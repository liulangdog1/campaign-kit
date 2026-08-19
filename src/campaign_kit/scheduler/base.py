"""Batch-scheduler abstractions: job states, retry policy, and a crash-safe ledger.

The campaign driver treats "run these commands on a cluster" as a pluggable
backend so the same driver runs unchanged against a real batch system and
against local subprocesses in CI. Two design decisions here carry most of the
weight:

1. Job failure is data, not an exception. One dead task out of a thousand must
   not abort the campaign, so every job ends in a :class:`JobState` and the
   caller decides what each terminal state means.
2. The driver process is assumed to be mortal. Cluster jobs outlive the Python
   process that submitted them, so :class:`JobLedger` persists every submission
   to disk atomically — after a crash, a fresh driver can reconnect to (or at
   least account for) work that is still running and money already spent.
"""

from __future__ import annotations

import abc
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any

__all__ = [
    "BatchScheduler",
    "JobLedger",
    "JobState",
    "RetryPolicy",
    "SchedulerConfig",
    "SchedulerError",
]


class SchedulerError(RuntimeError):
    """The scheduler itself misbehaved (submission rejected, queue tooling missing).

    Distinct from job failure on purpose: a job that dies is a normal,
    per-item outcome (:class:`JobState`), while this exception means the
    campaign cannot make progress at all and a human should look.
    """


class JobState(Enum):
    """Terminal and in-flight states of one batch job, normalized across backends.

    ``VANISHED`` is the silent case: the job left the queue with no terminal
    record and no output file. Batch systems can drop jobs without a trace
    (accounting records expire, queues get purged, admins intervene). Treating
    absence as "still running" would make :meth:`BatchScheduler.wait` poll
    forever; treating it as success would corrupt results with missing data.
    Giving absence its own terminal state forces the caller to handle it, and
    lets the retry policy treat it as infrastructure loss.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    NODE_FAIL = "node_fail"
    CANCELLED = "cancelled"
    VANISHED = "vanished"

    @property
    def is_terminal(self) -> bool:
        """True once the job can no longer change state on its own."""
        return self not in (JobState.PENDING, JobState.RUNNING)


@dataclass
class SchedulerConfig:
    """Resource request for one batch of jobs.

    Defaults are deliberately placeholders: this package never encodes a real
    cluster's partition or account names. Deployments substitute their own
    values at the call site or via their own configuration layer.
    """

    partition: str = "PARTITION_NAME"
    account: str = "ACCOUNT_NAME"
    time_limit_minutes: int = 60
    cpus_per_task: int = 1
    mem_gb: int = 4
    #: Environment modules to load before the command runs (site-specific names).
    modules: list[str] = field(default_factory=list)
    #: Extra raw directive lines (e.g. ``"#SBATCH --constraint=..."``) appended
    #: verbatim after the generated ones — the escape hatch for site quirks.
    extra_directives: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RetryPolicy:
    """Decides whether a finished job earns another attempt, and with what resources.

    The rules encode where each failure mode comes from:

    - ``TIMEOUT``: the job was healthy but the wall clock was underestimated.
      Resubmitting under the identical limit reproduces the kill, so the limit
      is escalated by ``walltime_multiplier_on_timeout`` each attempt.
    - ``NODE_FAIL`` / ``VANISHED``: the infrastructure failed, not the input.
      An unchanged resubmission is expected to succeed.
    - ``FAILED``: treated as deterministic — the same input will fail the same
      way again. Retrying burns queue allocation, delays every other job, and
      hides real bugs behind flaky-looking noise; the item should surface as a
      failure for the caller to drop or investigate.
    - ``CANCELLED``: an operator decision. Automatically resubmitting would
      fight the human who cancelled it.
    """

    max_attempts: int = 3
    walltime_multiplier_on_timeout: float = 2.0

    def next_config(
        self, state: JobState, config: SchedulerConfig, attempt: int
    ) -> SchedulerConfig | None:
        """Return the config for the next attempt, or None to stop retrying.

        ``attempt`` is the 1-based number of attempts already made, so the cap
        is on total submissions, not on retries.
        """
        if not state.is_terminal or state is JobState.COMPLETED:
            return None
        if attempt >= self.max_attempts:
            return None
        if state is JobState.TIMEOUT:
            longer = math.ceil(config.time_limit_minutes * self.walltime_multiplier_on_timeout)
            # Guarantee strict growth even for multipliers close to 1.
            return replace(config, time_limit_minutes=max(longer, config.time_limit_minutes + 1))
        if state in (JobState.NODE_FAIL, JobState.VANISHED):
            return config
        return None


class JobLedger:
    """Persistent on-disk record of every submitted job.

    Maps ``job_id`` to ``{state, attempts, spec, output_path, submitted_at}``.

    Why this exists: batch jobs outlive the Python process that submitted them.
    If the driver crashes (or is preempted) with jobs in flight, the scheduler
    still knows about those jobs — but without an on-disk record the restarted
    driver does not, and the compute already spent becomes untraceable. The
    ledger is the only durable binding between job ids and what they were for.

    Why atomic writes (write a temp file, then :func:`os.replace`): a crash in
    the middle of a plain overwrite leaves a truncated JSON file, which is
    worse than a stale one — it orphans *every* job, not just the newest. The
    replace is atomic on POSIX, so a reader observes either the old complete
    ledger or the new complete ledger, never a torn one.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._data: dict[str, dict[str, Any]] = {}
        if self._path.exists():
            self._data = json.loads(self._path.read_text())

    @property
    def path(self) -> Path:
        return self._path

    def record(
        self,
        job_id: str,
        *,
        spec: dict[str, Any],
        output_path: str | None = None,
        state: JobState = JobState.PENDING,
        attempts: int = 1,
        submitted_at: float | None = None,
    ) -> None:
        """Register a newly submitted job (overwrites any stale entry for the id)."""
        self._data[job_id] = {
            "state": state.name,
            "attempts": attempts,
            "spec": spec,
            "output_path": output_path,
            "submitted_at": time.time() if submitted_at is None else submitted_at,
        }
        self._flush()

    def update(
        self,
        job_id: str,
        *,
        state: JobState | None = None,
        attempts: int | None = None,
    ) -> None:
        """Update fields of a known job; raises KeyError for unknown ids.

        Raising (rather than silently creating a stub) keeps the invariant
        that every ledger row was written by a real submission.
        """
        entry = self._data[job_id]
        if state is not None:
            entry["state"] = state.name
        if attempts is not None:
            entry["attempts"] = attempts
        self._flush()

    def get(self, job_id: str) -> dict[str, Any] | None:
        entry = self._data.get(job_id)
        return dict(entry) if entry is not None else None

    def jobs(self) -> dict[str, dict[str, Any]]:
        return {job_id: dict(entry) for job_id, entry in self._data.items()}

    def __contains__(self, job_id: object) -> bool:
        return job_id in self._data

    def __len__(self) -> int:
        return len(self._data)

    def _flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, sort_keys=True))
        os.replace(tmp, self._path)


class BatchScheduler(abc.ABC):
    """Submits batches of commands and tracks them to a terminal state.

    Concrete backends implement :meth:`submit` and :meth:`poll`; the waiting
    and retry logic lives here so every backend behaves identically under
    failure — which is exactly the behavior that is hardest to test against a
    real cluster and therefore must not be reimplemented per backend.
    """

    def __init__(self, ledger: JobLedger | None = None) -> None:
        self.ledger = ledger

    @abc.abstractmethod
    def submit(self, commands: Sequence[list[str]], config: SchedulerConfig) -> list[str]:
        """Submit one job per command; returns job ids aligned with ``commands``."""

    @abc.abstractmethod
    def poll(self, job_ids: Sequence[str]) -> dict[str, JobState]:
        """Return the current state of every requested id, without blocking.

        Must return an entry for *every* requested id; ids the backend cannot
        account for come back as ``VANISHED`` rather than being dropped, so
        callers never have to handle a partial answer.
        """

    def wait(
        self,
        job_ids: Sequence[str],
        poll_interval: float = 2.0,
        backoff: float = 1.5,
        max_interval: float = 60.0,
    ) -> dict[str, JobState]:
        """Block until every job reaches a terminal state; returns the final states.

        Polling backs off exponentially from ``poll_interval`` up to
        ``max_interval``: short early intervals give fast turnaround for
        quick local jobs, while the cap keeps long campaigns from hammering
        the scheduler's accounting service for hours.
        """
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        if backoff < 1.0:
            raise ValueError("backoff must be >= 1.0")
        ids = list(dict.fromkeys(job_ids))
        if not ids:
            return {}
        interval = poll_interval
        while True:
            states = self.poll(ids)
            if all(state.is_terminal for state in states.values()):
                return states
            time.sleep(interval)
            interval = min(interval * backoff, max_interval)

    def run_with_retries(
        self,
        commands: Sequence[list[str]],
        config: SchedulerConfig,
        policy: RetryPolicy | None = None,
        *,
        poll_interval: float = 2.0,
        backoff: float = 1.5,
        max_interval: float = 60.0,
    ) -> list[JobState]:
        """Submit, wait, and resubmit per :class:`RetryPolicy`; one final state per command.

        Resubmissions are grouped by their (possibly escalated) time limit so
        each group still goes out as a single batch submission rather than one
        call per job.
        """
        policy = RetryPolicy() if policy is None else policy
        argvs = [list(c) for c in commands]
        final: list[JobState | None] = [None] * len(argvs)
        job_ids = self.submit(argvs, config)
        active: dict[str, tuple[int, SchedulerConfig, int]] = {
            job_id: (index, config, 1) for index, job_id in enumerate(job_ids)
        }
        while active:
            states = self.wait(list(active), poll_interval, backoff, max_interval)
            retries: list[tuple[int, SchedulerConfig, int]] = []
            for job_id, (index, cfg, attempt) in active.items():
                state = states[job_id]
                nxt = policy.next_config(state, cfg, attempt)
                if nxt is None:
                    final[index] = state
                else:
                    retries.append((index, nxt, attempt + 1))
            active = {}
            for limit in sorted({cfg.time_limit_minutes for _, cfg, _ in retries}):
                group = [item for item in retries if item[1].time_limit_minutes == limit]
                new_ids = self.submit([argvs[index] for index, _, _ in group], group[0][1])
                for new_id, (index, cfg, attempt) in zip(new_ids, group, strict=True):
                    active[new_id] = (index, cfg, attempt)
                    if self.ledger is not None and new_id in self.ledger:
                        self.ledger.update(new_id, attempts=attempt)
        results: list[JobState] = []
        for outcome in final:
            if outcome is None:  # pragma: no cover - every index reaches a terminal state
                raise SchedulerError("internal error: job ended without a recorded state")
            results.append(outcome)
        return results

    def _record_states(self, states: Mapping[str, JobState]) -> None:
        """Mirror observed states into the ledger (ids the ledger knows about only)."""
        if self.ledger is None:
            return
        for job_id, state in states.items():
            if job_id in self.ledger:
                self.ledger.update(job_id, state=state)
