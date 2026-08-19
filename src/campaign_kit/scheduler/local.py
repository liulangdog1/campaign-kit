"""Local-subprocess backend: the same scheduler interface, no cluster required.

This is what demos and CI run against. It matters that the *interface* is
identical to the cluster backend — including a real TIMEOUT path — because the
retry logic in `BatchScheduler.run_with_retries` is exactly the code that is
hardest to exercise against a real queue. Here a TIMEOUT is enforced by
killing the child once it exceeds ``time_limit_minutes``, so policy behavior
is testable in seconds.

Concurrency is capped semaphore-style but without threads: children are only
started inside ``submit``/``poll`` calls, up to ``max_concurrent`` at a time,
and finished children free their slot on the next ``poll``. Poll-driven
progress keeps the backend single-threaded and deterministic under test.

NODE_FAIL and CANCELLED never occur locally; a child either exits 0
(COMPLETED), exits nonzero or cannot start (FAILED), or is killed on wall
clock (TIMEOUT). Unknown ids poll as VANISHED for parity with the cluster
backend's silent-loss semantics.
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from campaign_kit.scheduler.base import BatchScheduler, JobLedger, JobState, SchedulerConfig

__all__ = ["LocalScheduler"]


@dataclass
class _LocalJob:
    """Book-keeping for one locally run command."""

    argv: list[str]
    time_limit_s: float
    output_path: Path
    state: JobState = JobState.PENDING
    proc: subprocess.Popen[bytes] | None = None
    output_file: IO[bytes] | None = None
    started_at: float | None = None


class LocalScheduler(BatchScheduler):
    """Runs each command as a local subprocess, at most ``max_concurrent`` at once."""

    def __init__(
        self,
        max_concurrent: int = 2,
        workdir: str | Path | None = None,
        ledger: JobLedger | None = None,
    ) -> None:
        super().__init__(ledger=ledger)
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        self._max_concurrent = max_concurrent
        if workdir is None:
            workdir = tempfile.mkdtemp(prefix="campaign_kit_local_")
        self._workdir = Path(workdir)
        self._jobs: dict[str, _LocalJob] = {}
        self._seq = 0

    def submit(self, commands: Sequence[list[str]], config: SchedulerConfig) -> list[str]:
        job_ids: list[str] = []
        for argv in commands:
            self._seq += 1
            job_id = f"local-{self._seq}"
            job = _LocalJob(
                argv=list(argv),
                time_limit_s=config.time_limit_minutes * 60.0,
                output_path=self._workdir / f"{job_id}.out",
            )
            self._jobs[job_id] = job
            job_ids.append(job_id)
            if self.ledger is not None:
                self.ledger.record(
                    job_id,
                    spec={
                        "argv": list(argv),
                        "time_limit_minutes": config.time_limit_minutes,
                    },
                    output_path=str(job.output_path),
                )
        self._advance()
        return job_ids

    def poll(self, job_ids: Sequence[str]) -> dict[str, JobState]:
        # Reap first so finished children free their slots, then start queued
        # work — this is the whole "semaphore without threads" mechanism.
        self._reap()
        self._advance()
        states: dict[str, JobState] = {}
        for job_id in job_ids:
            job = self._jobs.get(job_id)
            states[job_id] = JobState.VANISHED if job is None else job.state
        self._record_states(states)
        return states

    # ------------------------------------------------------------ internals

    def _running(self) -> int:
        return sum(1 for job in self._jobs.values() if job.state is JobState.RUNNING)

    def _advance(self) -> None:
        """Start queued jobs until the concurrency cap is reached."""
        free = self._max_concurrent - self._running()
        for job in self._jobs.values():
            if free <= 0:
                break
            if job.state is not JobState.PENDING:
                continue
            self._start(job)
            if job.state is JobState.RUNNING:
                free -= 1

    def _start(self, job: _LocalJob) -> None:
        self._workdir.mkdir(parents=True, exist_ok=True)
        output_file = job.output_path.open("wb")
        try:
            # List argv, no shell — identical hygiene to the cluster backend.
            job.proc = subprocess.Popen(  # noqa: S603 - argv list, shell never involved
                job.argv, stdout=output_file, stderr=subprocess.STDOUT
            )
        except OSError:
            # An unlaunchable command (missing binary, bad permissions) is a
            # deterministic FAILED, not an exception: one bad item must not
            # abort the batch.
            output_file.close()
            job.state = JobState.FAILED
            return
        job.output_file = output_file
        job.started_at = time.monotonic()
        job.state = JobState.RUNNING

    def _reap(self) -> None:
        """Collect exit statuses and enforce wall-clock limits."""
        for job in self._jobs.values():
            if job.state is not JobState.RUNNING or job.proc is None:
                continue
            returncode = job.proc.poll()
            if returncode is None:
                started = job.started_at if job.started_at is not None else time.monotonic()
                if time.monotonic() - started > job.time_limit_s:
                    job.proc.kill()
                    job.proc.wait()
                    self._finish(job, JobState.TIMEOUT)
                continue
            self._finish(job, JobState.COMPLETED if returncode == 0 else JobState.FAILED)

    def _finish(self, job: _LocalJob, state: JobState) -> None:
        job.state = state
        if job.output_file is not None:
            job.output_file.close()
            job.output_file = None
