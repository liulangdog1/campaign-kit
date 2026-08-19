"""Batch backend for a workload manager with ``sbatch``/``sacct``/``squeue`` tooling.

Nothing here touches the queue tooling at import time: the binaries are only
invoked inside :meth:`SlurmScheduler.submit` and :meth:`SlurmScheduler.poll`,
so the module imports (and unit tests monkeypatching :func:`subprocess.run`
work) on machines with no batch system installed.

Security note, stated once and followed everywhere: every external process is
started with ``subprocess.run(argv_list)`` — never ``shell=True``, and never a
command string built by interpolating user data. User-supplied command
arguments reach a shell only inside the generated batch script, where each
argument is individually quoted with :func:`shlex.join`.
"""

from __future__ import annotations

import re
import subprocess
import uuid
from collections.abc import Sequence
from pathlib import Path
from shlex import join as shlex_join
from shlex import quote as shlex_quote

from campaign_kit.scheduler.base import (
    BatchScheduler,
    JobLedger,
    JobState,
    SchedulerConfig,
    SchedulerError,
)

__all__ = ["SlurmScheduler"]

_SUBMIT_RE = re.compile(r"Submitted batch job (\d+)")

# Raw scheduler state codes normalized to JobState. Preemption and boot
# failures count as infrastructure loss (NODE_FAIL) because an unchanged
# resubmission is the right response; DEADLINE is a wall-clock kill, so it
# gets the TIMEOUT escalation path.
_STATE_MAP: dict[str, JobState] = {
    "PENDING": JobState.PENDING,
    "CONFIGURING": JobState.PENDING,
    "REQUEUED": JobState.PENDING,
    "RESIZING": JobState.PENDING,
    "SUSPENDED": JobState.PENDING,
    "RUNNING": JobState.RUNNING,
    "COMPLETING": JobState.RUNNING,
    "STAGE_OUT": JobState.RUNNING,
    "COMPLETED": JobState.COMPLETED,
    "TIMEOUT": JobState.TIMEOUT,
    "DEADLINE": JobState.TIMEOUT,
    "NODE_FAIL": JobState.NODE_FAIL,
    "PREEMPTED": JobState.NODE_FAIL,
    "BOOT_FAIL": JobState.NODE_FAIL,
    "CANCELLED": JobState.CANCELLED,
    "FAILED": JobState.FAILED,
    "OUT_OF_MEMORY": JobState.FAILED,
}


def _map_state(raw: str) -> JobState | None:
    """Normalize a raw state string ('CANCELLED by 123', 'FAILED+') to a JobState."""
    token = raw.strip().split()[0].rstrip("+") if raw.strip() else ""
    return _STATE_MAP.get(token)


def _parse_bracket_row(job_id: str) -> tuple[str, list[tuple[int, int]]] | None:
    """Parse an array-bracket id like ``123_[0-4,7%2]`` into (base, index intervals).

    Intervals are kept unexpanded so a huge declared range cannot blow up
    memory; membership is checked per wanted id instead.
    """
    base, sep, rest = job_id.partition("_[")
    if not sep or not rest.endswith("]"):
        return None
    body = rest[:-1].split("%", 1)[0]  # drop any throttle suffix
    intervals: list[tuple[int, int]] = []
    for token in body.split(","):
        lo_s, dash, hi_s = token.strip().partition("-")
        try:
            lo = int(lo_s)
            hi = int(hi_s) if dash else lo
        except ValueError:
            continue
        intervals.append((lo, hi))
    return base, intervals


def _parse_states(text: str, wanted: set[str]) -> dict[str, JobState]:
    """Parse '<job_id>|<state>' lines, resolving array-bracket rows to task ids.

    A not-yet-started array shows up as one row like ``123_[0-9]`` rather than
    one row per task; without resolving it, every such task would look absent
    and be misread as VANISHED. Only the indices inside the brackets belong to
    the row — a task missing from both the explicit rows and the bracket range
    really is unaccounted for.
    """
    states: dict[str, JobState] = {}
    bracket_rows: list[tuple[str, list[tuple[int, int]], JobState]] = []
    for line in text.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 2:
            continue
        job_id, raw_state = parts[0].strip(), parts[1]
        if "." in job_id:
            continue  # per-step accounting rows; only whole tasks matter here
        if "_[" in job_id:
            parsed = _parse_bracket_row(job_id)
            if parsed is not None:
                bracket_rows.append(
                    (parsed[0], parsed[1], _map_state(raw_state) or JobState.PENDING)
                )
            continue
        if job_id in wanted:
            state = _map_state(raw_state)
            if state is not None:
                states[job_id] = state
    for job_id in wanted - states.keys():
        base, sep, index_s = job_id.rpartition("_")
        if not sep:
            continue
        try:
            index = int(index_s)
        except ValueError:
            continue
        for row_base, intervals, row_state in bracket_rows:
            if row_base == base and any(lo <= index <= hi for lo, hi in intervals):
                states[job_id] = row_state
                break
    return states


class SlurmScheduler(BatchScheduler):
    """Runs each batch of commands as one job array on the cluster's queue.

    One array per :meth:`submit` call (instead of one job per command) keeps
    the scheduler database load flat no matter how many candidates a campaign
    round produces, and gives all tasks identical resources by construction.

    ``workdir`` holds generated scripts and per-task output files. Output
    files double as evidence: a job absent from both accounting and the queue
    is judged COMPLETED if its output file exists and VANISHED if not.
    """

    def __init__(
        self,
        workdir: str | Path = "campaign_jobs",
        ledger: JobLedger | None = None,
        *,
        sbatch_bin: str = "sbatch",
        sacct_bin: str = "sacct",
        squeue_bin: str = "squeue",
    ) -> None:
        super().__init__(ledger=ledger)
        self._workdir = Path(workdir)
        self._script_dir = self._workdir / "scripts"
        self._log_dir = self._workdir / "logs"
        self._sbatch_bin = sbatch_bin
        self._sacct_bin = sacct_bin
        self._squeue_bin = squeue_bin
        self._outputs: dict[str, Path] = {}

    # ------------------------------------------------------------------ submit

    def submit(self, commands: Sequence[list[str]], config: SchedulerConfig) -> list[str]:
        if not commands:
            return []
        self._script_dir.mkdir(parents=True, exist_ok=True)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        script_path = self._script_dir / f"array_{uuid.uuid4().hex[:12]}.sh"
        script_path.write_text(self._build_script(commands, config))
        # List argv, no shell: the script path is the only argument, so nothing
        # user-controlled can be reinterpreted by a shell here.
        argv = [self._sbatch_bin, str(script_path)]
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, check=False)
        except FileNotFoundError as exc:
            raise SchedulerError(f"batch submit tool not found: {self._sbatch_bin!r}") from exc
        if proc.returncode != 0:
            raise SchedulerError(f"submission failed (rc={proc.returncode}): {proc.stderr.strip()}")
        match = _SUBMIT_RE.search(proc.stdout)
        if match is None:
            raise SchedulerError(f"could not parse job id from submit output: {proc.stdout!r}")
        base_id = match.group(1)
        job_ids = [f"{base_id}_{index}" for index in range(len(commands))]
        for job_id, task_argv in zip(job_ids, commands, strict=True):
            output_path = self._log_dir / f"{job_id}.out"
            self._outputs[job_id] = output_path
            if self.ledger is not None:
                self.ledger.record(
                    job_id,
                    spec={
                        "argv": list(task_argv),
                        "time_limit_minutes": config.time_limit_minutes,
                    },
                    output_path=str(output_path),
                )
        return job_ids

    def _build_script(self, commands: Sequence[list[str]], config: SchedulerConfig) -> str:
        """Render the array script; the task index dispatches into a shell ``case``."""
        lines = [
            "#!/bin/bash",
            f"#SBATCH --partition={config.partition}",
            f"#SBATCH --account={config.account}",
            # A bare number is interpreted as minutes by the scheduler.
            f"#SBATCH --time={config.time_limit_minutes}",
            f"#SBATCH --cpus-per-task={config.cpus_per_task}",
            f"#SBATCH --mem={config.mem_gb}G",
            f"#SBATCH --array=0-{len(commands) - 1}",
            f"#SBATCH --output={self._log_dir}/%A_%a.out",
        ]
        lines.extend(config.extra_directives)
        lines.extend(f"module load {shlex_quote(module)}" for module in config.modules)
        lines.append("")
        lines.append('case "${SLURM_ARRAY_TASK_ID}" in')
        for index, task_argv in enumerate(commands):
            # shlex.join quotes each argument individually — the only point
            # where user data meets a shell, and it meets it inert.
            lines.append(f"  {index}) exec {shlex_join(task_argv)} ;;")
        lines.append('  *) echo "no such array task" >&2; exit 64 ;;')
        lines.append("esac")
        lines.append("")
        return "\n".join(lines)

    # -------------------------------------------------------------------- poll

    def poll(self, job_ids: Sequence[str]) -> dict[str, JobState]:
        ids = list(dict.fromkeys(job_ids))
        if not ids:
            return {}
        found = self._query_sacct(ids)
        if found is None:
            found = self._query_squeue(ids)
        if found is None:
            raise SchedulerError(f"neither {self._sacct_bin!r} nor {self._squeue_bin!r} is usable")
        states = {job_id: found.get(job_id) or self._classify_absent(job_id) for job_id in ids}
        self._record_states(states)
        return states

    def _query_sacct(self, ids: list[str]) -> dict[str, JobState] | None:
        """Ask the accounting database; None means the tool itself is unusable."""
        argv = [
            self._sacct_bin,
            "--jobs",
            ",".join(ids),
            "--format=JobID,State",
            "--noheader",
            "--parsable2",
            "--allocations",
        ]
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, check=False)
        except FileNotFoundError:
            return None
        if proc.returncode != 0:
            return None
        return _parse_states(proc.stdout, set(ids))

    def _query_squeue(self, ids: list[str]) -> dict[str, JobState] | None:
        """Fallback for sites without accounting; sees queued/running jobs only."""
        argv = [
            self._squeue_bin,
            "--jobs",
            ",".join(ids),
            "--noheader",
            "--format=%i|%T",
        ]
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, check=False)
        except FileNotFoundError:
            return None
        if proc.returncode != 0:
            # The queue tool errors on wholly unknown ids; an empty answer is
            # still an answer (all jobs absent -> output-file check decides).
            return {}
        return _parse_states(proc.stdout, set(ids))

    def _classify_absent(self, job_id: str) -> JobState:
        """Judge a job that neither accounting nor the queue can see.

        The output file is the tie-breaker: accounting records expire, but the
        artifact a finished job wrote does not. Present -> COMPLETED; absent ->
        VANISHED (the silent-loss case the ledger exists to surface).
        """
        output = self._outputs.get(job_id)
        if output is None and self.ledger is not None:
            entry = self.ledger.get(job_id)
            if entry is not None and entry.get("output_path"):
                output = Path(entry["output_path"])
        if output is not None and output.exists():
            return JobState.COMPLETED
        return JobState.VANISHED
