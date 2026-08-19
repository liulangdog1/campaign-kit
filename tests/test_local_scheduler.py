"""LocalScheduler against real subprocesses.

This backend is the CI stand-in for a cluster, so its terminal states have to
mean the same things: exit 0 -> COMPLETED, nonzero or unlaunchable -> FAILED,
wall-clock kill -> TIMEOUT, unknown id -> VANISHED. The concurrency cap is
observed through polling, which is the only mechanism the backend has.
"""

from __future__ import annotations

import time
from pathlib import Path

from campaign_kit.scheduler import JobLedger, JobState, LocalScheduler, SchedulerConfig


def test_local_completed_and_failed(tmp_path: Path) -> None:
    sched = LocalScheduler(max_concurrent=2, workdir=tmp_path)
    ids = sched.submit([["true"], ["false"]], SchedulerConfig())
    states = sched.wait(ids, poll_interval=0.05)
    assert states[ids[0]] is JobState.COMPLETED
    assert states[ids[1]] is JobState.FAILED


def test_local_missing_binary_is_failed_not_raised(tmp_path: Path) -> None:
    sched = LocalScheduler(max_concurrent=1, workdir=tmp_path)
    ids = sched.submit([["campaign-kit-no-such-binary"]], SchedulerConfig())
    states = sched.wait(ids, poll_interval=0.05)
    assert states[ids[0]] is JobState.FAILED


def test_local_unknown_id_polls_as_vanished(tmp_path: Path) -> None:
    sched = LocalScheduler(workdir=tmp_path)
    assert sched.poll(["ghost-1"])["ghost-1"] is JobState.VANISHED


def test_local_enforces_wall_clock_as_timeout(tmp_path: Path) -> None:
    sched = LocalScheduler(max_concurrent=1, workdir=tmp_path)
    config = SchedulerConfig(time_limit_minutes=0.005)  # type: ignore[arg-type]  # 0.3 s
    ids = sched.submit([["sleep", "5"]], config)
    states = sched.wait(ids, poll_interval=0.05)
    assert states[ids[0]] is JobState.TIMEOUT


def test_local_concurrency_cap_is_respected(tmp_path: Path) -> None:
    sched = LocalScheduler(max_concurrent=1, workdir=tmp_path)
    ids = sched.submit([["sleep", "0.3"], ["sleep", "0.3"], ["sleep", "0.3"]], SchedulerConfig())
    deadline = time.monotonic() + 30.0
    while True:
        states = sched.poll(ids)
        running = sum(1 for s in states.values() if s is JobState.RUNNING)
        assert running <= 1  # the cap must hold at every observation
        if all(s.is_terminal for s in states.values()):
            break
        assert time.monotonic() < deadline, "jobs did not finish in time"
        time.sleep(0.05)
    assert all(states[i] is JobState.COMPLETED for i in ids)


def test_local_scheduler_records_states_in_ledger(tmp_path: Path) -> None:
    ledger = JobLedger(tmp_path / "ledger.json")
    sched = LocalScheduler(max_concurrent=2, workdir=tmp_path / "work", ledger=ledger)
    ids = sched.submit([["true"], ["false"]], SchedulerConfig())
    sched.wait(ids, poll_interval=0.05)
    reloaded = JobLedger(tmp_path / "ledger.json")
    first = reloaded.get(ids[0])
    second = reloaded.get(ids[1])
    assert first is not None and first["state"] == "COMPLETED"
    assert second is not None and second["state"] == "FAILED"
