"""State classification and retry behavior of the cluster scheduler backend.

``subprocess.run`` is monkeypatched, so no queue tooling is required: the
fake plays the roles of the submit tool (hands out array job ids) and the
accounting tool (answers state queries from a scripted table). What stays
real is everything worth testing — script generation, id bookkeeping, state
normalization, absence classification, and the retry loop's escalation.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from campaign_kit.scheduler import (
    JobLedger,
    JobState,
    RetryPolicy,
    SchedulerConfig,
    SlurmScheduler,
)


def _directive(script: str, key: str) -> str:
    for line in script.splitlines():
        if line.startswith(f"#SBATCH {key}="):
            return line.split("=", 1)[1]
    raise AssertionError(f"directive {key} not found in script:\n{script}")


class _FakeCluster:
    """Callable standing in for ``subprocess.run`` on a queue-managed cluster.

    ``auto_state`` (when set) marks every submitted task with that raw
    accounting state immediately, which lets the retry loop run to completion
    without the test ever touching the scheduler's internals.
    """

    def __init__(self, auto_state: str | None = None) -> None:
        self.sacct_states: dict[str, str] = {}
        self.submitted_scripts: list[str] = []
        self.time_limits: list[int] = []
        self._next_base = 1000
        self._auto_state = auto_state

    def __call__(self, argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        tool = Path(argv[0]).name
        if tool == "sbatch":
            script = Path(argv[-1]).read_text()
            self.submitted_scripts.append(script)
            self.time_limits.append(int(_directive(script, "--time")))
            n_tasks = int(_directive(script, "--array").split("-")[1]) + 1
            base = str(self._next_base)
            self._next_base += 1
            if self._auto_state is not None:
                for index in range(n_tasks):
                    self.sacct_states[f"{base}_{index}"] = self._auto_state
            return subprocess.CompletedProcess(argv, 0, f"Submitted batch job {base}\n", "")
        if tool == "sacct":
            ids = argv[argv.index("--jobs") + 1].split(",")
            out = "".join(f"{i}|{self.sacct_states[i]}\n" for i in ids if i in self.sacct_states)
            return subprocess.CompletedProcess(argv, 0, out, "")
        raise AssertionError(f"unexpected tool invoked: {argv}")


def _patched(monkeypatch: pytest.MonkeyPatch, fake: _FakeCluster) -> None:
    monkeypatch.setattr("campaign_kit.scheduler.slurm.subprocess.run", fake)


def test_slurm_state_classification(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeCluster()
    _patched(monkeypatch, fake)
    sched = SlurmScheduler(workdir=tmp_path)
    ids = sched.submit([["prog", str(i)] for i in range(6)], SchedulerConfig())
    assert len(ids) == 6
    base = ids[0].rsplit("_", 1)[0]
    fake.sacct_states.update(
        {
            f"{base}_0": "COMPLETED",
            f"{base}_1": "FAILED",
            f"{base}_2": "TIMEOUT",
            f"{base}_3": "NODE_FAIL",
            f"{base}_4": "CANCELLED by 1234",  # raw suffix must not confuse parsing
            # base_5 deliberately absent from accounting -> silent loss
        }
    )
    states = sched.poll(ids)
    assert states[ids[0]] is JobState.COMPLETED
    assert states[ids[1]] is JobState.FAILED
    assert states[ids[2]] is JobState.TIMEOUT
    assert states[ids[3]] is JobState.NODE_FAIL
    assert states[ids[4]] is JobState.CANCELLED
    assert states[ids[5]] is JobState.VANISHED

    # Absent from accounting but with an output file on disk: the artifact is
    # terminal evidence, so the job is judged COMPLETED, not VANISHED.
    (tmp_path / "logs" / f"{ids[5]}.out").write_text("done\n")
    assert sched.poll(ids)[ids[5]] is JobState.COMPLETED


def test_timeout_retry_escalates_walltime_up_to_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeCluster(auto_state="TIMEOUT")
    _patched(monkeypatch, fake)
    sched = SlurmScheduler(workdir=tmp_path)
    policy = RetryPolicy(max_attempts=3, walltime_multiplier_on_timeout=2.0)
    final = sched.run_with_retries(
        [["prog", "a"]],
        SchedulerConfig(time_limit_minutes=10),
        policy,
        poll_interval=0.01,
    )
    assert final == [JobState.TIMEOUT]
    # Three submissions total (the attempt cap), each with a doubled wall
    # clock, and no fourth attempt after the cap.
    assert fake.time_limits == [10, 20, 40]
    assert len(fake.submitted_scripts) == 3


def test_retry_policy_rules() -> None:
    cfg = SchedulerConfig(time_limit_minutes=30)
    policy = RetryPolicy(max_attempts=3, walltime_multiplier_on_timeout=2.0)
    # Success and deterministic/operator outcomes never retry.
    assert policy.next_config(JobState.COMPLETED, cfg, 1) is None
    assert policy.next_config(JobState.FAILED, cfg, 1) is None
    assert policy.next_config(JobState.CANCELLED, cfg, 1) is None
    # Infrastructure loss resubmits unchanged.
    assert policy.next_config(JobState.NODE_FAIL, cfg, 1) == cfg
    assert policy.next_config(JobState.VANISHED, cfg, 2) == cfg
    # A wall-clock kill escalates by the multiplier.
    escalated = policy.next_config(JobState.TIMEOUT, cfg, 1)
    assert escalated is not None
    assert escalated.time_limit_minutes == 60
    # The cap counts total attempts, and non-terminal states never retry.
    assert policy.next_config(JobState.TIMEOUT, cfg, 3) is None
    assert policy.next_config(JobState.RUNNING, cfg, 1) is None


def test_submit_script_uses_placeholders_and_array(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeCluster()
    _patched(monkeypatch, fake)
    sched = SlurmScheduler(workdir=tmp_path)
    sched.submit([["prog", "x y"], ["prog", "b"]], SchedulerConfig())
    script = fake.submitted_scripts[0]
    assert _directive(script, "--partition") == "PARTITION_NAME"
    assert _directive(script, "--account") == "ACCOUNT_NAME"
    assert _directive(script, "--array") == "0-1"
    assert "'x y'" in script  # arguments are individually shell-quoted


def test_job_ledger_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    ledger = JobLedger(path)
    ledger.record("1000_0", spec={"argv": ["prog"]}, output_path="out/1000_0.out")
    assert "1000_0" in ledger
    assert len(ledger) == 1
    ledger.update("1000_0", state=JobState.COMPLETED, attempts=2)

    reloaded = JobLedger(path)  # a fresh driver process reading the same file
    entry = reloaded.get("1000_0")
    assert entry is not None
    assert entry["state"] == "COMPLETED"
    assert entry["attempts"] == 2
    assert entry["spec"] == {"argv": ["prog"]}
    with pytest.raises(KeyError):
        reloaded.update("unknown", state=JobState.FAILED)
