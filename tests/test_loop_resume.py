"""Kill-and-resume behavior of the campaign driver.

The invariant under test: labeling compute is spent at submit time, so a
crash between submit and collect must never lead to the same structure being
submitted (paid for) twice. The counting labeler computes each label exactly
once, at submit; re-collecting after a resume reads stored results instead of
recomputing, exactly like polling a cluster for already-finished jobs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from campaign_kit.loop import STATE_FILENAME, Campaign, CampaignConfig, CampaignState
from campaign_kit.protocols import Labeler
from tests._doubles import (
    AdapterSelector,
    CountingLabeler,
    CrashOnCollect,
    JitterProposer,
    TinyLinearModel,
    jittered_structures,
    labeled_dataset,
)


def _make_campaign(labeler: Labeler, checkpoint_dir: Path) -> Campaign:
    """A fresh driver, as after a process restart: new models, new selector."""
    config = CampaignConfig(
        max_rounds=3,
        label_budget=100,
        batch_size=4,
        band_factor=2.0,
        plateau_patience=10,  # keep the plateau stop out of this test's way
        seed=7,
        checkpoint_dir=checkpoint_dir,
        holdout_fraction=0.25,
    )
    return Campaign(
        labeler=labeler,
        committee=[TinyLinearModel() for _ in range(3)],
        selector=AdapterSelector(batch_size=4),
        proposer=JitterProposer(),
        config=config,
    )


def test_loop_resume(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "ckpt"
    state_path = checkpoint_dir / STATE_FILENAME
    inner = CountingLabeler()
    labeler = CrashOnCollect(inner, crash_on_call=2)  # dies inside round 1's collect
    initial = labeled_dataset(jittered_structures(12, scale=0.2, seed=1))

    with pytest.raises(RuntimeError, match="simulated kill"):
        _make_campaign(labeler, checkpoint_dir).run(initial=initial)

    # The submit checkpoint of the interrupted round must be on disk: claimed
    # ids, persisted handles, and the partial counters of round 1.
    mid = CampaignState.load(state_path)
    assert len(mid.records) == 1
    assert len(mid.pending) == 4
    assert mid.labels_submitted == 8
    assert mid.in_flight is not None
    assert mid.in_flight["round_index"] == 1

    # Restart from the checkpoint with a fresh driver (same labeler backend,
    # as on a real cluster where the queue outlives the driver process).
    records = _make_campaign(labeler, checkpoint_dir).run()
    assert len(records) == 3
    assert [r.round_index for r in records] == [0, 1, 2]
    assert all(r.n_submitted == 4 for r in records)
    # The interrupted round was completed from its persisted handles.
    assert records[1].n_merged == 4

    # ZERO duplicate labels: compute spent == unique structures submitted.
    assert inner.label_calls == len(set(inner.submitted_ids))
    assert len(inner.submitted_ids) == len(set(inner.submitted_ids))
    assert inner.label_calls == 12  # 3 rounds x batch 4, paid exactly once each

    # The final dataset holds no duplicate structure ids, and the holdout
    # stayed disjoint from the training rows.
    final = CampaignState.load(state_path)
    assert len(final.dataset) == len(final.dataset.ids)
    assert final.dataset.ids.isdisjoint(final.holdout.ids)
    assert set(inner.submitted_ids) <= final.labeled_ids

    # Running again after completion is a no-op: no new labels, same records.
    again = _make_campaign(labeler, checkpoint_dir).run()
    assert len(again) == 3
    assert inner.label_calls == 12


def test_label_budget_is_exact(tmp_path: Path) -> None:
    config = CampaignConfig(
        max_rounds=10,
        label_budget=6,
        batch_size=4,
        band_factor=2.0,
        plateau_patience=10,
        seed=3,
        checkpoint_dir=tmp_path / "ckpt",
        holdout_fraction=0.25,
    )
    labeler = CountingLabeler()
    campaign = Campaign(
        labeler=labeler,
        committee=[TinyLinearModel() for _ in range(2)],
        selector=AdapterSelector(batch_size=4),
        proposer=JitterProposer(),
        config=config,
    )
    records = campaign.run(initial=labeled_dataset(jittered_structures(8, scale=0.2, seed=2)))
    # Round 0 spends 4, round 1 is clipped to the remaining 2, then the stop.
    assert labeler.label_calls == 6
    assert sum(r.n_submitted for r in records) == 6
    assert [r.n_submitted for r in records] == [4, 2]
