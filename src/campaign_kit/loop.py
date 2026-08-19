"""Campaign driver: propose -> fence -> submit -> collect -> merge -> retrain -> check.

The loop is written to survive being killed at any point. Every phase ends with
a checkpoint of the full :class:`CampaignState`, and every side effect is either
idempotent or guarded by bookkeeping the checkpoint carries: structure ids are
"claimed" at *submit* time (a crash between submit and collect can never label
the same geometry twice), pending job handles are persisted (a restart
re-collects in-flight work instead of resubmitting it), and merging is
idempotent for free because `Dataset.merge` deduplicates on content-hashed ids
— the driver leans on that rather than tracking which results were merged.

Model weights are deliberately *not* checkpointed: the committee is a pure
function of (dataset, seeds), so a resume simply refits. That keeps the
checkpoint format independent of any model backend.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from campaign_kit.protocols import (
    Dataset,
    JobHandle,
    Labeler,
    Model,
    Proposer,
    Selector,
    Structure,
)

__all__ = ["CampaignConfig", "RoundRecord", "CampaignState", "Campaign"]

logger = logging.getLogger(__name__)

STATE_FILENAME = "campaign_state.json"

#: A domain fence maps a batch of structures to one score per structure
#: (larger = further outside the trusted region). A boolean array is also
#: accepted and is interpreted directly as a keep-mask (True = keep).
DomainFence = Callable[[Sequence[Structure]], np.ndarray]


@dataclass
class CampaignConfig:
    """Tunables of one campaign. All defaults are generic starting points.

    ``label_budget`` counts label *submissions* (not successes): a failed
    calculation still spends compute. ``band_factor`` oversamples proposals —
    the proposer is asked for ``ceil(band_factor * batch_size)`` candidates so
    the fence and selector have a real pool to cut from (at 1.0 the selector
    has no choice). ``fence_threshold`` drops candidates whose fence score
    exceeds it; the scale is set by the fence you supply, with 1.0 the
    conventional "edge of the trusted region". The plateau stop needs
    ``plateau_patience`` consecutive rounds whose relative holdout-RMSE
    improvement is below ``plateau_rel_tol`` — patience guards against
    stopping on one noisy round. ``holdout_fraction`` of the *initial* dataset
    is split off once, at campaign start, and kept fixed so the plateau metric
    is comparable between rounds. ``seed`` is the root of every stochastic
    choice (split, proposals, committee fits), making a campaign reproducible.
    """

    max_rounds: int = 20
    label_budget: int = 500
    batch_size: int = 16
    band_factor: float = 4.0
    fence_threshold: float = 1.0
    plateau_patience: int = 3
    plateau_rel_tol: float = 0.01
    seed: int = 0
    checkpoint_dir: str | Path = "campaign_checkpoints"
    holdout_fraction: float = 0.2


@dataclass
class RoundRecord:
    """What happened in one round, in counts. One structured log line mirrors it."""

    round_index: int
    n_proposed: int
    n_fenced_out: int
    n_submitted: int
    n_failed: int
    n_merged: int
    dataset_size: int
    holdout_rmse: float | None


def _structure_to_json(s: Structure) -> dict[str, Any]:
    # Positions are rounded at 1e-10 (matching Structure.structure_id) so a
    # JSON round trip cannot change a structure's identity.
    return {
        "species": list(s.species),
        "positions": np.round(s.positions, 10).tolist(),
        "tag": s.tag,
    }


def _structure_from_json(d: dict[str, Any]) -> Structure:
    return Structure(
        species=tuple(d["species"]),
        positions=np.asarray(d["positions"], dtype=float),
        tag=d.get("tag", ""),
    )


def _dataset_to_json(ds: Dataset) -> list[dict[str, Any]]:
    return [
        {
            "structure": _structure_to_json(s),
            "energy": float(e),
            "forces": None if f is None else np.asarray(f, dtype=float).tolist(),
        }
        for s, e, f in zip(ds.structures, ds.energies, ds.forces, strict=True)
    ]


def _dataset_from_json(rows: list[dict[str, Any]]) -> Dataset:
    ds = Dataset()
    for row in rows:
        ds.structures.append(_structure_from_json(row["structure"]))
        ds.energies.append(float(row["energy"]))
        f = row["forces"]
        ds.forces.append(None if f is None else np.asarray(f, dtype=float))
    return ds


@dataclass
class CampaignState:
    """Everything needed to resume a campaign after a kill, as one JSON file.

    ``labeled_ids`` means "claimed": an id enters the set at submit time, not
    when its result arrives — that invariant makes a crash between submit and
    collect safe. ``in_flight`` holds an interrupted round's partial counters
    so its :class:`RoundRecord` can still be completed on resume.
    """

    dataset: Dataset = field(default_factory=Dataset)
    holdout: Dataset = field(default_factory=Dataset)
    records: list[RoundRecord] = field(default_factory=list)
    labeled_ids: set[str] = field(default_factory=set)
    pending: list[JobHandle] = field(default_factory=list)
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    labels_submitted: int = 0
    in_flight: dict[str, int] | None = None

    def save(self, path: str | Path) -> None:
        """Serialize to JSON, atomically (write a temp file, then rename)."""
        payload = {
            "dataset": _dataset_to_json(self.dataset),
            "holdout": _dataset_to_json(self.holdout),
            "records": [asdict(r) for r in self.records],
            "labeled_ids": sorted(self.labeled_ids),
            "pending": [asdict(h) for h in self.pending],
            "rng_state": self.rng.bit_generator.state,
            "labels_submitted": self.labels_submitted,
            "in_flight": self.in_flight,
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, target)

    @classmethod
    def load(cls, path: str | Path) -> CampaignState:
        payload = json.loads(Path(path).read_text())
        rng = np.random.default_rng()
        rng.bit_generator.state = payload["rng_state"]
        return cls(
            dataset=_dataset_from_json(payload["dataset"]),
            holdout=_dataset_from_json(payload["holdout"]),
            records=[RoundRecord(**r) for r in payload["records"]],
            labeled_ids=set(payload["labeled_ids"]),
            pending=[JobHandle(**h) for h in payload["pending"]],
            rng=rng,
            labels_submitted=int(payload["labels_submitted"]),
            in_flight=payload["in_flight"],
        )


class Campaign:
    """Drives the active-learning loop against the frozen protocol seams.

    The labeler's ``submit``/``collect`` pair is the only external IO in the
    loop; everything else is deterministic given the state file, which is what
    makes kill-and-resume testable.
    """

    def __init__(
        self,
        labeler: Labeler,
        committee: Sequence[Model],
        selector: Selector,
        proposer: Proposer,
        config: CampaignConfig,
        domain_fence: DomainFence | None = None,
    ) -> None:
        if not committee:
            raise ValueError("committee must contain at least one model")
        self.labeler = labeler
        self.committee = list(committee)
        self.selector = selector
        self.proposer = proposer
        self.config = config
        self.domain_fence = domain_fence
        self.state_path = Path(config.checkpoint_dir) / STATE_FILENAME

    def run(self, initial: Dataset | None = None) -> list[RoundRecord]:
        """Execute rounds until a stop condition; resume from a checkpoint if one
        exists (``initial`` seeds a fresh campaign and is ignored on resume)."""
        if self.state_path.exists():
            state = CampaignState.load(self.state_path)
            logger.info("resuming from %s at round %d", self.state_path, len(state.records))
        else:
            state = self._fresh_state(initial if initial is not None else Dataset())
            self._checkpoint(state, "init")
        self._fit_committee(state, generation=len(state.records))
        if state.pending:
            # Resume path: re-collect persisted handles instead of resubmitting.
            logger.info("re-collecting %d pending handles", len(state.pending))
            self._collect_merge_retrain(state)
        while True:
            reason = self._stop_reason(state)
            if reason is not None:
                logger.info("campaign stopped: %s", reason)
                break
            self._run_round(state)
        return state.records

    def _fresh_state(self, initial: Dataset) -> CampaignState:
        rng = np.random.default_rng(self.config.seed)
        n = len(initial)
        n_hold = int(n * self.config.holdout_fraction)
        hold_idx = set(rng.permutation(n)[:n_hold].tolist())
        train, hold = Dataset(), Dataset()
        for i in range(n):
            dst = hold if i in hold_idx else train
            dst.structures.append(initial.structures[i])
            dst.energies.append(initial.energies[i])
            dst.forces.append(initial.forces[i])
        return CampaignState(dataset=train, holdout=hold, labeled_ids=train.ids | hold.ids, rng=rng)

    def _run_round(self, state: CampaignState) -> None:
        round_index = len(state.records)
        batch = min(self.config.batch_size, self.config.label_budget - state.labels_submitted)

        # Propose: oversample, then drop anything already claimed. The pool is
        # not persisted — after a crash a fresh pool is proposed, spending no labels.
        raw = self.proposer.propose(math.ceil(self.config.band_factor * batch), state.rng)
        pool: list[Structure] = []
        seen: set[str] = set()
        for s in raw:
            sid = s.structure_id
            if sid not in state.labeled_ids and sid not in seen:
                seen.add(sid)
                pool.append(s)
        self._checkpoint(state, "propose")

        # Fence: refuse to spend labels outside the trusted domain.
        if self.domain_fence is not None and pool:
            verdict = np.asarray(self.domain_fence(pool))
            keep = verdict if verdict.dtype == bool else verdict <= self.config.fence_threshold
            fenced = [s for s, k in zip(pool, keep.tolist(), strict=True) if k]
        else:
            fenced = pool
        self._checkpoint(state, "fence")

        # Select + submit: ids are claimed *now*, before any result exists.
        order = self.selector.rank(fenced, self.committee) if fenced else []
        selected = [fenced[i] for i in order[:batch]]
        handles = self.labeler.submit(selected) if selected else []
        state.labeled_ids.update(s.structure_id for s in selected)
        state.labels_submitted += len(handles)
        state.pending = list(handles)
        state.in_flight = {
            "round_index": round_index,
            "n_proposed": len(raw),
            "n_fenced_out": len(pool) - len(fenced),
            "n_submitted": len(handles),
        }
        self._checkpoint(state, "submit")

        self._collect_merge_retrain(state)

    def _collect_merge_retrain(self, state: CampaignState) -> None:
        info = state.in_flight or {}
        round_index = info.get("round_index", len(state.records))

        # Collect and merge share one durable point: results become durable only
        # via the merged dataset, and Dataset.merge is idempotent, so
        # re-collecting after a crash here cannot duplicate rows.
        results = self.labeler.collect(state.pending) if state.pending else []
        n_failed = sum(1 for r in results if r.status.is_failure)
        n_merged = state.dataset.merge(results)
        state.pending = []
        self._checkpoint(state, "collect+merge")

        self._fit_committee(state, generation=round_index + 1)
        rmse = self._holdout_rmse(state)
        record = RoundRecord(
            round_index=round_index,
            n_proposed=info.get("n_proposed", 0),
            n_fenced_out=info.get("n_fenced_out", 0),
            n_submitted=info.get("n_submitted", 0),
            n_failed=n_failed,
            n_merged=n_merged,
            dataset_size=len(state.dataset),
            holdout_rmse=rmse,
        )
        state.records.append(record)
        state.in_flight = None
        self._checkpoint(state, "retrain")
        logger.info(
            "round=%(round_index)d proposed=%(n_proposed)d fenced_out=%(n_fenced_out)d "
            "submitted=%(n_submitted)d failed=%(n_failed)d merged=%(n_merged)d "
            "dataset=%(dataset_size)d holdout_rmse=%(holdout_rmse)s",
            asdict(record),
        )

    def _fit_committee(self, state: CampaignState, generation: int) -> None:
        if len(state.dataset) == 0:
            return
        for member, model in enumerate(self.committee):
            # Seeds derive from (root seed, refit generation, member index) via
            # SeedSequence: members differ from each other, refits differ
            # between rounds, no ad-hoc arithmetic collisions.
            seq = np.random.SeedSequence(self.config.seed, spawn_key=(generation, member))
            model.fit(state.dataset, seed=int(seq.generate_state(1)[0]))

    def _holdout_rmse(self, state: CampaignState) -> float | None:
        if len(state.holdout) == 0 or len(state.dataset) == 0:
            return None
        preds = np.mean(
            [np.asarray(m.predict(state.holdout.structures)) for m in self.committee], axis=0
        )
        err = preds - state.holdout.energy_array()
        return float(np.sqrt(np.mean(err * err)))

    def _stop_reason(self, state: CampaignState) -> str | None:
        if len(state.records) >= self.config.max_rounds:
            return f"max_rounds ({self.config.max_rounds}) reached"
        if state.labels_submitted >= self.config.label_budget:
            return f"label_budget ({self.config.label_budget}) exhausted"
        if self._plateaued(state.records):
            return (
                f"holdout RMSE plateau (rel. improvement < {self.config.plateau_rel_tol} "
                f"for {self.config.plateau_patience} rounds)"
            )
        return None

    def _plateaued(self, records: Sequence[RoundRecord]) -> bool:
        rmses = [r.holdout_rmse for r in records if r.holdout_rmse is not None]
        patience = self.config.plateau_patience
        if len(rmses) < patience + 1:
            return False
        recent = rmses[-(patience + 1) :]
        for prev, curr in zip(recent, recent[1:], strict=False):
            improvement = (prev - curr) / prev if prev > 0 else 0.0
            if improvement >= self.config.plateau_rel_tol:
                return False
        return True

    def _checkpoint(self, state: CampaignState, phase: str) -> None:
        state.save(self.state_path)
        logger.debug("checkpoint after phase %r -> %s", phase, self.state_path)
