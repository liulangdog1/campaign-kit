"""Demo 01: a complete active-learning campaign on an analytic multi-well surface.

What this shows
---------------
The full loop from ``campaign_kit.loop.Campaign`` — propose, fence, select,
label, merge, retrain, stop — running end to end in well under a minute. The
expensive part of a real campaign, the labeler, is replaced by a closed-form
multi-well energy surface, so every mechanism of the loop can be watched
directly. Because the labeler seam (``submit``/``collect``) is the only
difference from production, the per-round table printed here is exactly the
bookkeeping a cluster campaign would produce.

The surface is a two-dimensional multi-well landscape (a soft confinement plus
three Gaussian wells) embedded into a 3-atom structure: two "frame" atoms pin
a reference edge, and the third atom's in-plane position is the surface
coordinate. The embedding exists because everything in campaign_kit speaks
``Structure`` — descriptors, fences and selectors operate on geometry, never
on the hidden (x, y) — which keeps the demo honest about what the library
actually sees.

Failure is part of the demo on purpose: the labeler injects deterministic
convergence failures at a small rate, so the ``failed`` column is nonzero and
the budget accounting (submissions cost budget, failures buy nothing) is
visible.

Run:  python examples/01_active_learning_loop.py
"""

from __future__ import annotations

import math
import tempfile
import warnings
from collections.abc import Sequence
from typing import cast

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from campaign_kit.backends import SklearnMLPModel
from campaign_kit.domain import DomainFence, TrainingDomain
from campaign_kit.loop import Campaign, CampaignConfig, RoundRecord
from campaign_kit.protocols import (
    Dataset,
    JobHandle,
    LabelResult,
    LabelStatus,
    Model,
    Predictions,
    Structure,
)
from campaign_kit.selection import CompositeSelector

# The trusted campaign region of the 2D surface coordinate. Kept strictly at
# y > 0: the demo descriptor is distance-based, and distances cannot tell a
# point from its mirror image across the frame axis, so the region is chosen
# on one side of that axis.
X_RANGE: tuple[float, float] = (0.0, 3.2)
Y_RANGE: tuple[float, float] = (0.2, 2.4)

# Gaussian wells as (depth, x0, y0, width), in generic energy / length units.
# Three wells of different depths make the surface genuinely multi-modal, so
# a model trained on a handful of points has something real left to learn.
WELLS: tuple[tuple[float, float, float, float], ...] = (
    (-6.0, 0.7, 0.7, 0.40),
    (-4.5, 1.6, 1.8, 0.30),
    (-7.5, 2.5, 0.6, 0.35),
)

# Separation of the two fixed frame atoms in the 3-atom embedding.
FRAME_SPAN: float = 3.0


def multiwell_energy(x: float, y: float) -> float:
    """Analytic surface: soft harmonic confinement plus three Gaussian wells."""
    energy = 0.4 * ((x - 1.6) ** 2 + (y - 1.2) ** 2)
    for depth, x0, y0, width in WELLS:
        energy += depth * math.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2.0 * width**2))
    return energy


def embed_point(x: float, y: float) -> Structure:
    """Embed a 2D surface coordinate as a 3-atom probe structure.

    Atoms A and B are a fixed frame; atom C's in-plane position *is* the
    surface coordinate. The indirection matters: the library never sees
    (x, y), only geometry, so descriptors and fences work exactly as they
    would on a real structure.
    """
    positions = np.array([[0.0, 0.0, 0.0], [FRAME_SPAN, 0.0, 0.0], [x, y, 0.0]], dtype=float)
    return Structure(species=("A", "B", "C"), positions=positions)


def surface_energy(structure: Structure) -> float:
    """Recover the surface coordinate from the probe atom and evaluate the surface."""
    x, y, _ = structure.positions[2]
    return multiwell_energy(float(x), float(y))


class AnalyticLabeler:
    """Instant ``Labeler``: the analytic surface stands in for a batch queue.

    Two deliberate realism features:

    - A small, deterministic failure rate. Whether a structure fails is a
      pseudo-random draw keyed on (seed, structure id), so the same structure
      fails on every attempt — which is how a genuine convergence failure
      behaves, and why the campaign driver claims ids at submit time instead
      of retrying them forever.
    - Pending work is held in memory only. A production labeler persists its
      queue so a process restart can re-collect; this demo one cannot, which
      is fine because the demo never restarts.
    """

    def __init__(self, failure_rate: float = 0.08, seed: int = 0) -> None:
        if not 0.0 <= failure_rate < 1.0:
            raise ValueError("failure_rate must be in [0, 1)")
        self._failure_rate = failure_rate
        self._seed = seed
        self._pending: dict[str, Structure] = {}

    def submit(self, candidates: Sequence[Structure]) -> list[JobHandle]:
        handles: list[JobHandle] = []
        for structure in candidates:
            sid = structure.structure_id
            self._pending[sid] = structure
            handles.append(JobHandle(job_id=f"analytic-{sid}", structure_id=sid))
        return handles

    def collect(self, handles: Sequence[JobHandle]) -> list[LabelResult]:
        results: list[LabelResult] = []
        for handle in handles:
            structure = self._pending.pop(handle.structure_id)
            if self._fails(handle.structure_id):
                results.append(
                    LabelResult.failure(
                        structure,
                        LabelStatus.FAILED_CONVERGENCE,
                        message="injected demo failure",
                    )
                )
            else:
                results.append(LabelResult.success(structure, surface_energy(structure)))
        return results

    def _fails(self, structure_id: str) -> bool:
        # Content-keyed determinism: the id is a content hash, so the verdict
        # is a fixed property of the geometry for a given labeler seed.
        draw = np.random.default_rng([self._seed, int(structure_id, 16)]).random()
        return bool(draw < self._failure_rate)


class RegionProposer:
    """Uniform candidates over the campaign region, plus occasional wild ones.

    The wild fraction produces near-collision geometries (the probe atom
    almost on top of a frame atom) on purpose: real candidate generators
    (dynamics, geometry perturbation) sometimes emit unphysical structures,
    and the demo fence should be seen rejecting them rather than never
    firing. Short contacts are the direction the inverse-distance descriptor
    treats as far from any sane training set, so these are exactly the
    candidates a distance fence can catch.
    """

    def __init__(self, wild_fraction: float = 0.12) -> None:
        self._wild_fraction = wild_fraction

    def propose(self, n: int, rng: np.random.Generator) -> list[Structure]:
        out: list[Structure] = []
        for _ in range(n):
            if rng.random() < self._wild_fraction:
                # Almost on top of frame atom A: an unphysical short contact.
                x = float(rng.uniform(0.02, 0.10))
                y = float(rng.uniform(0.02, 0.10))
            else:
                x = float(rng.uniform(*X_RANGE))
                y = float(rng.uniform(*Y_RANGE))
            out.append(embed_point(x, y))
        return out


def _committee_predictions(models: Sequence[Model], pool: Sequence[Structure]) -> Predictions:
    """Committee statistics (mean, global spread, per-atom spread) from raw models.

    Mirrors ``campaign_kit.committee.Committee.predict_mean_and_spread``, but
    for an externally managed list of already-fitted models — which is what
    ``Campaign`` holds. Spreads are population standard deviations (ddof=0):
    the committee is the whole population of interest, not a sample.
    """
    energies = np.stack([np.asarray(m.predict(pool), dtype=float) for m in models])
    per_atom_by_member = [m.predict_per_atom(pool) for m in models]
    per_atom_spread = [
        np.stack([per_atom_by_member[i][j] for i in range(len(models))]).std(axis=0)
        for j in range(len(pool))
    ]
    return Predictions(
        energies=energies.mean(axis=0),
        per_structure_spread=energies.std(axis=0),
        per_atom_spread=per_atom_spread,
    )


class ModelListSelector:
    """Adapts ``Campaign``'s committee (a plain model list) to ``CompositeSelector``.

    The ``Selector`` protocol deliberately leaves the committee argument
    untyped, so a driver can hand over whatever it holds — here, the sequence
    of fitted models. ``CompositeSelector`` scores a ``Predictions`` object,
    so this wrapper computes the committee statistics once per round and
    delegates the actual ranking.
    """

    def __init__(self, inner: CompositeSelector) -> None:
        self._inner = inner

    def rank(self, pool: Sequence[Structure], committee: object) -> list[int]:
        if len(pool) == 0:
            return []
        models = cast(Sequence[Model], committee)
        return self._inner.rank(pool, _committee_predictions(models, pool))


def print_round_table(records: Sequence[RoundRecord]) -> None:
    header = (
        f"{'round':>5} {'proposed':>9} {'fenced out':>11} {'submitted':>10} "
        f"{'failed':>7} {'merged':>7} {'dataset':>8} {'holdout RMSE':>13}"
    )
    print(header)
    print("-" * len(header))
    for r in records:
        rmse = "n/a" if r.holdout_rmse is None else f"{r.holdout_rmse:.3f}"
        print(
            f"{r.round_index:>5d} {r.n_proposed:>9d} {r.n_fenced_out:>11d} "
            f"{r.n_submitted:>10d} {r.n_failed:>7d} {r.n_merged:>7d} "
            f"{r.dataset_size:>8d} {rmse:>13}"
        )


def main() -> None:
    # Demo-size MLP fits stop on max_iter rather than on the optimizer
    # tolerance; the holdout-RMSE column is the convergence signal that
    # matters here, so the per-fit warnings are noise.
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    print("Demo 01: active-learning campaign on an analytic multi-well surface")
    print()

    labeler = AnalyticLabeler(failure_rate=0.08, seed=5)

    # Seed data: a small random sample of the region, labeled through the same
    # submit/collect seam the campaign itself uses (so a couple of seed points
    # may fail, exactly as they would on a queue).
    seed_rng = np.random.default_rng(2)
    seed_structures = [
        embed_point(float(seed_rng.uniform(*X_RANGE)), float(seed_rng.uniform(*Y_RANGE)))
        for _ in range(48)
    ]
    initial = Dataset()
    initial.merge(labeler.collect(labeler.submit(seed_structures)))
    print(
        f"seed dataset: {len(initial)} labeled structures "
        f"({len(seed_structures) - len(initial)} seed labels failed)"
    )

    # The fence is calibrated once, from the seed data's own spacing: reject
    # anything farther from the seed set than seed points ever are from each
    # other. Static by choice — the demo shows it cutting the proposer's wild
    # candidates before any label is spent on them.
    domain = TrainingDomain(initial)
    fence = DomainFence.from_train_quantile(domain, quantile=1.0)
    print(f"domain fence threshold (from seed spacing): {fence.threshold:.3f}")
    print()

    committee: list[Model] = [
        SklearnMLPModel(hidden_layer_sizes=(64, 64), max_iter=3000) for _ in range(4)
    ]
    selector = ModelListSelector(CompositeSelector(batch_size=16, band_factor=3.0, seed=11))

    with tempfile.TemporaryDirectory(prefix="campaign_demo01_") as checkpoint_dir:
        config = CampaignConfig(
            max_rounds=8,
            label_budget=128,
            batch_size=16,
            band_factor=4.0,
            plateau_patience=3,
            plateau_rel_tol=0.005,
            seed=7,
            checkpoint_dir=checkpoint_dir,
            holdout_fraction=0.3,
        )
        campaign = Campaign(
            labeler=labeler,
            committee=committee,
            selector=selector,
            proposer=RegionProposer(),
            config=config,
            domain_fence=fence.check,
        )
        records = campaign.run(initial)

    print_round_table(records)

    rmses = [r.holdout_rmse for r in records if r.holdout_rmse is not None]
    submitted = sum(r.n_submitted for r in records)
    print()
    print(
        f"holdout RMSE {rmses[0]:.3f} -> {rmses[-1]:.3f} "
        f"({100.0 * (1.0 - rmses[-1] / rmses[0]):.0f}% lower) "
        f"for {submitted} submitted labels on top of {len(seed_structures)} seed labels"
    )


if __name__ == "__main__":
    main()
