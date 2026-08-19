"""Shared test doubles: fast stand-ins for labelers, models, and proposers.

The doubles mirror the operational semantics of the real components rather
than their cost. The counting labeler spends its "compute" at submit time --
exactly like a cluster backend, where dispatching the job is the expensive
act and collecting only reads results back. That choice is what lets the
resume test equate "labeler work" with "structures submitted" and detect
double-spending across a crash/resume boundary.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from campaign_kit.domain import default_descriptor
from campaign_kit.protocols import (
    Dataset,
    JobHandle,
    LabelResult,
    Model,
    Predictions,
    Structure,
)
from campaign_kit.selection import CompositeSelector

BASE_SPECIES: tuple[str, ...] = ("C", "O", "H")
BASE_POSITIONS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.3, 0.0, 0.0],
        [0.4, 1.1, 0.2],
    ]
)


def analytic_energy(structure: Structure) -> float:
    """Smooth closed-form target so every test double labels consistently."""
    d = default_descriptor(structure)
    return float(d.sum() + 0.5 * float((d * d).sum()))


def jittered_structures(
    n: int,
    scale: float,
    seed: int,
    base: np.ndarray | None = None,
    species: tuple[str, ...] = BASE_SPECIES,
) -> list[Structure]:
    """Random displacements of one base geometry; continuous, so ids never collide."""
    rng = np.random.default_rng(seed)
    positions = BASE_POSITIONS if base is None else base
    return [
        Structure(species, positions + scale * rng.standard_normal(positions.shape))
        for _ in range(n)
    ]


def labeled_dataset(structures: Sequence[Structure]) -> Dataset:
    """A Dataset labeled by the analytic target, bypassing any labeler."""
    ds = Dataset()
    for s in structures:
        ds.structures.append(s)
        ds.energies.append(analytic_energy(s))
        ds.forces.append(None)
    return ds


class TinyLinearModel:
    """`Model` double: ridge least squares on the default descriptor.

    The fitted weights get a seed-scaled jitter so a committee of these
    models disagrees deterministically -- spread-based selection then has a
    real, reproducible signal without any expensive backend.
    """

    def __init__(self, jitter: float = 1e-3) -> None:
        self._jitter = float(jitter)
        self._weights: np.ndarray | None = None

    def fit(self, dataset: Dataset, seed: int) -> None:
        x = self._features(dataset.structures)
        y = dataset.energy_array()
        ridge = 1e-6 * np.eye(x.shape[1])
        weights = np.linalg.solve(x.T @ x + ridge, x.T @ y)
        rng = np.random.default_rng(seed)
        self._weights = weights + self._jitter * rng.standard_normal(weights.shape)

    def predict(self, structures: Sequence[Structure]) -> np.ndarray:
        if self._weights is None:
            raise RuntimeError("TinyLinearModel.fit must be called before predicting")
        return self._features(structures) @ self._weights

    def predict_per_atom(self, structures: Sequence[Structure]) -> list[np.ndarray]:
        totals = self.predict(structures)
        return [np.full(s.n_atoms, e / s.n_atoms) for s, e in zip(structures, totals, strict=True)]

    @staticmethod
    def _features(structures: Sequence[Structure]) -> np.ndarray:
        rows = np.vstack([default_descriptor(s) for s in structures])
        return np.hstack([rows, np.ones((rows.shape[0], 1))])


class CountingLabeler:
    """`Labeler` double that pays for each label exactly once, at submit.

    ``collect`` returns stored results without recomputing, mirroring how a
    cluster labeler polls for finished jobs. ``label_calls`` therefore counts
    compute spent; comparing it against the number of unique submitted ids
    detects any double submission after a crash and resume.
    """

    def __init__(self) -> None:
        self.label_calls = 0
        self.submitted_ids: list[str] = []
        self._results: dict[str, LabelResult] = {}
        self._counter = 0

    def submit(self, candidates: Sequence[Structure]) -> list[JobHandle]:
        handles: list[JobHandle] = []
        for s in candidates:
            self._counter += 1
            job_id = f"job-{self._counter}"
            self.label_calls += 1
            self.submitted_ids.append(s.structure_id)
            self._results[job_id] = LabelResult.success(s, analytic_energy(s))
            handles.append(JobHandle(job_id=job_id, structure_id=s.structure_id))
        return handles

    def collect(self, handles: Sequence[JobHandle]) -> list[LabelResult]:
        return [self._results[h.job_id] for h in handles]


class CrashOnCollect:
    """Wraps a labeler and raises on the Nth collect call -- a mid-round kill.

    The crash lands after the driver's submit checkpoint, which is exactly
    the window where a naive loop would resubmit (and re-pay for) the same
    structures on resume.
    """

    def __init__(self, inner: CountingLabeler, crash_on_call: int) -> None:
        self.inner = inner
        self._crash_on_call = crash_on_call
        self._calls = 0
        self._armed = True

    def submit(self, candidates: Sequence[Structure]) -> list[JobHandle]:
        return self.inner.submit(candidates)

    def collect(self, handles: Sequence[JobHandle]) -> list[LabelResult]:
        self._calls += 1
        if self._armed and self._calls == self._crash_on_call:
            self._armed = False
            raise RuntimeError("simulated kill")
        return self.inner.collect(handles)


class JitterProposer:
    """`Proposer` double: jitter one base geometry with the campaign rng."""

    def __init__(self, scale: float = 0.15) -> None:
        self._scale = scale

    def propose(self, n: int, rng: np.random.Generator) -> list[Structure]:
        return [
            Structure(
                BASE_SPECIES,
                BASE_POSITIONS + self._scale * rng.standard_normal(BASE_POSITIONS.shape),
            )
            for _ in range(n)
        ]


class ModelListAdapter:
    """Presents a plain list of models as an object with ``predict -> Predictions``.

    The campaign driver hands its selector the raw model list, while the
    shipped selectors want either a `Predictions` or an object exposing
    ``predict(pool)``; this adapter is the bridge a driver-level integration
    needs.
    """

    def __init__(self, models: Sequence[Model]) -> None:
        self._models = list(models)

    def predict(self, pool: Sequence[Structure]) -> Predictions:
        stacked = np.stack([np.asarray(m.predict(pool), dtype=float) for m in self._models])
        member_pa = [m.predict_per_atom(pool) for m in self._models]
        per_atom_spread = [
            np.stack([pa[j] for pa in member_pa]).std(axis=0) for j in range(len(pool))
        ]
        return Predictions(
            energies=stacked.mean(axis=0),
            per_structure_spread=stacked.std(axis=0),
            per_atom_spread=per_atom_spread,
        )


class AdapterSelector:
    """`Selector` for the loop tests: composite selection over the model list."""

    def __init__(self, batch_size: int, seed: int = 0) -> None:
        self._inner = CompositeSelector(batch_size=batch_size, seed=seed)

    def rank(self, pool: Sequence[Structure], committee: object) -> list[int]:
        assert isinstance(committee, list)  # the campaign driver passes its model list
        return self._inner.rank(pool, ModelListAdapter(committee))
