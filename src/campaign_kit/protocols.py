"""Core interfaces of the campaign toolkit.

Everything in this package is written against the three small interfaces below
(`Labeler`, `Model`, `Selector`) plus a handful of plain dataclasses. The point
of the seam: in production the labeler dispatches multi-hour cluster jobs, while
in the demos and tests it is a closed-form analytic function. Nothing downstream
can tell the difference, which is what makes the campaign driver testable on a
laptop.

Failure is a first-class outcome. A label request can come back as a
non-converged calculation, a wall-clock timeout, or a dead node — and campaign
code that pretends otherwise falls over on the first real cluster run. The type
system here makes the failure path visible: `LabelResult` always carries a
`LabelStatus`, and downstream consumers must decide what to do with non-success
statuses rather than crash on a missing number.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = [
    "Structure",
    "LabelStatus",
    "JobHandle",
    "LabelResult",
    "Predictions",
    "Dataset",
    "Labeler",
    "Model",
    "Selector",
    "Proposer",
]


@dataclass(frozen=True)
class Structure:
    """An atomistic configuration: element symbols plus Cartesian positions.

    Identity is content-based (see `structure_id`), so the same geometry
    submitted twice deduplicates naturally in `Dataset.merge` — that property is
    what makes re-collection of an already-collected batch idempotent.
    """

    species: tuple[str, ...]
    positions: np.ndarray  # shape (n_atoms, 3), Angstrom or demo units
    tag: str = ""

    def __post_init__(self) -> None:
        pos = np.asarray(self.positions, dtype=float)
        if pos.ndim != 2 or pos.shape[1] != 3 or pos.shape[0] != len(self.species):
            raise ValueError(f"positions must have shape ({len(self.species)}, 3), got {pos.shape}")
        object.__setattr__(self, "positions", pos)

    @property
    def n_atoms(self) -> int:
        return len(self.species)

    @property
    def structure_id(self) -> str:
        """Stable content hash of species + positions (12 hex chars).

        Positions are rounded to 1e-10 before hashing so that round-tripping
        through JSON/text serialization cannot silently change identity.
        """
        h = hashlib.sha1()
        h.update(",".join(self.species).encode())
        h.update(np.round(self.positions, 10).tobytes())
        return h.hexdigest()[:12]


class LabelStatus(Enum):
    """Terminal outcome of one label request.

    The distinction matters operationally: a convergence failure at the same
    geometry will fail again and should be dropped, while a timeout is often
    recoverable by resubmitting with a longer wall clock (see
    `scheduler.base.RetryPolicy`).
    """

    SUCCEEDED = "succeeded"
    FAILED_CONVERGENCE = "failed_convergence"
    FAILED_TIMEOUT = "failed_timeout"
    FAILED_NODE = "failed_node"
    FAILED_UNKNOWN = "failed_unknown"

    @property
    def is_failure(self) -> bool:
        return self is not LabelStatus.SUCCEEDED


@dataclass(frozen=True)
class JobHandle:
    """Reference to one in-flight label request.

    `attempt` counts resubmissions of the same structure, so the scheduler's
    retry cap is enforceable per job rather than globally.
    """

    job_id: str
    structure_id: str
    attempt: int = 0
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class LabelResult:
    """Outcome of one label request — success or classified failure.

    `energy`/`forces` are None unless `status` is SUCCEEDED; consumers should
    branch on `status`, never on the presence of the payload.
    """

    structure: Structure
    status: LabelStatus
    energy: float | None = None
    forces: np.ndarray | None = None  # shape (n_atoms, 3) when present
    message: str = ""

    @classmethod
    def success(
        cls,
        structure: Structure,
        energy: float,
        forces: np.ndarray | None = None,
    ) -> LabelResult:
        return cls(structure=structure, status=LabelStatus.SUCCEEDED, energy=energy, forces=forces)

    @classmethod
    def failure(cls, structure: Structure, status: LabelStatus, message: str = "") -> LabelResult:
        if not status.is_failure:
            raise ValueError("failure() requires a failure status")
        return cls(structure=structure, status=status, message=message)


@dataclass
class Predictions:
    """Committee output for a batch of structures.

    Two disagreement signals are provided deliberately:

    - `per_structure_spread[i]`: scalar committee spread for structure i.
    - `per_atom_spread[i]`: shape (n_atoms,) local spread for structure i.

    The local one is the important one. A structure can be confidently
    predicted overall while one region of it is extrapolating; a global scalar
    averages that signal away. Selection ranks on the local maximum, not the
    global mean.
    """

    energies: np.ndarray  # (n,) committee-mean predictions
    per_structure_spread: np.ndarray  # (n,)
    per_atom_spread: list[np.ndarray]  # n arrays of shape (n_atoms,)

    def __len__(self) -> int:
        return int(self.energies.shape[0])


@dataclass
class Dataset:
    """Labeled training set with content-hash deduplication.

    `merge` is idempotent: merging the same results twice cannot create
    duplicate rows, because rows are keyed by `Structure.structure_id`. This is
    the property the campaign checkpoint/resume test relies on.
    """

    structures: list[Structure] = field(default_factory=list)
    energies: list[float] = field(default_factory=list)
    forces: list[np.ndarray | None] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.structures)

    @property
    def ids(self) -> set[str]:
        return {s.structure_id for s in self.structures}

    def merge(self, results: Iterable[LabelResult]) -> int:
        """Merge successful results; skip failures and duplicates.

        Returns the number of rows actually added.
        """
        known = self.ids
        added = 0
        for r in results:
            if r.status.is_failure:
                continue
            sid = r.structure.structure_id
            if sid in known:
                continue
            assert r.energy is not None
            self.structures.append(r.structure)
            self.energies.append(float(r.energy))
            self.forces.append(r.forces)
            known.add(sid)
            added += 1
        return added

    def energy_array(self) -> np.ndarray:
        return np.asarray(self.energies, dtype=float)


@runtime_checkable
class Labeler(Protocol):
    """Produces ground-truth labels for candidate structures.

    In production this dispatches cluster jobs and `collect` blocks (or polls)
    until every handle reaches a terminal state. In the demos it evaluates an
    analytic function and returns immediately. `collect` must return one
    `LabelResult` per handle — failures included — and must never raise merely
    because a calculation failed.
    """

    def submit(self, candidates: Sequence[Structure]) -> list[JobHandle]: ...

    def collect(self, handles: Sequence[JobHandle]) -> list[LabelResult]: ...


@runtime_checkable
class Model(Protocol):
    """A single trainable regressor: energy per structure.

    `seed` controls every source of stochasticity in the fit (init, shuffling),
    so a committee of models differing only in seed measures model variance,
    not data variance.
    """

    def fit(self, dataset: Dataset, seed: int) -> None: ...

    def predict(self, structures: Sequence[Structure]) -> np.ndarray: ...

    def predict_per_atom(self, structures: Sequence[Structure]) -> list[np.ndarray]:
        """Optional per-atom energy decomposition; default falls back to uniform."""
        ...


@runtime_checkable
class Selector(Protocol):
    """Ranks a candidate pool; lower index = select first."""

    def rank(self, pool: Sequence[Structure], committee: object) -> list[int]: ...


@runtime_checkable
class Proposer(Protocol):
    """Generates candidate structures for one campaign round."""

    def propose(self, n: int, rng: np.random.Generator) -> list[Structure]: ...
