"""Descriptor-space domain membership: distances, fences, and extrapolation checks.

An interpolation model is trustworthy only near its training data, but "near"
has to be measured in some space. This module fixes that space (a geometric
descriptor per structure), measures how far a query lies from the training set
in it, and turns the measurement into an admit/reject fence plus a diagnostic
for the most dangerous failure mode: confident agreement far from the data.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from campaign_kit.protocols import Dataset, Predictions, Structure

__all__ = [
    "DomainFence",
    "TrainingDomain",
    "default_descriptor",
    "silent_extrapolation_check",
]


def default_descriptor(structure: Structure) -> np.ndarray:
    """Flattened, sorted inverse pairwise distances of a structure.

    Why this choice: distances make the descriptor invariant to translation and
    rotation, sorting makes it invariant to atom ordering, and the *inverse*
    weights short contacts — the region where an interatomic energy model is
    most sensitive — while distant pairs decay toward zero instead of dominating
    the Euclidean norm. Entries are sorted in descending order, so index 0 is
    always the closest contact.

    Limitations, stated up front: sorting discards which pair produced which
    entry and ignores element identity, so chemically distinct structures can
    collide; and the descriptor length depends on the atom count, so pools with
    mixed sizes need a custom ``descriptor_fn``. Structures with fewer than two
    atoms map to an empty vector. Coincident atoms produce ``inf`` entries;
    that is deliberate — such a geometry should look maximally far from any
    sane training set rather than be silently patched.
    """
    pos = structure.positions
    n = pos.shape[0]
    if n < 2:
        return np.zeros(0)
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=-1))
    upper = dist[np.triu_indices(n, k=1)]
    with np.errstate(divide="ignore"):
        inv = 1.0 / upper
    return np.sort(inv)[::-1].copy()


def _stack_descriptors(
    descriptor_fn: Callable[[Structure], np.ndarray],
    structures: Sequence[Structure],
) -> np.ndarray:
    """Apply the descriptor to every structure and stack into an (n, dim) matrix."""
    rows = [np.asarray(descriptor_fn(s), dtype=float).ravel() for s in structures]
    dims = {row.size for row in rows}
    if len(dims) > 1:
        raise ValueError(
            f"descriptor dimensionality is inconsistent across structures: {sorted(dims)}; "
            "distance comparisons need one shared descriptor space"
        )
    return np.vstack(rows)


def _pairwise_sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances between rows of `a` and rows of `b`."""
    sq = (a * a).sum(axis=1)[:, None] + (b * b).sum(axis=1)[None, :] - 2.0 * (a @ b.T)
    return np.maximum(sq, 0.0)


def _local_spread(predictions: Predictions) -> np.ndarray:
    """Per-structure disagreement: max per-atom spread, else the global scalar.

    The maximum over atoms is used because a structure can be well predicted on
    average while one local environment is extrapolating; a mean would average
    that signal away.
    """
    scores = np.asarray(predictions.per_structure_spread, dtype=float).copy()
    per_atom = predictions.per_atom_spread
    if per_atom is not None and len(per_atom) == scores.shape[0]:
        for i, atom_spread in enumerate(per_atom):
            arr = np.asarray(atom_spread, dtype=float)
            if arr.size > 0:
                scores[i] = float(arr.max())
    return scores


class TrainingDomain:
    """The region of descriptor space occupied by a training set.

    Built once from labeled structures, then queried many times. The primary
    measurement is `distance`: nearest-neighbour Euclidean distance in
    descriptor space, which asks "how far is this query from the closest thing
    the model has actually seen?" — a question a covariance model cannot answer
    for a multi-modal training set. `mahalanobis` is offered as a secondary,
    shape-aware measurement with its degenerate regime documented.
    """

    def __init__(
        self,
        data: Dataset | Sequence[Structure],
        descriptor_fn: Callable[[Structure], np.ndarray] = default_descriptor,
        covariance_regularization: float = 1e-6,
    ) -> None:
        """Build the domain from a `Dataset` or a plain list of structures.

        `covariance_regularization` is the ridge added to the empirical
        covariance (scaled by its mean diagonal) so that `mahalanobis` is
        always computable; it only matters in the degenerate regime described
        there.
        """
        structures = list(data.structures) if isinstance(data, Dataset) else list(data)
        if not structures:
            raise ValueError("TrainingDomain needs at least one training structure")
        if covariance_regularization <= 0.0:
            raise ValueError("covariance_regularization must be positive")
        self._descriptor_fn = descriptor_fn
        self._train = _stack_descriptors(descriptor_fn, structures)
        self._covariance_regularization = float(covariance_regularization)
        self._mean: np.ndarray | None = None
        self._cov_reg: np.ndarray | None = None

    @property
    def n_train(self) -> int:
        return int(self._train.shape[0])

    @property
    def dim(self) -> int:
        return int(self._train.shape[1])

    def _descriptors(self, structures: Sequence[Structure]) -> np.ndarray:
        queries = _stack_descriptors(self._descriptor_fn, structures)
        if queries.shape[1] != self.dim:
            raise ValueError(f"query descriptor dim {queries.shape[1]} != training dim {self.dim}")
        return queries

    def distance(self, structures: Sequence[Structure]) -> np.ndarray:
        """Euclidean distance from each query to its nearest training point.

        Nearest-neighbour rather than distance-to-mean: training sets collected
        along reaction paths are curved and multi-modal, and the centroid of
        such a set can lie in a region with no data at all.
        """
        queries = self._descriptors(structures)
        if queries.shape[0] == 0:
            return np.zeros(0)
        sq = _pairwise_sq_dists(queries, self._train)
        return np.sqrt(sq.min(axis=1))

    def mahalanobis(self, structures: Sequence[Structure]) -> np.ndarray:
        """Mahalanobis distance from each query to the training mean.

        Useful when the training cloud is a single anisotropic blob: it rescales
        each direction by how much the data actually varies along it. It
        degenerates when ``n_train < dim`` (common here, since descriptor
        dimension grows quadratically with atom count): the empirical covariance
        is then rank-deficient, the ridge term dominates every unsampled
        direction, and distances along those directions measure the
        regularization constant rather than the data. In that regime prefer
        `distance`, which stays meaningful.
        """
        queries = self._descriptors(structures)
        if queries.shape[0] == 0:
            return np.zeros(0)
        if self._mean is None or self._cov_reg is None:
            mean = self._train.mean(axis=0)
            centered = self._train - mean
            denom = max(self.n_train - 1, 1)
            cov = (centered.T @ centered) / denom
            diag_scale = float(np.trace(cov)) / self.dim if np.trace(cov) > 0.0 else 1.0
            cov_reg = cov + self._covariance_regularization * diag_scale * np.eye(self.dim)
            self._mean = mean
            self._cov_reg = cov_reg
        diffs = queries - self._mean
        solved = np.linalg.solve(self._cov_reg, diffs.T)
        sq = np.maximum((diffs.T * solved).sum(axis=0), 0.0)
        return np.sqrt(sq)


class DomainFence:
    """Admit/reject gate on descriptor-space distance to the training set.

    `threshold` is the largest nearest-neighbour distance (same units as the
    descriptor space) still considered inside the domain. There is no default:
    a sensible value depends on the descriptor and the density of the training
    set, so the caller must either supply one explicitly or derive one from the
    training data via `from_train_quantile`.
    """

    def __init__(self, domain: TrainingDomain, threshold: float) -> None:
        if threshold <= 0.0:
            raise ValueError("threshold must be positive")
        self._domain = domain
        self._threshold = float(threshold)

    @property
    def threshold(self) -> float:
        return self._threshold

    @classmethod
    def from_train_quantile(cls, domain: TrainingDomain, quantile: float) -> DomainFence:
        """Calibrate the threshold from the training set's own spacing.

        The threshold is set to the given quantile of leave-one-out
        nearest-neighbour distances among the training points. The rationale: a
        query no farther from the training set than training points typically
        are from each other is being interpolated, not extrapolated. Requires
        at least two training points.
        """
        if not 0.0 < quantile <= 1.0:
            raise ValueError("quantile must be in (0, 1]")
        if domain.n_train < 2:
            raise ValueError("from_train_quantile needs at least two training points")
        sq = _pairwise_sq_dists(domain._train, domain._train)
        np.fill_diagonal(sq, np.inf)
        nn = np.sqrt(sq.min(axis=1))
        threshold = float(np.quantile(nn, quantile))
        if threshold <= 0.0:
            raise ValueError(
                "calibrated threshold is zero (duplicate training structures dominate "
                "this quantile); raise the quantile or deduplicate the training set"
            )
        return cls(domain, threshold)

    def check(self, structures: Sequence[Structure]) -> np.ndarray:
        """Boolean admit mask: True where the structure is inside the fence."""
        return self._domain.distance(structures) <= self._threshold

    def reject_indices(self, structures: Sequence[Structure]) -> list[int]:
        """Indices of structures outside the fence, in pool order."""
        mask = self.check(structures)
        return [int(i) for i in np.flatnonzero(~mask)]


def silent_extrapolation_check(
    structures: Sequence[Structure],
    committee_predictions: Predictions,
    domain: TrainingDomain,
    distance_quantile: float = 0.75,
    spread_quantile: float = 0.25,
) -> list[str]:
    """Warn about structures that are far from the training data yet show low
    committee spread.

    Low ensemble variance is not a sufficient out-of-domain signal. Committee
    members share the same architecture, the same training set, and the same
    inductive biases; far from the data they can agree confidently and be wrong
    together, because their errors are correlated exactly where no data
    constrains them. Spread measures model-to-model variance, not
    model-to-truth error. This check therefore pairs the spread with an
    independent geometric criterion — descriptor-space distance to the training
    set — and flags the combination "far and quiet", which is the case neither
    signal catches alone.

    This is a mitigation, not a solution. It flags a suspicious combination of
    two proxies; it does not rank per-point error, and no continuous signal
    available here is known to do so reliably. That remains an open problem,
    and downstream code should treat flagged structures as candidates for
    labeling, not as measured failures.

    Thresholds are batch-relative: a structure is "far" when its distance is at
    or above `distance_quantile` of the batch's distances, and "quiet" when its
    local spread (max per-atom spread, or the per-structure scalar when
    per-atom is unavailable) is at or below `spread_quantile` of the batch's
    spreads. Both quantiles are ordinary parameters with generic defaults;
    with very small batches the quantiles are noisy and the check loses power.

    Returns one human-readable warning string per flagged structure, in pool
    order; an empty list means nothing was flagged.
    """
    n = len(structures)
    if n == 0:
        return []
    if len(committee_predictions) != n:
        raise ValueError(f"predictions cover {len(committee_predictions)} structures, pool has {n}")
    distances = domain.distance(structures)
    spreads = _local_spread(committee_predictions)
    dist_cut = float(np.quantile(distances, distance_quantile))
    spread_cut = float(np.quantile(spreads, spread_quantile))
    warnings: list[str] = []
    for i in range(n):
        if distances[i] >= dist_cut and spreads[i] <= spread_cut:
            warnings.append(
                f"structure {i} (id {structures[i].structure_id}): descriptor distance "
                f"{distances[i]:.4g} >= batch q{distance_quantile:g} ({dist_cut:.4g}) while "
                f"committee spread {spreads[i]:.4g} <= batch q{spread_quantile:g} "
                f"({spread_cut:.4g}); low spread here is not evidence of accuracy"
            )
    return warnings
