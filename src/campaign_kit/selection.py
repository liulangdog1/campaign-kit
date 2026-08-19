"""Candidate selection policies for one active-learning round.

Three selectors, one design decision. `QueryByCommittee` ranks by where the
committee disagrees; `FarthestPointSampling` maximizes geometric diversity; and
`CompositeSelector` chains them, because each criterion alone spends the
labeling budget badly (see its docstring for the argument).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

from campaign_kit.domain import default_descriptor
from campaign_kit.protocols import Predictions, Structure

__all__ = [
    "CompositeSelector",
    "FarthestPointSampling",
    "QueryByCommittee",
]


def _as_predictions(pool: Sequence[Structure], committee: object) -> Predictions:
    """Resolve the `committee` argument into a `Predictions` for this pool.

    The `Selector` protocol types `committee` as `object` so drivers can pass
    either a live committee (anything with ``predict(pool) -> Predictions``) or
    precomputed `Predictions` — the latter keeps selectors testable without
    fitting any model.
    """
    if isinstance(committee, Predictions):
        predictions = committee
    else:
        predict = getattr(committee, "predict", None)
        if not callable(predict):
            raise TypeError(
                "committee must be a Predictions instance or expose predict(pool) -> Predictions"
            )
        result = predict(pool)
        if not isinstance(result, Predictions):
            raise TypeError(
                f"committee.predict must return Predictions, got {type(result).__name__}"
            )
        predictions = result
    if len(predictions) != len(pool):
        raise ValueError(f"predictions cover {len(predictions)} structures, pool has {len(pool)}")
    return predictions


def _local_disagreement(predictions: Predictions) -> np.ndarray:
    """Per-structure score: max per-atom spread, else the per-structure scalar.

    The maximum over atoms — not the mean — is the score, because a structure
    can be confidently predicted overall while a single local environment is
    extrapolating; averaging hides exactly the atoms worth labeling.
    """
    scores = np.asarray(predictions.per_structure_spread, dtype=float).copy()
    per_atom = predictions.per_atom_spread
    if per_atom is not None and len(per_atom) == scores.shape[0]:
        for i, atom_spread in enumerate(per_atom):
            arr = np.asarray(atom_spread, dtype=float)
            if arr.size > 0:
                scores[i] = float(arr.max())
    return scores


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
            "farthest-point selection needs one shared descriptor space"
        )
    return np.vstack(rows)


class QueryByCommittee:
    """Rank candidates by descending local committee disagreement.

    Implements the `Selector` protocol. Disagreement is a cheap proxy for
    expected model improvement: where committee members differ, at least one of
    them is wrong, so a label there is guaranteed to correct somebody. The
    score is local (max over atoms) for the reason given in
    `_local_disagreement`; ties keep pool order, so the ranking is
    deterministic.

    Query-by-committee in the sense of Seung, Opper & Sompolinsky,
    COLT 1992.
    """

    def rank(self, pool: Sequence[Structure], committee: object) -> list[int]:
        if len(pool) == 0:
            return []
        predictions = _as_predictions(pool, committee)
        scores = _local_disagreement(predictions)
        order = np.argsort(-scores, kind="stable")
        return [int(i) for i in order]


class FarthestPointSampling:
    """Greedy max-min diversity selection in a descriptor space.

    Each pick is the pool point farthest (Euclidean, in descriptor space) from
    everything already picked — the classic 2-approximation to the k-center
    problem, which is what "cover the pool with k labels" actually is.

    Determinism: given the same pool and an equally seeded `rng`, `select`
    returns the same indices. The rng is consumed only for the first pick; when
    a `reference` set is provided the first pick is instead the pool point
    farthest from the reference (useful for extending an existing training set
    rather than re-covering it) and the rng is not consumed at all.

    Greedy max-min (k-center) selection in the sense of Gonzalez,
    Theor. Comput. Sci. 38, 293-306 (1985).
    """

    def __init__(
        self,
        descriptor_fn: Callable[[Structure], np.ndarray],
        reference: Sequence[Structure] | None = None,
    ) -> None:
        self._descriptor_fn = descriptor_fn
        self._reference: np.ndarray | None = None
        if reference is not None and len(reference) > 0:
            self._reference = _stack_descriptors(descriptor_fn, reference)

    def select(self, pool: Sequence[Structure], k: int, rng: np.random.Generator) -> list[int]:
        """Return indices of up to `k` diverse pool members, in pick order."""
        n = len(pool)
        if n == 0 or k <= 0:
            return []
        k = min(k, n)
        descriptors = _stack_descriptors(self._descriptor_fn, pool)
        if self._reference is not None:
            if self._reference.shape[1] != descriptors.shape[1]:
                raise ValueError(
                    f"reference descriptor dim {self._reference.shape[1]} != "
                    f"pool descriptor dim {descriptors.shape[1]}"
                )
            diffs = descriptors[:, None, :] - self._reference[None, :, :]
            min_dist = np.sqrt((diffs * diffs).sum(axis=-1)).min(axis=1)
            first = int(np.argmax(min_dist))
        else:
            first = int(rng.integers(n))
            min_dist = np.full(n, np.inf)
        selected = [first]
        min_dist = np.minimum(min_dist, np.linalg.norm(descriptors - descriptors[first], axis=1))
        min_dist[first] = -np.inf
        while len(selected) < k:
            nxt = int(np.argmax(min_dist))
            selected.append(nxt)
            min_dist = np.minimum(min_dist, np.linalg.norm(descriptors - descriptors[nxt], axis=1))
            min_dist[nxt] = -np.inf
        return selected


class CompositeSelector:
    """Disagreement preselection, then diversity reduction — in that order.

    The two-stage structure is the design decision. Disagreement alone returns
    near-duplicates: the most uncertain region of the pool fills the whole
    batch with slight variants of one geometry, and labels two through ten of
    it buy almost nothing. Diversity alone wastes budget on regions the
    committee already predicts confidently. So `QueryByCommittee` first keeps a
    wide band of the most uncertain candidates (``band_factor * batch_size``),
    and `FarthestPointSampling` then reduces that band to `batch_size` spread-
    out picks. Every label is therefore both informative (in the band) and
    non-redundant (chosen by max-min distance).

    Implements the `Selector` protocol: `rank` returns a full permutation of
    the pool with the diversified batch first (in pick order), then the
    remaining candidates in disagreement order — so a driver that takes the
    first `batch_size` indices gets exactly the composite selection.
    Deterministic: the stored `seed` re-seeds the diversity stage on every
    `rank` call.
    """

    def __init__(
        self,
        batch_size: int,
        band_factor: float = 4.0,
        descriptor_fn: Callable[[Structure], np.ndarray] = default_descriptor,
        seed: int = 0,
    ) -> None:
        """`band_factor` trades off the two criteria: 1.0 collapses to pure
        disagreement ranking; large values approach pure diversity. The default
        of 4.0 is a generic middle ground, not a tuned constant.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if band_factor < 1.0:
            raise ValueError("band_factor must be >= 1.0 (the band must hold the batch)")
        self._batch_size = int(batch_size)
        self._band_factor = float(band_factor)
        self._descriptor_fn = descriptor_fn
        self._seed = int(seed)
        self._qbc = QueryByCommittee()

    def rank(self, pool: Sequence[Structure], committee: object) -> list[int]:
        n = len(pool)
        if n == 0:
            return []
        qbc_order = self._qbc.rank(pool, committee)
        band_size = min(n, math.ceil(self._band_factor * self._batch_size))
        band = qbc_order[:band_size]
        fps = FarthestPointSampling(self._descriptor_fn)
        rng = np.random.default_rng(self._seed)
        picked_local = fps.select([pool[i] for i in band], self._batch_size, rng)
        picked = [band[j] for j in picked_local]
        picked_set = set(picked)
        remainder = [i for i in qbc_order if i not in picked_set]
        return picked + remainder
