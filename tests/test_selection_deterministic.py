"""Determinism and ranking semantics of the selection policies.

Selection decides where the labeling budget goes, so it must be exactly
reproducible from its seed: two drivers running the same campaign must pick
the same structures. Rankings are checked against hand-computed expectations
where the semantics (local disagreement, stable ties, fallback) are the
contract.
"""

from __future__ import annotations

import numpy as np

from campaign_kit.domain import default_descriptor
from campaign_kit.protocols import Predictions, Structure
from campaign_kit.selection import CompositeSelector, FarthestPointSampling, QueryByCommittee
from tests._doubles import jittered_structures


def _pool(n: int = 24, seed: int = 5) -> list[Structure]:
    return jittered_structures(n, scale=0.3, seed=seed)


def _random_predictions(pool: list[Structure], seed: int) -> Predictions:
    rng = np.random.default_rng(seed)
    return Predictions(
        energies=rng.standard_normal(len(pool)),
        per_structure_spread=rng.uniform(0.0, 1.0, len(pool)),
        per_atom_spread=[rng.uniform(0.0, 1.0, s.n_atoms) for s in pool],
    )


def test_composite_selector_same_seed_gives_identical_ranking() -> None:
    pool = _pool()
    preds = _random_predictions(pool, seed=1)
    first = CompositeSelector(batch_size=5, seed=11).rank(pool, preds)
    second = CompositeSelector(batch_size=5, seed=11).rank(pool, preds)
    assert first == second
    assert sorted(first) == list(range(len(pool)))


def test_composite_selector_is_repeatable_on_one_instance() -> None:
    pool = _pool()
    preds = _random_predictions(pool, seed=1)
    selector = CompositeSelector(batch_size=5, seed=11)
    assert selector.rank(pool, preds) == selector.rank(pool, preds)


def test_composite_selector_other_seed_still_a_full_permutation() -> None:
    # A different seed is allowed to reorder the batch; the contract that must
    # hold regardless is a full permutation of the pool.
    pool = _pool()
    preds = _random_predictions(pool, seed=1)
    ranking = CompositeSelector(batch_size=5, seed=12).rank(pool, preds)
    assert sorted(ranking) == list(range(len(pool)))


def test_qbc_ranks_by_local_disagreement_with_stable_ties() -> None:
    pool = jittered_structures(4, scale=0.2, seed=9)
    preds = Predictions(
        energies=np.zeros(4),
        per_structure_spread=np.zeros(4),  # must be ignored when per-atom exists
        per_atom_spread=[
            np.array([0.1, 0.05, 0.02]),
            np.array([0.9, 0.1, 0.0]),
            np.array([0.5, 0.2, 0.1]),
            np.array([0.9, 0.3, 0.2]),  # ties with index 1 on the local max
        ],
    )
    # Scores are the per-atom maxima [0.1, 0.9, 0.5, 0.9]; the tie between
    # indices 1 and 3 must keep pool order.
    assert QueryByCommittee().rank(pool, preds) == [1, 3, 2, 0]


def test_qbc_falls_back_to_per_structure_spread() -> None:
    pool = jittered_structures(4, scale=0.2, seed=9)
    preds = Predictions(
        energies=np.zeros(4),
        per_structure_spread=np.array([0.2, 0.9, 0.5, 0.1]),
        per_atom_spread=[np.zeros(0)] * 4,  # empty local signal -> global scalar
    )
    assert QueryByCommittee().rank(pool, preds) == [1, 2, 0, 3]


def test_fps_deterministic_given_equal_seeds() -> None:
    pool = _pool(30, seed=2)
    fps = FarthestPointSampling(default_descriptor)
    first = fps.select(pool, 6, np.random.default_rng(3))
    second = fps.select(pool, 6, np.random.default_rng(3))
    assert first == second
    assert len(set(first)) == 6


def test_fps_with_reference_does_not_consume_rng() -> None:
    pool = _pool(30, seed=2)
    reference = _pool(5, seed=8)
    fps = FarthestPointSampling(default_descriptor, reference=reference)
    # Differently seeded generators must not matter: the first pick comes from
    # the reference set, not from the rng.
    first = fps.select(pool, 4, np.random.default_rng(0))
    second = fps.select(pool, 4, np.random.default_rng(1234))
    assert first == second
