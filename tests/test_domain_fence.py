"""Domain membership: the fence and the silent-extrapolation diagnostic.

The fence must do two jobs: admit queries that look like the training data
and refuse to spend labels far outside it. The silent-extrapolation check
covers the failure mode the fence alone cannot express to a user: a
committee that agrees confidently exactly where no data constrains it.
"""

from __future__ import annotations

import numpy as np
import pytest

from campaign_kit.domain import DomainFence, TrainingDomain, silent_extrapolation_check
from campaign_kit.protocols import Predictions, Structure
from tests._doubles import BASE_POSITIONS, BASE_SPECIES, jittered_structures


def _train_domain() -> TrainingDomain:
    return TrainingDomain(jittered_structures(20, scale=0.05, seed=3))


def _far_structure() -> Structure:
    # Doubling every coordinate halves every inverse distance: far from the
    # training cluster in descriptor space, by construction.
    return Structure(BASE_SPECIES, BASE_POSITIONS * 2.0)


def test_training_points_have_zero_distance() -> None:
    train = jittered_structures(15, scale=0.05, seed=3)
    domain = TrainingDomain(train)
    assert np.allclose(domain.distance(train[:5]), 0.0, atol=1e-12)


def test_fence_admits_near_and_rejects_far() -> None:
    domain = _train_domain()
    fence = DomainFence.from_train_quantile(domain, quantile=1.0)
    near = jittered_structures(5, scale=0.03, seed=21)
    pool = [*near, _far_structure()]
    mask = fence.check(pool)
    assert mask[:5].all()
    assert not mask[5]
    assert fence.reject_indices(pool) == [5]


def test_fence_explicit_threshold_and_validation() -> None:
    domain = _train_domain()
    far = _far_structure()
    generous = DomainFence(domain, threshold=1e6)
    assert generous.check([far])[0]  # a huge threshold admits everything
    assert generous.threshold == 1e6
    with pytest.raises(ValueError):
        DomainFence(domain, threshold=0.0)
    with pytest.raises(ValueError):
        DomainFence.from_train_quantile(domain, quantile=1.5)


def test_mahalanobis_is_finite_and_orders_far_above_near() -> None:
    domain = _train_domain()
    near = jittered_structures(1, scale=0.03, seed=22)[0]
    values = domain.mahalanobis([near, _far_structure()])
    assert np.isfinite(values).all()
    assert values[1] > values[0]


def test_silent_extrapolation_flags_the_far_and_quiet_structure() -> None:
    domain = _train_domain()
    pool = [*jittered_structures(3, scale=0.01, seed=33), _far_structure()]
    # The far structure is given the LOWEST committee spread: the correlated-
    # error case where the ensemble is confidently wrong together.
    spreads = (0.5, 0.6, 0.7, 0.01)
    preds = Predictions(
        energies=np.zeros(4),
        per_structure_spread=np.asarray(spreads),
        per_atom_spread=[np.full(3, v) for v in spreads],
    )
    warnings = silent_extrapolation_check(pool, preds, domain)
    assert len(warnings) == 1
    assert pool[3].structure_id in warnings[0]


def test_silent_extrapolation_needs_the_quiet_half_too() -> None:
    # Control that can fail: same far structure, but now LOUD (highest
    # spread). If the check fired on distance alone, this would still flag.
    domain = _train_domain()
    pool = [*jittered_structures(3, scale=0.01, seed=33), _far_structure()]
    spreads = (0.5, 0.6, 0.7, 5.0)
    preds = Predictions(
        energies=np.zeros(4),
        per_structure_spread=np.asarray(spreads),
        per_atom_spread=[np.full(3, v) for v in spreads],
    )
    assert silent_extrapolation_check(pool, preds, domain) == []
