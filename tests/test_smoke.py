"""Fast, deterministic smoke tests for the invariants the rest of the kit leans on.

No ML training happens here: a trivial in-file `Model` keeps the suite fast and
seed-exact, so a CI failure points at the kit's plumbing (hashing, dedup,
committee wiring, fencing, ranking) rather than at optimizer noise.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from campaign_kit import Dataset, LabelResult, LabelStatus, Predictions, Structure
from campaign_kit.committee import Committee
from campaign_kit.domain import DomainFence, TrainingDomain
from campaign_kit.selection import QueryByCommittee


class MeanModel:
    """Minimal `Model`: predicts the training mean, shifted slightly per seed.

    The seed shift makes committee spread nonzero and exactly predictable,
    which lets the committee test assert numbers instead of shapes only.
    """

    def __init__(self) -> None:
        self._value = 0.0

    def fit(self, dataset: Dataset, seed: int) -> None:
        self._value = float(dataset.energy_array().mean()) + 0.1 * seed

    def predict(self, structures: Sequence[Structure]) -> np.ndarray:
        return np.full(len(structures), self._value)

    def predict_per_atom(self, structures: Sequence[Structure]) -> list[np.ndarray]:
        return [np.full(s.n_atoms, self._value / s.n_atoms) for s in structures]


def chain(offset: float) -> Structure:
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0 + offset, 0.0, 0.0]])
    return Structure(species=("X", "X", "X"), positions=positions)


def labeled_dataset(offsets: Sequence[float]) -> Dataset:
    dataset = Dataset()
    dataset.merge([LabelResult.success(chain(o), energy=float(o)) for o in offsets])
    return dataset


def test_structure_id_is_content_based() -> None:
    a, b = chain(0.1), chain(0.1)
    assert a is not b
    assert a.structure_id == b.structure_id
    assert a.structure_id != chain(0.2).structure_id


def test_dataset_merge_is_idempotent_and_skips_failures() -> None:
    dataset = Dataset()
    results = [
        LabelResult.success(chain(0.0), energy=1.0),
        LabelResult.failure(chain(0.1), LabelStatus.FAILED_TIMEOUT, "walltime"),
    ]
    assert dataset.merge(results) == 1
    assert dataset.merge(results) == 0  # re-collection after a crash adds nothing
    assert len(dataset) == 1


def test_label_result_failure_rejects_success_status() -> None:
    with pytest.raises(ValueError):
        LabelResult.failure(chain(0.0), LabelStatus.SUCCEEDED)


def test_committee_mean_and_spread_are_seed_exact() -> None:
    committee = Committee(MeanModel, n_members=3, base_seed=0)
    committee.fit(labeled_dataset([0.0, 0.2, 0.4]))
    preds = committee.predict_mean_and_spread([chain(0.0), chain(0.5)])
    # members predict mean + 0.1 * {0, 1, 2} -> committee mean = mean + 0.1
    assert preds.energies == pytest.approx([0.3, 0.3])
    expected_spread = float(np.std([0.0, 0.1, 0.2]))
    assert preds.per_structure_spread == pytest.approx([expected_spread] * 2)
    assert [len(a) for a in preds.per_atom_spread] == [3, 3]


def test_domain_fence_admits_train_and_rejects_far_points() -> None:
    train = [chain(o) for o in np.linspace(0.0, 0.5, 12)]
    domain = TrainingDomain(train)
    fence = DomainFence.from_train_quantile(domain, quantile=1.0)
    assert fence.check(train).all()
    far = chain(5.0)
    assert fence.reject_indices([train[0], far]) == [1]


def test_qbc_ranks_by_local_disagreement() -> None:
    pool = [chain(o) for o in (0.0, 0.1, 0.2)]
    preds = Predictions(
        energies=np.zeros(3),
        per_structure_spread=np.array([0.5, 0.1, 0.3]),
        per_atom_spread=[
            np.array([0.1, 0.1, 0.5]),  # local max 0.5
            np.array([0.9, 0.1, 0.1]),  # local max 0.9 -> most disputed
            np.array([0.2, 0.2, 0.2]),  # local max 0.2
        ],
    )
    assert QueryByCommittee().rank(pool, preds) == [1, 0, 2]
