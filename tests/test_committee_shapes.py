"""Committee output shapes and the algebra of the two-head reconstruction.

`Predictions` is the load-bearing interface between models and selection, so
its shapes are contractual: one energy and one scalar spread per structure,
one per-atom array whose length matches that structure's atom count. The
two-head identities and the residual corrector's composition are checked
against hand-computed expectations.
"""

from __future__ import annotations

import numpy as np
import pytest

from campaign_kit.backends import SklearnMLPModel
from campaign_kit.committee import Committee, ResidualCorrector, TwoHeadCommittee
from campaign_kit.protocols import Dataset
from tests._doubles import (
    TinyLinearModel,
    analytic_energy,
    jittered_structures,
    labeled_dataset,
)


def _dataset(n: int = 12, seed: int = 1) -> Dataset:
    return labeled_dataset(jittered_structures(n, scale=0.2, seed=seed))


def test_committee_prediction_shapes() -> None:
    committee = Committee(lambda: TinyLinearModel(jitter=1e-2), n_members=3, base_seed=5)
    committee.fit(_dataset())
    pool = jittered_structures(4, scale=0.2, seed=77)
    preds = committee.predict_mean_and_spread(pool)
    assert len(preds) == 4
    assert preds.energies.shape == (4,)
    assert preds.per_structure_spread.shape == (4,)
    assert len(preds.per_atom_spread) == 4
    for structure, spread in zip(pool, preds.per_atom_spread, strict=True):
        assert spread.shape == (structure.n_atoms,)
        assert (spread >= 0.0).all()
    # Members differ only by seed, and the jitter guarantees they disagree.
    assert (preds.per_structure_spread > 0.0).all()


def test_committee_predict_is_member_mean() -> None:
    committee = Committee(lambda: TinyLinearModel(jitter=1e-2), n_members=3, base_seed=5)
    committee.fit(_dataset())
    pool = jittered_structures(5, scale=0.2, seed=8)
    manual = np.mean([m.predict(pool) for m in committee.members], axis=0)
    assert np.allclose(committee.predict(pool), manual)


def test_committee_requires_fit_before_predict() -> None:
    committee = Committee(TinyLinearModel, n_members=2)
    assert not committee.is_fitted
    with pytest.raises(RuntimeError):
        committee.predict(jittered_structures(2, scale=0.2, seed=1))


def test_two_head_reconstruct_identities() -> None:
    structures = jittered_structures(10, scale=0.2, seed=1)
    dataset_a = labeled_dataset(structures)
    dataset_b = Dataset(
        structures=list(structures),
        energies=[0.1 * analytic_energy(s) for s in structures],
        forces=[None] * len(structures),
    )
    head_a = Committee(lambda: TinyLinearModel(jitter=1e-3), n_members=3, base_seed=0)
    head_b = Committee(lambda: TinyLinearModel(jitter=1e-3), n_members=3, base_seed=100)
    two = TwoHeadCommittee(head_a, head_b)
    two.fit(dataset_a, dataset_b)

    pool = jittered_structures(5, scale=0.2, seed=9)
    plus, minus = two.reconstruct(pool)
    pred_a = head_a.predict_mean_and_spread(pool)
    pred_b = head_b.predict_mean_and_spread(pool)
    assert np.allclose(plus.energies, pred_a.energies + 0.5 * pred_b.energies)
    assert np.allclose(minus.energies, pred_a.energies - 0.5 * pred_b.energies)
    expected = np.sqrt(pred_a.per_structure_spread**2 + 0.25 * pred_b.per_structure_spread**2)
    assert np.allclose(plus.per_structure_spread, expected)
    assert np.allclose(minus.per_structure_spread, expected)
    for pa, pb, pp in zip(
        pred_a.per_atom_spread, pred_b.per_atom_spread, plus.per_atom_spread, strict=True
    ):
        assert np.allclose(pp, np.sqrt(pa**2 + 0.25 * pb**2))


def test_two_head_fit_requires_matching_structures() -> None:
    head_a = Committee(TinyLinearModel, n_members=2)
    head_b = Committee(TinyLinearModel, n_members=2)
    dataset_a = labeled_dataset(jittered_structures(6, scale=0.2, seed=1))
    dataset_b = labeled_dataset(jittered_structures(6, scale=0.2, seed=2))
    with pytest.raises(ValueError):
        TwoHeadCommittee(head_a, head_b).fit(dataset_a, dataset_b)


def test_residual_corrector_adds_learned_residual() -> None:
    dataset = _dataset()
    committee = Committee(lambda: TinyLinearModel(jitter=0.05), n_members=3, base_seed=0)
    committee.fit(dataset)
    residual_model = TinyLinearModel(jitter=0.0)
    corrector = ResidualCorrector(committee, residual_model, seed=4)
    corrector.fit(dataset)
    pool = jittered_structures(4, scale=0.2, seed=13)
    expected = committee.predict(pool) + residual_model.predict(pool)
    assert np.allclose(corrector.predict(pool), expected)


def test_residual_corrector_rejects_unfitted_committee() -> None:
    with pytest.raises(ValueError):
        ResidualCorrector(Committee(TinyLinearModel, n_members=2), TinyLinearModel())


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_sklearn_backend_per_atom_sums_to_total() -> None:
    # The per-atom decomposition is a proxy, but its one hard guarantee is
    # that the contributions sum exactly to the predicted total.
    model = SklearnMLPModel(hidden_layer_sizes=(8,), max_iter=300)
    model.fit(_dataset(10, seed=1), seed=0)
    pool = jittered_structures(3, scale=0.2, seed=2)
    totals = model.predict(pool)
    for energy, per_atom in zip(totals, model.predict_per_atom(pool), strict=True):
        assert per_atom.shape == (3,)
        assert np.isclose(per_atom.sum(), energy, rtol=1e-8, atol=1e-10)
