"""Committee ensembles: disagreement as an uncertainty signal.

A committee is N copies of the same model class trained on the same data but
with different seeds. Because the seed is the only difference, the spread of
their predictions measures *model* variance — how much the fit is under-
determined by the data — which is exactly the quantity an active-learning
selector wants to maximize when picking what to label next.

The per-atom spread exists because the global one is not enough: a structure
can be confidently predicted overall while one region of it is extrapolating,
and averaging over atoms washes that signal away. Selection should rank on the
local maximum, not the global mean (see `Predictions`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from campaign_kit.protocols import Dataset, Model, Predictions, Structure

__all__ = ["Committee", "TwoHeadCommittee", "ResidualCorrector"]


class Committee:
    """N independently-seeded models trained on one dataset.

    The factory is called eagerly so a misconfigured backend fails at
    construction, not mid-campaign. Member i is fitted with ``base_seed + i``:
    deterministic, gap-free seeds make any single member reproducible in
    isolation when debugging a disagreement.
    """

    def __init__(
        self,
        model_factory: Callable[[], Model],
        n_members: int = 4,
        base_seed: int = 0,
    ) -> None:
        if n_members < 1:
            raise ValueError(f"n_members must be >= 1, got {n_members}")
        self._members: list[Model] = [model_factory() for _ in range(n_members)]
        self._base_seed = base_seed
        self._fitted = False

    @property
    def n_members(self) -> int:
        return len(self._members)

    @property
    def members(self) -> tuple[Model, ...]:
        """Read-only view; mutating a member invalidates the spread's meaning."""
        return tuple(self._members)

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, dataset: Dataset) -> None:
        """Train every member on the same data, varying only the seed.

        Same data is deliberate: bootstrap resampling would mix data variance
        into the disagreement signal, and for the small datasets typical of
        expensive-label campaigns that noise dominates.
        """
        if len(dataset) == 0:
            raise ValueError("cannot fit a committee on an empty dataset")
        for i, member in enumerate(self._members):
            member.fit(dataset, seed=self._base_seed + i)
        self._fitted = True

    def predict(self, structures: Sequence[Structure]) -> np.ndarray:
        """Committee-mean energies, shape (n,)."""
        self._require_fitted()
        stacked = np.stack([m.predict(structures) for m in self._members])
        return stacked.mean(axis=0)

    def predict_mean_and_spread(self, structures: Sequence[Structure]) -> Predictions:
        """Mean energies plus two disagreement signals: global and per-atom.

        Why per-atom: a structure can be confidently predicted overall while
        one region extrapolates — a global scalar averages the signal away.
        Each member decomposes its total energy over atoms via
        ``predict_per_atom``, and the spread is the across-member standard
        deviation of each atom's contribution, so disagreement is attributed
        to the atoms the members actually disagree about.

        Spreads are population standard deviations (ddof=0) over members: the
        committee is the whole population of interest, not a sample from one.
        """
        self._require_fitted()
        member_energies = np.stack([m.predict(structures) for m in self._members])
        member_pa = [m.predict_per_atom(structures) for m in self._members]
        per_atom_spread = [
            np.stack([member_pa[i][j] for i in range(self.n_members)]).std(axis=0)
            for j in range(len(structures))
        ]
        return Predictions(
            energies=member_energies.mean(axis=0),
            per_structure_spread=member_energies.std(axis=0),
            per_atom_spread=per_atom_spread,
        )

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Committee.fit must be called before predicting")


class TwoHeadCommittee:
    """Two committees over complementary targets: a mean-like and a difference-like one.

    When two related target surfaces must be learned, fitting their average
    (head_a) and their difference (head_b) separately is often better
    conditioned than fitting each surface directly: the difference is usually
    smaller and smoother than either surface, so it gets its own model
    capacity instead of being the tiny residual of two large numbers.

    ``reconstruct`` recovers the two surfaces as ``a + b/2`` and ``a - b/2``.
    Combined spreads use the independent-error rule
    ``sqrt(spread_a**2 + 0.25 * spread_b**2)`` — an approximation that ignores
    cross-head correlation, defensible because the heads are separate
    committees trained on separate targets, and stated here so nobody
    mistakes the combined spread for a calibrated error bar.
    """

    def __init__(self, head_a: Committee, head_b: Committee) -> None:
        self.head_a = head_a
        self.head_b = head_b

    def fit(self, dataset_a: Dataset, dataset_b: Dataset) -> None:
        """Fit each head on its own target; datasets must cover the same structures."""
        if dataset_a.ids != dataset_b.ids:
            raise ValueError("head_a and head_b datasets must contain the same structures")
        self.head_a.fit(dataset_a)
        self.head_b.fit(dataset_b)

    def reconstruct(self, structures: Sequence[Structure]) -> tuple[Predictions, Predictions]:
        """Combined predictions for the two surfaces: (a + b/2, a - b/2).

        Both returned `Predictions` carry the same combined spread — the
        independent-error combination is symmetric in the sign of b/2.
        """
        pred_a = self.head_a.predict_mean_and_spread(structures)
        pred_b = self.head_b.predict_mean_and_spread(structures)
        spread = np.sqrt(pred_a.per_structure_spread**2 + 0.25 * pred_b.per_structure_spread**2)
        per_atom = [
            np.sqrt(sa**2 + 0.25 * sb**2)
            for sa, sb in zip(pred_a.per_atom_spread, pred_b.per_atom_spread, strict=True)
        ]
        plus = Predictions(
            energies=pred_a.energies + 0.5 * pred_b.energies,
            per_structure_spread=spread,
            per_atom_spread=per_atom,
        )
        minus = Predictions(
            energies=pred_a.energies - 0.5 * pred_b.energies,
            per_structure_spread=spread.copy(),
            per_atom_spread=[s.copy() for s in per_atom],
        )
        return plus, minus


class ResidualCorrector:
    """A frozen committee plus a second model fitted on its residuals.

    Why two levels: the committee is kept frozen so its disagreement signal —
    the thing selection depends on — stays intact, while a separate model
    absorbs whatever systematic bias the committee has against the labels.
    Re-fitting the committee itself on the corrected target would entangle
    the uncertainty estimate with the correction.
    """

    def __init__(self, committee: Committee, residual_model: Model, seed: int = 0) -> None:
        if not committee.is_fitted:
            raise ValueError("ResidualCorrector requires an already-fitted committee")
        self._committee = committee
        self._residual_model = residual_model
        self._seed = seed
        self._fitted = False

    @property
    def committee(self) -> Committee:
        return self._committee

    def fit(self, dataset: Dataset) -> None:
        """Fit the residual model on (label - committee_mean); the committee is untouched."""
        if len(dataset) == 0:
            raise ValueError("cannot fit a residual corrector on an empty dataset")
        base = self._committee.predict(dataset.structures)
        residuals = dataset.energy_array() - base
        residual_dataset = Dataset(
            structures=list(dataset.structures),
            energies=[float(r) for r in residuals],
            forces=[None] * len(dataset),
        )
        self._residual_model.fit(residual_dataset, seed=self._seed)
        self._fitted = True

    def predict(self, structures: Sequence[Structure]) -> np.ndarray:
        """Committee mean plus the learned residual correction, shape (n,)."""
        if not self._fitted:
            raise RuntimeError("ResidualCorrector.fit must be called before predicting")
        return self._committee.predict(structures) + self._residual_model.predict(structures)
