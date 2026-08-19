"""A small sklearn MLP backend for demos and tests.

This backend exists so the whole campaign loop — committee, selector, driver —
can run on a laptop in seconds. It is honest about its two shortcuts:

1. The default descriptor (flattened upper-triangle inverse distances) is
   permutation-sensitive and requires a fixed atom count. That is fine for
   demos where atom ordering is fixed by construction; production descriptors
   are pluggable via ``descriptor_fn``.
2. The per-atom energy decomposition is a proxy, not a physical partition.
   See ``SklearnMLPModel.predict_per_atom`` for exactly what it does and where
   it stops being trustworthy.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from campaign_kit.protocols import Dataset, Structure

__all__ = ["SklearnMLPModel", "inverse_distance_descriptor"]


def inverse_distance_descriptor(structure: Structure) -> np.ndarray:
    """Flattened upper-triangle 1/r_ij, shape (n_atoms * (n_atoms - 1) / 2,).

    Inverse distances rather than distances: the descriptor then grows where
    atoms approach each other — the region where energies vary fastest — and
    decays smoothly toward zero at dissociation instead of diverging.

    Permutation-SENSITIVE: swapping two atom labels changes the vector even
    though the geometry is identical. Acceptable for demos where atom ordering
    is fixed across all structures; use a proper invariant descriptor in
    production.
    """
    n = structure.n_atoms
    if n < 2:
        raise ValueError("inverse_distance_descriptor needs at least 2 atoms")
    diff = structure.positions[:, None, :] - structure.positions[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    iu = np.triu_indices(n, k=1)
    return 1.0 / dist[iu]


class SklearnMLPModel:
    """`Model` implementation: descriptor vector -> MLPRegressor -> energy.

    Inputs are standardized inside a pipeline because MLP training is badly
    conditioned on raw inverse distances (their scales differ by orders of
    magnitude across atom pairs). ``seed`` is passed to ``random_state`` at
    fit time so a committee of these models differing only in seed measures
    model variance, as the `Model` protocol requires.
    """

    def __init__(
        self,
        descriptor_fn: Callable[[Structure], np.ndarray] = inverse_distance_descriptor,
        hidden_layer_sizes: tuple[int, ...] = (64, 64),
        max_iter: int = 2000,
        alpha: float = 1e-4,
        per_atom_mode: Literal["sensitivity", "uniform"] = "sensitivity",
        fd_step: float = 1e-3,
    ) -> None:
        self._descriptor_fn = descriptor_fn
        self._hidden_layer_sizes = hidden_layer_sizes
        self._max_iter = max_iter
        self._alpha = alpha
        self._per_atom_mode = per_atom_mode
        self._fd_step = fd_step
        self._pipe: Pipeline | None = None
        self._n_features: int | None = None

    def fit(self, dataset: Dataset, seed: int) -> None:
        x = np.stack([self._descriptor_fn(s) for s in dataset.structures])
        y = dataset.energy_array()
        self._pipe = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=self._hidden_layer_sizes,
                max_iter=self._max_iter,
                alpha=self._alpha,
                random_state=seed,
            ),
        )
        self._pipe.fit(x, y)
        self._n_features = x.shape[1]

    def predict(self, structures: Sequence[Structure]) -> np.ndarray:
        pipe = self._require_fitted()
        x = np.stack([self._descriptor_fn(s) for s in structures])
        return np.asarray(pipe.predict(x), dtype=float)

    def predict_per_atom(self, structures: Sequence[Structure]) -> list[np.ndarray]:
        """Distribute each predicted energy over atoms by a sensitivity proxy.

        Per-atom contribution = ``energy / n_atoms * (1 + z_i)`` where ``z_i``
        is the atom's finite-difference sensitivity — |change in the predicted
        energy when atom i's coordinates are displaced by ``fd_step``| — after
        normalizing the sensitivities to mean 0, so contributions always sum
        exactly to the total energy. Cost: one extra descriptor evaluation per
        atom plus a single batched pipeline call per structure.

        LIMITATION — read before trusting the localization. This is a proxy
        for where the model's output depends on the geometry, not a physical
        energy partition. With ``per_atom_mode="uniform"`` (energy/n_atoms),
        per-atom spread degenerates to member disagreement on the same uniform
        split: it still localizes when members disagree about the total, but
        only via atom-count weighting, never pointing at a specific atom. The
        sensitivity mode is a cheap, honest step up; for production use a
        backend with real per-atom energies (e.g. a graph-network model whose
        readout is a sum of atomic contributions).
        """
        pipe = self._require_fitted()
        base = self.predict(structures)
        out: list[np.ndarray] = []
        for s, e0 in zip(structures, base, strict=True):
            n = s.n_atoms
            if self._per_atom_mode == "uniform":
                out.append(np.full(n, e0 / n))
                continue
            perturbed = np.empty((n, int(self._n_features or 0)))
            for i in range(n):
                pos = s.positions.copy()
                pos[i] += self._fd_step
                perturbed[i] = self._descriptor_fn(Structure(s.species, pos))
            sens = np.abs(np.asarray(pipe.predict(perturbed), dtype=float) - e0) / self._fd_step
            mean_sens = float(sens.mean())
            z = sens / mean_sens - 1.0 if mean_sens > 0.0 else np.zeros(n)
            out.append(e0 / n * (1.0 + z))
        return out

    def _require_fitted(self) -> Pipeline:
        if self._pipe is None:
            raise RuntimeError("SklearnMLPModel.fit must be called before predicting")
        return self._pipe
