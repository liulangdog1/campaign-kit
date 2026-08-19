"""Reduce a set of reaction-path images to a few collective coordinates.

A converged reaction path — for example, a small molecule approaching a
transition-metal complex — moves almost entirely inside a low-dimensional
subspace of the full 3M Cartesian coordinates. This module extracts that
subspace by principal component analysis (PCA) of rigid-body-aligned
displacement vectors, so downstream tools can describe progress along the
path with a handful of amplitudes instead of 3M numbers.

Why alignment comes first: the raw Cartesian difference between two images
mixes internal deformation with overall rotation and translation of the whole
system. The Kabsch superposition removes the rigid-body part, so the PCA sees
only shape change.

Why mass weighting is an option rather than a hard-coded detail: multiplying
each displacement coordinate by sqrt(mass) changes the metric the PCA
optimizes in, and therefore changes what the components *mean*.

- Mass-weighted components resemble vibrational normal modes: heavy atoms move
  less per unit amplitude, and the leading components track the directions
  that dominate the kinetic energy of the motion. Prefer this when the basis
  feeds dynamics- or mode-analysis-style consumers.
- Unweighted components are purely geometric: every atom counts equally, and
  the leading components track the largest Cartesian rearrangements, which are
  often light-atom motions. Prefer this when the basis is used as a plain
  geometric compression of the path.

The two settings produce genuinely different subspaces, so the choice is
stored on the fitted object and honored consistently by ``project`` and
``reconstruct``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from campaign_kit.protocols import Structure

__all__ = ["ATOMIC_MASSES", "PathPCA", "kabsch_align"]

# Standard atomic weights (amu) for elements that commonly appear in
# simulation campaigns. The table is deliberately small: an unknown symbol
# raises immediately, forcing the caller to pass explicit masses instead of
# silently falling back to a wrong default.
ATOMIC_MASSES: dict[str, float] = {
    "H": 1.008,
    "He": 4.0026,
    "Li": 6.94,
    "Be": 9.0122,
    "B": 10.81,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "Ne": 20.180,
    "Na": 22.990,
    "Mg": 24.305,
    "Al": 26.982,
    "Si": 28.085,
    "P": 30.974,
    "S": 32.06,
    "Cl": 35.45,
    "Ar": 39.948,
    "K": 39.098,
    "Ca": 40.078,
    "Sc": 44.956,
    "Ti": 47.867,
    "V": 50.942,
    "Cr": 51.996,
    "Mn": 54.938,
    "Fe": 55.845,
    "Co": 58.933,
    "Ni": 58.693,
    "Cu": 63.546,
    "Zn": 65.38,
    "Br": 79.904,
    "Ru": 101.07,
    "Rh": 102.91,
    "Pd": 106.42,
    "Ag": 107.87,
    "I": 126.90,
    "Pt": 195.08,
    "Au": 196.97,
}


def _masses_from_species(species: Sequence[str]) -> np.ndarray:
    unknown = sorted({s for s in species if s not in ATOMIC_MASSES})
    if unknown:
        raise ValueError(
            f"no internal mass for element symbol(s) {unknown}; pass explicit masses instead"
        )
    return np.asarray([ATOMIC_MASSES[s] for s in species], dtype=float)


def kabsch_align(
    P: np.ndarray,
    Q: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Rigidly superimpose ``P`` onto ``Q``; return the transformed copy of ``P``.

    The weighted centroids are matched (translation removed) and the
    least-squares optimal rotation is applied, so the returned coordinates
    differ from ``Q`` only by internal deformation.

    Determinant sign fix: the unconstrained SVD solution over all orthogonal
    matrices can be an *improper* rotation — a reflection — whenever
    ``det(V @ U.T) < 0``, which happens routinely for near-planar geometries.
    A reflection would invert the handedness of the structure, which is never a
    physical superposition. The fix flips the sign of the third singular
    direction (``D = diag(1, 1, det)``), restricting the optimum to proper
    rotations at the cost of a slightly larger residual than the unconstrained
    minimum.

    ``weights`` is a per-atom weight vector of shape (M,); ``None`` means
    uniform. Mass weights make heavy atoms dominate the superposition, which
    is the right choice when the basis itself is mass-weighted.
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3 or P.shape != Q.shape:
        raise ValueError(f"P and Q must both have shape (M, 3); got {P.shape} and {Q.shape}")
    n_atoms = P.shape[0]
    if weights is None:
        w = np.full(n_atoms, 1.0 / n_atoms)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (n_atoms,):
            raise ValueError(f"weights must have shape ({n_atoms},), got {w.shape}")
        total = float(w.sum())
        if total <= 0.0 or np.any(w < 0.0):
            raise ValueError("weights must be non-negative with a positive sum")
        w = w / total

    centroid_p = w @ P
    centroid_q = w @ Q
    p_centered = P - centroid_p
    q_centered = Q - centroid_q

    covariance = (p_centered * w[:, None]).T @ q_centered
    u_mat, _, vt_mat = np.linalg.svd(covariance)
    sign = float(np.sign(np.linalg.det(vt_mat.T @ u_mat.T)))
    correction = np.diag([1.0, 1.0, sign])
    rotation = vt_mat.T @ correction @ u_mat.T
    return p_centered @ rotation.T + centroid_q


class PathPCA:
    """PCA basis for a set of path images, expanded about one reference image.

    Deliberate modeling choice: displacements are taken relative to the chosen
    *reference image*, not the ensemble mean, and the PCA runs on that
    uncentered matrix. Amplitude zero therefore reconstructs the reference
    geometry exactly — the property downstream consumers of a path basis rely
    on, because the reference is usually a meaningful anchor point such as an
    endpoint of the path.

    Fitted attributes (``None`` before ``fit``/``from_state``):

    - ``components_``: (k, 3M) orthonormal directions, row-major over atoms
      then x/y/z. In the mass-weighted setting these live in sqrt(mass)-scaled
      coordinates; ``reconstruct`` undoes the scaling.
    - ``explained_variance_ratio_``: (k,) fraction of total displacement
      variance captured by each component.
    - ``cumulative_variance_ratio_``: (k,) running sum of the above — the
      curve to inspect when choosing a truncation.
    - ``reference_``: (M, 3) the expansion origin geometry.
    - ``mass_weighted_`` / ``masses_``: the metric the basis was built in.
    """

    def __init__(self) -> None:
        self.components_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None
        self.cumulative_variance_ratio_: np.ndarray | None = None
        self.reference_: np.ndarray | None = None
        self.mass_weighted_: bool = False
        self.masses_: np.ndarray | None = None
        self.n_atoms_: int | None = None

    # ------------------------------------------------------------------ fit

    def fit(
        self,
        images: np.ndarray,
        reference_index: int = 0,
        mass_weighted: bool = True,
        masses: np.ndarray | None = None,
        species: Sequence[str] | None = None,
    ) -> PathPCA:
        """Build the basis from ``images`` of shape (N, M, 3).

        When ``mass_weighted`` is true, per-atom masses are required: pass
        them explicitly via ``masses`` (takes precedence), or pass ``species``
        so they can be looked up in the internal table. When ``mass_weighted``
        is false, ``masses``/``species`` are ignored and the Kabsch alignment
        uses uniform weights, keeping the whole pipeline in one consistent
        (purely geometric) metric.
        """
        arr = np.asarray(images, dtype=float)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"images must have shape (N, M, 3), got {arr.shape}")
        n_images, n_atoms, _ = arr.shape
        if n_images < 2:
            raise ValueError("need at least 2 images to build a displacement basis")
        if not -n_images <= reference_index < n_images:
            raise ValueError(
                f"reference_index {reference_index} out of range for {n_images} images"
            )

        mass_vec: np.ndarray | None = None
        if mass_weighted:
            if masses is not None:
                mass_vec = np.asarray(masses, dtype=float)
            elif species is not None:
                if len(species) != n_atoms:
                    raise ValueError(f"species has {len(species)} entries for {n_atoms} atoms")
                mass_vec = _masses_from_species(species)
            else:
                raise ValueError("mass_weighted fit requires either masses or species")
            if mass_vec.shape != (n_atoms,):
                raise ValueError(f"masses must have shape ({n_atoms},), got {mass_vec.shape}")
            if np.any(mass_vec <= 0.0):
                raise ValueError("masses must be strictly positive")

        reference = arr[reference_index]
        aligned = np.stack([kabsch_align(img, reference, mass_vec) for img in arr])
        displacements = (aligned - reference).reshape(n_images, 3 * n_atoms)
        if mass_vec is not None:
            displacements = displacements * np.repeat(np.sqrt(mass_vec), 3)

        _, singular_values, vt_mat = np.linalg.svd(displacements, full_matrices=False)
        total_variance = float((singular_values**2).sum())
        if total_variance == 0.0:
            raise ValueError("all images are identical to the reference; no variance to model")

        self.components_ = vt_mat
        self.explained_variance_ratio_ = singular_values**2 / total_variance
        self.cumulative_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)
        self.reference_ = reference.copy()
        self.mass_weighted_ = mass_weighted
        self.masses_ = mass_vec
        self.n_atoms_ = n_atoms
        return self

    @classmethod
    def from_state(
        cls,
        components: np.ndarray,
        explained_variance_ratio: np.ndarray,
        reference: np.ndarray,
        mass_weighted: bool,
        masses: np.ndarray | None,
    ) -> PathPCA:
        """Rebuild a fitted basis from serialized arrays.

        Exists so that ``coords.emit.read_basis`` can round-trip a basis file
        without re-running the fit; it is not intended as a general
        constructor.
        """
        reference = np.asarray(reference, dtype=float)
        if reference.ndim != 2 or reference.shape[1] != 3:
            raise ValueError(f"reference must have shape (M, 3), got {reference.shape}")
        n_atoms = reference.shape[0]
        components = np.asarray(components, dtype=float)
        if components.ndim != 2 or components.shape[1] != 3 * n_atoms:
            raise ValueError(
                f"components must have shape (k, {3 * n_atoms}), got {components.shape}"
            )
        ratios = np.asarray(explained_variance_ratio, dtype=float)
        if ratios.shape != (components.shape[0],):
            raise ValueError("explained_variance_ratio length must match component count")
        mass_vec: np.ndarray | None = None
        if mass_weighted:
            if masses is None:
                raise ValueError("a mass-weighted basis cannot be rebuilt without masses")
            mass_vec = np.asarray(masses, dtype=float)
            if mass_vec.shape != (n_atoms,):
                raise ValueError(f"masses must have shape ({n_atoms},), got {mass_vec.shape}")

        pca = cls()
        pca.components_ = components
        pca.explained_variance_ratio_ = ratios
        pca.cumulative_variance_ratio_ = np.cumsum(ratios)
        pca.reference_ = reference
        pca.mass_weighted_ = mass_weighted
        pca.masses_ = mass_vec
        pca.n_atoms_ = n_atoms
        return pca

    # ------------------------------------------------------------ inspection

    @property
    def n_components_(self) -> int:
        self._require_fitted()
        assert self.components_ is not None
        return int(self.components_.shape[0])

    def select_n_components(self, threshold: float = 0.98) -> int:
        """Smallest component count whose cumulative variance reaches ``threshold``.

        ``threshold`` is a documented, campaign-specific parameter — explicitly
        NOT a universal constant. The right value depends on how smooth the
        path is and how much reconstruction error the downstream consumer
        tolerates; choose it by inspecting ``cumulative_variance_ratio_`` and
        ``reconstruction_error`` rather than trusting the default.
        """
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")
        self._require_fitted()
        assert self.cumulative_variance_ratio_ is not None
        cumulative = self.cumulative_variance_ratio_
        # Tiny tolerance so threshold=1.0 is not defeated by float round-off.
        index = int(np.searchsorted(cumulative, threshold - 1e-12))
        return min(index, len(cumulative) - 1) + 1

    # ------------------------------------------------------- transformations

    def project(self, structures_or_array: Sequence[Structure] | np.ndarray) -> np.ndarray:
        """Amplitudes of each geometry in the basis; shape (N, k).

        Input is either an array of shape (N, M, 3) (a single (M, 3) geometry
        is promoted to N=1) or a sequence of ``Structure``. Every geometry is
        Kabsch-aligned to the stored reference first, with the same weighting
        the basis was built in, so the amplitudes measure internal deformation
        only.
        """
        self._require_fitted()
        assert self.components_ is not None
        geometries = self._coerce_geometries(structures_or_array)
        aligned = self._align(geometries)
        return self._displacements(aligned) @ self.components_.T

    def reconstruct(self, amplitudes: np.ndarray) -> np.ndarray:
        """Cartesian geometries for the given amplitudes; shape (N, M, 3).

        Inverts ``project`` up to basis truncation: the mass weighting is
        undone and the reference geometry added back, so the output is in
        plain Cartesian coordinates regardless of the fitting metric.
        """
        self._require_fitted()
        assert self.components_ is not None
        assert self.reference_ is not None
        assert self.n_atoms_ is not None
        amp = np.atleast_2d(np.asarray(amplitudes, dtype=float))
        k = self.components_.shape[0]
        if amp.ndim != 2 or amp.shape[1] != k:
            raise ValueError(f"amplitudes must have shape (N, {k}), got {amp.shape}")
        weighted = amp @ self.components_
        if self.masses_ is not None:
            weighted = weighted / np.repeat(np.sqrt(self.masses_), 3)
        return weighted.reshape(-1, self.n_atoms_, 3) + self.reference_

    def reconstruction_error(self, images: Sequence[Structure] | np.ndarray) -> np.ndarray:
        """Per-image RMSD between each input and its basis round trip; shape (N,).

        Each image is aligned to the reference, projected, and reconstructed;
        the RMSD is then taken between the aligned input and the
        reconstruction. The RMSD is *unweighted* Cartesian even for a
        mass-weighted basis — a deliberate interpretability choice, so the
        error is always in plain geometric distance units.
        """
        self._require_fitted()
        assert self.components_ is not None
        geometries = self._coerce_geometries(images)
        aligned = self._align(geometries)
        amplitudes = self._displacements(aligned) @ self.components_.T
        rebuilt = self.reconstruct(amplitudes)
        squared = ((aligned - rebuilt) ** 2).sum(axis=2)
        return np.sqrt(squared.mean(axis=1))

    # -------------------------------------------------------------- internal

    def _require_fitted(self) -> None:
        if self.components_ is None:
            raise RuntimeError("PathPCA is not fitted; call fit() or from_state() first")

    def _coerce_geometries(
        self, structures_or_array: Sequence[Structure] | np.ndarray
    ) -> np.ndarray:
        assert self.n_atoms_ is not None
        if isinstance(structures_or_array, np.ndarray):
            arr = np.asarray(structures_or_array, dtype=float)
            if arr.ndim == 2:
                arr = arr[None, :, :]
        else:
            arr = np.stack([np.asarray(s.positions, dtype=float) for s in structures_or_array])
        if arr.ndim != 3 or arr.shape[1] != self.n_atoms_ or arr.shape[2] != 3:
            raise ValueError(
                f"expected geometries of shape (N, {self.n_atoms_}, 3), got {arr.shape}"
            )
        return arr

    def _align(self, geometries: np.ndarray) -> np.ndarray:
        assert self.reference_ is not None
        return np.stack([kabsch_align(g, self.reference_, self.masses_) for g in geometries])

    def _displacements(self, aligned: np.ndarray) -> np.ndarray:
        assert self.reference_ is not None
        flat = (aligned - self.reference_).reshape(aligned.shape[0], -1)
        if self.masses_ is not None:
            flat = flat * np.repeat(np.sqrt(self.masses_), 3)
        return flat
