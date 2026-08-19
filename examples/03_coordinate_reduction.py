"""Demo 03: recover the intrinsic dimensionality of a synthetic path.

A smooth motion of 8 pseudo-atoms is built from exactly two independent
displacement modes, embedded in full Cartesian space (24 coordinates), and
lightly corrupted with noise. ``PathPCA`` should look at the 24-dimensional
images and report that two components explain essentially everything — and the
demo checks that it does, rather than asking the reader to eyeball it.

Two constructions make the test honest:

- The two mode vectors are projected orthogonal to the six rigid-body motions
  (translations and infinitesimal rotations) of the base geometry. PathPCA
  removes rigid-body motion by Kabsch alignment before the PCA, so a mode with
  a rigid component would be partially absorbed by the alignment and the
  "exactly two components" ground truth would be blurred.
- The two amplitude sequences are linearly independent functions of the path
  parameter, so the images genuinely span both modes instead of a diagonal
  line through them.

The end of the demo exercises the emit/read round trip: the truncated basis is
written to a file and read back, and the reloaded basis must project the
images identically.

Run:  python examples/03_coordinate_reduction.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from campaign_kit.coords import PathPCA, read_basis, write_basis

N_ATOMS: int = 8
N_IMAGES: int = 40
NOISE_SIGMA: float = 0.01  # Cartesian noise, generic length units


def check(condition: bool, label: str) -> None:
    """Print a check result and stop the demo on failure — self-validating output."""
    print(f"  [{'ok' if condition else 'FAIL'}] {label}")
    if not condition:
        raise SystemExit(1)


def rigid_body_modes(base: np.ndarray) -> np.ndarray:
    """Orthonormal basis of the 6 rigid-body motions of a geometry, shape (6, 3M).

    Three uniform translations plus three infinitesimal rotations about the
    centroid. Needed so the synthetic modes can be made purely internal (see
    module docstring for why that matters).
    """
    n = base.shape[0]
    centered = base - base.mean(axis=0)
    modes = []
    for axis in range(3):
        t = np.zeros((n, 3))
        t[:, axis] = 1.0
        modes.append(t.ravel())
    for axis in range(3):
        unit = np.zeros(3)
        unit[axis] = 1.0
        modes.append(np.cross(centered, unit).ravel())
    stacked = np.stack(modes)
    # QR orthonormalization; rows stay a basis of the same 6D space.
    q, _ = np.linalg.qr(stacked.T)
    return q.T[:6]


def internal_mode(rng: np.random.Generator, excluded_span: np.ndarray) -> np.ndarray:
    """A random unit displacement field orthogonal to ``excluded_span`` rows."""
    vec = np.asarray(rng.normal(size=excluded_span.shape[1]))
    vec -= excluded_span.T @ (excluded_span @ vec)
    return vec / float(np.linalg.norm(vec))


def main() -> None:
    print("Demo 03: path PCA recovers a known 2-mode motion from 24 Cartesian coordinates")
    print()

    rng = np.random.default_rng(5)
    base = rng.uniform(-1.5, 1.5, size=(N_ATOMS, 3))

    rigid = rigid_body_modes(base)
    mode_1 = internal_mode(rng, rigid)
    mode_2 = internal_mode(rng, np.vstack([rigid, mode_1[None, :]]))

    # Two linearly independent amplitude profiles along the path parameter.
    t = np.linspace(0.0, 1.0, N_IMAGES)
    amp_1 = 1.2 * t
    amp_2 = 0.8 * np.sin(np.pi * t)

    displacements = amp_1[:, None] * mode_1[None, :] + amp_2[:, None] * mode_2[None, :]
    images = (
        base[None, :, :]
        + displacements.reshape(N_IMAGES, N_ATOMS, 3)
        + rng.normal(0.0, NOISE_SIGMA, size=(N_IMAGES, N_ATOMS, 3))
    )

    # Pseudo-atoms: uniform weights, so the basis is purely geometric and no
    # element identities (or masses) enter the demo at all.
    pca = PathPCA().fit(images, reference_index=0, mass_weighted=False)

    assert pca.cumulative_variance_ratio_ is not None
    print("cumulative explained variance:")
    for k, value in enumerate(pca.cumulative_variance_ratio_[:6], start=1):
        bar = "#" * int(round(50 * value))
        print(f"  k={k}  {value:.5f}  {bar}")
    print("  ...")
    print()

    chosen = pca.select_n_components()  # documented default threshold: 0.98
    print(f"components chosen at the default variance threshold: {chosen}")
    check(chosen == 2, "chosen component count equals the true intrinsic dimensionality (2)")

    # Truncate to the chosen components and measure the round-trip error.
    assert pca.components_ is not None
    assert pca.explained_variance_ratio_ is not None
    assert pca.reference_ is not None
    truncated = PathPCA.from_state(
        components=pca.components_[:chosen],
        explained_variance_ratio=pca.explained_variance_ratio_[:chosen],
        reference=pca.reference_,
        mass_weighted=False,
        masses=None,
    )
    errors = truncated.reconstruction_error(images)
    print(
        f"  2-component round-trip RMSD per image: mean {errors.mean():.4f}, "
        f"max {errors.max():.4f}  (injected noise sigma: {NOISE_SIGMA})"
    )
    check(
        float(errors.max()) < 4.0 * NOISE_SIGMA,
        "truncated reconstruction error sits at the injected noise floor",
    )
    print()

    # Emit the truncated basis and read it back — the file, not the in-memory
    # object, is what a downstream consumer would receive.
    with tempfile.TemporaryDirectory(prefix="campaign_demo03_") as tmp:
        prefix = Path(tmp) / "path_basis"
        atom_order = [f"P{i + 1}" for i in range(N_ATOMS)]
        json_path = write_basis(
            prefix,
            truncated,
            atom_order=atom_order,
            meta={"demo": "03_coordinate_reduction", "n_images": N_IMAGES},
        )
        print(f"basis written: {json_path.name} ({json_path.stat().st_size} bytes)")
        loaded = read_basis(json_path)

        assert loaded.pca.components_ is not None
        assert truncated.components_ is not None
        check(loaded.format_version == 1, "format version round-trips")
        check(loaded.atom_order == atom_order, "atom order round-trips")
        check(
            np.allclose(loaded.pca.components_, truncated.components_),
            "component matrix round-trips",
        )
        check(
            np.allclose(loaded.pca.project(images), truncated.project(images)),
            "reloaded basis projects the images identically",
        )

    print()
    print("all checks passed")


if __name__ == "__main__":
    main()
