"""Path-PCA fidelity and the basis-file round trip.

The synthetic path is built to have a known answer: images move inside an
exactly two-dimensional internal subspace (both directions are projected
clean of rigid-body translation and rotation, so the Kabsch alignment cannot
leak variance into extra components). The basis must recover that dimension,
reconstruct the images to numerical precision, and survive serialization
byte-faithfully.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from campaign_kit.coords import PathPCA, read_basis, write_basis

BASE = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.4, 0.0, 0.0],
        [0.0, 1.2, 0.3],
        [0.7, 0.7, 1.1],
    ]
)
ATOM_ORDER = ["C", "O", "H", "H"]
REFERENCE_INDEX = 12  # center of the 5x5 grid: amplitudes (0, 0) -> exactly BASE


def _rigid_space(base: np.ndarray) -> np.ndarray:
    """Orthonormal basis (6, 3M) of infinitesimal rigid-body motions at `base`."""
    m = base.shape[0]
    modes = []
    for axis in range(3):
        translation = np.zeros((m, 3))
        translation[:, axis] = 1.0
        modes.append(translation.ravel())
    centered = base - base.mean(axis=0)
    for rotation_axis in np.eye(3):
        modes.append(np.cross(centered, rotation_axis).ravel())
    q, _ = np.linalg.qr(np.stack(modes, axis=1))
    return q.T


def _internal_direction(base: np.ndarray, seed: int) -> np.ndarray:
    """A unit displacement field orthogonal to all six rigid-body motions."""
    rng = np.random.default_rng(seed)
    direction = rng.standard_normal(base.size)
    rigid = _rigid_space(base)
    direction = direction - rigid.T @ (rigid @ direction)
    return (direction / np.linalg.norm(direction)).reshape(base.shape)


def _two_mode_images() -> np.ndarray:
    d1 = _internal_direction(BASE, seed=1)
    d2 = _internal_direction(BASE, seed=2)
    # Orthogonalize the second direction against the first so the two motion
    # amplitudes land in two distinct principal components.
    flat = d2.ravel() - (d2.ravel() @ d1.ravel()) * d1.ravel()
    d2 = (flat / np.linalg.norm(flat)).reshape(BASE.shape)
    amplitudes_1 = np.linspace(-0.15, 0.15, 5)
    amplitudes_2 = np.linspace(-0.1, 0.1, 5)
    return np.stack([BASE + t * d1 + s * d2 for t in amplitudes_1 for s in amplitudes_2])


def _fitted() -> tuple[PathPCA, np.ndarray]:
    images = _two_mode_images()
    pca = PathPCA().fit(images, reference_index=REFERENCE_INDEX, mass_weighted=False)
    return pca, images


def test_two_mode_path_needs_exactly_two_components() -> None:
    pca, _ = _fitted()
    assert pca.explained_variance_ratio_ is not None
    assert pca.explained_variance_ratio_[0] < 0.98  # one component is not enough
    assert pca.select_n_components(threshold=0.98) == 2


def test_reconstruction_error_is_numerically_zero_with_full_basis() -> None:
    pca, images = _fitted()
    errors = pca.reconstruction_error(images)
    assert errors.shape == (len(images),)
    assert errors.max() < 1e-8


def test_reference_image_projects_to_zero_amplitudes() -> None:
    pca, images = _fitted()
    amplitudes = pca.project(images)
    assert np.allclose(amplitudes[REFERENCE_INDEX], 0.0, atol=1e-10)


def test_emit_read_roundtrip_preserves_basis_and_metadata(tmp_path: Path) -> None:
    pca, images = _fitted()
    meta: dict[str, object] = {"purpose": "unit test", "grid": [5, 5]}
    json_path = write_basis(tmp_path / "basis", pca, atom_order=ATOM_ORDER, meta=meta)
    assert json_path == tmp_path / "basis.json"

    loaded = read_basis(tmp_path / "basis")  # prefix form must also resolve
    assert loaded.format_version == 1
    assert loaded.atom_order == ATOM_ORDER
    assert loaded.meta == meta
    assert loaded.pca.components_ is not None
    assert pca.components_ is not None
    assert loaded.pca.reference_ is not None
    assert pca.reference_ is not None
    assert np.array_equal(loaded.pca.components_, pca.components_)
    assert np.array_equal(loaded.pca.reference_, pca.reference_)
    assert np.allclose(loaded.pca.project(images), pca.project(images))
    assert loaded.pca.select_n_components(threshold=0.98) == 2


def test_emit_sidecar_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the large-basis path: components go to an .npz sidecar and the
    # JSON only references it by basename.
    monkeypatch.setattr("campaign_kit.coords.emit.INLINE_COMPONENT_LIMIT", 1)
    pca, images = _fitted()
    json_path = write_basis(tmp_path / "big", pca, atom_order=ATOM_ORDER, meta={})
    assert (tmp_path / "big.npz").exists()
    document = json.loads(json_path.read_text())
    assert document["components"] == {"npz_file": "big.npz", "key": "components"}

    loaded = read_basis(json_path)
    assert loaded.pca.components_ is not None
    assert pca.components_ is not None
    assert np.array_equal(loaded.pca.components_, pca.components_)
    assert np.allclose(loaded.pca.project(images), pca.project(images))


def test_mass_weighted_roundtrip_keeps_masses_and_reference(tmp_path: Path) -> None:
    images = _two_mode_images()
    pca = PathPCA().fit(
        images, reference_index=REFERENCE_INDEX, mass_weighted=True, species=ATOM_ORDER
    )
    assert pca.masses_ is not None
    write_basis(tmp_path / "mw", pca, atom_order=ATOM_ORDER, meta={})
    loaded = read_basis(tmp_path / "mw.json")
    assert loaded.pca.masses_ is not None
    assert np.array_equal(loaded.pca.masses_, pca.masses_)
    assert np.allclose(loaded.pca.project(images), pca.project(images))
    # Amplitude zero must rebuild the reference geometry, in plain Cartesian
    # coordinates, even though the basis lives in the mass-weighted metric.
    assert pca.reference_ is not None
    rebuilt = loaded.pca.reconstruct(np.zeros((1, loaded.pca.n_components_)))
    assert np.allclose(rebuilt[0], pca.reference_)


def test_read_rejects_unknown_format_version(tmp_path: Path) -> None:
    pca, _ = _fitted()
    json_path = write_basis(tmp_path / "v", pca, atom_order=ATOM_ORDER, meta={})
    document = json.loads(json_path.read_text())
    document["format_version"] = 999
    json_path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="format_version"):
        read_basis(json_path)


def test_mass_weighted_fit_requires_known_masses() -> None:
    images = _two_mode_images()
    with pytest.raises(ValueError, match="mass"):
        PathPCA().fit(images, mass_weighted=True, species=["Zz", "O", "H", "H"])
