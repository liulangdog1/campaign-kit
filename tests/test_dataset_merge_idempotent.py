"""Dataset.merge deduplication — the property crash recovery leans on.

The campaign driver deliberately does not track which results were merged;
it re-collects after a crash and relies on content-hash deduplication to keep
the dataset clean. These tests pin that contract down, including the id
stability under tiny position perturbations below the hashing precision.
"""

from __future__ import annotations

import numpy as np

from campaign_kit.protocols import Dataset, LabelResult, LabelStatus, Structure
from tests._doubles import analytic_energy, jittered_structures


def test_merge_is_idempotent_and_skips_failures() -> None:
    structures = jittered_structures(4, scale=0.2, seed=1)
    results = [LabelResult.success(s, analytic_energy(s)) for s in structures[:3]]
    results.append(
        LabelResult.failure(structures[3], LabelStatus.FAILED_CONVERGENCE, "did not converge")
    )
    ds = Dataset()
    assert ds.merge(results) == 3  # failures carry no label row
    assert ds.merge(results) == 0  # merging the same results again adds nothing
    assert len(ds) == 3
    assert ds.ids == {s.structure_id for s in structures[:3]}


def test_merge_deduplicates_within_a_single_call() -> None:
    s = jittered_structures(1, scale=0.2, seed=2)[0]
    ds = Dataset()
    added = ds.merge([LabelResult.success(s, 1.0), LabelResult.success(s, 2.0)])
    assert added == 1
    assert ds.energies == [1.0]  # first result wins; the duplicate is dropped


def test_structure_id_survives_sub_precision_noise() -> None:
    # Positions are hashed after rounding at 1e-10, so serialization noise
    # below that precision cannot change identity (and therefore cannot defeat
    # deduplication after a JSON round trip).
    positions = np.array([[0.0, 0.0, 0.0], [1.234567890123, 0.0, 0.0]])
    a = Structure(("C", "O"), positions)
    b = Structure(("C", "O"), positions + 1e-12)
    assert a.structure_id == b.structure_id

    ds = Dataset()
    assert ds.merge([LabelResult.success(a, 1.0), LabelResult.success(b, 2.0)]) == 1


def test_energy_array_matches_rows() -> None:
    structures = jittered_structures(3, scale=0.2, seed=6)
    ds = Dataset()
    ds.merge([LabelResult.success(s, float(i)) for i, s in enumerate(structures)])
    assert np.array_equal(ds.energy_array(), np.array([0.0, 1.0, 2.0]))
