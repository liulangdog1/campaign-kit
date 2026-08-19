"""Farthest-point sampling must actually spread its picks.

The claim behind FPS is geometric: greedy max-min selection covers the pool
better than chance. The test measures the minimum pairwise descriptor
distance within the selected batch — the quantity FPS greedily maximizes —
and requires it to beat every one of a set of fixed-seed random draws, not
just their average.
"""

from __future__ import annotations

import numpy as np

from campaign_kit.domain import default_descriptor
from campaign_kit.selection import FarthestPointSampling
from tests._doubles import jittered_structures


def _min_pairwise_distance(descriptors: np.ndarray, indices: list[int]) -> float:
    chosen = descriptors[np.asarray(indices)]
    diff = chosen[:, None, :] - chosen[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=-1))
    upper = dist[np.triu_indices(len(indices), k=1)]
    return float(upper.min())


def test_fps_spread_beats_random_selection() -> None:
    pool = jittered_structures(50, scale=0.4, seed=4)
    descriptors = np.vstack([default_descriptor(s) for s in pool])

    fps_indices = FarthestPointSampling(default_descriptor).select(
        pool, 8, np.random.default_rng(0)
    )
    fps_score = _min_pairwise_distance(descriptors, fps_indices)

    random_scores = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        picks = rng.choice(len(pool), size=8, replace=False).tolist()
        random_scores.append(_min_pairwise_distance(descriptors, picks))

    # Deterministic (all seeds fixed): the greedy max-min batch is better
    # spread than every random batch tried.
    assert fps_score > max(random_scores)
