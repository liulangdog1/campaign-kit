"""Demo 02: low committee spread is not evidence of accuracy.

The setup is a deliberate trap. A committee is trained on two windows of a
two-atom energy curve — short separations and long separations — and then
queried in between and beyond. The true curve hides a feature (an attractive
well) in the unsampled middle window. Committee members share one
architecture, one training set, and one set of inductive biases, so in that
gap they all bridge the two windows with the same smooth guess: they agree
with each other and are wrong together. The table below shows the spread
sitting at its in-window scale while the true error grows by more than an
order of magnitude.

The contrast that makes this the interesting failure mode: *beyond* the outer
window, where the committee openly extrapolates, the spread does rise — the
committee announces that problem itself. In the gap it stays quiet. A model
can fail loudly or silently, and only the silent case needs an independent
signal.

That independent signal is geometric. Descriptor-space distance to the
training set does not care what the models believe: the ``DomainFence`` built
on it rejects every gap row, and ``silent_extrapolation_check`` flags the
"far and quiet" combination specifically.

Run:  python examples/02_domain_detection.py
"""

from __future__ import annotations

import math

import numpy as np

from campaign_kit.backends import SklearnMLPModel
from campaign_kit.committee import Committee
from campaign_kit.domain import DomainFence, TrainingDomain, silent_extrapolation_check
from campaign_kit.protocols import Dataset, Structure

# The two training windows of atom-atom separation. The models never see a
# label between them or beyond the outer one.
WINDOW_NEAR: tuple[float, float] = (1.0, 1.9)
WINDOW_FAR: tuple[float, float] = (3.3, 4.5)
N_PER_WINDOW: int = 48

# Hidden feature: an attractive well centered inside the unsampled gap. At the
# window edges its contribution is numerically zero, so the training data
# carries no trace of it — that is the whole point.
HIDDEN_DEPTH: float = -2.5
HIDDEN_CENTER: float = 2.6
HIDDEN_WIDTH: float = 0.25


def true_energy(r: float) -> float:
    """Bonded-well curve plus the hidden feature, in generic energy/length units.

    The first term is a standard anharmonic well (repulsive at short range,
    flattening toward zero at large separation); the second is a Gaussian dip
    the models never get to see.
    """
    near = 4.0 * (1.0 - math.exp(-1.8 * (r - 1.4))) ** 2 - 4.0
    hidden = HIDDEN_DEPTH * math.exp(-(((r - HIDDEN_CENTER) / HIDDEN_WIDTH) ** 2))
    return near + hidden


def two_atom(r: float) -> Structure:
    """A two-atom structure at separation ``r`` — the minimal geometry with a descriptor."""
    return Structure(
        species=("A", "B"),
        positions=np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=float),
    )


def main() -> None:
    print("Demo 02: silent extrapolation — where committee spread stays low and the error does not")
    print()

    rng = np.random.default_rng(3)
    train_r = np.sort(
        np.concatenate(
            [
                rng.uniform(*WINDOW_NEAR, size=N_PER_WINDOW),
                rng.uniform(*WINDOW_FAR, size=N_PER_WINDOW),
            ]
        )
    )
    train = Dataset(
        structures=[two_atom(float(r)) for r in train_r],
        energies=[true_energy(float(r)) for r in train_r],
        forces=[None] * len(train_r),
    )

    # A deliberately smooth committee (one modest hidden layer, strong weight
    # decay): each member settles on nearly the same low-curvature bridge
    # across the gap, which is exactly what makes the failure there silent.
    committee = Committee(
        model_factory=lambda: SklearnMLPModel(hidden_layer_sizes=(64,), max_iter=3000, alpha=0.1),
        n_members=6,
        base_seed=1,
    )
    committee.fit(train)

    domain = TrainingDomain(train)
    fence = DomainFence.from_train_quantile(domain, quantile=0.99)

    # Rows the models were trained around (accuracy baseline), and query rows:
    # a scan through the unsampled gap, then past the outer window.
    control_r = [1.5, 1.7, 1.85, 3.5, 4.0, 4.4]
    query_r = [2.0, 2.15, 2.3, 2.45, 2.6, 2.75, 2.9, 3.05, 4.8, 5.1, 5.4]
    queries = [two_atom(r) for r in query_r]

    control_pred = committee.predict_mean_and_spread([two_atom(r) for r in control_r])
    control_err = np.abs(control_pred.energies - np.array([true_energy(r) for r in control_r]))
    predictions = committee.predict_mean_and_spread(queries)
    abs_error = np.abs(predictions.energies - np.array([true_energy(r) for r in query_r]))
    distances = domain.distance(queries)
    admitted = fence.check(queries)

    # The check screens *candidate* structures, so it runs on the query rows
    # only. "Quiet" is batch-relative; this batch is dominated by the silent
    # region itself, so the default "quiet = bottom quartile" is over-strict
    # and the median is the honest reading of "no louder than typical here".
    # (With the generic defaults the check still flags the r = 2.30 row.)
    flags = silent_extrapolation_check(queries, predictions, domain, spread_quantile=0.5)
    # Warning strings start with "structure <pool index>" (format defined in
    # campaign_kit.domain), so the flagged indices can be recovered.
    flagged = {int(w.split()[1]) for w in flags}

    print(
        f"training windows: r in [{WINDOW_NEAR[0]:.1f}, {WINDOW_NEAR[1]:.1f}] and "
        f"[{WINDOW_FAR[0]:.1f}, {WINDOW_FAR[1]:.1f}]  ({len(train)} labels)   "
        f"fence threshold: {fence.threshold:.4f}"
    )
    print()
    header = (
        f"{'r':>5} {'set':>9} {'dist to train':>14} {'spread':>8} "
        f"{'|true error|':>13} {'fence':>7} {'silent-check':>13}"
    )
    print(header)
    print("-" * len(header))
    rows = sorted(
        [
            (r, "train-win", None, s, e, None, None)
            for r, s, e in zip(
                control_r, control_pred.per_structure_spread, control_err, strict=True
            )
        ]
        + [
            (
                query_r[i],
                "query",
                float(distances[i]),
                predictions.per_structure_spread[i],
                abs_error[i],
                bool(admitted[i]),
                i in flagged,
            )
            for i in range(len(query_r))
        ]
    )
    for r, kind, dist, spread, err, inside, is_flagged in rows:
        dist_txt = "~0" if dist is None else f"{dist:.4f}"
        fence_txt = "-" if inside is None else ("inside" if inside else "OUT")
        flag_txt = "-" if is_flagged is None else ("FLAGGED" if is_flagged else "")
        print(
            f"{r:>5.2f} {kind:>9} {dist_txt:>14} {spread:>8.3f} "
            f"{err:>13.3f} {fence_txt:>7} {flag_txt:>13}".rstrip()
        )

    gap_idx = [i for i, r in enumerate(query_r) if r < 3.3]
    beyond_idx = [i for i, r in enumerate(query_r) if r > 4.5]
    print()
    print(
        f"in the gap   : spread {predictions.per_structure_spread[gap_idx].min():.3f}-"
        f"{predictions.per_structure_spread[gap_idx].max():.3f} "
        f"(in-window scale: {control_pred.per_structure_spread.min():.3f}-"
        f"{control_pred.per_structure_spread.max():.3f}) "
        f"while |error| reaches {abs_error[gap_idx].max():.3f} — SILENT"
    )
    print(
        f"beyond r={WINDOW_FAR[1]:.1f}: spread rises to "
        f"{predictions.per_structure_spread[beyond_idx].max():.3f} — the committee "
        f"announces that extrapolation itself"
    )
    edge_ratio = abs_error[query_r.index(4.8)] / float(np.median(control_err))
    print(
        "note the admitted r=4.80 row: it sits just past the outer window, inside the\n"
        f"calibrated threshold, with {edge_ratio:.0f}x the median in-window error — a\n"
        "fence shrinks the risk of silent extrapolation; it does not eliminate it"
    )
    print()
    print(
        "(the check ranks on local, per-atom spread, so its warnings quote that\n"
        " statistic — for these 2-atom structures it is about half the table's\n"
        " global spread)"
    )
    for w in flags:
        print(f"warning: {w}")
    print()
    print("Lesson:")
    print(
        "  Committee spread measures model-to-model variance, and the members share the\n"
        "  same data and biases — in the unsampled gap they agree on the same smooth\n"
        "  bridge and are wrong together, so variance alone is not an out-of-domain\n"
        "  signal. Distance to the training data in descriptor space is independent of\n"
        "  what the models believe; gate on it, and treat 'far but quiet' as the case\n"
        "  to distrust most."
    )


if __name__ == "__main__":
    main()
