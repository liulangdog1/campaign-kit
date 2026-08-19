# campaign-kit

[![CI](https://github.com/liulangdog1/campaign-kit/actions/workflows/ci.yml/badge.svg)](https://github.com/liulangdog1/campaign-kit/actions/workflows/ci.yml)

Active-learning orchestration and coordinate reduction for simulation campaigns whose ground-truth
labels are expensive batch calculations.

## The problem

A single reference calculation on a system like a transition-metal complex costs hundreds of
CPU-hours, so a training set is bought a few dozen labels at a time. The binding constraint is not
model architecture; it is deciding *which* structures are worth labeling next, and noticing when
the surrogate has silently left its training domain and is extrapolating with unwarranted
confidence. Managing such a campaign by hand — tracking queued jobs, classifying failures,
deduplicating labels, restarting after a dead node — does not scale past the first week and does
not survive the first cluster incident. This kit is that management layer, written so the whole
loop also runs on a laptop against an analytic stand-in labeler.

## Quickstart

```sh
git clone https://github.com/liulangdog1/campaign-kit.git && cd campaign-kit
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python examples/02_domain_detection.py
```

Output (a committee trained on two windows of a two-atom energy curve, then queried in the
unsampled gap between them and beyond the outer edge):

```
training windows: r in [1.0, 1.9] and [3.3, 4.5]  (96 labels)   fence threshold: 0.0256

    r       set  dist to train   spread  |true error|   fence  silent-check
---------------------------------------------------------------------------
 1.50 train-win             ~0    0.028         0.016       -             -
 1.85 train-win             ~0    0.026         0.044       -             -
 2.30     query         0.0982    0.046         0.254     OUT       FLAGGED
 2.45     query         0.1107    0.049         1.359     OUT       FLAGGED
 2.60     query         0.0872    0.054         2.098     OUT       FLAGGED
 2.90     query         0.0474    0.061         0.224     OUT
 3.50 train-win             ~0    0.053         0.158       -             -
 4.40 train-win             ~0    0.046         0.152       -             -
 4.80     query         0.0156    0.078         0.278  inside
 5.40     query         0.0388    0.116         0.442     OUT

in the gap   : spread 0.033-0.063 (in-window scale: 0.008-0.053) while |error| reaches 2.098 — SILENT
             : that worst gap error is 62x the median in-window error (0.034); the committee never says so
beyond r=4.5: spread rises to 0.116 — the committee announces that extrapolation itself
```

**What "in-window" means here.** Six control rows are evaluated from inside the training windows —
r = 1.50, 1.70, 1.85, 3.50, 4.00, 4.40. The median of their absolute errors (0.034) is the
denominator for every `Nx` figure the demo prints, and the range of their committee spreads
(0.008-0.053) is the "in-window scale" the gap rows are compared against. Both are computed in
`examples/02_domain_detection.py`; nothing here is hand-derived.

(Table abridged; the script prints all 20 rows.) In the gap between the two training windows the
committee members agree on the same smooth bridge and are wrong together: spread sits at
0.033-0.063, inside the in-window scale, while the true error reaches 2.098 — **62x** the median
in-window error. Past the outer window the same committee announces its extrapolation loudly. The
fence — a geometric criterion calibrated on the training set's own spacing — flags every gap row
without spending a label, and the demo's closing lines show its limit too: one admitted row just
past the outer window still carries 8x the median in-window error.

## What's here

**Module A — campaign orchestration** (`campaign_kit.loop`, `committee`, `selection`, `domain`,
`scheduler`):

- A resumable campaign driver: every phase writes a checkpoint, structures are keyed by content
  hash, and dataset merge is idempotent — a campaign killed mid-round resumes without re-buying
  any label.
- Typed failure handling end to end: a label request terminates as a classified outcome
  (convergence failure, timeout, node failure, vanished job), and a retry policy decides per
  class what is worth resubmitting.
- Two-stage batch selection (query-by-committee band, then farthest-point diversification) plus a
  training-domain fence and a silent-extrapolation check that pairs committee spread with an
  independent geometric criterion.

**Module B — coordinate reduction** (`campaign_kit.coords`):

- Path-PCA over rigidly aligned images of a reaction path: mass-weighted displacement modes with
  explained-variance bookkeeping, projection, and exact reconstruction.
- A versioned on-disk basis format (JSON with an optional binary sidecar) so externally authored
  optimizers can consume the reduced coordinates without importing this package's internals.

## Design notes

The reasoning behind the load-bearing choices — the abstract labeler seam, per-atom rather than
per-structure variance, why ensemble variance alone is not an out-of-domain signal, crash-safe
resumption, the scheduler failure taxonomy, and the two-stage selector — is written up in
[docs/DESIGN.md](docs/DESIGN.md).

## Scope

What this repository deliberately does and does not contain is stated in
[docs/SCOPE.md](docs/SCOPE.md).

## Status

Developed alongside two manuscripts in preparation. This repository contains infrastructure only —
no reference data, trained weights, or results.

## License

MIT — see [LICENSE](LICENSE).
