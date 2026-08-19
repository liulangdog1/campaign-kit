# campaign-kit

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
git clone <repository-url> && cd campaign-kit
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python examples/02_domain_detection.py
```

Output (a committee trained on the short-range window of a two-atom energy curve, then queried far
past it — the true curve hides a feature the training window carries no trace of):

```
training window: r in [1.0, 2.4]  (64 labels)   fence threshold: 0.0387

    r  dist to train   spread  |true error|   fence  silent-check
-----------------------------------------------------------------
 1.20         0.0034    0.071         0.208  inside
 1.50         0.0105    0.032         0.005  inside
 2.10         0.0040    0.013         0.000  inside
 2.30         0.0008    0.036         0.076  inside
 2.70         0.0528    0.208         0.289     OUT
 3.00         0.0899    0.312         0.365     OUT
 3.80         0.1601    0.516         0.478     OUT
 4.20         0.1851    0.587         2.733     OUT
 4.70         0.2105    0.661         0.209     OUT
 5.00         0.2232    0.699         0.116     OUT

inside the fence : median spread 0.034, worst |error| 0.208
outside the fence: max spread    0.699, worst |error| 2.733 (13x the worst in-domain error)
```

(Table abridged; the script prints all 14 query rows.) Committee spread grows smoothly with
distance and carries no hint of the hidden feature near r = 4.2, where the true error jumps an
order of magnitude above its neighbours at essentially the same spread. The fence — a geometric
criterion calibrated on the training set's own spacing — rejects every extrapolating row before
any label is spent.

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
