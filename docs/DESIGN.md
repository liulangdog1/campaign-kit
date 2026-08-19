# Design notes

Why the load-bearing pieces of campaign-kit are shaped the way they are. Each section states a
failure mode observed in real campaigns and the structural decision that answers it.

## Why an abstract `Labeler`

The expensive backend — a batch quantum-chemistry code dispatched to a cluster — is hidden behind
a two-method protocol: `submit(candidates) -> handles`, `collect(handles) -> results`. Nothing
downstream can tell whether a label came from a multi-hour cluster job or from a closed-form
analytic function that returns instantly.

That seam exists for testability, not elegance. A campaign driver whose only integration test
requires a cluster allocation is effectively untested: the interesting code paths (crash between
submit and collect, partial batch failure, duplicate collection) are exactly the ones you cannot
afford to provoke on real hardware. With the analytic labeler, the full loop — including kill and
resume — runs in seconds in CI. The demos and the test suite are the same code that runs in
production with one constructor argument changed.

The protocol also forces failure into the type system: `collect` must return one result per
handle, failures included, and may not raise merely because a calculation failed. A labeler that
can only report success is a labeler that turns the first node failure into a crashed campaign.

## Why per-atom variance, not per-structure

Committee disagreement is the selection signal, and the kit computes it per atom, ranking on the
local maximum rather than the structure-level scalar.

The reason is that a structure is not uniformly novel. A candidate can be well covered by training
data almost everywhere while one region — a stretched contact, an unusual local arrangement — is
extrapolating. Averaged over all atoms, that localized uncertainty is diluted below the selection
threshold; a global scalar systematically under-ranks exactly the structures that carry new
information in a small region. Ranking on the per-atom maximum keeps a locally novel structure
selectable even when the rest of it is boring. The structure-level scalar is still reported, and
selection falls back to it when a backend cannot decompose its prediction.

## Why variance alone is an insufficient out-of-domain signal

It is tempting to treat low committee spread as evidence of accuracy. It is not. Committee members
share an architecture, a training set, and inductive biases; far from the data, nothing constrains
them, and their errors become correlated. A committee can agree confidently and be wrong together.
Spread measures model-to-model variance, never model-to-truth error.

The kit therefore pairs spread with an independent geometric criterion: nearest-neighbour distance
to the training set in descriptor space (`TrainingDomain`), with a fence whose threshold is
calibrated from the training set's own spacing (`DomainFence.from_train_quantile`). The
`silent_extrapolation_check` flags the specific combination neither signal catches alone — far
from the data yet quiet.

This is a mitigation, not a solution. The check flags a suspicious conjunction of two proxies; it
does not rank per-point error, and no continuous signal available here is known to do that
reliably. Flagged structures are candidates for labeling, not measured failures. The honest
statement of the state of the art is that detecting confident extrapolation from model-side
signals alone remains open, and the kit's design treats it as such rather than pretending the
ensemble bar is a certificate.

## Why campaigns must be resumable

A real campaign spans days to weeks of wall clock, across which node failures, queue drains, and
maintenance windows are not exceptional events — they are the baseline. A driver that must restart
from scratch after an interruption re-spends labels that each cost hundreds of CPU-hours, which is
the one resource the campaign exists to conserve.

Resumability rests on three mechanisms working together:

1. **Content-hash identity.** A structure's id is a hash of its species and (rounded) positions,
   so identity survives serialization and process boundaries.
2. **Idempotent merge.** `Dataset.merge` keys rows by that id: merging the same results twice
   cannot create duplicates. After a crash between collect and merge, the driver simply
   re-collects; the merge is a no-op for anything already absorbed.
3. **Phase-level checkpoints.** The campaign state — dataset, holdout, pending handles, RNG state,
   budget counters — is written atomically after each phase. Labels are claimed at submit time, so
   a crash between submit and collect cannot cause the same structure to be paid for twice.

Model weights are deliberately not checkpointed: they are cheap to recompute from the dataset,
and refitting on resume avoids a whole class of stale-weights-versus-dataset consistency bugs.

## Scheduler failure taxonomy

The scheduler layer classifies every job into a terminal state rather than a boolean, because the
correct reaction differs per class:

| State | Meaning | Reaction |
| --- | --- | --- |
| `COMPLETED` | Terminal evidence of success | Collect the output. |
| `FAILED` | The calculation itself failed | Do not retry: the same input fails again. |
| `TIMEOUT` | Wall clock exhausted | Retry with an escalated time limit. |
| `NODE_FAIL` | Hardware died under the job | Retry unchanged. |
| `CANCELLED` | A human or policy killed it | Do not retry: the cancellation was a decision. |
| `VANISHED` | No record anywhere | See below. |

`VANISHED` is the case naive drivers get wrong. Batch accounting databases expire records, and a
job can be absent from both the accounting query and the live queue while having finished
perfectly well. Classifying "no record" as failure triggers a pointless (and possibly expensive)
resubmission; classifying it as success invents data. The kit demands terminal evidence: a job
absent from both sources is `COMPLETED` only if its expected output file exists, and `VANISHED`
otherwise — a state that is eligible for resubmission precisely because nothing is known about it.

Retries are governed by an explicit `RetryPolicy` with a per-job attempt cap, so a
deterministically failing input cannot consume the queue forever.

## Why two-stage selection

Ranking a pool purely by committee disagreement has a known pathology: the top of the ranking is
redundant. If the committee is uncertain about a region, it is uncertain about every candidate in
that region, and a greedy top-k selection buys the same information k times.

The kit therefore selects in two stages. First, query-by-committee keeps a band of the most
disputed candidates (a small multiple of the batch size) — this is the informativeness filter.
Then farthest-point sampling in descriptor space reduces the band to the batch — this is the
diversity filter, and it operates only among candidates already known to be informative. The order
matters: diversifying first would spend the batch on well-understood regions; ranking alone would
spend it on one region. The band factor is the single knob trading the two off.
