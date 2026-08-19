# Scope

What this repository is, and — just as deliberately — what it is not.

## What this is

campaign-kit is the orchestration layer extracted from active-learning research campaigns on
expensive atomistic reference calculations. It contains the parts of that work that are generic
engineering: the campaign driver with checkpoint/resume, the committee and selection machinery,
the training-domain fence, the batch-scheduler adapters with their failure taxonomy, and the
path-PCA coordinate reduction with its on-disk basis format. Everything runs against abstract
interfaces (`Labeler`, `Model`, `Selector`, `Proposer`), so the whole loop is exercisable on a
laptop with analytic stand-ins.

## Where module B stops

The coordinate-reduction module (`campaign_kit.coords`) deliberately stops at emitting a
coordinate-basis file: a versioned JSON document (plus an optional binary sidecar) holding the
reduced basis, the reference geometry, the atom ordering, and the metadata needed to project and
reconstruct.

What consumes that file — geometry optimizers, line searches, constrained-path refinement,
workflow engines — is externally authored software. Those tools are intentionally not included
here and not re-implemented here. The basis file is the interface: it is self-describing enough
that a downstream tool needs no import of this package to use it.

## What is deliberately absent

- **No reference data.** No computed energies, no labeled datasets.
- **No trained weights.** No model checkpoints of any kind, for any backend.
- **No geometries.** No structures of any studied system.
- **No results.** No numbers from any campaign or manuscript; every default in this codebase is a
  generic, documented parameter, not a tuned value.

The demos and tests generate their own synthetic data from closed-form functions at runtime.

## Authorship boundary

This repository contains only code written for release here. Anything whose authorship or
provenance was uncertain — third-party snippets, collaborator contributions without an explicit
release, configuration tied to specific institutional infrastructure — was left out rather than
included provisionally. Scheduler examples use placeholder identifiers (for example
`PARTITION_NAME`, `ACCOUNT_NAME`) that must be replaced with site-specific values.
