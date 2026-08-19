"""Write and read a fitted path-PCA basis as a small, self-describing file.

This module is the end of the coordinate-reduction pipeline inside
``campaign_kit``: it *emits* the basis and stops. The downstream consumer of a
basis file — a path optimizer, a workflow engine, an interpolation service —
is external to this package and deliberately out of scope; nothing here
imports a basis for any purpose beyond the verbatim round trip.

File format specification (``format_version`` = 1)
--------------------------------------------------
A basis is one JSON document, optionally accompanied by one ``.npz`` sidecar.
``write_basis("some/dir/basis", ...)`` produces ``some/dir/basis.json`` and,
for large bases, ``some/dir/basis.npz``.

Top-level JSON keys:

- ``format_version`` (int): currently 1. Readers must reject versions they
  do not know rather than guess.
- ``library_version`` (str): ``campaign_kit.__version__`` at write time, for
  provenance only — it does not affect parsing.
- ``n_atoms`` (int): number of atoms M.
- ``atom_order`` (list[str]): element symbol per row of the reference
  geometry. The basis is only meaningful for geometries whose atoms are
  supplied in exactly this order.
- ``mass_weighted`` (bool): the metric the basis was built in.
- ``masses`` (list[float] | null): per-atom masses (amu); present whenever
  ``mass_weighted`` is true, because reconstruction is impossible without
  them.
- ``reference`` (nested list, (M, 3)): the expansion-origin geometry.
  Amplitude zero reconstructs exactly this structure.
- ``explained_variance_ratio`` (list[float]): one entry per component.
- ``components``: one of
    * a nested list of shape (k, 3M) — inline form, used when the array has
      at most ``INLINE_COMPONENT_LIMIT`` elements so the JSON stays small
      enough to diff and grep;
    * an object ``{"npz_file": "<basename>.npz", "key": "components"}`` —
      sidecar form for large bases. The sidecar is referenced by *basename*
      and must sit next to the JSON, so the pair can be moved or renamed
      together.
- ``meta`` (object): caller-supplied metadata, stored verbatim. Must be
  JSON-serializable; the library never interprets it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from campaign_kit import __version__
from campaign_kit.coords.pca import PathPCA

__all__ = ["FORMAT_VERSION", "INLINE_COMPONENT_LIMIT", "LoadedBasis", "read_basis", "write_basis"]

FORMAT_VERSION: int = 1

# Inline-vs-sidecar cutoff, in array elements (floats). 100k elements is a
# few MB of JSON text — the point where a text file stops being pleasant to
# diff or grep and binary storage starts paying for itself. It is a packaging
# knob, not a correctness threshold: both forms round-trip identically.
INLINE_COMPONENT_LIMIT: int = 100_000


@dataclass(frozen=True)
class LoadedBasis:
    """Everything ``read_basis`` recovers from one basis file."""

    pca: PathPCA
    atom_order: list[str]
    meta: dict[str, object]
    format_version: int
    library_version: str


def write_basis(
    path_prefix: str | Path,
    pca: PathPCA,
    atom_order: list[str],
    meta: dict[str, object],
) -> Path:
    """Serialize a fitted basis to ``<path_prefix>.json`` (+ optional ``.npz``).

    Returns the path of the JSON file written. The atom ordering is stored
    explicitly because the flattened component vectors are meaningless without
    it — a reader has no other way to know which coordinates belong to which
    atom.
    """
    if pca.components_ is None or pca.reference_ is None:
        raise ValueError("cannot write an unfitted PathPCA")
    assert pca.explained_variance_ratio_ is not None
    assert pca.n_atoms_ is not None
    if len(atom_order) != pca.n_atoms_:
        raise ValueError(f"atom_order has {len(atom_order)} entries for {pca.n_atoms_} atoms")

    prefix = Path(path_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_name(prefix.name + ".json")

    components: object
    if pca.components_.size <= INLINE_COMPONENT_LIMIT:
        components = pca.components_.tolist()
    else:
        npz_path = prefix.with_name(prefix.name + ".npz")
        np.savez_compressed(npz_path, components=pca.components_)
        components = {"npz_file": npz_path.name, "key": "components"}

    document = {
        "format_version": FORMAT_VERSION,
        "library_version": __version__,
        "n_atoms": pca.n_atoms_,
        "atom_order": list(atom_order),
        "mass_weighted": pca.mass_weighted_,
        "masses": None if pca.masses_ is None else pca.masses_.tolist(),
        "reference": pca.reference_.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "components": components,
        "meta": meta,
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return json_path


def read_basis(path: str | Path) -> LoadedBasis:
    """Load a basis written by ``write_basis``; accepts the JSON path or the prefix.

    The returned ``PathPCA`` is fully functional (``project``, ``reconstruct``,
    ``select_n_components``) without re-fitting — that is the round-trip
    guarantee this pair of functions exists for.
    """
    json_path = Path(path)
    if json_path.suffix != ".json":
        json_path = json_path.with_name(json_path.name + ".json")
    with json_path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)

    version = document.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"unsupported basis format_version {version!r}; this reader knows {FORMAT_VERSION}"
        )

    raw_components = document["components"]
    if isinstance(raw_components, dict):
        # Sidecar form: resolved relative to the JSON so the pair is relocatable.
        npz_path = json_path.parent / raw_components["npz_file"]
        with np.load(npz_path) as archive:
            components = np.asarray(archive[raw_components["key"]], dtype=float)
    else:
        components = np.asarray(raw_components, dtype=float)

    masses = document["masses"]
    pca = PathPCA.from_state(
        components=components,
        explained_variance_ratio=np.asarray(document["explained_variance_ratio"], dtype=float),
        reference=np.asarray(document["reference"], dtype=float),
        mass_weighted=bool(document["mass_weighted"]),
        masses=None if masses is None else np.asarray(masses, dtype=float),
    )
    return LoadedBasis(
        pca=pca,
        atom_order=list(document["atom_order"]),
        meta=dict(document["meta"]),
        format_version=int(version),
        library_version=str(document["library_version"]),
    )
