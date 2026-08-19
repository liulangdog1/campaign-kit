"""Optional adapter for a pretrained MACE graph-network calculator.

MACE is deliberately NOT a dependency of this package: the demos must install
in seconds. The import is guarded at construction so the failure is a clear
install hint at the point of use, not a confusing ImportError at package
import time.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from campaign_kit.protocols import Dataset, Structure

__all__ = ["MACEModel"]

_INSTALL_HINT = (
    "MACEModel requires the optional 'mace-torch' package (which brings 'ase'). "
    "Install it with: pip install mace-torch"
)


class MACEModel:
    """Inference-only `Model` wrapper around a pretrained MACE calculator.

    Energy-only by design: training a graph-network potential is an external
    workflow with its own tooling, so ``fit`` raises rather than pretending.
    Use this adapter to plug an already-trained potential into the committee
    and selection machinery.
    """

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        try:
            from mace.calculators import MACECalculator
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(_INSTALL_HINT) from exc
        self._calc = MACECalculator(model_paths=model_path, device=device)

    def fit(self, dataset: Dataset, seed: int) -> None:
        raise NotImplementedError(
            "MACEModel is inference-only; train the potential with the MACE "
            "toolchain and pass the resulting model_path here"
        )

    def predict(self, structures: Sequence[Structure]) -> np.ndarray:
        return np.asarray([self._evaluate(s)[0] for s in structures], dtype=float)

    def predict_per_atom(self, structures: Sequence[Structure]) -> list[np.ndarray]:
        """Real per-atom energies when the calculator exposes them, else uniform."""
        return [self._evaluate(s)[1] for s in structures]

    def _evaluate(self, structure: Structure) -> tuple[float, np.ndarray]:
        from ase import Atoms

        atoms = Atoms(symbols=list(structure.species), positions=structure.positions)
        atoms.calc = self._calc
        energy = float(atoms.get_potential_energy())
        node_energy = self._calc.results.get("node_energy")
        if node_energy is not None:
            per_atom = np.asarray(node_energy, dtype=float)
        else:
            per_atom = np.full(structure.n_atoms, energy / structure.n_atoms)
        return energy, per_atom
