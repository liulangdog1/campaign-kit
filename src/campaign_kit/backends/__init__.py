"""Model backends implementing the `Model` protocol.

Re-exports are lazy (module-level ``__getattr__``) so that importing
``campaign_kit.backends`` never pays for sklearn, and never touches the
optional MACE dependency unless ``MACEModel`` is actually accessed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from campaign_kit.backends.mace import MACEModel
    from campaign_kit.backends.sklearn_mlp import SklearnMLPModel, inverse_distance_descriptor

__all__ = ["MACEModel", "SklearnMLPModel", "inverse_distance_descriptor"]


def __getattr__(name: str) -> object:
    if name in ("SklearnMLPModel", "inverse_distance_descriptor"):
        from campaign_kit.backends import sklearn_mlp

        return getattr(sklearn_mlp, name)
    if name == "MACEModel":
        from campaign_kit.backends import mace

        return mace.MACEModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
