"""campaign-kit: active-learning orchestration and coordinate reduction."""

from campaign_kit.protocols import (
    Dataset,
    JobHandle,
    Labeler,
    LabelResult,
    LabelStatus,
    Model,
    Predictions,
    Proposer,
    Selector,
    Structure,
)

__version__ = "0.1.0"

__all__ = [
    "Dataset",
    "JobHandle",
    "Labeler",
    "LabelResult",
    "LabelStatus",
    "Model",
    "Predictions",
    "Proposer",
    "Selector",
    "Structure",
    "__version__",
]
