"""DGNorm SDK: the dual-granularity phenotype normalization pipeline.

Public verification release. Only the DGNorm SDK and the supporting modules
it imports are included; the full experimental harness, corpora, model
weights and dense indexes are obtained separately.
"""
from .dgnorm import (
    DgnormConfig,
    DgnormPipeline,
    DatasetAdapter,
    RagHpoAdapter,
    Bc8T3Adapter,
    GscAdapter,
    ExactAnchorTier,
    NormDocument,
    DgnormRunResult,
)

__all__ = [
    "DgnormPipeline", "DgnormConfig", "DatasetAdapter",
    "RagHpoAdapter", "Bc8T3Adapter", "GscAdapter", "ExactAnchorTier",
    "NormDocument", "DgnormRunResult",
]

__version__ = "0.1.0"
