"""DGNorm: dual-granularity phenotype normalization.

A dataset-agnostic normalization pipeline of five independent stages,
designed to be composed through :class:`DgnormPipeline` and consumed
from any corpus through a :class:`DatasetAdapter`:

    DgnormPipeline(config).run(adapter)

Stages (each importable and independently reusable):

    dgnorm.extract       stage 1 -- dual extractors (flash + gen)
    dgnorm.retrieve      stage 2 -- SapBERT candidate delimitation
    dgnorm.shortcircuit  stage 3 -- in-block equality-family short-circuit
    dgnorm.select        stage 4 -- source-sentence constrained selection
    dgnorm.evaluate      stage 5 -- corpus/observation micro

The equality rule runs before selection: phrases whose code is fixed by the
string are settled with no LLM call, and only misses reach the selector.
"""
from .adapters import Bc8T3Adapter, DatasetAdapter, GscAdapter, RagHpoAdapter
from .pipeline import DgnormConfig, DgnormPipeline
from .types import DgnormRunResult, NormDocument

__all__ = [
    "DgnormPipeline",
    "DgnormConfig",
    "DatasetAdapter",
    "RagHpoAdapter",
    "Bc8T3Adapter",
    "GscAdapter",
    "NormDocument",
    "DgnormRunResult",
]