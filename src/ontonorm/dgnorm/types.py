"""Unified document type flowing through the DGNorm pipeline.

The five stages of DGNorm know nothing about the raw dataset behind a
document. Every dataset adapter normalizes its input into NormDocument,
so the pipeline itself has exactly one code path for RAG-HPO, BC8-T3 and
any future corpus.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class NormDocument:
    """One normalized document after adapter conversion.

    obs_id:
        Stable corpus-local identifier (RAG-HPO: ``CSC-<n>``; BC8-T3:
        the StringID column). Used as the cache key and the eval unit.
    text:
        Raw clinical text, exactly as the corpus provides it.
    gold_terms:
        Positive key-finding HPO ids (empty for observations with no gold).
    negated_terms:
        HPO ids that are normal/negated findings; they are excluded from
        the gold set but kept separate so the adapter contract is explicit.
    """

    obs_id: str
    text: str
    gold_terms: List[str] = field(default_factory=list)
    negated_terms: List[str] = field(default_factory=list)


@dataclass
class DgnormRunResult:
    """Everything one pipeline run produced, cache-aware and resumable."""

    dataset: str
    docs: List[NormDocument]
    phrases: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    selection: Dict[str, Dict[str, str]] = field(default_factory=dict)
    selection_stats: Dict[str, object] = field(default_factory=dict)
    shortcircuit_map: Dict[str, str] = field(default_factory=dict)
    shortcircuit_stats: Dict[str, object] = field(default_factory=dict)
    final_preds: Dict[str, List[str]] = field(default_factory=dict)
    metrics_gold: Optional[Dict[str, object]] = None
    metrics_full: Optional[Dict[str, object]] = None