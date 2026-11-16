"""Dataset adapters: raw corpus files -> NormDocument.

DGNorm's pipeline stages are corpus-agnostic; the only dataset-specific
code lives here. Adapters normalize the *raw* benchmark files into the
unified NormDocument shape (obs_id / text / gold_terms / negated_terms),
so a single pipeline run handles RAG-HPO's xlsx and BC8-T3's TSV (or any
future corpus) without any change to the main flow.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from .types import NormDocument


class DatasetAdapter(ABC):
    """Minimal contract: a corpus -> ordered list of documents."""

    name: str = "abstract"

    @abstractmethod
    def load(self) -> List[NormDocument]:
        ...


# ---------------------------------------------------------------------------
# RAG-HPO (Genome Medicine 2025) -- xlsx workbook
# ---------------------------------------------------------------------------

class RagHpoAdapter(DatasetAdapter):
    """RAG-HPO corpus from the official xlsx workbook.

    Reuses the existing RAG-HPO loader so the parse logic (case-id
    normalisation, multi-id cells, GSC/CSC sheets, gold extraction) stays
    in one place and remains exactly as before.
    """

    name = "raghpo"

    def __init__(self, xlsx_path: str, corpus: str = "csc",
                 min_terms: int = 0) -> None:
        self.xlsx_path = xlsx_path
        self.corpus = corpus
        self.min_terms = min_terms

    def load(self) -> List[NormDocument]:
        from ..data.raghpo_loader import load_raghpo
        obs = load_raghpo(self.xlsx_path, self.corpus, min_terms=self.min_terms)
        # RAG-HPO loader already strips negated findings (the corpus records
        # none); gold_terms are the positive key-finding set.
        return [NormDocument(obs_id=o.obs_id, text=o.text,
                             gold_terms=list(o.gold_terms))
                for o in obs]


# ---------------------------------------------------------------------------
# BC8-T3 (BioCreative VIII Track 3) -- TSV files
# ---------------------------------------------------------------------------

class Bc8T3Adapter(DatasetAdapter):
    """BC8-T3 dev/train TSV, keeping key findings in gold and normal
    findings (Negated==X) out of the gold set.

    The TSV can hold several rows per observation (one per annotated
    finding); the loader groups them. Reading only key findings for gold
    follows the official Subtask-3a scoring: a normal finding is not a
    target concept; predicting it is a false positive.
    """

    name = "bc8t3"

    def __init__(self, tsv_path: str) -> None:
        self.tsv_path = tsv_path

    def load(self) -> List[NormDocument]:
        from ..data.loaders import load_bc8_tsv
        obs = load_bc8_tsv(self.tsv_path)
        return [NormDocument(obs_id=o.obs_id, text=o.text,
                             gold_terms=list(o.gold_terms),
                             negated_terms=list(o.negated_terms))
                for o in obs]


# ---------------------------------------------------------------------------
# GSC+ (PhenoTagger corpus.zip, corpus/GSC) -- parsed JSON documents
# ---------------------------------------------------------------------------

class GscAdapter(DatasetAdapter):
    """GSC+ corpus parsed from PhenoTagger's corpus.zip into
    data/gsc/gsc_{test,dev}.json (obs_id=PMID, text=title+abstract,
    gold_terms=unique HPO codes). Span/mention granularity is collapsed to
    the code set, matching our corpus-level micro convention.
    """

    name = "gsc"

    def __init__(self, json_path: str) -> None:
        self.json_path = json_path

    def load(self) -> List[NormDocument]:
        import json
        docs = json.load(open(self.json_path, encoding="utf-8"))
        return [NormDocument(obs_id=d["obs_id"], text=d["text"],
                             gold_terms=list(d.get("gold_terms", [])))
                for d in docs]