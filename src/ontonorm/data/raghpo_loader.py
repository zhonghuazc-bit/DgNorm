"""Loader for the RAG-HPO benchmark sets (Genome Medicine 2025).

The benchmark ships as a spreadsheet rather than a flat file, with three
complications that have to be handled before it can stand in for a BC8-T3
style split:

* the ``CSC`` annotation sheet has a handful of cells holding two
  comma-joined HPO ids, so the term column is not always one id;
* case identifiers are stored as text with stray whitespace, so they must be
  normalised before the sheets can be joined;
* the ``GSC`` sheet carries a ``Category`` column whose only observed value is
  ``Abnormal``, so no annotation is flagged as negated and no separate
  negative set exists. Descriptions such as "absent fibula" denote the
  abnormality itself (HP:0002990), not the negation of a finding, so they
  belong in the gold set exactly as they are.

Observations are returned as :class:`Bc8Observation` so the rest of the
pipeline is corpus-agnostic: ``gold_terms`` is the positive key-finding set,
``negated_terms`` stays empty because this corpus records no negated findings.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import openpyxl

from .loaders import Bc8Observation

XLSX_DEFAULT = "data/raghpo/RAG-HPO_tests.xlsx"

_SHEETS = {
    # corpus -> (input sheet, term column, annotation sheet, term column)
    "csc": ("CSC Input", 1, "CSC Manual Annotations", 2),
    "gsc": ("GSC Input", 2, "GSC Manual Annotations ", 3),
}


def _norm_id(value) -> str:
    """Case ids appear as text with stray spaces; compare them normalised."""
    return re.sub(r"\s+", "", str(value or "")).upper()


def _split_terms(raw) -> List[str]:
    """Split an annotation cell into individual HPO ids."""
    out: List[str] = []
    for part in str(raw or "").split(","):
        part = part.strip().upper()
        if re.fullmatch(r"HP:\d{7}", part):
            out.append(part)
    return out


def load_raghpo(
    xlsx_path: str,
    corpus: str = "csc",
    min_terms: int = 1,
) -> List[Bc8Observation]:
    """Load one RAG-HPO corpus into observations.

    Parameters
    ----------
    xlsx_path:
        Path to ``RAG-HPO_tests.xlsx``.
    corpus:
        ``"csc"`` for the case-study cohort (free-text case reports) or
        ``"gsc"`` for the gold-standard corpus (disease entries).
    min_terms:
        Drop observations whose gold set is smaller than this. ``0`` keeps
        the unannotated cases that carry input text but no gold row, which is
        how the published recall denominators are built.
    """
    if corpus not in _SHEETS:
        raise ValueError(f"corpus must be one of {sorted(_SHEETS)}")
    in_sheet, in_col, ann_sheet, term_col = _SHEETS[corpus]

    wb = openpyxl.load_workbook(xlsx_path, data_only=True)

    notes: Dict[str, str] = {}
    order: List[str] = []
    for row in wb[in_sheet].iter_rows(min_row=2, values_only=True):
        if row[0] is None:
            continue
        cid = _norm_id(row[0])
        text = row[in_col]
        if not text or cid in notes:
            continue
        notes[cid] = str(text)
        order.append(cid)

    gold: Dict[str, List[str]] = {}
    for row in wb[ann_sheet].iter_rows(min_row=2, values_only=True):
        if row[0] is None:
            continue
        cid = _norm_id(row[0])
        for term in _split_terms(row[term_col]):
            gold.setdefault(cid, []).append(term)

    obs: List[Bc8Observation] = []
    for cid in order:
        terms = gold.get(cid, [])
        if len(terms) < min_terms:
            continue
        obs.append(Bc8Observation(
            obs_id=f"{corpus.upper()}-{cid}",
            text=notes[cid],
            # dedupe while preserving annotation order
            gold_terms=list(dict.fromkeys(terms)),
        ))
    return obs


def load_raghpo_all(xlsx_path: str = XLSX_DEFAULT) -> Dict[str, List[Bc8Observation]]:
    return {c: load_raghpo(xlsx_path, c) for c in _SHEETS}


if __name__ == "__main__":
    import statistics

    for name, corpus in load_raghpo_all().items():
        sizes = [len(o.gold_terms) for o in corpus]
        lengths = [len(o.text) for o in corpus]
        uniq = {t for o in corpus for t in o.gold_terms}
        print(f"{name}: cases={len(corpus)} gold_pairs={sum(sizes)} "
              f"unique_hpo={len(uniq)}")
        print(f"  gold per case: min={min(sizes)} max={max(sizes)} "
              f"mean={statistics.mean(sizes):.2f}")
        print(f"  note chars  : min={min(lengths)} max={max(lengths)} "
              f"mean={statistics.mean(lengths):.0f}")