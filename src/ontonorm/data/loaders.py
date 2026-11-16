"""Loaders for the public datasets used by OntoNorm-Dx.

- BC8-T3: BioCreative VIII Task 3 genotype-phenotype normalization corpus
  (TSV with columns StringID, ParticipantCount, Text, Term, Negated, Spans).
- Phenopacket Store: GA4GH phenopackets (JSON) with gold standard diseases.
- HPOA: disease-to-phenotype annotations used for Task B gold phenotype cues.
"""
from __future__ import annotations

import csv
import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple


# ---------------------------------------------------------------------------
# BC8-T3
# ---------------------------------------------------------------------------

@dataclass
class Bc8Observation:
    obs_id: str
    text: str
    gold_terms: List[str] = field(default_factory=list)  # HPO IDs, positive key findings only
    gold_spans: List[Tuple[int, int]] = field(default_factory=list)  # (start, end)
    negated_terms: List[str] = field(default_factory=list)  # HPO IDs annotated as negated/normal
    negated: bool = False


def load_bc8_tsv(path: str) -> List[Bc8Observation]:
    """Load a BC8-T3 tab-separated file into observations.

    Multiple rows may share one observation (same StringID) when several
    findings are annotated; we group them. ``Term`` may be a valid HPO id
    or "NA" for unannotated rows.

    Following the official Subtask-3a scoring, a *negated* finding (a
    normal finding, e.g. "Normal lips and tongue") is **not** a gold
    term: correctly ignoring it is a true negative and predicting it is a
    false positive. Negated terms are therefore kept in
    :attr:`Bc8Observation.negated_terms`, separate from ``gold_terms``.
    """
    obs_by_id: Dict[str, Bc8Observation] = {}
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            sid = row["StringID"].strip()
            text = row["Text"].strip()
            term = row["Term"].strip()
            neg = (row.get("Negated") or "").strip().lower() in {"x", "true", "1"}
            spans_raw = (row.get("Spans") or "").strip()
            spans = []
            if spans_raw and spans_raw not in {"NA", ""}:
                for sp in spans_raw.split(","):
                    sp = sp.strip()
                    if "-" in sp:
                        a, b = sp.split("-", 1)
                        spans.append((int(a), int(b)))
            if sid not in obs_by_id:
                obs_by_id[sid] = Bc8Observation(sid, text)
            rec = obs_by_id[sid]
            if term and term != "NA":
                if neg:
                    rec.negated_terms.append(term)
                    rec.negated = True
                else:
                    rec.gold_terms.append(term)
                    rec.gold_spans.extend(spans)
    return list(obs_by_id.values())


# ---------------------------------------------------------------------------
# Phenopacket Store
# ---------------------------------------------------------------------------

@dataclass
class PhenoCase:
    case_id: str
    phenotypes: List[Tuple[str, str]] = field(default_factory=list)  # (HPO id, label)
    excluded: List[Tuple[str, str]] = field(default_factory=list)
    diseases: List[Tuple[str, str]] = field(default_factory=list)  # (disease id, label)
    sex: str = ""
    age: Optional[Dict] = None

    @property
    def positive_phenotypes(self) -> List[str]:
        return [hp for hp, _ in self.phenotypes]


class PhenopacketStore:
    """Iterator-compatible access to a Phenopacket Store zip archive."""

    def __init__(self, zip_path: str, prefix: str = "") -> None:
        self.zip_path = zip_path
        self.prefix = prefix
        self._z = None

    def open(self) -> "PhenopacketStore":
        self._z = zipfile.ZipFile(self.zip_path)
        return self

    def close(self) -> None:
        if self._z:
            self._z.close()
            self._z = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    def case_names(self) -> List[str]:
        assert self._z is not None
        return [n for n in self._z.namelist()
                if n.endswith(".json") and n.startswith(self.prefix)]

    def iter_cases(self, limit: Optional[int] = None) -> Iterator[PhenoCase]:
        assert self._z is not None
        names = self.case_names()
        if limit:
            names = names[:limit]
        for n in names:
            data = json.loads(self._z.read(n))
            cid = data.get("id") or n
            pf = []
            ex = []
            for feat in data.get("phenotypicFeatures", []) or []:
                if not feat:
                    continue
                typ = feat.get("type") or {}
                hpid = typ.get("id", "")
                label = typ.get("label", "")
                if feat.get("excluded"):
                    ex.append((hpid, label))
                else:
                    pf.append((hpid, label))
            dis = []
            for d in data.get("diseases", []) or []:
                term = (d or {}).get("term") or {}
                dis.append((term.get("id", ""), term.get("label", "")))
            subj = data.get("subject") or {}
            yield PhenoCase(
                case_id=cid,
                phenotypes=pf,
                excluded=ex,
                diseases=dis,
                sex=subj.get("sex", ""),
                age=subj.get("age", None),
            )


# ---------------------------------------------------------------------------
# HPOA
# ---------------------------------------------------------------------------

@dataclass
class HpoaRecord:
    database_id: str  # e.g. OMIM:619340
    disease_name: str
    hpo_id: str
    reference: str
    evidence: str


def load_hpoa(path: str) -> Dict[str, List[HpoaRecord]]:
    """Load phenotype.hpoa into {disease_id -> [records]}."""
    out: Dict[str, List[HpoaRecord]] = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            rec = HpoaRecord(
                database_id=parts[0],
                disease_name=parts[1],
                hpo_id=parts[3],
                reference=parts[4],
                evidence=parts[5],
            )
            out.setdefault(rec.database_id, []).append(rec)
    return out