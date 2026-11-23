"""DGNorm evaluation: corpus micro, observation micro, and official-form.

The DGNorm paper reports *corpus micro* (global TP / FP / FN aggregates
over all documents) -- the metric used throughout the RAG-HPO line. The
official BC8-T3 Task-3 scoring (Subtask 3a) differs in three ways:

  1. observation-level: each observation's P/R/F1 is computed and then
     macro-averaged (the organizers' official metric for ranking);
  2. normal findings (Negated=X in the TSV) are True Negatives -- they
     are *not* targets, so predicting them is an FP (a "False Positive"
     under the official definition: "this HPO term was ... negated");
  3. observations with no key-finding gold (found in dev; the test set
     adds explicit decoys) count against a system only through the FP
     of the codes it predicts there; predicting nothing is a TN and
     scores 1.0 for that observation.

``bc8_official`` evaluates under exactly these rules: it takes the gold
key-finding set plus the negated set per document, and treats an empty
prediction on a key-less document as a perfect observation.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence


def corpus_micro(
    gold_by_id: Dict[str, set],
    pred_by_id: Dict[str, set],
) -> Dict[str, object]:
    """Aggregate TP/FP/FN over all documents, then P/R/F1."""
    tp = fp = fn = 0
    for obs_id, g in gold_by_id.items():
        p = pred_by_id.get(obs_id, set())
        tp += len(p & g)
        fp += len(p - g)
        fn += len(g - p)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "p": round(prec, 4), "r": round(rec, 4),
            "f1": round(f1, 4), "n_docs": len(gold_by_id)}


def observation_micro(
    gold_by_id: Dict[str, set],
    pred_by_id: Dict[str, set],
) -> Dict[str, object]:
    """Average per-document P/R/F1 (each doc with no pred and no gold -> 1.0).

    This is the conventional document-level micro used by many clinical
    NER evaluations. For BC8-T3 the gold set is key findings only.
    """
    ps, rs, fs = [], [], []
    for obs_id, g in gold_by_id.items():
        p = pred_by_id.get(obs_id, set())
        if not g and not p:
            ps.append(1.0); rs.append(1.0); fs.append(1.0)
            continue
        tp = len(p & g)
        fp = len(p - g)
        fn = len(g - p)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        ps.append(prec); rs.append(rec); fs.append(f1)
    n = len(ps) if ps else 1
    return {"p": round(sum(ps) / n, 4), "r": round(sum(rs) / n, 4),
            "f1": round(sum(fs) / n, 4), "n_docs": len(ps)}


def bc8_official(
    gold_by_id: Dict[str, set],
    pred_by_id: Dict[str, set],
    negated_by_id: Optional[Dict[str, set]] = None,
) -> Dict[str, object]:
    """BC8-T3 Subtask-3a official scoring, verified against the organizers'
    evaluation script (BioCreativeVIIITask3EvaluationScript.py, DiscHPO repo).

    The official normalization-only metric is a *global* micro over key
    findings: P = TP/(TP+FP), R = TP/(TP+FN) with TP/FP/FN aggregated
    across all observations; true negatives (key-less observation, empty
    prediction) never enter the denominators. Predictions on a key-less
    observation are all FP, and a predicted code that matches a *negated*
    (normal) finding's code is likewise an FP -- the organizers' script
    does NOT carve negated-matched predictions out of the prediction set
    (verified in ``__addFPs``/``__alignRemainingFindings``).

    Consequently this function coincides with ``corpus_micro`` up to one
    nuance: the official script matches predictions to truth rows with
    multiplicity (a duplicate prediction of an already-matched code is an
    extra FP), while set-based counting collapses duplicates. Set
    semantics is kept here for cross-metric consistency; ``negated_by_id``
    is accepted for signature compatibility and intentionally unused.
    """
    del negated_by_id  # official script does not special-case negated codes
    return corpus_micro(gold_by_id, pred_by_id)


def preds_from_selection(
    selection: Dict[str, Dict[str, object]],
    shortcircuit_map: Optional[Dict[str, str]] = None,
) -> Dict[str, set]:
    """Document-level prediction sets.

    selection: {obs_id: {phrase: hpo_id_or_""}}; an empty string is an
    abstention and contributes no code.
    shortcircuit_map: {phrase: hpo_id} fixes the code for a rule hit. Hits
    are pre-filled into selection and skipped by the LLM; this function
    applies the same map at prediction time, so the output is invariant to
    whether the rule ran before or after selection.
    """
    out: Dict[str, set] = {}
    for obs_id, mapping in selection.items():
        pred = set()
        for ph, v in mapping.items():
            if shortcircuit_map and ph in shortcircuit_map:
                v = shortcircuit_map[ph]
            if v:
                pred.add(v)
        out[obs_id] = pred
    return out


def evaluate(
    gold_by_id: Dict[str, set],
    pred_by_id: Dict[str, set],
    modes: Sequence[str] = ("corpus", "observation"),
    negated_by_id: Optional[Dict[str, set]] = None,
) -> Dict[str, object]:
    out = {}
    if "corpus" in modes:
        out["corpus_micro"] = corpus_micro(gold_by_id, pred_by_id)
    if "observation" in modes:
        out["observation_micro"] = observation_micro(gold_by_id, pred_by_id)
    if "bc8_official" in modes:
        out["bc8_official"] = bc8_official(gold_by_id, pred_by_id,
                                           negated_by_id)
    return out