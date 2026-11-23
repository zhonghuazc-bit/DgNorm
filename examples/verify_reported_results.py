#!/usr/bin/env python3
"""Verify the published DGNorm CSC result without models, network or keys.

Pure standard library. Two protocols are checked against the same 112 gold
sets and final prediction sets:

* main result (paper Tables 3-5): normalized (collapsed) strict matching with
  per-document macro averaging, which yields F1 0.7490;
* release cross-check: strict exact matching at corpus-level micro, which
  yields F1 0.7441.

The collapse uses a minimal HPO is_a descendant table shipped in
``verification/``; no ontology file, model weight, index or network is needed.
The script loads the SDK's evaluate.py directly by file path, so it also works
before ``pip install -e .``.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VER = ROOT / "verification"
EVAL = ROOT / "src" / "ontonorm" / "dgnorm" / "evaluate.py"


def load_evaluate():
    spec = importlib.util.spec_from_file_location("dgnorm_evaluate", EVAL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def collapse(code_set, desc):
    """Drop a code if a more specific code in the same set is its descendant."""
    codes = list(code_set)
    return {c for c in codes
            if not any(c != d and d in desc.get(c, ()) for d in codes)}


def collapsed_macro(gold, pred, desc):
    g = {k: collapse(set(v), desc) for k, v in gold.items()}
    p = {k: collapse(set(v), desc) for k, v in pred.items()}
    sp = sr = sf = 0.0
    n = 0
    for oid, gs in g.items():
        ps = p.get(oid, set())
        if not gs and not ps:
            continue
        tp = len(ps & gs)
        fp = len(ps) - tp
        fn = len(gs) - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        sp += precision
        sr += recall
        sf += f1
        n += 1
    return {"p": round(sp / n, 4), "r": round(sr / n, 4),
            "f1": round(sf / n, 4), "n_docs": n}


def main() -> None:
    ev = load_evaluate()

    selection = {row["obs_id"]: dict(row["map"])
                 for row in read_jsonl(VER / "selection.jsonl")}
    short = read_json(VER / "shortcircuit_map.json")
    gold = {k: set(v) for k, v in read_json(VER / "gold_csc_strict.json").items()}
    published = {k: set(v) for k, v in read_json(
        VER / "predictions_csc_strict.json").items()}
    desc = {k: set(v) for k, v in read_json(
        VER / "hpo_descendants_scope.json").items()}
    expected_main = read_json(VER / "expected_metrics_collapsed.json")
    expected_micro = read_json(VER / "expected_metrics.json")

    assert len(selection) == len(gold) == len(published) == 112
    assert selection.keys() == gold.keys() == published.keys()

    preds = ev.preds_from_selection(selection, short)
    assert preds == published, "final predictions differ from the released set"

    # --- main result: collapsed strict per-document macro (paper Tables 3-5) ---
    main_metrics = collapsed_macro(gold, published, desc)
    for key in ("p", "r", "f1", "n_docs"):
        assert main_metrics[key] == expected_main[key], (
            "main", key, main_metrics[key], expected_main[key])

    # --- release cross-check: strict corpus-level micro ---
    micro = ev.corpus_micro(gold, preds)
    locked = expected_micro["main"]
    for key in ("tp", "fp", "fn", "p", "r", "f1", "n_docs"):
        assert micro[key] == locked[key], ("micro", key, micro[key], locked[key])

    sens = expected_micro["endpoint_sensitivity"]
    assert sens["main"]["f1"] == 0.7441
    assert sens["glm"]["f1"] == 0.7308
    assert sens["qwen"]["f1"] == 0.7117
    for name in sens:
        m = sens[name]
        assert abs(2 * m["p"] * m["r"] / (m["p"] + m["r"]) - m["f1"]) < 0.00015, name
        assert m["n_docs"] == 112

    print("MAIN RESULT (paper Tables 3-5): normalized collapsed strict, per-document macro")
    print("  CSC 112 docs: P %.4f  R %.4f  F1 %.4f" %
          (main_metrics["p"], main_metrics["r"], main_metrics["f1"]))
    print("RELEASE CROSS-CHECK: strict exact match, corpus-level micro")
    print("  TP %d / FP %d / FN %d, P %.4f  R %.4f  F1 %.4f" %
          (micro["tp"], micro["fp"], micro["fn"],
           micro["p"], micro["r"], micro["f1"]))
    print("Endpoint sensitivity (micro): DeepSeek 0.7441, GLM 0.7308, Qwen 0.7117")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
