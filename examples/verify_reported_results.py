#!/usr/bin/env python3
"""Verify the published DGNorm CSC result without models, network or keys.

Pure standard library. Loads the SDK's evaluate.py directly by file path,
so this script also works before `pip install -e .`.
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


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> None:
    ev = load_evaluate()

    selection = {row["obs_id"]: dict(row["map"])
                 for row in read_jsonl(VER / "selection.jsonl")}
    short = json.loads((VER / "shortcircuit_map.json").read_text(encoding="utf-8"))
    gold = {k: set(v) for k, v in json.loads(
        (VER / "gold_csc_strict.json").read_text(encoding="utf-8")).items()}
    published = {k: set(v) for k, v in json.loads(
        (VER / "predictions_csc_strict.json").read_text(encoding="utf-8")).items()}
    expected = json.loads((VER / "expected_metrics.json").read_text(encoding="utf-8"))

    assert len(selection) == len(gold) == len(published) == 112
    assert selection.keys() == gold.keys() == published.keys()

    preds = ev.preds_from_selection(selection, short)
    assert preds == published, "final predictions differ from the released set"

    metrics = ev.corpus_micro(gold, preds)
    locked = expected["main"]
    for key in ("tp", "fp", "fn", "p", "r", "f1", "n_docs"):
        assert metrics[key] == locked[key], (key, metrics[key], locked[key])

    sens = expected["endpoint_sensitivity"]
    assert sens["main"]["f1"] == 0.7441
    assert sens["glm"]["f1"] == 0.7308
    assert sens["qwen"]["f1"] == 0.7117
    for name in sens:
        m = sens[name]
        assert abs(2 * m["p"] * m["r"] / (m["p"] + m["r"]) - m["f1"]) < 0.00015, name
        assert m["n_docs"] == 112

    print(f"CSC strict corpus micro: TP {metrics['tp']} / "
          f"FP {metrics['fp']} / FN {metrics['fn']}")
    print(f"P {metrics['p']:.4f}  R {metrics['r']:.4f}  F1 {metrics['f1']:.4f}")
    print("Endpoint sensitivity: DeepSeek 0.7441, GLM 0.7308, Qwen 0.7117")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
