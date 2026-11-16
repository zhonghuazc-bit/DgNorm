"""Checkpoint plumbing for the paid stages of the DGNorm pipeline.

Every LLM call in DGNorm is persisted as a jsonl line keyed by obs_id, so
an interrupted run never re-pays and a rerun over a full cache is free.
The line schemas are inherited from the original research scripts and must
not change: existing RAG-HPO artifacts keep being valid caches.

Schemas:
  extraction: {"obs_id": "<id>", "phrases": [...], "error": ""}
  selection:  {"obs_id": "<id>", "map": {"<phrase>": "HP:..."}}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in
            p.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_keyed(path: str, key: str = "obs_id") -> Dict[str, Dict[str, Any]]:
    return {r[key]: r for r in read_jsonl(path)}


def load_phrases_map(path: str) -> Dict[str, List[str]]:
    return {r["obs_id"]: r.get("phrases", []) for r in read_jsonl(path)}


def load_selection_map(path: str) -> Dict[str, Dict[str, str]]:
    return {r["obs_id"]: dict(r.get("map", {})) for r in read_jsonl(path)}


class JsonlAppender:
    """Append-only writer; open once, flush each line, safe to resume."""

    def __init__(self, path: str, resume: bool = True) -> None:
        self.path = path
        self.done: set = set()
        if resume and Path(path).exists():
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.done.add(json.loads(line)[key_of(line)])
        self._fh = open(path, "a" if resume else "w", encoding="utf-8")

    def write(self, rec: Dict[str, Any]) -> None:
        self.done.add(rec["obs_id"])
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def key_of(line: str) -> str:
    return "obs_id"