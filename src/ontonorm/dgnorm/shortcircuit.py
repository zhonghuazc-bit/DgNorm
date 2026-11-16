"""Stage 4 of DGNorm: in-block equality-family short-circuit.

Deterministic, zero-LLM. For each phrase, three equality rules are tried
in cascade over the phrase's own SapBERT top-K block, restricted to
phrases whose index entry maps to a single hpo id:

  1. sorted-token signature + token count equality
  2. token-set equality
  3. cleaned full-string equality

A hit overrides the LLM pick (or fills an abstention). Two safety design
constraints inherited from the ablation: the match domain is the
candidate block (never the whole library), and only the equality family
is used (no substring/fuzzy rules, which measured harmful).
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np  # noqa: E402

PAT = re.compile(r"[^a-z0-9 ]")


def clean_text(txt: str) -> str:
    return PAT.sub("", (txt or "").replace("_", " ").lower()).strip()


class ShortCircuitTier:
    """Equality-family short-circuit over each phrase's own top-K block."""

    def __init__(self, index_npz: str, model_dir: str, k: int = 20) -> None:
        data = np.load(index_npz, allow_pickle=True)
        self.matrix = np.asarray(data["matrix"], dtype=np.float32)
        self.lib = [p for p in data["phrases"]]
        self.lib_clean = [clean_text(p) for p in self.lib]
        self.lib_ids = [set(x) for x in data["phrase_ids"]]
        self.k = k
        from sentence_transformers import SentenceTransformer  # noqa: E402
        self.model = SentenceTransformer(model_dir)

    def candidates(self, phrase: str) -> List[Tuple[str, str]]:
        """(cleaned_lib_phrase, hpo_id) for single-id phrases in the top-K block."""
        vec = self.model.encode([phrase], normalize_embeddings=True)[0]
        s = self.matrix @ vec
        idx = np.argpartition(-s, min(self.k, len(s)))[:self.k]
        idx = idx[np.argsort(-s[idx])]
        return [(self.lib_clean[i], next(iter(self.lib_ids[i])))
                for i in idx if len(self.lib_ids[i]) == 1]

    def _match(self, phrase: str) -> Optional[str]:
        cp = clean_text(phrase)
        toks = cp.split()
        pset = set(toks)
        sig = " ".join(sorted(toks))
        cands = self.candidates(phrase)
        # rule 1: sorted-token signature + token count, all candidates first
        for ct, h in cands:
            ctoks = ct.split()
            if " ".join(sorted(ctoks)) == sig and len(ctoks) == len(toks):
                return h
        # rule 2: token-set equality
        for ct, h in cands:
            if set(ct.split()) == pset:
                return h
        # rule 3: cleaned full-string equality
        for ct, h in cands:
            if ct == cp:
                return h
        return None

    def run(
        self,
        phrases: List[str],
        selection: Dict[str, str],
    ) -> Tuple[Dict[str, Dict[str, str]], Dict[str, object]]:
        """Apply short-circuit over all phrases.

        Returns (per-doc mappings, stats). Each phrase keeps its LLM pick
        unless the short-circuit hits; a hit overrides (or fills abstention).
        """
        short: Dict[str, str] = {}
        for p in phrases:
            hit = self._match(p)
            if hit:
                short[p] = hit
        stats = {"shortcircuited": len(short)}
        return short, stats