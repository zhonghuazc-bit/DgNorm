"""Stage 2 of DGNorm: candidate delimitation (SapBERT dense retrieval).

One embedding per phrasing, matched against the HPO surface-form index;
each phrase gets its own top-K block. This stage compresses the search
space from the full ontology to <=K candidates per phrase, zero LLM cost.
The index (phrase -> hpo-id-set matrix) is built once and reused; the
index is dataset-agnostic, so BC8-T3 and RAG-HPO share it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

K_DEFAULT = 20


def _load_index(index_npz: str) -> tuple:
    data = np.load(index_npz, allow_pickle=True)
    return (list(data["phrases"]),
            [set(x) for x in data["phrase_ids"]],
            np.asarray(data["matrix"], dtype=np.float32))


class CandidateIndex:
    """HPO surface-form index loaded from a prebuilt npz."""

    def __init__(self, index_npz: str, model_dir: str,
                 dtype: str = "float32") -> None:
        self.phrases, self.phrase_ids, self.matrix = _load_index(index_npz)
        self.model = SentenceTransformer(model_dir)
        self._phrase_index = {p: i for i, p in enumerate(self.phrases)}

    def embed(self, queries: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(list(queries), batch_size=128,
                              normalize_embeddings=True,
                              show_progress_bar=False, convert_to_numpy=True),
            dtype=np.float32)

    def topk_for_query(self, vec: np.ndarray, k: int = K_DEFAULT) -> List[tuple]:
        """Return [(phrase, id_set, score)] for the top-k phrases of one vector."""
        sims = self.matrix @ vec  # (n_phrases,)
        idx = np.argpartition(-sims, min(k, len(sims)))[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(self.phrases[i], self.phrase_ids[i], float(sims[i]))
                for i in idx]

    def topk_terms_for_query(self, vec: np.ndarray,
                             k: int = K_DEFAULT) -> List[Tuple[str, str, float]]:
        """Top-k *terms* for one query vector, mirroring the validated mm8
        candidate block: every phrase in the index votes for the hpo ids it
        carries, each term keeps its best phrase score, then the terms are
        ranked by score and truncated to k.

        Returns [(hpid, best_phrase, score)] in score-descending order.
        """
        sims = self.matrix @ vec  # (n_phrases,)
        best: Dict[str, Tuple[float, str]] = {}
        for i in range(len(self.phrases)):
            s = float(sims[i])
            for hpid in self.phrase_ids[i]:
                cur = best.get(hpid)
                if cur is None or s > cur[0]:
                    best[hpid] = (s, self.phrases[i])
        ranked = [(h, ph, s) for h, (s, ph) in best.items()]
        ranked.sort(key=lambda x: (-x[2], x[0]))
        return ranked[:k]

    def topk_for_phrases(self, phrases: Sequence[str],
                         k: int = K_DEFAULT) -> Dict[str, List[tuple]]:
        vecs = self.embed(phrases)
        return {p: self.topk_for_query(vecs[i], k=k)
                for i, p in enumerate(phrases)}

    def topk_terms_for_phrases(self, phrases: Sequence[str],
                               k: int = K_DEFAULT) -> Dict[str, List[Tuple[str, str, float]]]:
        vecs = self.embed(phrases)
        return {p: self.topk_terms_for_query(vecs[i], k=k)
                for i, p in enumerate(phrases)}

    def top_k_unique_ids(self, phrases: Sequence[str],
                         k: int = K_DEFAULT) -> Dict[str, Set[str]]:
        """Top-k phrases -> unique hpo-id set, for reachability probes."""
        out = {}
        for p, hits in self.topk_for_phrases(phrases, k=k).items():
            ids: Set[str] = set()
            for _ph, id_set, _s in hits:
                ids |= set(id_set)
            out[p] = ids
        return out