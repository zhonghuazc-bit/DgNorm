"""Stage 4 of DGNorm: source-sentence-injected constrained selection.

Phrases already settled by the equality rule are skipped. For every
remaining phrase, one LLM call (temperature 0) chooses exactly one HPO
code inside that phrase's candidate block, or abstains. The user prompt
injects the source sentence containing the phrase, the document-wide
phrase list, and the candidate terms with definitions. A hard guard nulls
any pick outside the phrase's own block.

The code is a class so the pipeline can drive it over any dataset's
documents with full resumable checkpointing.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..tiers.llm_tier import FatalApiError, LlmClient

_SYSTEM = """You are a clinical phenotype coding assistant.

You are given ONE phenotypic phrase extracted from a clinical document, the \
full list of phrases extracted from the same document, and a numbered list of \
candidate Human Phenotype Ontology (HPO) terms that a retriever proposed for \
the phrase.

Your job is to pick the single HPO term from the list that best matches the \
phrase.

Rules:
- Pick exactly one term from the list. Do not invent an ID not on the list.
- Use the document-wide phrase list as context: judge the granularity the \
case warrants. If another phrase in the list already covers a broader or \
related finding, prefer the term that keeps the case-level annotation set \
coherent and non-redundant.
- Prefer the most specific term that the phrase supports. If the phrase \
describes a 'low muscle tone', pick 'Hypotonia' rather than a generic \
'Abnormality of muscle tone'.
- If none of the candidates is a reasonable match for the phrase, return an \
empty selection.
- Respond with JSON only, exactly one of:
{"selected": "HP:0000000"}   or   {"selected": ""}"""

MAX_CTX_CHARS = 400
_SENT_RE = re.compile(r"[^\n.!?]+[.!?]*")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _morph_pattern(words: List[str]) -> str:
    parts = []
    for w in words:
        if w.endswith("ies"):
            stem = w[:-3] + "y"
            parts.append(f"{re.escape(stem)}(?:ies|s)?")
        elif w.endswith("s") and not w.endswith("ss"):
            parts.append(f"{re.escape(w[:-1])}s?")
        else:
            parts.append(f"{re.escape(w)}s?")
    return r"\b" + r"\s+".join(parts) + r"\b"


def anchor_sentence(text: str, phrase: str) -> Tuple[Optional[str], str]:
    """Return (sentence, tier) with tier in {exact, fuzzy, morph, none}."""
    t = _norm(text)
    spans = [(m.start(), m.end()) for m in _SENT_RE.finditer(t)]

    def sent_for(lo: int, hi: int) -> Optional[str]:
        for a, b in spans:
            if a <= lo and hi <= b:
                s = t[a:b].strip()
                if len(s) > MAX_CTX_CHARS:
                    s = s[:MAX_CTX_CHARS] + "..."
                return s
        return None

    p = _norm(phrase)
    i = t.find(p)
    if i >= 0:
        s = sent_for(i, i + len(p))
        if s:
            return s, "exact"
    words = p.split()
    for w in range(len(words), 2, -1):
        for start in range(0, len(words) - w + 1):
            frag = " ".join(words[start:start + w])
            if len(frag) < 20 and w < 4:
                continue
            j = t.find(frag)
            if j >= 0:
                s = sent_for(j, j + len(frag))
                if s:
                    return s, "fuzzy"
    m = re.search(_morph_pattern(words), t)
    if m:
        s = sent_for(m.start(), m.end())
        if s:
            return s, "morph"
    return None, "none"


def parse_single(raw: str) -> str:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return ""
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return ""
    sel = data.get("selected")
    if isinstance(sel, str):
        sel = sel.strip().upper()
        return sel if sel.startswith("HP:") else ""
    if isinstance(sel, dict):
        for k in ("hpo", "hpo_id", "id", "code"):
            v = (sel.get(k) or "").strip().upper()
            if v.startswith("HP:"):
                return v
    return ""


def load_definitions(hpo_json: str) -> Dict[str, str]:
    hp = json.loads(Path(hpo_json).read_text(encoding="utf-8"))
    defs = {}
    for n in hp["graphs"][0]["nodes"]:
        c = n.get("id", "").split("/")[-1].replace("_", ":")
        d = (n.get("meta") or {}).get("definition")
        if isinstance(d, dict) and d.get("val"):
            defs[c] = d["val"]
    return defs


def _candidate_lines(cands, hpo, defs: Dict[str, str], k: int) -> List[str]:
    lines = []
    for c in cands[:k]:
        dfn = defs.get(c.hpid, "")
        if dfn:
            lines.append(f"{c.hpid} {hpo.label(c.hpid)}: "
                         f"{re.sub(r'\s+', ' ', dfn)[:220]}")
        else:
            lines.append(f"{c.hpid} {hpo.label(c.hpid)}")
    return lines


class SelectionTier:
    """Constrained selection, resumable and off-block guarded."""

    def __init__(
        self,
        hpo,
        candidate_index,
        hpo_json: str,
        llm: Optional[LlmClient] = None,
        k: int = 20,
        out_path: Optional[str] = None,
    ) -> None:
        self.hpo = hpo
        self.index = candidate_index
        self.defs = load_definitions(hpo_json)
        self.client = llm or LlmClient(temperature=0.0)
        self.k = k
        self.out_path = out_path

    def select_doc(self, doc_text: str, phrases: List[str],
                   done_map: Optional[dict] = None) -> dict:
        """Run selection over one document's phrases, resuming from done_map.

        ``done_map`` holds phrases already settled by the equality rule;
        those are returned unchanged and never reach the LLM.

        Returns {phrase: hpo_code}.
        """
        mapping: dict = dict(done_map or {})
        todo = [p for p in phrases if p not in mapping]
        if not todo:
            return mapping
        vecs = self.index.embed(todo)
        plist = "\n".join(f"- {p}" for p in phrases)
        for i, ph in enumerate(todo):
            # term-level candidate block: each hpo id takes its best phrase
            # score; terms ranked by score; truncated to k.
            term_hits = self.index.topk_terms_for_query(vecs[i], k=self.k)
            cand_ids = [h for h, _ph, _s in term_hits]
            class _C:
                def __init__(self, hpid):
                    self.hpid = hpid
            cand_objs = [_C(h) for h in cand_ids]
            lines = _candidate_lines(cand_objs, self.hpo, self.defs, self.k)
            ctx = plist.replace(f"- {ph}\n", f"- {ph}  <-- current\n", 1) \
                if f"- {ph}\n" in plist else plist
            sent, tier = anchor_sentence(doc_text, ph)
            ctx_block = ""
            if sent:
                ctx_block = (f"\nSentence from the source document "
                             f"containing this phrase:\n\"{sent}\"\n")
            prompt = (f"Phenotypic phrase: \"{ph}\"\n"
                      f"{ctx_block}\n"
                      f"All phrases from the same document:\n{ctx}\n\n"
                      f"Candidate HPO terms:\n" + "\n".join(lines) +
                      "\n\nPick the single best match. Return JSON only.")
            best = ""
            for attempt in range(3):
                try:
                    raw = self.client.chat(prompt, max_tokens=128,
                                           system=_SYSTEM)
                    best = parse_single(raw)
                    break
                except FatalApiError:
                    raise
                except Exception:
                    time.sleep(1.0 * (attempt + 1))
            if best and best not in set(cand_ids):
                best = ""
            mapping[ph] = best
        return mapping