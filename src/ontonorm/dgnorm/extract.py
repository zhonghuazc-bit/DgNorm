"""Stage 1 of DGNorm: dual-granularity phrase extraction.

Two LLM calls per document, identical model and decoding (deepseek-chat,
temperature 0), differing only in the output spec -- the exact contrast
that models the expert's two-layer annotation cognition:

  * ``flash`` copies the document's own wording (surface layer);
  * ``gen`` produces an HPO-style normalized concept phrase per finding
    (normalization layer).

The union, order-preserving and deduplicated, is the phrase set that
feeds candidate delimitation. Both extractors cache per-document to
jsonl, so re-running a cached corpus is free, and the two caches are
kept in separate files keyed by phase name.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from ..tiers.llm_tier import FatalApiError, LlmClient
from .types import NormDocument as _NormDoc  # noqa: F401 (typing only)

FLASH_SYSTEM = """You read a clinical document and list the phenotypic findings it \
reports.

Return one short phrase per finding. Each phrase must be the finding as the \
document words it: a noun phrase or a short clause, copied from the document \
rather than paraphrased, naming exactly one finding.

Rules:
- List every abnormal finding the document reports. A long case report often \
contains 10 to 20 of them, so read the whole document before answering.
- Phrase length: at most 12 words. Split compound findings into separate \
phrases when they name different findings.
- Do not include normal findings, absent findings, findings the document \
rules out, measurements, instruments, or body regions that could not be \
assessed.
- Do not include family history, previous conditions, or the reason for \
referral unless the document states them as findings of this case.
- Do not output HPO identifiers, codes, or ontology numbers. Output only the \
phrasing.
- Respond with JSON only:
{"phrases": ["...", "..."]}"""

GEN_SYSTEM = """You read a clinical document and list every phenotypic finding it \
reports.
Output rules:
1. Respond with JSON only:
{"phrases": ["...", "..."]}
2. Each finding is ONE short phrase of 1-6 words, written the way HPO concept \
names are written, e.g. 'Fatigue', 'Syncope', 'Joint contractures', \
'Severe short stature'.
3. Normalize the wording: output the standard name of the clinical finding \
rather than the document's own wording. A described sign or test result \
should become the named finding it indicates (e.g. 'spells of loss of \
consciousness' -> 'Loss of consciousness'); a diagnosed disease name is kept \
as its own phrase.
4. Split: if one sentence reports several findings, output one phrase per \
finding.
5. A long case report often contains 10 to 20 findings, so read the whole \
document before answering.
6. Do not output normal findings, treatments, medications or family history \
as findings of this patient."""


def parse_phrases(raw: str) -> Optional[List[str]]:
    """Tolerant parse of the phrase list, or None if nothing usable."""
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    items = data.get("phrases")
    if not isinstance(items, list):
        for alt in ("findings", "terms", "results", "items", "keys"):
            if isinstance(data.get(alt), list):
                items = data[alt]
                break
    if not isinstance(items, list):
        return None
    out: List[str] = []
    for it in items:
        if isinstance(it, str):
            p = it.strip()
        elif isinstance(it, dict):
            p = str(it.get("phrase") or it.get("text") or it.get("mention")
                     or "").strip()
        else:
            continue
        if p and len(p.split()) <= 20:
            out.append(p)
    return list(dict.fromkeys(out))


def _user_prompt(text: str) -> str:
    return (f"Clinical document:\n\"{text}\"\n\n"
            "List the phenotypic findings this document reports. "
            "Return JSON only.")


def extract_phrases(
    docs: List[_NormDoc],
    out_path: str,
    phase: str = "flash",
    llm_client: Optional[LlmClient] = None,
    resume: bool = True,
    max_calls: Optional[int] = None,
) -> Dict[str, List[str]]:
    """Run one extractor (``phase`` in {"flash", "gen"}) over the corpus.

    Returns {obs_id: [phrases]}. Already-cached ids are skipped (free
    rerun); each new document is appended to ``out_path`` as it completes,
    so an interrupted run resumes without re-paying.
    """
    if phase not in ("flash", "gen"):
        raise ValueError(f"phase must be 'flash' or 'gen', got {phase!r}")
    system = FLASH_SYSTEM if phase == "flash" else GEN_SYSTEM

    from ..dgnorm.caching import load_phrases_map

    cache = load_phrases_map(out_path)
    todo = [d for d in docs if d.obs_id not in cache]
    if max_calls is not None:
        todo = todo[:max_calls]

    client = llm_client or LlmClient(temperature=0.0)
    if todo:
        try:
            client.preflight()
        except FatalApiError as e:
            raise SystemExit(f"preflight failed: {e}") from e

    done: Dict[str, List[str]] = dict(cache)
    if todo:
        fh = open(out_path, "a" if resume else "w", encoding="utf-8")
        errs = empty = 0
        t0 = time.time()
        for i, d in enumerate(todo):
            raw = None
            for attempt in range(3):
                try:
                    raw = client.chat(_user_prompt(d.text), max_tokens=3072,
                                      system=system)
                    break
                except FatalApiError:
                    raise
                except Exception:
                    time.sleep(1.0 * (attempt + 1))
            phrases = parse_phrases(raw) if raw else None
            if phrases is None:
                errs += 1
                rec = {"obs_id": d.obs_id, "phrases": [], "error": "unparseable"}
            elif not phrases:
                empty += 1
                rec = {"obs_id": d.obs_id, "phrases": [], "error": ""}
            else:
                rec = {"obs_id": d.obs_id, "phrases": phrases, "error": ""}
            done[d.obs_id] = rec["phrases"]
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            if (i + 1) % 20 == 0:
                print(f"  [{phase}] {i+1}/{len(todo)} "
                      f"({time.time()-t0:.0f}s) errs={errs} empty={empty}",
                      flush=True)
        fh.close()
    return done


def combine(flash: Dict[str, List[str]], gen: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Order-preserving deduplicated union (flash first, gen appended)."""
    return {k: list(dict.fromkeys(flash.get(k, []) + gen.get(k, [])))
            for k in {**flash, **gen}}