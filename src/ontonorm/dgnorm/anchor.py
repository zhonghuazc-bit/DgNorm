# -*- coding: utf-8 -*-
"""Stage 3.5 (optional): exact-surface anchor for multi-code output.

Motivation (GSC): the gold annotates a single mention at several HPO
granularities (ancestor + specific terms). The single-code selector can
only emit one code per phrase; the free multi-code selector recovers
recall (+13.6 pt) but explodes precision (+658 FP) because the LLM picks
near-neighbours from the top-K block instead of the exact surface match.

ExactAnchorTier is a pure-rule, zero-LLM layer: for each phrase it looks
up the phrase's canonical label + synonyms across the WHOLE HPO surface
library (not a candidate block) with exact equality. Every hit is a code
the phrase text corresponds to verbatim, so emitting all hits is safe --
the same constraint that keeps FastHPOCR's dictionary matching precise.

Anchors are produced BEFORE the LLM selection and passed to it:
  - the LLM's candidate block is restricted to the anchor codes when a
    non-empty anchor exists (the LLM may drop some, but may not invent
    near-neighbours outside the anchor);
  - when no anchor hits, the LLM keeps its normal top-K choice.
This turns "free multi-select" into "exact-anchored multi-select".

Default OFF (config.exact_anchor=False), the pipeline is byte-identical.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Set

_CLEAN = re.compile(r"[^a-z0-9 ]")


def _clean(s: str) -> str:
    return _CLEAN.sub("", (s or "").replace("_", " ").lower()).strip()


class ExactAnchorTier:
    """Exact-surface anchor lookups over the full HPO label/synonym index."""

    def __init__(self, hpo) -> None:
        self.hpo = hpo
        # index: cleaned surface -> set(code)
        self._surf: Dict[str, Set[str]] = {}
        for hpid in hpo.graph.nodes:
            lbl = _clean(hpo.label(hpid))
            if lbl:
                self._surf.setdefault(lbl, set()).add(hpid)
            for syn in getattr(hpo, "synonyms", {}).get(hpid, []) or []:
                cs = _clean(syn)
                if cs:
                    self._surf.setdefault(cs, set()).add(hpid)

    def anchor(self, phrase: str) -> Set[str]:
        """HPO codes whose label/synonym equals the phrase exactly (cleaned).

        Returns the full set of codes for every matching surface -- the
        multi-code anchor the phrase text verbatim supports.
        """
        k = _clean(phrase)
        if not k:
            return set()
        direct = set(self._surf.get(k, set()))
        # token-set equality for order/plural-inflection robustness
        toks = set(k.split())
        if len(k.split()) == 1:
            return direct
        for surf, ids in self._surf.items():
            if surf != k and set(surf.split()) == toks \
                    and len(surf.split()) == len(k.split()):
                direct |= ids
        return direct

    def restrict_candidates(
        self,
        phrase: str,
        cand_ids: List[str],
    ) -> tuple[Optional[List[str]], Set[str]]:
        """Restrict the LLM candidate block to the anchor codes.

        Returns (anchored_block_or_None, anchor_codes). If the phrase has a
        non-empty anchor, the block is the anchor codes ∩ candidate block
        (falling back to the full candidate block when the anchor is
        entirely outside the block -- the retriever may miss the exact hit);
        if no anchor, returns (None, set()).
        """
        anc = self.anchor(phrase)
        if not anc:
            return None, set()
        anc_in_block = [c for c in cand_ids if c in anc]
        if anc_in_block:
            return anc_in_block, anc
        return None, anc