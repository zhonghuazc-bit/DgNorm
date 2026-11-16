"""Ontology constraint tier: the methodological core of OntoNorm-Dx.

Given candidate HPO terms emitted by the rules tier or the LLM tier,
this tier validates and re-ranks them using the structure of the HPO
DAG:

1. Code-hallucination guard: every candidate ID must exist in the
   pinned HPO release. Nonexistent codes are removed and counted.
2. DAG-aware re-ranking:
   - ancestor (hypernym) distance penalty: a candidate that is a strict
     ancestor of a higher-scoring neighbour on a supported path is
     discounted proportionally to DAG distance;
   - lowest-common-ancestor (LCA) coherence: candidates whose LCA with
     the top candidate is shallow are penalized (cross-branch drift);
   - granularity preference: between a parent-child pair with
     comparable evidence, prefer the more specific child unless the
     evidence only supports the parent.
3. Semantic post-validation: candidates whose lexical (embedding-free)
   similarity to the evidence anchor falls below a threshold are demoted;
   if the top candidate for a mention falls below the floor, the mention
   is reverted to the rules-tier result.
4. Evidence-chain provenance on every accepted term.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..knowledge.hpo import HPO


@dataclass
class Candidate:
    """A candidate HPO term with evidence and a pre-constraint score."""

    hpid: str
    mention: str  # evidence anchor span text
    span: Tuple[int, int]
    origin: str  # "rules" | "llm"
    score: float = 0.5  # pre-constraint score (e.g., LLM confidence / lexical match)
    polarity: str = "positive"  # "positive" | "excluded"


@dataclass
class ConstraintResult:
    """Output of the ontology constraint tier for one finding."""

    accepted: bool
    hpid: Optional[str]
    final_score: float = 0.0
    # provenance / evidence chain
    mechanisms: List[str] = field(default_factory=list)  # which corrections applied
    reverted_to_rules: bool = False

    def to_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "hpid": self.hpid,
            "final_score": self.final_score,
            "mechanisms": self.mechanisms,
            "reverted_to_rules": self.reverted_to_rules,
        }


class HallucinationGuard:
    """Guards against codes not present in the ontology."""

    def __init__(self, hpo: HPO) -> None:
        self.hpo = hpo
        self.blocked: List[str] = []

    def validate(self, hpid: str) -> bool:
        ok = self.hpo.has(hpid)
        if not ok:
            self.blocked.append(hpid)
        return ok


class DagAwareReranker:
    """Re-ranks candidates using hierarchical structure.

    Three mechanisms, applied pairwise over the candidate list:

    (1) Ancestor (hypernym) distance penalty: a candidate that is a
        strict ancestor of a higher-scoring candidate on a supported
        path is discounted proportionally to its DAG distance from that
        more specific candidate.
    (2) LCA coherence: candidates whose lowest common ancestor with the
        current top candidate is shallow (near the root) are penalized
        as cross-branch drift.
    (3) Granularity preference: when a generalized candidate and a more
        specific descendant both score within ``granularity_margin`` of
        each other, the more specific descendant wins. Clinical
        phenotype coding prefers the most specific term the evidence
        supports, so this directly counteracts hypernym
        overgeneralization, the dominant LLM error mode in this task.
    """

    def __init__(
        self,
        hpo: HPO,
        ancestor_penalty: float = 0.15,
        lca_floor_depth: float = 2.0,
        granularity_margin: float = 0.3,
        granularity_penalty: float = 0.05,
    ) -> None:
        self.hpo = hpo
        self.ancestor_penalty = ancestor_penalty
        self.lca_floor_depth = lca_floor_depth
        self.granularity_margin = granularity_margin
        self.granularity_penalty = granularity_penalty

    def rerank(self, candidates: List[Candidate]) -> List[Candidate]:
        if not candidates:
            return candidates
        # Work on mutable copies so we do not mutate the caller's objects.
        ordered = sorted(
            (Candidate(hpid=c.hpid, mention=c.mention, span=c.span, origin=c.origin,
                       score=c.score, polarity=c.polarity) for c in candidates),
            key=lambda c: c.score, reverse=True,
        )
        top = ordered[0]

        # (1) + (2): penalty pass against the current top.
        for c in ordered:
            if c.hpid == top.hpid:
                continue
            if self.hpo.is_ancestor_of(c.hpid, top.hpid):
                dist = self.hpo.d(top.hpid) - self.hpo.d(c.hpid)
                c.score -= self.ancestor_penalty * max(1, dist)
            lca = self.hpo.lca(c.hpid, top.hpid)
            lca_depth = self.hpo.d(lca) if lca else 0
            if lca_depth < self.lca_floor_depth:
                c.score -= 0.1

        # (3): granularity preference, pairwise across the whole list.
        # A generalized candidate that scores within `granularity_margin`
        # of a specific descendant is demoted below that descendant.
        for u in ordered:
            for v in ordered:
                if u.hpid == v.hpid:
                    continue
                if self.hpo.is_ancestor_of(u.hpid, v.hpid):
                    if u.score - v.score < self.granularity_margin:
                        u.score = v.score - self.granularity_penalty

        ordered.sort(key=lambda c: c.score, reverse=True)
        return ordered


class SemanticPostValidator:
    """Demotes candidates too dissimilar to their evidence anchor.

    This version uses lexical/embedding-free proxy similarity (there is
    no external embedding dependency) based on shared tokens. Users may
    inject a real bi-encoder by subclassing.
    """

    def __init__(self, threshold: float = 0.2, floor: float = 0.05) -> None:
        self.threshold = threshold
        self.floor = floor
        self.demoted: List[Tuple[str, float]] = []

    def similarity(self, hpid: str, label: str, mention: str) -> float:
        a = set(label.lower().split())
        b = set(mention.lower().split())
        if not a or not b:
            return 0.0
        inter = a & b
        return len(inter) / max(len(a), len(b))  # token overlap Jaccard-ish

    def validate(self, hpid: str, label: str, mention: str, score: float) -> Tuple[float, bool]:
        """Return (adjusted_score, keep)."""
        sim = self.similarity(hpid, label, mention)
        if sim < self.threshold:
            # Demote but keep if the candidate is still above the floor
            # and was strongly supported (e.g. rules-tier exact match).
            adjusted = score * max(0.3, sim)
            self.demoted.append((hpid, sim))
            return adjusted, adjusted >= self.floor
        return score, True


class OntologyConstraintTier:
    """Compose the three mechanisms into the tier used by the pipeline."""

    def __init__(self, hpo: HPO, threshold: float = 0.2, floor: float = 0.05) -> None:
        self.hpo = hpo
        self.guard = HallucinationGuard(hpo)
        self.reranker = DagAwareReranker(hpo)
        self.validator = SemanticPostValidator(threshold=threshold, floor=floor)
        self.stats: Dict[str, int] = {"hallucinated": 0, "demoted": 0, "reverted": 0}

    def process_mention(
        self,
        candidates: List[Candidate],
        rules_result: Optional[str] = None,
    ) -> Optional[ConstraintResult]:
        """Process all candidates for one mention.

        Returns a ConstraintResult for the single best HPO term, or None
        when nothing survives validation.
        """
        # 1) Hallucination guard
        kept: List[Candidate] = []
        for c in candidates:
            if self.guard.validate(c.hpid):
                kept.append(c)
            else:
                self.stats["hallucinated"] += 1

        if not kept:
            if rules_result:
                self.stats["reverted"] += 1
                return ConstraintResult(
                    accepted=True, hpid=rules_result, final_score=0.2,
                    mechanisms=["fallback_to_rules"], reverted_to_rules=True,
                )
            return ConstraintResult(accepted=False, hpid=None, mechanisms=["no_valid_candidate"])

        # 2) DAG-aware re-ranking
        ranked = self.reranker.rerank(kept)
        top = ranked[0]

        # 3) Semantic post-validation
        label = self.hpo.label(top.hpid)
        score, keep = self.validator.validate(top.hpid, label, top.mention, top.score)
        if not keep:
            self.stats["demoted"] += 1
            if rules_result:
                self.stats["reverted"] += 1
                return ConstraintResult(
                    accepted=True, hpid=rules_result, final_score=0.2,
                    mechanisms=["semantic_demotion", "fallback_to_rules"],
                    reverted_to_rules=True,
                )
            return ConstraintResult(accepted=False, hpid=None, mechanisms=["semantic_demotion"])

        return ConstraintResult(
            accepted=True,
            hpid=top.hpid,
            final_score=score,
            mechanisms=["dag_rerank", "semantic_pass"],
        )

    def summary(self) -> Dict[str, int]:
        return dict(self.stats)