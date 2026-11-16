"""Human Phenotype Ontology access layer for OntoNorm-Dx.

Loads an OBO Graphs JSON (or OBO file) into a NetworkX DAG with label,
synonym and ID indexes, and provides the ontology-level operations used
by the constraint tier: ancestor/descendant queries, lowest common
ancestor, granularity depth, and lexical lookup.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx


@dataclass
class HPO:
    """An HPO release loaded from obo-graphs JSON.

    Parameters
    ----------
    source: path to hp.json (obo-graphs) or hp.obo.
    """

    source: str
    alt_id_map: Optional[Dict[str, str]] = None
    graph: nx.DiGraph = field(default_factory=nx.DiGraph, init=False)
    labels: Dict[str, str] = field(default_factory=dict, init=False)
    synonyms: Dict[str, List[str]] = field(default_factory=dict, init=False)
    id2label: Dict[str, str] = field(default_factory=dict, init=False)
    label2ids: Dict[str, List[str]] = field(default_factory=dict, init=False)
    depth: Dict[str, int] = field(default_factory=dict, init=False)
    _lower_index: Dict[str, List[str]] = field(default_factory=dict, init=False)
    # A load-time alternative-identifier switch (channel A2). A map
    # obsolete_id -> primary_id; when set, obsolete ids are folded into their
    # primary term in the graph, labels and indexes so that a term annotated
    # under an obsolete id becomes reachable by structure and by retrieval.
    # The map itself is built from the release's own hasAlternativeId axioms.
    _alt_id_map: Dict[str, str] = field(default_factory=dict, init=False)
    # Reachability caches. ``networkx.ancestors`` walks the whole graph on
    # every call; the constraint tier calls it O(n^2) times per candidate
    # list, which made a 56-point calibration grid take minutes instead of
    # seconds. The graph is immutable after load, so memoising is safe.
    _anc_cache: Dict[str, frozenset] = field(default_factory=dict, init=False)
    _desc_cache: Dict[str, frozenset] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._load()

    def _load(self) -> None:
        if self.source.endswith(".json"):
            self._load_json(self.source)
        else:
            raise ValueError("Only obo-graphs JSON (.json) is supported in this version")

    def _load_json(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        graph = data["graphs"][0]
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        is_a = graph.get("meta", {}).get("isA", [])

        # Fold obsolete ids into primary terms before building indexes, if the
        # alt-id switch is on. remap(old) -> primary. Graph nodes for obsolete
        # ids are orphan (no is_a edge reaches them), so folding means: the
        # primary term carries the obsolete id's label/synonyms; retrieval and
        # structure queries then surface the primary term for either spelling.
        if self.alt_id_map:
            self._alt_id_map = dict(self.alt_id_map)

        # Nodes
        for n in nodes:
            nid = n["id"]
            # Strip full IRI to HP:0000001 form
            short = nid.split("/")[-1].replace("_", ":")
            self.graph.add_node(short)
            lbl = n.get("lbl", "")
            self.labels[short] = lbl
            if lbl:
                self.id2label[short] = lbl
                self.label2ids.setdefault(lbl.lower(), []).append(short)
            # xrefs / synonyms
            meta = n.get("meta", {})
            syns = []
            for s in meta.get("synonyms", []):
                syns.append(s.get("val", ""))
            self.synonyms[short] = syns

        # After loading all nodes, remap: any index entry whose key is an
        # obsolete id is re-pointed to its primary term. A primary term that
        # already exists keeps its own label; if it was absent (a bare folder),
        # it inherits the obsolete id's label/synonyms.
        if self._alt_id_map:
            # Build a reverse: primary -> obsolete ids that map to it
            for old, primary in list(self._alt_id_map.items()):
                if old not in self.graph or primary == old:
                    continue
                # Move the obsolete id's label/synonym/indices onto primary
                if primary not in self.graph:
                    self.graph.add_node(primary)
                if primary not in self.labels or not self.labels.get(primary):
                    if self.labels.get(old):
                        self.labels[primary] = self.labels[old]
                        self.id2label[primary] = self.labels[old]
                for s in self.synonyms.get(old, []):
                    if s and s not in self.synonyms.get(primary, []):
                        self.synonyms.setdefault(primary, []).append(s)
                # Merge edges: primary inherits the obsolete id's parents and
                # children, so structure queries reach it from either side.
                for p in list(self.graph.predecessors(old)):
                    self.graph.add_edge(p, primary)
                for c in list(self.graph.successors(old)):
                    self.graph.add_edge(primary, c)
                # The obsolete id itself stays as an alias-free orphan; drop
                # it from query surfaces by removing its label index
                # (kept in graph so has() stays true, but not retrievable).
                if self.labels.get(old):
                    self.label2ids.pop(self.labels[old].lower(), None)
                # Re-index primary under old's lower forms
                if self.labels.get(primary):
                    self.label2ids.setdefault(self.labels[primary].lower(), []).append(primary)
                for s in self.synonyms.get(primary, []):
                    self._lower_index.setdefault(s.lower(), []).append(primary)

        # Edges (is_a). We store parent -> child so that networkx
        # ancestors() == superclasses and descendants() == subclasses.
        for e in edges:
            pred = e.get("pred")
            if pred and ("is_a" in pred or "subClassOf" in pred):
                sub = e["sub"].split("/")[-1].replace("_", ":")
                obj = e["obj"].split("/")[-1].replace("_", ":")
                # remap obsolete endpoints to primary before adding
                sub = self._alt_id_map.get(sub, sub)
                obj = self._alt_id_map.get(obj, obj)
                if sub in self.graph and obj in self.graph:
                    self.graph.add_edge(obj, sub)  # parent -> child

        # Legacy isA array (some releases carry it on the graph meta)
        for pair in is_a:
            sub = pair["sub"].split("/")[-1].replace("_", ":")
            obj = pair["obj"].split("/")[-1].replace("_", ":")
            sub = self._alt_id_map.get(sub, sub)
            obj = self._alt_id_map.get(obj, obj)
            if sub in self.graph and obj in self.graph:
                self.graph.add_edge(obj, sub)

        # Compute depth (distance from root, i.e. longest path length to "All")
        roots = [n for n in self.graph.nodes if self.graph.in_degree(n) == 0]
        for r in roots:
            path_lens = nx.single_source_shortest_path_length(self.graph, r)
            for n, d in path_lens.items():
                self.depth[n] = max(self.depth.get(n, 0), d)

        # Lower-case index for the rules tier
        for label, ids in self.label2ids.items():
            for i in ids:
                self._lower_index.setdefault(label, []).append(i)
        for hpid, syns in self.synonyms.items():
            for s in syns:
                self._lower_index.setdefault(s.lower(), []).append(hpid)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def has(self, hpid: str) -> bool:
        return hpid in self.graph

    def label(self, hpid: str) -> str:
        return self.labels.get(hpid, "")

    def ancestors(self, hpid: str) -> set:
        """Return strict superclasses (more general terms) of ``hpid``."""
        if hpid not in self.graph:
            return set()
        cached = self._anc_cache.get(hpid)
        if cached is None:
            cached = frozenset(nx.ancestors(self.graph, hpid))
            self._anc_cache[hpid] = cached
        return set(cached)

    def descendants(self, hpid: str) -> set:
        """Return strict subclasses (more specific terms) of ``hpid``."""
        if hpid not in self.graph:
            return set()
        cached = self._desc_cache.get(hpid)
        if cached is None:
            cached = frozenset(nx.descendants(self.graph, hpid))
            self._desc_cache[hpid] = cached
        return set(cached)

    def parents(self, hpid: str) -> set:
        if hpid not in self.graph:
            return set()
        return set(self.graph.predecessors(hpid))

    def is_ancestor_of(self, a: str, b: str) -> bool:
        """True if a is a strict ancestor of b (a more general term)."""
        return a in self.ancestors(b)

    def lca(self, a: str, b: str) -> Optional[str]:
        """Lowest common ancestor of two terms (closest common parent)."""
        if a == b:
            return a
        anc_a = self.ancestors(a) | {a}
        anc_b = self.ancestors(b) | {b}
        common = anc_a & anc_b
        if not common:
            return None
        # The LCA is the common ancestor with max depth
        return max(common, key=lambda x: self.depth.get(x, 0))

    def lexical_lookup(self, text: str) -> List[str]:
        """Return HPO IDs whose label or synonym matches ``text`` exactly (case-insensitive)."""
        key = text.strip().lower()
        return list(self._lower_index.get(key, []))

    def lexical_search_stem(self, text: str, min_len: int = 3) -> List[Tuple[str, str]]:
        """Stem-ish search: return (hpid, matched_label) where the HPO label
        contains the token as a substring (word-boundary aware). Cheap and
        useful for the rules tier. Order: exact > starts-with > contains.
        """
        key = text.strip().lower()
        if len(key) < min_len:
            return []
        toks = re.escape(key)
        exact = self._lower_index.get(key, [])
        results: List[Tuple[str, str, int]] = []
        for lbl, ids in self._lower_index.items():
            if lbl == key:
                continue
            for hpid in ids:
                if key in lbl:
                    if lbl.startswith(key):
                        results.append((hpid, self.labels.get(hpid, ""), 1))
                    else:
                        results.append((hpid, self.labels.get(hpid, ""), 2))
        results.sort(key=lambda x: (x[2], len(x[1])))
        return [(h, l) for h, l, _ in results[:10]]

    def common_ancestor_depth(self, a: str, b: str) -> int:
        l = self.lca(a, b)
        return self.depth.get(l, 0) if l else 0

    def d(self, hpid: str) -> int:
        return self.depth.get(hpid, 0)

    def __len__(self) -> int:
        return len(self.graph)