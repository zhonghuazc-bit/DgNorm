"""DGNorm pipeline: one entry point, five independent stages.

The pipeline is dataset-agnostic: it takes a DatasetAdapter, converts
the raw corpus into NormDocument via the adapter, then runs the five
stages over those documents:

    1. extract   -- dual-granularity phrase extraction (flash + gen)
    2. retrieve  -- SapBERT candidate delimitation (top-K block per phrase)
    3. select    -- source-sentence-injected constrained selection (mm8)
    4. shortcut  -- in-block equality-family short-circuit (zero LLM)
    5. evaluate  -- corpus micro (and observation micro for BC8-T3 style)

Every stage is independently reusable (import the module, run it over
any phrases/docs), and the paid stages checkpoint to jsonl so a rerun
over a cached corpus is free. Existing RAG-HPO artifacts
(raghpo_phrases.jsonl / raghpo_phrases_gen.jsonl /
raghpo_mr_mm8_ctxsent_sel.jsonl) remain valid caches and are picked up
when their paths are supplied, so the SDK reproduces the published
0.7441 line without re-paying.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ..tiers.llm_tier import LlmClient
from .adapters import DatasetAdapter
from .anchor import ExactAnchorTier
from .caching import load_phrases_map, load_selection_map
from .evaluate import evaluate, preds_from_selection
from .extract import combine, extract_phrases
from .retrieve import CandidateIndex
from .select import SelectionTier
from .shortcircuit import ShortCircuitTier
from .types import DgnormRunResult, NormDocument


@dataclass
class DgnormConfig:
    """Paths and settings for one pipeline run.

    hpo_json:     HP ontology release (hp.json, obo-graphs).
    sapbert_dir:  local SapBERT model dir (sentence-transformers format).
    index_npz:    prebuilt HPO surface-form dense index (npz with
                  phrases / phrase_ids / matrix arrays).
    out_dir:      where this dataset's fresh checkpoints are written
                  (default results/dgnorm/<dataset>).
    flash_cache / gen_cache / sel_cache:
        Optional existing jsonl caches to reuse instead of the out_dir
        defaults (e.g. the published RAG-HPO artifacts).
    k:            candidate block size.
    """

    dataset: str
    hpo_json: str
    sapbert_dir: str
    index_npz: str
    out_dir: str = "results/dgnorm"
    flash_cache: Optional[str] = None
    gen_cache: Optional[str] = None
    sel_cache: Optional[str] = None
    k: int = 20
    resume: bool = True
    # multi-code selection: one phrase may map to several HPO terms
    # (mirrors dictionary matching; for corpora whose gold annotates
    # several granularities of one finding, e.g. GSC). Default False keeps
    # the mm8 single-code selection byte-identical.
    select_multi: bool = False
    # exact-surface anchor (only meaningful with select_multi=True):
    # restrict the LLM multi-select to codes whose label/synonym matches
    # the phrase verbatim in the full HPO library (FastHPOCR-style exact
    # matching), curbing the free multi-select's false-positive explosion.
    # Default False keeps the pipeline byte-identical.
    exact_anchor: bool = False

    def _paths(self) -> Dict[str, Path]:
        base = Path(self.out_dir) / self.dataset
        base.mkdir(parents=True, exist_ok=True)
        return {
            "flash": Path(self.flash_cache) if self.flash_cache else base / "phrases_flash.jsonl",
            "gen": Path(self.gen_cache) if self.gen_cache else base / "phrases_gen.jsonl",
            "sel": Path(self.sel_cache) if self.sel_cache else base / "selection.jsonl",
        }


class DgnormPipeline:
    """Composed DGNorm. Adapters supply documents; stages run in order."""

    def __init__(self, config: DgnormConfig, llm: Optional[LlmClient] = None) -> None:
        self.config = config
        self.llm = llm or LlmClient(temperature=0.0)

    # ------------------------------------------------------------------
    # stage helpers (each independently callable)
    # ------------------------------------------------------------------

    def run_extraction(self, docs: List[NormDocument]) -> Dict[str, List[str]]:
        p = self.config._paths()
        flash = extract_phrases(docs, str(p["flash"]), "flash", self.llm,
                                resume=self.config.resume)
        gen = extract_phrases(docs, str(p["gen"]), "gen", self.llm,
                              resume=self.config.resume)
        return combine(flash, gen)

    def run_selection(
        self, docs: List[NormDocument], phrases: Dict[str, List[str]],
    ) -> Dict[str, dict]:
        p = self.config._paths()
        # lazily load shared resources so a stage can run standalone
        if not hasattr(self, "hpo"):
            self.hpo = self._load_hpo(self.config.hpo_json)
        if not hasattr(self, "index"):
            self.index = CandidateIndex(self.config.index_npz,
                                        self.config.sapbert_dir)
        sel = SelectionTier(self.hpo, self.index, self.config.hpo_json,
                            self.llm, k=self.config.k,
                            multi=self.config.select_multi,
                            anchor=ExactAnchorTier(self.hpo)
                            if self.config.exact_anchor else None)
        existing = load_selection_map(str(p["sel"])) if self.config.resume else {}
        out: Dict[str, dict] = dict(existing)
        fh = open(p["sel"], "a" if self.config.resume else "w", encoding="utf-8")
        t0 = time.time()
        n_phrases = 0
        for i, d in enumerate(docs):
            if d.obs_id in out:
                continue
            mapping = sel.select_doc(d.text, phrases.get(d.obs_id, []))
            out[d.obs_id] = mapping
            fh.write(json.dumps({"obs_id": d.obs_id, "map": mapping},
                                ensure_ascii=False) + "\n")
            fh.flush()
            n_phrases += len(mapping)
            if (i + 1) % 20 == 0:
                print(f"  [select] {i+1}/{len(docs)} docs, {n_phrases} phrases "
                      f"({time.time()-t0:.0f}s)", flush=True)
        fh.close()
        return out

    def run_shortcircuit(
        self, phrases: List[str],
    ) -> Dict[str, str]:
        sc = ShortCircuitTier(self.config.index_npz, self.config.sapbert_dir,
                              k=self.config.k)
        short, stats = sc.run(phrases, {})
        self.shortcircuit_stats = stats
        return short

    # ------------------------------------------------------------------
    # full run
    # ------------------------------------------------------------------

    def run(self, adapter: DatasetAdapter,
            max_docs: Optional[int] = None) -> DgnormRunResult:
        cfg = self.config
        docs = adapter.load()
        if max_docs is not None:
            docs = docs[:max_docs]
        print(f"[dgnorm] dataset={cfg.dataset} docs={len(docs)} "
              f"gold={sum(bool(d.gold_terms) for d in docs)}", flush=True)

        # 1) extraction
        phrases = self.run_extraction(docs)
        n_phrases = sum(len(v) for v in phrases.values())
        print(f"[dgnorm] phrases={n_phrases} over {len(phrases)} docs",
              flush=True)

        # 2) index / hpo shared resources
        self.hpo = self._load_hpo(cfg.hpo_json)
        self.index = CandidateIndex(cfg.index_npz, cfg.sapbert_dir)

        # 3) selection
        selection = self.run_selection(docs, phrases)
        print(f"[dgnorm] selection done for {len(selection)} docs", flush=True)

        # 4) short-circuit
        all_phrases = sorted({ph for v in phrases.values() for ph in v})
        short = self.run_shortcircuit(all_phrases)
        print(f"[dgnorm] short-circuited {len(short)}/{len(all_phrases)} phrases",
              flush=True)

        # 5) evaluate
        gold = {d.obs_id: set(d.gold_terms) for d in docs}
        negated = {d.obs_id: set(d.negated_terms)
                   for d in docs if d.negated_terms}
        preds = preds_from_selection(selection, short)
        metrics = evaluate(gold, preds, modes=("corpus", "observation",
                                               "bc8_official"),
                           negated_by_id=negated)

        result = DgnormRunResult(
            dataset=cfg.dataset,
            docs=docs,
            phrases=phrases,
            selection=selection,
            shortcircuit_map=short,
            shortcircuit_stats=getattr(self, "shortcircuit_stats", {}),
            final_preds={k: sorted(v) for k, v in preds.items()},
            metrics_gold={"corpus": sum(len(v) for v in gold.values()),
                          "docs": len(gold)},
            metrics_full=metrics,
        )
        print("[dgnorm] corpus_micro:", json.dumps(metrics["corpus_micro"]),
              flush=True)
        print("[dgnorm] observation_micro:",
              json.dumps(metrics["observation_micro"]), flush=True)
        if "bc8_official" in metrics:
            print("[dgnorm] bc8_official:",
                  json.dumps(metrics["bc8_official"]), flush=True)
        return result

    @staticmethod
    def _load_hpo(hpo_json: str):
        from ..knowledge.hpo import HPO
        return HPO(hpo_json)