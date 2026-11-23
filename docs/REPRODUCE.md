# Reproducing the full DGNorm run

The one-command verifier in `examples/` needs no external resources. A
full rerun does, and those resources are not redistributed here.

## External resources

| Resource | Source | Used for |
|---|---|---|
| HPO release hp 2026-09-01, `hp.json` | HPO consortium distribution | ontology surface forms and `is_a` graph |
| SapBERT `sapbert-mnli-stsb` | public model release, converted to sentence-transformers format | phrase embeddings |
| `dense_index_sapbert.npz` | built locally from the HPO labels and synonyms with the model above | candidate blocks |
| `RAG-HPO_tests.xlsx`, CSC sheet | supplementary data of the RAG-HPO paper | documents and gold codes |
| OpenAI-compatible LLM endpoint | provider of choice | the two extraction passes and constrained selection |

The dense index stores `phrases`, `phrase_ids` and `matrix` arrays; build
it over the union of HPO labels and synonyms from the same release. The
paper fixes the candidate block at `k = 20`.

## Configuration

`configs/dgnorm_csc.json` records the main configuration and the locked
counts. The pipeline reads paths through `DgnormConfig` and reuses JSONL
checkpoints, so a rerun over a cached corpus costs no model calls:

```python
from ontonorm.dgnorm import DgnormConfig, DgnormPipeline, RagHpoAdapter

config = DgnormConfig(
    dataset="csc",
    hpo_json="path/to/hp.json",
    sapbert_dir="path/to/sapbert-mnli-stsb",
    index_npz="path/to/dense_index_sapbert.npz",
    out_dir="results/dgnorm",
    k=20,
    resume=True,
)
result = DgnormPipeline(config).run(RagHpoAdapter("path/to/RAG-HPO_tests.xlsx"))
print(result.metrics_full["corpus_micro"])
```

Set the provider key in the environment, for example
`DEEPSEEK_API_KEY`, and pass the matching `LlmClient` configuration to
`DgnormPipeline`. No key is stored in this repository.

## Metric definitions

The paper reports two protocols, both recomputed by the verifier from the
same prediction sets.

**Main result (Tables 3, 4, 5): normalized strict matching, per-document
macro.** Each document's gold and prediction sets are first normalized by
collapsing them to their most specific codes: a code is dropped when a
strict descendant of it is present in the same set. The collapse uses the
hp 2026-09-01 `is_a` graph; `verification/hpo_descendants_scope.json` ships
exactly the descendant relation restricted to codes appearing in these 112
documents, so normalization is reproducible offline. A normalized code is a
true positive only when it exactly equals a normalized gold code of the same
document. Per-document precision, recall and F1 are computed and then averaged
with equal weight over all documents. This yields the paper's CSC headline
F1 0.7490 (P 0.7606, R 0.7499), locked in
`verification/expected_metrics_collapsed.json`.

**Release cross-check: strict matching, corpus-level micro.** Without
normalization, a code is a true positive only when it exactly equals a gold
code of the same document. TP, FP and FN are summed over all documents and
then divided to give precision, recall and F1. This yields F1 0.7441
(TP 1361, FP 503, FN 433), locked in `verification/expected_metrics.json`.
The two protocols answer different questions and are not interchangeable;
the verifier asserts both so the released predictions are checked against
the headline number as well as the stricter aggregate.

## Endpoint sensitivity

Table 6 of the paper reports three endpoints on the same 112 documents
under strict corpus-level micro: DeepSeek 4.1 Flash 0.7441, GLM-5.3-Flash
0.7308, Qwen3.8-Flash 0.7117. Those runs differ only in the endpoint; their
caches and the zero-model-call rescope are summarized in
`verification/expected_metrics.json`.
