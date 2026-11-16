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

## Metric definition

Strict matching counts a predicted HPO code as true positive only when it
is exactly a gold code of the same document. Precision, recall and F1 are
computed from TP, FP and FN aggregated over all documents, then divided.
This is corpus-level micro, not per-case macro, and it is distinct from
the paper's folded matching and from the LEAP paper's ancestor-relaxed
scoring. `verification/expected_metrics.json` fixes the locked counts.

## Endpoint sensitivity

Table 6 of the paper reports three endpoints on the same 112 documents
under the same strict corpus-level micro: DeepSeek 4.1 Flash 0.7441,
GLM-5.3-Flash 0.7308, Qwen3.8-Flash 0.7117. Those runs differ only in
the endpoint; their caches and the zero-model-call rescope are summarized
in `verification/expected_metrics.json`.
