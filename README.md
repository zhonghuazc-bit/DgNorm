# DGNorm SDK — verification release

DGNorm models expert annotation as dual-granularity extraction for Human
Phenotype Ontology (HPO) normalization of clinical case reports. One
instruction-tuned LLM reads each document twice under two output
specifications, once copying the original wording and once emitting
standard concept names; the phrase union is bounded by candidates over
ontology surface forms; deterministic equality rules inside each phrase's
own candidate block settle every case decidable without judgment, and only
the remaining phrases go to a source-sentence-injected constrained choice
that assigns one code or abstains.

This repository contains the SDK and the artifacts required to verify the
published main result. It deliberately does **not** contain the full
experimental harness, all four corpora, the four comparator baselines,
SapBERT weights, the HPO release, the dense index, API keys or per-run
logs. Those remain with the original sources listed in
`docs/REPRODUCE.md`.

## Verify the reported number in one command

```bash
python examples/verify_reported_results.py
```

The verifier uses only the Python standard library. It needs no network,
no API key, no model weights and no HPO file. It loads the 112 per-phrase
selection records and the deterministic short-circuit map in
`verification/`, rebuilds the final HPO code sets, and recomputes strict
corpus-level micro precision, recall and F1 against the 112 gold sets.
Expected output ends with:

```
CSC strict corpus micro: TP 1361 / FP 503 / FN 433
P 0.7302  R 0.7586  F1 0.7441
ALL CHECKS PASSED
```

It also asserts the endpoint sensitivity readouts of the paper's Table 6
on the same 112 documents: DeepSeek 4.1 Flash 0.7441, GLM-5.3-Flash
0.7308, Qwen3.8-Flash 0.7117.

## What is included

| Path | Content |
|---|---|
| `src/ontonorm/dgnorm/` | the SDK modules: extraction, candidate delimitation, equality rule, constrained selection, evaluation, adapters, pipeline |
| `src/ontonorm/tiers/`, `src/ontonorm/knowledge/`, `src/ontonorm/data/` | the minimal supporting modules imported by the SDK |
| `verification/` | 112 gold sets, 112 final prediction sets, per-phrase selection records, short-circuit map, locked metrics |
| `configs/dgnorm_csc.json` | the main configuration and locked counts |
| `examples/verify_reported_results.py` | pure-stdlib verification |
| `docs/REPRODUCE.md` | how to obtain external resources and rerun the full pipeline |

The SDK is importable after `pip install -e .`:

```python
from ontonorm.dgnorm import DgnormConfig, DgnormPipeline, RagHpoAdapter
```

## Scope of the claim

The released artifacts verify the **strict corpus-level micro** result on
the 112 gold-bearing CSC documents. They do not reproduce the folded
matching, per-case macro, the other three corpora or the comparator
reruns; those require the external corpora and baseline repositories.
The four comparator systems were rerun from their official released code,
and the paper's own DGNorm implementation is the code in this repository.

## License

Apache License 2.0. See `LICENSE`.
