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
selection records, the deterministic short-circuit map and a minimal HPO
is_a descendant table shipped in `verification/`, rebuilds the final HPO
code sets, and recomputes both scoring protocols used in the paper against
the 112 gold sets. Expected output ends with:

```
MAIN RESULT (paper Tables 3-5): normalized collapsed strict, per-document macro
  CSC 112 docs: P 0.7606  R 0.7499  F1 0.7490
RELEASE CROSS-CHECK: strict exact match, corpus-level micro
  TP 1361 / FP 503 / FN 433, P 0.7302  R 0.7586  F1 0.7441
Endpoint sensitivity (micro): DeepSeek 0.7441, GLM 0.7308, Qwen 0.7117
ALL CHECKS PASSED
```

The **main result** is the paper's headline number: F1 0.7490 on CSC under
normalized strict matching with per-document macro averaging, the protocol
of Tables 3, 4 and 5. Normalization collapses each document's code set to
its most specific codes by dropping any code that has a descendant in the
same set; the shipped descendant table is the exact hp 2026-09-01 is_a
relation restricted to the codes appearing in these 112 documents. The
**release cross-check** is the stricter corpus-level micro reading (0.7441)
and the three-endpoint sensitivity of Table 6; both are computed from the
same prediction sets and are reported for transparency, not as substitutes
for the main result.

## What is included

| Path | Content |
|---|---|
| `src/ontonorm/dgnorm/` | the SDK modules: extraction, candidate delimitation, equality rule, constrained selection, evaluation, adapters, pipeline |
| `src/ontonorm/tiers/`, `src/ontonorm/knowledge/`, `src/ontonorm/data/` | the minimal supporting modules imported by the SDK |
| `verification/` | 112 gold sets, 112 final prediction sets, per-phrase selection records, short-circuit map, minimal HPO descendant table, locked metrics for both protocols |
| `configs/dgnorm_csc.json` | the main configuration and locked counts for both protocols |
| `examples/verify_reported_results.py` | pure-stdlib verification |
| `docs/REPRODUCE.md` | how to obtain external resources and rerun the full pipeline |

The SDK is importable after `pip install -e .`:

```python
from ontonorm.dgnorm import DgnormConfig, DgnormPipeline, RagHpoAdapter
```

## Scope of the claim

The released artifacts reproduce the paper's **main result on CSC**: F1
0.7490 under normalized strict matching with per-document macro on the 112
gold-bearing documents, using the same prediction sets that produced the
published tables. They also report the strict corpus-level micro reading
and the three-endpoint sensitivity over the same documents. The other three
corpora and the comparator reruns require the external corpora and baseline
repositories and are not redistributed here. The four comparator systems
were rerun from their official released code, and the paper's own DGNorm
implementation is the code in this repository.

## License

Apache License 2.0. See `LICENSE`.
