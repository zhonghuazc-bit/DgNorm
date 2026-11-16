"""Long-text selection prompt for narrative clinical documents.

The shipped selector prompt in ``selection_tier`` is written for one short
dysmorphology observation: it names the corpus, calls the input a "short
clinical observation", and its rules are phrased per finding, which implicitly
expects one or two terms per case. A RAG-HPO case report averages 1,887
characters and 16 gold terms, so three things have to change while the
architecture stays identical:

* the corpus named in the system prompt;
* the expectation of how many findings a document carries, stated explicitly
  so the model does not stop after the first few;
* the token budget, which must grow with the expected number of emitted terms.

Nothing else differs: the model still selects only from the supplied candidate
list, still abstains when there is nothing to report, and still may use the
off-list ``extra`` channel. Keeping the rule set byte-identical apart from
these points is what lets the corpus comparison mean "the placement effect
reproduces on another corpus" rather than "two different systems were run".

The prompt also states that findings absent from the text must not be
selected, which the short-observation prompt leaves implicit via "key
findings of this observation"; on a long narrative the model otherwise tends
to annotate the case history of a published report, which includes findings
the annotators did not code.
"""
from __future__ import annotations

# Used when the selector runs over narrative documents (case reports,
# disease-entry summaries) rather than short examination observations.
LONG_TEXT_SYSTEM = """You are a clinical phenotype coding assistant working \
on narrative clinical documents: published case reports and disease summaries.

You are given a whole clinical document and a numbered list of candidate \
Human Phenotype Ontology (HPO) terms that an automatic retriever proposed \
from it.

Your job is to select, from that list, every term that is an abnormal \
phenotypic finding reported in this document.

Rules:
- A document reports MANY findings, often 10 to 20 of them. Go through the \
document systematically and select all of them. Do not stop after the first \
few findings you find.
- Select ONLY terms that appear in the candidate list. Do not invent IDs.
- Select a term only if the document actually states that finding. Do not \
select a finding merely because it is typical, expected, or commonly \
associated with the condition discussed; do not select findings that the \
document lists as absent, denied, ruled out, or normalised.
- Terms in the candidate list that name a normal finding, an absent finding, \
or an instrument, measurement or body region that could not be assessed must \
NOT be selected.
- Prefer the most specific candidate the document supports. Do not select \
both a term and its more general parent.
- History of previous conditions, family history, and findings explicitly \
excluded by the author are not findings of this document.
- If the document clearly states an abnormal finding that the candidate list \
does not cover, you may add it to "extra".
- Respond with JSON only, exactly:
{"selected": ["HP:0000000"], "extra": [{"hpo": "HP:0000000"}],
 "abstain": false}
Use an empty list for "selected" and "extra" when nothing applies, and set \
"abstain" to true only when the document reports no abnormal finding at all."""

# Annotated variant: same task, but each candidate line carries the two
# priors that measurably separate true from false positives on this task --
# how often the term is annotated across disease resources, and how often this
# corpus's own vocabulary uses it. Structural features (depth, descendant
# count, ancestor count) are deliberately withheld: their discriminative power
# measured at or below chance on the frozen predictions, so showing them would
# spend the model's attention on noise.
#
# The point of this variant is to test whether the selector's weakness is
# missing evidence or missing capacity to use evidence it already has. On the
# narrative corpus the pool already contains 0.85 oracle F1 while the pipeline
# realises 0.63, so the gap is in the choosing, not the offering.
ANNOTATED_SYSTEM = LONG_TEXT_SYSTEM.replace(
    "Your job is to select, from that list, every term that is an abnormal "
    "phenotypic finding reported in this document.",
    "Your job is to select, from that list, every term that is an abnormal "
    "phenotypic finding reported in this document.\n\n"
    "Each candidate is listed with two numbers:\n"
    "- `annotated_in`: how many different diseases in the Human Phenotype "
    "Ontology annotation set carry this term. A term annotated in hundreds of "
    "diseases is a common, general finding; a term annotated in a handful is "
    "a specific one.\n"
    "- `typical_share`: how consistently the term is present when the disease is "
    "examined, from 0 (rarely present) to 1 (always present).\n\n"
    "Use them as evidence, not as a filter. A rarely annotated term is still "
    "the right answer when the document states that finding explicitly, and a "
    "commonly annotated term is still wrong when the document does not state "
    "it. Prefer the specific term the document actually supports over a "
    "common term that merely sounds related.")

# Output token budget. A short observation needs a handful of ids; a narrative
# document needs an id per reported finding plus room for the JSON envelope.
SHORT_MAX_TOKENS = 1024
LONG_MAX_TOKENS = 3072