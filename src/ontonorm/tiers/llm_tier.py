"""LLM tier: evidence-anchored candidate generation via chat API.

The LLM receives the clinical observation and returns, in JSON:

    {
      "findings": [
        {"span": "high arched palate", "start": 7, "end": 25,
         "hpo": ["HP:0000218", "HP:0000153"],
         "polarity": "positive",
         "rationale": "..."}
      ]
    }

The evidence anchor (span) grounds every emitted HPO candidate, which
is what allows the ontology constraint tier to validate and re-rank
against the text. Provider-agnostic: any OpenAI-compatible chat API
works (DeepSeek, OpenAI, local vLLM endpoints).
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

from ..knowledge.hpo import HPO
from ..tiers.ontology_tier import Candidate as OntoCandidate

_EXTRACTION_SYSTEM = """You are a clinical phenotype coding assistant.
Given a short clinical observation (from a dysmorphology physical exam),
list every key finding and map it to the most specific applicable Human
Phenotype Ontology (HPO) term.

Rules:
- Quote the exact text span of each finding with its character offsets.
- Provide exactly ONE HPO ID per finding, the most specific one.
- Only use HPO IDs you are confident exist. If unsure, still propose the
  best guess; the system will validate against the ontology.
- Set polarity to "positive" for abnormal findings, "excluded" for
  normal/negated findings.
- Be concise. No rationale, no extra fields.
- Respond with JSON only, in this exact schema:
{"findings": [{"span": string, "start": int, "end": int,
               "hpo": [string], "polarity": string}]}"""


@dataclass
class LlmFinding:
    span: str
    start: int
    end: int
    hpo_ids: List[str]
    polarity: str
    rationale: str = ""


class FatalApiError(RuntimeError):
    """Non-retryable API error (bad key, exhausted quota, no balance)."""


@dataclass
class LlmResult:
    observation_id: Optional[str]
    text: str
    findings: List[LlmFinding] = field(default_factory=list)
    raw: str = ""
    error: Optional[str] = None


class LlmClient:
    """Minimal OpenAI-compatible chat client (no SDK dependency)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-chat",
        temperature: float = 0.3,
        timeout: int = 90,
        key_env: str = "DEEPSEEK_API_KEY",
        thinking_budget: Optional[int] = None,
    ) -> None:
        self.api_key = api_key or os.environ.get(key_env, "")
        self.key_env = key_env
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.thinking_budget = thinking_budget

    def chat(self, user: str, max_tokens: int = 2048,
             system: Optional[str] = None) -> str:
        if not self.api_key:
            raise FatalApiError(
                f"No API key; set {self.key_env} or pass api_key=.")
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system or _EXTRACTION_SYSTEM},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        if self.thinking_budget is not None:
            # GLM-style reasoning models: bound the hidden reasoning so the
            # visible content is not starved by the chain-of-thought.
            payload["thinking"] = {
                "type": "enabled",
                "thinking_budget": self.thinking_budget,
            }
        try:
            resp = requests.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else 0
            if code in (401, 402, 403):
                body = (e.response.text or "")[:200] if e.response is not None else ""
                raise FatalApiError(
                    f"HTTP {code} from {url} -- API key rejected or quota/balance "
                    f"exhausted. Not retryable. {body}"
                ) from e
            raise
        data = resp.json()
        msg = data["choices"][0]["message"]
        content = msg.get("content") or ""
        # GLM-style reasoning models can return an empty ``content`` when the
        # whole budget went to ``reasoning_content`` (common at temperature 0).
        # Surfacing the raw reasoning text keeps a JSON block recoverable
        # upstream instead of silently returning "".
        if not content.strip() and msg.get("reasoning_content"):
            content = msg["reasoning_content"] or ""
        return content

    def preflight(self) -> None:
        """Fail fast (before a long batch) if the endpoint is unusable."""
        self.chat("Reply with JSON only: {\"ok\":1}", max_tokens=16)


class LlmTier:
    """Runs the LLM extraction over observations and yields candidates.

    :meth:`run` returns the parsed findings; :meth:`to_candidates`
    converts findings into OntoNorm ontology-tier candidates.
    """

    def __init__(
        self,
        client: LlmClient,
        hpo: HPO,
        max_findings: int = 8,
        retries: int = 2,
        fewshot=None,
        long_text: bool = False,
    ) -> None:
        self.client = client
        self.hpo = hpo
        self.max_findings = max_findings
        self.retries = retries
        self.fewshot = fewshot
        # Narrative documents carry far more findings per document than a
        # short examination observation does, so the per-response cap and the
        # token budget both have to scale. Only the corpus wording and the
        # budget change; the extraction task is identical.
        self.long_text = long_text
        # 8 findings is the right cap for a one-line observation and a silent
        # truncation on a case report carrying 16, so the cap is lifted rather
        # than left to bite the tail off every document.
        if long_text and max_findings == 8:
            max_findings = 64
        # Set after a non-retryable failure (bad key / no balance) so that a
        # long batch aborts immediately instead of retrying every case.
        self.fatal_error: Optional[str] = None

    def _prompt(self, text: str) -> str:
        if self.long_text:
            # Few-shot exemplars come from the short-observation corpus, so
            # they are dropped rather than mixed across corpora.
            return (
                "Clinical document:\n"
                f'"{text}"\n\n'
                "This document reports MANY abnormal findings, often 10 to 20. "
                "List every one of them and map each to the most specific "
                "applicable Human Phenotype Ontology (HPO) term. Do not stop "
                "after the first few. Quote the exact text span of each "
                "finding with its character offsets, and report only findings "
                "this document actually states. Return JSON only."
            )
        head = ""
        if self.fewshot is not None:
            block = self.fewshot.prompt_block(text)
            if block:
                head = block + "\n\n"
        return (
            head
            + "Clinical observation:\n"
            f'"{text}"\n\n'
            "Extract all key findings and map them to HPO terms. "
            "Return JSON only."
        )

    def run(self, observation_id: Optional[str], text: str) -> LlmResult:
        if self.fatal_error:
            return LlmResult(observation_id, text, error=self.fatal_error)
        last_err: Optional[Exception] = None
        max_tokens = 4096 if self.long_text else 2048
        for attempt in range(self.retries + 1):
            try:
                raw = self.client.chat(self._prompt(text),
                                       max_tokens=max_tokens)
                parsed = self._parse(raw)
                if parsed is None:
                    raise ValueError("unparseable LLM JSON output")
                return LlmResult(observation_id, text, parsed, raw)
            except FatalApiError as e:
                # Auth / balance / quota: retrying cannot help.
                self.fatal_error = str(e)
                return LlmResult(observation_id, text, error=self.fatal_error)
            except Exception as e:  # retry transient API / parse failures
                last_err = e
                time.sleep(1.0 * (attempt + 1))
        return LlmResult(observation_id, text, error=str(last_err))

    def _parse(self, raw: str) -> Optional[List[LlmFinding]]:
        # Tolerate fenced or backticked JSON.
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
        findings = data.get("findings")
        if not isinstance(findings, list):
            # Some models nest under "results" or return a dict
            findings = data.get("results") or data.get("items") or []
        out: List[LlmFinding] = []
        for f in (findings or [])[: self.max_findings]:
            try:
                # Accept multiple field names across model behaviors.
                hpo_ids = f.get("hpo") or f.get("hpo_id") or f.get("hpo_ids") or f.get("id")
                if isinstance(hpo_ids, str):
                    hpo_ids = [hpo_ids]
                if not isinstance(hpo_ids, list):
                    hpo_ids = []
                span = f.get("span") or f.get("mention") or f.get("term") or ""
                start = f.get("start", -1)
                end = f.get("end", -1)
                try:
                    start = int(start)
                except (TypeError, ValueError):
                    start = -1
                try:
                    end = int(end)
                except (TypeError, ValueError):
                    end = -1
                out.append(
                    LlmFinding(
                        span=str(span),
                        start=start,
                        end=end,
                        hpo_ids=hpo_ids,
                        polarity=str(f.get("polarity", "positive")) or "positive",
                        rationale=str(f.get("rationale", "")),
                    )
                )
            except (ValueError, TypeError):
                continue
        return out

    def to_candidates(self, findings: List[LlmFinding]) -> List[OntoCandidate]:
        cands: List[OntoCandidate] = []
        for f in findings:
            pol = "positive" if f.polarity == "positive" else "excluded"
            score = 0.8  # LLM baseline confidence; more specific ordering pushed
            # Candidates ordered most-specific-first get degrading scores.
            for rank, hpid in enumerate(f.hpo_ids):
                cands.append(
                    OntoCandidate(
                        hpid=hpid,
                        mention=f.span,
                        span=(f.start, f.end) if f.start >= 0 else (0, 0),
                        origin="llm",
                        score=max(0.5, 0.8 - rank * 0.15),
                        polarity=pol,
                    )
                )
        return cands