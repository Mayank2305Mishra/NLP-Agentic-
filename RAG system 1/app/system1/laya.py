"""Local Laya decision engine: typed choice/score, batched in one call.

No generation, no network. Deterministic heuristics so the benchmark is
reproducible. All related questions are answered in a single decide() call.
"""
from __future__ import annotations

import re
import time
from typing import Any

from app.system1.schemas import Decision, DecisionResult, QuestionSpec
from app.domain.errors import System1UnavailableError

_SYNTHESIS_HINTS = re.compile(
    r"\b(summariz|synthes|overall|business event|across|trend|pattern|compare|overview)\b", re.IGNORECASE)
_FILTER_HINTS = re.compile(
    r"\b(insurance|travel|hotel|booking|flight|domain|intent|discount|urgency|stress)\b", re.IGNORECASE)
_CAUSAL_HINTS = re.compile(
    r"\b(because|caused|due to|led to|resulted in|triggered|why|reason)\b", re.IGNORECASE)


class LayaDecisionEngine:
    """Production default: local System-1 decision layer."""

    engine_name = "laya-local"

    def __init__(self, *, available: bool = True):
        self._available = available

    @property
    def available(self) -> bool:
        return self._available

    def ping(self) -> bool:
        if not self._available:
            raise System1UnavailableError("Laya engine unavailable (local mode disabled)")
        return True

    def decide(self, state: dict[str, Any], questions: dict[str, QuestionSpec]) -> DecisionResult:
        t0 = time.perf_counter()
        if not self._available:
            raise System1UnavailableError("Laya engine unavailable")
        out: dict[str, Decision] = {}
        for name, spec in questions.items():
            out[name] = self._answer_one(name, spec, state)
        ms = (time.perf_counter() - t0) * 1000.0
        return DecisionResult(decisions=out, latency_ms=round(ms, 2), engine=self.engine_name)

    def _answer_one(self, name: str, spec: QuestionSpec, state: dict[str, Any]) -> Decision:
        query = str(state.get("query_text", ""))
        n_chunks = int(state.get("num_chunks", 0) or 0)
        mean_score = float(state.get("mean_score", 0.0) or 0.0)
        if spec.kind == "score":
            return Decision(question=name, kind="score",
                            value=round(self._sufficiency_score(n_chunks, mean_score), 3),
                            confidence=0.6)
        handler = {
            "task_type": self._task_type,
            "requires_metadata_filter": self._metadata_filter,
            "evidence_sufficient": self._evidence_sufficient,
            "answer_grounded": self._answer_grounded,
        }.get(name)
        if handler is None:
            # Unknown question: abstain to first option with low confidence.
            fallback = spec.options[0] if spec.options else None
            return Decision(question=name, kind=spec.kind, value=fallback, confidence=0.0)
        value, conf = handler(query, n_chunks, mean_score, state)
        return Decision(question=name, kind=spec.kind, value=value, confidence=conf)

    def _task_type(self, query, n_chunks, mean_score, state):
        if _SYNTHESIS_HINTS.search(query):
            return "task2", 0.7
        return "task1", 0.6

    def _metadata_filter(self, query, n_chunks, mean_score, state):
        if state.get("domain") or state.get("intent"):
            return "yes", 0.9
        if _FILTER_HINTS.search(query):
            return "yes", 0.55
        return "no", 0.7

    def _evidence_sufficient(self, query, n_chunks, mean_score, state):
        if n_chunks == 0:
            return "insufficient", 0.95
        if n_chunks >= 3 and mean_score > 0.05:
            return "sufficient", 0.7
        if n_chunks >= 1 and _CAUSAL_HINTS.search(query):
            return "sufficient", 0.5
        return "insufficient", 0.6

    def _sufficiency_score(self, n_chunks: int, mean_score: float) -> float:
        base = min(1.0, n_chunks / 5.0) * 0.6 + min(1.0, max(0.0, mean_score) * 4.0) * 0.4
        return max(0.0, min(1.0, base))

    def _answer_grounded(self, query, n_chunks, mean_score, state):
        n_cited = int(state.get("num_cited", 0) or 0)
        n_valid = int(state.get("num_valid_ids", 0) or 0)
        if n_cited == 0:
            return "unsupported", 0.8
        if n_valid == n_cited and n_cited > 0:
            return "grounded", 0.7
        if n_valid > 0:
            return "partially_grounded", 0.6
        return "unsupported", 0.8
