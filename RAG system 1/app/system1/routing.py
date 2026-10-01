"""Jev-compatible adapter: same state+questions->decisions semantics, local only.

Jev compatibility is an interface (AGENTS.md section 10): we implement the
request/response shape locally through Laya and never call a hosted Jev
endpoint with raw conversations.
"""
from __future__ import annotations

from typing import Any, Protocol

from app.system1.laya import LayaDecisionEngine
from app.system1.schemas import DecisionResult, QuestionSpec


class System1DecisionEngine(Protocol):
    def decide(self, state: object, questions: dict) -> DecisionResult: ...


class JevCompatibleDecisionEngine:
    """Adapter exposing Jev-style {state, questions} -> {decisions} over Laya."""

    engine_name = "jev-compatible-local"

    def __init__(self, inner: LayaDecisionEngine | None = None):
        self._inner = inner or LayaDecisionEngine()

    def decide(self, state: object, questions: dict) -> DecisionResult:
        state_dict = dict(state) if isinstance(state, dict) else {"state": state}
        parsed: dict[str, QuestionSpec] = {}
        for k, v in (questions or {}).items():
            if isinstance(v, QuestionSpec):
                parsed[k] = v
            elif isinstance(v, dict):
                parsed[k] = QuestionSpec.model_validate(v)
            elif isinstance(v, (list, tuple)):
                parsed[k] = QuestionSpec(kind="choice", options=list(v))
            else:
                parsed[k] = QuestionSpec(kind="choice", options=[str(v)])
        result = self._inner.decide(state_dict, parsed)
        result.engine = self.engine_name
        return result

    def to_jev_request(self, state: Any, questions: dict) -> dict:
        return {"state": state, "questions": questions, "engine": self.engine_name}

    @staticmethod
    def from_jev_response(payload: dict) -> DecisionResult:
        return DecisionResult.model_validate(payload)


def get_engine(mode: str = "local", *, enabled: bool = True) -> System1DecisionEngine:
    if mode == "jev-compatible":
        return JevCompatibleDecisionEngine(LayaDecisionEngine(available=enabled))
    return LayaDecisionEngine(available=enabled)
