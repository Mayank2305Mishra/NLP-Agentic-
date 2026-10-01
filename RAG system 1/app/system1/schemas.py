"""Typed System-1 decision schemas (Laya-style choice/score, Jev-compatible)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class QuestionSpec(BaseModel):
    kind: Literal["choice", "score", "noul"] = "choice"
    options: list[str] = Field(default_factory=list)
    prompt: str = ""


class Decision(BaseModel):
    question: str
    kind: str = "choice"
    value: str | float | None = None
    confidence: float = 0.0


class DecisionResult(BaseModel):
    decisions: dict[str, Decision] = Field(default_factory=dict)
    latency_ms: float = 0.0
    engine: str = "laya-local"
    degraded: bool = False

    def get(self, name: str, default: Any = None) -> Any:
        d = self.decisions.get(name)
        return d.value if d else default


# Canonical batched questions for the pipeline (one forward pass).
def routing_questions() -> dict[str, QuestionSpec]:
    return {
        "task_type": QuestionSpec(
            kind="choice", options=["task1", "task2"],
            prompt="Is the request causal rationale extraction (task1) or synthesis (task2)?",
        ),
        "requires_metadata_filter": QuestionSpec(
            kind="choice", options=["yes", "no"],
            prompt="Does the query name a domain/intent that should gate retrieval?",
        ),
    }


def evidence_questions() -> dict[str, QuestionSpec]:
    return {
        "evidence_sufficient": QuestionSpec(
            kind="choice", options=["sufficient", "insufficient"],
            prompt="Is the retrieved evidence sufficient to answer without fabrication?",
        ),
        "evidence_quality_score": QuestionSpec(
            kind="score", options=[],
            prompt="Rate evidence sufficiency 0..1.",
        ),
    }


def answer_questions() -> dict[str, QuestionSpec]:
    return {
        "answer_grounded": QuestionSpec(
            kind="choice", options=["grounded", "partially_grounded", "unsupported"],
            prompt="Is every claim traceable to cited evidence?",
        ),
    }
