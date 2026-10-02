"""Deterministic Laya decision policies (AGENTS_SYSTEM_4.md §17, §19, §44).

Thresholds are benchmark-calibrated via :class:`Settings`. The policy is a
pure function of typed evidence features — no remote dependency, no invented
evidence, and every decision is typed and logged (§18, §40).
"""

from __future__ import annotations

import re
from typing import Protocol

from system4.config.settings import Settings
from system4.decision.schemas import task_route_of
from system4.evidence.gate import EvidenceFeatures
from system4.schema import AnswerGate, CompressionLevel, Decision, EvidenceAction


class DecisionPolicy(Protocol):
    name: str

    def decide(
        self, task: str, features: EvidenceFeatures, budget: int, settings: Settings
    ) -> Decision: ...


_TASK2_HINTS = re.compile(
    r"\b(summar(y|ise|ize)|overall|entire conversation|end to end|chronolog|"
    r"what happened|whole interaction|as a whole|recap)\b",
    re.I,
)
_TASK1_HINTS = re.compile(r"\b(why|cause|caused|reason|because|what led to|root cause)\b", re.I)


def route_task(query: str) -> str:
    """Deterministic task routing for ``task='auto'`` (§4 ``task_route`` contract)."""
    if _TASK2_HINTS.search(query):
        return "task2"
    if _TASK1_HINTS.search(query):
        return "task1"
    return "task1"


class DeterministicPolicy:
    """The default local-first policy (§19 example policy, made explicit)."""

    name = "deterministic"

    def decide(
        self, task: str, features: EvidenceFeatures, budget: int, settings: Settings
    ) -> Decision:
        route = task_route_of(task)
        if (
            features.top_rerank_score <= 0
            or features.evidence_count == 0
            or features.evidence_tokens == 0
        ):
            return Decision(
                route, EvidenceAction.REJECT, CompressionLevel.NONE, AnswerGate.ABSTAIN,
                "no relevant evidence",
            )
        if features.contradiction_count > 0:
            return Decision(
                route, EvidenceAction.EXPAND, CompressionLevel.LIGHT,
                AnswerGate.ANSWER_WITH_UNCERTAINTY,
                "contradictory evidence requires context",
            )
        if features.causal_coverage < settings.laya_causal_coverage_expand:
            return Decision(
                route, EvidenceAction.EXPAND, CompressionLevel.LIGHT,
                AnswerGate.ANSWER_WITH_UNCERTAINTY,
                "causal chain incomplete",
            )
        level = (
            CompressionLevel.AGGRESSIVE
            if features.evidence_tokens > budget
            else CompressionLevel(settings.default_compression)
        )
        if features.outcome_coverage <= 0:
            return Decision(
                route, EvidenceAction.ACCEPT, level, AnswerGate.ANSWER_WITH_UNCERTAINTY,
                "evidence lacks an explicit outcome",
            )
        return Decision(
            route, EvidenceAction.ACCEPT, level, AnswerGate.ANSWER,
            "sufficient consistent evidence",
        )


def decide_with_fallback(
    task: str,
    features: EvidenceFeatures,
    budget: int,
    settings: Settings,
    policy: DecisionPolicy | None = None,
) -> Decision:
    """§43: any policy failure falls back to the deterministic policy."""
    try:
        return (policy or DeterministicPolicy()).decide(task, features, budget, settings)
    except Exception:
        try:
            return DeterministicPolicy().decide(task, features, budget, settings)
        except Exception:
            return Decision(
                task_route_of(task), EvidenceAction.REJECT, CompressionLevel.NONE,
                AnswerGate.ABSTAIN, "decision layer failure — failing closed",
                policy="fail-closed",
            )
