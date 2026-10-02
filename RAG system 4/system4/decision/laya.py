"""Laya: the fast System-1 decision layer around expensive stages (§17–§20).

Laya does not replace retrieval, never invents evidence, and emits only typed,
logged decisions. It sits behind a :class:`DecisionPolicy` so the application
stays independent of any protocol/remote layer.
"""

from __future__ import annotations

from system4.config.settings import Settings
from system4.decision.jev import JevAdapter
from system4.decision.policies import DecisionPolicy, DeterministicPolicy, decide_with_fallback, route_task
from system4.evidence.gate import EvidenceFeatures
from system4.schema import AnswerGate, CompressionLevel, Decision, EvidenceAction


class Laya:
    def __init__(self, policy: DecisionPolicy | None = None, adapter: JevAdapter | None = None) -> None:
        self.policy = policy or DeterministicPolicy()
        self.adapter = adapter

    @property
    def name(self) -> str:
        return getattr(self.adapter, "endpoint", None) and "jev" or self.policy.name

    def route(self, query: str) -> str:
        """Typed task-route decision (§18 ``task_route`` contract)."""
        return route_task(query)

    def decide(
        self, task: str, features: EvidenceFeatures, budget: int, settings: Settings
    ) -> Decision:
        if self.adapter is not None:
            return self.adapter.decide(task, features, budget, settings)
        return decide_with_fallback(task, features, budget, settings, self.policy)


def decide(
    task: str,
    top_score: float,
    evidence_count: int,
    causal_coverage: float,
    contradiction_count: int,
    tokens: int,
    budget: int,
) -> Decision:
    """Compatibility wrapper around the typed feature-based policy (§18, §19)."""
    features = EvidenceFeatures(
        top_rerank_score=top_score,
        conversation_count=1,
        entity_consistency=0.0,
        causal_coverage=causal_coverage,
        outcome_coverage=1.0 if causal_coverage >= 1 else 0.0,
        contradiction_count=contradiction_count,
        evidence_tokens=tokens,
        evidence_count=evidence_count,
    )
    return decide_with_fallback(task, features, budget, Settings())
