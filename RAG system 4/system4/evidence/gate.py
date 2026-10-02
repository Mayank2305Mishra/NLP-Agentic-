"""Structured evidence features consumed by Laya (AGENTS_SYSTEM_4.md §19)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from system4.config.settings import Settings
from system4.evidence.causal_scoring import contradiction_count
from system4.evidence.packer import estimate_tokens
from system4.schema import CHAIN_ACTION_SIDE, CHAIN_CAUSE_SIDE, CHAIN_OUTCOME_SIDE, Candidate, Evidence


@dataclass(frozen=True)
class EvidenceFeatures:
    top_rerank_score: float
    conversation_count: int
    entity_consistency: float
    causal_coverage: float
    outcome_coverage: float
    contradiction_count: int
    evidence_tokens: int
    evidence_count: int

    def as_dict(self) -> dict:
        return asdict(self)


def causal_coverage(evidence: list[Evidence]) -> float:
    """Fraction of the causal chain segments present: cause / action / outcome (§16)."""
    roles = {role for item in evidence for role in item.roles}
    present = 0
    if roles & {r.value for r in CHAIN_CAUSE_SIDE}:
        present += 1
    if roles & {r.value for r in CHAIN_ACTION_SIDE}:
        present += 1
    if roles & {r.value for r in CHAIN_OUTCOME_SIDE}:
        present += 1
    return present / 3


def outcome_coverage(evidence: list[Evidence]) -> float:
    roles = {role for item in evidence for role in item.roles}
    return 1.0 if roles & {r.value for r in CHAIN_OUTCOME_SIDE} else 0.0


def compute_features(
    evidence: list[Evidence],
    ranked: list[Candidate],
    settings: Settings,
) -> EvidenceFeatures:
    top = ranked[0].rerank_score if ranked else 0.0
    return EvidenceFeatures(
        top_rerank_score=round(top, 6),
        conversation_count=len({item.turn.transcript_id for item in evidence}),
        entity_consistency=round(ranked[0].consistency_score, 6) if ranked else 0.0,
        causal_coverage=round(causal_coverage(evidence), 6),
        outcome_coverage=round(outcome_coverage(evidence), 6),
        contradiction_count=contradiction_count(evidence),
        evidence_tokens=sum(estimate_tokens(item.turn.text, settings) for item in evidence),
        evidence_count=len(evidence),
    )
