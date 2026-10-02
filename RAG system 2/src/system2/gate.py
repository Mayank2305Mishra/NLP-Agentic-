from __future__ import annotations

from .config import Config
from .decisions import DecisionLog
from .schema import EvidenceConversation, GateDecision

SUFFICIENT = "SUFFICIENT"
EXPAND = "EXPAND"
INSUFFICIENT = "INSUFFICIENT"


def evaluate_evidence_gate(
    candidates: list[EvidenceConversation],
    config: Config,
    decisions: DecisionLog,
    causal_gap: bool,
) -> GateDecision:
    if not candidates:
        decision = decisions.record(INSUFFICIENT, "no conversation candidates retrieved", 0.0)
        return GateDecision(INSUFFICIENT, decision.reason, 0.0)

    best = candidates[0]
    best_turn_score = max((t.score for t in best.turns), default=0.0)
    combined = 0.5 * best.score + 0.5 * best_turn_score

    if best.score < config.evidence.min_conversation_score and best_turn_score < config.evidence.min_score:
        decision = decisions.record(
            INSUFFICIENT,
            "top candidate below score thresholds",
            round(combined, 4),
            transcript_id=best.transcript_id,
            conversation_score=round(best.score, 4),
            best_turn_score=round(best_turn_score, 4),
        )
        return GateDecision(
            INSUFFICIENT,
            decision.reason,
            round(combined, 4),
            transcript_id=best.transcript_id,
            detail=decision.detail,
        )

    if combined < config.evidence.min_score:
        decision = decisions.record(
            INSUFFICIENT,
            "combined evidence score below minimum",
            round(combined, 4),
            transcript_id=best.transcript_id,
            conversation_score=round(best.score, 4),
            best_turn_score=round(best_turn_score, 4),
        )
        return GateDecision(
            INSUFFICIENT,
            decision.reason,
            round(combined, 4),
            transcript_id=best.transcript_id,
            detail=decision.detail,
        )

    if causal_gap and config.evidence.expand_causal_gap:
        decision = decisions.record(
            EXPAND,
            "lexical hit relevant but causal chain incomplete",
            round(combined, 4),
            transcript_id=best.transcript_id,
        )
        return GateDecision(
            EXPAND,
            decision.reason,
            round(combined, 4),
            transcript_id=best.transcript_id,
            detail=decision.detail,
        )

    decision = decisions.record(
        SUFFICIENT,
        "conversation and causal evidence present",
        round(combined, 4),
        transcript_id=best.transcript_id,
        conversation_score=round(best.score, 4),
        best_turn_score=round(best_turn_score, 4),
        candidates=len(candidates),
    )
    return GateDecision(
        SUFFICIENT,
        decision.reason,
        round(combined, 4),
        transcript_id=best.transcript_id,
        detail=decision.detail,
    )


def causal_chain_incomplete(evidence: list[EvidenceConversation]) -> bool:
    if not evidence:
        return True
    text = " ".join(t.text_normalized for c in evidence for t in c.turns)
    from .cues import ACTION_CUES, EFFECT_CUES, cue_hits

    has_outcome = bool(cue_hits(text, EFFECT_CUES))
    has_action = bool(cue_hits(text, ACTION_CUES))
    return not (has_outcome and has_action)
