"""Relevant turn selection and neighbor expansion (AGENTS_SYSTEM_4.md §13, §14).

Selected turns retain transcript id, turn number, speaker, exact text,
selection score, and selection reason. Post-call artifacts are excluded by
default (§24). Expansion is a configurable local window, never a full
conversation dump (§14).
"""

from __future__ import annotations

from system4.config.settings import Settings
from system4.evidence.causal_scoring import detect_roles
from system4.retrieval.index import tokens
from system4.retrieval.metadata import content_terms
from system4.schema import CAUSAL_ROLES, CausalRole, Conversation, Evidence

#: Spec §13 selection reasons, derived from roles.
_REASON_LABELS = {
    "cause": "cause evidence",
    "trigger": "cause evidence",
    "problem": "cause evidence",
    "customer_motivation": "customer motivation",
    "agent_action": "agent action",
    "mechanism": "mechanism evidence",
    "effect": "effect evidence",
    "outcome": "outcome",
    "uncertainty": "uncertainty",
    "contradiction": "contradiction",
    "clarification": "clarification",
    "query_match": "query match",
    "context": "context",
}


def select(query: str, conversation: Conversation, settings: Settings) -> list[Evidence]:
    query_terms = set(tokens(query))
    role_terms = content_terms(query)
    selected: list[Evidence] = []
    for turn in conversation.turns:
        if turn.post_call and not settings.include_post_call:
            continue
        detected = detect_roles(turn, role_terms, settings)
        roles = [role for role, _ in detected]
        turn_terms = set(tokens(turn.text))
        overlap = len(query_terms & turn_terms) / max(1, len(query_terms))
        causal_hits = len([role for role in roles if role in {r.value for r in CAUSAL_ROLES}])
        score = overlap + 0.12 * causal_hits
        if score <= settings.selection_min_score and causal_hits == 0:
            continue
        labels = [_REASON_LABELS.get(role, role) for role in roles]
        selected.append(
            Evidence(
                turn=turn,
                score=round(score, 6),
                roles=roles,
                reason=", ".join(dict.fromkeys(labels)),
                retrieval_stage="turn_selection",
            )
        )
    selected.sort(key=lambda item: (-item.score, item.turn.turn_id))
    return selected[: settings.turns_per_conversation]


def expand(
    evidence: list[Evidence], conversation: Conversation, settings: Settings
) -> list[Evidence]:
    """Add the local causal neighborhood around each selected turn (§14)."""
    by_id = {item.turn.turn_id: item for item in evidence}
    turns = {turn.turn_id: turn for turn in conversation.turns}
    window = max(0, settings.neighbor_window)
    for item in list(evidence):
        for turn_id in range(
            max(1, item.turn.turn_id - window), item.turn.turn_id + window + 1
        ):
            if turn_id in by_id:
                continue
            turn = turns.get(turn_id)
            if turn is None or (turn.post_call and not settings.include_post_call):
                continue
            by_id[turn_id] = Evidence(
                turn=turn,
                score=round(item.score * 0.7, 6),
                roles=[CausalRole.CAUSAL_NEIGHBOR.value],
                reason=f"adjacent causal context (±{window} turns of {item.turn.turn_id})",
                retrieval_stage="expansion",
            )
    return sorted(by_id.values(), key=lambda item: (item.turn.transcript_id, item.turn.turn_id))
