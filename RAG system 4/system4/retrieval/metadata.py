"""Conversation consistency gate (AGENTS_SYSTEM_4.md §10).

Metadata is supporting evidence, never a replacement for transcript evidence:
the gate only filters obviously incompatible candidates and scores the rest.
A metadata-only match can never become final causal evidence because final
evidence always comes from selected turns of a gated conversation.
"""

from __future__ import annotations

import re
from typing import Callable

from system4.config.settings import Settings
from system4.retrieval.index import tokens
from system4.schema import Candidate, Conversation

_CAPITALIZED = re.compile(r"\b[A-Z][a-zA-Z]+(?:[-'][A-Za-z]+)?\b")
_NUMERIC = re.compile(r"\b\d{2,}\b")

STOP = frozenset(
    "the a an and or is are was were to of in on for with at by from that this "
    "these those it its i you your we our they their he she do does did not no "
    "if then as so about into over under please thanks thank".split()
)


def content_terms(text: str) -> set[str]:
    return {term for term in tokens(text) if term not in STOP and len(term) > 1}


def entity_terms(text: str) -> set[str]:
    """Named-entity-like signals: capitalized words and id-like numerics."""
    found = {match.group(0).lower() for match in _CAPITALIZED.finditer(text)}
    found |= set(_NUMERIC.findall(text))
    return {term for term in found if term not in STOP}


def _overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left)


def consistency(
    query: str, raw_query: str, conversation: Conversation
) -> tuple[float, dict[str, float]]:
    """Score how consistent a conversation is with the query (§10 signals)."""
    query_terms = content_terms(query)
    body_terms: set[str] = set()
    body_entities: set[str] = set()
    for turn in conversation.turns:
        body_terms |= content_terms(turn.text)
        body_entities |= entity_terms(turn.text)

    domain_compat = _overlap(query_terms, content_terms(conversation.domain or ""))
    intent_compat = _overlap(query_terms, content_terms(conversation.intent or ""))
    reason_terms = content_terms(conversation.reason_for_call or "")
    reason_similarity = _overlap(query_terms, reason_terms)
    query_entities = entity_terms(raw_query)
    entity_overlap = _overlap(
        query_entities, body_entities | entity_terms(conversation.reason_for_call or "")
    )
    topic_coverage = _overlap(query_terms, body_terms)

    breakdown = {
        "domain_compatibility": domain_compat,
        "intent_compatibility": intent_compat,
        "reason_for_call_similarity": reason_similarity,
        "entity_overlap": entity_overlap,
        "query_topic_coverage": topic_coverage,
    }
    score = (
        0.20 * domain_compat
        + 0.15 * intent_compat
        + 0.25 * reason_similarity
        + 0.15 * entity_overlap
        + 0.25 * topic_coverage
    )
    return round(min(1.0, score), 6), breakdown


def apply_gate(
    candidates: list[Candidate],
    conversations: dict[str, Conversation],
    query: str,
    raw_query: str,
    settings: Settings,
) -> list[Candidate]:
    """Filter inconsistent candidates while always keeping a floor (§10).

    The floor prevents a metadata-only mismatch from emptying the pool before
    the reranker sees transcript evidence.
    """
    for candidate in candidates:
        conversation = conversations.get(candidate.transcript_id)
        if conversation is None:
            candidate.consistency_score = 0.0
            continue
        score, breakdown = consistency(query, raw_query, conversation)
        candidate.consistency_score = score
        candidate.consistency_breakdown = breakdown
        candidate.metadata_score = max(
            breakdown["domain_compatibility"],
            breakdown["intent_compatibility"],
            breakdown["reason_for_call_similarity"],
        )
    kept = [
        candidate
        for candidate in candidates
        if candidate.consistency_score >= settings.min_consistency_score
    ]
    if len(kept) < settings.consistency_keep_floor:
        by_consistency = sorted(
            (c for c in candidates if c not in kept),
            key=lambda c: (-c.consistency_score, c.transcript_id),
        )
        kept.extend(by_consistency[: settings.consistency_keep_floor - len(kept)])
    # Preserve broad-retrieval rank order among survivors.
    order = {candidate.transcript_id: index for index, candidate in enumerate(candidates)}
    return sorted(kept, key=lambda c: order[c.transcript_id])
