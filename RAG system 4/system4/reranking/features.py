"""Deterministic reranker features (AGENTS_SYSTEM_4.md §12, §3.2).

Features use ABSOLUTE scales, not cross-candidate max-normalization: making
the top candidate always look perfect on every feature is exactly the
"similarity is confidence" trap §3.2 forbids. Only the rank-fusion feature
is normalized (ranks are our own combined signal, no cross-system mixing).

``query_relevance`` combines only query-linked signals; when it is exactly
zero the candidate is scored zero regardless of how causal or dense the
conversation is — an unrelated conversation can never ride in on its own
causal vocabulary.
"""

from __future__ import annotations

import math

from system4.config.settings import Settings
from system4.retrieval.index import tokens
from system4.retrieval.metadata import content_terms, entity_terms
from system4.schema import Candidate, Conversation

FEATURE_NAMES = (
    "semantic_relevance",
    "lexical_relevance",
    "entity_match",
    "metadata_match",
    "causal_density",
    "action_density",
    "outcome_density",
    "fusion",
    "query_coverage",
    "phrase_overlap",
    "query_relevance",
)

_ACTION_MARKERS = ("i can", "i will", "we can", "let me", "i'll", "approved", "process")
_OUTCOME_MARKERS = ("resolved", "fix", "refund", "cancelled", "canceled", "confirmed", "closed")


def _density(text: str, terms: tuple[str, ...], turns: int) -> float:
    lowered = text.lower()
    hits = sum(lowered.count(term) for term in terms)
    return min(1.0, math.log1p(hits) / math.log1p(max(4, turns * 2)))


def _bigrams(text: str) -> set[str]:
    words = [w for w in tokens(text) if len(w) > 2]
    return {f"{a} {b}" for a, b in zip(words, words[1:])}


def _squash(value: float, scale: float = 6.0) -> float:
    """Absolute monotone map to [0, 1): 0 → 0, scale → 0.63."""
    if value <= 0:
        return 0.0
    return 1.0 - math.exp(-value / scale)


def _max_norm(values: list[float]) -> list[float]:
    peak = max(values, default=0.0)
    if peak <= 0:
        return [0.0 for _ in values]
    return [min(1.0, value / peak) for value in values]


def extract_features(
    query: str,
    raw_query: str,
    candidates: list[Candidate],
    conversations: dict[str, Conversation],
    settings: Settings,
) -> list[dict[str, float]]:
    """Compute one absolute-scale feature dict per candidate (§12 feature family)."""
    query_terms = content_terms(query)
    query_bigrams = _bigrams(query)
    query_entities = entity_terms(raw_query)

    rows: list[dict[str, float]] = []
    for candidate in candidates:
        conversation = conversations.get(candidate.transcript_id)
        if conversation is None:
            rows.append({name: 0.0 for name in FEATURE_NAMES})
            continue
        body = " ".join(turn.text for turn in conversation.turns)
        body_terms = content_terms(body)
        metadata = " ".join(
            part or ""
            for part in (conversation.domain, conversation.intent, conversation.reason_for_call)
        )
        coverage = len(query_terms & body_terms) / max(1, len(query_terms))
        phrase = len(query_bigrams & _bigrams(body)) / max(1, len(query_bigrams))
        entity_hits = len(query_entities & (entity_terms(body) | entity_terms(metadata)))
        entities = entity_hits / max(1, len(query_entities))
        rows.append(
            {
                # absolute, cross-system-safe scales:
                "semantic_relevance": float(candidate.dense_score or 0.0),
                "lexical_relevance": _squash(float(candidate.lexical_score or 0.0)),
                "entity_match": entities,
                "metadata_match": candidate.consistency_score,
                "query_coverage": coverage,
                "phrase_overlap": phrase,
                "causal_density": _density(body, settings.causal_terms, len(conversation.turns)),
                "action_density": _density(body, _ACTION_MARKERS, len(conversation.turns)),
                "outcome_density": _density(body, _OUTCOME_MARKERS, len(conversation.turns)),
                # rank fusion normalized across the candidate set (ranks only):
                "fusion": candidate.fusion_score,
            }
        )

    for row, value in zip(rows, _max_norm([row["fusion"] for row in rows])):
        row["fusion"] = round(value, 6)
    for row in rows:
        row["query_relevance"] = round(
            max(
                row["semantic_relevance"],
                row["lexical_relevance"],
                row["metadata_match"],
                row["query_coverage"],
                row["entity_match"],
            ),
            6,
        )
        for name in FEATURE_NAMES:
            row[name] = round(min(1.0, max(0.0, row[name])), 6)
    return rows


def weighted_score(features: dict[str, float], settings: Settings) -> float:
    """Configurable weighted rerank score (§12); weights are benchmark-calibrated.

    Zero query relevance means zero score: causal density of an unrelated
    conversation must never substitute for query relevance (§3.2).
    """
    if features.get("query_relevance", 0.0) <= 0.0:
        return 0.0
    score = (
        settings.w_semantic * features.get("semantic_relevance", 0.0)
        + settings.w_lexical * features.get("lexical_relevance", 0.0)
        + settings.w_entity * features.get("entity_match", 0.0)
        + settings.w_metadata * features.get("metadata_match", 0.0)
        + settings.w_causal_density * features.get("causal_density", 0.0)
        + settings.w_action_density * features.get("action_density", 0.0)
        + settings.w_outcome_density * features.get("outcome_density", 0.0)
        + settings.w_fusion * features.get("fusion", 0.0)
    )
    return round(min(1.0, score), 6)
