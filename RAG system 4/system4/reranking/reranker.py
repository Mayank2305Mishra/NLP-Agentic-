"""Swappable reranker abstraction (AGENTS_SYSTEM_4.md §11, §12, §36).

Implementations: ``none`` (fusion passthrough), ``feature`` (deterministic
default), ``cross_encoder`` (optional learned reranker). The pipeline depends
only on the :class:`Reranker` protocol, so implementations are benchmark-
swappable without touching retrieval code.
"""

from __future__ import annotations

from typing import Protocol

from system4.config.settings import Settings
from system4.reranking.features import extract_features, weighted_score
from system4.retrieval.metadata import content_terms
from system4.schema import Candidate, Conversation

RERANKER_VERSION = "feature-v2"


class Reranker(Protocol):
    name: str
    version: str

    def rerank(
        self,
        query: str,
        raw_query: str,
        candidates: list[Candidate],
        conversations: dict[str, Conversation],
        settings: Settings,
    ) -> list[Candidate]: ...


class PassthroughReranker:
    """No-op reranker: orders by normalized fusion score (benchmark baseline §36).

    Still applies the query-relevance gate (§3.2): a candidate with zero query
    relevance scores zero no matter how well it fused.
    """

    name = "none"
    version = "passthrough-v1"

    def rerank(self, query, raw_query, candidates, conversations, settings):
        peak = max((c.fusion_score for c in candidates), default=0.0) or 1.0
        query_terms = content_terms(query)
        for candidate in candidates:
            conversation = conversations.get(candidate.transcript_id)
            body_terms: set[str] = set()
            if conversation is not None:
                for turn in conversation.turns:
                    body_terms |= content_terms(turn.text)
            relevance = max(
                float(candidate.dense_score or 0.0),
                float(candidate.lexical_score or 0.0),
                candidate.consistency_score,
                len(query_terms & body_terms) / max(1, len(query_terms)),
            )
            candidate.rerank_score = (
                round(min(1.0, candidate.fusion_score / peak), 6) if relevance > 0 else 0.0
            )
        ordered = sorted(candidates, key=lambda c: (-c.rerank_score, c.transcript_id))
        return ordered[: settings.rerank_k]


class FeatureReranker:
    """Deterministic weighted-feature reranker (§12)."""

    name = "feature"
    version = RERANKER_VERSION

    def rerank(self, query, raw_query, candidates, conversations, settings):
        if not candidates:
            return []
        feature_rows = extract_features(query, raw_query, candidates, conversations, settings)
        for candidate, row in zip(candidates, feature_rows):
            candidate.features = row
            candidate.rerank_score = weighted_score(row, settings)
        ordered = sorted(candidates, key=lambda c: (-c.rerank_score, c.transcript_id))
        return ordered[: settings.rerank_k]


def make_reranker(settings: Settings):
    """Factory with graceful degradation: cross-encoder → feature (§11, §43)."""
    kind = (settings.reranker or "feature").lower()
    if kind == "none":
        return PassthroughReranker()
    if kind == "cross_encoder":
        try:
            from system4.reranking.cross_encoder import CrossEncoderReranker

            return CrossEncoderReranker(settings)
        except Exception:
            # §43: reranker failure falls back to deterministic feature ranking.
            return FeatureReranker()
    return FeatureReranker()


def rerank(query, candidates, conversations, settings, raw_query=None):
    """Backwards-compatible helper using the deterministic feature reranker."""
    return FeatureReranker().rerank(
        query, raw_query if raw_query is not None else query, candidates, conversations, settings
    )
