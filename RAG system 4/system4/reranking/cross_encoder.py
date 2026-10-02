"""Optional cross-encoder reranker (AGENTS_SYSTEM_4.md §11).

Loaded lazily so the local-first core never requires heavy dependencies.
If ``sentence-transformers`` is unavailable the factory falls back to the
deterministic feature reranker.
"""

from __future__ import annotations

import importlib.util

from system4.config.settings import Settings
from system4.schema import Candidate, Conversation

CROSS_ENCODER_VERSION = "cross-encoder-v1"


def available() -> bool:
    return importlib.util.find_spec("sentence_transformers") is not None


class CrossEncoderReranker:
    name = "cross_encoder"
    version = CROSS_ENCODER_VERSION

    def __init__(self, settings: Settings) -> None:
        if not available():
            raise RuntimeError("sentence-transformers is not installed")
        from sentence_transformers import CrossEncoder  # local import: optional dep

        self._model = CrossEncoder(settings.cross_encoder_model)
        self._max_chars = settings.cross_encoder_max_chars

    def rerank(self, query, raw_query, candidates, conversations, settings):
        if not candidates:
            return []
        pairs = []
        keep: list[Candidate] = []
        for candidate in candidates:
            conversation = conversations.get(candidate.transcript_id)
            if conversation is None:
                continue
            metadata = " ".join(
                part or ""
                for part in (
                    conversation.domain,
                    conversation.intent,
                    conversation.reason_for_call,
                )
            )
            body = " ".join(turn.text for turn in conversation.turns)
            pairs.append((query, f"{metadata} {body}"[: self._max_chars]))
            keep.append(candidate)
        if not pairs:
            return []
        scores = self._model.predict(pairs)
        peak = max(float(s) for s in scores) or 1.0
        floor = min(float(s) for s in scores)
        span = (peak - floor) or 1.0
        for candidate, score in zip(keep, scores):
            candidate.rerank_score = round((float(score) - floor) / span, 6)
        ordered = sorted(keep, key=lambda c: (-c.rerank_score, c.transcript_id))
        return ordered[: settings.rerank_k]
