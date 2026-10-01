"""Deterministic metadata filters (allowed: small, no reranker/BM25)."""
from __future__ import annotations

from app.domain.models import Chunk


def apply_metadata_filter(chunks: list[Chunk], *, domain: str | None = None,
                          intent: str | None = None) -> list[Chunk]:
    out = chunks
    if domain:
        out = [c for c in out if (c.domain or "").lower() == domain.lower()]
    if intent:
        out = [c for c in out if (c.intent or "").lower() == intent.lower()]
    return out


def matches(chunk: Chunk, *, domain: str | None, intent: str | None) -> bool:
    if domain and (chunk.domain or "").lower() != domain.lower():
        return False
    if intent and (chunk.intent or "").lower() != intent.lower():
        return False
    return True
