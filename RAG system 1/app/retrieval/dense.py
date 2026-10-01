"""Dense retriever: embed query -> vector search -> metadata -> EvidenceChunk.

Retrieval knows nothing about generation (AGENTS.md section 3).
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from app.config import Settings
from app.domain.models import Chunk, EvidenceChunk, IndexManifest, RetrievalQuery
from app.indexing.builder import get_query_provider, load_chunks, load_vector_store, verify_manifest
from app.indexing.embeddings import QueryEmbeddingCache
from app.observability.trace import RequestTrace
from app.retrieval.filters import matches


class Retriever(Protocol):
    def retrieve(self, query: RetrievalQuery, trace: RequestTrace | None = None) -> list[EvidenceChunk]: ...


class DenseRetriever:
    """System 1 baseline: traditional dense-vector retrieval only."""

    def __init__(self, settings: Settings):
        self.settings = settings
        manifest: IndexManifest = verify_manifest(
            settings, allow_stale=settings.allow_stale_index)
        self.manifest = manifest
        self.chunks: dict[str, Chunk] = load_chunks(settings)
        self.store = load_vector_store(settings, manifest)
        provider = get_query_provider(settings, manifest)
        self.embed_cache = QueryEmbeddingCache(provider)

    def retrieve(self, query: RetrievalQuery, trace: RequestTrace | None = None) -> list[EvidenceChunk]:
        top_k = query.top_k or self.settings.retrieval_top_k
        # Over-fetch to allow metadata filtering + per-transcript diversity.
        fetch_k = min(len(self.chunks), max(top_k * 4, top_k + 10))
        if trace:
            trace.start("embedding")
        qvec, _hit = self.embed_cache.embed_query(query.text)
        if trace:
            trace.end("embedding")
        if trace:
            trace.start("vector_search")
        hits = self.store.search(np.asarray(qvec), fetch_k)
        if trace:
            trace.end("vector_search")
        out: list[EvidenceChunk] = []
        per_transcript: dict[str, int] = {}
        rank = 0
        for cid, score in hits:
            c = self.chunks.get(cid)
            if c is None:
                continue
            if not matches(c, domain=query.domain, intent=query.intent):
                continue
            n = per_transcript.get(c.transcript_id, 0)
            if n >= self.settings.evidence_max_per_transcript:
                continue
            per_transcript[c.transcript_id] = n + 1
            out.append(EvidenceChunk(
                chunk_id=c.chunk_id, transcript_id=c.transcript_id, text=c.text,
                score=round(float(score), 4), rank=rank,
                metadata={"domain": c.domain, "intent": c.intent,
                          "reason_for_call": c.reason_for_call, "speakers": c.speakers},
                turn_start=c.turn_start, turn_end=c.turn_end,
            ))
            rank += 1
            if len(out) >= top_k:
                break
        return out
