"""Evidence packing: dedupe, diversity, context-budget protection.

- Drop exact-duplicate chunk texts (keep best rank).
- Cap chunks per transcript (diversity) — enforced in retriever; re-checked here.
- Fit within max_context_tokens by dropping the lowest-ranked chunk first,
  only at chunk boundaries. Record truncation.
- If nothing fits -> ContextBudgetError (insufficient_context), never an
  unsupported generation request.
"""
from __future__ import annotations

import hashlib

from app.config import Settings
from app.domain.errors import ContextBudgetError
from app.domain.models import EvidenceChunk, EvidencePack
from app.indexing.chunker import estimate_tokens


def _fingerprint(text: str) -> str:
    norm = " ".join(text.split())
    return hashlib.sha256(norm.encode()).hexdigest()[:16]


def build_evidence_pack(chunks: list[EvidenceChunk], settings: Settings) -> EvidencePack:
    top_k = settings.evidence_top_k
    max_per = settings.evidence_max_per_transcript
    budget = settings.evidence_max_context_tokens
    # Dedupe exact duplicates (keep first = best rank).
    seen: set[str] = set()
    deduped: list[EvidenceChunk] = []
    for c in sorted(chunks, key=lambda x: x.rank):
        fp = _fingerprint(c.text)
        if fp in seen:
            continue
        seen.add(fp)
        deduped.append(c)
    # Diversity cap.
    per: dict[str, int] = {}
    diverse: list[EvidenceChunk] = []
    for c in deduped:
        n = per.get(c.transcript_id, 0)
        if n >= max_per:
            continue
        per[c.transcript_id] = n + 1
        diverse.append(c)
        if len(diverse) >= top_k:
            break
    # Context budget: drop lowest-ranked until within budget.
    pack = list(diverse)
    dropped: list[str] = []
    truncated = len(deduped) > len(diverse) or len(chunks) > len(deduped)
    while pack and sum(estimate_tokens(c.text) for c in pack) > budget:
        removed = pack.pop()  # lowest rank last
        dropped.append(removed.chunk_id)
        truncated = True
    if not pack:
        raise ContextBudgetError(
            "insufficient_context: no evidence fits within the context budget",
            context={"budget_tokens": budget, "candidates": len(diverse)},
        )
    # Re-rank sequentially.
    for i, c in enumerate(pack):
        c.rank = i
    total = sum(estimate_tokens(c.text) for c in pack)
    return EvidencePack(chunks=pack, truncated=truncated, context_tokens=total,
                        dropped_chunk_ids=dropped)


def format_evidence_block(pack: EvidencePack) -> str:
    parts = []
    for c in pack.chunks:
        parts.append(
            f"[source_id: {c.chunk_id} | transcript: {c.transcript_id} "
            f"| turns {c.turn_start + 1}-{c.turn_end + 1} | score {c.score}]\n{c.text}"
        )
    return "\n\n---\n\n".join(parts)
