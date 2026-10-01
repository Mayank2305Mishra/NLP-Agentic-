"""Conversation-aware chunker: overlapping turn windows, never char-splitting.

Chunk text format:
    Domain: ... | Intent: ... | Reason: ...
    Turn 001 | Agent: ...
    Turn 002 | Customer: ...

Stable IDs: {transcript_id}:chunk:{index}. Raw and normalized text kept
separately. Token length estimated at ~4 chars/token (no char==token assumption).
"""
from __future__ import annotations

import hashlib
import json

from app.domain.models import Chunk, NormalizedTranscript


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def chunking_config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


def chunk_transcript(nt: NormalizedTranscript, *, target_tokens: int = 450,
                     max_tokens: int = 600, overlap_turns: int = 2,
                     include_metadata: bool = True) -> list[Chunk]:
    turns = nt.turns
    if not turns:
        return []
    # Greedy turn windows: accumulate until target_tokens, hard-stop at max.
    windows: list[tuple[int, int]] = []  # (start_idx, end_idx) inclusive
    start = 0
    while start < len(turns):
        end = start
        acc: list[str] = []
        while end < len(turns):
            line = f"Turn {turns[end].index + 1:03d} | {turns[end].speaker}: {turns[end].text}"
            trial = acc + [line]
            if estimate_tokens("\n".join(trial)) > max_tokens and acc:
                break
            acc = trial
            end += 1
            if estimate_tokens("\n".join(acc)) >= target_tokens:
                break
        windows.append((start, max(start, end - 1)))
        if end >= len(turns):
            break
        start = max(end - overlap_turns, start + 1)  # overlap, always progress
    chunks: list[Chunk] = []
    for ci, (s, e) in enumerate(windows):
        wturns = turns[s:e + 1]
        lines = [f"Turn {t.index + 1:03d} | {t.speaker}: {t.text}" for t in wturns]
        norm_lines = [f"Turn {t.index + 1:03d} | {t.speaker}: {t.retrieval_text}" for t in wturns]
        header = ""
        if include_metadata:
            parts = []
            if nt.domain:
                parts.append(f"Domain: {nt.domain}")
            if nt.intent:
                parts.append(f"Intent: {nt.intent}")
            if nt.reason_for_call:
                parts.append(f"Reason: {nt.reason_for_call}")
            if parts:
                header = " | ".join(parts) + "\n"
        text = header + "\n".join(lines)
        norm_text = (header + "\n".join(norm_lines)) if header else "\n".join(norm_lines)
        speakers = sorted({t.speaker for t in wturns})
        chunks.append(
            Chunk(
                chunk_id=f"{nt.transcript_id}:chunk:{ci}",
                transcript_id=nt.transcript_id,
                chunk_index=ci,
                turn_start=wturns[0].index,
                turn_end=wturns[-1].index,
                speakers=speakers,
                domain=nt.domain,
                intent=nt.intent,
                reason_for_call=nt.reason_for_call,
                text=text,
                normalized_text=norm_text,
                source_hash=nt.source_hash,
            )
        )
    return chunks
