"""Context compression after reranking and selection (AGENTS_SYSTEM_4.md §21).

Compression never rewrites meaning: it drops greetings, filler, near-duplicates,
and (aggressively) context-only turns, then enforces the token budget by
priority. Originals stay on ``Evidence.turn.text``; dropped turn ids are
reported for auditing, and the compressed text is stored on
``Evidence.compressed_text``.
"""

from __future__ import annotations

import re

from system4.config.settings import Settings
from system4.schema import CAUSAL_ROLES, CompressionLevel, Evidence

_FILLER = re.compile(
    r"^(hi|hello|hey|good (morning|afternoon|evening)|"
    r"thank(s| you)( so much| again| for calling| so much for calling)?|"
    r"you'?re welcome|okay|ok|sure|yes|yeah|no|bye|goodbye|mm[- ]?hmm|alright|great"
    r"|perfect|got it|hold on|one moment|please hold)[.!?, ]*$",
    re.I,
)

_MEANINGFUL = CAUSAL_ROLES | {"query_match", "contradiction", "clarification", "uncertainty"}


def is_filler(item: Evidence) -> bool:
    return bool(_FILLER.match(item.turn.text.strip())) and not (
        _MEANINGFUL.intersection(item.roles) - {"query_match"}
    )


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def compress(
    evidence: list[Evidence],
    level: CompressionLevel | str,
    budget: int,
    settings: Settings,
) -> tuple[list[Evidence], dict]:
    """Compress evidence to the budget at the requested level (§21, §37)."""
    if isinstance(level, str):
        level = CompressionLevel(level)
    from system4.evidence.packer import estimate_tokens  # local import: avoid cycle

    stats: dict = {
        "level": level.value,
        "input_items": len(evidence),
        "input_tokens": sum(estimate_tokens(item.turn.text, settings) for item in evidence),
        "dropped_filler": 0,
        "dropped_duplicate": 0,
        "dropped_context": 0,
        "dropped_budget": 0,
        "dropped_turns": [],
    }
    if level is CompressionLevel.NONE or not evidence:
        kept = sorted(evidence, key=lambda i: (i.turn.transcript_id, i.turn.turn_id))
        stats["output_items"] = len(kept)
        stats["output_tokens"] = sum(estimate_tokens(i.turn.text, settings) for i in kept)
        return kept, stats

    working = list(evidence)
    if level in (CompressionLevel.LIGHT, CompressionLevel.AGGRESSIVE):
        keep = [i for i in working if not is_filler(i)] or working
        stats["dropped_filler"] = len(working) - len(keep)
        working = keep
        # Near-duplicate removal across the same transcript.
        seen: dict[tuple[str, str], Evidence] = {}
        deduped: list[Evidence] = []
        for item in sorted(working, key=lambda i: (i.turn.transcript_id, i.turn.turn_id)):
            key = (item.turn.transcript_id, _normalize(item.turn.text))
            if key in seen and key[1]:
                stats["dropped_duplicate"] += 1
                stats["dropped_turns"].append(
                    {"transcript_id": item.turn.transcript_id, "turn_id": item.turn.turn_id}
                )
                continue
            seen[key] = item
            deduped.append(item)
        working = deduped or working

    if level is CompressionLevel.AGGRESSIVE:
        meaningful = [i for i in working if _MEANINGFUL.intersection(i.roles)]
        if meaningful:
            stats["dropped_context"] = len(working) - len(meaningful)
            stats["dropped_turns"].extend(
                {"transcript_id": i.turn.transcript_id, "turn_id": i.turn.turn_id}
                for i in working
                if i not in meaningful
            )
            working = meaningful

    # Budget trim by priority (score desc), always keeping the strongest item.
    kept: list[Evidence] = []
    used = 0
    for item in sorted(working, key=lambda i: (-i.score, i.turn.transcript_id, i.turn.turn_id)):
        cost = estimate_tokens(item.turn.text, settings) + 10
        if kept and used + cost > budget:
            stats["dropped_budget"] += 1
            stats["dropped_turns"].append(
                {"transcript_id": item.turn.transcript_id, "turn_id": item.turn.turn_id}
            )
            continue
        used += cost
        kept.append(item)
    if not kept and working:
        kept = [max(working, key=lambda i: i.score)]

    kept.sort(key=lambda i: (i.turn.transcript_id, i.turn.turn_id))
    for item in kept:
        item.compressed_text = item.turn.text
    stats["output_items"] = len(kept)
    stats["output_tokens"] = sum(estimate_tokens(i.turn.text, settings) for i in kept)
    return kept, stats
