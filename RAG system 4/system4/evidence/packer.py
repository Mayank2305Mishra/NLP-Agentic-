"""Evidence pack builder (AGENTS_SYSTEM_4.md §16, §22).

Produces the compact, provenance-explicit pack consumed by the LLM:
``SOURCE`` blocks per transcript with chronological turns and roles.
Budgets are enforced with a deterministic token estimator; chronology inside
each source is a correctness property, not presentation polish.
"""

from __future__ import annotations

import math

from system4.config.settings import Settings
from system4.schema import Evidence

#: Turns are never dropped for budget if they are the first item overall.
_OVERHEAD_PER_TURN = 10


def estimate_tokens(text: str, settings: Settings | None = None) -> int:
    """Deterministic token estimate: chars/4 (default) or word count."""
    estimator = (settings.token_estimator if settings else "chars4") or "chars4"
    if estimator == "words":
        return max(1, len(text.split()))
    return max(1, math.ceil(len(text) / 4))


def _blocks(evidence: list[Evidence]) -> list[tuple[str, list[Evidence]]]:
    """Group evidence by transcript, strongest conversation first (§23)."""
    groups: dict[str, list[Evidence]] = {}
    order: list[str] = []
    for item in sorted(evidence, key=lambda i: (i.turn.transcript_id, i.turn.turn_id)):
        if item.turn.transcript_id not in groups:
            groups[item.turn.transcript_id] = []
            order.append(item.turn.transcript_id)
        groups[item.turn.transcript_id].append(item)
    best = {
        tid: max(item.score for item in items) for tid, items in groups.items()
    }
    order.sort(key=lambda tid: (-best[tid], tid))
    return [(tid, sorted(groups[tid], key=lambda i: i.turn.turn_id)) for tid in order]


def pack(
    evidence: list[Evidence], budget: int, settings: Settings | None = None
) -> tuple[str, list[Evidence]]:
    if not evidence:
        return "", []
    lines: list[str] = []
    chosen: list[Evidence] = []
    used = 0
    for block_index, (transcript_id, items) in enumerate(_blocks(evidence)):
        head = items[0].turn
        header = [
            f"SOURCE: {transcript_id}",
            f"DOMAIN: {head.domain or 'unknown'}",
            f"INTENT: {head.intent or 'unknown'}",
        ]
        header_cost = estimate_tokens("\n".join(header), settings) + 4
        if chosen and used + header_cost > budget:
            break  # budget exhausted (§3.4): no partial blocks after the first
        header_start = len(lines)
        if block_index > 0:
            lines.append("")
        lines.extend(header)
        used += header_cost
        block_turns = 0
        for item in items:
            turn = item.turn
            body = item.compressed_text or turn.text
            cost = estimate_tokens(body, settings) + _OVERHEAD_PER_TURN
            if chosen and used + cost > budget:
                break  # keep chronology within the block; never re-order
            text = body if turn.post_call is False else f"{body} [POST-CALL ARTIFACT]"
            lines.append(f"[Turn {turn.turn_id:03d} | {turn.speaker}] {text}")
            lines.append(f"Roles: {', '.join(item.roles)}  Reason: {item.reason}")
            lines.append("")
            used += cost
            chosen.append(item)
            block_turns += 1
        if block_turns:
            roles = sorted({role for item in items for role in item.roles})
            lines.append(f"Evidence roles ({transcript_id}): {', '.join(roles)}")
            used += 4
        else:
            # Nothing from this block fit; drop its header to avoid a dangling SOURCE.
            del lines[header_start:]
            used -= header_cost
    if not chosen and evidence:
        # Never return an empty pack when evidence exists.
        first = min(evidence, key=lambda i: (i.turn.transcript_id, i.turn.turn_id))
        chosen = [first]
        lines = [
            f"SOURCE: {first.turn.transcript_id}",
            f"[Turn {first.turn.turn_id:03d} | {first.turn.speaker}] "
            f"{first.compressed_text or first.turn.text}",
            f"Roles: {', '.join(first.roles)}",
        ]
    return "\n".join(lines).strip(), chosen
