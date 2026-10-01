from __future__ import annotations

from .schema import EvidenceConversation, EvidenceItem, EvidencePack, EvidenceWindow
from .textnorm import estimate_tokens

SEPARATOR = "\n" + "=" * 72 + "\n"
MULTI_HEADER = (
    "NOTE: the blocks below are SEPARATE conversations with distinct transcript ids. "
    "Never merge them into one interaction.\n"
)


def _window_items(window: EvidenceWindow, turns_by_id: dict[int, EvidenceItem]) -> list[EvidenceItem]:
    return [turns_by_id[t] for t in window.turn_ids if t in turns_by_id]


def _items_for(
    windows: list[EvidenceWindow], turns_by_id: dict[int, EvidenceItem]
) -> list[EvidenceItem]:
    ids = {i.turn_id for w in windows for i in _window_items(w, turns_by_id)}
    return sorted((turns_by_id[i] for i in ids if i in turns_by_id), key=lambda i: i.turn_id)


def pack_evidence(
    conversations: list[EvidenceConversation],
    task: str,
    max_tokens: int,
    turns_by_conv: dict[str, dict[int, EvidenceItem]],
) -> EvidencePack:
    ordered = sorted(conversations, key=lambda c: c.score, reverse=True)
    selected: list[EvidenceConversation] = []
    used = 0

    for conv in ordered:
        turns_by_id = turns_by_conv.get(conv.transcript_id, {})
        windows = sorted(conv.windows, key=lambda w: w.score, reverse=True)
        kept: list[EvidenceWindow] = []
        for window in windows:
            trial = kept + [window]
            items = _items_for(trial, turns_by_id)
            if not items:
                continue
            block_tokens = estimate_tokens(_render_block(conv, items))
            if used + block_tokens <= max_tokens or not kept:
                kept = trial
            else:
                break
        items = _items_for(kept, turns_by_id)
        if not items:
            continue
        block = _render_block(conv, items)
        block_tokens = estimate_tokens(block)
        if selected and used + block_tokens > max_tokens:
            continue
        used += block_tokens
        selected.append(
            EvidenceConversation(
                transcript_id=conv.transcript_id,
                score=conv.score,
                meta=conv.meta,
                turns=items,
                windows=kept,
                components=conv.components,
            )
        )

    blocks = [_render_block(c, c.turns) for c in selected]
    text = SEPARATOR.join(blocks)
    if len(blocks) > 1:
        text = MULTI_HEADER + text
    allowed = [c.transcript_id for c in selected]
    return EvidencePack(
        task=task,
        conversations=selected,
        token_estimate=estimate_tokens(text),
        allowed_transcript_ids=allowed,
        text=text,
    )


def _render_block(conv: EvidenceConversation, items: list[EvidenceItem]) -> str:
    meta = conv.meta
    lines = [
        f"CONVERSATION: {conv.transcript_id}",
        f"DOMAIN: {meta.get('domain') or 'unknown'}",
        f"INTENT: {meta.get('intent') or 'unknown'}",
        f"REASON: {meta.get('reason_for_call') or 'unknown'}",
        "",
    ]
    for item in items:
        speaker = (item.speaker_original or item.speaker or "unknown").upper()
        lines.append(f"TURN {item.turn_id:03d} | {speaker}")
        lines.append(item.text_original.strip())
        lines.append("")
    return "\n".join(lines).rstrip()


def turns_index(conversations: list[EvidenceConversation]) -> dict[str, dict[int, EvidenceItem]]:
    return {c.transcript_id: {item.turn_id: item for item in c.turns} for c in conversations}
