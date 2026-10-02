"""Record normalization and post-call artifact detection (AGENTS_SYSTEM_4.md §7, §24).

Normalization runs once at ingestion and persists a deterministic representation.
Source data is never mutated; speaker identity and original text are preserved
(only whitespace is collapsed, meaning is unchanged).
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterator

from system4.schema import Conversation, Turn

_WS = re.compile(r"\s+")

TURN_ID_KEYS = ("turn_id", "turn_number", "turn_index", "turn")

#: Trailing segments that look like internal/post-call notes rather than
#: customer dialogue (§24). Detected structurally, never used as causal
#: evidence unless the task explicitly requests internal notes.
_INTERNAL_NOTE = re.compile(
    r"\b(internal note|internal only|post[- ]call|after[- ]call|after call notes?"
    r"|qa note|quality note|agent notes?|rep notes?|representative notes?"
    r"|wrap[- ]?up notes?|case notes?|resolution notes?|supervisor notes?"
    r"|for the record|escalation notes?)\b",
    re.I,
)
_TERMINATION = re.compile(
    r"\b(goodbye|good bye|bye bye|thanks for calling|thank you for calling"
    r"|have a (great|good|nice) (day|evening|afternoon))\b",
    re.I,
)


def collapse_ws(text: str) -> str:
    return _WS.sub(" ", text).strip()


def _text(value: Any) -> str | None:
    if isinstance(value, str):
        collapsed = collapse_ws(value)
        return collapsed or None
    return None


def conversation_hash(conversation: Conversation) -> str:
    """Deterministic content hash for reproducibility (§7)."""
    payload = json.dumps(
        {
            "transcript_id": conversation.transcript_id,
            "domain": conversation.domain,
            "intent": conversation.intent,
            "reason_for_call": conversation.reason_for_call,
            "time_of_interaction": conversation.time_of_interaction,
            "turns": [[t.turn_id, t.speaker, t.text] for t in conversation.turns],
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def looks_internal(text: str) -> bool:
    return bool(_INTERNAL_NOTE.search(text))


def is_termination(text: str) -> bool:
    return bool(_TERMINATION.search(text))


def mark_post_call(turns: list[Turn]) -> list[Turn]:
    """Mark the trailing run of internal-note turns as post-call artifacts (§24).

    Post-call segments are classified separately; they never become customer
    causal evidence by default.
    """
    if not turns:
        return turns
    start = len(turns)
    while start > 0 and looks_internal(turns[start - 1].text):
        start -= 1
    if start == len(turns):
        return turns
    return [
        turn if index < start else Turn(
            turn.transcript_id, turn.turn_id, turn.speaker, turn.text,
            turn.domain, turn.intent, turn.reason_for_call, post_call=True,
        )
        for index, turn in enumerate(turns)
    ]


def _turn_id(raw: Any, fallback: int, last: int) -> int:
    """Preserve the original turn number when present and sane (§6, §7)."""
    if isinstance(raw, bool):
        return fallback
    if isinstance(raw, int):
        candidate = raw
    elif isinstance(raw, str) and raw.strip().lstrip("-").isdigit():
        candidate = int(raw.strip())
    else:
        return fallback
    if candidate > last:  # keep strictly increasing ids for citation safety
        return candidate
    return fallback


def normalize_record(record: dict[str, Any]) -> Conversation | None:
    if not isinstance(record, dict):
        return None
    transcript_id = str(record.get("transcript_id") or "").strip()
    raw_turns = record.get("conversation")
    if not transcript_id or not isinstance(raw_turns, list):
        return None
    domain = _text(record.get("domain"))
    intent = _text(record.get("intent"))
    reason = _text(record.get("reason_for_call"))
    time_of_interaction = _text(record.get("time_of_interaction"))
    turns: list[Turn] = []
    last = 0
    for index, raw in enumerate(raw_turns):
        if not isinstance(raw, dict):
            continue
        text = _text(raw.get("text"))
        if not text:
            continue
        speaker = _text(raw.get("speaker")) or "Unknown"
        turn_id = _turn_id(raw.get("turn_id", raw.get("turn_number")), index + 1, last)
        last = turn_id
        turns.append(
            Turn(transcript_id, turn_id, speaker, text, domain, intent, reason)
        )
    if not turns:
        return None
    turns = mark_post_call(turns)
    return Conversation(transcript_id, domain, intent, reason, turns, time_of_interaction)


def iter_records(data: Any) -> Iterator[dict[str, Any]]:
    if isinstance(data, list):
        yield from (row for row in data if isinstance(row, dict))
    elif isinstance(data, dict):
        records = data.get("records", [])
        if isinstance(records, list):
            yield from (row for row in records if isinstance(row, dict))
