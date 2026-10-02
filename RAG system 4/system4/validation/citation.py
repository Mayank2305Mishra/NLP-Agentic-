"""Citation validation: referenced turns must exist and belong to the retrieved
conversation set (AGENTS_SYSTEM_4.md §29.3–§29.4).

Citations are integers for single-conversation answers and
``"TRANSCRIPT_ID:TURN"`` strings for cross-conversation answers. Plain
integers are unambiguous only when a turn number appears in exactly one
selected transcript; ambiguous integers are rejected rather than guessed.
"""

from __future__ import annotations

import re

from system4.schema import Evidence

_CITED = re.compile(r"^\s*(?P<tid>.+?)\s*:\s*(?:turn\s*)?(?P<turn>\d+)\s*$", re.I)


def turns_by_transcript(evidence: list[Evidence]) -> dict[str, set[int]]:
    mapping: dict[str, set[int]] = {}
    for item in evidence:
        mapping.setdefault(item.turn.transcript_id, set()).add(item.turn.turn_id)
    return mapping


def parse_citation(value: object, mapping: dict[str, set[int]]) -> tuple[str, int] | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        if len(mapping) == 1:
            transcript_id = next(iter(mapping))
            return (transcript_id, value) if value in mapping[transcript_id] else None
        # Ambiguous across transcripts: valid only if unique to one transcript.
        owners = [tid for tid, turns in mapping.items() if value in turns]
        return (owners[0], value) if len(owners) == 1 else None
    if isinstance(value, str):
        match = _CITED.match(value)
        if match:
            transcript_id = match.group("tid")
            turn = int(match.group("turn"))
            if transcript_id in mapping and turn in mapping[transcript_id]:
                return transcript_id, turn
    return None


def collect_citations(answer: object) -> list[object]:
    values: list[object] = []
    if not isinstance(answer, dict):
        return values
    for rationale in answer.get("causal_rationales") or []:
        if isinstance(rationale, dict):
            values.extend(rationale.get("evidence_turns") or [])
    values.extend(answer.get("evidence") or [])
    return values


def validate_citations(answer: object, evidence: list[Evidence]) -> tuple[bool, list[object]]:
    """Return ``(all_valid, invalid_citations)`` (§29.3, §29.4)."""
    mapping = turns_by_transcript(evidence)
    citations = collect_citations(answer)
    if not mapping:
        # No evidence at all: any citation is unsupported.
        return (not citations), list(citations)
    invalid = [value for value in citations if parse_citation(value, mapping) is None]
    return not invalid, invalid


def collect_valid(answer: object, evidence: list[Evidence]) -> set[tuple[str, int]]:
    mapping = turns_by_transcript(evidence)
    valid: set[tuple[str, int]] = set()
    for value in collect_citations(answer):
        parsed = parse_citation(value, mapping)
        if parsed:
            valid.add(parsed)
    return valid
