"""Transcript consistency validation (AGENTS_SYSTEM_4.md §23, §29.4).

A single answer must never merge customers from unrelated conversations
unless the user explicitly requested comparative or aggregate analysis.
"""

from __future__ import annotations

from system4.schema import Evidence


def selected_transcripts(evidence: list[Evidence]) -> set[str]:
    return {item.turn.transcript_id for item in evidence}


def transcript_consistency(
    answer: object,
    evidence: list[Evidence],
    cross_conversation: bool,
) -> list[str]:
    """Return identity-consistency error codes (empty means consistent)."""
    errors: list[str] = []
    selected = selected_transcripts(evidence)
    if not cross_conversation and len(selected) > 1:
        errors.append("CROSS_CONVERSATION_EVIDENCE")
    if not isinstance(answer, dict):
        return errors

    declared = answer.get("transcript_id")
    if declared is not None and str(declared) not in selected:
        errors.append("UNKNOWN_TRANSCRIPT_ID")
    declared_list = answer.get("source_transcripts")
    if isinstance(declared_list, list):
        for value in declared_list:
            if str(value) not in selected:
                errors.append("UNKNOWN_TRANSCRIPT_ID")
                break

    # Every citation must resolve inside the retrieved conversation set (§29.4).
    from system4.validation.citation import collect_citations, parse_citation, turns_by_transcript

    mapping = turns_by_transcript(evidence)
    for value in collect_citations(answer):
        if parse_citation(value, mapping) is None:
            errors.append("CITATION_OUTSIDE_EVIDENCE")
            break
    return sorted(set(errors))
