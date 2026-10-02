"""Causal shape guardrails that lexical grounding alone cannot provide (§16, §27)."""

from __future__ import annotations

import re

from system4.schema import Evidence
from system4.validation.citation import parse_citation, turns_by_transcript


def _normal(value: object) -> str:
    return re.sub(r"\W+", "", str(value).lower())


def validate_causal_shape(
    answer: dict,
    task: str,
    evidence: list[Evidence],
    cross_conversation: bool,
) -> list[str]:
    errors: list[str] = []
    if answer.get("task") != task:
        errors.append("TASK_MISMATCH")
    if not cross_conversation and len({item.turn.transcript_id for item in evidence}) > 1:
        errors.append("CROSS_CONVERSATION_EVIDENCE")

    mapping = turns_by_transcript(evidence)
    mechanism_roles = {
        (item.turn.transcript_id, item.turn.turn_id)
        for item in evidence
        if "mechanism" in item.roles
    }

    for rationale in answer.get("causal_rationales") or []:
        if not isinstance(rationale, dict):
            errors.append("MALFORMED_RATIONALE")
            continue
        cause, effect = rationale.get("cause", ""), rationale.get("effect", "")
        if not str(cause).strip() or not str(effect).strip():
            errors.append("MISSING_CAUSE_OR_EFFECT")
        elif _normal(cause) == _normal(effect):
            errors.append("CAUSE_EFFECT_IDENTICAL")
        if str(cause).rstrip().endswith("?") or str(effect).rstrip().endswith("?"):
            errors.append("QUESTION_AS_CAUSAL_CLAIM")
        mechanism = rationale.get("mechanism")
        if str(mechanism or "").strip():
            # Mechanism claims must be supported by an explicit mechanism turn (§15).
            cited = {
                parsed
                for value in (rationale.get("evidence_turns") or [])
                if (parsed := parse_citation(value, mapping)) is not None
            }
            if not mechanism_roles or (cited and not (cited & mechanism_roles)):
                errors.append("MECHANISM_UNSUPPORTED")
        turns = [
            value for value in (rationale.get("evidence_turns") or [])
            if isinstance(value, int) and not isinstance(value, bool)
        ]
        if turns != sorted(turns):
            errors.append("EVIDENCE_NOT_CHRONOLOGICAL")

    if task == "task1" and not answer.get("causal_rationales") and not answer.get("insufficient_evidence"):
        errors.append("MISSING_CAUSAL_RATIONALE")
    if task == "task2" and not answer.get("summary") and not answer.get("insufficient_evidence"):
        errors.append("MISSING_SUMMARY")
    return sorted(set(errors))
