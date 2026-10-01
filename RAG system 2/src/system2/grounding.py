from __future__ import annotations

import logging
from typing import Any

from .config import Config
from .cues import content_words
from .decisions import DecisionLog
from .schema import EvidencePack, GroundingReport, QueryPlan

log = logging.getLogger("system2.grounding")

TASK1_STRENGTHS = {"explicit", "strongly_supported", "inferred", "insufficient"}
TASK2_ROLES = {"customer", "agent", "business"}


def _claim_turns(item: dict[str, Any]) -> list[int]:
    raw = item.get("evidence_turns") or []
    out = []
    for value in raw:
        try:
            out.append(int(value))
        except (TypeError, ValueError):
            continue
    return out


def support_ratio(claim_text: str, evidence_text: str) -> float:
    words = content_words(claim_text)
    if not words:
        return 0.0
    low = " " + evidence_text.lower() + " "
    hits = sum(1 for w in words if f" {w} " in low)
    return hits / len(words)


def validate_schema(answer: dict[str, Any], task: str) -> list[str]:
    issues: list[str] = []
    if answer.get("task") not in (task, None):
        issues.append(f"task field mismatch: {answer.get('task')!r} != {task!r}")
    if "confidence" in answer:
        try:
            conf = float(answer["confidence"])
            if not 0.0 <= conf <= 1.0:
                issues.append(f"confidence out of range: {conf}")
        except (TypeError, ValueError):
            issues.append("confidence is not numeric")
    if not isinstance(answer.get("insufficient_evidence", False), bool):
        issues.append("insufficient_evidence must be boolean")
    if task == "task1":
        rationales = answer.get("causal_rationales")
        if not isinstance(rationales, list):
            issues.append("causal_rationales must be a list")
            return issues
        for idx, item in enumerate(rationales):
            if not isinstance(item, dict):
                issues.append(f"causal_rationales[{idx}] must be an object")
                continue
            if not isinstance(item.get("cause"), str):
                issues.append(f"causal_rationales[{idx}].cause must be a string")
            if not isinstance(item.get("effect"), str):
                issues.append(f"causal_rationales[{idx}].effect must be a string")
            strength = item.get("strength")
            if strength not in TASK1_STRENGTHS:
                issues.append(f"causal_rationales[{idx}].strength invalid: {strength!r}")
            if not isinstance(_claim_turns(item), list):
                issues.append(f"causal_rationales[{idx}].evidence_turns invalid")
    else:
        chain = answer.get("causal_chain")
        if not isinstance(chain, list):
            issues.append("causal_chain must be a list")
            return issues
        for idx, item in enumerate(chain):
            if not isinstance(item, dict):
                issues.append(f"causal_chain[{idx}] must be an object")
                continue
            if not isinstance(item.get("event"), str):
                issues.append(f"causal_chain[{idx}].event must be a string")
            role = item.get("role")
            if role not in TASK2_ROLES:
                issues.append(f"causal_chain[{idx}].role invalid: {role!r}")
        if not isinstance(answer.get("summary", ""), str):
            issues.append("summary must be a string")
    return issues


def validate_grounding(
    answer: dict[str, Any] | None,
    task: str,
    plan: QueryPlan,
    pack: EvidencePack | None,
    config: Config,
    decisions: DecisionLog,
) -> GroundingReport:
    if answer is None:
        decisions.record("REJECT_ANSWER", "no parseable answer")
        return GroundingReport(False, False, [], ["unparseable output"])

    issues = validate_schema(answer, task)
    claims: list[dict[str, Any]] = []
    contamination = False

    valid_turns: dict[int, set[str]] = {}
    if pack:
        for conv in pack.conversations:
            for item in conv.turns:
                valid_turns.setdefault(item.turn_id, set()).add(conv.transcript_id)

    single_conversation = bool(pack) and len(pack.conversations) == 1
    pack_ids = [c.transcript_id for c in pack.conversations] if pack else []
    if len(pack_ids) > 1 and not plan.comparison:
        contamination = True
        issues.append(f"evidence pack contains multiple conversations without comparison: {pack_ids}")

    items = answer.get("causal_rationales") or [] if task == "task1" else answer.get("causal_chain") or []
    resolved_transcripts: set[str] = set()
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        turns = _claim_turns(item)
        claim_text = " ".join(
            str(item.get(k, "")) for k in ("cause", "effect", "mechanism", "event", "summary")
        )
        claim: dict[str, Any] = {"index": idx, "turns": turns}
        bad_turns = [t for t in turns if t not in valid_turns]
        claim["unknown_turns"] = bad_turns
        if bad_turns:
            issues.append(f"claim {idx}: turns not present in retrieved evidence: {bad_turns}")

        evidence_text_parts: list[str] = []
        per_turn_sets: list[set[str]] = []
        if pack:
            for conv in pack.conversations:
                for ev in conv.turns:
                    if ev.turn_id in turns:
                        evidence_text_parts.append(ev.text_original)
                        if ev.turn_id not in bad_turns:
                            per_turn_sets.append(set([conv.transcript_id]))

        claim_transcripts: set[str] = set()
        if turns and not bad_turns:
            if single_conversation:
                claim_transcripts = {pack_ids[0]}
            elif per_turn_sets:
                intersection = set.intersection(*per_turn_sets) if per_turn_sets else set()
                if not intersection:
                    contamination = True
                    issues.append(
                        f"claim {idx}: cited turns cannot come from one conversation: {sorted(t for t in turns)}"
                    )
                elif len(intersection) > 1 and not plan.comparison:
                    issues.append(
                        f"claim {idx}: ambiguous turn citation across conversations {sorted(intersection)}"
                    )
                claim_transcripts = intersection
        resolved_transcripts |= claim_transcripts
        claim["transcripts"] = sorted(claim_transcripts)

        ratio = support_ratio(claim_text, " ".join(evidence_text_parts))
        claim["support_ratio"] = round(ratio, 4)
        claim["supported"] = ratio >= config.grounding.min_claim_support
        if turns and not claim["supported"]:
            issues.append(f"claim {idx}: low lexical support ({ratio:.2f})")
        claims.append(claim)

    if len(resolved_transcripts) > 1 and not plan.comparison:
        contamination = True
        issues.append(f"answer resolves to multiple conversations: {sorted(resolved_transcripts)}")

    passed = not issues
    report = GroundingReport(passed, contamination, claims, issues)

    unsupported = [c for c in claims if c.get("turns") and not c.get("supported")]
    if contamination and config.grounding.reject_on_contamination:
        decisions.record(
            "REJECT_ANSWER",
            "cross-conversation contamination detected",
            detail={"issues": issues},
        )
    elif answer is not None and (issues and not claims and not answer.get("insufficient_evidence")):
        decisions.record("REJECT_ANSWER", "schema invalid", detail={"issues": issues})
    elif unsupported or issues:
        decisions.record(
            "FLAG_ANSWER",
            "answer accepted with grounding flags",
            detail={"issues": issues, "unsupported_claims": len(unsupported)},
        )
    else:
        decisions.record("ACCEPT_ANSWER", "grounding validation passed")
    return report
