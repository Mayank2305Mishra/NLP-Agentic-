"""Final answer gate (AGENTS_SYSTEM_4.md §30).

Composes schema validity, transcript identity, evidence existence, grounding,
and causal-shape checks into one typed decision. Unsupported claims are
removed or marked uncertain before the answer leaves the system; abstention
is a first-class, high-quality outcome.
"""

from __future__ import annotations

from dataclasses import dataclass

from system4.config.settings import Settings
from system4.schema import AnswerGate, Decision, Evidence, GroundingStatus
from system4.validation.grounding import claim_words, support_ratio


@dataclass
class GateOutcome:
    gate: AnswerGate
    reason: str
    grounding_status: str
    grounding_ratio: float
    pruned_claims: int = 0


def _cl(answer: dict) -> list[dict]:
    return [r for r in answer.get("causal_rationales") or [] if isinstance(r, dict)]


def _evidence_words(evidence: list[Evidence]) -> set[str]:
    words: set[str] = set()
    for item in evidence:
        words |= claim_words(item.turn.text)
    return words


def _prune_task1(answer: dict, evidence: list[Evidence], settings: Settings) -> int:
    words = _evidence_words(evidence)
    kept: list[dict] = []
    pruned = 0
    for rationale in _cl(answer):
        support = max(
            support_ratio(rationale.get(key, ""), words)
            for key in ("cause", "effect", "mechanism")
            if str(rationale.get(key) or "").strip()
        ) if any(str(rationale.get(key) or "").strip() for key in ("cause", "effect", "mechanism")) else 0.0
        if support < settings.min_claim_support:
            pruned += 1
            continue
        kept.append(rationale)
    if pruned:
        answer["causal_rationales"] = kept
    return pruned


def _prune_task2(answer: dict, evidence: list[Evidence], settings: Settings) -> int:
    words = _evidence_words(evidence)
    pruned = 0
    uncertainties = [str(u) for u in answer.get("uncertainties") or []]
    for key in ("key_causes", "effects", "mechanisms"):
        kept_items: list[str] = []
        for item in answer.get(key) or []:
            if support_ratio(item, words) < settings.min_claim_support:
                uncertainties.append(f"Unverified {key[:-1]} removed by grounding gate: {item}")
                pruned += 1
            else:
                kept_items.append(item)
        if pruned:
            answer[key] = kept_items
    if pruned:
        answer["uncertainties"] = uncertainties
    return pruned


def final_gate(
    answer: dict,
    evidence: list[Evidence],
    decision: Decision,
    grounding: tuple[str, float],
    causal_errors: list[str],
    consistency_errors: list[str],
    settings: Settings,
) -> GateOutcome:
    """Evaluate every §30 pass condition; may prune/uncertain-ify ``answer``."""
    grounding_status, ratio = grounding

    if not evidence:
        return GateOutcome(AnswerGate.ABSTAIN, "no evidence available", GroundingStatus.UNSUPPORTED.value, 0.0)
    if consistency_errors:
        return GateOutcome(
            AnswerGate.ABSTAIN,
            f"transcript consistency failed: {', '.join(consistency_errors)}",
            GroundingStatus.UNSUPPORTED.value,
            ratio,
        )
    fatal = [e for e in causal_errors if e != "MISSING_CAUSAL_RATIONALE"] if answer.get("insufficient_evidence") else causal_errors
    if fatal:
        return GateOutcome(
            AnswerGate.ABSTAIN,
            f"causal shape invalid: {', '.join(fatal)}",
            GroundingStatus.UNSUPPORTED.value if grounding_status == GroundingStatus.GROUNDED.value else grounding_status,
            ratio,
        )

    pruned = 0
    if grounding_status == GroundingStatus.CONTRADICTED.value:
        return GateOutcome(
            AnswerGate.ABSTAIN,
            "generated claim contradicts its cited evidence",
            grounding_status,
            ratio,
        )
    if grounding_status == GroundingStatus.UNSUPPORTED.value or ratio < settings.min_grounding_ratio:
        # Unsupported claims are removed even on the abstain path (§30);
        # the pipeline replaces the answer with the abstention shape either way.
        if answer.get("task") == "task1":
            pruned = _prune_task1(answer, evidence, settings)
        else:
            pruned = _prune_task2(answer, evidence, settings)
        return GateOutcome(
            AnswerGate.ABSTAIN,
            "generated claims are not sufficiently grounded",
            grounding_status,
            ratio,
            pruned,
        )

    # PARTIALLY_GROUNDED: remove unsupported claims, keep the grounded remainder.
    if answer.get("task") == "task1":
        pruned = _prune_task1(answer, evidence, settings)
        remaining = _cl(answer)
    else:
        pruned = _prune_task2(answer, evidence, settings)
        remaining = [1] if str(answer.get("summary") or "").strip() else []

    if not remaining:
        return GateOutcome(
            AnswerGate.ABSTAIN,
            "all claims removed by grounding gate",
            GroundingStatus.UNSUPPORTED.value,
            ratio,
            pruned,
        )

    answer["confidence"] = round(min(float(answer.get("confidence") or ratio), ratio), 4) if "confidence" in answer else round(ratio, 4)
    if grounding_status == GroundingStatus.PARTIALLY_GROUNDED.value or decision.answer_gate == AnswerGate.ANSWER_WITH_UNCERTAINTY or pruned:
        gate = AnswerGate.ANSWER_WITH_UNCERTAINTY
        reason = "answered with uncertainty after grounding review"
    else:
        gate = AnswerGate.ANSWER
        reason = "grounded answer passed all gates"
    return GateOutcome(gate, reason, grounding_status, ratio, pruned)
