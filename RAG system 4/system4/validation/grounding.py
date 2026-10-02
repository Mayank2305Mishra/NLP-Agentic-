"""Grounding validator (AGENTS_SYSTEM_4.md §29).

Checks every causal claim against retrieved evidence: the claim has support,
the cited turn exists, the cited turn belongs to the retrieved conversation,
the claim does not exceed the evidence, and contradictions are surfaced.
"""

from __future__ import annotations

import re

from system4.evidence.causal_scoring import CONTRADICTION_PAIRS
from system4.schema import Evidence, GroundingStatus
from system4.validation.citation import parse_citation, turns_by_transcript

WORDS = re.compile(r"[a-z0-9]+", re.I)
STOP = {
    "the", "a", "an", "and", "or", "is", "was", "were", "to", "of", "in", "that",
    "it", "this", "for", "with", "on", "at", "as", "be", "been", "are", "by", "from",
}


def claim_words(claim: object) -> set[str]:
    return {w.lower() for w in WORDS.findall(str(claim)) if w.lower() not in STOP and len(w) > 1}


def support_ratio(claim: object, evidence_words: set[str]) -> float:
    words = claim_words(claim)
    if not words:
        return 0.0
    return len(words & evidence_words) / len(words)


def _claims(answer: dict) -> list[str]:
    claims: list[str] = []
    for rationale in answer.get("causal_rationales") or []:
        if isinstance(rationale, dict):
            claims.extend(
                str(rationale.get(key) or "")
                for key in ("cause", "effect", "mechanism")
            )
    if answer.get("summary"):
        claims.append(str(answer["summary"]))
    claims.extend(str(item) for item in answer.get("key_causes") or [])
    claims.extend(str(item) for item in answer.get("effects") or [])
    claims.extend(str(item) for item in answer.get("mechanisms") or [])
    return [claim for claim in claims if claim.strip()]


def _cited_texts(answer: dict, evidence: list[Evidence]) -> list[str]:
    """Texts of turns actually cited by the answer, in citation order."""
    mapping = turns_by_transcript(evidence)
    by_key = {(i.turn.transcript_id, i.turn.turn_id): i.turn.text.lower() for i in evidence}
    texts: list[str] = []
    from system4.validation.citation import collect_citations

    for value in collect_citations(answer):
        parsed = parse_citation(value, mapping)
        if parsed and parsed in by_key:
            texts.append(by_key[parsed])
    if not texts:
        texts = [item.turn.text.lower() for item in evidence]
    return texts


def _contradicts(claim: str, cited_texts: list[str]) -> bool:
    lowered = claim.lower()
    for positive, negative in CONTRADICTION_PAIRS:
        if negative in lowered and not any(negative in text for text in cited_texts):
            if any(positive in text and negative not in text for text in cited_texts):
                return True
    return False


def claim_support(answer: object, evidence: list[Evidence]) -> dict[str, float]:
    """Per-claim support ratios used by the answer gate for pruning (§30)."""
    if not isinstance(answer, dict):
        return {}
    evidence_words: set[str] = set()
    for item in evidence:
        evidence_words |= claim_words(item.turn.text)
    return {claim: support_ratio(claim, evidence_words) for claim in _claims(answer)}


def validate(
    answer: dict,
    evidence: list[Evidence],
    min_ratio: float = 0.55,
) -> tuple[str, float]:
    """Return ``(grounding_status, mean_support_ratio)`` (§29 statuses)."""
    if not isinstance(answer, dict):
        return GroundingStatus.UNSUPPORTED.value, 0.0
    if not evidence:
        return GroundingStatus.UNSUPPORTED.value, 0.0

    mapping = turns_by_transcript(evidence)
    from system4.validation.citation import collect_citations

    cited_values = collect_citations(answer)
    for value in cited_values:
        if parse_citation(value, mapping) is None:
            # §29.3/§29.4: unknown turn or turn outside retrieved conversation.
            return GroundingStatus.UNSUPPORTED.value, 0.0

    claims = _claims(answer)
    has_claims = bool(claims)
    evidence_words: set[str] = set()
    for item in evidence:
        evidence_words |= claim_words(item.turn.text)
    cited_texts = _cited_texts(answer, evidence)

    if any(_contradicts(claim, cited_texts) for claim in claims):
        return GroundingStatus.CONTRADICTED.value, 0.0

    ratios = [support_ratio(claim, evidence_words) for claim in claims]
    if ratios:
        ratio = sum(ratios) / len(ratios)
        if not cited_values:
            return GroundingStatus.UNSUPPORTED.value, ratio
        if ratio >= min_ratio:
            return GroundingStatus.GROUNDED.value, round(ratio, 4)
        if ratio >= min_ratio / 2:
            return GroundingStatus.PARTIALLY_GROUNDED.value, round(ratio, 4)
        return GroundingStatus.UNSUPPORTED.value, round(ratio, 4)

    # No claims to check: grounded only when citations were validated and exist.
    if cited_values and has_claims is False:
        return GroundingStatus.PARTIALLY_GROUNDED.value, 0.5
    return GroundingStatus.UNSUPPORTED.value, 0.0
