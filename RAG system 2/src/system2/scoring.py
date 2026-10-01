from __future__ import annotations

from typing import Sequence

from .cues import ACTION_CUES, CAUSE_CUES, CONTENT_STOPWORDS, EFFECT_CUES, PROBLEM_CUES, cue_hits
from .config import RetrievalConfig
from .schema import QueryPlan, ScoreComponents

BM25_SQUASH_K = 10.0

LEXICAL_BM25 = 0.60
LEXICAL_TERM_OVERLAP = 0.25
LEXICAL_PHRASE = 0.15

DEFAULT_TURN_WEIGHTS = {
    "lexical": 0.55,
    "causal": 0.15,
    "problem": 0.10,
    "action": 0.10,
    "outcome": 0.10,
}


def squash_bm25(bm25_raw: float) -> float:
    s = max(0.0, -float(bm25_raw))
    return s / (s + BM25_SQUASH_K)


def min_max(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def _fraction(needles: Sequence[str], haystack: str) -> float:
    if not needles:
        return 0.0
    low = " " + (haystack or "") + " "
    hits = sum(1 for n in needles if f" {n} " in low or n in low)
    return hits / len(needles)


def term_overlap(plan: QueryPlan, text: str) -> float:
    terms = [t for t in plan.terms if t not in CONTENT_STOPWORDS]
    if not terms:
        terms = plan.terms
    return _fraction(terms[:10], text)


def phrase_score(plan: QueryPlan, text: str) -> float:
    if not plan.phrases:
        return 0.0
    return _fraction(plan.phrases, text)


def cue_density(hits: list[str]) -> float:
    return min(1.0, len(hits) / 3.0)


def metadata_score(plan: QueryPlan, meta: dict, explicit: bool) -> float:
    score = 0.0
    intent = meta.get("intent")
    if plan.explicit_intent and intent and plan.explicit_intent == intent:
        score += 1.0
    elif intent and intent.lower() in plan.normalized_query:
        score += 0.8
    domain = meta.get("domain")
    if plan.metadata_hints.get("domain") and domain:
        if str(domain).lower() in plan.normalized_query:
            score += 0.5
    if plan.metadata_hints.get("month") and plan.metadata_hints["month"] in (
        (meta.get("time_of_interaction") or "").lower()
    ):
        score += 0.3
    if explicit and score == 0.0:
        score = 0.1
    return min(1.0, score)


def conversation_score(
    plan: QueryPlan,
    conversation_text: str,
    meta: dict,
    config: RetrievalConfig,
    bm25_raw: float,
) -> tuple[float, ScoreComponents]:
    w = config.weights
    comps = ScoreComponents(bm25_raw=bm25_raw)
    comps.bm25 = squash_bm25(bm25_raw)
    comps.metadata = metadata_score(plan, meta, bool(plan.explicit_intent)) if config.metadata_boost else 0.0
    comps.phrase = phrase_score(plan, conversation_text)
    comps.causal = cue_density(cue_hits(conversation_text, CAUSE_CUES)) if config.causal_scoring else 0.0
    comps.outcome = (
        cue_density(cue_hits(conversation_text, EFFECT_CUES + ACTION_CUES))
        if config.causal_scoring
        else 0.0
    )
    final = (
        w.bm25 * comps.bm25
        + w.metadata * comps.metadata
        + w.phrase * comps.phrase
        + w.causal * comps.causal
        + w.outcome * comps.outcome
    )
    return final, comps


def turn_score(
    plan: QueryPlan,
    text: str,
    meta: dict,
    config: RetrievalConfig,
    bm25_raw: float,
) -> tuple[float, ScoreComponents]:
    comps = ScoreComponents(bm25_raw=bm25_raw)
    comps.bm25 = squash_bm25(bm25_raw)
    overlap = term_overlap(plan, text)
    comps.phrase = phrase_score(plan, text)
    lexical = LEXICAL_BM25 * comps.bm25 + LEXICAL_TERM_OVERLAP * overlap + LEXICAL_PHRASE * comps.phrase
    comps.metadata = metadata_score(plan, meta, bool(plan.explicit_intent)) if config.metadata_boost else 0.0
    if config.causal_scoring:
        comps.causal = cue_density(cue_hits(text, CAUSE_CUES))
        comps.outcome = cue_density(cue_hits(text, EFFECT_CUES))
        comps.action = cue_density(cue_hits(text, ACTION_CUES))
        comps.problem = cue_density(cue_hits(text, PROBLEM_CUES))
    weights = DEFAULT_TURN_WEIGHTS
    final = (
        weights["lexical"] * lexical
        + weights["causal"] * comps.causal
        + weights["problem"] * comps.problem
        + weights["action"] * comps.action
        + weights["outcome"] * comps.outcome
        + 0.10 * comps.metadata
    )
    final = min(1.0, final)
    return final, comps
