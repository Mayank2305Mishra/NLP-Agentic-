from __future__ import annotations

import hashlib
import logging
import re
from typing import Sequence

from .cues import (
    ACTION_CUES,
    CAUSE_CUES,
    CONTENT_STOPWORDS,
    EFFECT_CUES,
    PROBLEM_CUES,
    SYNONYM_MAP,
    TEMPORAL_CUES,
    cue_hits,
)
from .schema import QueryPlan
from .store import build_match, quote_term
from .textnorm import normalize_text

log = logging.getLogger("system3.query")

ROLE_WORDS = frozenset(
    """customer agent caller representative rep advisor adviser specialist supervisor
    manager consultant operator associate you me i we they he she""".split()
)

_COMPARISON_RE = re.compile(
    r"\b(compare (these |the |two |multiple )?(conversations|interactions|calls|customers)"
    r"|compare across (conversations|interactions|customers|calls)"
    r"|across (all )?(the )?(conversations|interactions|customers|calls)"
    r"|multiple conversations|different customers)\b",
    re.IGNORECASE,
)
_TASK2_RE = re.compile(
    r"\b(summar(y|ise|ize)|summarize the|entire (call|conversation)|whole (call|conversation)"
    r"|end to end|walk through|causal chain|timeline|chronolog\w*|story of|narrative)\b",
    re.IGNORECASE,
)
_TASK2_COMPARE_RE = re.compile(r"\b(compare|comparison|versus|vs\.?|difference between)\b", re.I)
_QUOTED_RE = re.compile(r'"([^"]{2,80})"')
_MONTH_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
    re.IGNORECASE,
)


def classify_task(raw_query: str) -> str:
    if _TASK2_RE.search(raw_query) or _TASK2_COMPARE_RE.search(raw_query):
        return "task2"
    return "task1"


def detect_comparison(raw_query: str) -> bool:
    return bool(_COMPARISON_RE.search(raw_query))


class QueryProcessor:
    def __init__(self, intents: Sequence[str], domains: Sequence[str] | None = None):
        self.intents = [(i, normalize_text(i)) for i in intents if i]
        self.domains = [(d, normalize_text(d)) for d in (domains or []) if d]

    def process(self, raw_query: str, task: str | None = None) -> QueryPlan:
        normalized = normalize_text(raw_query)
        tokens = normalized.split()
        quoted = [normalize_text(p) for p in _QUOTED_RE.findall(raw_query)]
        quoted = [p for p in quoted if p]

        all_content = [t for t in tokens if t not in CONTENT_STOPWORDS]
        retrieval_terms = [t for t in all_content if t not in STOPWORDISH] or all_content
        if not retrieval_terms:
            retrieval_terms = [t for t in tokens if t not in CONTENT_STOPWORDS][:8]
        if not retrieval_terms:
            retrieval_terms = tokens[:8]

        phrases = quoted
        causal_terms = sorted(set(cue_hits(normalized, CAUSE_CUES)))
        outcome_terms = sorted(
            set(cue_hits(normalized, EFFECT_CUES)) | set(cue_hits(normalized, ACTION_CUES))
        )
        event_terms = sorted(
            set(cue_hits(normalized, PROBLEM_CUES)) | set(cue_hits(normalized, TEMPORAL_CUES))
        )

        expanded: list[str] = []
        for term in retrieval_terms:
            expanded.extend(SYNONYM_MAP.get(term, [term]))
        if causal_terms:
            for term in ("cause", "resolve", "resolution"):
                expanded.extend(SYNONYM_MAP.get(term, []))
        expanded = sorted(set(expanded) - set(retrieval_terms))

        explicit_intent, metadata_hints = self._metadata(all_content, normalized)

        strict_terms = [
            t
            for t in retrieval_terms
            if t not in LOW_VALUE_TERMS and t not in ROLE_WORDS and t not in STOPWORDISH
        ]
        strict_terms = strict_terms or [t for t in retrieval_terms if t not in ROLE_WORDS] or retrieval_terms
        strict_terms = (phrases + strict_terms)[:12]
        strict_match = build_match(strict_terms, "AND")
        loose_terms = (strict_terms + retrieval_terms + expanded)[:20]
        loose_match = build_match(loose_terms, "OR")

        detected = task if task in ("task1", "task2") else classify_task(raw_query)
        return QueryPlan(
            raw_query=raw_query,
            query_id=hashlib.sha256(normalized.encode()).hexdigest()[:12],
            normalized_query=normalized,
            terms=retrieval_terms,
            phrases=phrases,
            causal_terms=causal_terms,
            event_terms=event_terms,
            outcome_terms=outcome_terms,
            expanded_terms=expanded,
            metadata_hints=metadata_hints,
            explicit_intent=explicit_intent,
            task=detected,
            comparison=detect_comparison(raw_query),
            fts_match=strict_match,
            fts_match_loose=loose_match,
        )

    @staticmethod
    def _token_match(term: str, norm: str) -> bool:
        if not term or not norm:
            return False
        if term == norm:
            return True
        if norm.startswith(term + " ") or term.startswith(norm + " "):
            return True
        if len(term) > 3 and len(norm) > 3 and term.rstrip("s") == norm.rstrip("s"):
            return True
        return False

    def _metadata(self, terms: list[str], normalized: str) -> tuple[str | None, dict]:
        hints: dict = {}
        explicit: str | None = None
        for original, norm in self.intents:
            if norm and (norm in normalized or norm.replace(" ", "") in normalized.replace(" ", "")):
                explicit = original
                hints["intent"] = original
                break
        if explicit is None:
            # A shared word like "customer" is only a hint, never grounds for
            # a hard filter. Require a distinctive non-role query token.
            matches: list[tuple[int, str]] = []
            for original, norm in self.intents:
                intent_tokens = norm.split()
                overlap = [
                    term
                    for term in terms
                    if term not in ROLE_WORDS
                    and len(term) >= 5
                    and any(self._token_match(term, tok) for tok in intent_tokens)
                ]
                if overlap:
                    matches.append((len(overlap), original))
            if matches:
                hints["intent"] = max(matches)[1]
        for original, norm in self.domains:
            if norm and norm in normalized:
                hints["domain"] = original
        month = _MONTH_RE.search(normalized)
        if month:
            hints["month"] = month.group(1)
        return explicit, hints


STOPWORDISH = frozenset(
    """why how what when where who which did do does was were are is be been being have has
    had will would can could should may might must about into over under again very just also
    any some all each other more most such than then there here""".split()
)

LOW_VALUE_TERMS = frozenset(
    """call calls talking talked speaking said says tell told asked asking question
    thing things stuff moment time times day today please help hi hello okay ok
    want wants like get got make made""".split()
)
