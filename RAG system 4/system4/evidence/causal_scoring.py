"""Causal evidence model: typed roles, reasons, contradiction signals (§15, §19).

Mechanisms are only assigned on explicit explanatory markers — never merely
because cause and effect appear in adjacent turns (§15).
"""

from __future__ import annotations

from system4.config.settings import Settings
from system4.schema import CausalRole, Evidence, Turn

_ROLE_MARKERS: dict[str, tuple[str, ...]] = {
    "cause": ("because", "due to", "caused by", "reason is", "reason for", "since the"),
    "trigger": ("when the", "after the", "as soon as", "the moment", "first time", "started when"),
    "problem": (
        "issue", "problem", "unable", "cannot", "can't", "could not", "couldn't",
        "failed", "failure", "error", "broken", "damaged", "wrong", "overcharged",
        "charged twice", "not working", "stuck", "declined",
    ),
    "customer_motivation": (
        "i want", "i need", "i'm trying", "i am trying", "i would like", "i'd like",
        "looking for", "hoping to", "asking because", "i was charged", "i want to avoid",
    ),
    "agent_action": (
        "i can", "i will", "we can", "we will", "let me", "i'll", "approved",
        "process", "escalate", "update your", "send you", "apply the", "book",
    ),
    "mechanism": (
        "the system", "system will", "our system", "automatically", "our policy",
        "policy requires", "the process", "how this works", "how it works",
        "the reason it", "triggered a", "updates the", "reflects the",
    ),
    "effect": ("as a result", "so now", "therefore", "which means", "this means", "because of that"),
    "outcome": (
        "resolve", "fix", "refund", "cancelled", "canceled", "cancel ", "closed", "close ",
        "confirmed", "approved", "booked", "saved", "upgraded", "restored", "reinstalled",
        "no longer", "works again",
    ),
    "uncertainty": (
        "not sure", "unclear", "uncertain", "might be", "maybe", "pending",
        "cannot confirm", "can't confirm", "unable to confirm", "i guess", "hopefully",
    ),
    "clarification": ("just to confirm", "let me confirm", "to clarify", "what do you mean", "are you saying"),
}

#: Explicit opposition pairs used for contradiction signals (§19). Both sides
#: must appear in *different* turns to count as a contradiction.
CONTRADICTION_PAIRS: tuple[tuple[str, str], ...] = (
    ("approved", "denied"),
    ("waived", "declined"),
    ("accepted", "rejected"),
    ("resolved", "unresolved"),
    ("enabled", "disabled"),
    ("refund approved", "refund denied"),
)

_MECHANISM_SPEAKERS = ("agent", "representative", "support", "specialist")


def _is_agent(speaker: str) -> bool:
    lowered = speaker.lower()
    return any(marker in lowered for marker in _MECHANISM_SPEAKERS) or lowered == "unknown"


def detect_roles(turn: Turn, query_terms: set[str], settings: Settings) -> list[tuple[str, str]]:
    """Return ``(role, reason)`` pairs for one turn (§15 example shape)."""
    lowered = turn.text.lower()
    roles: list[tuple[str, str]] = []

    for role, markers in _ROLE_MARKERS.items():
        if role == "mechanism" and not _is_agent(turn.speaker):
            continue  # mechanisms are agent explanations, never inferred from adjacency
        for marker in markers:
            if marker in lowered:
                roles.append((role, f"turn contains {role.replace('_', ' ')} marker '{marker}'"))
                break

    if query_terms:
        turn_terms = {word for word in lowered.split() if word.strip(".,!?")}
        overlap = query_terms & turn_terms
        if overlap:
            roles.append(
                ("query_match", f"turn matches query terms: {', '.join(sorted(overlap)[:5])}")
            )
    if not roles:
        roles.append(("context", "no causal or query signal in turn"))
    return roles


def assign_roles(evidence: list[Evidence], query: str, settings: Settings) -> None:
    """Annotate evidence items with causal roles and reasons (in place)."""
    query_terms = {term for term in query.split() if len(term) > 2}
    for item in evidence:
        detected = detect_roles(item.turn, query_terms, settings)
        item.roles = [role for role, _ in detected]
        item.reason = "; ".join(reason for _, reason in detected)


def contradiction_count(evidence: list[Evidence]) -> int:
    """Count explicit opposition pairs across selected turns (§19)."""
    texts = [item.turn.text.lower() for item in evidence]
    found = 0
    for positive, negative in CONTRADICTION_PAIRS:
        has_positive = any(positive in text and negative not in text for text in texts)
        has_negative = any(negative in text for text in texts)
        if has_positive and has_negative:
            found += 1
    return found
