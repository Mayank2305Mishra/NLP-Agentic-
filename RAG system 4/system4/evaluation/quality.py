"""Generation quality metrics (AGENTS_SYSTEM_4.md §31).

Works purely on benchmark rows + traces so it can run with or without an LLM.
"""

from __future__ import annotations

from collections import Counter

_REQUIRED = {
    "task1": ("causal_rationales", "confidence", "insufficient_evidence"),
    "task2": ("summary", "key_causes", "effects", "mechanisms", "uncertainties", "evidence", "insufficient_evidence"),
}


def _complete(answer: dict, task: str) -> bool:
    return all(key in answer for key in _REQUIRED.get(task, ()))


def evaluate_quality(rows: list[dict]) -> dict:
    if not rows:
        return {}
    grounding = Counter(row.get("grounding") for row in rows)
    answered = [row for row in rows if not row.get("insufficient_evidence")]
    abstained = [row for row in rows if row.get("insufficient_evidence")]
    answerable = [row for row in rows if row.get("answerable", True)]
    unanswerable = [row for row in rows if row.get("answerable") is False]

    correct_abstains = sum(1 for row in unanswerable if row.get("insufficient_evidence"))
    correct_answers = sum(
        1 for row in answerable
        if not row.get("insufficient_evidence") and row.get("hit") is not False
    )
    return {
        "queries": len(rows),
        "answered": len(answered),
        "abstained": len(abstained),
        "answer_rate": round(len(answered) / len(rows), 4),
        "abstention_rate": round(len(abstained) / len(rows), 4),
        "grounding_distribution": dict(grounding),
        "grounded_rate": round(
            grounding.get("GROUNDED", 0) / max(1, len(answered)), 4
        ),
        "unsupported_claim_rate": round(
            sum(
                1
                for row in answered
                if row.get("grounding") in {"UNSUPPORTED", "CONTRADICTED"}
            )
            / max(1, len(answered)),
            4,
        ),
        "json_valid_rate": round(
            sum(1 for row in rows if row.get("json_valid", True)) / len(rows), 4
        ),
        "answer_completeness": round(
            sum(1 for row in rows if row.get("complete", True)) / len(rows), 4
        ),
        "abstention_precision": (
            round(correct_abstains / len(abstained), 4) if abstained else None
        ),
        "unanswerable_queries": len(unanswerable),
        "correct_abstains": correct_abstains,
        "answerable_correct": correct_answers,
        "answerable_total": len(answerable),
    }
