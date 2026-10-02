"""Retrieval metrics measured independently from generation (§31)."""

from __future__ import annotations

import math
from typing import Iterable


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / k


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for index, value in enumerate(retrieved, 1):
        if value in relevant:
            return 1.0 / index
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = 0.0
    for index, value in enumerate(retrieved[:k], 1):
        if value in relevant:
            dcg += 1.0 / math.log2(index + 1)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def turn_recall(retrieved_turns: Iterable[int], relevant_turns: set[int]) -> float:
    if not relevant_turns:
        return 0.0
    return len(set(retrieved_turns) & relevant_turns) / len(relevant_turns)


def contamination_rate(rows: list[dict]) -> float:
    """Fraction of single-answer queries whose evidence left the gold conversation (§3.1, §23)."""
    checked = [
        row for row in rows
        if row.get("expected") and row.get("retrieved") is not None
    ]
    if not checked:
        return 0.0
    contaminated = sum(
        1
        for row in checked
        if any(tid != row["expected"] for tid in row.get("retrieved") or [])
    )
    return contaminated / len(checked)


def evaluate_retrieval(rows: list[dict], ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict:
    scored = [row for row in rows if row.get("expected")]
    result: dict = {"scored_queries": len(scored)}
    if not scored:
        result.update({f"recall@{k}": None for k in ks})
        result.update({f"precision@{k}": None for k in ks})
        result.update({f"ndcg@{k}": None for k in ks})
        result["mrr"] = None
        result["conversation_recall"] = None
        result["turn_recall"] = None
        result["contamination_rate"] = 0.0
        return result
    for k in ks:
        result[f"recall@{k}"] = round(
            sum(recall_at_k(row["retrieved"], {row["expected"]}, k) for row in scored) / len(scored), 4
        )
        result[f"precision@{k}"] = round(
            sum(precision_at_k(row["retrieved"], {row["expected"]}, k) for row in scored) / len(scored), 4
        )
        result[f"ndcg@{k}"] = round(
            sum(ndcg_at_k(row["retrieved"], {row["expected"]}, k) for row in scored) / len(scored), 4
        )
    result["mrr"] = round(
        sum(reciprocal_rank(row["retrieved"], {row["expected"]}) for row in scored) / len(scored), 4
    )
    result["conversation_recall"] = result["recall@1"]
    turn_rows = [row for row in scored if row.get("expected_turns")]
    result["turn_recall"] = (
        round(
            sum(turn_recall(row.get("retrieved_turns") or [], set(row["expected_turns"])) for row in turn_rows)
            / len(turn_rows),
            4,
        )
        if turn_rows
        else None
    )
    result["contamination_rate"] = round(contamination_rate(scored), 4)
    return result
