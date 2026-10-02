from __future__ import annotations

import math
from typing import Any


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, round((len(ordered) - 1) * p))], 3)


def per_question(question: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    gold = set(question.get("transcript_ids", []))
    evidence = result.get("evidence") or []
    ids = [item.get("transcript_id") for item in evidence if isinstance(item, dict)]
    relevant = [item for item in ids if item in gold]
    rank = next((i + 1 for i, item in enumerate(ids) if item in gold), None)
    turns = {item.get("turn_id") for item in evidence if isinstance(item, dict) and item.get("transcript_id") in gold}
    gold_turns = set(question.get("gold_turns", []))
    return {
        "conversation_recall": bool(gold & set(ids)) if gold else None,
        "reciprocal_rank": 1 / rank if rank else 0,
        "evidence_precision": len(relevant) / len(ids) if ids else 0,
        "evidence_recall": len(turns & gold_turns) / len(gold_turns) if gold_turns else None,
        "contamination": bool(set(ids) - gold),
        "irrelevant_transcript_count": len(set(ids) - gold),
        "latency_ms": (result.get("trace") or {}).get("timing_ms", {}).get("request"),
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in records if not r.get("errors")]
    metric = lambda key: [r["metrics"][key] for r in ok if r["metrics"].get(key) is not None]
    latency = [float(v) for v in metric("latency_ms") if v is not None]
    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None
    return {
        "queries": len(records), "failures": len(records) - len(ok),
        "conversation_recall": mean(metric("conversation_recall")), "mrr": mean(metric("reciprocal_rank")),
        "evidence_precision": mean(metric("evidence_precision")), "evidence_recall": mean(metric("evidence_recall")),
        "contamination_rate": mean(metric("contamination")),
        "latency_ms": {"mean": mean(latency), "p50": percentile(latency, .5), "p95": percentile(latency, .95)},
    }
