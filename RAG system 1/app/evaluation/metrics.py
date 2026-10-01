"""Engineering quality proxies (NOT official hackathon scores).

Until official labels/ground truth exist, report: structured-output validity,
citation validity, grounding rate, evidence sufficiency. Human rubric (0-4)
dimensions are preserved per-case for later aggregation.
"""
from __future__ import annotations

import statistics


def latency_stats(samples: list[float]) -> dict:
    if not samples:
        return {"count": 0}
    s = sorted(samples)
    def pct(p: float) -> float:
        return s[min(len(s) - 1, int(p / 100 * len(s)))]
    return {"count": len(s), "mean": round(statistics.mean(s), 2),
            "p50": round(pct(50), 2), "p90": round(pct(90), 2),
            "p95": round(pct(95), 2), "max": round(max(s), 2)}


def quality_summary(cases: list[dict]) -> dict:
    n = len(cases)
    if not n:
        return {"cases": 0}
    ok = [c for c in cases if c.get("success")]
    def rate(key, val=True):
        return round(sum(1 for c in ok if c.get(key) == val) / n, 4) if n else 0.0
    grounded = sum(1 for c in ok if c.get("grounded"))
    return {
        "cases": n,
        "success_rate": round(len(ok) / n, 4),
        "failure_rate": round((n - len(ok)) / n, 4),
        "structured_validity_rate": rate("structured_valid"),
        "citation_validity_rate": rate("citations_valid"),
        "grounding_rate": round(grounded / n, 4),
        "evidence_sufficiency_rate": round(
            sum(1 for c in ok if not c.get("insufficient_evidence")) / n, 4) if n else 0.0,
    }


# Human rubric dimensions (0-4), stored per case, aggregated only afterwards.
RUBRIC_DIMENSIONS = ["causal_correctness", "evidence_grounding", "completeness", "synthesis_quality"]


def empty_rubric() -> dict:
    return {d: None for d in RUBRIC_DIMENSIONS}
