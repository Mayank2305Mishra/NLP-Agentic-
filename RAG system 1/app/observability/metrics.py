"""In-memory request metrics (latency breakdown, cache hits, failures)."""
from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass
class RequestMetrics:
    request_id: str
    task: str = ""
    retrieval_ms: float = 0.0
    embedding_ms: float = 0.0
    generation_ms: float = 0.0
    validation_ms: float = 0.0
    total_ms: float = 0.0
    num_chunks: int = 0
    context_tokens: int = 0
    validation_failures: int = 0
    retry_count: int = 0
    cache_hit: bool = False
    laya_ms: float = 0.0
    success: bool = True
    error_code: str | None = None


class MetricsRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self._records: list[RequestMetrics] = []

    def record(self, m: RequestMetrics) -> None:
        with self._lock:
            self._records.append(m)

    def snapshot(self) -> list[RequestMetrics]:
        with self._lock:
            return list(self._records)

    def summary(self) -> dict:
        recs = self.snapshot()
        if not recs:
            return {"count": 0}
        totals = sorted(r.total_ms for r in recs)
        import statistics

        def pct(p: float) -> float:
            if not totals:
                return 0.0
            idx = min(len(totals) - 1, int(p / 100 * len(totals)))
            return totals[idx]

        return {
            "count": len(recs),
            "mean_ms": round(statistics.mean(totals), 2),
            "p50_ms": round(pct(50), 2),
            "p95_ms": round(pct(95), 2),
            "max_ms": round(max(totals), 2),
            "failure_rate": round(sum(1 for r in recs if not r.success) / len(recs), 4),
        }


REGISTRY = MetricsRegistry()
