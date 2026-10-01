"""Per-request trace with stage timings (AGENTS.md section 26)."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field


@dataclass
class StageTiming:
    name: str
    duration_ms: float


@dataclass
class RequestTrace:
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    stages: list[StageTiming] = field(default_factory=list)
    _starts: dict = field(default_factory=dict, repr=False)

    def start(self, stage: str) -> None:
        self._starts[stage] = time.perf_counter()

    def end(self, stage: str) -> float:
        started = self._starts.pop(stage, None)
        if started is None:
            return 0.0
        ms = (time.perf_counter() - started) * 1000.0
        self.stages.append(StageTiming(name=stage, duration_ms=round(ms, 2)))
        return ms

    def stage_ms(self, stage: str) -> float:
        for s in self.stages:
            if s.name == stage:
                return s.duration_ms
        return 0.0

    def stage_ms_total(self, stage: str) -> float:
        """Sum across repeated stages (task2 runs task1 generation first)."""
        return round(sum(s.duration_ms for s in self.stages if s.name == stage), 2)

    def total_ms(self) -> float:
        for s in self.stages:
            if s.name == "request":
                return s.duration_ms  # outer span is the true total (no double-count)
        return round(sum(s.duration_ms for s in self.stages), 2)

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "stages": [{"stage": s.name, "latency_ms": s.duration_ms} for s in self.stages],
            "total_ms": self.total_ms(),
        }


class Timer:
    """Context manager: with Timer(trace, 'embedding'): ..."""

    def __init__(self, trace: RequestTrace, stage: str):
        self.trace = trace
        self.stage = stage

    def __enter__(self):
        self.trace.start(self.stage)
        return self

    def __exit__(self, *exc):
        self.trace.end(self.stage)
        return False
