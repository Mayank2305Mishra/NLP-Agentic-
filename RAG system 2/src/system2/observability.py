from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Any, Iterator

STAGE_KEYS = [
    "query_normalization_ms",
    "metadata_filter_ms",
    "fts_retrieval_ms",
    "conversation_ranking_ms",
    "turn_ranking_ms",
    "neighbor_expansion_ms",
    "evidence_pack_ms",
    "evidence_gate_ms",
    "generation_ttft_ms",
    "generation_ms",
    "parse_ms",
    "grounding_ms",
    "answer_gate_ms",
    "total_ms",
]


@dataclass
class StageTimer:
    trace: dict[str, Any] = field(default_factory=dict)
    _start: float = field(default_factory=time.perf_counter)

    @classmethod
    def start(cls) -> "StageTimer":
        timer = cls(trace={k: None for k in STAGE_KEYS})
        timer._start = time.perf_counter()
        timer.trace["total_ms"] = 0.0
        return timer

    @contextmanager
    def stage(self, key: str) -> Iterator[None]:
        begin = time.perf_counter()
        try:
            yield
        finally:
            self.trace[key] = round((time.perf_counter() - begin) * 1000, 3)

    def set(self, key: str, value: float | None) -> None:
        self.trace[key] = value

    def finish(self) -> dict[str, Any]:
        self.trace["total_ms"] = round((time.perf_counter() - self._start) * 1000, 3)
        return self.trace


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]
