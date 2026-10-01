from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("system2.decisions")

DECISIONS = (
    "CLASSIFY_TASK",
    "APPLY_METADATA_FILTER",
    "ACCEPT_CANDIDATE",
    "REJECT_CANDIDATE",
    "EXPAND",
    "EXPAND_EVIDENCE",
    "SUFFICIENT",
    "SUFFICIENT_EVIDENCE",
    "INSUFFICIENT",
    "INSUFFICIENT_EVIDENCE",
    "ACCEPT_ANSWER",
    "FLAG_ANSWER",
    "REJECT_ANSWER",
    "ALLOW_COMPARISON",
)


@dataclass
class Decision:
    decision: str
    reason: str
    score: float | None = None
    transcript_id: str | None = None
    request_id: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        if self.decision not in DECISIONS:
            raise ValueError(f"unknown decision {self.decision!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DecisionLog:
    """Local typed decision/gating layer (Laya-style): every gate emits a Decision."""

    def __init__(self, request_id: str, path: Path | None = None):
        self.request_id = request_id
        self.path = path
        self.records: list[Decision] = []

    def record(
        self,
        decision: str,
        reason: str,
        score: float | None = None,
        transcript_id: str | None = None,
        **detail: Any,
    ) -> Decision:
        rec = Decision(
            decision=decision,
            reason=reason,
            score=score,
            transcript_id=transcript_id,
            request_id=self.request_id,
            detail=detail,
        )
        self.records.append(rec)
        log.debug("decision %s %s", rec.decision, rec.reason)
        return rec

    def flush(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            for rec in self.records:
                fh.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
