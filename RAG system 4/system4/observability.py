"""Structured request traces (AGENTS_SYSTEM_4.md §40, §41).

Traces carry identifiers, scores, counts, and hashes — never raw transcript
text by default. A defensive redactor strips content-like keys in case a
caller passes richer structures. Logging failures never break request
handling.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

#: Keys whose values could contain conversation content (§41).
_SENSITIVE_KEYS = frozenset(
    {
        "text",
        "content",
        "prompt",
        "evidence_pack",
        "raw",
        "answer_text",
        "transcript",
        "conversation",
    }
)


def redact(value: Any) -> Any:
    """Recursively drop content-like keys; keep ids, scores, counts, hashes."""
    if isinstance(value, dict):
        return {
            key: ("[REDACTED]" if key.lower() in _SENSITIVE_KEYS else redact(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return [redact(item) for item in value]
    return value


class TraceLogger:
    """Append-only JSONL trace sink with size-based rotation."""

    def __init__(
        self,
        directory: str | Path | None = None,
        enabled: bool = True,
        max_bytes: int = 5_000_000,
        backups: int = 3,
    ) -> None:
        self.directory = Path(directory) if directory else None
        self.enabled = bool(enabled and self.directory)
        self.max_bytes = max_bytes
        self.backups = backups

    @classmethod
    def from_settings(cls, settings: Any) -> "TraceLogger":
        return cls(
            directory=getattr(settings, "trace_dir", None),
            enabled=getattr(settings, "trace_enabled", True),
            max_bytes=getattr(settings, "trace_max_bytes", 5_000_000),
        )

    @property
    def path(self) -> Path | None:
        if self.directory is None:
            return None
        return self.directory / "system4.jsonl"

    def _rotate(self, target: Path) -> None:
        if not target.exists() or target.stat().st_size < self.max_bytes:
            return
        for index in range(self.backups - 1, 0, -1):
            source = target.with_suffix(f".{index}.jsonl")
            if source.exists():
                source.replace(target.with_suffix(f".{index + 1}.jsonl"))
        target.replace(target.with_suffix(".1.jsonl"))

    def emit(self, trace: dict[str, Any]) -> None:
        if not self.enabled or self.directory is None:
            return
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            target = self.path
            assert target is not None
            self._rotate(target)
            line = json.dumps(redact(trace), default=str, ensure_ascii=False)
            with target.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except Exception:
            pass  # observability must never fail a request (§40)

    def read(self, limit: int | None = None) -> list[dict[str, Any]]:
        target = self.path
        if target is None or not target.exists():
            return []
        rows: list[dict[str, Any]] = []
        with target.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows[-limit:] if limit else rows


def now_ms() -> float:
    return round(time.perf_counter() * 1000, 3)
