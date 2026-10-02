"""Source record loading (AGENTS_SYSTEM_4.md §6, §7).

Reads without mutating source data; malformed records are counted, never
fabricated (metadata nulls stay null).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from system4.ingestion.normalize import iter_records, normalize_record
from system4.schema import Conversation


def read_records(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def load_conversations(
    path: str | Path, stats: dict[str, int] | None = None
) -> Iterator[Conversation]:
    """Yield normalized conversations; malformed records are skipped and counted."""
    data = read_records(path)
    read = skipped = 0
    for record in iter_records(data):
        read += 1
        conversation = normalize_record(record)
        if conversation is None:
            skipped += 1
            continue
        yield conversation
    if stats is not None:
        stats["records_read"] = read
        records = data if isinstance(data, list) else data.get("records", []) if isinstance(data, dict) else []
        stats["records_total"] = len(records) if isinstance(records, list) else read
        stats["records_skipped"] = skipped
