"""Frozen evaluation splits (dev/val/test) with discipline guardrails.

Split is deterministic (sha-based) from transcript_ids. Once a test set is
frozen (generated/benchmarks/splits.json), the harness records the split hash
in every report; tuning against test is a process violation the code cannot
fully prevent, so the manifest + report make it auditable.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

DEFAULT_SPLIT = {"dev": 0.7, "val": 0.15, "test": 0.15}


def assign_split(transcript_id: str) -> str:
    h = int(hashlib.sha256(transcript_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < DEFAULT_SPLIT["dev"]:
        return "dev"
    if h < DEFAULT_SPLIT["dev"] + DEFAULT_SPLIT["val"]:
        return "val"
    return "test"


def build_splits(transcript_ids: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {"dev": [], "val": [], "test": []}
    for tid in transcript_ids:
        out[assign_split(tid)].append(tid)
    return out


def save_splits(settings, transcript_ids: list[str]) -> Path:
    from app.config import Settings as S  # noqa
    splits = build_splits(sorted(set(transcript_ids)))
    payload = {"splits": {k: sorted(v) for k, v in splits.items()},
               "counts": {k: len(v) for k, v in splits.items()},
               "hash": hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest()[:16]}
    path = Path(settings.benchmarks_dir) / "splits.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_splits(settings) -> dict | None:
    path = Path(settings.benchmarks_dir) / "splits.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None
