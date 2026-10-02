from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Config

log = logging.getLogger("system3.sampling")


def _load_conversation_meta(config: Config) -> list[dict[str, Any]]:
    path = config.dataset.conversations_jsonl
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python -m system3.cli build-data` first"
        )
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    return rows


def stratified_sample(config: Config) -> dict[str, Any]:
    rows = _load_conversation_meta(config)
    seed = config.dataset.sample_seed
    target = config.dataset.sample_size
    key = config.dataset.stratification_key

    strata: dict[Any, list[str]] = defaultdict(list)
    for row in rows:
        strata[row.get(key)].append(row["transcript_id"])

    total = len(rows)
    selected: list[str] = []
    shortfall: dict[str, int] = {}
    rng = random.Random(seed)

    ideal = {k: target * (len(v) / total) for k, v in strata.items()}
    for k, ids in strata.items():
        rng.shuffle(ids)
        want = int(round(ideal[k]))
        if want > len(ids):
            shortfall[str(k)] = want - len(ids)
            want = len(ids)
        selected.extend(ids[:want])

    selected = sorted(set(selected))
    if len(selected) < target:
        remaining = [i for ids in strata.values() for i in ids if i not in set(selected)]
        rng.shuffle(remaining)
        selected.extend(remaining[: target - len(selected)])
        selected = sorted(set(selected))

    manifest = {
        "seed": seed,
        "source_hash": _source_hash(config),
        "source_count": total,
        "sample_count": len(selected),
        "stratification_key": key,
        "strata_count": len(strata),
        "strata_shortfall": shortfall,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selected_transcript_ids": selected,
    }
    out_path = config.dataset.sampling_manifest
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)
    log.info("wrote sampling manifest: %s ids -> %s", len(selected), out_path)
    return manifest


def _source_hash(config: Config) -> str:
    meta_path = config.dataset.processed_path / "source_meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as fh:
            return json.load(fh).get("source_hash", "")
    return ""


def load_population(config: Config, population: str) -> set[str] | None:
    if population == "full":
        return None
    manifest_path = config.dataset.sampling_manifest
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found — run `python -m system3.cli sample` first"
        )
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    return set(manifest["selected_transcript_ids"])
