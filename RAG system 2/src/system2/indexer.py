from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Iterator

from .config import Config
from .schema import Conversation, Turn
from .store import LexicalStore

log = logging.getLogger("system2.indexer")


def _iter_meta(path: Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


def _iter_turn_groups(path: Path) -> Iterator[list[Turn]]:
    group: list[Turn] = []
    current = None
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            tid = row["transcript_id"]
            if current is not None and tid != current:
                yield group
                group = []
            current = tid
            group.append(
                Turn(
                    transcript_id=row["transcript_id"],
                    turn_id=row["turn_id"],
                    speaker_original=row["speaker_original"],
                    speaker=row["speaker"],
                    text_original=row["text_original"],
                    text_normalized=row["text_normalized"],
                    domain=row.get("domain"),
                    intent=row.get("intent"),
                    reason_for_call=row.get("reason_for_call"),
                    time_of_interaction=row.get("time_of_interaction"),
                    is_post_call_artifact=bool(row.get("is_post_call_artifact")),
                    speaker_inferred=bool(row.get("speaker_inferred")),
                )
            )
    if group:
        yield group


def build_index(
    config: Config, population: str = "full", reset: bool = False
) -> dict[str, int]:
    if not config.dataset.conversations_jsonl.exists():
        raise FileNotFoundError("processed data missing — run build-data first")
    from .sampling import load_population

    allowed = load_population(config, population)
    store = LexicalStore(config.dataset.db_path)
    if reset and config.dataset.db_path.exists():
        store.close()
        config.dataset.db_path.unlink()
        for suffix in ("-wal", "-shm"):
            side = Path(str(config.dataset.db_path) + suffix)
            if side.exists():
                side.unlink()
        store = LexicalStore(config.dataset.db_path)
    store.create_schema()

    begin = time.perf_counter()
    meta_by_id = {m["transcript_id"]: m for m in _iter_meta(config.dataset.conversations_jsonl)}
    turn_groups = _iter_turn_groups(config.dataset.turns_jsonl)

    batch: list[Conversation] = []
    inserted_convs = 0
    inserted_turns = 0
    skipped = 0

    def flush() -> None:
        nonlocal inserted_convs, inserted_turns
        if not batch:
            return
        inserted_convs += store.insert_conversations(batch, config.retrieval.conversation_text_mode)
        inserted_turns += store.insert_turns(batch)
        batch.clear()

    for turns in turn_groups:
        tid = turns[0].transcript_id
        if allowed is not None and tid not in allowed:
            skipped += 1
            continue
        meta = meta_by_id.get(tid)
        if meta is None:
            skipped += 1
            continue
        conv = Conversation(
            transcript_id=tid,
            time_of_interaction=meta.get("time_of_interaction"),
            domain=meta.get("domain"),
            intent=meta.get("intent"),
            reason_for_call=meta.get("reason_for_call"),
            turns=turns,
        )
        batch.append(conv)
        if len(batch) >= 200:
            flush()
    flush()

    log.info("building FTS indexes...")
    store.build_fts()
    store.optimize()
    counts = store.counts()
    build_s = round(time.perf_counter() - begin, 2)
    counts["build_seconds"] = build_s
    counts["population"] = population
    counts["skipped_conversations"] = skipped
    stats_path = config.dataset.index_path / "index_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(counts, fh, indent=2)
    db_size = config.dataset.db_path.stat().st_size
    counts["db_size_bytes"] = db_size
    log.info("index built in %.1fs: %s", build_s, counts)
    store.close()
    return counts
