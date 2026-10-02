from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from .config import Config
from .schema import Conversation, Turn
from .textnorm import infer_speaker_by_position, is_post_call_artifact, normalize_speaker, normalize_text

log = logging.getLogger("system3.dataset")

REQUIRED_FIELDS = ("transcript_id", "conversation")


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def load_raw(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("raw dataset must be a JSON array")
    return data


def _validate_record(record: dict[str, Any], idx: int) -> list[str]:
    errors = []
    for field in REQUIRED_FIELDS:
        if field not in record:
            errors.append(f"record[{idx}] missing {field}")
    conv = record.get("conversation")
    if not isinstance(conv, list):
        errors.append(f"record[{idx}] conversation is not a list")
    else:
        for t_idx, turn in enumerate(conv):
            if not isinstance(turn, dict) or "text" not in turn:
                errors.append(f"record[{idx}].conversation[{t_idx}] missing text")
    return errors


def parse_conversation(record: dict[str, Any]) -> Conversation:
    transcript_id = str(record["transcript_id"])
    meta = {
        "domain": record.get("domain"),
        "intent": record.get("intent"),
        "reason_for_call": record.get("reason_for_call"),
        "time_of_interaction": record.get("time_of_interaction"),
    }
    conversation = Conversation(transcript_id=transcript_id, turns=[], **meta)
    previous_canonical: str | None = None
    previous_text: str | None = None
    for t_idx, raw_turn in enumerate(record.get("conversation") or []):
        speaker_original = str(raw_turn.get("speaker") or "").strip()
        text_original = str(raw_turn.get("text") or "")
        canonical, inferred = infer_speaker_by_position(speaker_original, previous_canonical)
        artifact = is_post_call_artifact(speaker_original, text_original, previous_text)
        turn = Turn(
            transcript_id=transcript_id,
            turn_id=t_idx,
            speaker_original=speaker_original,
            speaker=canonical,
            text_original=text_original,
            text_normalized=normalize_text(text_original),
            is_post_call_artifact=artifact,
            speaker_inferred=inferred,
            **meta,
        )
        conversation.turns.append(turn)
        previous_canonical = canonical if canonical != "unknown" else previous_canonical
        previous_text = text_original
    return conversation


def iter_conversations(records: Iterable[dict[str, Any]]) -> Iterable[Conversation]:
    for record in records:
        yield parse_conversation(record)


def write_processed(conversations: list[Conversation], config: Config) -> dict[str, Any]:
    processed_dir = config.dataset.processed_path
    processed_dir.mkdir(parents=True, exist_ok=True)
    turns_path = config.dataset.turns_jsonl
    conv_path = config.dataset.conversations_jsonl
    speaker_counts: Counter[str] = Counter()
    artifact_count = 0
    turn_count = 0
    with open(conv_path, "w", encoding="utf-8") as conv_fh, open(
        turns_path, "w", encoding="utf-8"
    ) as turn_fh:
        for conv in conversations:
            meta = {
                "transcript_id": conv.transcript_id,
                "time_of_interaction": conv.time_of_interaction,
                "domain": conv.domain,
                "intent": conv.intent,
                "reason_for_call": conv.reason_for_call,
                "turn_count": len(conv.turns),
            }
            conv_fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
            for turn in conv.turns:
                turn_fh.write(json.dumps(asdict(turn), ensure_ascii=False) + "\n")
                speaker_counts[turn.speaker] += 1
                turn_count += 1
                artifact_count += int(turn.is_post_call_artifact)
    stats = {
        "conversation_count": len(conversations),
        "turn_count": turn_count,
        "speaker_counts": dict(speaker_counts),
        "post_call_artifact_turns": artifact_count,
        "intent_count": len({c.intent for c in conversations}),
        "domain_count": len({c.domain for c in conversations}),
    }
    with open(processed_dir / "dataset_stats.json", "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    return stats


def build_processed(config: Config) -> dict[str, Any]:
    raw_path = config.dataset.raw_path
    log.info("loading raw dataset from %s", raw_path)
    records = load_raw(raw_path)
    errors: list[str] = []
    for idx, record in enumerate(records):
        errors.extend(_validate_record(record, idx))
        if len(errors) > 20:
            break
    if errors:
        raise ValueError("schema validation failed:\n" + "\n".join(errors))
    conversations = list(iter_conversations(records))
    stats = write_processed(conversations, config)
    stats["source_hash"] = file_sha256(raw_path)
    stats["source_path"] = str(raw_path)
    with open(config.dataset.processed_path / "source_meta.json", "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    log.info("processed dataset: %s", stats)
    return stats
