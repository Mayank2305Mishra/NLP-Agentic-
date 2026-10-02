from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "benchmarks" / "datasets"
SPLITS = ROOT / "benchmarks" / "splits"


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def normalise_fixture(path: Path) -> list[dict[str, Any]]:
    """Convert the existing fixture format without inventing answers or evidence."""
    raw = json.loads(path.read_text())
    source = raw["questions"] if isinstance(raw, dict) else raw
    rows = []
    for item in source:
        rows.append({
            "question_id": item["question_id"], "task": item.get("task", "task1"),
            "query": item.get("query", item.get("question", "")),
            "transcript_ids": [item["transcript_id"]] if item.get("transcript_id") else [],
            "gold_turns": item.get("gold_turns", item.get("expected_evidence_turns", [])),
            "gold_evidence": item.get("gold_evidence", []),
            "reference_answer": item.get("reference_answer"),
            "gold_causes": item.get("gold_causes", [item["expected_cause"]] if item.get("expected_cause") else []),
            "gold_effects": item.get("gold_effects", [item["expected_effect"]] if item.get("expected_effect") else []),
            "gold_outcomes": item.get("gold_outcomes", []),
            "difficulty": item.get("difficulty", "unknown"),
            "metadata": {"category": item.get("category"), "anchor_terms": item.get("anchor_terms", [])},
        })
    return rows


def build(fixture: Path, seed: int = 42) -> dict[str, Any]:
    rows = normalise_fixture(fixture)
    ids = [row["question_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("question_id values must be unique")
    shuffled = rows[:]
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    dev_end, test_end = max(1, int(n * .6)), max(2, int(n * .8))
    groups = {"dev": shuffled[:dev_end], "test": shuffled[dev_end:test_end], "stress": shuffled[test_end:]}
    DATASETS.mkdir(parents=True, exist_ok=True)
    _write_jsonl(DATASETS / "benchmark_questions.jsonl", rows)
    for name, split_rows in groups.items():
        _write_jsonl(SPLITS / f"{name}.jsonl", split_rows)
    manifest = {
        "benchmark_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "source_fixture": str(fixture), "benchmark_hash": _hash(rows), "seed": seed,
        "counts": {name: len(split_rows) for name, split_rows in groups.items()},
        "split_hashes": {name: _hash(split_rows) for name, split_rows in groups.items()},
    }
    (DATASETS / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    validate()
    return manifest


def validate() -> dict[str, Any]:
    split_rows = {name: _read_jsonl(SPLITS / f"{name}.jsonl") for name in ("dev", "test", "stress")}
    seen_questions: set[str] = set()
    seen_transcripts: set[str] = set()
    for name, rows in split_rows.items():
        for row in rows:
            required = {"question_id", "task", "query", "transcript_ids", "gold_turns"}
            missing = required - row.keys()
            if missing or not row["query"]:
                raise ValueError(f"{name}: invalid item {row.get('question_id')}: missing {sorted(missing)}")
            if row["question_id"] in seen_questions:
                raise ValueError(f"question leak: {row['question_id']}")
            seen_questions.add(row["question_id"])
            overlap = set(row["transcript_ids"]) & seen_transcripts
            if overlap:
                raise ValueError(f"transcript leak into {name}: {sorted(overlap)}")
            seen_transcripts.update(row["transcript_ids"])
    return {"valid": True, "questions": len(seen_questions), "transcripts": len(seen_transcripts)}


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--fixture", required=True, type=Path)
    build_parser.add_argument("--seed", type=int, default=42)
    sub.add_parser("validate")
    args = parser.parse_args()
    result = build(args.fixture, args.seed) if args.command == "build" else validate()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
