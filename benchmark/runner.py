from __future__ import annotations

import argparse
import json
import platform
import shutil
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .datasets import _read_jsonl, validate
from .metrics import aggregate, per_question
from .registry import answer

ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def run(systems: list[str], split: str, limit: int | None, question_id: str | None, seed: int, stage: str) -> Path:
    validate()
    questions = _read_jsonl(ROOT / "benchmarks" / "splits" / f"{split}.jsonl")
    if question_id:
        questions = [q for q in questions if q["question_id"] == question_id]
    if limit:
        questions = questions[:limit]
    if not questions:
        raise ValueError("no matching benchmark questions")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{seed}"
    directory = ROOT / "benchmarks" / "runs" / run_id
    directory.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    for question in questions:
        for system in systems:
            started = time.perf_counter()
            try:
                result = answer(system, question)
                trace = result.setdefault("trace", {})
                trace.setdefault("timing_ms", {}).setdefault("request", round((time.perf_counter() - started) * 1000, 3))
                records.append({"run_id": run_id, "question_id": question["question_id"], "system": system,
                                "task": question["task"], "answer": result["answer"], "evidence": result["evidence"],
                                "trace": trace, "usage": result["usage"], "system_metadata": result["system_metadata"],
                                "metrics": per_question(question, result), "errors": []})
            except Exception as exc:
                records.append({"run_id": run_id, "question_id": question["question_id"], "system": system,
                                "task": question["task"], "errors": [str(exc)], "metrics": {}})
    _write(directory / "questions.jsonl", questions)
    _write(directory / "results.jsonl", records)
    _write(directory / "failures.jsonl", [r for r in records if r["errors"]])
    _write(directory / "traces.jsonl", [{"question_id": r["question_id"], "system": r["system"], "trace": r.get("trace", {})} for r in records])
    _write(directory / "retrieval_results.jsonl", [{"question_id": r["question_id"], "system": r["system"], "evidence": r.get("evidence", []), "metrics": r.get("metrics", {})} for r in records])
    summary = {system: aggregate([r for r in records if r["system"] == system]) for system in systems}
    (directory / "aggregate_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    (directory / "latency_metrics.json").write_text(json.dumps({s: v["latency_ms"] for s, v in summary.items()}, indent=2) + "\n")
    (directory / "token_metrics.json").write_text(json.dumps({"note": "Provider token counters are retained per query in results.jsonl; unavailable values are null."}, indent=2) + "\n")
    (directory / "quality_metrics.json").write_text(json.dumps({s: {k: v[k] for k in ("evidence_precision", "evidence_recall", "contamination_rate") } for s, v in summary.items()}, indent=2) + "\n")
    (directory / "contamination_metrics.json").write_text(json.dumps({s: v["contamination_rate"] for s, v in summary.items()}, indent=2) + "\n")
    configs = directory / "configs"; configs.mkdir()
    for system in systems:
        shutil.copy2(ROOT / "benchmark" / "configs" / f"{system}.yaml", configs / f"{system}.yaml")
    manifest = {"run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(), "split": split,
                "systems": systems, "seed": seed, "stage": stage, "python": platform.python_version(),
                "platform": platform.platform(), "questions": len(questions)}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (directory / "environment.json").write_text(json.dumps({"python": platform.python_version(), "platform": platform.platform()}, indent=2) + "\n")
    return directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired, contract-based RAG benchmark runner")
    parser.add_argument("--systems", nargs="+", required=True)
    parser.add_argument("--split", default="test", choices=("dev", "test", "stress"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--question-id")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage", default="full", choices=("full", "retrieval"))
    args = parser.parse_args()
    print(run(args.systems, args.split, args.limit, args.question_id, args.seed, args.stage))


if __name__ == "__main__":
    main()
