from __future__ import annotations

import json
import logging
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import Config
from ..pipeline import Pipeline
from .fixture import load_fixture

log = logging.getLogger("system2.benchmark")

RETRIEVAL_STAGE_KEYS = [
    "query_normalization_ms",
    "metadata_filter_ms",
    "fts_retrieval_ms",
    "conversation_ranking_ms",
    "turn_ranking_ms",
    "neighbor_expansion_ms",
    "evidence_pack_ms",
    "evidence_gate_ms",
]


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return round(ordered[idx], 3)


def _mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 4) if values else None


def _latency_summary(values: list[float]) -> dict[str, Any]:
    return {
        "mean_ms": _mean(values),
        "p50_ms": _percentile(values, 50),
        "p95_ms": _percentile(values, 95),
        "count": len(values),
    }


def run_benchmark(
    config: Config,
    fixture_path: str | None = None,
    use_llm: bool = True,
    limit: int | None = None,
    ablation: str | None = None,
) -> dict[str, Any]:
    if ablation:
        config.ablation = {"mode": ablation}
    fixture = load_fixture(config, fixture_path)
    questions = fixture["questions"][:limit] if limit else fixture["questions"]
    pipeline = Pipeline(config)

    rows: list[dict[str, Any]] = []
    retrieval_ms: list[float] = []
    generation_ms: list[float] = []
    total_ms: list[float] = []

    for question in questions:
        expected_tid = question["transcript_id"]
        expected_turns = set(question["expected_evidence_turns"])
        begin = time.perf_counter()
        outcome = _run_question(pipeline, question, use_llm)
        wall = round((time.perf_counter() - begin) * 1000, 3)

        candidate_ids = [c["transcript_id"] for c in outcome["candidates"]]
        rank = (
            candidate_ids.index(expected_tid) + 1 if expected_tid in candidate_ids else None
        )
        evidence_ids = [e["transcript_id"] for e in outcome["evidence"]]
        retrieved_turns: set[tuple[str, int]] = set()
        expected_retrieved: set[int] = set()
        for ev in outcome["evidence"]:
            for turn in ev["turns"]:
                key = (ev["transcript_id"], turn["turn_id"])
                retrieved_turns.add(key)
                if ev["transcript_id"] == expected_tid and turn["turn_id"] in expected_turns:
                    expected_retrieved.add(turn["turn_id"])
        relevant_total = len(expected_retrieved)
        retrieved_total = len(retrieved_turns)
        evidence_recall = relevant_total / len(expected_turns) if expected_turns else 0.0
        evidence_precision = relevant_total / retrieved_total if retrieved_total else 0.0
        irrelevant_rate = (
            sum(1 for tid in evidence_ids if tid != expected_tid) / len(evidence_ids)
            if evidence_ids
            else 1.0
        )

        trace = outcome["trace"]
        r_ms = sum(trace.get(k) or 0.0 for k in RETRIEVAL_STAGE_KEYS)
        g_ms = trace.get("generation_ms") or 0.0
        t_ms = trace.get("total_ms") or wall
        retrieval_ms.append(r_ms)
        if g_ms:
            generation_ms.append(g_ms)
        total_ms.append(t_ms)

        rows.append(
            {
                "question_id": question["question_id"],
                "category": question["category"],
                "expected_transcript_id": expected_tid,
                "rank": rank,
                "hit_at_1": rank == 1,
                "hit_at_3": rank is not None and rank <= 3,
                "selected_relevant": expected_tid in evidence_ids,
                "evidence_recall": round(evidence_recall, 4),
                "evidence_precision": round(evidence_precision, 4),
                "irrelevant_conversation_rate": round(irrelevant_rate, 4),
                "evidence_gate": outcome["gate"],
                "answer_gate": outcome.get("answer_gate"),
                "json_valid": outcome.get("json_valid"),
                "grounding_pass": outcome.get("grounding_pass"),
                "contamination": outcome.get("contamination", False),
                "insufficient_evidence": outcome.get("insufficient_evidence"),
                "unsupported_claim_rate": outcome.get("unsupported_claim_rate"),
                "retrieval_ms": round(r_ms, 3),
                "generation_ms": round(g_ms, 3),
                "total_ms": round(t_ms, 3),
                "errors": outcome.get("errors", []),
            }
        )

    summary = _summarize(rows, retrieval_ms, generation_ms, total_ms, use_llm)
    result = {
        "summary": summary,
        "rows": rows,
        "reproducibility": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "fixture_version": fixture.get("version"),
            "fixture_seed": fixture.get("seed"),
            "source_hash": fixture.get("source_hash"),
            "config_hash": config.config_hash,
            "dataset_population": "index",
            "sample_seed": config.dataset.sample_seed,
            "model": config.generation.model,
            "provider": config.generation.provider,
            "temperature": config.generation.temperature,
            "ablation": (config.ablation or {}).get("mode", "full"),
            "use_llm": use_llm,
            "question_count": len(questions),
        },
    }
    _save(config, result)
    return result


def _run_question(pipeline: Pipeline, question: dict[str, Any], use_llm: bool) -> dict[str, Any]:
    if not use_llm:
        outcome = pipeline.retrieve_only(question["question"], task=question["task"])
        retrieval = outcome["retrieval"]
        pack = retrieval.pack
        evidence_convs = pack.conversations if pack and pack.conversations else retrieval.evidence
        return {
            "candidates": [
                {"transcript_id": c.transcript_id} for c in retrieval.candidates
            ],
            "evidence": [
                {
                    "transcript_id": e.transcript_id,
                    "turns": [{"turn_id": t.turn_id} for t in e.turns],
                }
                for e in evidence_convs
            ],
            "gate": retrieval.gate.decision,
            "trace": outcome["trace"],
            "contamination": bool(
                pack and len(pack.allowed_transcript_ids) > 1 and not question.get("comparison")
            ),
            "errors": [],
        }

    result = pipeline.ask(question["question"], task=question["task"])
    answer = result.answer or {}
    claims = result.grounding.get("claims") or []
    unsupported = [c for c in claims if c.get("turns") and not c.get("supported")]
    claim_count = sum(1 for c in claims if c.get("turns"))
    return {
        "candidates": [{"transcript_id": c["transcript_id"]} for c in result.candidates],
        "evidence": [
            {"transcript_id": e["transcript_id"], "turns": e["turns"]} for e in result.evidence
        ],
        "gate": result.evidence_gate.get("decision"),
        "answer_gate": result.answer_gate.get("decision"),
        "trace": result.trace,
        "json_valid": result.parse_ok and result.answer is not None,
        "grounding_pass": result.grounding.get("passed", False),
        "contamination": result.grounding.get("contamination", False),
        "insufficient_evidence": bool(answer.get("insufficient_evidence")),
        "unsupported_claim_rate": round(len(unsupported) / claim_count, 4) if claim_count else 0.0,
        "errors": result.errors,
    }


def _summarize(
    rows: list[dict[str, Any]],
    retrieval_ms: list[float],
    generation_ms: list[float],
    total_ms: list[float],
    use_llm: bool,
) -> dict[str, Any]:
    n = len(rows) or 1
    summary: dict[str, Any] = {
        "questions": len(rows),
        "retrieval": {
            "hit_at_1_rate": round(sum(1 for r in rows if r["hit_at_1"]) / n, 4),
            "recall_at_3": round(sum(1 for r in rows if r["hit_at_3"]) / n, 4),
            "relevant_conversation_retrieval_rate": round(
                sum(1 for r in rows if r["selected_relevant"]) / n, 4
            ),
            "evidence_precision": _mean([r["evidence_precision"] for r in rows]),
            "evidence_recall": _mean([r["evidence_recall"] for r in rows]),
            "irrelevant_conversation_rate": _mean(
                [r["irrelevant_conversation_rate"] for r in rows]
            ),
            "contamination_rate": round(sum(1 for r in rows if r["contamination"]) / n, 4),
            "gate_insufficient_rate": round(
                sum(1 for r in rows if r["evidence_gate"] == "INSUFFICIENT") / n, 4
            ),
        },
        "latency": {
            "retrieval": _latency_summary(retrieval_ms),
            "generation": _latency_summary(generation_ms),
            "total": _latency_summary(total_ms),
        },
    }
    if use_llm:
        summary["generation"] = {
            "json_validity": round(sum(1 for r in rows if r["json_valid"]) / n, 4),
            "grounding_pass_rate": round(sum(1 for r in rows if r["grounding_pass"]) / n, 4),
            "unsupported_claim_rate": _mean(
                [r["unsupported_claim_rate"] for r in rows if r["unsupported_claim_rate"] is not None]
            ),
            "insufficient_evidence_rate": round(
                sum(1 for r in rows if r["insufficient_evidence"]) / n, 4
            ),
        }
    by_category: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_category.setdefault(row["category"], []).append(row)
    summary["by_category"] = {
        cat: {
            "count": len(items),
            "recall_at_3": round(sum(1 for r in items if r["hit_at_3"]) / len(items), 4),
            "selected_relevant": round(
                sum(1 for r in items if r["selected_relevant"]) / len(items), 4
            ),
        }
        for cat, items in by_category.items()
    }
    return summary


def _save(config: Config, result: dict[str, Any]) -> Path:
    out_dir = Path(__file__).resolve().parents[3] / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    mode = (config.ablation or {}).get("mode", "full")
    path = out_dir / f"{stamp}_{mode}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    log.info("benchmark results written to %s", path)
    return path
