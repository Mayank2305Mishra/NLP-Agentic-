"""Benchmark orchestration (AGENTS_SYSTEM_4.md §34–§38).

Runs a fixed question set against a fixed index with the LLM, prompts, and
schema held constant; only the retrieval/evidence architecture (or a single
component) varies per run. Supports repeats for variance and variant matrices
for reranker/compression comparisons.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from system4.evaluation.latency import aggregate_latency
from system4.evaluation.quality import evaluate_quality
from system4.evaluation.retrieval import evaluate_retrieval
from system4.schema import SYSTEM_VERSION


@dataclass
class BenchmarkConfig:
    repeats: int = 1
    detail: str = "detailed"
    cross_conversation: bool = False
    variant: str = "default"
    warm: bool = True


def _row_from(question: dict, result: Any, repeat: int) -> dict:
    answer = result.answer
    trace = result.trace
    expected = question.get("transcript_id") or question.get("expected")
    retrieved = trace.get("selected_transcript_ids", [])
    return {
        "id": question.get("id"),
        "category": question.get("category"),
        "task": question.get("task", "task1"),
        "query_hash": question.get("query_hash"),
        "repeat": repeat,
        "expected": expected,
        "expected_turns": question.get("expected_turns"),
        "answerable": question.get("answerable", True),
        "retrieved": retrieved,
        "retrieved_turns": trace.get("selected_turns", []),
        "hit": (expected in retrieved) if expected else None,
        "insufficient_evidence": bool(answer.get("insufficient_evidence")),
        "grounding": trace.get("grounding_status"),
        "grounding_ratio": trace.get("grounding_ratio"),
        "json_valid": trace.get("output_parse_valid", True),
        "complete": _complete(answer, question.get("task", "task1")),
        "rationale_count": len(answer.get("causal_rationales") or []),
        "latency_ms": trace.get("total_ms"),
        "retrieval_latency_ms": trace.get("candidate_retrieval_ms"),
        "llm_generation_ms": trace.get("llm_generation_ms"),
        "context_tokens": trace.get("context_tokens"),
        "generation_mode": trace.get("generation_mode"),
        "answer_gate": trace.get("answer_gate"),
        "trace": trace,
    }


def _complete(answer: dict, task: str) -> bool:
    required = {
        "task1": ("causal_rationales", "confidence", "insufficient_evidence"),
        "task2": ("summary", "key_causes", "effects", "mechanisms", "uncertainties", "evidence", "insufficient_evidence"),
    }.get(task, ())
    return all(key in answer for key in required)


def run_benchmark(rag: object, questions: list[dict], config: BenchmarkConfig | None = None) -> dict:
    config = config or BenchmarkConfig()
    rows: list[dict] = []
    traces: list[dict] = []
    started = time.perf_counter()
    for repeat in range(config.repeats):
        for question in questions:
            result = rag.ask(  # type: ignore[attr-defined]
                question["query"],
                question.get("task", "task1"),
                cross_conversation=config.cross_conversation,
                detail=config.detail,
            )
            row = _row_from(question, result, repeat)
            rows.append(row)
            traces.append(result.trace)
    wall_ms = round((time.perf_counter() - started) * 1000, 2)
    latency = aggregate_latency(traces)
    return {
        "system": "system4",
        "system_version": SYSTEM_VERSION,
        "variant": config.variant,
        "repeats": config.repeats,
        "warm": config.warm,
        "questions": len(questions),
        "rows": rows,
        "retrieval": evaluate_retrieval(rows),
        "quality": evaluate_quality(rows),
        "latency": latency,
        "wall_ms": wall_ms,
    }


def run_matrix(
    index: object,
    questions: list[dict],
    variants: dict[str, Any],
    base_settings: Any,
    source_hash: str = "",
    build_rag=None,
) -> dict:
    """Compare component variants with everything else fixed (§36, §37, §38).

    ``variants`` maps a variant name to a fully-built ``Settings`` instance.
    """
    results: dict[str, dict] = {}
    for name, settings in variants.items():
        rag = build_rag(settings) if build_rag else _default_build(index, settings, source_hash)
        report = run_benchmark(rag, questions, BenchmarkConfig(variant=name, repeats=1, warm=False))
        results[name] = report
    comparison = _compare(results)
    return {
        "system": "system4",
        "system_version": SYSTEM_VERSION,
        "kind": "matrix",
        "variants": results,
        "comparison": comparison,
    }


def _default_build(index: Any, settings: Any, source_hash: str) -> Any:
    from system4.pipeline import QualityRAG

    return QualityRAG(index, settings, source_hash=source_hash)


def _compare(results: dict[str, dict]) -> dict:
    rows = []
    for name, report in results.items():
        retrieval = report.get("retrieval", {})
        quality = report.get("quality", {})
        latency = report.get("latency", {}).get("stages", {}).get("total_ms", {})
        rows.append(
            {
                "variant": name,
                "recall@1": retrieval.get("recall@1"),
                "recall@5": retrieval.get("recall@5"),
                "mrr": retrieval.get("mrr"),
                "contamination_rate": retrieval.get("contamination_rate"),
                "grounded_rate": quality.get("grounded_rate"),
                "abstention_precision": quality.get("abstention_precision"),
                "answer_rate": quality.get("answer_rate"),
                "mean_total_ms": latency.get("mean"),
                "mean_context_tokens": report.get("latency", {}).get("context_tokens"),
            }
        )
    return {"rows": rows}
