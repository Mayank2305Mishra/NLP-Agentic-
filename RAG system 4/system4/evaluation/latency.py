"""Latency aggregation over structured traces (AGENTS_SYSTEM_4.md §32, §33)."""

from __future__ import annotations

import statistics

#: Canonical stage order from §32 (extra stages are appended automatically).
STAGE_ORDER = (
    "query_normalization_ms",
    "task_route_ms",
    "metadata_filter_ms",
    "dense_retrieval_ms",
    "lexical_retrieval_ms",
    "candidate_merge_ms",
    "candidate_retrieval_ms",
    "contamination_gate_ms",
    "reranker_ms",
    "turn_selection_ms",
    "expansion_ms",
    "laya_ms",
    "laya_redecide_ms",
    "compression_ms",
    "evidence_pack_ms",
    "prompt_build_ms",
    "llm_ttft_ms",
    "llm_generation_ms",
    "output_parse_ms",
    "grounding_validation_ms",
    "causal_validation_ms",
    "consistency_validation_ms",
    "answer_gate_ms",
    "total_ms",
)


def _stats(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "p50": None, "p95": None, "max": None}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "mean": round(statistics.fmean(values), 2),
        "p50": round(statistics.median(values), 2),
        "p95": round(ordered[p95_index], 2),
        "max": round(ordered[-1], 2),
    }


def aggregate_latency(traces: list[dict]) -> dict:
    stage_names: list[str] = []
    for name in STAGE_ORDER:
        if any(name in trace for trace in traces):
            stage_names.append(name)
    for trace in traces:
        for key in trace:
            if key.endswith("_ms") and key not in stage_names:
                stage_names.append(key)

    stages = {
        name: _stats(
            [float(trace[name]) for trace in traces if isinstance(trace.get(name), (int, float))]
        )
        for name in stage_names
    }
    input_tokens = [float(t.get("input_tokens", 0) or 0) for t in traces]
    output_tokens = [float(t.get("output_tokens", 0) or 0) for t in traces]
    context_tokens = [float(t.get("context_tokens", 0) or 0) for t in traces]
    tps = [
        float(t.get("generation_tokens_per_second", 0) or 0)
        for t in traces
        if t.get("generation_tokens_per_second")
    ]
    return {
        "requests": len(traces),
        "stages": stages,
        "input_tokens": round(statistics.fmean(input_tokens), 1) if input_tokens else 0,
        "output_tokens": round(statistics.fmean(output_tokens), 1) if output_tokens else 0,
        "context_tokens": round(statistics.fmean(context_tokens), 1) if context_tokens else 0,
        "generation_tokens_per_second": round(statistics.fmean(tps), 2) if tps else 0,
        "error_requests": sum(1 for t in traces if t.get("errors")),
    }
