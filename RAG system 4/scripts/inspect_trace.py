"""Inspect structured request traces (AGENTS_SYSTEM_4.md §40).

Usage:
    python scripts/inspect_trace.py                     # latest N requests
    python scripts/inspect_trace.py --last 50
    python scripts/inspect_trace.py --request-id <id>
    python scripts/inspect_trace.py --aggregate          # stage latency summary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from system4.evaluation.latency import aggregate_latency
from system4.observability import TraceLogger


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def print_request(trace: dict) -> None:
    print(
        f"[{trace.get('request_id', '?')[:8]}] task={trace.get('task')} "
        f"gate={trace.get('answer_gate')} grounding={trace.get('grounding_status')} "
        f"candidates={trace.get('candidate_conversations')} "
        f"selected={len(trace.get('selected_turns') or [])} "
        f"total={_fmt(trace.get('total_ms'))}ms"
    )
    stages = {k: v for k, v in trace.items() if k.endswith("_ms")}
    for name, value in sorted(stages.items(), key=lambda kv: -(kv[1] or 0)):
        print(f"    {name:<32} {_fmt(value):>10}")
    if trace.get("laya_decisions"):
        print(f"    laya: {trace.get('laya_decisions')} -> {trace.get('laya_answer_gates')} ({trace.get('laya_reason')})")
    if trace.get("errors"):
        for err in trace["errors"]:
            print(f"    ERROR {err.get('stage')}: {err.get('error')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect System 4 traces")
    parser.add_argument("--trace-dir", default=None, help="defaults to SYSTEM4_TRACE_DIR or artifacts/traces")
    parser.add_argument("--last", type=int, default=5)
    parser.add_argument("--request-id", default=None)
    parser.add_argument("--aggregate", action="store_true", help="stage latency summary across all traces")
    args = parser.parse_args()

    trace_dir = args.trace_dir
    logger = TraceLogger(directory=trace_dir, enabled=True)
    traces = logger.read()
    if not traces:
        print("no traces found (did the pipeline run with trace_enabled?)")
        return

    if args.aggregate:
        stats = aggregate_latency(traces)
        print(json.dumps(stats, indent=2))
        return
    if args.request_id:
        matched = [t for t in traces if t.get("request_id") == args.request_id]
        if not matched:
            print(f"request {args.request_id} not found")
            return
        for trace in matched:
            print_request(trace)
        return
    for trace in traces[-args.last:]:
        print_request(trace)
        print()


if __name__ == "__main__":
    main()
