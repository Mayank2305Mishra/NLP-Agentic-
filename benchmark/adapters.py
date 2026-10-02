"""JSON-stdin adapters; these translate public system results, not retrieval internals."""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _require_ollama() -> None:
    """Fail before a per-question retry loop when the required generator is down."""
    try:
        with socket.create_connection(("127.0.0.1", 11434), timeout=.5):
            return
    except OSError as exc:
        raise RuntimeError("required Ollama generator at 127.0.0.1:11434 is unavailable") from exc


def _timing(trace: dict[str, Any], total: float | None = None) -> dict[str, Any]:
    stages = trace.get("stages", [])
    values = {item.get("stage"): item.get("latency_ms") for item in stages if isinstance(item, dict)}
    values["request"] = trace.get("total_ms", total)
    return values


def _evidence(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep IDs/scores/turns only: the shared suite need not duplicate transcript text."""
    rows: list[dict[str, Any]] = []
    for item in items:
        if "turn" in item and isinstance(item["turn"], dict):
            item = {**item, **item["turn"]}
        turns = item.get("turns") if isinstance(item.get("turns"), list) else [item]
        for turn in turns:
            rows.append({"transcript_id": item.get("transcript_id", turn.get("transcript_id")),
                         "turn_id": turn.get("turn_id", turn.get("index", item.get("turn_start"))),
                         "turn_start": item.get("turn_start"), "turn_end": item.get("turn_end"),
                         "score": item.get("score")})
    return rows


def system1(payload: dict[str, Any]) -> dict[str, Any]:
    _require_ollama()
    project = ROOT / "RAG system 1"; os.chdir(project); sys.path.insert(0, str(project))
    from app.config import get_settings
    from app.main import QueryService
    out = QueryService(get_settings()).run(task=payload["task"], query=payload["query"], save_trace=False)
    if not out.get("success"):
        raise RuntimeError((out.get("error") or {}).get("message", "System 1 request failed"))
    return {"answer": out.get("result") or {}, "evidence": _evidence(out.get("evidence", [])),
            "trace": {"timing_ms": _timing(out.get("trace", {}), out.get("metrics", {}).get("total_ms")), "raw": out.get("trace", {})},
            "usage": out.get("metrics", {}), "system_metadata": out.get("system", {})}


def lexical(payload: dict[str, Any], name: str) -> dict[str, Any]:
    _require_ollama()
    project = ROOT / f"RAG system {name[-1]}"; os.chdir(project); sys.path.insert(0, str(project / "src"))
    module = __import__(f"system{name[-1]}.config", fromlist=["load_config"])
    pipeline_module = __import__(f"system{name[-1]}.pipeline", fromlist=["Pipeline"])
    result = pipeline_module.Pipeline(module.load_config()).ask(payload["query"], task=payload["task"]).to_dict()
    return {"answer": result.get("answer") or {}, "evidence": _evidence(result.get("evidence", [])),
            "trace": {"timing_ms": {**result.get("trace", {}), "request": result.get("trace", {}).get("total_ms")}, "raw": result.get("trace", {})},
            "usage": result.get("tokens", {}), "system_metadata": {"model": result.get("model"), "provider": result.get("provider"), "config_hash": result.get("config_hash")}}


def system4(payload: dict[str, Any]) -> dict[str, Any]:
    project = ROOT / "RAG system 4"; os.chdir(project); sys.path.insert(0, str(project))
    from system4.ingestion.artifacts import load
    from system4.pipeline import QualityRAG
    index = project / "artifacts" / "benchmark-index.pkl"
    if not index.exists():
        raise RuntimeError(f"System 4 index is missing: {index}; build it before the benchmark")
    result = QualityRAG(load(index)["index"]).ask(payload["query"], payload["task"]).public()
    trace = result["trace"]
    return {"answer": result["answer"], "evidence": _evidence(result["evidence"]),
            "trace": {"timing_ms": {**trace, "request": trace.get("total_ms")}, "raw": trace},
            "usage": {"context": trace.get("context_tokens"), "input": None, "output": None, "total": None},
            "system_metadata": {"index": str(index), "generation": "deterministic fallback unless OLLAMA_MODEL is set"}}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--system", required=True, choices=("system1", "system2", "system3", "system4")); args = parser.parse_args()
    payload = json.load(sys.stdin)
    fn = {"system1": system1, "system2": lambda p: lexical(p, "system2"), "system3": lambda p: lexical(p, "system3"), "system4": system4}[args.system]
    try:
        print(json.dumps(fn(payload), default=str))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
