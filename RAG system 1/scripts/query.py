"""CLI: python -m scripts.query — same QueryService as UI/benchmark/tests.

Usage:
  python -m scripts.query --task task1 --query "..." [--domain X --intent Y --top-k 8]
  python -m scripts.query --health
  python -m scripts.query --task task1 --query "..." --debug-trace
"""
from __future__ import annotations

import argparse
import json
import sys

from app.config import get_settings


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run one RAG query through the canonical pipeline.")
    ap.add_argument("--task", default="task1", choices=["task1", "task2"])
    ap.add_argument("--query", default="")
    ap.add_argument("--domain", default=None)
    ap.add_argument("--intent", default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--health", action="store_true")
    ap.add_argument("--debug-trace", action="store_true")
    ap.add_argument("--allow-stale-index", action="store_true")
    args = ap.parse_args(argv)
    settings = get_settings()
    if args.allow_stale_index:
        settings.allow_stale_index = True

    if args.health:
        from app.observability.health import check_health
        report = check_health(settings)
        print(json.dumps(report, indent=2))
        return 0 if report["overall_ok"] else 2

    if not args.query.strip():
        print("Empty query: pass --query \"...\"", file=sys.stderr)
        return 2
    from app.main import QueryService
    svc = QueryService(settings)
    out = svc.run(task=args.task, query=args.query, domain=args.domain,
                  intent=args.intent, top_k=args.top_k)
    if args.debug_trace:
        print(json.dumps(out, indent=2, default=str)[:50_000])
    else:
        slim = {k: out.get(k) for k in ("request_id", "task", "success", "result", "info",
                                        "metrics", "evidence_truncated", "error")}
        slim["evidence_ids"] = [c["chunk_id"] for c in out.get("evidence", [])]
        print(json.dumps(slim, indent=2, default=str)[:50_000])
    return 0 if out.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
