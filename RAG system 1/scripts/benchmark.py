"""CLI: python -m scripts.benchmark — reproducible harness over QueryService."""
from __future__ import annotations

import argparse

from app.config import get_settings


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the System-1 benchmark.")
    ap.add_argument("--task", default="task1", choices=["task1", "task2"])
    ap.add_argument("--num-cases", type=int, default=20)
    ap.add_argument("--split", default="dev", choices=["dev", "val", "test", "all"])
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    settings = get_settings()
    from app.evaluation.benchmark import run_benchmark
    report = run_benchmark(settings, task=args.task, num_cases=args.num_cases,
                           split=args.split, top_k=args.top_k, out_path=args.out)
    import json
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2, default=str))
    print(f"cases={len(report['cases'])} saved={report.get('saved_to')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
