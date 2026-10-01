from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import load_config


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def cmd_build_data(args: argparse.Namespace) -> int:
    from .dataset import build_processed

    config = load_config(args.config)
    stats = build_processed(config)
    print(json.dumps(stats, indent=2))
    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    from .sampling import stratified_sample

    config = load_config(args.config)
    manifest = stratified_sample(config)
    slim = {k: v for k, v in manifest.items() if k != "selected_transcript_ids"}
    print(json.dumps(slim, indent=2))
    return 0


def cmd_build_index(args: argparse.Namespace) -> int:
    from .indexer import build_index

    config = load_config(args.config)
    stats = build_index(config, population=args.population, reset=args.reset)
    print(json.dumps(stats, indent=2))
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from .pipeline import Pipeline

    config = load_config(args.config)
    if args.ablation:
        config.ablation = {"mode": args.ablation}
    pipeline = Pipeline(config)
    result = pipeline.ask(args.question, task=args.task)
    output = result.to_dict()
    if not args.full:
        output.pop("raw_output", None)
    print(json.dumps(output, indent=2, ensure_ascii=False, default=str))
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from .benchmarking.runner import run_benchmark

    config = load_config(args.config)
    results = run_benchmark(
        config,
        fixture_path=args.fixture,
        use_llm=not args.retrieval_only,
        limit=args.limit,
        ablation=args.ablation,
    )
    print(json.dumps(results["summary"], indent=2, default=str))
    return 0


def cmd_fixture(args: argparse.Namespace) -> int:
    from .benchmarking.fixture import build_fixture

    config = load_config(args.config)
    path = build_fixture(config, count=args.count, seed=args.seed)
    print(str(path))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import subprocess

    app = args.app
    cmd = [sys.executable, "-m", "streamlit", "run", app, "--server.port", str(args.port)]
    return subprocess.call(cmd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="system2", description="System 2 lexical RAG")
    parser.add_argument("--config", default=None, help="path to config yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-data", help="validate raw dataset and write processed artifacts")
    p.set_defaults(func=cmd_build_data)

    p = sub.add_parser("sample", help="stratified conversation-level sample + manifest")
    p.set_defaults(func=cmd_sample)

    p = sub.add_parser("build-index", help="build SQLite FTS/BM25 index")
    p.add_argument("--population", choices=["full", "sample"], default="full")
    p.add_argument("--reset", action="store_true", help="drop existing index first")
    p.set_defaults(func=cmd_build_index)

    p = sub.add_parser("ask", help="run one question end to end")
    p.add_argument("question")
    p.add_argument("--task", choices=["task1", "task2"], default=None)
    p.add_argument("--ablation", default=None)
    p.add_argument("--full", action="store_true", help="include raw model output")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("benchmark", help="run the labeled fixture benchmark")
    p.add_argument("--fixture", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--retrieval-only", action="store_true")
    p.add_argument("--ablation", default=None)
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("fixture", help="build the retrieval evaluation fixture")
    p.add_argument("--count", type=int, default=64)
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=cmd_fixture)

    p = sub.add_parser("serve", help="launch the Streamlit UI")
    p.add_argument("--app", default="app/streamlit_app.py")
    p.add_argument("--port", type=int, default=8501)
    p.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging("INFO")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
