"""Run the reproducible System 4 benchmark.

Usage:
    python scripts/run_benchmark.py --index artifacts/index.pkl \\
        --questions benchmarks/questions.json --output artifacts/benchmark

Add ``--matrix`` to compare reranker and compression variants (§36, §37).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from system4.config.settings import Settings
from system4.evaluation.benchmark import BenchmarkConfig, run_benchmark, run_matrix
from system4.evaluation.reports import write
from system4.ingestion.artifacts import load_artifact
from system4.pipeline import QualityRAG
from system4.schema import SYSTEM_VERSION


def _variants(base: Settings) -> dict[str, Settings]:
    variants = {
        "no_reranker": dataclasses.replace(base, reranker="none", cache_answers=False),
        "feature_reranker": dataclasses.replace(base, reranker="feature", cache_answers=False),
        "no_compression": dataclasses.replace(base, default_compression="NONE", cache_answers=False),
        "light_compression": dataclasses.replace(base, default_compression="LIGHT", cache_answers=False),
        "aggressive_compression": dataclasses.replace(base, default_compression="AGGRESSIVE", cache_answers=False),
    }
    return variants


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible System 4 benchmark")
    parser.add_argument("--index", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", default="artifacts/benchmark")
    parser.add_argument("--repeats", type=int, default=1, help="repeat queries for variance (§34)")
    parser.add_argument("--detail", default="detailed", choices=["concise", "detailed", "thorough"])
    parser.add_argument("--matrix", action="store_true", help="compare reranker/compression variants")
    parser.add_argument("--reranker", default=None, choices=["none", "feature", "cross_encoder"])
    parser.add_argument("--compression", default=None, choices=["NONE", "LIGHT", "AGGRESSIVE"])
    args = parser.parse_args()

    artifact = load_artifact(args.index, expected_version=SYSTEM_VERSION)
    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    settings = Settings(cache_answers=False)
    if args.reranker:
        settings = dataclasses.replace(settings, reranker=args.reranker)
    if args.compression:
        settings = dataclasses.replace(settings, default_compression=args.compression)

    if args.matrix:
        report = run_matrix(
            artifact["index"],
            questions,
            _variants(settings),
            settings,
            source_hash=artifact.get("source_hash", ""),
        )
    else:
        rag = QualityRAG(artifact["index"], settings, source_hash=artifact.get("source_hash", ""))
        report = run_benchmark(
            rag, questions, BenchmarkConfig(repeats=args.repeats, detail=args.detail)
        )
    json_path, md_path = write(args.output, report)
    if report.get("kind") == "matrix":
        summary = {"variants": list(report["variants"]), "comparison": report["comparison"]["rows"]}
    else:
        summary = {k: v for k, v in report.items() if k not in {"rows"}}
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
