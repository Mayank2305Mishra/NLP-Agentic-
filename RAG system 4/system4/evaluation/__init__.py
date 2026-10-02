"""Evaluation package: retrieval, quality, latency, benchmark, reports (§31–§38)."""

from system4.evaluation.benchmark import BenchmarkConfig, run_benchmark, run_matrix
from system4.evaluation.latency import aggregate_latency
from system4.evaluation.quality import evaluate_quality
from system4.evaluation.retrieval import evaluate_retrieval
from system4.evaluation.reports import markdown_summary, write, write_json

__all__ = [
    "BenchmarkConfig",
    "aggregate_latency",
    "evaluate_quality",
    "evaluate_retrieval",
    "markdown_summary",
    "run_benchmark",
    "run_matrix",
    "write",
    "write_json",
]
