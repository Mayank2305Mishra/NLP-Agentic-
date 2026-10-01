"""Reproducible benchmark harness over the canonical QueryService.

Records run_id, git/source hashes, all configs, hardware, per-case latency
split (retrieval vs generation), and quality proxies. Same protocol is meant
for all four future RAG systems.
"""
from __future__ import annotations

import json
import platform
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.config import Settings
from app.evaluation.metrics import latency_stats, quality_summary

# Fixed demo evaluation questions (dev-style; NOT a frozen test set).
DEFAULT_QUESTIONS = [
    ("task1", "Why did customers request urgent changes or express stress about their bookings?"),
    ("task1", "What caused discount or promotion complaints and what was the effect?"),
    ("task1", "Extract the cause and effect when a customer missed a flight or connection."),
    ("task1", "Why did insurance customers call about claims and what happened as a result?"),
    ("task2", "Synthesize the main business reasons for urgent booking changes and their effects."),
    ("task2", "Summarize promotion-related issues and their business impact across conversations."),
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                       text=True).strip()[:12]
    except Exception:
        return "nogit"


def run_benchmark(settings: Settings, *, task: str = "task1", num_cases: int = 20,
                  split: str = "dev", top_k: int | None = None,
                  out_path: str | None = None,
                  questions: list[tuple[str, str]] | None = None) -> dict:
    from app.indexing.builder import verify_manifest
    from app.ingestion.loader import load_raw_records
    from app.main import QueryService

    run_id = uuid.uuid4().hex[:12]
    manifest = verify_manifest(settings, allow_stale=True)
    _, source_sha, _ = load_raw_records(settings.source_path)
    qs = questions or DEFAULT_QUESTIONS
    qs = [(t, q) for t, q in qs if t == task][: max(1, num_cases)]
    svc = QueryService(settings)
    cases: list[dict] = []
    for i, (t, q) in enumerate(qs):
        c0 = time.perf_counter()
        out = svc.run(task=t, query=q, top_k=top_k, save_trace=False)
        wall = (time.perf_counter() - c0) * 1000.0
        m = out.get("metrics", {})
        res = out.get("result") or {}
        cases.append({
            "case_id": i, "task": t, "query": q, "success": bool(out.get("success")),
            "wall_ms": round(wall, 2),
            "retrieval_ms": m.get("retrieval_ms", 0.0),
            "generation_ms": m.get("generation_ms", 0.0),
            "total_ms": m.get("total_ms", wall),
            "num_chunks": m.get("num_chunks", 0),
            "structured_valid": out.get("success", False) and isinstance(res, dict),
            "citations_valid": out.get("success", False),
            "grounded": bool(out.get("success")) and not res.get("insufficient_evidence", False),
            "insufficient_evidence": bool(res.get("insufficient_evidence", False)),
            "error": (out.get("error") or {}).get("code"),
        })
    totals = [c["total_ms"] for c in cases]
    report = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system": "System 1: Traditional Dense Vector RAG",
        "git_commit": _git_commit(),
        "source_sha256": source_sha,
        "index_manifest_hash": manifest.chunking_config_hash,
        "embedding_model": manifest.embedding_model,
        "embedding_dimension": manifest.embedding_dimension,
        "embedding_normalize": manifest.embedding_normalize,
        "index_type": manifest.index_type,
        "num_chunks": manifest.num_chunks,
        "chunking_config": manifest.chunking_config,
        "retrieval_config": settings.retrieval_config(),
        "generation": settings.generation_config(),
        "laya_config": {"enabled": settings.laya_enabled, "mode": settings.laya_mode},
        "prompt_versions": {"task1": settings.prompt_version_task1,
                            "task2": settings.prompt_version_task2},
        "hardware": {"platform": platform.platform(), "cpu": platform.processor(),
                     "python": platform.python_version()},
        "task": task, "split": split,
        "latency": latency_stats(totals),
        "retrieval_latency": latency_stats([c["retrieval_ms"] for c in cases]),
        "generation_latency": latency_stats([c["generation_ms"] for c in cases]),
        "quality_proxies": quality_summary(cases),
        "note": "Quality figures are engineering proxies, NOT official hackathon scores.",
        "cases": cases,
    }
    dest = Path(out_path) if out_path else (
        Path(settings.benchmarks_dir) / f"benchmark_{task}_{run_id}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["saved_to"] = str(dest)
    return report
