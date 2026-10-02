"""Broad candidate retrieval: dense + lexical with reciprocal-rank fusion (§8, §9).

Both streams stay independently observable in traces. Raw scores from
incompatible retrievers are never averaged; fusion is rank-based, and the
per-stream scores are preserved on the candidate for later normalization.
"""

from __future__ import annotations

import time

from system4.config.settings import Settings
from system4.retrieval import dense, lexical
from system4.retrieval.index import LocalIndex
from system4.schema import Candidate


def _now() -> float:
    return time.perf_counter()


def _stamp(timings: dict[str, float] | None, name: str, started: float) -> None:
    if timings is not None:
        timings[name] = round((_now() - started) * 1000, 2)


def retrieve(
    index: LocalIndex,
    query: str,
    settings: Settings,
    timings: dict[str, float] | None = None,
) -> list[Candidate]:
    """Return a rank-fused candidate pool traceable to its retrieval sources."""
    started = _now()
    try:
        dense_results = dense.search(index, query, settings.dense_k)
        dense_error: str | None = None
    except Exception as exc:  # §43: fall back to lexical candidates
        dense_results = []
        dense_error = f"{type(exc).__name__}: {exc}"
    _stamp(timings, "dense_retrieval_ms", started)
    if dense_error and timings is not None:
        timings.setdefault("dense_fallbacks", 0)
        timings["dense_fallbacks"] += 1

    started = _now()
    try:
        lexical_results = lexical.search(index, query, settings.lexical_k)
        lexical_error: str | None = None
    except Exception as exc:  # §43: fall back to dense candidates
        lexical_results = []
        lexical_error = f"{type(exc).__name__}: {exc}"
    _stamp(timings, "lexical_retrieval_ms", started)
    if lexical_error and timings is not None:
        timings.setdefault("lexical_fallbacks", 0)
        timings["lexical_fallbacks"] += 1

    started = _now()
    merged: dict[str, Candidate] = {}
    for source, results in (("dense", dense_results), ("lexical", lexical_results)):
        for rank, (transcript_id, score) in enumerate(results, 1):
            candidate = merged.setdefault(transcript_id, Candidate(transcript_id))
            if source not in candidate.source_retrievers:
                candidate.source_retrievers.append(source)
            if source == "dense":
                candidate.dense_score = score
            else:
                candidate.lexical_score = score
            # Rank-based fusion: no incompatible score averaging (§9).
            candidate.fusion_score += 1.0 / (settings.rrf_k + rank)
    output = sorted(
        merged.values(), key=lambda item: (-item.fusion_score, item.transcript_id)
    )
    _stamp(timings, "candidate_merge_ms", started)
    return output
