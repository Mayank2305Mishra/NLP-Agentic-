from __future__ import annotations
from system4.retrieval.index import LocalIndex, query_vector


def search(index: LocalIndex, query: str, limit: int) -> list[tuple[str, float]]:
    vector = query_vector(query, index.dimensions)
    scores = [(tid, sum(weight * index.vectors[tid].get(key, 0.0) for key, weight in vector.items())) for tid in index.conversations]
    return sorted(scores, key=lambda row: (-row[1], row[0]))[:limit]
