from __future__ import annotations
import math
from collections import defaultdict
from system4.retrieval.index import LocalIndex, tokens


def search(index: LocalIndex, query: str, limit: int) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float); total = max(1, len(index.conversations))
    for term in set(tokens(query)):
        idf = math.log((total + 1) / (index.doc_freq.get(term, 0) + 1)) + 1
        for transcript_id, frequency in index.inverted.get(term, {}).items(): scores[transcript_id] += idf * (1 + math.log(frequency))
    return sorted(scores.items(), key=lambda row: (-row[1], row[0]))[:limit]
