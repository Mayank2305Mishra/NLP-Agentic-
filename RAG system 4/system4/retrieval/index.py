from __future__ import annotations

from collections import Counter, defaultdict
import hashlib, math, re
from dataclasses import dataclass

from system4.config.settings import Settings
from system4.schema import Conversation

TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.I)


def tokens(text: str) -> list[str]: return TOKEN.findall(text.lower())


@dataclass
class LocalIndex:
    conversations: dict[str, Conversation]
    inverted: dict[str, dict[str, int]]
    doc_freq: dict[str, int]
    vectors: dict[str, dict[int, float]]
    dimensions: int

    @classmethod
    def build(cls, conversations: list[Conversation], settings: Settings) -> "LocalIndex":
        docs, inverted, df, vectors = {}, defaultdict(dict), Counter(), {}
        for conversation in conversations:
            docs[conversation.transcript_id] = conversation
            terms = tokens(" ".join(filter(None, [conversation.domain or "", conversation.intent or "", conversation.reason_for_call or "", *(t.text for t in conversation.turns)])))
            counts = Counter(terms)
            for term, count in counts.items(): inverted[term][conversation.transcript_id] = count
            df.update(counts)
            vectors[conversation.transcript_id] = _vector(counts, settings.hash_dimensions)
        return cls(docs, dict(inverted), dict(df), vectors, settings.hash_dimensions)


def _bucket(term: str, dimensions: int) -> int:
    # Stable across processes so persisted artifacts are reproducible.
    return int.from_bytes(hashlib.blake2b(term.encode(), digest_size=8).digest(), "big") % dimensions


def _vector(counts: Counter[str], dimensions: int) -> dict[int, float]:
    vector: dict[int, float] = defaultdict(float)
    for term, count in counts.items(): vector[_bucket(term, dimensions)] += 1.0 + math.log(count)
    norm = math.sqrt(sum(v * v for v in vector.values())) or 1.0
    return {k: v / norm for k, v in vector.items()}


def query_vector(query: str, dimensions: int) -> dict[int, float]: return _vector(Counter(tokens(query)), dimensions)
