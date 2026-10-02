"""Cache keying and bounded caches (AGENTS_SYSTEM_4.md §39).

Cache keys always include source hash, system version, query hash, retrieval
config hash, reranker version, prompt version, and model identifier — final
answers are never cached across different evidence versions without them.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from typing import Any


def stable_hash(*parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def answer_cache_key(
    *,
    source_hash: str,
    system_version: str,
    query: str,
    task: str,
    cross_conversation: bool,
    detail: str,
    settings_fingerprint: str,
    reranker: str,
    prompt_version: str,
    model_identifier: str,
) -> str:
    """Full §39 key for final-answer caching."""
    return stable_hash(
        source_hash,
        system_version,
        hashlib.sha256(query.strip().lower().encode()).hexdigest(),
        task,
        cross_conversation,
        detail,
        settings_fingerprint,
        reranker,
        prompt_version,
        model_identifier,
    )


class AnswerCache:
    """Small thread-safe LRU cache for final answers (§39)."""

    def __init__(self, maxsize: int = 256) -> None:
        self.maxsize = max(1, maxsize)
        self._data: OrderedDict[str, tuple] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> tuple | None:
        with self._lock:
            value = self._data.get(key)
            if value is not None:
                self._data.move_to_end(key)
            return value

    def set(self, key: str, value: tuple) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        return len(self._data)
