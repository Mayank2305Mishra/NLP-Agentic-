from __future__ import annotations

import logging
import time
from collections import OrderedDict
from typing import Any, Callable

log = logging.getLogger("system2.cache")


class LRUCache:
    def __init__(self, max_entries: int = 256, enabled: bool = True):
        self.max_entries = max_entries
        self.enabled = enabled
        self._data: OrderedDict[str, Any] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        if key in self._data:
            self._data.move_to_end(key)
            self.hits += 1
            return self._data[key]
        self.misses += 1
        return None

    def set(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self.max_entries:
            self._data.popitem(last=False)

    def memo(self, key: str, factory: Callable[[], Any]) -> Any:
        cached = self.get(key)
        if cached is not None:
            return cached
        value = factory()
        self.set(key, value)
        return value

    def stats(self) -> dict[str, Any]:
        return {"hits": self.hits, "misses": self.misses, "size": len(self._data), "enabled": self.enabled}


def index_signature(db_path) -> tuple[str, float, int]:
    try:
        stat = time.time()
        import os

        st = os.stat(db_path)
        return (str(db_path), st.st_mtime, st.st_size)
    except OSError:
        return (str(db_path), 0.0, 0)
