from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

log = logging.getLogger("system2.structured")


class StructuredLogger:
    def __init__(self, path: Path, level: str = "INFO", debug_evidence: bool = False):
        self.path = path
        self.level = level.upper()
        self.debug_evidence = debug_evidence
        self._lock = threading.Lock()

    def emit(self, payload: dict[str, Any]) -> None:
        if self.level == "DEBUG" and not self.debug_evidence:
            payload = {k: v for k, v in payload.items() if k != "evidence"}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(payload, ensure_ascii=False, default=str)
            with self._lock:
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except OSError as exc:
            log.warning("failed to write structured log: %s", exc)
        log.debug("request summary: %s", json.dumps(payload, default=str))
