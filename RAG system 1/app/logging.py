"""Structured logging. Never log raw conversations at INFO."""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

_configured = False


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "event": getattr(record, "event", record.getMessage()),
            "logger": record.name,
        }
        for key in ("request_id", "latency_ms", "top_k", "result_count", "extra"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info and record.exc_info[0] is not None:
            payload["exc"] = self.formatException(record.exc_info)[:2000]
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> logging.Logger:
    global _configured
    handler = logging.StreamHandler(sys.stderr)  # never pollute CLI stdout JSON
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger("rag1")
    root.handlers = [handler]
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.propagate = False
    _configured = True
    return root


def get_logger(name: str = "rag1") -> logging.Logger:
    if not _configured:
        configure_logging()
    return logging.getLogger(f"rag1.{name}")


def log_event(logger: logging.Logger, level: int, event: str, **fields) -> None:
    logger.log(level, event, extra={"event": event, **fields})
