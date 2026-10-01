"""Query normalization (empty/whitespace rejected before retrieval)."""
from __future__ import annotations

from app.domain.errors import DataValidationError
from app.domain.models import RetrievalQuery
from app.ingestion.normalizer import normalize_text


def normalize_query(text: str, *, task: str = "task1", domain=None,
                    intent=None, top_k: int = 8) -> RetrievalQuery:
    if task not in ("task1", "task2"):
        from app.domain.errors import ConfigurationError
        raise ConfigurationError(f"Unknown task '{task}' (expected task1|task2)")
    if text is None or not str(text).strip():
        raise DataValidationError("Empty query: please provide a question or request.")
    cleaned = normalize_text(str(text))
    if not cleaned:
        raise DataValidationError("Empty query: please provide a question or request.")
    return RetrievalQuery(text=cleaned, task=task, domain=domain, intent=intent, top_k=top_k)
