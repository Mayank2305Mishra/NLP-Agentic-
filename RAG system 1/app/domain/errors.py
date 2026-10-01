"""Typed error taxonomy (AGENTS.md section 23)."""
from __future__ import annotations

import uuid


class RAGError(Exception):
    code: str = "rag_error"

    def __init__(self, message: str, *, context: dict | None = None, correlation_id: str | None = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.correlation_id = correlation_id or uuid.uuid4().hex[:12]

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "context": self.context,
            "correlation_id": self.correlation_id,
        }


class DataNotFoundError(RAGError):
    code = "data_not_found"


class DataValidationError(RAGError):
    code = "data_validation"


class IndexNotFoundError(RAGError):
    code = "index_not_found"


class IndexStaleError(RAGError):
    code = "index_stale"


class EmbeddingModelError(RAGError):
    code = "embedding_model"


class VectorStoreError(RAGError):
    code = "vector_store"


class System1UnavailableError(RAGError):
    code = "system1_unavailable"


class OllamaUnavailableError(RAGError):
    code = "ollama_unavailable"


class OllamaTimeoutError(RAGError):
    code = "ollama_timeout"


class OllamaProtocolError(RAGError):
    code = "ollama_protocol"


class GenerationParseError(RAGError):
    code = "generation_parse"


class GroundingValidationError(RAGError):
    code = "grounding_validation"


class ContextBudgetError(RAGError):
    code = "context_budget"


class ConfigurationError(RAGError):
    code = "configuration"
