"""Shared generation primitives (backend-agnostic).

Only the Ollama adapter (ollama_client.py) knows the endpoint contract.
The pipeline depends on the `generate(prompt, request_id)` interface only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


TRANSIENT_STATUS = {429, 500, 502, 503, 504}


class GeneratorClient(Protocol):
    """Any generator: OllamaClient or test fakes."""

    def generate(self, prompt: str, *, request_id: str = "-") -> GenerationResponse: ...


@dataclass
class GenerationResponse:
    text: str
    http_status: int
    model: str | None
    latency_ms: float
    retries: int
    raw: dict | None = None
    usage: dict | None = None  # e.g. {"input_tokens":..,"output_tokens":..,"total_tokens":..}


def extract_text(data: dict | list) -> str:
    """Pull assistant text from OpenAI-style or common local-LLM shapes."""
    from app.domain.errors import GenerationParseError

    if isinstance(data, dict):
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0] or {}
            msg = first.get("message") if isinstance(first, dict) else None
            if isinstance(msg, dict) and msg.get("content"):
                c = msg["content"]
                return "".join(x.get("text", "") for x in c) if isinstance(c, list) else str(c)
            for key in ("text", "content", "output"):
                if first.get(key):
                    return str(first[key])
        for key in ("response", "text", "content", "output", "result"):
            if data.get(key):
                return str(data[key])
        if "error" in data:
            raise GenerationParseError(f"Generator error payload: {str(data['error'])[:300]}")
    return ""
