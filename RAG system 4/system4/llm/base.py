"""Provider-independent LLM contract (AGENTS_SYSTEM_4.md §25).

Benchmarks keep the LLM fixed when comparing retrieval architectures; the
pipeline only depends on these protocols, never on a concrete provider.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol


class LLMError(RuntimeError):
    """Raised when a provider fails; the pipeline degrades per §43."""


@dataclass
class GenerationResult:
    text: str
    ttft_ms: float = 0.0
    generation_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    streaming: bool = False


class LLM(Protocol):
    def generate(self, prompt: str, **kwargs: object) -> str: ...


class MeasuredLLM(Protocol):
    """Optional richer contract exposing TTFT and token usage (§32)."""

    identifier: str

    def generate(self, prompt: str, **kwargs: object) -> str: ...

    def generate_measured(self, prompt: str, **kwargs: object) -> GenerationResult: ...


def measure(llm: object, prompt: str, **kwargs: object) -> GenerationResult:
    """Generate with per-stage latency instrumentation when available (§32)."""
    measured = getattr(llm, "generate_measured", None)
    if callable(measured):
        return measured(prompt, **kwargs)
    started = time.perf_counter()
    text = llm.generate(prompt, **kwargs)  # type: ignore[attr-defined]
    elapsed = round((time.perf_counter() - started) * 1000, 2)
    # Non-instrumented adapters: ttft unknown, reported as 0.0 (never fabricated).
    return GenerationResult(text=str(text), ttft_ms=0.0, generation_ms=elapsed)


def llm_identifier(llm: object | None) -> str:
    if llm is None:
        return "deterministic-fallback"
    identifier = getattr(llm, "identifier", None)
    if identifier:
        return str(identifier)
    return f"{type(llm).__name__}"
