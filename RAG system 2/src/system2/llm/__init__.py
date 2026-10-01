from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class GenerationResult:
    text: str
    model: str
    provider: str
    ttft_ms: float | None = None
    generation_ms: float = 0.0
    prompt_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def tokens_per_second(self) -> float | None:
        if not self.output_tokens or not self.generation_ms:
            return None
        return round(self.output_tokens / (self.generation_ms / 1000.0), 2)


@runtime_checkable
class LLMClient(Protocol):
    provider: str
    model: str

    def generate(
        self,
        system: str,
        prompt: str,
        *,
        temperature: float = 0.0,
        num_predict: int = 1400,
        timeout_s: int = 300,
    ) -> GenerationResult: ...


def build_client(config) -> LLMClient:
    from ..config import Config

    assert isinstance(config, Config)
    gen = config.generation
    if gen.provider == "ollama":
        from .ollama import OllamaClient

        return OllamaClient(
            base_url=gen.ollama_url,
            model=gen.model,
        )
    if gen.provider == "bonsai":
        from .bonsai import BonsaiClient

        return BonsaiClient(
            base_url=gen.bonsai_url,
            model=gen.bonsai_model or gen.model,
        )
    raise ValueError(f"unknown LLM provider: {gen.provider!r}")
