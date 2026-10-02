from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Protocol, TypeVar, runtime_checkable

import requests

#: Bounded retry policy for connection-level failures only (endpoint not up
#: yet, transient refused connection). Never retried after a response starts.
DEFAULT_CONNECT_RETRIES = 2
RETRY_BACKOFF_S = 0.25

_T = TypeVar("_T")


@dataclass
class GenerationResult:
    text: str
    model: str
    provider: str
    ttft_ms: float | None = None
    generation_ms: float = 0.0
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    prompt_eval_ms: float | None = None
    eval_ms: float | None = None

    @property
    def tokens_per_second(self) -> float | None:
        output_tokens = self.output_tokens
        duration_ms = self.eval_ms or self.generation_ms
        if not output_tokens or not duration_ms:
            return None
        return round(output_tokens / (duration_ms / 1000.0), 2)

    @property
    def input_tokens_per_second(self) -> float | None:
        if not self.prompt_tokens or not self.prompt_eval_ms:
            return None
        return round(self.prompt_tokens / (self.prompt_eval_ms / 1000.0), 2)


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


def pooled_session(pool_maxsize: int = 16) -> requests.Session:
    """Shared keep-alive HTTP session: avoids a TCP handshake per generation
    request and pools connections across parallel benchmark workers."""
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=4, pool_maxsize=pool_maxsize, max_retries=0
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def retry_connection_failures(
    send: Callable[[], _T],
    attempts: int,
    backoff_s: float = RETRY_BACKOFF_S,
) -> _T:
    """Run ``send`` with bounded retries on connection-level failures only.

    HTTP errors from a live endpoint are NOT retried here — the caller decides
    via ``raise_for_status``. Only ConnectionError/Timeout before any bytes
    are delivered are considered transient.
    """
    attempts = max(1, attempts)
    last: requests.RequestException | None = None
    for attempt in range(attempts):
        try:
            return send()
        except (requests.ConnectionError, requests.Timeout) as exc:
            last = exc
            if attempt + 1 >= attempts:
                raise
            time.sleep(backoff_s * (attempt + 1))
    raise last if last else RuntimeError("unreachable")  # pragma: no cover


def build_client(config) -> LLMClient:
    from ..config import Config

    assert isinstance(config, Config)
    gen = config.generation
    if gen.provider == "ollama":
        from .ollama import OllamaClient

        return OllamaClient(
            base_url=gen.ollama_url,
            model=gen.model,
            connect_retries=gen.connect_retries,
        )
    if gen.provider == "bonsai":
        from .bonsai import BonsaiClient

        return BonsaiClient(
            base_url=gen.bonsai_url,
            model=gen.bonsai_model or gen.model,
            connect_retries=gen.connect_retries,
        )
    if gen.provider == "groq":
        from .groq import GroqClient

        return GroqClient(model=gen.model, connect_retries=gen.connect_retries)
    raise ValueError(f"unknown LLM provider: {gen.provider!r}")
