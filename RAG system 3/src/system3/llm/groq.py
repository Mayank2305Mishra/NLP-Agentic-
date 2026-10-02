from __future__ import annotations

import logging
import os
import time
from typing import Any

import requests

from . import (
    DEFAULT_CONNECT_RETRIES,
    GenerationResult,
    pooled_session,
    retry_connection_failures,
)

log = logging.getLogger("system3.llm.groq")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqClient:
    """Optional external benchmark adapter. Never the default: local runs use Ollama/Bonsai.

    Requires GROQ_API_KEY in the environment; the key is never stored in config files.
    """

    provider = "groq"

    def __init__(
        self,
        model: str = "",
        base_url: str = "",
        api_key: str = "",
        connect_retries: int = DEFAULT_CONNECT_RETRIES,
    ):
        self.base_url = base_url or GROQ_URL
        self.model = model
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.connect_retries = max(0, connect_retries)
        self._session = pooled_session()

    def generate(
        self,
        system: str,
        prompt: str,
        *,
        temperature: float = 0.0,
        num_predict: int = 1400,
        timeout_s: int = 300,
    ) -> GenerationResult:
        if not self.api_key:
            raise RuntimeError("Groq endpoint not configured (set GROQ_API_KEY)")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": num_predict,
            "stream": False,
        }
        begin = time.perf_counter()
        resp = retry_connection_failures(
            lambda: self._session.post(
                self.base_url, json=payload, headers=headers, timeout=timeout_s
            ),
            attempts=self.connect_retries + 1,
        )
        resp.raise_for_status()
        data = resp.json()
        elapsed = round((time.perf_counter() - begin) * 1000, 3)
        text = self._extract(data)
        usage = data.get("usage") or {}
        return GenerationResult(
            text=text,
            model=self.model or "groq",
            provider=self.provider,
            ttft_ms=None,
            generation_ms=elapsed,
            prompt_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )

    @staticmethod
    def _extract(data: dict[str, Any]) -> str:
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            message = first.get("message") or {}
            if isinstance(message.get("content"), str):
                return message["content"]
            if isinstance(first.get("text"), str):
                return first["text"]
        if isinstance(data.get("content"), str):
            return data["content"]
        raise ValueError(f"unrecognized Groq response shape: {list(data)}")
