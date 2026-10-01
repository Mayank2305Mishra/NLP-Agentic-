from __future__ import annotations

import logging
import os
import time
from typing import Any

import requests

from . import GenerationResult

log = logging.getLogger("system2.llm.bonsai")


class BonsaiClient:
    """Adapter for the existing local Bonsai endpoint (OpenAI-style or simple text JSON)."""

    provider = "bonsai"

    def __init__(self, base_url: str = "", model: str = ""):
        self.base_url = (base_url or os.environ.get("BONSAI_URL", "")).rstrip("/")
        self.model = model or os.environ.get("BONSAI_MODEL", "")
        self.api_key = os.environ.get("BONSAI_API_KEY", "")

    def generate(
        self,
        system: str,
        prompt: str,
        *,
        temperature: float = 0.0,
        num_predict: int = 1400,
        timeout_s: int = 300,
    ) -> GenerationResult:
        if not self.base_url:
            raise RuntimeError("Bonsai endpoint not configured (set generation.bonsai_url or BONSAI_URL)")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
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
        resp = requests.post(self.base_url, json=payload, headers=headers, timeout=timeout_s)
        resp.raise_for_status()
        data = resp.json()
        elapsed = round((time.perf_counter() - begin) * 1000, 3)
        text = self._extract(data)
        usage = data.get("usage") or {}
        return GenerationResult(
            text=text,
            model=self.model or "bonsai",
            provider=self.provider,
            ttft_ms=None,
            generation_ms=elapsed,
            prompt_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )

    @staticmethod
    def _extract(data: dict[str, Any]) -> str:
        if isinstance(data.get("response"), str):
            return data["response"]
        if isinstance(data.get("text"), str):
            return data["text"]
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first.get("text"), str):
                return first["text"]
            message = first.get("message") or {}
            if isinstance(message.get("content"), str):
                return message["content"]
        if isinstance(data.get("content"), str):
            return data["content"]
        raise ValueError(f"unrecognized Bonsai response shape: {list(data)}")
