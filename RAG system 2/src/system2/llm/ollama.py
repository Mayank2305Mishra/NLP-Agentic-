from __future__ import annotations

import logging
import time

import requests

from . import GenerationResult

log = logging.getLogger("system2.llm.ollama")


class OllamaClient:
    provider = "ollama"

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen3.5:4b-mlx"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(
        self,
        system: str,
        prompt: str,
        *,
        temperature: float = 0.0,
        num_predict: int = 1400,
        timeout_s: int = 300,
    ) -> GenerationResult:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
            "format": "json",
            "think": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
        }
        begin = time.perf_counter()
        ttft_ms: float | None = None
        chunks: list[str] = []
        prompt_tokens: int | None = None
        output_tokens: int | None = None

        with requests.post(
            f"{self.base_url}/api/chat", json=payload, stream=True, timeout=timeout_s
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                import json as _json

                data = _json.loads(line)
                if ttft_ms is None:
                    ttft_ms = round((time.perf_counter() - begin) * 1000, 3)
                content = (data.get("message") or {}).get("content")
                if content:
                    chunks.append(content)
                if data.get("prompt_eval_count") is not None:
                    prompt_tokens = data["prompt_eval_count"]
                if data.get("eval_count") is not None:
                    output_tokens = data["eval_count"]
                if data.get("done"):
                    break

        generation_ms = round((time.perf_counter() - begin) * 1000, 3)
        return GenerationResult(
            text="".join(chunks),
            model=self.model,
            provider=self.provider,
            ttft_ms=ttft_ms,
            generation_ms=generation_ms,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
        )
