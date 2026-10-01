"""Ollama generation adapter — the ONLY module that knows the endpoint contract.

Local Ollama server (default http://127.0.0.1:11434), primary model
qwen3.5:4b-mlx. Uses the native POST /api/chat with `"format": "json"`
(guaranteed JSON mode) and `"stream": false`, thinking disabled for minimal
token use. No API key, nothing leaves the laptop.

- Explicit connect/read timeouts, bounded retries (transient failures only:
  connection errors + HTTP 429/5xx; never 400s, never indefinite).
- Records latency, HTTP status, model name when returned.
"""
from __future__ import annotations

import random
import time

import httpx

from app.config import Settings
from app.domain.errors import (
    GenerationParseError,
    OllamaProtocolError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)
from app.generation.base import GenerationResponse, TRANSIENT_STATUS


class OllamaClient:
    def __init__(self, settings: Settings, *, transport: httpx.BaseTransport | None = None):
        self.settings = settings
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.ollama_model
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(
                settings.generation_timeout_seconds,
                connect=settings.generation_connect_timeout_seconds,
            ),
            transport=transport,
        )

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    def health(self) -> dict:
        """Server reachability + model presence. Sends no conversation data."""
        out: dict = {"backend": "ollama", "base_url": self.base_url,
                     "model": self.model, "reachable": False, "model_present": False}
        try:
            r = self._client.get("/api/tags")
            out["reachable"] = True
            out["status"] = r.status_code
            if r.status_code == 200:
                try:
                    models = [m.get("name") for m in r.json().get("models", [])
                              if isinstance(m, dict)]
                    out["models"] = models
                    out["model_present"] = self.model in models
                    if not out["model_present"]:
                        out["detail"] = (f"model '{self.model}' not pulled; "
                                         f"run: ollama pull {self.model}")
                except Exception:
                    pass
        except (httpx.ConnectError, httpx.TimeoutException, OSError) as e:
            out["detail"] = f"unreachable (is ollama running?): {e}"
        return out

    def _payload(self, prompt: str) -> dict:
        s = self.settings
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a careful causal-reasoning assistant. Follow the task prompt exactly."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "format": "json",
            "think": False,
            "options": {
                "temperature": s.generation_temperature,
                "top_p": s.generation_top_p,
                "num_predict": s.generation_max_tokens,
            },
        }

    def generate(self, prompt: str, *, request_id: str = "-") -> GenerationResponse:
        max_retries = max(0, self.settings.generation_max_retries)
        retries = 0
        t0 = time.perf_counter()
        for attempt in range(max_retries + 1):
            try:
                r = self._client.post("/api/chat", json=self._payload(prompt))
            except (httpx.ConnectError, httpx.TimeoutException, TimeoutError) as e:
                if attempt < max_retries:
                    retries += 1
                    time.sleep(min(4.0, 0.3 * (2 ** attempt)) + random.uniform(0, 0.2))
                    continue
                if isinstance(e, httpx.TimeoutException):
                    raise OllamaTimeoutError(f"Ollama request timed out: {e}",
                                             context={"request_id": request_id}) from e
                raise OllamaUnavailableError(
                    f"Ollama unreachable at {self.base_url} (is it running?). {e}",
                    context={"request_id": request_id, "base_url": self.base_url}) from e
            ms = (time.perf_counter() - t0) * 1000.0
            if r.status_code in TRANSIENT_STATUS and attempt < max_retries:
                retries += 1
                time.sleep(min(4.0, 0.3 * (2 ** attempt)) + random.uniform(0, 0.2))
                continue
            if r.status_code == 400:
                raise OllamaProtocolError(f"Ollama rejected the request (400): {r.text[:300]}",
                                          context={"request_id": request_id, "status": 400})
            if r.status_code == 404:
                raise OllamaProtocolError(
                    f"Ollama model not found (404): {r.text[:300]} "
                    f"(run: ollama pull {self.model})",
                    context={"request_id": request_id, "status": 404})
            if r.status_code >= 400:
                raise OllamaUnavailableError(f"Ollama error (HTTP {r.status_code}): {r.text[:300]}",
                                             context={"request_id": request_id,
                                                      "status": r.status_code})
            try:
                data = r.json()
            except Exception as e:
                raise OllamaProtocolError(f"Ollama returned malformed JSON: {e}",
                                          context={"request_id": request_id}) from e
            text = self._extract_text(data)
            if not text or not text.strip():
                raise GenerationParseError("Ollama returned empty text",
                                           context={"request_id": request_id})
            model = data.get("model", self.model) if isinstance(data, dict) else self.model
            return GenerationResponse(text=text, http_status=r.status_code, model=model,
                                      latency_ms=round(ms, 2), retries=retries,
                                      raw=data if self.settings.app_debug else None)
        raise OllamaUnavailableError("Ollama failed after retries",
                                     context={"request_id": request_id})

    @staticmethod
    def _extract_text(data: dict) -> str:
        try:
            msg = data["message"]
            content = msg.get("content", "") if isinstance(msg, dict) else ""
            return "".join(c.get("text", "") for c in content) if isinstance(content, list) else str(content)
        except (KeyError, TypeError, AttributeError) as e:
            if isinstance(data, dict) and "error" in data:
                raise OllamaProtocolError(f"Ollama error payload: {str(data['error'])[:300]}")
            raise GenerationParseError(f"Unexpected Ollama response shape: {e}",
                                       context={"preview": str(data)[:300]})
