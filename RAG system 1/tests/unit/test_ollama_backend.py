"""Unit: Ollama backend (mocked HTTP, no network)."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

import httpx
import pytest

from app.config import Settings
from app.domain.errors import OllamaProtocolError
from app.generation.ollama_client import OllamaClient


def settings_for(**kw) -> Settings:
    tmp = Path(tempfile.mkdtemp())
    base = dict(source_path=tmp / "x.json", generated_dir=tmp / "g",
                ollama_model="qwen3.5:9b-mlx")
    base.update(kw)
    return Settings(**base)


def ollama_ok_payload(text='{"task":"task1"}') -> dict:
    return {"model": "qwen3.5:9b-mlx", "created_at": "2026-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": text}, "done": True}


def transport_for(status: int, payload: dict) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        body = json.loads(request.content.decode())
        assert body["model"] == "qwen3.5:9b-mlx"
        assert body["stream"] is False
        assert body["format"] == "json"
        assert body["think"] is False
        assert body["options"]["temperature"] == 0.0
        return httpx.Response(status, json=payload)
    return httpx.MockTransport(handler)


def test_ollama_generate_ok():
    client = OllamaClient(settings_for(), transport=transport_for(200, ollama_ok_payload()))
    resp = client.generate("prompt", request_id="r1")
    assert resp.text == '{"task":"task1"}' and resp.http_status == 200


def test_ollama_no_retry_on_400():
    client = OllamaClient(settings_for(), transport=transport_for(400, {"error": "bad"}))
    with pytest.raises(OllamaProtocolError):
        client.generate("prompt")


def test_ollama_404_names_model_pull():
    client = OllamaClient(settings_for(), transport=transport_for(404, {"error": "not found"}))
    with pytest.raises(OllamaProtocolError) as e:
        client.generate("prompt")
    assert "ollama pull" in str(e.value)


def test_ollama_service_uses_ollama_client(tmp_path):
    from app.main import QueryService
    s = Settings(source_path=tmp_path / "x.json", generated_dir=tmp_path / "g")
    assert isinstance(QueryService(s).generator, OllamaClient)
