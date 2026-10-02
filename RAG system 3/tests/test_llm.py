from __future__ import annotations

import json

import requests

from system3.llm import retry_connection_failures
from system3.llm.ollama import OllamaClient, PartialStreamError

PARTIAL_LINE = json.dumps({"message": {"content": '{"cau'}, "done": False})
DONE_LINE = json.dumps(
    {
        "message": {"content": '" : "charge"}'},
        "eval_count": 4,
        "eval_duration": 4_000_000,
        "prompt_eval_count": 10,
        "prompt_eval_duration": 1_000_000,
        "done": True,
    }
)


class FakeResponse:
    def __init__(self, lines, raise_after_first=None):
        self._lines = lines
        self._raise_after_first = raise_after_first

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_lines(self):
        for idx, line in enumerate(self._lines):
            yield line
            if self._raise_after_first is not None and idx == 0:
                raise self._raise_after_first


class FlakySession:
    def __init__(self, fail_times=0, lines=None, stream_exc=None):
        self.fail_times = fail_times
        self.lines = lines if lines is not None else [DONE_LINE]
        self.stream_exc = stream_exc
        self.calls = 0

    def post(self, *args, **kwargs):
        self.calls += 1
        if self.fail_times > 0:
            self.fail_times -= 1
            raise requests.ConnectionError("connection refused")
        return FakeResponse(self.lines, self.stream_exc)


def test_ollama_retries_connection_failures_then_succeeds(monkeypatch):
    monkeypatch.setattr("system3.llm.ollama.RETRY_BACKOFF_S", 0.0)
    client = OllamaClient(connect_retries=2)
    session = FlakySession(fail_times=2)
    client._session = session

    result = client.generate("sys", "prompt")

    assert session.calls == 3
    assert result.text == '" : "charge"}'
    assert result.output_tokens == 4
    assert result.provider == "ollama"


def test_ollama_gives_up_after_bounded_retries(monkeypatch):
    monkeypatch.setattr("system3.llm.ollama.RETRY_BACKOFF_S", 0.0)
    client = OllamaClient(connect_retries=1)
    session = FlakySession(fail_times=99)
    client._session = session

    try:
        client.generate("sys", "prompt")
    except requests.ConnectionError:
        pass
    else:
        raise AssertionError("expected ConnectionError after retries exhausted")
    assert session.calls == 2


def test_ollama_never_retries_after_partial_stream(monkeypatch):
    monkeypatch.setattr("system3.llm.ollama.RETRY_BACKOFF_S", 0.0)
    client = OllamaClient(connect_retries=3)
    session = FlakySession(
        lines=[PARTIAL_LINE], stream_exc=requests.Timeout("read timed out")
    )
    client._session = session

    try:
        client.generate("sys", "prompt")
    except PartialStreamError:
        pass
    else:
        raise AssertionError("expected PartialStreamError for mid-stream failure")
    assert session.calls == 1, "partial output must never be retried"


def test_retry_connection_failures_helper():
    state = {"failures": 2}

    def flaky():
        if state["failures"] > 0:
            state["failures"] -= 1
            raise requests.ConnectionError("refused")
        return "ok"

    assert retry_connection_failures(flaky, attempts=3, backoff_s=0.0) == "ok"

    def always_fails():
        raise requests.Timeout("down")

    try:
        retry_connection_failures(always_fails, attempts=2, backoff_s=0.0)
    except requests.Timeout:
        pass
    else:
        raise AssertionError("expected Timeout after attempts exhausted")

    # HTTP-level errors are not connection failures and must propagate at once.
    calls = {"n": 0}

    def http_error():
        calls["n"] += 1
        raise requests.HTTPError("500")

    try:
        retry_connection_failures(http_error, attempts=3, backoff_s=0.0)
    except requests.HTTPError:
        pass
    else:
        raise AssertionError("expected HTTPError")
    assert calls["n"] == 1
