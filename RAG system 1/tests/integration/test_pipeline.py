"""Integration: full pipeline query->retrieve->evidence->prompt->generate->validate
with mocked generator + stub retriever (no network, no index needed)."""
from __future__ import annotations

import json

from app.config import Settings
from app.domain.models import EvidenceChunk
from app.generation.base import GenerationResponse
from app.main import QueryService
from app.system1.laya import LayaDecisionEngine


CHUNK_TEXT = ("Turn 001 | Customer: my flight was cancelled because the airline "
              "had a staff shortage so I missed my connection")


class StubRetriever:
    def __init__(self, chunks):
        self._chunks = chunks

    def retrieve(self, query, trace=None):
        return self._chunks


class FakeGeneratorTask1:
    """Canned Task1 extraction citing the stub chunk (valid grounding)."""
    def generate(self, prompt, *, request_id="-"):
        assert "t-int:chunk:0" in prompt  # evidence actually passed to generator
        body = {"task": "task1", "causal_rationales": [{
            "cause": "airline staff shortage", "effect": "missed connection",
            "mechanism": "flight cancellation",
            "evidence": [{"source_id": "t-int:chunk:0", "turn_start": 0, "turn_end": 0,
                          "quote": "my flight was cancelled because the airline had a staff shortage"}],
            "strength": "explicit"}], "confidence": 0.9, "insufficient_evidence": False}
        return GenerationResponse(text=json.dumps(body), http_status=200, model="fake",
                              latency_ms=1.0, retries=0)


class FakeGeneratorTask2Chain:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt, *, request_id="-"):
        self.calls += 1
        if self.calls == 1:  # task-1 leg inside task2 flow
            body = {"task": "task1", "causal_rationales": [{
                "cause": "airline staff shortage", "effect": "missed connection", "mechanism": "cancellation",
                "evidence": [{"source_id": "t-int:chunk:0", "turn_start": 0, "turn_end": 0,
                              "quote": "my flight was cancelled because the airline had a staff shortage"}],
                "strength": "explicit"}], "confidence": 0.9, "insufficient_evidence": False}
        else:  # synthesis leg consumes task1 json
            assert "TASK-1 EXTRACTION" in prompt
            body = {"task": "task2", "synthesis": "Staff shortages caused cancellations and missed connections.",
                    "key_causes": ["airline staff shortage"], "business_effects": ["missed connections"],
                    "supporting_evidence": ["t-int:chunk:0"], "uncertainties": [], "confidence": 0.85}
        return GenerationResponse(text=json.dumps(body), http_status=200, model="fake",
                              latency_ms=1.0, retries=0)


class FakeGeneratorInvalidThenFixed:
    """First call invalid JSON -> exercises the single bounded repair retry."""
    def __init__(self):
        self.calls = 0

    def generate(self, prompt, *, request_id="-"):
        self.calls += 1
        if self.calls == 1:
            return GenerationResponse(text="oops not json", http_status=200, model="fake",
                                  latency_ms=1.0, retries=0)
        body = {"task": "task1", "causal_rationales": [], "confidence": 0.0,
                "insufficient_evidence": True}
        return GenerationResponse(text=json.dumps(body), http_status=200, model="fake",
                              latency_ms=1.0, retries=0)


def make_service(tmp_path, generator) -> QueryService:
    s = Settings(source_path=tmp_path / "x.json", generated_dir=tmp_path / "g")
    chunks = [EvidenceChunk(chunk_id="t-int:chunk:0", transcript_id="t-int", text=CHUNK_TEXT,
                            score=0.9, rank=0, turn_start=0, turn_end=0)]
    return QueryService(s, retriever=StubRetriever(chunks), generator=generator,
                        laya_engine=LayaDecisionEngine())


def test_task1_end_to_end(tmp_path):
    svc = make_service(tmp_path, FakeGeneratorTask1())
    out = svc.run(task="task1", query="Why was the connection missed?")
    assert out["success"] is True
    assert out["result"]["causal_rationales"][0]["strength"] == "explicit"
    assert out["trace"]["stages"]  # latency breakdown recorded
    assert out["metrics"]["num_chunks"] == 1


def test_task2_consumes_task1(tmp_path):
    svc = make_service(tmp_path, FakeGeneratorTask2Chain())
    out = svc.run(task="task2", query="Why did the staff shortage cause missed connections? Summarize impact")
    assert out["success"] is True
    assert "task1" in out  # extraction preserved alongside synthesis
    assert out["result"]["supporting_evidence"] == ["t-int:chunk:0"]


def test_single_repair_retry(tmp_path):
    svc = make_service(tmp_path, FakeGeneratorInvalidThenFixed())
    out = svc.run(task="task1", query="Why?")
    assert out["success"] is True and out["result"]["insufficient_evidence"] is True


def test_empty_query_rejected_before_retrieval(tmp_path):
    svc = make_service(tmp_path, FakeGeneratorTask1())
    out = svc.run(task="task1", query="   ")
    assert out["success"] is False and out["error"]["code"] == "data_validation"


def test_insufficient_evidence_short_circuit(tmp_path):
    s = Settings(source_path=tmp_path / "x.json", generated_dir=tmp_path / "g")
    svc = QueryService(s, retriever=StubRetriever([]), generator=FakeGeneratorTask1(),
                       laya_engine=LayaDecisionEngine())
    out = svc.run(task="task1", query="Why?")
    # No chunks -> evidence gate insufficient -> typed insufficient result, no generation
    assert out["success"] is True and out["result"]["insufficient_evidence"] is True
