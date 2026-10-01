"""Regression tests for bug fixes (AGENTS.md 50.4: one regression test per fix)."""
from __future__ import annotations

import json

import pytest

from app.domain.errors import GenerationParseError, GroundingValidationError
from app.domain.models import EvidenceChunk, EvidencePack, Task1Result
from app.generation.base import GenerationResponse
from app.generation.parser import extract_json_object, validate_task1_grounding
from app.main import llm_label
from app.observability.redaction import redact, redact_dict
from app.observability.trace import RequestTrace


UUID_SOURCE_ID = "4bda0582-1677-4e58-af92-7399046a59f5:chunk:1"


# --- 1. redaction must not corrupt citation IDs ----------------------------
def test_redaction_preserves_source_and_chunk_ids():
    assert UUID_SOURCE_ID in redact(UUID_SOURCE_ID)
    out = redact_dict({"source_id": UUID_SOURCE_ID,
                       "supporting_evidence": [UUID_SOURCE_ID],
                       "chunk_id": "t1:chunk:0"})
    assert out["source_id"] == UUID_SOURCE_ID
    assert out["supporting_evidence"] == [UUID_SOURCE_ID]
    assert out["chunk_id"] == "t1:chunk:0"


def test_redaction_still_redacts_real_pii():
    assert "[redacted-phone]" in redact("call me at 415-555-0123 tomorrow")
    assert "[redacted-email]" in redact("write to jane.doe@example.com")
    assert redact_dict({"api_key": "sk-abc123"})["api_key"] == "[redacted]"


# --- 2. bounded JSON repair for omitted commas ------------------------------
def test_extract_json_recovers_missing_commas():
    broken = '{\n  "task": "task1"\n  "confidence": 0.9\n  "causal_rationales": []\n}'
    assert extract_json_object(broken) == {"task": "task1", "confidence": 0.9,
                                           "causal_rationales": []}


def test_extract_json_recovers_missing_comma_between_objects():
    broken = ('{"causal_rationales": [{"cause": "a", "effect": "b"}'
              '{"cause": "c", "effect": "d"}]}')
    data = extract_json_object(broken)
    assert [r["cause"] for r in data["causal_rationales"]] == ["a", "c"]


def test_extract_json_still_rejects_garbage():
    with pytest.raises(GenerationParseError):
        extract_json_object("not json at all {{{")
    with pytest.raises(GenerationParseError):
        extract_json_object('{"task": "task1" truncated garbage')


# --- 3. grounding: typographic punctuation must not fail verbatim quotes ----
def _pack(text: str) -> EvidencePack:
    c = EvidenceChunk(chunk_id="t1:chunk:0", transcript_id="t1", text=text,
                      score=0.9, rank=0, turn_start=0, turn_end=1)
    return EvidencePack(chunks=[c], truncated=False, context_tokens=20)


def _task1(quote: str) -> Task1Result:
    return Task1Result.model_validate({"task": "task1", "causal_rationales": [{
        "cause": "staff shortage", "effect": "missed connection", "mechanism": "",
        "evidence": [{"source_id": "t1:chunk:0", "turn_start": 0, "turn_end": 1,
                      "quote": quote}],
        "strength": "explicit"}], "confidence": 0.9, "insufficient_evidence": False})


def test_grounding_tolerates_ascii_apostrophes_against_curly_source():
    pack = _pack("Turn 001 | Agent: that\u2019s correct the fare you\u2019re seeing "
                 "includes the weekend surcharge and nothing else applies here")
    # model copies with straight apostrophes — must still be grounded
    assert validate_task1_grounding(
        _task1("that's correct the fare you're seeing includes the weekend surcharge"), pack
    )["grounded"] is True


def test_grounding_still_rejects_paraphrase():
    pack = _pack("Turn 001 | Agent: the flight was cancelled due to a crew shortage "
                 "so we rebooked you on the morning service tomorrow")
    with pytest.raises(GroundingValidationError):
        validate_task1_grounding(
            _task1("the flight got cancelled because the crew did not show up"), pack)


def test_grounding_rejects_quote_from_another_chunk():
    """A verbatim quote that belongs to a different pack is still a bad citation."""
    pack = _pack("Turn 001 | Agent: your refund has been approved today")
    with pytest.raises(GroundingValidationError):
        validate_task1_grounding(
            _task1("turn 002 | agent: loyalty points were doubled last season"), pack)


# --- 4. repair prompt must carry the evidence pack --------------------------
def test_repair_template_has_evidence_and_quote_rule():
    from app.tasks.base import REPAIR_TEMPLATE
    assert "{evidence}" in REPAIR_TEMPLATE
    assert "{allowed_ids}" in REPAIR_TEMPLATE
    assert "QUOTE RULE" in REPAIR_TEMPLATE


def test_repair_retry_sends_evidence_to_generator():
    """Grounding failure on call 1 -> call 2 must contain verbatim source text."""
    from app.config import Settings
    from app.tasks.task1_extraction import run_task1

    pack = _pack("Turn 001 | Agent: the flight was cancelled due to a crew shortage "
                 "so we rebooked you on the morning service tomorrow")
    prompts: list[str] = []

    class Flaky:
        def generate(self, prompt, *, request_id="-"):
            prompts.append(prompt)
            if len(prompts) == 1:  # paraphrased quote -> grounding failure
                body = {"task": "task1", "causal_rationales": [{
                    "cause": "crew shortage", "effect": "rebooked", "mechanism": "",
                    "evidence": [{"source_id": "t1:chunk:0", "turn_start": 0, "turn_end": 1,
                                  "quote": "the crew did not show up so the flight was axed"}],
                    "strength": "explicit"}], "confidence": 0.9, "insufficient_evidence": False}
            else:
                body = {"task": "task1", "causal_rationales": [{
                    "cause": "crew shortage", "effect": "rebooked", "mechanism": "",
                    "evidence": [{"source_id": "t1:chunk:0", "turn_start": 0, "turn_end": 1,
                                  "quote": "the flight was cancelled due to a crew shortage"}],
                    "strength": "explicit"}], "confidence": 0.9, "insufficient_evidence": False}
            return GenerationResponse(text=json.dumps(body), http_status=200, model="fake",
                                      latency_ms=1.0, retries=0)

    trace = RequestTrace()
    settings = Settings(source_path="/nonexistent.json", generated_dir="/tmp/g-regress")
    result, info = run_task1(settings=settings, client=Flaky(), query_text="why?",
                             pack=pack, trace=trace)
    assert info["validation_failures"] == 1
    assert len(prompts) == 2
    # first prompt is the task prompt, second is the repair — both must carry evidence
    assert "Turn 001 | Agent: the flight was cancelled" in prompts[0]
    assert "Turn 001 | Agent: the flight was cancelled" in prompts[1]
    assert result.causal_rationales


# --- 5. diagnostics must report the configured model ------------------------
def test_llm_label_reports_configured_model():
    assert "qwen3.5:4b-mlx" in llm_label("qwen3.5:4b-mlx")
    assert "my-model" in llm_label("my-model")


# --- 6. repeated stages are summed, not silently dropped --------------------
def test_stage_ms_total_sums_repeated_stages():
    import time
    t = RequestTrace()
    for _ in range(2):
        t.start("generation_request")
        time.sleep(0.01)
        t.end("generation_request")
    assert t.stage_ms_total("generation_request") >= 15.0
    assert t.stage_ms_total("generation_request") > t.stage_ms("generation_request")
