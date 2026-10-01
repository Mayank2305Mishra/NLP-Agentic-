from __future__ import annotations

import json

import pytest

from system2.dataset import build_processed, parse_conversation
from system2.indexer import build_index
from system2.pipeline import Pipeline
from system2.sampling import load_population, stratified_sample
from system2.textnorm import normalize_text
from tests.conftest import FakeLLM


def _raw_dataset(path, conversations):
    records = []
    for conv in conversations:
        records.append(
            {
                "transcript_id": conv.transcript_id,
                "time_of_interaction": conv.time_of_interaction,
                "domain": conv.domain,
                "intent": conv.intent,
                "reason_for_call": conv.reason_for_call,
                "conversation": [
                    {"speaker": t.speaker_original, "text": t.text_original}
                    for t in conv.turns
                ],
            }
        )
    path.write_text(json.dumps(records), encoding="utf-8")


def test_build_processed_and_sampling_and_index(config, sample_conversations):
    _raw_dataset(config.dataset.raw_path, sample_conversations)
    stats = build_processed(config)
    assert stats["conversation_count"] == 6
    assert stats["turn_count"] == 48
    assert len(stats["source_hash"]) == 64

    manifest = stratified_sample(config)
    assert manifest["sample_count"] == 6
    assert manifest["seed"] == 42
    again = stratified_sample(config)
    assert manifest["selected_transcript_ids"] == again["selected_transcript_ids"]

    counts = build_index(config, population="sample")
    assert counts["conversations"] == 6
    assert counts["turns"] == 48
    assert counts["conv_fts"] == 6


def test_parse_conversation_normalizes(config):
    conv = parse_conversation(
        {
            "transcript_id": "x1",
            "time_of_interaction": "2025-01-01 00:00:00",
            "domain": "Insurance",
            "intent": "Cancellations",
            "reason_for_call": "r",
            "conversation": [
                {"speaker": "Agent", "text": "Hello   THERE"},
                {"speaker": "Isabella", "text": "i want to cancel"},
                {"speaker": "Note", "text": "[Key Events] internal summary"},
            ],
        }
    )
    assert conv.turns[0].text_normalized == "hello there"
    assert conv.turns[0].text_original == "Hello   THERE"
    assert conv.turns[1].speaker == "customer"
    assert conv.turns[1].speaker_inferred is True
    assert conv.turns[2].is_post_call_artifact is True


def test_sample_population_helper(config, sample_conversations):
    _raw_dataset(config.dataset.raw_path, sample_conversations)
    build_processed(config)
    stratified_sample(config)
    pop = load_population(config, "sample")
    assert len(pop) == 6
    assert load_population(config, "full") is None


def test_pipeline_end_to_end_task1(config, store):
    fake = FakeLLM(
        payload={
            "task": "task1",
            "causal_rationales": [
                {
                    "cause": "duplicate charge on the account",
                    "effect": "agent processed a refund",
                    "mechanism": "the refund reverses the duplicate charge",
                    "evidence_turns": [3, 4, 6],
                    "strength": "explicit",
                }
            ],
            "confidence": 0.85,
            "insufficient_evidence": False,
        }
    )
    pipeline = Pipeline(config, store=store, client=fake)
    result = pipeline.ask("why was the customer charged twice and what did the agent do?")

    assert result.parse_ok
    assert result.answer["insufficient_evidence"] is False
    assert result.evidence_gate["decision"] in ("SUFFICIENT", "EXPAND")
    assert [e["transcript_id"] for e in result.evidence] == ["conv-refund"]
    assert result.answer_gate["decision"] == "ACCEPT_ANSWER"
    assert result.grounding["passed"] is True
    assert result.trace["generation_ms"] is not None
    assert result.trace["total_ms"] > 0
    assert fake.calls and "EVIDENCE PACK" in fake.calls[0]["prompt"]


def test_pipeline_insufficient_skips_llm(config, store):
    fake = FakeLLM(payload={"task": "task1"})
    pipeline = Pipeline(config, store=store, client=fake)
    result = pipeline.ask("quantum chromodynamics lecture notes please")
    assert fake.calls == []
    assert result.evidence_gate["decision"] == "INSUFFICIENT"
    assert result.answer["insufficient_evidence"] is True
    assert result.answer_gate["decision"] == "ACCEPT_ANSWER"
    assert result.trace["generation_ms"] == 0.0


def test_pipeline_contamination_rejected(config, store):
    fake = FakeLLM(
        payload={
            "task": "task1",
            "causal_rationales": [
                {
                    "cause": "refund requested in one call",
                    "effect": "cancellation happened in another call",
                    "mechanism": None,
                    "evidence_turns": [3, 99],
                    "strength": "explicit",
                }
            ],
            "confidence": 0.9,
            "insufficient_evidence": False,
        }
    )
    pipeline = Pipeline(config, store=store, client=fake)
    result = pipeline.ask("why did customers ask for refunds?")
    assert result.answer_gate["decision"] in ("REJECT_ANSWER", "FLAG_ANSWER")
    assert result.grounding["passed"] is False


def test_pipeline_retries_invalid_json(config, store):
    fake = FakeLLM(raw="not json at all")
    pipeline = Pipeline(config, store=store, client=fake)
    result = pipeline.ask("why did the customer want a refund for the duplicate charge?")
    assert len(fake.calls) >= 2
    assert result.parse_ok is False or result.answer is None
    assert result.errors


def test_pipeline_task2(config, store):
    fake = FakeLLM(
        payload={
            "task": "task2",
            "summary": "customer reported a duplicate charge and the agent refunded it",
            "causal_chain": [
                {"step": 1, "event": "duplicate charge reported", "role": "customer", "evidence_turns": [3]},
                {"step": 2, "event": "agent processed refund", "role": "agent", "evidence_turns": [4, 6]},
            ],
            "key_causes": ["duplicate charge"],
            "effects": ["refund completed"],
            "uncertainties": [],
            "confidence": 0.8,
            "insufficient_evidence": False,
        }
    )
    pipeline = Pipeline(config, store=store, client=fake)
    result = pipeline.ask(
        "summarize the entire call end to end: why was the customer charged twice?",
        task="task2",
    )
    assert result.task == "task2"
    assert result.parse_ok
    assert result.answer["causal_chain"]


def test_structured_log_written(config, store):
    fake = FakeLLM(payload={"task": "task1", "causal_rationales": [], "confidence": 0.0, "insufficient_evidence": True})
    pipeline = Pipeline(config, store=store, client=fake)
    pipeline.ask("why was the customer charged twice?")
    log_path = config.logging.log_path
    assert log_path.exists()
    last = json.loads(log_path.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert last["request_id"]
    assert last["total_ms"] > 0
    assert "trace" in last


@pytest.mark.parametrize("ablation", ["pure_bm25", "bm25_metadata", "bm25_causal", "bm25_expansion", "full"])
def test_ablation_modes_run(config, store, ablation):
    from system2.retriever import RetrievalSettings

    config.ablation = {"mode": ablation}
    pipeline = Pipeline(config, store=store, client=FakeLLM(payload={"task": "task1", "causal_rationales": [], "confidence": 0.0, "insufficient_evidence": True}))
    pipeline.retriever.settings = RetrievalSettings.from_config(config)
    result = pipeline.ask("why did the customer want a refund for the duplicate charge?")
    assert result.request_id
    assert result.trace["total_ms"] > 0
