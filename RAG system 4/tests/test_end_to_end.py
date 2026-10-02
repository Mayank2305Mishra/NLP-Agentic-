"""End-to-end pipeline tests (§4, §47 integration, §48 acceptance)."""

from __future__ import annotations

from system4.config.settings import Settings
from system4.pipeline import QualityRAG
from system4.retrieval.index import LocalIndex
from system4.schema import Conversation, Turn
from tests.test_regression import REQUIRED_STAGES, _diverse_index


def fixture_index():
    first = Conversation("refund-1", "Retail", "Refund", "Customer asks for a refund", [
        Turn("refund-1", 1, "Customer", "I need a refund because the item arrived damaged."),
        Turn("refund-1", 2, "Agent", "I can approve the refund after receiving the photo."),
        Turn("refund-1", 3, "Customer", "Thanks, that resolves the issue."),
    ])
    second = Conversation("insurance-1", "Insurance", "Policy", "Discount inquiry", [Turn("insurance-1", 1, "Customer", "Can I get a policy discount?")])
    return LocalIndex.build([first, second], Settings(dense_k=2, lexical_k=2, min_rerank_score=0.01))


def test_single_conversation_evidence_and_trace():
    result = QualityRAG(fixture_index(), Settings(dense_k=2, lexical_k=2, min_rerank_score=0.01)).ask("Why was the refund approved?", "task1")
    assert {e.turn.transcript_id for e in result.evidence} == {"refund-1"}
    assert result.trace["system"] == "system4"
    assert "candidate_retrieval_ms" in result.trace


def test_abstains_for_no_match():
    result = QualityRAG(fixture_index(), Settings(min_rerank_score=.95)).ask("astronomy telescope", "task1")
    assert result.answer["insufficient_evidence"] is True


def test_causal_gate_rejects_identical_cause_and_effect():
    from system4.validation.causal import validate_causal_shape
    evidence = [Turn("x", 1, "Customer", "A problem happened.")]
    from system4.schema import Evidence
    errors = validate_causal_shape({"task": "task1", "causal_rationales": [{"cause": "same", "effect": "same", "evidence_turns": [1]}]}, "task1", [Evidence(evidence[0], 1, [], "test")], False)
    assert "CAUSE_EFFECT_IDENTICAL" in errors


def test_all_stage_latencies_instrumented():
    """§32: every major stage reports its own timing — never only total latency."""
    settings = Settings(dense_k=2, lexical_k=2, min_rerank_score=0.01, cache_answers=False)
    result = QualityRAG(fixture_index(), settings).ask("Why was the refund approved?", "task1")
    for stage in REQUIRED_STAGES:
        assert stage in result.trace, f"missing {stage}"
        assert isinstance(result.trace[stage], (int, float))
    assert result.trace["total_ms"] >= 0


def test_answer_is_deterministic_and_cache_hit_on_repeat():
    settings = Settings(dense_k=2, lexical_k=2, min_rerank_score=0.01, cache_answers=True)
    rag = QualityRAG(fixture_index(), settings)
    first = rag.ask("Why was the refund approved?", "task1")
    second = rag.ask("Why was the refund approved?", "task1")
    assert second.trace.get("cache_hit") is True
    assert first.answer == second.answer
    # Cached traces keep provenance but flag the hit.
    assert second.trace["selected_transcript_ids"] == first.trace["selected_transcript_ids"]


def test_auto_task_routing():
    settings = Settings(dense_k=2, lexical_k=2, min_rerank_score=0.01, cache_answers=False)
    rag = QualityRAG(fixture_index(), settings)
    one = rag.ask("Why was the refund approved after the photo?", "auto")
    assert one.answer["task"] == "task1"
    two = rag.ask("Summarize this conversation end to end", "auto")
    assert two.answer["task"] == "task2"


def test_task2_answer_shape_matches_contract():
    """§28/§30: task2 answers (including abstentions) follow the schema."""
    settings = Settings(dense_k=2, lexical_k=2, min_rerank_score=0.01, cache_answers=False)
    rag = QualityRAG(fixture_index(), settings)
    result = rag.ask("Why was the refund approved after the photo?", "task2")
    answer = result.answer
    for key in ("task", "summary", "key_causes", "effects", "mechanisms", "uncertainties", "evidence", "insufficient_evidence"):
        assert key in answer, f"missing {key}"
    # Abstention shape must also be complete (§30).
    abstain = rag.ask("astronomy telescope question", "task2")
    if abstain.answer["insufficient_evidence"]:
        for key in ("summary", "key_causes", "effects", "evidence"):
            assert key in abstain.answer


def test_debug_trace_exposes_selection_and_laya():
    settings = Settings(dense_k=2, lexical_k=2, min_rerank_score=0.01, cache_answers=False)
    result = QualityRAG(fixture_index(), settings).ask(
        "Why was the refund approved?", "task1", debug=True
    )
    debug = result.trace["debug"]
    assert debug["selection"], "selection reasons exposed in debug mode (§45)"
    assert all("reason" in row and "roles" in row for row in debug["selection"])
    assert debug["laya"], "Laya decisions exposed in debug mode (§45)"
    assert debug["candidates"] and "rerank_score" in debug["candidates"][0]
    assert debug["features"]


def test_no_debug_block_by_default():
    settings = Settings(dense_k=2, lexical_k=2, min_rerank_score=0.01, cache_answers=False)
    result = QualityRAG(fixture_index(), settings).ask("Why was the refund approved?", "task1")
    assert "debug" not in result.trace


def test_result_public_serializes():
    settings = Settings(dense_k=2, lexical_k=2, min_rerank_score=0.01, cache_answers=False)
    result = QualityRAG(fixture_index(), settings).ask("Why was the refund approved?", "task1")
    payload = result.public()
    assert set(payload) == {"answer", "evidence", "trace"}
    import json
    json.dumps(payload)  # must be JSON-serializable


def test_input_validation():
    settings = Settings(cache_answers=False)
    rag = QualityRAG(fixture_index(), settings)
    try:
        rag.ask("", "task1")
        raise AssertionError("expected ValueError for empty query")
    except ValueError:
        pass
    try:
        rag.ask("hello", "task9")
        raise AssertionError("expected ValueError for unknown task")
    except ValueError:
        pass


def test_diverse_index_single_source_answer():
    """§23 default: single-interaction analysis keeps exactly one source."""
    settings = Settings(dense_k=8, lexical_k=8, min_rerank_score=0.01, cache_answers=False)
    rag = QualityRAG(_diverse_index(), settings)
    result = rag.ask("Why was the refund approved after the cancellation?", "task1")
    assert len(result.answer["source_transcripts"]) == 1
