"""Unit tests: dense/lexical retrieval, rank fusion, consistency gate (§8–§10)."""

from __future__ import annotations

from system4.config.settings import Settings
from system4.retrieval import dense, lexical
from system4.retrieval.broad_retrieval import retrieve
from system4.retrieval.index import LocalIndex, tokens
from system4.retrieval.metadata import apply_gate, consistency
from tests.conftest import conv


def _index() -> LocalIndex:
    refund = conv(
        "refund-1",
        [(1, "Customer", "I need a refund because the item arrived damaged."),
         (2, "Agent", "I will approve your refund today.")],
        domain="Retail", intent="Refunds", reason="Customer wants a refund for a damaged item.",
    )
    booking = conv(
        "booking-1",
        [(1, "Customer", "My hotel booking has the wrong dates."),
         (2, "Agent", "Let me fix the booking dates for you.")],
        domain="Travel", intent="Booking Errors", reason="Booking dates are wrong.",
    )
    return LocalIndex.build([refund, booking], Settings())


def test_tokenizer_is_deterministic():
    assert tokens("Refund, REFUND! refund") == ["refund", "refund", "refund"]


def test_dense_prefers_matching_conversation():
    results = dense.search(_index(), "refund for damaged item", 5)
    assert results[0][0] == "refund-1"
    assert 0.0 <= results[0][1] <= 1.0 + 1e-9


def test_lexical_prefers_matching_conversation():
    results = lexical.search(_index(), "damaged refund", 5)
    assert results[0][0] == "refund-1"
    assert results[0][1] > 0


def test_retrieve_fuses_both_streams_with_provenance():
    settings = Settings(dense_k=5, lexical_k=5)
    timings: dict = {}
    candidates = retrieve(_index(), "refund damaged item", settings, timings)
    top = candidates[0]
    assert top.transcript_id == "refund-1"
    assert set(top.source_retrievers) <= {"dense", "lexical"}
    assert top.dense_score is not None or top.lexical_score is not None
    assert top.fusion_score > 0
    assert {"dense_retrieval_ms", "lexical_retrieval_ms", "candidate_merge_ms"} <= set(timings)


def test_retrieve_returns_empty_on_broken_index():
    class Broken:
        conversations = {}

    candidates = retrieve(Broken(), "anything", Settings(dense_k=3, lexical_k=3), {})
    assert candidates == []


def test_consistency_scores_metadata_compatibility():
    refund = conv("r", [(1, "Customer", "refund damaged")], domain="Retail",
                  intent="Refunds", reason="Refund for damaged item")
    score, breakdown = consistency("refund damaged item", "refund damaged item", refund)
    assert score > 0.2
    assert breakdown["reason_for_call_similarity"] > 0
    unrelated = conv("u", [(1, "Customer", "hello")], domain="Insurance",
                     intent="Claims", reason="Policy renewal")
    weak, _ = consistency("refund damaged item", "refund damaged item", unrelated)
    assert weak < score


def test_gate_filters_inconsistent_but_keeps_floor():
    settings = Settings(min_consistency_score=0.9, consistency_keep_floor=2)
    target = conv("target", [(1, "Customer", "refund damaged item")],
                  domain="Retail", intent="Refunds", reason="Refund for damaged item")
    noise = conv("noise", [(1, "Customer", "hi")], domain="Insurance",
                 intent="Claims", reason="Policy question")
    index = LocalIndex.build([target, noise], settings)
    candidates = retrieve(index, "refund damaged item", settings)
    gated = apply_gate(candidates, index.conversations, "refund damaged item",
                       "refund damaged item", settings)
    # Threshold is impossible (0.9) but the floor keeps candidates observable (§10).
    assert len(gated) == 2
    assert gated[0].transcript_id == "target"
    assert all(c.consistency_breakdown for c in gated)
