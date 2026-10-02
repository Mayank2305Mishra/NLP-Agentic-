"""Unit tests: grounding validator and transcript-scoped citations (§29)."""

from __future__ import annotations

from system4.schema import Evidence, GroundingStatus
from system4.validation.citation import parse_citation, validate_citations
from system4.validation.grounding import validate
from tests.conftest import conv


def _items(tid: str = "t1") -> list[Evidence]:
    conversation = conv(tid, [
        (1, "Customer", "I need a refund because the item was damaged"),
        (2, "Agent", "I approved the refund after the photo"),
        (3, "Customer", "The refund fixed the issue thanks"),
    ])
    return [Evidence(turn=t, score=1.0, roles=["cause"], reason="test") for t in conversation.turns]


def test_grounded_claim_with_valid_citation():
    answer = {
        "task": "task1",
        "causal_rationales": [{
            "cause": "item was damaged refund",
            "effect": "refund approved issue fixed",
            "evidence_turns": [1, 2, 3],
        }],
    }
    status, ratio = validate(answer, _items())
    assert status == GroundingStatus.GROUNDED.value
    assert ratio >= 0.55


def test_citation_to_unknown_turn_is_unsupported():
    answer = {"task": "task1", "causal_rationales": [{
        "cause": "damaged item refund", "effect": "refund approved", "evidence_turns": [99],
    }]}
    status, ratio = validate(answer, _items())
    assert status == GroundingStatus.UNSUPPORTED.value


def test_citation_to_other_conversation_is_unsupported():
    evidence = _items("t1")
    answer = {"task": "task1", "evidence": ["other-transcript:1"],
              "causal_rationales": [{"cause": "damaged refund", "effect": "approved refund"}]}
    status, _ = validate(answer, evidence)
    assert status == GroundingStatus.UNSUPPORTED.value


def test_named_citation_valid_when_transcript_selected():
    evidence = _items("t1")
    answer = {"task": "task1", "evidence": ["t1:2"],
              "causal_rationales": [{"cause": "damaged item", "effect": "refund approved"}]}
    ok, invalid = validate_citations(answer, evidence)
    assert ok and not invalid


def test_ambiguous_integer_citation_across_transcripts_is_rejected():
    a = [Evidence(turn=t, score=1.0, roles=["cause"], reason="x")
         for t in conv("a", [(1, "Customer", "refund because damaged")]).turns]
    b = [Evidence(turn=t, score=1.0, roles=["outcome"], reason="x")
         for t in conv("b", [(1, "Agent", "refund approved")]).turns]
    mapping = {"a": {1}, "b": {1}}
    assert parse_citation(1, mapping) is None  # ambiguous → never guessed (§29.4)
    answer = {"task": "task1", "evidence": [1]}
    ok, invalid = validate_citations(answer, a + b)
    assert not ok and invalid == [1]


def test_contradicted_claim_is_flagged():
    evidence = [Evidence(turn=t, score=1.0, roles=["outcome"], reason="x")
                for t in conv("t", [(1, "Agent", "your refund was approved")]).turns]
    answer = {"task": "task1", "evidence": [1],
              "causal_rationales": [{"cause": "the refund was denied", "effect": "refund approved"}]}
    status, _ = validate(answer, evidence)
    assert status == GroundingStatus.CONTRADICTED.value


def test_partially_grounded_threshold_band():
    answer = {
        "task": "task1",
        "evidence": [1, 2, 3],
        "causal_rationales": [{
            "cause": "completely unrelated topic about astronomy telescope",
            "effect": "refund approved",
        }],
    }
    status, _ = validate(answer, _items())
    assert status in {GroundingStatus.PARTIALLY_GROUNDED.value, GroundingStatus.UNSUPPORTED.value}


def test_unsupported_without_citations_is_rejected():
    answer = {"task": "task1", "causal_rationales": [{
        "cause": "damaged item refund", "effect": "refund approved",
    }]}
    status, _ = validate(answer, _items())
    assert status == GroundingStatus.UNSUPPORTED.value  # claims must cite turns


def test_no_evidence_is_unsupported():
    status, ratio = validate({"task": "task1", "causal_rationales": []}, [])
    assert status == GroundingStatus.UNSUPPORTED.value and ratio == 0.0
