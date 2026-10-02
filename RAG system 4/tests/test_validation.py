"""Unit tests: causal shape, transcript consistency, final answer gate (§23, §27, §30)."""

from __future__ import annotations

from system4.config.settings import Settings
from system4.decision.policies import DeterministicPolicy
from system4.evidence.gate import EvidenceFeatures
from system4.schema import AnswerGate, Evidence, GroundingStatus
from system4.validation.answer_gate import final_gate
from system4.validation.causal import validate_causal_shape
from system4.validation.consistency import transcript_consistency
from tests.conftest import conv

S = Settings()


def _items(tid: str = "t1") -> list[Evidence]:
    conversation = conv(tid, [
        (1, "Customer", "I need a refund because the item was damaged"),
        (2, "Agent", "I approved the refund after the photo"),
        (3, "Customer", "The refund fixed the issue thanks"),
    ])
    return [Evidence(turn=t, score=1.0, roles=["cause", "outcome"], reason="test")
            for t in conversation.turns]


def _decision():
    features = EvidenceFeatures(0.5, 1, 0.5, 1.0, 1.0, 0, 100, 3)
    return DeterministicPolicy().decide("task1", features, 1200, S)


def test_causal_shape_accepts_valid_answer():
    answer = {"task": "task1", "causal_rationales": [{
        "cause": "item was damaged", "effect": "refund approved",
        "mechanism": None, "evidence_turns": [1, 2],
    }], "insufficient_evidence": False}
    assert validate_causal_shape(answer, "task1", _items(), False) == []


def test_task_mismatch_detected():
    answer = {"task": "task2", "causal_rationales": [{
        "cause": "a", "effect": "b", "evidence_turns": [1],
    }], "insufficient_evidence": False}
    assert "TASK_MISMATCH" in validate_causal_shape(answer, "task1", _items(), False)


def test_cross_conversation_evidence_rejected_when_not_requested():
    evidence = _items("a") + _items("b")
    answer = {"task": "task1", "causal_rationales": [{
        "cause": "x", "effect": "y", "evidence_turns": [1],
    }], "insufficient_evidence": False}
    errors = validate_causal_shape(answer, "task1", evidence, False)
    assert "CROSS_CONVERSATION_EVIDENCE" in errors
    # ...but allowed when explicitly requested.
    assert validate_causal_shape(answer, "task1", evidence, True) == []


def test_mechanism_invention_detected():
    answer = {"task": "task1", "causal_rationales": [{
        "cause": "item was damaged", "effect": "refund approved",
        "mechanism": "the system automatically reprocessed the ledger",
        "evidence_turns": [1, 2],
    }], "insufficient_evidence": False}
    errors = validate_causal_shape(answer, "task1", _items(), False)
    assert "MECHANISM_UNSUPPORTED" in errors  # §15/§27: never invent mechanisms


def test_question_and_identical_claims_rejected():
    answer = {"task": "task1", "causal_rationales": [
        {"cause": "was the refund denied?", "effect": "refund approved", "evidence_turns": [1, 2]},
        {"cause": "same text", "effect": "same text", "evidence_turns": [1, 2]},
    ], "insufficient_evidence": False}
    errors = validate_causal_shape(answer, "task1", _items(), False)
    assert "QUESTION_AS_CAUSAL_CLAIM" in errors
    assert "CAUSE_EFFECT_IDENTICAL" in errors


def test_chronological_citation_order_enforced():
    answer = {"task": "task1", "causal_rationales": [{
        "cause": "damaged", "effect": "approved", "evidence_turns": [3, 1],
    }], "insufficient_evidence": False}
    assert "EVIDENCE_NOT_CHRONOLOGICAL" in validate_causal_shape(answer, "task1", _items(), False)


def test_consistency_flags_unknown_transcripts():
    answer = {"task": "task1", "source_transcripts": ["ghost-id"], "evidence": [1]}
    errors = transcript_consistency(answer, _items(), False)
    assert "UNKNOWN_TRANSCRIPT_ID" in errors


def test_consistency_accepts_single_source():
    answer = {"task": "task1", "source_transcripts": ["t1"], "evidence": [1]}
    assert transcript_consistency(answer, _items(), False) == []


def test_answer_gate_passes_grounded_answer():
    answer = {"task": "task1", "causal_rationales": [{
        "cause": "the item was damaged refund", "effect": "refund approved issue fixed",
        "evidence_turns": [1, 2, 3],
    }], "confidence": 0.9, "insufficient_evidence": False}
    outcome = final_gate(answer, _items(), _decision(), (GroundingStatus.GROUNDED.value, 0.9),
                         [], [], S)
    assert outcome.gate is AnswerGate.ANSWER


def test_answer_gate_abstains_without_evidence():
    outcome = final_gate({"task": "task1"}, [], _decision(), ("UNSUPPORTED", 0.0), [], [], S)
    assert outcome.gate is AnswerGate.ABSTAIN


def test_answer_gate_abstains_on_contradiction():
    answer = {"task": "task1", "causal_rationales": [{
        "cause": "a", "effect": "b", "evidence_turns": [1],
    }]}
    outcome = final_gate(answer, _items(), _decision(),
                         (GroundingStatus.CONTRADICTED.value, 0.0), [], [], S)
    assert outcome.gate is AnswerGate.ABSTAIN


def test_answer_gate_prunes_unsupported_claims():
    answer = {"task": "task1", "causal_rationales": [
        {"cause": "the item was damaged refund", "effect": "refund approved issue fixed",
         "evidence_turns": [1, 2, 3]},
        {"cause": "unrelated astronomy telescope launch", "effect": "mars colonization failed",
         "evidence_turns": [1]},
    ], "confidence": 0.9, "insufficient_evidence": False}
    outcome = final_gate(answer, _items(), _decision(),
                         (GroundingStatus.PARTIALLY_GROUNDED.value, 0.6), [], [], S)
    assert outcome.gate in {AnswerGate.ANSWER_WITH_UNCERTAINTY, AnswerGate.ABSTAIN}
    if outcome.gate is AnswerGate.ANSWER_WITH_UNCERTAINTY:
        assert len(answer["causal_rationales"]) == 1  # unsupported claim removed (§30)
        assert answer["causal_rationales"][0]["cause"].startswith("the item")


def test_answer_gate_abstains_when_every_claim_pruned():
    answer = {"task": "task1", "causal_rationales": [
        {"cause": "unrelated astronomy telescope", "effect": "mars colonization failed",
         "evidence_turns": [1]},
    ], "confidence": 0.9, "insufficient_evidence": False}
    outcome = final_gate(answer, _items(), _decision(),
                         (GroundingStatus.PARTIALLY_GROUNDED.value, 0.3), [], [], S)
    assert outcome.gate is AnswerGate.ABSTAIN
    assert answer["causal_rationales"] == []
