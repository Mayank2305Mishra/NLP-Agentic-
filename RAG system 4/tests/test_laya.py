"""Unit tests: Laya typed decisions, policies, Jev adapter (§17–§20)."""

from __future__ import annotations

import pytest

from system4.config.settings import Settings
from system4.decision.jev import JevAdapter, from_protocol, to_protocol
from system4.decision.laya import Laya, decide
from system4.decision.policies import DeterministicPolicy, decide_with_fallback, route_task
from system4.evidence.gate import EvidenceFeatures
from system4.schema import AnswerGate, CompressionLevel, EvidenceAction, TaskRoute

S = Settings()


def _features(**overrides) -> EvidenceFeatures:
    base = dict(
        top_rerank_score=0.5,
        conversation_count=1,
        entity_consistency=0.5,
        causal_coverage=1.0,
        outcome_coverage=1.0,
        contradiction_count=0,
        evidence_tokens=100,
        evidence_count=6,
    )
    base.update(overrides)
    return EvidenceFeatures(**base)


def test_accept_path():
    decision = DeterministicPolicy().decide("task1", _features(), 1200, S)
    assert decision.evidence_action is EvidenceAction.ACCEPT
    assert decision.answer_gate is AnswerGate.ANSWER
    assert decision.compression_level is CompressionLevel.LIGHT
    assert decision.task_route is TaskRoute.TASK_1


def test_reject_when_no_evidence():
    decision = DeterministicPolicy().decide("task1", _features(evidence_count=0, evidence_tokens=0), 1200, S)
    assert decision.evidence_action is EvidenceAction.REJECT
    assert decision.answer_gate is AnswerGate.ABSTAIN


def test_expand_when_causal_coverage_low():
    decision = DeterministicPolicy().decide("task1", _features(causal_coverage=0.0), 1200, S)
    assert decision.evidence_action is EvidenceAction.EXPAND
    assert decision.answer_gate is AnswerGate.ANSWER_WITH_UNCERTAINTY


def test_contradiction_forces_uncertainty():
    decision = DeterministicPolicy().decide("task1", _features(contradiction_count=1), 1200, S)
    assert decision.evidence_action is EvidenceAction.EXPAND
    assert decision.answer_gate is AnswerGate.ANSWER_WITH_UNCERTAINTY


def test_aggressive_compression_over_budget():
    decision = DeterministicPolicy().decide("task1", _features(evidence_tokens=5000), 1200, S)
    assert decision.compression_level is CompressionLevel.AGGRESSIVE


def test_missing_outcome_yields_uncertainty_not_answer():
    decision = DeterministicPolicy().decide(
        "task2", _features(outcome_coverage=0.0, causal_coverage=2 / 3), 2200, S
    )
    assert decision.answer_gate is AnswerGate.ANSWER_WITH_UNCERTAINTY


def test_policy_failure_falls_back_to_deterministic():
    class Exploding:
        name = "exploding"

        def decide(self, *args, **kwargs):
            raise RuntimeError("boom")

    decision = decide_with_fallback("task1", _features(), 1200, S, Exploding())
    assert decision.evidence_action is EvidenceAction.ACCEPT


def test_fail_closed_when_even_fallback_breaks(monkeypatch):
    class Exploding:
        name = "exploding"

        def decide(self, *args, **kwargs):
            raise RuntimeError("boom")

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("system4.decision.policies.DeterministicPolicy.decide", boom)
    decision = decide_with_fallback("task1", _features(), 1200, S, Exploding())
    assert decision.answer_gate is AnswerGate.ABSTAIN
    assert decision.policy == "fail-closed"


def test_task_routing():
    assert route_task("Why was the refund denied?") == "task1"
    assert route_task("Summarize this conversation end to end") == "task2"
    assert route_task("what happened here") == "task2"
    assert route_task("something neutral") == "task1"


def test_laya_facade_and_legacy_decide():
    laya = Laya()
    assert laya.route("why did this happen") == "task1"
    decision = laya.decide("task1", _features(), 1200, S)
    assert isinstance(decision.evidence_action, EvidenceAction)
    legacy = decide("task1", 0.5, 6, 1.0, 0, 100, 1200)
    assert legacy.answer_gate is AnswerGate.ANSWER


def test_jev_protocol_roundtrip():
    decision = DeterministicPolicy().decide("task1", _features(), 1200, S)
    payload = to_protocol(decision, _features())
    assert payload["schema"] == "jev.system-one.decision.v1"
    restored = from_protocol(payload)
    assert restored == decision


def test_jev_adapter_local_first_when_endpoint_unreachable():
    adapter = JevAdapter(endpoint="http://127.0.0.1:1/jev", timeout=0.2)
    decision = adapter.decide("task1", _features(), 1200, S)
    # Remote failure must never block the local policy (§20, §43).
    assert decision.answer_gate is AnswerGate.ANSWER


def test_jev_adapter_without_endpoint_is_pure_local():
    adapter = JevAdapter(endpoint=None)
    decision = adapter.decide("task1", _features(contradiction_count=2), 1200, S)
    assert decision.answer_gate is AnswerGate.ANSWER_WITH_UNCERTAINTY
