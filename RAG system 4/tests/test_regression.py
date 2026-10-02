"""Regression + acceptance tests (AGENTS_SYSTEM_4.md §47, §48).

Stores known failure cases: unrelated insurance/refund mixing, weak-similarity
candidates, missing outcomes, post-call notes, noisy speakers, contradictory
evidence, and the §32 instrumentation contract.
"""

from __future__ import annotations

import json

from system4.config.settings import Settings
from system4.pipeline import QualityRAG
from system4.retrieval.index import LocalIndex
from system4.schema import Conversation, Turn
from tests.conftest import conv

#: §32 stage timings that must always be present in a trace.
REQUIRED_STAGES = (
    "query_normalization_ms",
    "metadata_filter_ms",
    "dense_retrieval_ms",
    "lexical_retrieval_ms",
    "candidate_merge_ms",
    "candidate_retrieval_ms",
    "reranker_ms",
    "turn_selection_ms",
    "expansion_ms",
    "laya_ms",
    "compression_ms",
    "evidence_pack_ms",
    "prompt_build_ms",
    "llm_ttft_ms",
    "llm_generation_ms",
    "output_parse_ms",
    "grounding_validation_ms",
    "answer_gate_ms",
    "total_ms",
)


def _settings(**kwargs) -> Settings:
    base = dict(dense_k=8, lexical_k=8, min_rerank_score=0.01, cache_answers=False)
    base.update(kwargs)
    return Settings(**base)


def _diverse_index(settings: Settings | None = None) -> LocalIndex:
    """Refund conversation plus an insurance/discount distractor with shared jargon (§3.1)."""
    refund = conv(
        "refund-9",
        [
            (1, "Customer", "I want a refund because my order was cancelled by mistake"),
            (2, "Agent", "I can approve the refund since the cancellation was our error"),
            (3, "Customer", "Great, the refund fixed my problem, thank you"),
        ],
        domain="Retail", intent="Refunds",
        reason="Customer requests a refund after an erroneous cancellation.",
    )
    insurance = conv(
        "insurance-7",
        [
            (1, "Customer", "I am calling about a refund option on my insurance discount"),
            (2, "Agent", "The insurance policy discount cannot be refunded"),
        ],
        domain="Insurance", intent="Discounts",
        reason="Insurance discount refund inquiry.",
    )
    noisy = conv(
        "noisy-3",
        [
            (1, "um Customer", "hi um so like um my internet is not working i think"),
            (2, "", "okay let me um check that for you right away"),
        ],
        domain="Utilities", intent="Technical Support",
        reason="Internet outage reported with heavy conversational noise.",
    )
    return LocalIndex.build([refund, insurance, noisy], settings or _settings())


def test_no_cross_conversation_contamination_on_regression_suite():
    """§48: no cross-conversation contamination; one answer = one interaction."""
    rag = QualityRAG(_diverse_index(), _settings())
    result = rag.ask("Why was the refund approved after the cancellation?", "task1")
    assert result.answer["source_transcripts"] == ["refund-9"]
    assert {e.turn.transcript_id for e in result.evidence} == {"refund-9"}
    assert set(result.trace["selected_transcript_ids"]) == {"refund-9"}


def test_weak_similarity_abstains():
    """§3.2: similarity is not evidence quality — weak candidates are refused."""
    rag = QualityRAG(_diverse_index(), _settings(min_rerank_score=0.9))
    result = rag.ask("quantum chromodynamics lattice", "task1")
    assert result.answer["insufficient_evidence"] is True
    assert result.trace["answer_gate"] == "ABSTAIN"


def test_missing_outcome_becomes_uncertainty_not_hallucination():
    """§16: represent the missing chain part as uncertainty."""
    incomplete = conv(
        "incomplete-1",
        [
            (1, "Customer", "The payment failed because my card was expired"),
            (2, "Agent", "I can update the card details on file"),
        ],
        domain="Retail", intent="Payments", reason="Payment failed due to expired card.",
    )
    settings = _settings()
    rag = QualityRAG(LocalIndex.build([incomplete], settings), settings)
    result = rag.ask("Why did the payment fail?", "task1")
    if result.answer.get("causal_rationales"):
        for rationale in result.answer["causal_rationales"]:
            assert rationale.get("mechanism") is None, "mechanism must never be invented"
    assert result.answer["grounding_status"] in {"GROUNDED", "PARTIALLY_GROUNDED", "UNSUPPORTED"}
    # Laya observes the missing outcome (§19).
    assert "outcome" not in " ".join(
        role for e in result.evidence for role in e.roles
    ) or result.trace["laya_answer_gates"]


def test_post_call_notes_are_not_customer_evidence():
    """§24: post-call artifacts classified separately, excluded by default."""
    conversation = Conversation(
        "post-1", "Retail", "Refunds", "Refund question",
        [
            Turn("post-1", 1, "Customer", "I need a refund because the item was broken"),
            Turn("post-1", 2, "Agent", "I approved the refund for you"),
            Turn("post-1", 3, "QA Agent", "Internal note: refund approved by supervisor review", post_call=True),
        ],
    )
    settings = _settings()
    rag = QualityRAG(LocalIndex.build([conversation], settings), settings)
    result = rag.ask("Why was the refund approved?", "task1")
    assert all(not e.turn.post_call for e in result.evidence)
    assert all(e.turn.turn_id != 3 for e in result.evidence)


def test_contradictory_evidence_surfaces_uncertainty():
    """§19/§47: contradictory evidence is never silently discarded."""
    contradictory = conv(
        "contra-1",
        [
            (1, "Customer", "I need a refund because the item was broken"),
            (2, "Agent", "Your refund was approved by the system"),
            (3, "Agent", "Actually your refund was denied after review"),
        ],
        domain="Retail", intent="Refunds", reason="Refund status inquiry.",
    )
    settings = _settings()
    rag = QualityRAG(LocalIndex.build([contradictory], settings), settings)
    result = rag.ask("Was the refund approved or denied?", "task1")
    gates = result.trace.get("laya_answer_gates", [])
    assert result.trace["laya_decisions"]  # decisions typed and observable (§18)
    # Laya must force uncertainty when opposite outcomes appear across turns (§19).
    assert "ANSWER_WITH_UNCERTAINTY" in gates or "ABSTAIN" in gates


def test_noisy_speakers_are_handled():
    """§47: noisy speakers and hesitations must not break the pipeline."""
    rag = QualityRAG(_diverse_index(), _settings())
    result = rag.ask("Why is the internet not working?", "task1")
    assert result.trace["system"] == "system4"
    assert result.trace["selected_transcript_ids"] == ["noisy-3"]


def test_every_answer_is_traceable_and_instrumented():
    """§48: every answer has traceable evidence; all major stages instrumented."""
    rag = QualityRAG(_diverse_index(), _settings())
    result = rag.ask("Why was the refund approved after the cancellation?", "task1")
    for stage in REQUIRED_STAGES:
        assert stage in result.trace, f"missing stage timing: {stage}"
    if not result.answer.get("insufficient_evidence"):
        assert result.evidence, "an answer must cite evidence"
        assert all(item.turn.transcript_id for item in result.evidence)
        assert result.answer["source_transcripts"]


def test_llm_can_be_swapped_without_touching_retrieval():
    """§48: provider swap requires no retrieval changes (§25)."""

    class StubLLM:
        identifier = "stub:test"

        def generate(self, prompt: str, **kwargs: object) -> str:
            assert "<CONVERSATION_EVIDENCE>" in prompt
            return json.dumps({
                "task": "task1",
                "causal_rationales": [{
                    "cause": "the order was cancelled by mistake refund",
                    "effect": "refund approved cancellation was our error",
                    "mechanism": None,
                    "evidence_turns": [1, 2, 3],
                    "strength": "explicit",
                }],
                "confidence": 0.9,
                "insufficient_evidence": False,
            })

    settings = _settings()
    rag = QualityRAG(_diverse_index(), settings, llm=StubLLM())
    result = rag.ask("Why was the refund approved after the cancellation?", "task1")
    assert result.trace["generation_mode"] == "llm"
    assert result.answer.get("causal_rationales"), result.answer
    assert result.answer["grounding_status"] in {"GROUNDED", "PARTIALLY_GROUNDED"}


def test_broken_llm_degrades_to_deterministic_fallback():
    """§43: LLM failure must degrade, never fabricate or crash."""

    class BrokenLLM:
        identifier = "broken:test"

        def generate(self, prompt: str, **kwargs: object) -> str:
            raise RuntimeError("provider down")

    settings = _settings()
    rag = QualityRAG(_diverse_index(), settings, llm=BrokenLLM())
    result = rag.ask("Why was the refund approved after the cancellation?", "task1")
    assert result.trace["generation_mode"] == "deterministic_fallback"
    assert any(e["stage"] == "llm_generation" for e in result.trace["errors"])
    # Deterministic fallback still yields an evidence-cited, schema-valid answer.
    assert result.answer["task"] == "task1"
    assert result.evidence


def test_invalid_model_json_degrades_to_fallback():
    """§43: JSON parse failure uses the deterministic evidence-cited answer."""

    class GarbageLLM:
        identifier = "garbage:test"

        def generate(self, prompt: str, **kwargs: object) -> str:
            return "I could not produce JSON, sorry."

    settings = _settings()
    rag = QualityRAG(_diverse_index(), settings, llm=GarbageLLM())
    result = rag.ask("Why was the refund approved after the cancellation?", "task1")
    assert result.trace["generation_mode"] == "parse_fallback"
    assert any(e["stage"] == "output_parse" for e in result.trace["errors"])


def test_traces_never_contain_transcript_text():
    """§41: default logging carries ids/scores/counts, never raw content."""
    rag = QualityRAG(_diverse_index(), _settings())
    result = rag.ask("Why was the refund approved after the cancellation?", "task1")
    blob = json.dumps(result.trace, default=str)
    assert "refund because my order" not in blob
    assert "insurance discount" not in blob
