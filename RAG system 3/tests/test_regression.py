"""Regression tests for the eight known example categories (spec §37)."""

from __future__ import annotations

import pytest

from system3.pipeline import Pipeline
from tests.conftest import FakeLLM, make_conversation


def _conv_pricing():
    return make_conversation(
        "conv-pricing",
        "Billing & Payments",
        [
            "good afternoon how can i assist you today",
            "i checked my bill and i was overcharged this month",
            "i am sorry let me review the charges on your account",
            "the price is much higher than i was quoted and i am unhappy",
            "i can see the error i will issue a credit for the difference",
            "thank you i do not want to be overcharged again",
            "the credit has been applied your price is correct now",
            "great that resolves my concern",
        ],
    )


def _conv_compare():
    return make_conversation(
        "conv-compare",
        "Plan Comparisons",
        [
            "hello how can i help you",
            "i want to compare plans before i make a decision",
            "i can walk you through the differences between plans",
            "the other plan looks cheaper but i am not sure",
            "both plans have benefits let me explain",
            "i will compare the options with my spouse",
            "sure take your time",
            "thanks i will choose the plan that fits best",
        ],
    )


@pytest.fixture
def regression_store(store):
    extra = [_conv_pricing(), _conv_compare()]
    store.insert_conversations(extra, "full")
    store.insert_turns(extra)
    store.build_fts()
    yield store


REGRESSION_CASES = [
    ("refund", "conv-refund", "why did the customer ask for a refund for the duplicate charge"),
    ("cancellation", "conv-cancellation", "why did the customer cancel the policy because it is too expensive"),
    ("booking_error", "conv-booking", "the reservation shows the wrong date after the booking error"),
    ("pricing_dissatisfaction", "conv-pricing", "why was the customer overcharged and unhappy about the price"),
    ("plan_comparison", "conv-compare", "which plan did the customer decide to choose"),
    ("upsell", "conv-upsell", "how did the customer react to the premium upsell offer"),
    ("product_defect", "conv-defect", "why is the device broken and the screen cracked"),
    ("service_complaint", "conv-complaint", "what complaint did the customer make about the service"),
]


@pytest.mark.parametrize(
    "category,expected_tid,query",
    REGRESSION_CASES,
    ids=[c[0] for c in REGRESSION_CASES],
)
def test_regression_retrieval_ranking(config, regression_store, category, expected_tid, query):
    pipeline = Pipeline(config, store=regression_store, client=FakeLLM())
    outcome = pipeline.retrieve_only(query, task="task1")
    retrieval = outcome["retrieval"]

    assert retrieval.candidates, f"{category}: no candidates"
    assert (
        retrieval.candidates[0].transcript_id == expected_tid
    ), f"{category}: expected {expected_tid} first, got {retrieval.candidates[0].transcript_id}"
    assert retrieval.gate.decision != "INSUFFICIENT", f"{category}: gate INSUFFICIENT"

    pack = retrieval.pack
    assert pack is not None, f"{category}: no evidence pack"
    assert pack.allowed_transcript_ids == [
        expected_tid
    ], f"{category}: pack must stay conversation-local"
