"""Shared fixtures: small deterministic conversations (no full-dataset dependency)."""

from __future__ import annotations

import pytest

from system4.config.settings import Settings
from system4.retrieval.index import LocalIndex
from system4.schema import Conversation, Turn


def conv(tid: str, turns: list[tuple[int, str, str]], *, domain: str | None = "Retail",
         intent: str | None = "Refunds", reason: str | None = None) -> Conversation:
    return Conversation(
        transcript_id=tid,
        domain=domain,
        intent=intent,
        reason_for_call=reason,
        turns=[Turn(tid, n, speaker, text, domain, intent, reason) for n, speaker, text in turns],
    )


@pytest.fixture
def refund_conversation() -> Conversation:
    return conv(
        "refund-1",
        [
            (1, "Customer", "I need a refund because the item arrived damaged."),
            (2, "Agent", "I can approve the refund after receiving the photo."),
            (3, "Customer", "Thanks, that resolves the issue."),
        ],
        domain="Retail",
        intent="Refunds",
        reason="Customer asks for a refund of a damaged item.",
    )


@pytest.fixture
def insurance_conversation() -> Conversation:
    return conv(
        "insurance-1",
        [(1, "Customer", "Can I get a policy discount on my insurance?")],
        domain="Insurance",
        intent="Discounts",
        reason="Discount inquiry for insurance policy.",
    )


@pytest.fixture
def settings() -> Settings:
    return Settings(dense_k=8, lexical_k=8, min_rerank_score=0.01, cache_answers=False)


@pytest.fixture
def index(refund_conversation, insurance_conversation, settings) -> LocalIndex:
    return LocalIndex.build([refund_conversation, insurance_conversation], settings)
