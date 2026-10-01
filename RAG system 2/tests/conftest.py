from __future__ import annotations

import json
from pathlib import Path

import pytest

from system2.config import Config, DatasetConfig, EvidenceConfig, LoggingConfig, RetrievalConfig
from system2.llm import GenerationResult
from system2.pipeline import Pipeline
from system2.schema import Conversation, Turn
from system2.store import LexicalStore
from system2.textnorm import normalize_text


class FakeLLM:
    provider = "fake"
    model = "fake-model"

    def __init__(self, payload: dict | None = None, raw: str | None = None):
        self.payload = payload
        self.raw = raw
        self.calls: list[dict] = []

    def generate(self, system, prompt, *, temperature=0.0, num_predict=1400, timeout_s=300):
        self.calls.append({"system": system, "prompt": prompt})
        if self.raw is not None:
            text = self.raw
        else:
            text = json.dumps(self.payload or {})
        return GenerationResult(
            text=text,
            model=self.model,
            provider=self.provider,
            ttft_ms=1.0,
            generation_ms=10.0,
            prompt_tokens=100,
            output_tokens=20,
        )


def make_conversation(
    transcript_id: str,
    intent: str,
    texts: list[str],
    speakers: list[str] | None = None,
    domain: str = "Insurance",
    reason: str = "test reason",
) -> Conversation:
    conv = Conversation(
        transcript_id=transcript_id,
        time_of_interaction="2025-06-01 10:00:00",
        domain=domain,
        intent=intent,
        reason_for_call=reason,
        turns=[],
    )
    speakers = speakers or ["Agent", "Customer"] * (len(texts) // 2 + 1)
    for idx, text in enumerate(texts):
        original = speakers[idx] if idx < len(speakers) else "Agent"
        conv.turns.append(
            Turn(
                transcript_id=transcript_id,
                turn_id=idx,
                speaker_original=original,
                speaker="agent" if original == "Agent" else "customer",
                text_original=text,
                text_normalized=normalize_text(text),
            )
        )
    return conv


@pytest.fixture
def config(tmp_path: Path) -> Config:
    cfg = Config(
        dataset=DatasetConfig(
            path=str(tmp_path / "raw.json"),
            processed_dir=str(tmp_path / "processed"),
            index_dir=str(tmp_path / "index"),
            sample_size=6,
            sample_seed=42,
        ),
        retrieval=RetrievalConfig(),
        evidence=EvidenceConfig(min_score=0.20, min_conversation_score=0.20),
        logging=LoggingConfig(path=str(tmp_path / "logs" / "system2.jsonl")),
    )
    return cfg


@pytest.fixture
def sample_conversations() -> list[Conversation]:
    return [
        make_conversation(
            "conv-refund",
            "Billing & Payments",
            [
                "thank you for calling insurance support how can i help you today",
                "hi um i am really unhappy because i was charged twice this month",
                "i am sorry to hear that let me check your account",
                "the duplicate charge made me so frustrated i want a refund",
                "i can process a refund right now for the duplicate charge",
                "ok please do i need the money back as soon as possible",
                "the refund has been processed it will arrive in three days",
                "great that resolves my issue thank you",
            ],
        ),
        make_conversation(
            "conv-booking",
            "Bookings",
            [
                "thanks for calling the booking desk how can i assist",
                "hello i booked a room for july but the reservation shows the wrong date",
                "let me pull up your reservation i see the error",
                "this booking error caused me to miss my flight",
                "i will rebook you for the correct date at no extra cost",
                "that works for me please update the reservation",
                "your reservation is updated and confirmed",
                "perfect problem solved",
            ],
        ),
        make_conversation(
            "conv-upsell",
            "Upgrades & Add-ons",
            [
                "hello how can i help you today",
                "i was wondering about upgrading my plan",
                "we can upgrade you to the premium plan today",
                "i do not want the upsell unless the price is lower",
                "i can offer a discount on the premium plan",
                "maybe later i need to think about it",
                "sure take your time i will keep the note on your file",
                "thanks i will call back if i decide",
            ],
        ),
        make_conversation(
            "conv-complaint",
            "Service Complaints",
            [
                "good evening how may i help you",
                "i have a complaint your service has been terrible",
                "i understand your frustration tell me what happened",
                "the agent never called me back after i reported the problem",
                "i can escalate this for you today",
                "i would appreciate that thank you",
                "i have escalated the case and you will hear back tomorrow",
                "ok i hope this actually gets resolved",
            ],
        ),
        make_conversation(
            "conv-cancellation",
            "Cancellations",
            [
                "hi thanks for calling how can i help",
                "i want to cancel my policy because it is too expensive",
                "may i ask why you are cancelling",
                "the price increased and i cannot afford it",
                "i can offer a lower price if you stay",
                "no i already decided to cancel",
                "i have cancelled your policy effective today",
                "thank you that is all i needed",
            ],
        ),
        make_conversation(
            "conv-defect",
            "Claims",
            [
                "hello how can i help you",
                "my device is broken and not working since yesterday",
                "i am sorry what exactly is not working",
                "the screen is cracked after the fall",
                "i can file a claim for the damaged device",
                "please do that",
                "your claim is submitted and approved for replacement",
                "thanks for the quick resolution",
            ],
        ),
    ]


@pytest.fixture
def store(config: Config, sample_conversations: list[Conversation]) -> LexicalStore:
    store = LexicalStore(config.dataset.db_path)
    store.create_schema()
    store.insert_conversations(sample_conversations, "full")
    store.insert_turns(sample_conversations)
    store.build_fts()
    yield store
    store.close()


@pytest.fixture
def pipeline(config: Config, store: LexicalStore) -> Pipeline:
    return Pipeline(config, store=store, client=FakeLLM())
