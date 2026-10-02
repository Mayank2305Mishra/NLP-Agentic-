"""Unit tests: normalization, post-call detection, artifact versioning (§7, §24)."""

from __future__ import annotations

import json

from system4.config.settings import Settings
from system4.ingestion.artifacts import load_artifact, save, source_hash
from system4.ingestion.loader import load_conversations
from system4.ingestion.normalize import conversation_hash, mark_post_call, normalize_record
from system4.schema import SYSTEM_VERSION, Turn


def _turn(n: int, text: str, speaker: str = "Agent") -> Turn:
    return Turn("t1", n, speaker, text)


def test_normalize_preserves_speaker_text_and_metadata():
    record = {
        "transcript_id": "abc",
        "domain": "Travel",
        "intent": "Refund Policy",
        "reason_for_call": "Refund for canceled flight",
        "conversation": [
            {"speaker": "Nora Diaz", "text": "  hello   there  "},
            {"speaker": "Sophia Patel", "text": "how can I help"},
        ],
    }
    conversation = normalize_record(record)
    assert conversation is not None
    assert conversation.turns[0].speaker == "Nora Diaz"
    assert conversation.turns[0].text == "hello there"
    assert conversation.turns[0].turn_id == 1
    assert conversation.domain == "Travel"
    assert conversation.reason_for_call == "Refund for canceled flight"


def test_normalize_preserves_original_turn_numbers():
    record = {
        "transcript_id": "abc",
        "conversation": [
            {"turn_number": 10, "speaker": "A", "text": "first"},
            {"turn_number": 11, "speaker": "B", "text": "second"},
        ],
    }
    conversation = normalize_record(record)
    assert [t.turn_id for t in conversation.turns] == [10, 11]


def test_normalize_skips_malformed_records():
    assert normalize_record({}) is None
    assert normalize_record({"transcript_id": "x", "conversation": "not-a-list"}) is None
    assert normalize_record({"transcript_id": "x", "conversation": [{"speaker": "A"}]}) is None


def test_metadata_nulls_stay_null():
    conversation = normalize_record(
        {"transcript_id": "x", "conversation": [{"speaker": "A", "text": "hi"}]}
    )
    assert conversation.domain is None and conversation.intent is None


def test_post_call_notes_are_marked_and_excluded_by_default():
    turns = [
        _turn(1, "hi thanks for calling"),
        _turn(2, "the refund was approved"),
        _turn(3, "Internal note: customer called back twice about the refund"),
        _turn(4, "QA note: handle with care next time"),
    ]
    marked = mark_post_call(turns)
    assert [t.post_call for t in marked] == [False, False, True, True]


def test_no_post_call_without_trailing_notes():
    turns = [_turn(1, "hello"), _turn(2, "goodbye")]
    assert all(not t.post_call for t in mark_post_call(turns))


def test_conversation_hash_deterministic():
    a = normalize_record({"transcript_id": "x", "conversation": [{"speaker": "A", "text": "hi"}]})
    b = normalize_record({"transcript_id": "x", "conversation": [{"speaker": "A", "text": "hi"}]})
    assert conversation_hash(a) == conversation_hash(b)


def test_loader_counts_skipped(tmp_path):
    source = tmp_path / "data.json"
    source.write_text(json.dumps([
        {"transcript_id": "ok", "conversation": [{"speaker": "A", "text": "hi"}]},
        {"transcript_id": "", "conversation": [{"speaker": "A", "text": "hi"}]},
        "not-a-dict",
    ]))
    stats: dict = {}
    conversations = list(load_conversations(source, stats))
    assert len(conversations) == 1
    assert stats["records_skipped"] == 1


def test_source_hash_and_artifact_version_check(tmp_path):
    source = tmp_path / "data.json"
    source.write_text("[]")
    assert len(source_hash(source)) == 64

    artifact = tmp_path / "index.pkl"
    save(artifact, {"index": object(), "version": SYSTEM_VERSION})
    assert load_artifact(artifact, expected_version=SYSTEM_VERSION)["version"] == SYSTEM_VERSION
    try:
        load_artifact(artifact, expected_version="other-version")
        raise AssertionError("expected version mismatch error")
    except ValueError as exc:
        assert "rebuild" in str(exc)


def test_settings_fingerprint_is_stable_and_config_sensitive():
    a, b = Settings(), Settings()
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != Settings(dense_k=99).fingerprint()
    # Runtime-only fields must not change answer cache keys (§39).
    assert Settings().fingerprint() == Settings(trace_enabled=False).fingerprint()
