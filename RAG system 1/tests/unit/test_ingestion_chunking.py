"""Unit: normalization rules + conversation-aware chunking."""
from __future__ import annotations

from app.indexing.chunker import chunk_transcript, estimate_tokens
from app.ingestion.normalizer import normalize_record, normalize_text


def test_normalize_keeps_negation_numbers_entities():
    t = "I did NOT get booking AB-12345 for $200 on Monday!!  [pause]  um..."
    out = normalize_text(t)
    assert "NOT" in out or "not" in out.lower()
    assert "AB-12345" in out or "ab-12345" in out.lower()
    assert "200" in out
    assert "[pause]" not in out


def test_normalize_record_preserves_verbatim_and_order():
    rec = {"transcript_id": "t1", "domain": None,
           "conversation": [{"speaker": "Agent", "text": "  hello   world "},
                            {"speaker": "Customer", "text": ""},
                            {"speaker": "WeirdBot", "text": "beep"}]}
    nt = normalize_record(rec)
    assert [t.speaker for t in nt.turns] == ["Agent", "WeirdBot"]  # unknown label preserved
    assert nt.turns[0].text == "hello   world"  # verbatim (outer ws only)
    assert nt.turns[0].retrieval_text == "hello world"
    assert nt.domain is None  # null metadata never fabricated
    assert nt.speaker_anomaly  # unknown label flagged, not crashed


def test_single_speaker_allowed_with_anomaly():
    rec = {"transcript_id": "s", "conversation": [{"speaker": "Agent", "text": "only me"}]}
    nt = normalize_record(rec)
    assert len(nt.turns) == 1 and nt.speaker_anomaly


def test_chunk_ids_stable_and_windows_overlap():
    # Short turns so each window holds several turns and overlap is observable.
    turns = [{"speaker": "Agent" if i % 2 == 0 else "Customer", "text": f"msg {i} ok"}
             for i in range(12)]
    rec = {"transcript_id": "abc", "domain": "Travel", "intent": "X",
           "reason_for_call": "Y", "conversation": turns}
    nt = normalize_record(rec)
    chunks = chunk_transcript(nt, target_tokens=12, max_tokens=30, overlap_turns=2)
    assert len(chunks) >= 2
    assert chunks[0].chunk_id == "abc:chunk:0"
    assert all(c.chunk_id.startswith("abc:chunk:") for c in chunks)
    # overlap: second window starts before first window ends
    assert chunks[1].turn_start <= chunks[0].turn_end
    # turn order preserved inside chunk
    assert "Turn 001" in chunks[0].text


def test_very_short_transcript_single_chunk():
    rec = {"transcript_id": "short", "conversation": [{"speaker": "A", "text": "hi"}]}
    chunks = chunk_transcript(normalize_record(rec))
    assert len(chunks) == 1


def test_estimate_tokens_not_char_count():
    assert estimate_tokens("abcd") == 1  # 4 chars ~= 1 token
