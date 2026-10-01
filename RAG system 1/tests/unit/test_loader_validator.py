"""Unit: loader + validator edge cases (golden 8/10/11 + malformed JSON)."""
from __future__ import annotations

import json

import pytest

from app.domain.errors import DataValidationError
from app.ingestion.loader import load_raw_records
from app.ingestion.validator import validate_records


def write(tmp_path, name, content: bytes):
    p = tmp_path / name
    p.write_bytes(content)
    return p


def test_malformed_json_reports_line(tmp_path):
    p = write(tmp_path, "bad.json", b'[{"a": 1},]')
    with pytest.raises(DataValidationError) as e:
        load_raw_records(p)
    assert "line" in str(e.value).lower()


def test_broken_utf8_fails(tmp_path):
    p = write(tmp_path, "bad2.json", b'[{"transcript_id": "\xff"}]')
    with pytest.raises(DataValidationError):
        load_raw_records(p)


def test_root_must_be_array(tmp_path):
    p = write(tmp_path, "obj.json", json.dumps({"a": 1}).encode())
    with pytest.raises(DataValidationError):
        load_raw_records(p)


def test_object_wrapper_supported(tmp_path):
    recs = [{"transcript_id": "a", "conversation": [{"speaker": "Agent", "text": "hi"}]}]
    p = write(tmp_path, "w.json", json.dumps({"transcripts": recs}).encode())
    out, _, _ = load_raw_records(p)
    assert out == recs


def test_duplicate_ids_fail_closed():
    rec = {"transcript_id": "dup", "conversation": [{"speaker": "A", "text": "hi"}]}
    with pytest.raises(DataValidationError):
        validate_records([rec, dict(rec)], on_duplicate="fail")


def test_duplicate_keep_first_policy():
    rec = {"transcript_id": "dup", "conversation": [{"speaker": "A", "text": "hi"}]}
    usable, summary = validate_records([rec, dict(rec)], on_duplicate="keep_first")
    assert len(usable) == 1 and summary.excluded == 1


def test_empty_conversation_excluded():
    usable, summary = validate_records([{"transcript_id": "e", "conversation": []}])
    assert usable == [] and summary.excluded == 1


def test_malformed_turns_skipped_record_kept():
    rec = {"transcript_id": "m", "conversation": [
        {"speaker": "Agent", "text": ""},          # empty -> skip
        {"nope": 1},                                # malformed -> skip
        {"speaker": "Customer", "text": "real issue because X"},
    ]}
    usable, summary = validate_records([rec])
    assert len(usable) == 1 and summary.reports[0].ok


def test_missing_transcript_id_excluded():
    usable, summary = validate_records([{"conversation": [{"speaker": "A", "text": "x"}]}])
    assert usable == [] and summary.excluded == 1
