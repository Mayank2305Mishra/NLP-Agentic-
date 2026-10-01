"""Unit: evidence dedupe/diversity/budget + query validation + redaction + config."""
from __future__ import annotations

import pytest

from app.config import Settings
from app.domain.errors import ContextBudgetError, DataValidationError
from app.domain.models import EvidenceChunk
from app.observability.redaction import redact, redact_dict
from app.retrieval.evidence import build_evidence_pack
from app.retrieval.query import normalize_query


def mk(cid, tid, text, score=0.9, rank=0):
    return EvidenceChunk(chunk_id=cid, transcript_id=tid, text=text, score=score,
                         rank=rank, turn_start=0, turn_end=1)


def test_evidence_dedupe_and_diversity(tmp_path):
    s = Settings(source_path=tmp_path / "x.json", generated_dir=tmp_path / "g",
                 evidence_top_k=4, evidence_max_per_transcript=1)
    chunks = [mk("a:chunk:0", "a", "same text same text", 0.9, 0),
              mk("b:chunk:0", "b", "same text same text", 0.8, 1),  # dup text
              mk("a:chunk:1", "a", "other text here", 0.7, 2),      # same transcript cap
              mk("c:chunk:0", "c", "third unique piece", 0.6, 3)]
    pack = build_evidence_pack(chunks, s)
    ids = [c.chunk_id for c in pack.chunks]
    assert "b:chunk:0" not in ids  # duplicate removed
    assert ids.count("a:chunk:0") == 1
    assert "c:chunk:0" in ids


def test_context_budget_drops_lowest_ranked(tmp_path):
    s = Settings(source_path=tmp_path / "x.json", generated_dir=tmp_path / "g",
                 evidence_max_context_tokens=30)
    chunks = [mk("a:chunk:0", "a", "word " * 20, 0.9, 0),   # ~25 tokens, fits alone
              mk("b:chunk:0", "b", "word " * 16, 0.1, 1)]   # ~20 tokens; together over budget
    pack = build_evidence_pack(chunks, s)
    assert pack.truncated and [c.chunk_id for c in pack.chunks] == ["a:chunk:0"]


def test_empty_candidates_raise_budget_error(tmp_path):
    s = Settings(source_path=tmp_path / "x.json", generated_dir=tmp_path / "g")
    with pytest.raises(ContextBudgetError):
        build_evidence_pack([], s)


def test_empty_and_whitespace_query_rejected():
    with pytest.raises(DataValidationError):
        normalize_query("   ")
    with pytest.raises(DataValidationError):
        normalize_query("")


def test_unknown_task_rejected():
    with pytest.raises(Exception):
        normalize_query("hello", task="task9")


def test_redaction():
    assert "[redacted-email]" in redact("contact a@b.com now")
    assert "[redacted-phone]" in redact("call 415-555-1234 now")
    d = redact_dict({"authorization": "Bearer abc", "msg": "mail a@b.com"})
    assert d["authorization"] == "[redacted]"
    assert "[redacted-email]" in d["msg"]
