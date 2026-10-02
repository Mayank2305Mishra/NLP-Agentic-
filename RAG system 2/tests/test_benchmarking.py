from __future__ import annotations

import json

from system2.benchmarking.runner import run_benchmark
from tests.conftest import FakeLLM


def _write_fixture(path, questions):
    payload = {
        "version": 1,
        "seed": 7,
        "count": len(questions),
        "source_hash": "a" * 64,
        "questions": questions,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _questions():
    return [
        {
            "question_id": "q001",
            "category": "refund",
            "question": "why was the customer charged twice and what did the agent do?",
            "transcript_id": "conv-refund",
            "task": "task1",
            "expected_evidence_turns": [3, 4, 6],
        },
        {
            "question_id": "q002",
            "category": "cancellation",
            "question": "why did the customer cancel the policy",
            "transcript_id": "conv-cancellation",
            "task": "task1",
            "expected_evidence_turns": [1, 5],
        },
    ]


def test_benchmark_retrieval_only(config, store, tmp_path):
    fixture = _write_fixture(tmp_path / "fixture.json", _questions())
    out_dir = tmp_path / "results"

    result = run_benchmark(
        config,
        fixture_path=str(fixture),
        use_llm=False,
        output_dir=out_dir,
    )

    summary = result["summary"]
    assert summary["questions"] == 2
    assert summary["retrieval"]["hit_at_1_rate"] == 1.0
    assert summary["retrieval"]["recall_at_3"] == 1.0
    assert summary["retrieval"]["contamination_rate"] == 0.0

    total = summary["latency"]["total"]
    assert total["p50_ms"] is not None and total["p95_ms"] is not None
    assert summary["latency"]["retrieval"]["mean_ms"] is not None

    operational = summary["operational"]
    assert operational["cold_start_ms"] is not None
    assert operational["peak_rss_mb"] is None or operational["peak_rss_mb"] > 0
    assert operational["index_hash"]
    assert operational["index_size_bytes"] > 0

    repro = result["reproducibility"]
    assert repro["fixture_seed"] == 7
    assert repro["config_hash"]
    assert repro["prompt_version"]
    assert repro["source_hash"] == "a" * 64
    assert repro["git_commit"] is None or isinstance(repro["git_commit"], str)

    written = list(out_dir.glob("*.json"))
    assert len(written) == 1, "benchmark must write results to the configured output dir"
    on_disk = json.loads(written[0].read_text(encoding="utf-8"))
    assert on_disk["summary"]["questions"] == 2


def test_benchmark_with_llm_reports_generation_metrics(config, store, tmp_path):
    fixture = _write_fixture(tmp_path / "fixture.json", _questions())
    fake = FakeLLM(
        payload={
            "task": "task1",
            "causal_rationales": [
                {
                    "cause": "duplicate charge on the account",
                    "effect": "agent processed a refund",
                    "mechanism": "the refund reverses the duplicate charge",
                    "evidence_turns": [3, 4, 6],
                    "strength": "explicit",
                }
            ],
            "confidence": 0.85,
            "insufficient_evidence": False,
        }
    )

    result = run_benchmark(
        config,
        fixture_path=str(fixture),
        use_llm=True,
        output_dir=tmp_path / "results-llm",
        client=fake,
    )

    summary = result["summary"]
    generation = summary["generation"]
    assert generation["json_validity"] == 1.0
    assert generation["grounding_pass_rate"] >= 0.5
    assert generation["unsupported_claim_rate"] is not None

    tokens = summary["tokens"]
    assert tokens["input_tokens_mean"] == 100
    assert tokens["output_tokens_mean"] == 20
    assert tokens["output_tokens_per_second_mean"] is not None

    row = result["rows"][0]
    assert row["input_tokens"] == 100
    assert row["output_tokens"] == 20
    assert row["json_valid"] is True
