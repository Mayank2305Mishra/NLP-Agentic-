from __future__ import annotations

from system3.health import run_health


def _ok_probe(config):
    return {"ok": True, "provider": "fake", "reachable": True}


def _down_probe(config):
    return {"ok": False, "provider": "fake", "reachable": False, "error": "down"}


def test_health_ok_with_index_and_probe(config, store):
    report = run_health(config, llm_probe=_ok_probe)
    assert report["status"] == "ok"
    assert report["config_hash"]
    checks = report["checks"]
    assert checks["index"]["ok"] is True
    assert checks["index"]["conversations"] == 6
    assert checks["index"]["conv_fts"] == 6
    assert checks["llm"]["ok"] is True
    assert checks["logs"]["ok"] is True
    # processed data is informational: absent build artifacts must not degrade
    assert checks["processed_data"]["informational"] is True


def test_health_degraded_when_llm_unreachable(config, store):
    report = run_health(config, llm_probe=_down_probe)
    assert report["status"] == "degraded"
    assert report["checks"]["llm"]["ok"] is False


def test_health_degraded_when_index_missing(config, tmp_path):
    config.dataset.index_dir = str(tmp_path / "no-index")
    report = run_health(config, llm_probe=_ok_probe)
    assert report["status"] == "degraded"
    assert report["checks"]["index"]["ok"] is False
    assert "build-index" in report["checks"]["index"]["error"]


def test_health_detects_unsynced_fts(config, store):
    # Base row inserted without a matching FTS entry -> desynced index.
    store.conn.execute(
        "INSERT INTO conversations (transcript_id, conversation_text) VALUES (?, ?)",
        ("conv-unsynced", "orphaned row"),
    )
    store.conn.commit()
    report = run_health(config, llm_probe=_ok_probe)
    assert report["checks"]["index"]["ok"] is False
    assert "out of sync" in report["checks"]["index"]["error"]
    assert report["status"] == "degraded"
