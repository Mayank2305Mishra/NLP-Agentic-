from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Callable

import requests

from .config import Config

log = logging.getLogger("system3.health")

#: Health checks must stay local by default (§34): the only network probe is
#: the configured LLM endpoint itself.
PROBE_TIMEOUT_S = 2.0


def _check_processed(config: Config) -> dict[str, Any]:
    """Informational: build-time artifacts (not required to serve requests)."""
    conv = config.dataset.conversations_jsonl
    turns = config.dataset.turns_jsonl
    ok = conv.exists() and turns.exists()
    return {
        "ok": ok,
        "informational": True,
        "conversations_jsonl": str(conv) if conv.exists() else None,
        "turns_jsonl": str(turns) if turns.exists() else None,
        "error": None if ok else "processed data missing — run `system3 build-data`",
    }


def _fts_in_sync(store: Any) -> bool:
    """FTS5 content-vs-index integrity check (rank=1 variant).

    Row-count comparison cannot detect desync for external-content tables —
    rowid scans are answered from the content table itself.
    """
    for table in ("conv_fts", "turns_fts"):
        try:
            store.conn.execute(
                f"INSERT INTO {table}({table}, rank) VALUES('integrity-check', 1)"
            )
        except sqlite3.DatabaseError:
            return False
    return True


def _check_index(config: Config, deep: bool = True) -> dict[str, Any]:
    db_path = config.dataset.db_path
    if not db_path.exists():
        return {
            "ok": False,
            "path": str(db_path),
            "error": "index missing — run `system3 build-index`",
        }
    from .store import LexicalStore

    store = LexicalStore(db_path)
    try:
        counts = store.counts()
        fts_ok = _fts_in_sync(store) if deep else True
    except Exception as exc:  # pragma: no cover - corrupt db surfaces here
        return {"ok": False, "path": str(db_path), "error": f"index unreadable: {exc}"}
    finally:
        store.close()
    populated = counts["conversations"] > 0
    ok = fts_ok and populated
    error = None
    if not populated:
        error = "index has no conversations — rebuild with `system3 build-index --reset`"
    elif not fts_ok:
        error = "FTS tables out of sync with base tables — rebuild with `system3 build-index --reset`"
    return {
        "ok": ok,
        "path": str(db_path),
        "size_bytes": db_path.stat().st_size,
        "deep_check": deep,
        **counts,
        "error": error,
    }


def probe_llm(config: Config, timeout_s: float = PROBE_TIMEOUT_S) -> dict[str, Any]:
    """Reachability probe for the configured provider. Never uploads data."""
    gen = config.generation
    provider = gen.provider
    if provider == "ollama":
        url = gen.ollama_url.rstrip("/") + "/api/tags"
        try:
            resp = requests.get(url, timeout=timeout_s)
            resp.raise_for_status()
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            present = any(
                m == gen.model or m.rsplit(":", 1)[0] == gen.model.rsplit(":", 1)[0]
                for m in models
            )
            return {
                "ok": present,
                "provider": provider,
                "model": gen.model,
                "url": gen.ollama_url,
                "reachable": True,
                "model_present": present,
                "error": None
                if present
                else f"model {gen.model!r} not pulled on this Ollama instance",
            }
        except (requests.RequestException, ValueError) as exc:
            return {
                "ok": False,
                "provider": provider,
                "model": gen.model,
                "url": gen.ollama_url,
                "reachable": False,
                "error": f"ollama unreachable: {type(exc).__name__}: {exc}",
            }
    if provider == "bonsai":
        url = gen.bonsai_url or os.environ.get("BONSAI_URL", "")
        if not url:
            return {
                "ok": False,
                "provider": provider,
                "error": "bonsai endpoint not configured (generation.bonsai_url or BONSAI_URL)",
            }
        try:
            # Any HTTP response (even 404) proves the endpoint is reachable.
            requests.get(url, timeout=timeout_s)
            return {"ok": True, "provider": provider, "url": url, "reachable": True}
        except requests.RequestException as exc:
            return {
                "ok": False,
                "provider": provider,
                "url": url,
                "reachable": False,
                "error": f"bonsai unreachable: {type(exc).__name__}: {exc}",
            }
    if provider == "groq":
        # External service: check credentials only, keep the probe local.
        has_key = bool(os.environ.get("GROQ_API_KEY"))
        return {
            "ok": has_key,
            "provider": provider,
            "model": gen.model,
            "error": None if has_key else "GROQ_API_KEY not set",
        }
    return {"ok": False, "provider": provider, "error": f"unknown provider {provider!r}"}


def _check_logs(config: Config) -> dict[str, Any]:
    log_path = config.logging.log_path
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        probe = log_path.parent / ".healthcheck"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return {"ok": True, "path": str(log_path)}
    except OSError as exc:
        return {"ok": False, "path": str(log_path), "error": f"log dir not writable: {exc}"}


def run_health(
    config: Config,
    llm_probe: Callable[[Config], dict[str, Any]] | None = None,
    deep: bool = True,
) -> dict[str, Any]:
    """Full service health report. ``status`` is ``ok`` only when every
    gating check (index, llm, logs) passes; processed data is informational.

    ``deep=False`` skips the FTS integrity scan (fast mode, ~ms).
    """
    probe = llm_probe or probe_llm
    checks = {
        "processed_data": _check_processed(config),
        "index": _check_index(config, deep=deep),
        "llm": probe(config),
        "logs": _check_logs(config),
    }
    gating = [checks["index"], checks["llm"], checks["logs"]]
    status = "ok" if all(c.get("ok") for c in gating) else "degraded"
    return {
        "status": status,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": config.config_hash,
        "provider": config.generation.provider,
        "model": config.generation.model,
        "checks": checks,
    }
