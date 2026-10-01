"""Dependency health checks (AGENTS.md section 22): each shown independently."""
from __future__ import annotations

from app.config import Settings
from app.domain.errors import RAGError


def check_health(settings: Settings) -> dict:
    checks: list[dict] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})

    # 1. source exists
    src = settings.source_path
    add("source_exists", src.exists(), str(src))
    # 2. source readable (+ record count)
    try:
        from app.ingestion.loader import load_raw_records
        recs, sha, _ = load_raw_records(src)
        add("source_readable", True, f"{len(recs)} records, sha {sha[:12]}")
    except RAGError as e:
        add("source_readable", False, e.message)
        recs, sha = [], ""
    except Exception as e:
        add("source_readable", False, f"{type(e).__name__}: {e}")
        recs, sha = [], ""
    # 3. index exists + matches source
    try:
        from app.indexing.builder import verify_manifest
        m = verify_manifest(settings)
        add("index_valid", True,
            f"{m.num_chunks} chunks / {m.num_transcripts} transcripts, {m.index_type}/{m.embedding_model}")
    except RAGError as e:
        add("index_valid", False, e.message)
    except Exception as e:
        add("index_valid", False, f"{type(e).__name__}: {e}")
    # Generator (Ollama): reachability + model presence. No conversation data.
    try:
        from app.generation.ollama_client import OllamaClient
        try:
            h = OllamaClient(settings).health()
        except Exception as e:
            add("ollama", False, f"{type(e).__name__}: {e}")
        else:
            ok = bool(h.get("reachable")) and bool(h.get("model_present"))
            add("ollama", ok, f"{h.get('base_url')} model={h.get('model')} "
                              f"present={h.get('model_present')} detail={h.get('detail', '')}")
    except Exception as e:
        add("ollama", False, f"{type(e).__name__}: {e}")
    # 6. laya trivial decision
    try:
        from app.system1.gates import build_engine
        from app.system1.schemas import QuestionSpec
        eng = build_engine(settings)
        if eng is None:
            add("laya", False, "disabled by config")
        else:
            eng.ping()
            r = eng.decide({"query_text": "health ping", "num_chunks": 2, "mean_score": 0.5},
                           {"evidence_sufficient": QuestionSpec(kind="choice",
                            options=["sufficient", "insufficient"])})
            add("laya", True, f"engine={r.engine} decision={r.get('evidence_sufficient')}")
    except RAGError as e:
        add("laya", False, e.message)
    except Exception as e:
        add("laya", False, f"{type(e).__name__}: {e}")
    # 7. embedding model available locally
    try:
        from app.indexing.embeddings import get_embedding_provider
        p = get_embedding_provider(settings.embedding_backend, settings.embedding_model,
                                   settings.embedding_dim, settings.embedding_normalize)
        v = p.embed(["health check"])
        add("embeddings", True, f"{p.name} dim={p.dim} shape={tuple(v.shape)}")
    except RAGError as e:
        add("embeddings", False, e.message)
    except Exception as e:
        add("embeddings", False, f"{type(e).__name__}: {e}")

    overall = all(c["ok"] for c in checks)
    return {"overall_ok": overall, "checks": checks,
            "config": {"source": str(src), "embedding": f"{settings.embedding_backend}/{settings.embedding_model}",
                       "generation": settings.generation_config()}}
