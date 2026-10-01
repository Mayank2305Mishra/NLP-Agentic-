"""Atomic index builder (AGENTS.md sections 19-20).

Order: validate -> normalize -> chunk -> embed (batches) -> write vectors
atomically -> write metadata -> write manifest LAST. A partial build is never
marked valid (tmp dir + atomic rename).
"""
from __future__ import annotations

import json
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from app.config import Settings
from app.domain.errors import IndexStaleError
from app.domain.models import Chunk, IndexManifest
from app.indexing.chunker import chunk_transcript, chunking_config_hash
from app.indexing.embeddings import EmbeddingProvider, get_embedding_provider
from app.indexing.vector_store import VectorStore, create_store, load_store
from app.ingestion.loader import load_raw_records
from app.ingestion.normalizer import normalize_record
from app.ingestion.validator import validate_records
from app.logging import get_logger

log = get_logger("indexing")

MANIFEST_NAME = "manifest.json"
CHUNKS_NAME = "chunks.jsonl"
VECTORS_SUBDIR = "vectors"


def build_index(settings: Settings, *, on_duplicate: str = "fail",
                progress: bool = False) -> IndexManifest:
    from app.logging import log_event
    import logging

    t0 = time.perf_counter()
    records, source_sha, resolved = load_raw_records(settings.source_path)
    log_event(log, logging.INFO, "index.source_loaded",
              extra={"path": str(resolved), "records": len(records)})
    usable, summary = validate_records(records, on_duplicate=on_duplicate)
    chunk_cfg = settings.chunking_config()
    provider = get_embedding_provider(
        settings.embedding_backend, settings.embedding_model,
        settings.embedding_dim, settings.embedding_normalize,
    )
    all_chunks: list[Chunk] = []
    for rec in usable:
        nt = normalize_record(rec, source_hash=source_sha[:16])
        if not nt.turns:
            continue
        all_chunks.extend(chunk_transcript(nt, **{
            "target_tokens": chunk_cfg["target_tokens"],
            "max_tokens": chunk_cfg["max_tokens"],
            "overlap_turns": chunk_cfg["overlap_turns"],
            "include_metadata": chunk_cfg["include_metadata"],
        }))
    if progress:
        print(f"chunked {len(all_chunks)} chunks from {len(usable)} transcripts")
    # Embed in batches.
    batch = max(1, settings.max_embedding_batch)
    vectors_list = []
    for i in range(0, len(all_chunks), batch):
        sl = all_chunks[i:i + batch]
        vectors_list.append(provider.embed([c.normalized_text for c in sl]))
        if progress and (i // batch) % 20 == 0:
            print(f"  embedded {min(i + batch, len(all_chunks))}/{len(all_chunks)}")
    import numpy as np
    vectors = np.vstack(vectors_list) if vectors_list else np.zeros((0, provider.dim), dtype=np.float32)
    store = create_store(settings.vector_store_type, provider.dim)
    store.add(vectors, [c.chunk_id for c in all_chunks])
    # Atomic write: tmp dir then rename.
    index_dir = Path(settings.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="index_build_", dir=str(index_dir)))
    try:
        vec_dir = tmp / VECTORS_SUBDIR
        store.save(vec_dir)
        with open(tmp / CHUNKS_NAME, "w", encoding="utf-8") as f:
            for c in all_chunks:
                f.write(c.model_dump_json() + "\n")
        manifest = IndexManifest(
            source_file=resolved.name,
            source_sha256=source_sha,
            embedding_model=provider.name,
            embedding_dimension=provider.dim,
            embedding_normalize=provider.normalize,
            embedding_backend=settings.embedding_backend,
            chunking_config_hash=chunking_config_hash(chunk_cfg),
            chunking_config=chunk_cfg,
            index_type=store.index_type,
            created_at=datetime.now(timezone.utc).isoformat(),
            num_transcripts=len(usable),
            num_chunks=len(all_chunks),
        )
        build_seconds = round(time.perf_counter() - t0, 2)
        (tmp / MANIFEST_NAME).write_text(
            json.dumps({**manifest.model_dump(), "build_seconds": build_seconds,
                        "index_bytes": _dir_size(vec_dir)}, indent=2),
            encoding="utf-8",
        )
        final_tmp = index_dir / ".final"
        if final_tmp.exists():
            shutil.rmtree(final_tmp)
        os_replace_dir(tmp, index_dir)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
    log_event(log, logging.INFO, "index.built",
              extra={"chunks": len(all_chunks), "transcripts": len(usable)})
    return load_manifest(settings)


def os_replace_dir(tmp: Path, index_dir: Path) -> None:
    """Atomically replace index_dir contents with tmp (manifest written last already)."""

    staging = index_dir.parent / (index_dir.name + ".new")
    if staging.exists():
        shutil.rmtree(staging)
    tmp.rename(staging)
    old = index_dir.parent / (index_dir.name + ".old")
    if old.exists():
        shutil.rmtree(old)
    if index_dir.exists():
        index_dir.rename(old)
    staging.rename(index_dir)
    if old.exists():
        shutil.rmtree(old)


def _dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def manifest_path(settings: Settings) -> Path:
    return Path(settings.index_dir) / MANIFEST_NAME


def load_manifest(settings: Settings) -> IndexManifest:
    mp = manifest_path(settings)
    if not mp.exists():
        from app.domain.errors import IndexNotFoundError
        raise IndexNotFoundError("No index found. Run: python -m scripts.build_index",
                                 context={"index_dir": str(settings.index_dir)})
    return IndexManifest.model_validate(json.loads(mp.read_text(encoding="utf-8")))


def verify_manifest(settings: Settings, *, allow_stale: bool = False) -> IndexManifest:
    """Refuse a stale index by default (AGENTS.md section 19)."""
    manifest = load_manifest(settings)
    _, source_sha, _ = load_raw_records(settings.source_path)
    if manifest.source_sha256 != source_sha and not (allow_stale or settings.allow_stale_index):
        raise IndexStaleError(
            "Index is stale: source hash changed. Rebuild with: python -m scripts.build_index",
            context={"manifest_sha": manifest.source_sha256[:12], "source_sha": source_sha[:12]},
        )
    if manifest.embedding_dimension <= 0:
        from app.domain.errors import VectorStoreError
        raise VectorStoreError("Manifest has invalid embedding dimension")
    return manifest


def load_chunks(settings: Settings) -> dict[str, Chunk]:
    path = Path(settings.index_dir) / CHUNKS_NAME
    if not path.exists():
        from app.domain.errors import IndexNotFoundError
        raise IndexNotFoundError("Chunk metadata missing; rebuild the index")
    out: dict[str, Chunk] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                c = Chunk.model_validate_json(line)
                out[c.chunk_id] = c
    return out


def load_vector_store(settings: Settings, manifest: IndexManifest) -> VectorStore:
    return load_store(manifest.index_type, Path(settings.index_dir) / VECTORS_SUBDIR,
                      manifest.embedding_dimension)


def get_query_provider(settings: Settings, manifest: IndexManifest) -> EmbeddingProvider:
    """Query-time embedding provider must match the manifest dimension."""
    provider = get_embedding_provider(
        settings.embedding_backend or manifest.embedding_backend,
        settings.embedding_model, settings.embedding_dim, settings.embedding_normalize,
    )
    if provider.dim != manifest.embedding_dimension:
        from app.domain.errors import EmbeddingModelError
        raise EmbeddingModelError(
            f"Embedding dim mismatch: config dim={provider.dim} vs index dim={manifest.embedding_dimension}. "
            "Fix EMBEDDING_DIM / EMBEDDING_BACKEND or rebuild the index."
        )
    return provider
