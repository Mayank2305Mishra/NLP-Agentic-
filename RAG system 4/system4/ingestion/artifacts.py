"""Artifact persistence with version validation (AGENTS_SYSTEM_4.md §7, §39)."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any


def source_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(data).hexdigest()


def save(path: str | Path, artifact: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        pickle.dump(artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def load_artifact(path: str | Path, expected_version: str | None = None) -> dict[str, Any]:
    """Load a persisted artifact and refuse incompatible versions (§39)."""
    artifact = load(path)
    if not isinstance(artifact, dict) or "index" not in artifact:
        raise ValueError(f"{path} is not a System 4 index artifact")
    version = artifact.get("version")
    if expected_version is not None and version != expected_version:
        raise ValueError(
            f"artifact version {version!r} != expected {expected_version!r}; "
            f"rebuild with: python scripts/build_index.py --source ... --output {path}"
        )
    return artifact
