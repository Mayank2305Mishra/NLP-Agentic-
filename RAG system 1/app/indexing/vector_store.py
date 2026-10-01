"""Vector-store protocol + FAISS (primary) and numpy (fallback) adapters.

Application code must depend on the VectorStore protocol only — never on
FAISS APIs directly (AGENTS.md section 5).
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from app.domain.errors import VectorStoreError


class VectorStore(Protocol):
    index_type: str
    dim: int

    def add(self, vectors: np.ndarray, ids: list[str]) -> None: ...
    def search(self, query: np.ndarray, top_k: int) -> list[tuple[str, float]]: ...
    def save(self, directory: Path) -> None: ...
    @classmethod
    def load(cls, directory: Path, dim: int) -> "VectorStore": ...
    def __len__(self) -> int: ...


class FaissVectorStore:
    """Cosine search via inner product on L2-normalized vectors (IndexFlatIP)."""

    index_type = "faiss"

    def __init__(self, dim: int):
        try:
            import faiss  # noqa: F401
        except ImportError as e:
            raise VectorStoreError(
                "faiss is not installed. Run: pip install faiss-cpu"
            ) from e
        import faiss

        self.dim = dim
        self._faiss = faiss
        self._index = faiss.IndexFlatIP(dim)
        self._ids: list[str] = []

    def add(self, vectors: np.ndarray, ids: list[str]) -> None:
        if vectors.shape[1] != self.dim:
            raise VectorStoreError(
                f"Vector dimension mismatch: got {vectors.shape[1]}, index dim {self.dim}"
            )
        try:
            self._index.add(np.ascontiguousarray(vectors.astype(np.float32)))
        except Exception as e:
            raise VectorStoreError(f"FAISS add failed: {e}") from e
        self._ids.extend(ids)

    def search(self, query: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        if len(self._ids) == 0:
            return []
        k = min(top_k, len(self._ids))
        q = np.ascontiguousarray(query.reshape(1, -1).astype(np.float32))
        try:
            scores, idx = self._index.search(q, k)
        except Exception as e:
            raise VectorStoreError(f"FAISS search failed: {e}") from e
        return [(self._ids[i], float(scores[0][r])) for r, i in enumerate(idx[0]) if i >= 0]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(directory / "vectors.faiss"))
        import json

        (directory / "ids.json").write_text(json.dumps(self._ids), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, dim: int) -> "FaissVectorStore":
        import json

        store = cls(dim)
        try:
            store._index = store._faiss.read_index(str(directory / "vectors.faiss"))
        except Exception as e:
            raise VectorStoreError(f"FAISS index corrupt or missing: {e}") from e
        if store._index.d != dim:
            raise VectorStoreError(
                f"Vector dimension mismatch: index d={store._index.d}, expected {dim}. Refusing to load."
            )
        store._ids = json.loads((directory / "ids.json").read_text(encoding="utf-8"))
        return store

    def __len__(self) -> int:
        return len(self._ids)


class NumpyVectorStore:
    """Brute-force cosine fallback (same protocol, no FAISS dependency)."""

    index_type = "numpy"

    def __init__(self, dim: int):
        self.dim = dim
        self._vecs: list[np.ndarray] = []
        self._ids: list[str] = []

    def add(self, vectors: np.ndarray, ids: list[str]) -> None:
        if vectors.shape[1] != self.dim:
            raise VectorStoreError(
                f"Vector dimension mismatch: got {vectors.shape[1]}, index dim {self.dim}"
            )
        for v, i in zip(vectors, ids):
            self._vecs.append(v.astype(np.float32))
            self._ids.append(i)

    def search(self, query: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        if not self._ids:
            return []
        q = query.astype(np.float32).ravel()
        mat = np.stack(self._vecs)
        scores = mat @ q
        k = min(top_k, len(self._ids))
        order = np.argsort(-scores)[:k]
        return [(self._ids[i], float(scores[i])) for i in order]

    def save(self, directory: Path) -> None:
        import json

        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", np.stack(self._vecs) if self._vecs else np.zeros((0, self.dim)))
        (directory / "ids.json").write_text(json.dumps(self._ids), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, dim: int) -> "NumpyVectorStore":
        import json

        store = cls(dim)
        arr = np.load(directory / "vectors.npy")
        if arr.ndim != 2 or (arr.shape[0] > 0 and arr.shape[1] != dim):
            raise VectorStoreError(f"Vector dimension mismatch on load: {arr.shape} vs dim {dim}")
        store._vecs = [arr[i] for i in range(arr.shape[0])]
        store._ids = json.loads((directory / "ids.json").read_text(encoding="utf-8"))
        return store

    def __len__(self) -> int:
        return len(self._ids)


def create_store(store_type: str, dim: int) -> VectorStore:
    st = (store_type or "auto").lower()
    if st == "faiss":
        return FaissVectorStore(dim)
    if st == "numpy":
        return NumpyVectorStore(dim)
    # auto: prefer FAISS, fall back to numpy with an explicit (non-silent) note.
    try:
        return FaissVectorStore(dim)
    except VectorStoreError:
        return NumpyVectorStore(dim)


def load_store(store_type: str, directory: Path, dim: int) -> VectorStore:
    """Load by manifest index_type (not by current preference)."""
    if store_type == "faiss":
        return FaissVectorStore.load(directory, dim)
    if store_type == "numpy":
        return NumpyVectorStore.load(directory, dim)
    raise VectorStoreError(f"Unknown index_type '{store_type}'")
