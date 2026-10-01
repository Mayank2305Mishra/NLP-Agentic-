"""Embedding providers. Protocol + two local backends.

- HashingEmbedding (default): sklearn HashingVectorizer, deterministic, no
  download, L2-normalized. Suitable for offline/laptop benchmark baseline.
- SentenceTransformerEmbedding (optional): used only when
  EMBEDDING_BACKEND=sentence-transformers and the package is installed.

The active model name/dim/normalization is recorded in the index manifest
and every benchmark result.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from app.domain.errors import EmbeddingModelError


class EmbeddingProvider(Protocol):
    name: str
    dim: int
    normalize: bool

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedding:
    """Deterministic char-word hashing embedding (no external download)."""

    def __init__(self, dim: int = 768, normalize: bool = True, alternate_sign: bool = False):
        from sklearn.feature_extraction.text import HashingVectorizer

        self._vec = HashingVectorizer(
            n_features=dim,
            alternate_sign=alternate_sign,  # deterministic non-negative
            norm=None,
            ngram_range=(1, 2),
            lowercase=True,
        )
        self.dim = dim
        self.normalize = normalize
        self.name = f"hashing-{dim}"

    def embed(self, texts: list[str]) -> np.ndarray:
        try:
            mat = self._vec.transform(texts or [""]).toarray().astype(np.float32)
        except Exception as e:
            raise EmbeddingModelError(f"Hashing embedding failed: {e}") from e
        if self.normalize:
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            mat = mat / norms
        return mat


class SentenceTransformerEmbedding:
    def __init__(self, model_name: str, normalize: bool = True):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise EmbeddingModelError(
                "sentence-transformers is not installed. "
                "Install with: pip install 'rag-system-1[embeddings-st]' "
                "or use EMBEDDING_BACKEND=hashing."
            ) from e
        try:
            self._model = SentenceTransformer(model_name)
        except Exception as e:
            raise EmbeddingModelError(f"Could not load embedding model '{model_name}': {e}") from e
        self.name = model_name
        self.normalize = normalize
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts or [""], normalize_embeddings=self.normalize,
                                  show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


def get_embedding_provider(backend: str = "hashing", model: str = "hashing-768",
                            dim: int = 768, normalize: bool = True) -> EmbeddingProvider:
    backend = (backend or "hashing").lower()
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedding(model, normalize=normalize)
    if backend == "hashing":
        return HashingEmbedding(dim=dim, normalize=normalize)
    raise EmbeddingModelError(
        f"Unknown EMBEDDING_BACKEND '{backend}' (expected hashing|sentence-transformers)"
    )


class QueryEmbeddingCache:
    """Small in-memory cache for repeated query embeddings (same config only)."""

    def __init__(self, provider: EmbeddingProvider, maxsize: int = 256):
        self.provider = provider
        self._cache: dict[str, np.ndarray] = {}
        self._maxsize = maxsize
        self.hits = 0
        self.misses = 0

    def embed_query(self, text: str) -> tuple[np.ndarray, bool]:
        if text in self._cache:
            self.hits += 1
            return self._cache[text], True
        vec = self.provider.embed([text])[0]
        self.misses += 1
        if len(self._cache) >= self._maxsize:
            self._cache.pop(next(iter(self._cache)))
        self._cache[text] = vec
        return vec, False
