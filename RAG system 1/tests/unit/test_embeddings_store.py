"""Unit: embeddings shape/normalization + vector-store ordering + dim guard."""
from __future__ import annotations

import numpy as np
import pytest

from app.domain.errors import EmbeddingModelError, VectorStoreError
from app.indexing.embeddings import HashingEmbedding, get_embedding_provider
from app.indexing.vector_store import NumpyVectorStore, create_store


def test_hashing_embedding_shape_and_norm():
    p = HashingEmbedding(dim=128)
    vecs = p.embed(["hello world", "another text"])
    assert vecs.shape == (2, 128)
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), [1.0, 1.0], atol=1e-5)


def test_unknown_backend_rejected():
    with pytest.raises(EmbeddingModelError):
        get_embedding_provider("openai", "x", 8, True)


def test_numpy_store_orders_by_similarity():
    s = NumpyVectorStore(dim=3)
    s.add(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32), ["a", "b"])
    hits = s.search(np.array([1, 0, 0], dtype=np.float32), 2)
    assert hits[0][0] == "a" and hits[0][1] > hits[1][1]


def test_dim_mismatch_refuses():
    s = NumpyVectorStore(dim=3)
    with pytest.raises(VectorStoreError):
        s.add(np.zeros((1, 4), dtype=np.float32), ["x"])


def test_faiss_store_roundtrip(tmp_path):
    pytest.importorskip("faiss")
    store = create_store("faiss", 8)
    assert store.index_type == "faiss"
    vecs = np.eye(8, dtype=np.float32)[:4]
    store.add(vecs, [f"c{i}" for i in range(4)])
    hits = store.search(vecs[2], 2)
    assert hits[0][0] == "c2"
    store.save(tmp_path)
    from app.indexing.vector_store import FaissVectorStore
    loaded = FaissVectorStore.load(tmp_path, 8)
    assert len(loaded) == 4
    with pytest.raises(VectorStoreError):
        FaissVectorStore.load(tmp_path, 16)  # dim mismatch refuses
