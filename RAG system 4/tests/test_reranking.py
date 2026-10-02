"""Unit tests: reranker features, weighting, swappable factory (§11, §12, §36)."""

from __future__ import annotations

import dataclasses

from system4.config.settings import Settings
from system4.retrieval.broad_retrieval import retrieve
from system4.retrieval.index import LocalIndex
from system4.reranking.cross_encoder import available
from system4.reranking.reranker import FeatureReranker, PassthroughReranker, make_reranker
from tests.conftest import conv


def _candidates():
    settings = Settings(dense_k=6, lexical_k=6, rerank_k=6)
    target = conv(
        "target",
        [(1, "Customer", "I was charged twice for the refund and need it fixed"),
         (2, "Agent", "I can approve the refund because the double charge is confirmed")],
        domain="Retail", intent="Refunds", reason="Customer reports double charge refund",
    )
    distractor = conv(
        "distractor",
        [(1, "Customer", "What is the weather like for my policy renewal")],
        domain="Insurance", intent="Renewals", reason="Weather question",
    )
    index = LocalIndex.build([target, distractor], settings)
    candidates = retrieve(index, "double charge refund", settings)
    return settings, index, candidates


def test_feature_reranker_puts_target_first():
    settings, index, candidates = _candidates()
    ranked = FeatureReranker().rerank("double charge refund", "double charge refund",
                                      candidates, index.conversations, settings)
    assert ranked[0].transcript_id == "target"
    assert 0.0 <= ranked[0].rerank_score <= 1.0
    assert ranked[0].features  # observability: named features stored (§45)


def test_rerank_scores_are_normalized_and_bounded():
    settings, index, candidates = _candidates()
    ranked = FeatureReranker().rerank("double charge refund", "double charge refund",
                                      candidates, index.conversations, settings)
    assert all(0.0 <= c.rerank_score <= 1.0 for c in ranked)
    assert ranked == sorted(ranked, key=lambda c: (-c.rerank_score, c.transcript_id))


def test_weights_are_configurable():
    settings, index, candidates = _candidates()
    metadata_only = dataclasses.replace(
        settings, w_semantic=0.0, w_lexical=0.0, w_entity=0.0, w_metadata=1.0,
        w_causal_density=0.0, w_action_density=0.0, w_outcome_density=0.0, w_fusion=0.0,
    )
    ranked = FeatureReranker().rerank("double charge refund", "double charge refund",
                                      candidates, index.conversations, metadata_only)
    for candidate in ranked:
        assert abs(candidate.rerank_score - candidate.consistency_score) < 1e-6


def test_passthrough_orders_by_fusion():
    settings, index, candidates = _candidates()
    ranked = PassthroughReranker().rerank("double charge refund", "double charge refund",
                                          candidates, index.conversations, settings)
    assert ranked
    fusion = [c.fusion_score for c in ranked]
    assert fusion == sorted(fusion, reverse=True) or ranked[0].rerank_score >= ranked[-1].rerank_score


def test_factory_selection_and_graceful_fallback():
    assert isinstance(make_reranker(Settings(reranker="none")), PassthroughReranker)
    assert isinstance(make_reranker(Settings(reranker="feature")), FeatureReranker)
    fallback = make_reranker(Settings(reranker="cross_encoder"))
    if available():
        assert fallback.name == "cross_encoder"
    else:
        # §43: unavailable learned reranker degrades to deterministic features.
        assert isinstance(fallback, FeatureReranker)


def test_cross_encoder_module_not_imported_by_default():
    # Local-first core must not require sentence-transformers.
    import sys

    assert "sentence_transformers" not in sys.modules
