"""Tests for beer.embeddings — FallbackEmbedder and get_embedder()."""
from __future__ import annotations
import pytest
from beer.embeddings import FallbackEmbedder, get_embedder, ESM2_AVAILABLE, SequenceEmbedder


def test_fallback_embedder_is_not_available():
    emb = FallbackEmbedder()
    assert emb.is_available() is False


def test_fallback_embedder_embed_returns_none(fus_lc):
    emb = FallbackEmbedder()
    result = emb.embed(fus_lc)
    assert result is None


def test_fallback_embedder_is_sequence_embedder():
    emb = FallbackEmbedder()
    assert isinstance(emb, SequenceEmbedder)


def test_get_embedder_returns_sequence_embedder():
    emb = get_embedder()
    assert isinstance(emb, SequenceEmbedder)


def test_get_embedder_without_esm2_returns_fallback():
    """When ESM2 is not installed, get_embedder() must return a FallbackEmbedder."""
    if ESM2_AVAILABLE:
        pytest.skip("ESM2 is installed — FallbackEmbedder path not active")
    emb = get_embedder()
    assert isinstance(emb, FallbackEmbedder)
    assert emb.is_available() is False


def test_get_embedder_availability_matches_esm2_flag():
    emb = get_embedder()
    assert emb.is_available() == ESM2_AVAILABLE
