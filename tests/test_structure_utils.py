"""Tests for beer.utils.structure — calc_disorder_profile and helpers."""
from __future__ import annotations
import pytest
from beer.utils.structure import calc_disorder_profile


# --- calc_disorder_profile ---

def test_disorder_profile_length(fus_lc):
    profile = calc_disorder_profile(fus_lc)
    assert len(profile) == len(fus_lc)


def test_disorder_profile_range(fus_lc):
    profile = calc_disorder_profile(fus_lc)
    for val in profile:
        assert 0.0 <= val <= 1.0, f"Disorder score {val} out of [0, 1]"


def test_disorder_profile_returns_list(short_seq):
    profile = calc_disorder_profile(short_seq)
    assert isinstance(profile, list)


def test_disorder_profile_short_seq_no_crash():
    """Sequence shorter than window should not raise."""
    short = "MAEG"
    profile = calc_disorder_profile(short, window=9)
    assert len(profile) == len(short)


def test_disorder_profile_with_unavailable_embedder(fus_lc, unavailable_embedder):
    """When embedder is unavailable, falls back to classical method."""
    profile = calc_disorder_profile(fus_lc, embedder=unavailable_embedder)
    assert len(profile) == len(fus_lc)
    for val in profile:
        assert 0.0 <= val <= 1.0


def test_disorder_profile_esm2_fallback_with_mock_embedder(
    fus_lc, mock_embedder, disorder_head_weights
):
    """When embedder is available with head weights, ESM2 path is used."""
    profile = calc_disorder_profile(
        fus_lc, embedder=mock_embedder, head=disorder_head_weights
    )
    assert len(profile) == len(fus_lc)
    # With zero weights + zero embedding + zero intercept, sigmoid(0) = 0.5
    for val in profile:
        assert val == pytest.approx(0.5, abs=1e-5)


def test_disorder_profile_no_embedder(short_seq):
    """Without embedder, classical method runs cleanly."""
    profile = calc_disorder_profile(short_seq, embedder=None, head=None)
    assert len(profile) == len(short_seq)


