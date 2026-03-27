"""Tests for beer.analysis.ptm."""
from __future__ import annotations
import pytest
from beer.analysis.ptm import scan_ptm_sites, summarize_ptm_sites


def test_scan_ptm_sites_returns_list(fus_lc):
    sites = scan_ptm_sites(fus_lc)
    assert isinstance(sites, list)


def test_scan_ptm_sites_dict_keys(fus_lc):
    """Each PTM site dict must contain required keys."""
    required = {"type", "position_1based", "context", "description", "confidence"}
    sites = scan_ptm_sites(fus_lc)
    for site in sites:
        assert required <= set(site.keys()), (
            f"PTM site missing keys: {required - set(site.keys())}"
        )


def test_scan_ptm_sites_position_1based(fus_lc):
    """Positions must be >= 1 and <= len(seq)."""
    sites = scan_ptm_sites(fus_lc)
    for site in sites:
        assert 1 <= site["position_1based"] <= len(fus_lc)


def test_scan_ptm_sites_confidence_valid(fus_lc):
    valid_confidences = {"high", "medium", "low"}
    sites = scan_ptm_sites(fus_lc)
    for site in sites:
        assert site["confidence"] in valid_confidences


def test_scan_ptm_sites_short_seq(short_seq):
    """Function must work on short sequences without raising."""
    sites = scan_ptm_sites(short_seq)
    assert isinstance(sites, list)


def test_summarize_ptm_sites_returns_dict(fus_lc):
    sites = scan_ptm_sites(fus_lc)
    summary = summarize_ptm_sites(sites)
    assert isinstance(summary, dict)


def test_summarize_ptm_sites_empty():
    """Summarising an empty list must return an empty or valid dict."""
    summary = summarize_ptm_sites([])
    assert isinstance(summary, dict)
