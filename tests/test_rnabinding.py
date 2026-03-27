"""Tests for beer.analysis.rnabinding."""
from __future__ import annotations
import pytest
from beer.analysis.rnabinding import calc_rbp_score, calc_rbp_profile


def test_calc_rbp_score_returns_dict(fus_lc):
    result = calc_rbp_score(fus_lc)
    assert isinstance(result, dict)


def test_calc_rbp_score_required_keys(fus_lc):
    result = calc_rbp_score(fus_lc)
    assert "mean_propensity" in result
    assert "fraction_rbp_residues" in result
    assert "motifs_found" in result


def test_calc_rbp_score_propensity_range(fus_lc):
    result = calc_rbp_score(fus_lc)
    assert -1.0 <= result["mean_propensity"] <= 1.0


def test_calc_rbp_score_short_seq(short_seq):
    result = calc_rbp_score(short_seq)
    assert isinstance(result, dict)
    assert 0.0 <= result["fraction_rbp_residues"] <= 1.0


def test_calc_rbp_score_charged_higher_propensity(charged_seq, hydrophobic_seq):
    """Basic (K/R-rich) sequences have higher mean RBP propensity than hydrophobic ones."""
    prop_charged = calc_rbp_score(charged_seq)["mean_propensity"]
    prop_hydro   = calc_rbp_score(hydrophobic_seq)["mean_propensity"]
    assert prop_charged >= prop_hydro


def test_calc_rbp_profile_length(fus_lc):
    window = 11  # default
    profile = calc_rbp_profile(fus_lc)
    assert isinstance(profile, list)
    # sliding window: returns len(seq) - window + 1 values
    assert len(profile) == len(fus_lc) - window + 1


def test_calc_rbp_profile_numeric(short_seq):
    profile = calc_rbp_profile(short_seq)
    assert all(isinstance(v, float) for v in profile)
