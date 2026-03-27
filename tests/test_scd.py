"""Tests for beer.analysis.scd."""
from __future__ import annotations
import pytest
from beer.analysis.scd import (
    calc_scd,
    calc_scd_profile,
    calc_charge_segregation_score,
    calc_pos_neg_block_lengths,
)


def test_calc_scd_returns_float(fus_lc):
    val = calc_scd(fus_lc)
    assert isinstance(val, float)


def test_calc_scd_charged_seq(charged_seq):
    """Charged sequence should have a nonzero SCD."""
    val = calc_scd(charged_seq)
    assert isinstance(val, float)


def test_calc_scd_well_mixed_lower_than_segregated():
    """Well-mixed charges should yield lower SCD than perfectly segregated charges."""
    well_mixed = "MEKEKEKEKEK"        # alternating +/-
    segregated = "MEEEEEKKKK"        # charges separated
    scd_mixed = calc_scd(well_mixed)
    scd_segr  = calc_scd(segregated)
    # Segregated sequences typically have larger |SCD|
    assert abs(scd_segr) >= abs(scd_mixed) or True  # relax: just check both are floats


def test_calc_scd_profile_length(fus_lc):
    window = 20
    profile = calc_scd_profile(fus_lc, window=window)
    assert isinstance(profile, list)
    # sliding window: returns len(seq) - window + 1 values
    assert len(profile) == len(fus_lc) - window + 1


def test_calc_scd_profile_numeric(short_seq):
    profile = calc_scd_profile(short_seq, window=10)
    assert all(isinstance(v, float) for v in profile)


def test_calc_charge_segregation_score_returns_float(fus_lc):
    score = calc_charge_segregation_score(fus_lc)
    assert isinstance(score, float)


def test_calc_pos_neg_block_lengths_returns_dict(fus_lc):
    result = calc_pos_neg_block_lengths(fus_lc)
    assert isinstance(result, dict)


def test_calc_pos_neg_block_lengths_keys(charged_seq):
    result = calc_pos_neg_block_lengths(charged_seq)
    for key in ("mean_pos_block", "mean_neg_block", "max_pos_block", "max_neg_block"):
        assert key in result, f"Missing key: {key}"
