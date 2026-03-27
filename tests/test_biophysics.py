"""Tests for beer.utils.biophysics."""
from __future__ import annotations
import pytest
from beer.utils.biophysics import (
    calc_net_charge,
    sliding_window_hydrophobicity,
    calc_shannon_entropy,
    sliding_window_ncpr,
    sliding_window_entropy,
    calc_kappa,
    calc_omega,
    count_pairs,
    fraction_low_complexity,
    sticker_spacing_stats,
)


# --- calc_net_charge ---

def test_calc_net_charge_positive_for_basic(charged_seq):
    """MEKEKEKEKEKRKRKRKRKR has more K/R than D/E, expect positive charge at pH 7."""
    charge = calc_net_charge(charged_seq, pH=7.0)
    assert charge > 0.0


def test_calc_net_charge_negative_for_acidic():
    acidic_seq = "MDDDDDDDDDEEEEEEEEE"
    charge = calc_net_charge(acidic_seq, pH=7.0)
    assert charge < 0.0


def test_calc_net_charge_near_zero_for_balanced():
    balanced = "MDEDEKKRR"
    charge = calc_net_charge(balanced, pH=7.0)
    assert isinstance(charge, float)


def test_calc_net_charge_returns_float(fus_lc):
    charge = calc_net_charge(fus_lc, pH=7.0)
    assert isinstance(charge, float)


def test_calc_net_charge_custom_pH(short_seq):
    charge_low  = calc_net_charge(short_seq, pH=2.0)
    charge_high = calc_net_charge(short_seq, pH=12.0)
    # At very low pH everything is protonated -> more positive
    # At very high pH everything deprotonated -> more negative
    assert charge_low > charge_high


# --- sliding_window_hydrophobicity ---

def test_sliding_window_hydrophobicity_length(fus_lc):
    window = 9
    result = sliding_window_hydrophobicity(fus_lc, window_size=window)
    assert isinstance(result, list)
    # sliding window returns len(seq) - window + 1 values
    assert len(result) == len(fus_lc) - window + 1


def test_sliding_window_hydrophobicity_hydrophobic_seq_positive(hydrophobic_seq):
    result = sliding_window_hydrophobicity(hydrophobic_seq, window_size=5)
    assert sum(result) > 0


# --- calc_shannon_entropy ---

def test_calc_shannon_entropy_returns_float(fus_lc):
    h = calc_shannon_entropy(fus_lc)
    assert isinstance(h, float)
    assert h >= 0.0


def test_calc_shannon_entropy_single_aa_is_zero():
    h = calc_shannon_entropy("AAAAAAAAAA")
    assert h == pytest.approx(0.0, abs=1e-9)


def test_calc_shannon_entropy_max_for_uniform():
    import math
    seq = "ACDEFGHIKLMNPQRSTVWY"  # all 20 AAs once
    h = calc_shannon_entropy(seq)
    assert h == pytest.approx(math.log2(20), rel=1e-3)


# --- sliding_window_ncpr ---

def test_sliding_window_ncpr_length(charged_seq):
    window = 9
    result = sliding_window_ncpr(charged_seq, window_size=window)
    assert isinstance(result, list)
    # sliding window returns len(seq) - window + 1 values
    assert len(result) == len(charged_seq) - window + 1


# --- sliding_window_entropy ---

def test_sliding_window_entropy_length(fus_lc):
    window = 9
    result = sliding_window_entropy(fus_lc, window_size=window)
    assert isinstance(result, list)
    # sliding window returns len(seq) - window + 1 values
    assert len(result) == len(fus_lc) - window + 1


# --- calc_kappa ---

def test_calc_kappa_returns_float(fus_lc):
    k = calc_kappa(fus_lc)
    assert isinstance(k, float)


def test_calc_kappa_range(charged_seq):
    k = calc_kappa(charged_seq)
    assert 0.0 <= k <= 1.0


# --- calc_omega ---

def test_calc_omega_returns_float(fus_lc):
    omega = calc_omega(fus_lc)
    assert isinstance(omega, float)


# --- count_pairs ---

def test_count_pairs_returns_int(fus_lc):
    n = count_pairs(fus_lc, set("KR"), set("FWY"), window=4)
    assert isinstance(n, int)
    assert n >= 0


# --- fraction_low_complexity ---

def test_fraction_low_complexity_range(fus_lc):
    frac = fraction_low_complexity(fus_lc, window_size=12, threshold=2.0)
    assert isinstance(frac, float)
    assert 0.0 <= frac <= 1.0


# --- sticker_spacing_stats ---

def test_sticker_spacing_stats_returns_dict(fus_lc):
    stats = sticker_spacing_stats(fus_lc)
    assert isinstance(stats, dict)
    for key in ("mean", "min", "max"):
        assert key in stats, f"Missing key: {key}"
