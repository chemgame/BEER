"""Tests for beer.analysis.aggregation."""
from __future__ import annotations
import pytest
from beer.analysis.aggregation import calc_aggregation_profile, calc_solubility_stats


def test_calc_aggregation_profile_length(fus_lc):
    profile = calc_aggregation_profile(fus_lc)
    assert isinstance(profile, list)
    assert len(profile) == len(fus_lc)


def test_calc_aggregation_profile_numeric(fus_lc):
    profile = calc_aggregation_profile(fus_lc)
    assert all(isinstance(v, float) for v in profile)


def test_calc_aggregation_profile_short_seq(short_seq):
    """Profile must still return a list for short sequences."""
    profile = calc_aggregation_profile(short_seq)
    assert isinstance(profile, list)
    assert len(profile) == len(short_seq)


def test_calc_aggregation_profile_hydrophobic_seq(hydrophobic_seq):
    """Hydrophobic sequences should yield higher aggregation scores."""
    profile_hydro = calc_aggregation_profile(hydrophobic_seq)
    profile_charged = calc_aggregation_profile("MEKEKEKEKEKRKRKRKRKR")
    assert sum(profile_hydro) > sum(profile_charged)


def test_calc_solubility_stats_keys(fus_lc):
    stats = calc_solubility_stats(fus_lc)
    assert isinstance(stats, dict)
    required = {"mean_camsolmt", "fraction_insoluble", "fraction_soluble",
                "mean_aggregation_propensity", "n_hotspots", "aggregation_hotspots",
                "camsolmt_profile", "camsolmt_smoothed"}
    for key in required:
        assert key in stats, f"Missing key: {key}"


def test_calc_solubility_stats_numeric_scalar_values(short_seq):
    stats = calc_solubility_stats(short_seq)
    for key in ("mean_camsolmt", "fraction_insoluble", "fraction_soluble",
                "mean_aggregation_propensity", "n_hotspots"):
        assert isinstance(stats[key], (int, float)), (
            f"stats[{key!r}] = {stats[key]!r} is not numeric"
        )


def test_calc_solubility_stats_fractions_in_range(fus_lc):
    stats = calc_solubility_stats(fus_lc)
    assert 0.0 <= stats["fraction_insoluble"] <= 1.0
    assert 0.0 <= stats["fraction_soluble"] <= 1.0
