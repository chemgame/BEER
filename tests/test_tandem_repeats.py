"""Tests for beer.analysis.tandem_repeats."""
from __future__ import annotations
import pytest
from beer.analysis.tandem_repeats import (
    find_tandem_repeats,
    find_direct_repeats,
    find_compositional_repeats,
    calc_repeat_stats,
)


REPEAT_SEQ = "MSTSTSTSTSTSTSTSTST"   # clear tandem repeats of "ST"


def test_find_tandem_repeats_returns_list(fus_lc):
    repeats = find_tandem_repeats(fus_lc)
    assert isinstance(repeats, list)


def test_find_tandem_repeats_detects_repeats(fus_lc):
    """FUS LC domain is known to contain tandem repeats."""
    repeats = find_tandem_repeats(fus_lc)
    assert len(repeats) > 0


def test_find_tandem_repeats_dict_keys():
    repeats = find_tandem_repeats(REPEAT_SEQ)
    required = {"unit", "n_copies", "start_1based", "end_1based", "unit_length", "total_length"}
    for r in repeats:
        assert required <= set(r.keys()), (
            f"Repeat dict missing keys: {required - set(r.keys())}"
        )


def test_find_tandem_repeats_positions_in_range():
    repeats = find_tandem_repeats(REPEAT_SEQ)
    for r in repeats:
        assert r["start_1based"] >= 1
        assert r["end_1based"] <= len(REPEAT_SEQ)
        assert r["n_copies"] >= 2


def test_find_direct_repeats_returns_list(fus_lc):
    repeats = find_direct_repeats(fus_lc)
    assert isinstance(repeats, list)


def test_find_compositional_repeats_returns_list(fus_lc):
    repeats = find_compositional_repeats(fus_lc)
    assert isinstance(repeats, list)


def test_calc_repeat_stats_returns_dict(fus_lc):
    stats = calc_repeat_stats(fus_lc)
    assert isinstance(stats, dict)


def test_calc_repeat_stats_short_seq(short_seq):
    stats = calc_repeat_stats(short_seq)
    assert isinstance(stats, dict)
