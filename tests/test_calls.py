"""Protein-level calls: signal peptide from the SP profile, TM segments from the CRF topology."""
from __future__ import annotations

import pytest

from beer.analysis.calls import (
    SP_MIN_LENGTH, TM_MIN_SEGMENT, signal_peptide_call, tm_segments,
)

THR = 0.8


def _profile(high_from: int, high_to: int, length: int = 120, hi=0.95, lo=0.1):
    return [hi if high_from <= i < high_to else lo for i in range(length)]


def test_n_terminal_run_is_called_with_its_end():
    assert signal_peptide_call(_profile(0, 18), THR) == (True, 18)


def test_run_may_start_after_the_initiator_residues():
    called, end = signal_peptide_call(_profile(2, 24), THR)
    assert called and end == 24


def test_run_starting_too_late_is_not_a_signal_peptide():
    assert signal_peptide_call(_profile(10, 40), THR) == (False, 0)


def test_short_run_is_not_called():
    assert signal_peptide_call(_profile(0, SP_MIN_LENGTH - 1), THR) == (False, 0)


def test_single_spike_is_not_called():
    """One high-scoring residue is not a signal peptide."""
    p = [0.1] * 120
    p[5] = 0.99
    assert signal_peptide_call(p, THR) == (False, 0)


@pytest.mark.parametrize("profile", [None, []])
def test_empty_profile(profile):
    assert signal_peptide_call(profile, THR) == (False, 0)


def test_default_threshold_is_the_heads_calibrated_value():
    from beer.models.metrics import f1_threshold
    just_above = f1_threshold("signal_peptide") + 0.01
    p = [just_above] * 20 + [0.0] * 100
    assert signal_peptide_call(p)[0]


def _topology(*segments, length=400):
    t = [0] * length
    for a, b in segments:          # 1-based inclusive
        for i in range(a - 1, b):
            t[i] = 1
    return t


def test_long_segments_are_reported():
    assert tm_segments(_topology((35, 59), (71, 93))) == [(35, 59), (71, 93)]


def test_beta_strand_length_segments_are_reported():
    """UniProt annotates beta-barrel strands and 13-17 residue helices as transmembrane."""
    strand = (40, 40 + TM_MIN_SEGMENT - 1)
    assert tm_segments(_topology(strand)) == [strand]


def test_segments_below_the_minimum_are_dropped():
    short = (22, 22 + TM_MIN_SEGMENT - 2)
    assert tm_segments(_topology(short, (100, 125))) == [(100, 125)]


def test_segment_touching_the_c_terminus_is_closed():
    assert tm_segments(_topology((381, 400))) == [(381, 400)]


def test_no_topology():
    assert tm_segments(None) == [] and tm_segments([]) == []
