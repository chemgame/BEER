"""Tests for beer.analysis.signal_peptide."""
from __future__ import annotations
import pytest
from beer.analysis.signal_peptide import predict_signal_peptide, predict_gpi_anchor


def test_predict_signal_peptide_returns_dict(short_seq):
    result = predict_signal_peptide(short_seq)
    assert isinstance(result, dict)


def test_predict_signal_peptide_required_keys(short_seq):
    """Keys returned by predict_signal_peptide per docstring."""
    result = predict_signal_peptide(short_seq)
    required = {"n_end", "h_start", "h_end", "h_length", "c_start", "c_has_axa",
                "cleavage_site", "h_region_seq", "h_region_score", "n_score"}
    assert required <= set(result.keys()), (
        f"Missing keys: {required - set(result.keys())}"
    )


def test_predict_signal_peptide_no_verdict(short_seq):
    """score and verdict must not be present — removed as scientifically unsound."""
    result = predict_signal_peptide(short_seq)
    assert "score" not in result
    assert "verdict" not in result


def test_predict_signal_peptide_h_region_score_is_float(short_seq):
    result = predict_signal_peptide(short_seq)
    assert isinstance(result["h_region_score"], float)


def test_predict_signal_peptide_fus_lc_no_crash(fus_lc):
    """predict_signal_peptide must not raise on FUS LC domain."""
    result = predict_signal_peptide(fus_lc)
    assert isinstance(result, dict)
    assert "h_region_score" in result


def test_predict_gpi_anchor_returns_dict(short_seq):
    result = predict_gpi_anchor(short_seq)
    assert isinstance(result, dict)


def test_predict_gpi_anchor_required_keys(short_seq):
    """Keys returned by predict_gpi_anchor per docstring."""
    result = predict_gpi_anchor(short_seq)
    required = {"omega_found", "omega_position", "omega_aa", "spacer_length",
                "spacer_ok", "tail_ok", "tail_start", "tail_seq", "tail_kd_mean"}
    assert required <= set(result.keys()), (
        f"Missing keys: {required - set(result.keys())}"
    )


def test_predict_gpi_anchor_no_verdict(fus_lc):
    """score and verdict must not be present — removed as scientifically unsound."""
    result = predict_gpi_anchor(fus_lc)
    assert "score" not in result
    assert "verdict" not in result


def test_predict_gpi_anchor_tail_kd_is_float(fus_lc):
    result = predict_gpi_anchor(fus_lc)
    assert isinstance(result["tail_kd_mean"], float)
