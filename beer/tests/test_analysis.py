"""Pytest tests for beer.analysis.core.AnalysisTools."""
from __future__ import annotations

import pytest

from beer.analysis.core import AnalysisTools


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def all20_result():
    """Analyze a sequence containing all 20 canonical amino acids."""
    seq = "ACDEFGHIKLMNPQRSTVWY"
    return seq, AnalysisTools.analyze_sequence(seq)


@pytest.fixture(scope="module")
def charge_result():
    """Analyze a sequence with equal K and E counts."""
    seq = "KKKKKKKKKKEEEEEEEEEE"
    return seq, AnalysisTools.analyze_sequence(seq)


@pytest.fixture(scope="module")
def hydrophobic_result():
    """Analyze a poly-leucine sequence."""
    seq = "LLLLLLLLLLL"
    return seq, AnalysisTools.analyze_sequence(seq)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBasicProperties:
    def test_result_is_dict(self, all20_result):
        _, result = all20_result
        assert isinstance(result, dict)

    def test_mol_weight_positive(self, all20_result):
        _, result = all20_result
        assert result["mol_weight"] > 0

    def test_iso_point_range(self, all20_result):
        _, result = all20_result
        assert 0 < result["iso_point"] < 14

    def test_seq_roundtrip(self, all20_result):
        seq, result = all20_result
        assert result["seq"] == seq

    def test_aa_counts_20_entries(self, all20_result):
        _, result = all20_result
        # aa_counts from BioPython only includes residues present; all 20 are present
        assert len(result["aa_counts"]) == 20

    def test_disorder_scores_length(self, all20_result):
        seq, result = all20_result
        assert len(result["disorder_scores"]) == len(seq)

    def test_report_sections_min_keys(self, all20_result):
        _, result = all20_result
        assert isinstance(result["report_sections"], dict)
        assert len(result["report_sections"]) >= 5


class TestLongSequence:
    def test_poly_ala_no_exception(self):
        seq = "A" * 200
        result = AnalysisTools.analyze_sequence(seq)
        assert result["mol_weight"] > 0


class TestChargeProperties:
    def test_fcr_high(self, charge_result):
        _, result = charge_result
        # 10 K + 10 E in a 20-aa sequence → FCR = 1.0
        assert result["fcr"] > 0.8

    def test_ncpr_near_zero(self, charge_result):
        _, result = charge_result
        # Equal positive and negative → NCPR ≈ 0
        assert abs(result["ncpr"]) < 0.1


class TestHydrophobicity:
    def test_gravy_leucine(self, hydrophobic_result):
        _, result = hydrophobic_result
        # Leucine KD = 3.8; poly-L GRAVY should be well above 2.0
        assert result["gravy"] > 2.0


class TestReportSectionsHtml:
    def test_all_sections_non_empty_strings(self):
        seq = "MASMTGGQQMG"
        result = AnalysisTools.analyze_sequence(seq)
        sections = result["report_sections"]
        assert isinstance(sections, dict)
        for key, value in sections.items():
            assert isinstance(value, str), f"Section '{key}' is not a string"
            assert len(value.strip()) > 0, f"Section '{key}' is empty"


class TestBiophysicsUtils:
    @pytest.mark.skip(reason="calc_mw not a public export; biophysics tested indirectly via analyze_sequence")
    def test_fasta_parse(self):
        pass

    def test_calc_net_charge_importable(self):
        """Verify that biophysics utilities can be imported directly."""
        from beer.utils.biophysics import calc_net_charge
        seq = "KKKKK"
        charge = calc_net_charge(seq, pH=7.0)
        assert charge > 0, "Poly-K should have a positive net charge at pH 7"

    def test_sliding_window_hydrophobicity(self):
        from beer.utils.biophysics import sliding_window_hydrophobicity
        seq = "LLLLLLLLLLL"
        profile = sliding_window_hydrophobicity(seq, window_size=9)
        assert len(profile) > 0
        assert all(v > 0 for v in profile), "All-leucine window should be hydrophobic"
