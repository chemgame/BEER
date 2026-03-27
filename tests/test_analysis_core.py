"""Integration tests for beer.analysis.core.AnalysisTools.analyze_sequence()."""
from __future__ import annotations
import pytest
from beer.analysis.core import AnalysisTools

# Required top-level keys in the result dict
REQUIRED_KEYS = {
    "report_sections",
    "mol_weight",
    "iso_point",
    "net_charge_7",
    "gravy",
    "aromaticity",
    "fcr",
    "ncpr",
    "arom_f",
    "disorder_scores",
    "disorder_f",
    "aa_counts",
    "aa_freq",
    "hydro_profile",
    "ncpr_profile",
    "entropy_profile",
    "seq",
    "window_size",
    "tm_helices",
    "motifs",
    "solub_stats",
    "ptm_sites",
    "sp_result",
    "gpi_result",
    "scd",
    "scd_profile",
    "scd_blocks",
    "rbp",
    "rbp_profile",
    "tandem_stats",
}

# Required report section names
REQUIRED_SECTIONS = {
    "Composition",
    "Properties",
    "Hydrophobicity",
    "Charge",
    "Disorder",
    "TM Helices",
    "LARKS",
    "Linear Motifs",
    "\u03b2-Aggregation & Solubility",
    "PTM Sites",
    "Signal Peptide & GPI",
    "Charge Decoration (SCD)",
    "RNA Binding",
    "Tandem Repeats",
}


def test_analyze_sequence_returns_dict(fus_lc):
    result = AnalysisTools.analyze_sequence(fus_lc)
    assert isinstance(result, dict)


def test_analyze_sequence_required_keys_present(fus_lc):
    result = AnalysisTools.analyze_sequence(fus_lc)
    missing = REQUIRED_KEYS - set(result.keys())
    assert not missing, f"Missing keys in result: {missing}"


def test_analyze_sequence_mol_weight_range(fus_lc):
    result = AnalysisTools.analyze_sequence(fus_lc)
    mw = result["mol_weight"]
    # FUS LC is ~62 aa; MW should be roughly 1000-10000 Da per 10 residues
    assert 1_000 < mw < 1_000_000, f"mol_weight={mw} is out of expected range"


def test_analyze_sequence_seq_preserved(fus_lc):
    result = AnalysisTools.analyze_sequence(fus_lc)
    assert result["seq"] == fus_lc


def test_analyze_sequence_report_sections_present(fus_lc):
    result = AnalysisTools.analyze_sequence(fus_lc)
    sections = result["report_sections"]
    assert isinstance(sections, dict)
    missing = REQUIRED_SECTIONS - set(sections.keys())
    assert not missing, f"Missing report sections: {missing}"


def test_analyze_sequence_report_sections_are_html(fus_lc):
    result = AnalysisTools.analyze_sequence(fus_lc)
    for name, html in result["report_sections"].items():
        assert isinstance(html, str), f"Section {name!r} is not a string"
        assert len(html) > 0, f"Section {name!r} is empty"


def test_analyze_sequence_hydro_profile_length(fus_lc):
    window_size = 9  # default
    result = AnalysisTools.analyze_sequence(fus_lc, window_size=window_size)
    # sliding window returns len(seq) - window + 1 values
    assert len(result["hydro_profile"]) == len(fus_lc) - window_size + 1


def test_analyze_sequence_disorder_scores_length(fus_lc):
    result = AnalysisTools.analyze_sequence(fus_lc)
    assert len(result["disorder_scores"]) == len(fus_lc)


def test_analyze_sequence_with_embedder_kwarg(fus_lc, unavailable_embedder):
    """Passing an unavailable embedder must not raise and should still return a result."""
    result = AnalysisTools.analyze_sequence(fus_lc, embedder=unavailable_embedder)
    assert isinstance(result, dict)
    assert "mol_weight" in result


def test_analyze_sequence_with_mock_embedder(
    fus_lc, mock_embedder
):
    """Passing a mock embedder must not raise."""
    result = AnalysisTools.analyze_sequence(fus_lc, embedder=mock_embedder)
    assert isinstance(result, dict)
    assert "disorder_scores" in result


def test_analyze_sequence_short_seq(short_seq):
    result = AnalysisTools.analyze_sequence(short_seq)
    assert isinstance(result, dict)
    assert result["mol_weight"] > 0


def test_analyze_sequence_custom_pH(short_seq):
    result_7  = AnalysisTools.analyze_sequence(short_seq, pH_value=7.0)
    result_2  = AnalysisTools.analyze_sequence(short_seq, pH_value=2.0)
    result_12 = AnalysisTools.analyze_sequence(short_seq, pH_value=12.0)
    # net_charge_7 is always pH=7.0 regardless of pH_value parameter
    assert result_7["net_charge_7"] == pytest.approx(result_2["net_charge_7"], abs=1e-6)
    assert result_7["net_charge_7"] == pytest.approx(result_12["net_charge_7"], abs=1e-6)


def test_analyze_sequence_fcr_ncpr_range(fus_lc):
    result = AnalysisTools.analyze_sequence(fus_lc)
    assert 0.0 <= result["fcr"] <= 1.0
    assert -1.0 <= result["ncpr"] <= 1.0


def test_analyze_sequence_arom_f_range(short_seq):
    result = AnalysisTools.analyze_sequence(short_seq)
    assert 0.0 <= result["arom_f"] <= 1.0
