"""analyze_sequence orchestrator: keys, values, N-end-rule half-life."""
from beer.analysis.core import AnalysisTools


REQUIRED_KEYS = (
    "seq", "mol_weight", "iso_point", "gravy", "net_charge_7",
    "fcr", "ncpr", "kappa", "instability", "instability_class",
    "half_life", "report_sections",
)


def test_analyze_sequence_has_required_keys():
    d = AnalysisTools.analyze_sequence("MASTKLVWQRDEFGHIKLMNP")
    for k in REQUIRED_KEYS:
        assert k in d, f"missing key {k}"
    assert "Properties" in d["report_sections"]


def test_all_alanine_gravy_and_charge():
    d = AnalysisTools.analyze_sequence("AAAAAAAAAA")
    # Kyte-Doolittle hydropathy of Ala is +1.8 → GRAVY of poly-Ala is 1.8.
    assert abs(d["gravy"] - 1.8) < 0.05
    # No charged residues → FCR and NCPR are 0.
    assert d["fcr"] == 0 and d["ncpr"] == 0
    assert d["mol_weight"] > 0


def test_n_end_rule_half_life():
    assert AnalysisTools.analyze_sequence("MAAAAAA")["half_life"] == "30 h"   # Met
    assert AnalysisTools.analyze_sequence("RAAAAAA")["half_life"] == "1 h"    # Arg
    assert AnalysisTools.analyze_sequence("VAAAAAA")["half_life"] == "100 h"  # Val


def test_instability_classification_label():
    d = AnalysisTools.analyze_sequence("MASTKLVWQRDEFGHIKLMNP")
    assert d["instability_class"] in ("stable", "unstable", "unknown")
