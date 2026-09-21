"""The Signal Peptide report section must agree with the protein-level call."""
from beer.analysis.calls import signal_peptide_call
from beer.analysis.signal_peptide import format_signal_report


def _section(profile):
    return format_signal_report("M" * len(profile), "#4361ee", profile)


def test_short_signal_peptide_is_reported():
    # A 22-residue signal peptide leaves the mean over residues 1-70 far below 0.5.
    profile = [0.95] * 22 + [0.02] * 300
    assert signal_peptide_call(profile)[0]
    html = _section(profile)
    assert "detected, predicted cleavage after residue 22" in html
    assert "not detected" not in html


def test_no_signal_peptide():
    html = _section([0.02] * 322)
    assert "not detected" in html


def test_run_starting_late_is_not_reported():
    profile = [0.02] * 10 + [0.95] * 30 + [0.02] * 200
    assert not signal_peptide_call(profile)[0]
    assert "not detected" in _section(profile)
