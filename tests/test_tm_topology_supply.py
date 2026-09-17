"""The transmembrane report needs a Viterbi topology, so check it is actually supplied.

The formatter is covered in test_head_report.py, which passes a topology in directly. The bug
these tests guard against was in the supply: the analysis produced a profile but no topology, so
the report counted zero segments for a membrane protein whose profile was drawn correctly.
"""
import numpy as np
import pytest

from beer.analysis.core import analyze_sequence
from beer.analysis.head_report import head_findings_html

SEQ = ("MNGTEGPNFYVPFSNATGVVRSPFEYPQYYLAEPWQFSMLAAYMFLLIVLGFPINFLTLYVTVQHKKLRTPLNYILLNLAVADLFMV"
       "LGGFTSTLYTSLHGYFVFGPTGCNLEGFFATLGGEIALWSLVVLAIERYVVVCKPMSNFRFGENHAIMGVAFTWVMALACAAPPLA")


class _StubEmbedder:
    """Stands in for ESM C. The assertions are about plumbing, not about the numbers."""

    def is_available(self) -> bool:
        return True

    def embed(self, seq: str):
        return np.zeros((len(seq), 1152), dtype=np.float32)


def test_analysis_supplies_tm_topology():
    """analyze_sequence must return a topology, not just the membrane probability profile."""
    data = analyze_sequence(SEQ, embedder=_StubEmbedder())
    if data.get("tm_bilstm_profile") is None:
        pytest.skip("transmembrane head weights are not available locally")
    topo = data.get("tm_topology")
    assert topo is not None, "tm_topology missing; the report would show zero segments"
    assert len(topo) == len(SEQ)


def test_report_without_topology_says_so():
    """With no topology the report must not claim a protein-level call it cannot make."""
    scores = [0.9] * len(SEQ)
    html = head_findings_html("transmembrane", scores, SEQ, tm_topology=None)
    assert "Membrane segments (Viterbi)</td><td>0" in html


def test_report_counts_segments_when_topology_present():
    """A topology with two membrane stretches must be reported as two segments."""
    topo = [0] * 20 + [1] * 22 + [0] * 20 + [1] * 22 + [0] * (len(SEQ) - 84)
    html = head_findings_html("transmembrane", [0.9] * len(SEQ), SEQ, tm_topology=topo)
    assert "Membrane segments (Viterbi)</td><td>2" in html
    assert "transmembrane protein" in html
