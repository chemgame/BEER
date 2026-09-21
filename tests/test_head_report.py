"""Protein-level calls and per-head findings for the 21 thresholded heads."""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from beer.analysis import calls
from beer.analysis.head_report import DATA_KEY_TO_TASK, head_findings_html
from beer.models import metrics

THRESHOLDED = sorted(t for t, m in metrics.HEAD_METRICS.items() if m.f1_threshold is not None)


def _profile(*spans, length=150, hi=0.99, lo=0.01):
    p = [lo] * length
    for a, b in spans:                 # 1-based inclusive
        for i in range(a - 1, b):
            p[i] = hi
    return p


def test_every_thresholded_head_has_a_rule_and_a_protein_auroc():
    for task in THRESHOLDED:
        assert task in calls.PROTEIN_RULES or task == "signal_peptide", task
        assert metrics.protein_auroc(task) is not None, task


def test_site_heads_are_thresholded_heads():
    assert calls.SITE_HEADS <= set(THRESHOLDED)


def test_runs_above():
    assert calls.runs_above(_profile((3, 5), (10, 10)), 0.5) == [(3, 5), (10, 10)]
    assert calls.runs_above(_profile((148, 150)), 0.5) == [(148, 150)]
    assert calls.runs_above([], 0.5) == [] and calls.runs_above(None, 0.5) == []


def test_a_single_residue_calls_a_site_head():
    call = calls.protein_call("active_site", _profile((70, 70)))
    assert call.called is True and call.spans == [(70, 70)]


def test_region_heads_need_their_minimum_run():
    k = calls.PROTEIN_RULES["disorder"].min_run
    assert calls.protein_call("disorder", _profile((20, 20 + k - 2))).called is False
    assert calls.protein_call("disorder", _profile((20, 20 + k - 1))).called is True


def test_the_call_uses_the_heads_calibrated_threshold():
    thr = metrics.f1_threshold("glycosylation")
    below = _profile((40, 40), hi=thr - 0.01)
    assert calls.protein_call("glycosylation", below).called is False


def test_transit_peptide_must_be_n_terminal():
    assert calls.protein_call("transit_peptide", _profile((1, 30))).called is True
    assert calls.protein_call("transit_peptide", _profile((40, 80))).called is False


def test_weak_heads_make_no_protein_call():
    weak = [t for t in THRESHOLDED if metrics.protein_auroc(t) < calls.MIN_PROTEIN_AUROC]
    assert weak, "expected at least one weak head in the table"
    for task in weak:
        call = calls.protein_call(task, _profile((30, 60)))
        assert call.called is None and call.spans, task


def test_rule_text_mentions_the_rule():
    assert "one residue" in calls.rule_text("active_site")
    assert "first" in calls.rule_text("transit_peptide")


@pytest.mark.parametrize("data_key,task", sorted(DATA_KEY_TO_TASK.items()))
def test_findings_html_renders_for_every_head(data_key, task):
    html = head_findings_html(task, _profile((30, 50), (90, 90)), "A" * 150,
                              tm_topology=[0] * 30 + [1] * 22 + [0] * 98)
    assert "<table" in html


def test_site_heads_list_residues_region_heads_list_regions():
    p, seq = _profile((30, 32), (90, 90)), "K" * 150
    assert "Predicted sites" in head_findings_html("acetylation", p, seq)
    assert "Lys-90" in head_findings_html("acetylation", p, seq)
    assert "Predicted regions" in head_findings_html("coiled_coil", _profile((30, 60)), seq)


def test_architecture_table_matches_the_weights():
    models_dir = pathlib.Path(metrics.__file__).parent
    for task, (arch, hidden) in metrics.HEAD_ARCHITECTURE.items():
        path = models_dir / f"{task}_head.npz"
        if not path.exists():
            pytest.skip("weights not present locally")
        z = np.load(path, allow_pickle=False)
        assert str(z["architecture"]) == arch, task
        assert int(z["lstm_hidden"]) == hidden, task


def test_architecture_note_describes_the_network():
    assert "CNN" in metrics.architecture_note("disorder")
    assert "384" in metrics.architecture_note("binding_site")
    assert "CRF" in metrics.architecture_note("transmembrane")
    assert "positional" in metrics.architecture_note("propeptide")


def test_protein_call_shows_measured_rates():
    from beer.models.metrics import PROTEIN_CALL_RATES
    for task, (tpr, fpr) in PROTEIN_CALL_RATES.items():
        assert 0.0 <= tpr <= 1.0 and 0.0 <= fpr <= 1.0, task
    from beer.analysis.head_report import head_findings_html
    html = head_findings_html("disorder", [0.9] * 40 + [0.01] * 60, "M" * 100)
    assert "held-out test proteins" in html and "% of proteins that do not" in html
