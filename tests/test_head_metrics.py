"""The metrics table must not drift from the weights it describes.

``beer.models.metrics`` mirrors the ``auc`` / ``auprc`` / ``f1_threshold`` fields
of the shipped ``*_head.npz`` files so the GUI can place threshold lines and show
AUROCs without touching disk.  If a head is ever retrained, these tests fail
until the table is updated with it.

They skip when the weight files are not present locally (a fresh checkout that
has not downloaded them yet, or CI).
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from beer.models import metrics as m

_MODELS_DIR = pathlib.Path(m.__file__).parent
_TOLERANCE = 5e-5   # the table is rounded for display


def _npz_path(task: str) -> pathlib.Path:
    return _MODELS_DIR / f"{task}_head.npz"


def _tasks_with_weights() -> list[str]:
    return [t for t in sorted(m.HEAD_METRICS) if _npz_path(t).exists()]


def test_table_covers_all_twenty_four_heads():
    assert len(m.HEAD_METRICS) == 24


def test_every_head_has_a_loader():
    from beer import models

    for task in m.HEAD_METRICS:
        assert hasattr(models, f"load_{task}_head"), f"no loader for {task}"


@pytest.mark.parametrize("task", sorted(m.HEAD_METRICS))
def test_metrics_match_shipped_weights(task):
    path = _npz_path(task)
    if not path.exists():
        pytest.skip(f"{path.name} not present locally")

    npz = np.load(path, allow_pickle=False)
    expected = m.HEAD_METRICS[task]

    assert float(npz["auc"]) == pytest.approx(expected.auroc, abs=_TOLERANCE)

    for field, value in (("auprc", expected.auprc), ("f1_threshold", expected.f1_threshold)):
        if field in npz:
            assert value is not None, f"{task}: {field} in the .npz but None in the table"
            assert float(npz[field]) == pytest.approx(value, abs=_TOLERANCE)
        else:
            assert value is None, f"{task}: {field} in the table but not in the .npz"


def test_weights_present_for_this_checkout():
    """Guard against the parametrised test silently skipping every head."""
    if not _tasks_with_weights():
        pytest.skip("no head weights in the package directory")
    assert len(_tasks_with_weights()) == 24


def test_viterbi_heads_still_get_a_threshold():
    """The two Viterbi-decoded heads have no F1-max point but must draw a line."""
    for task in ("transmembrane", "secondary_structure"):
        assert m.HEAD_METRICS[task].f1_threshold is None
    assert m.f1_threshold("transmembrane") == pytest.approx(0.81339)
    assert m.f1_threshold("secondary_structure") == m.DEFAULT_THRESHOLD


def test_ss3_aliases_resolve_to_the_shared_head():
    for alias in ("secondary_structure_helix", "secondary_structure_strand",
                  "secondary_structure_coil"):
        assert m.auroc(alias) == m.HEAD_METRICS["secondary_structure"].auroc
        assert m.f1_threshold(alias) == m.DEFAULT_THRESHOLD


def test_unknown_task_falls_back_without_raising():
    assert m.f1_threshold("not_a_head") == m.DEFAULT_THRESHOLD
    assert m.auroc("not_a_head") is None
    assert m.auroc_str("not_a_head") == ""
