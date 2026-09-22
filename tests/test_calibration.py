"""Calibration must change the numbers shown and nothing else.

The justification for calibrating is that the map is monotone: it rescales a probability without
reordering residues, and the head's threshold is mapped through the same function. So every call
BEER makes must be identical to the uncalibrated model. These tests assert that directly, with no
model and no embedder.

The threshold is derived from the fitted coefficients rather than stored beside them. An earlier
version stored it as a third number, and it drifted out of step with metrics.py for the five
rebuilt binding heads, which is exactly what the first test here caught.
"""
import numpy as np
import pytest

from beer.models import calibration
from beer.models.metrics import HEAD_METRICS, f1_threshold

CALIBRATED = sorted(calibration.PLATT)


def _raw_threshold(task: str) -> float:
    return HEAD_METRICS[task].f1_threshold


@pytest.mark.parametrize("task", CALIBRATED)
def test_calls_are_identical_after_calibration(task):
    """A residue is called positive on the calibrated scale exactly when it was before."""
    raw = np.linspace(1e-6, 1 - 1e-6, 4001)
    cal = np.asarray(calibration.apply(task, raw))
    before = raw >= _raw_threshold(task)
    after = cal >= calibration.calibrated_threshold(task, _raw_threshold(task))
    assert np.array_equal(before, after)


@pytest.mark.parametrize("task", CALIBRATED)
def test_map_is_monotone(task):
    """Ranking is preserved, so AUROC and average precision cannot change."""
    raw = np.linspace(1e-6, 1 - 1e-6, 2001)
    cal = np.asarray(calibration.apply(task, raw))
    assert np.all(np.diff(cal) > 0)


@pytest.mark.parametrize("task", CALIBRATED)
def test_threshold_accessor_uses_the_calibrated_scale(task):
    """f1_threshold must report the threshold on the scale the head now reports."""
    expected = calibration.calibrated_threshold(task, _raw_threshold(task))
    assert f1_threshold(task) == pytest.approx(expected)


@pytest.mark.parametrize("task", CALIBRATED)
def test_threshold_is_derived_not_stored(task):
    """Mapping the head's own threshold must agree with what the accessor returns.

    This is the invariant that broke when the threshold was stored separately.
    """
    entry = calibration.PLATT[task]
    assert len(entry) == 2, "storing a third value lets it drift out of step"
    raw = _raw_threshold(task)
    direct = np.asarray(calibration.apply(task, np.array([raw])))[0]
    assert calibration.calibrated_threshold(task, raw) == pytest.approx(direct)


@pytest.mark.parametrize("task", ["transmembrane", "secondary_structure"])
def test_uncalibrated_heads_are_untouched(task):
    """The two Viterbi-decoded heads are excluded by construction."""
    assert not calibration.is_calibrated(task)
    raw = [0.01, 0.25, 0.5, 0.75, 0.99]
    assert calibration.apply(task, raw) == raw
    assert calibration.calibrated_threshold(task, 0.42) == 0.42


def test_apply_accepts_lists_and_arrays_alike():
    task = CALIBRATED[0]
    values = [0.1, 0.5, 0.9]
    from_list = calibration.apply(task, values)
    from_array = calibration.apply(task, np.asarray(values))
    assert np.allclose(np.asarray(from_list), np.asarray(from_array))


def test_apply_tolerates_none():
    assert calibration.apply(CALIBRATED[0], None) is None
