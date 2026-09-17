"""Probability calibration for the prediction heads.

The heads are trained with a focal loss, which reweights easy negatives. That makes them good at
ranking residues and poor at reporting a frequency: a residue scored 0.9 is not positive nine
times in ten. A two-parameter Platt map on the logit corrects the scale,

    calibrated = sigmoid(a * logit(raw) + b)

fitted on each head's validation split and measured once on its test split. The map is monotone,
so it cannot change the order of residues. Area under the ROC curve and average precision are
therefore unchanged to the last digit, and because each head's decision threshold is mapped
through the same function, every call BEER makes is identical to before. Only the number shown
to the user changes.

A head keeps its map only if the calibration error measured on the test split actually improved.
The repeat head did not improve and is deliberately absent, so it reports raw probabilities.

Two heads are excluded by construction. The transmembrane head decodes with a CRF, where
rescaling the emissions against the transition scores would move the Viterbi path and change the
segments. The secondary-structure head is a three-class softmax whose per-residue argmax, and so
its Q3 accuracy, a shared scale cannot change.

Only the two fitted coefficients are stored. The calibrated threshold is derived from them and
from the head's own threshold whenever it is asked for, so the two can never disagree. Storing it
as a third number is what allowed them to drift once already.

This table is generated from the fitting run, not written by hand. Source values live in
data/benchmarks/calibration_platt.json, produced by scripts/fit_calibration.py.
"""
from __future__ import annotations

import math

#: task -> (a, b) of the fitted Platt map. Heads absent from this table are not calibrated.
PLATT: dict[str, tuple[float, float]] = {
    "acetylation":         (2.543208, -5.575222),
    "active_site":         (2.534832, -5.00519),
    "binding_site":        (2.323393, -2.959409),
    "coiled_coil":         (2.189034, -1.613947),
    "disorder":            (2.739265, -1.62805),
    "disulfide":           (2.001454, -3.294155),
    "dna_binding":         (2.267873, -3.232535),
    "glycosylation":       (2.295331, -4.087752),
    "intramembrane":       (1.631915, -2.677799),
    "lcd":                 (1.791741, -1.567413),
    "lipidation":          (2.537197, -6.575697),
    "methylation":         (1.847171, -5.311877),
    "motif":               (2.089562, -4.165324),
    "nucleotide_binding":  (0.620531, -0.888675),
    "phosphorylation":     (2.335535, -4.892683),
    "propeptide":          (3.697062, -2.310771),
    "rna_binding":         (1.688688, -1.546082),
    "signal_peptide":      (2.852116, -3.060202),
    "transit_peptide":     (2.299682, -2.012147),
    "ubiquitination":      (1.388266, -4.831482),
    "zinc_finger":         (2.865729, -4.468654),
}

_EPS = 1e-6


def _sigmoid_of_logit(a: float, b: float, p: float) -> float:
    p = min(max(float(p), _EPS), 1.0 - _EPS)
    return 1.0 / (1.0 + math.exp(-(a * math.log(p / (1.0 - p)) + b)))


def is_calibrated(task: str) -> bool:
    """True when *task* has a fitted map that improved its calibration."""
    return task in PLATT


def calibrated_threshold(task: str, raw_threshold: float) -> float:
    """The head's decision threshold on the calibrated scale.

    Derived from *raw_threshold* through the same map applied to the scores, so the pair always
    agrees and the calls are identical to the uncalibrated model.
    """
    entry = PLATT.get(task)
    if entry is None:
        return raw_threshold
    return _sigmoid_of_logit(entry[0], entry[1], raw_threshold)


def apply(task: str, values):
    """Map raw probabilities for *task* onto the calibrated scale.

    Accepts a sequence or a numpy array and returns the same shape. Heads without a map are
    returned unchanged, so this is safe to call unconditionally.
    """
    entry = PLATT.get(task)
    if entry is None or values is None:
        return values
    a, b = entry

    try:
        import numpy as np
    except ImportError:                                   # pragma: no cover
        np = None

    if np is not None and isinstance(values, np.ndarray):
        p = np.clip(values, _EPS, 1.0 - _EPS)
        return 1.0 / (1.0 + np.exp(-(a * np.log(p / (1.0 - p)) + b)))

    return [_sigmoid_of_logit(a, b, v) for v in values]
