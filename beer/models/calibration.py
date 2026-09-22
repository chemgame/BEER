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

Every thresholded head has a map. Whether a map helps was judged on the validation split alone,
by fitting it on one half and scoring it on the other, and it helped for all 22 heads.

Two heads are excluded by construction. The transmembrane head decodes with a CRF, where
rescaling the emissions against the transition scores would move the Viterbi path and change the
segments. The secondary-structure head is a three-class softmax whose per-residue argmax, and so
its Q3 accuracy, a shared scale cannot change.

Only the two fitted coefficients are stored. The calibrated threshold is derived from them and
from the head's own threshold whenever it is asked for, so the two can never disagree. Storing it
as a third number is what allowed them to drift once already.

This table is generated from the fitting run on the full validation split, not written by hand.
"""
from __future__ import annotations

import math

#: task -> (a, b) of the fitted Platt map. Heads absent from this table are not calibrated.
PLATT: dict[str, tuple[float, float]] = {
    "acetylation":        (2.398889, -5.443673),
    "active_site":        (2.466133, -4.908169),
    "binding_site":       (2.380925, -2.952080),
    "coiled_coil":        (2.164472, -1.568401),
    "disorder":           (2.071457, -1.715412),
    "disulfide":          (1.972934, -3.312016),
    "dna_binding":        (2.288768, -3.252406),
    "glycosylation":      (2.217385, -3.971966),
    "intramembrane":      (1.630952, -2.675041),
    "lcd":                (1.654834, -1.489530),
    "lipidation":         (2.555071, -6.693884),
    "methylation":        (1.851033, -5.247563),
    "motif":              (2.079278, -3.989837),
    "nucleotide_binding": (0.628539, -0.973822),
    "phosphorylation":    (2.355268, -4.893051),
    "propeptide":         (3.640716, -2.446070),
    "repeat":             (0.868249, -0.338762),
    "rna_binding":        (1.703737, -1.499341),
    "signal_peptide":     (2.962842, -3.239683),
    "transit_peptide":    (2.340948, -2.072095),
    "ubiquitination":     (1.401118, -4.842862),
    "zinc_finger":        (2.880222, -4.536060),
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
