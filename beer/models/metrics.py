"""Canonical evaluation metrics for the shipped v3.2.0 prediction heads.

Single source of truth for every AUROC, AUPRC and F1-maximising threshold that
BEER displays — the profile figures, the report sparklines, the structure-tab
binary colouring and the AI Predictions list all read from here.

The values are exactly those stored in the shipped ``*_head.npz`` files:
thresholds chosen on each head's validation split, metrics measured on its
held-out test split.  ``tests/test_head_metrics.py`` asserts
that this table and the weight files stay in sync whenever the ``.npz`` files
are available locally.

Mirroring the numbers here rather than reading them from the ``.npz`` at draw
time keeps the GUI from touching disk — or triggering a first-use weight
download — just to place a threshold line on a plot.
"""
from __future__ import annotations

from typing import NamedTuple


class HeadMetrics(NamedTuple):
    """Test-set metrics for one prediction head.

    ``auroc`` is Q3 accuracy for the secondary-structure head.  ``auprc`` and
    ``f1_threshold`` are ``None`` for the two sequence-labelling heads
    (transmembrane BiLSTM-CRF, SS3), which decode with Viterbi rather than by
    thresholding a probability.
    """

    auroc: float
    auprc: float | None
    f1_threshold: float | None


#: task name (matches the ``<task>_head.npz`` stem) → metrics
HEAD_METRICS: dict[str, HeadMetrics] = {
    "acetylation":         HeadMetrics(0.98462, 0.25873, 0.84387),
    "active_site":         HeadMetrics(0.98713, 0.47844, 0.80964),
    "binding_site":        HeadMetrics(0.90556, 0.48416, 0.67283),
    "coiled_coil":         HeadMetrics(0.97865, 0.88618, 0.63786),
    "disorder":            HeadMetrics(0.91247, 0.79703, 0.65091),
    "disulfide":           HeadMetrics(0.99874, 0.94309, 0.83614),
    "dna_binding":         HeadMetrics(0.96456, 0.48431, 0.72915),
    "glycosylation":       HeadMetrics(0.99141, 0.80917, 0.80819),
    "intramembrane":       HeadMetrics(0.99052, 0.75239, 0.78581),
    "lcd":                 HeadMetrics(0.97046, 0.76661, 0.65064),
    "lipidation":          HeadMetrics(0.98955, 0.50154, 0.92039),
    "methylation":         HeadMetrics(0.93758, 0.20345, 0.82442),
    "motif":               HeadMetrics(0.86770, 0.28061, 0.76315),
    "nucleotide_binding":  HeadMetrics(0.96981, 0.69167, 0.72601),
    "phosphorylation":     HeadMetrics(0.97315, 0.29494, 0.80981),
    "propeptide":          HeadMetrics(0.96243, 0.81080, 0.65580),
    "repeat":              HeadMetrics(0.95794, 0.92615, 0.53972),
    "rna_binding":         HeadMetrics(0.91965, 0.62245, 0.63909),
    "secondary_structure": HeadMetrics(0.87947, None, None),
    "signal_peptide":      HeadMetrics(0.99973, 0.99380, 0.74974),
    "transit_peptide":     HeadMetrics(0.99325, 0.93425, 0.68474),
    "transmembrane":       HeadMetrics(0.98885, None, None),
    "ubiquitination":      HeadMetrics(0.98322, 0.09694, 0.83290),
    "zinc_finger":         HeadMetrics(0.97925, 0.71793, 0.80561),
}

#: Protein-level AUROC on the held-out test split: each protein scored by its highest residue
#: probability, positives being proteins with at least one annotated residue.  The Viterbi
#: heads have none.
PROTEIN_AUROC: dict[str, float] = {
    "acetylation":         0.7878,
    "active_site":         0.9714,
    "binding_site":        0.9188,
    "coiled_coil":         0.9331,
    "disorder":            0.8484,
    "disulfide":           0.9618,
    "dna_binding":         0.9338,
    "glycosylation":       0.9645,
    "intramembrane":       0.9855,
    "lcd":                 0.9442,
    "lipidation":          0.7166,
    "methylation":         0.6938,
    "motif":               0.7085,
    "nucleotide_binding":  0.9807,
    "phosphorylation":     0.7936,
    "propeptide":          0.8097,
    "repeat":              0.9702,
    "rna_binding":         0.9807,
    "signal_peptide":      0.9979,
    "transit_peptide":     0.9944,
    "ubiquitination":      0.7778,
    "zinc_finger":         0.9186,
}

#: Protein-level AUROC on the validation split, scored the same way.  This, and not the test
#: value, decides whether a head makes a protein-level call at all.
PROTEIN_AUROC_VAL: dict[str, float] = {
    "acetylation":        0.7948,
    "active_site":        0.9688,
    "binding_site":       0.9013,
    "coiled_coil":        0.9199,
    "disorder":           0.8571,
    "disulfide":          0.9735,
    "dna_binding":        0.9109,
    "glycosylation":      0.9598,
    "intramembrane":      0.9703,
    "lcd":                0.9720,
    "lipidation":         0.8279,
    "methylation":        0.8524,
    "motif":              0.7130,
    "nucleotide_binding": 0.9552,
    "phosphorylation":    0.8119,
    "propeptide":         0.9130,
    "repeat":             0.9721,
    "rna_binding":        0.9742,
    "signal_peptide":     0.9997,
    "transit_peptide":    0.9967,
    "ubiquitination":     0.6934,
    "zinc_finger":        0.8994,
}

#: Protein-level true- and false-positive rates of the call rule each head actually uses, measured
#: once on its held-out test proteins: the fraction of proteins carrying the feature that the rule
#: calls, and the fraction of proteins lacking it that the rule calls anyway.  The test sets hold
#: far more feature-carrying proteins than a proteome does, so a positive call on an arbitrary
#: protein is less reliable than the true-positive rate alone suggests.
PROTEIN_CALL_RATES: dict[str, tuple[float, float]] = {
    "acetylation":        (0.7570, 0.3315),
    "active_site":        (0.9615, 0.1658),
    "binding_site":       (0.9399, 0.3810),
    "coiled_coil":        (0.9286, 0.1875),
    "disorder":           (0.7606, 0.1864),
    "disulfide":          (0.9637, 0.2161),
    "dna_binding":        (0.8878, 0.1362),
    "glycosylation":      (0.8881, 0.0310),
    "intramembrane":      (0.9569, 0.0514),
    "lcd":                (0.9261, 0.2178),
    "lipidation":         (0.4378, 0.2112),
    "methylation":        (0.3167, 0.1209),
    "motif":              (0.7138, 0.4159),
    "nucleotide_binding": (0.9814, 0.1758),
    "phosphorylation":    (0.8481, 0.4286),
    "propeptide":         (0.6846, 0.2115),
    "repeat":             (0.9252, 0.0556),
    "rna_binding":        (0.9278, 0.0780),
    "signal_peptide":     (0.9792, 0.0132),
    "transit_peptide":    (0.9872, 0.0366),
    "ubiquitination":     (0.9234, 0.6027),
    "zinc_finger":        (0.8452, 0.1421),
}

#: task → (architecture, LSTM hidden size per direction), as stored in the .npz files.
HEAD_ARCHITECTURE: dict[str, tuple[str, int]] = {
    "acetylation":         ("bilstm2", 256),
    "active_site":         ("bilstm2", 256),
    "binding_site":        ("bilstm2", 384),
    "coiled_coil":         ("bilstm2", 384),
    "disorder":            ("bilstm2_cnn", 256),
    "disulfide":           ("bilstm2", 256),
    "dna_binding":         ("bilstm2", 256),
    "glycosylation":       ("bilstm2", 256),
    "intramembrane":       ("bilstm2", 256),
    "lcd":                 ("bilstm2_cnn", 256),
    "lipidation":          ("bilstm2", 256),
    "methylation":         ("bilstm2", 384),
    "motif":               ("bilstm2_cnn", 256),
    "nucleotide_binding":  ("bilstm2", 384),
    "phosphorylation":     ("bilstm2", 384),
    "propeptide":          ("bilstm2_pos", 256),
    "repeat":              ("bilstm2_cnn", 256),
    "rna_binding":         ("bilstm2_cnn", 256),
    "secondary_structure": ("bilstm2_ss3_cnn", 256),
    "signal_peptide":      ("bilstm2", 256),
    "transit_peptide":     ("bilstm2", 256),
    "transmembrane":       ("bilstm2_crf", 256),
    "ubiquitination":      ("bilstm2", 256),
    "zinc_finger":         ("bilstm2", 256),
}

DEFAULT_THRESHOLD = 0.5

# The Viterbi-decoded transmembrane head still draws a probability curve, so it
# needs a reference line.  The value is a visual aid, not the head's operating
# point (its calls come from the Viterbi path).
_VITERBI_THRESHOLDS: dict[str, float] = {
    "transmembrane": 0.81339,
}

# The three SS3 profiles come from one head; they share its Q3 accuracy.
_ALIASES: dict[str, str] = {
    "secondary_structure_helix":  "secondary_structure",
    "secondary_structure_strand": "secondary_structure",
    "secondary_structure_coil":   "secondary_structure",
}


def _resolve(task: str) -> str:
    return _ALIASES.get(task, task)


def f1_threshold(task: str) -> float:
    """Decision threshold for *task*, on the scale BEER reports, or 0.5 if it has none.

    Heads with a fitted calibration report calibrated probabilities, so their threshold is
    returned on that same scale.  The two always move together, which is what keeps every call
    identical to the uncalibrated model.
    """
    task = _resolve(task)
    metrics = HEAD_METRICS.get(task)
    if metrics is not None and metrics.f1_threshold is not None:
        from beer.models.calibration import calibrated_threshold
        return calibrated_threshold(task, metrics.f1_threshold)
    return _VITERBI_THRESHOLDS.get(task, DEFAULT_THRESHOLD)


def auroc(task: str) -> float | None:
    """Test-set AUROC for *task* (Q3 accuracy for secondary structure)."""
    metrics = HEAD_METRICS.get(_resolve(task))
    return metrics.auroc if metrics else None


def auroc_str(task: str) -> str:
    """AUROC formatted for display; empty string for an unknown task."""
    value = auroc(task)
    return f"{value:.4f}" if value is not None else ""


def auprc(task: str) -> float | None:
    """Test-set AUPRC for *task*; ``None`` for the Viterbi-decoded heads."""
    metrics = HEAD_METRICS.get(_resolve(task))
    return metrics.auprc if metrics else None


def protein_call_rates(task: str) -> tuple[float, float] | None:
    """(true-positive rate, false-positive rate) of *task*'s protein-level rule on test proteins."""
    return PROTEIN_CALL_RATES.get(_resolve(task))


def protein_auroc_val(task: str) -> float | None:
    """Protein-level validation AUROC for *task*, the value the call gate uses."""
    return PROTEIN_AUROC_VAL.get(_resolve(task))


def protein_auroc(task: str) -> float | None:
    """Protein-level test AUROC for *task*; ``None`` where there is none."""
    return PROTEIN_AUROC.get(_resolve(task))


def architecture_note(task: str) -> str:
    """One sentence describing *task*'s network, for the report notes."""
    arch, hidden = HEAD_ARCHITECTURE.get(_resolve(task), ("bilstm2", 256))
    parts = ["ESM C 600M embeddings (1152-dim, frozen)"]
    if arch in ("bilstm2_cnn", "bilstm2_ss3_cnn"):
        parts.append("multi-scale 1-D CNN (kernel widths 3, 7 and 15; 64 filters each)")
    if arch == "bilstm2_pos":
        parts[0] += " plus a learned positional embedding"
    parts.append(f"2-layer bidirectional LSTM ({hidden} units per direction)")
    parts.append({"bilstm2_crf": "3-state CRF with Viterbi decoding (outside, membrane, inside)",
                  "bilstm2_ss3": "3-class softmax (helix, strand, coil)"}.get(arch, "linear layer and sigmoid"))
    return (" → ".join(parts) + ". Trained with focal loss on MMseqs2-clustered splits (30% "
            "identity); the decision threshold maximises F1 on the validation split.")
