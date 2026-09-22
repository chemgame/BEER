"""Protein-level calls derived from residue-level head outputs.

A residue-level probability profile answers "how likely is this residue to be X"; a report
sentence such as "signal peptide detected" answers a protein-level question. These helpers
make that step explicit and use each head's calibrated operating point.
"""
from __future__ import annotations

from typing import NamedTuple

from beer.models.metrics import f1_threshold, protein_auroc

# A signal peptide starts at the N-terminus; allow the first residues to sit below the
# threshold (initiator Met, n-region) before the above-threshold run begins.
SP_MAX_START = 3
# Shortest run of above-threshold residues accepted as a signal peptide.
SP_MIN_LENGTH = 10
# Only look for the run within the N-terminal window where signal peptides occur.
SP_WINDOW = 70

# Shortest Viterbi membrane segment reported as a transmembrane segment, chosen on the
# validation split. It sits below the ~20 residues a helix needs to span the bilayer because
# UniProt 'Transmembrane' annotations also cover beta-barrel strands and 13-17 residue helices.
TM_MIN_SEGMENT = 12
TM_STATE = 1   # state index of the membrane segment in the transmembrane CRF head


def signal_peptide_call(profile: "list[float] | None",
                        threshold: float | None = None) -> tuple[bool, int]:
    """Return (called, end) for an N-terminal signal peptide.

    *end* is the 1-based last residue of the above-threshold run (0 when not called).
    The call needs a contiguous run of at least ``SP_MIN_LENGTH`` residues above the head's
    F1-max threshold that starts within the first ``SP_MAX_START`` residues.
    """
    if not profile:
        return False, 0
    thr = f1_threshold("signal_peptide") if threshold is None else threshold
    window = list(profile[:SP_WINDOW])
    start = next((i for i in range(min(SP_MAX_START, len(window))) if window[i] > thr), None)
    if start is None:
        return False, 0
    end = start
    while end < len(window) and window[end] > thr:
        end += 1
    if end - start < SP_MIN_LENGTH:
        return False, 0
    return True, end


def tm_segments(topology: "list[int] | None",
                min_length: int = TM_MIN_SEGMENT,
                tm_state: int = TM_STATE) -> list[tuple[int, int]]:
    """Transmembrane segments from a Viterbi topology, as 1-based inclusive ranges.

    Only segments of at least *min_length* residues in the membrane state are returned.
    """
    if not topology:
        return []
    segments, start = [], None
    for i, state in enumerate(list(topology) + [None]):
        if state == tm_state and start is None:
            start = i
        elif state != tm_state and start is not None:
            if i - start >= min_length:
                segments.append((start + 1, i))
            start = None
    return segments


# ── Every other thresholded head ─────────────────────────────────────────────────────────
# Heads trained on single-residue annotations (the median positive run in their training
# labels is one residue): a protein carries the feature when any residue is called.
SITE_HEADS = frozenset({
    "acetylation", "active_site", "binding_site", "disulfide", "dna_binding",
    "glycosylation", "lipidation", "methylation", "nucleotide_binding",
    "phosphorylation", "rna_binding", "ubiquitination", "zinc_finger",
})


class ProteinRule(NamedTuple):
    """A protein carries the feature when it has a run of at least *min_run* consecutive
    residues above the head's threshold, starting within the first *max_start* residues
    when that is set (N-terminal targeting sequences)."""
    min_run: int
    max_start: int | None = None


#: Each minimum run maximises the protein-level MCC on the head's validation split.
PROTEIN_RULES: dict[str, ProteinRule] = {
    "acetylation":        ProteinRule(1),
    "active_site":        ProteinRule(1),
    "binding_site":       ProteinRule(1),
    "coiled_coil":        ProteinRule(8),
    "disorder":           ProteinRule(10),
    "disulfide":          ProteinRule(1),
    "dna_binding":        ProteinRule(2),
    "glycosylation":      ProteinRule(1),
    "intramembrane":      ProteinRule(10),
    "lcd":                ProteinRule(5),
    "lipidation":         ProteinRule(1),
    "methylation":        ProteinRule(1),
    "motif":              ProteinRule(2),
    "nucleotide_binding": ProteinRule(1),
    "phosphorylation":    ProteinRule(1),
    "propeptide":         ProteinRule(1),
    "repeat":             ProteinRule(1),
    "rna_binding":        ProteinRule(1),
    "transit_peptide":    ProteinRule(2, SP_MAX_START),
    "ubiquitination":     ProteinRule(1),
    "zinc_finger":        ProteinRule(1),
}

# Below this protein-level AUROC (v3.1 test split) a head separates proteins with and
# without the feature too weakly for a yes/no statement; only its residue scores are shown.
MIN_PROTEIN_AUROC = 0.80


class ProteinCall(NamedTuple):
    """*called* is None when the head makes no protein-level call (see MIN_PROTEIN_AUROC)."""
    called: bool | None
    spans: list[tuple[int, int]]      # 1-based inclusive runs that satisfy the rule


def runs_above(profile: "list[float] | None", threshold: float) -> list[tuple[int, int]]:
    """Maximal runs of residues scoring above *threshold*, as 1-based inclusive ranges."""
    runs, start = [], None
    for i, v in enumerate(list(profile or []) + [float("-inf")]):
        if v > threshold and start is None:
            start = i
        elif v <= threshold and start is not None:
            runs.append((start + 1, i))
            start = None
    return runs


def rule_text(task: str) -> str:
    """The protein-level rule for *task* in words."""
    if task == "signal_peptide":
        return (f"a run of at least {SP_MIN_LENGTH} residues above the threshold starting "
                f"within the first {SP_MAX_START} residues")
    rule = PROTEIN_RULES.get(task)
    if rule is None:
        return ""
    if rule.min_run == 1:
        text = "at least one residue above the threshold"
    else:
        text = f"a run of at least {rule.min_run} consecutive residues above the threshold"
    if rule.max_start:
        text += f" starting within the first {rule.max_start} residues"
    return text


def protein_call(task: str, profile: "list[float] | None") -> ProteinCall:
    """Protein-level call for a thresholded head from its residue profile."""
    if task == "signal_peptide":
        called, end = signal_peptide_call(profile)
        return ProteinCall(called, [(1, end)] if called else [])
    rule = PROTEIN_RULES.get(task)
    if rule is None or not profile:
        return ProteinCall(None, [])
    spans = [(s, e) for s, e in runs_above(profile, f1_threshold(task))
             if e - s + 1 >= rule.min_run and (rule.max_start is None or s <= rule.max_start)]
    if (protein_auroc(task) or 0.0) < MIN_PROTEIN_AUROC:
        return ProteinCall(None, spans)
    return ProteinCall(bool(spans), spans)
