"""Position-specific scoring matrix (PSSM) from an aligned MSA.

A PSSM gives, for every alignment column, a log-odds score per amino acid:

    score(a, j) = log2( (f(a, j) + pseudo) / p(a) )

where f(a, j) is the observed frequency of residue *a* in column *j*
(gap-excluded), p(a) is the background amino-acid frequency, and *pseudo*
is a small Laplace pseudocount that keeps unseen residues finite.

Reference
---------
Background composition: UniProtKB/Swiss-Prot average amino-acid frequencies.
"""
from __future__ import annotations
import math
from collections import Counter

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")

# UniProtKB/Swiss-Prot average amino-acid composition (background p(a)).
_BACKGROUND = {
    "A": 0.0825, "R": 0.0553, "N": 0.0406, "D": 0.0546, "C": 0.0138,
    "Q": 0.0393, "E": 0.0672, "G": 0.0707, "H": 0.0227, "I": 0.0591,
    "L": 0.0965, "K": 0.0580, "M": 0.0241, "F": 0.0386, "P": 0.0473,
    "S": 0.0660, "T": 0.0535, "W": 0.0110, "Y": 0.0292, "V": 0.0686,
}


def compute_pssm(
    aligned_seqs: list[str],
    pseudocount: float = 0.05,
) -> tuple[list[dict], list[float], list[int]]:
    """Build a PSSM from gap-containing aligned sequences of equal length.

    Parameters
    ----------
    aligned_seqs:
        Aligned sequences ('-' for gaps); all must be the same length.
    pseudocount:
        Laplace smoothing added to every per-column frequency.

    Returns
    -------
    (rows, conservation, coverage):
        rows         : per-column list of {aa: log2 odds score} dicts.
        conservation : per-column Shannon information content (bits, 0–~4.3).
        coverage     : per-column count of non-gap residues.
    """
    if not aligned_seqs:
        return [], [], []
    ncol = len(aligned_seqs[0])
    if any(len(s) != ncol for s in aligned_seqs):
        raise ValueError("All aligned sequences must have equal length.")

    rows: list[dict] = []
    conservation: list[float] = []
    coverage: list[int] = []

    for j in range(ncol):
        col = [s[j].upper() for s in aligned_seqs]
        counts = Counter(c for c in col if c in _BACKGROUND)
        n = sum(counts.values())
        coverage.append(n)

        scores: dict[str, float] = {}
        denom = n + pseudocount * 20
        for a in AA_ORDER:
            f = (counts.get(a, 0) + pseudocount) / denom if denom else pseudocount / (pseudocount * 20)
            scores[a] = math.log2(f / _BACKGROUND[a])
        rows.append(scores)
        # Relative-entropy conservation (Kullback–Leibler vs. background).
        kl = 0.0
        for a in AA_ORDER:
            f = counts.get(a, 0) / n if n else 0.0
            if f > 0:
                kl += f * math.log2(f / _BACKGROUND[a])
        conservation.append(max(kl, 0.0))

    return rows, conservation, coverage


def pssm_to_csv(
    rows: list[dict],
    conservation: list[float],
    coverage: list[int],
    consensus: str | None = None,
) -> str:
    """Serialise a PSSM to CSV text (header + one row per alignment column)."""
    header = ["position", "consensus", "coverage", "conservation_bits"] + AA_ORDER
    lines = [",".join(header)]
    for j, scores in enumerate(rows):
        cons = consensus[j] if consensus and j < len(consensus) else ""
        vals = [f"{scores[a]:.3f}" for a in AA_ORDER]
        lines.append(",".join(
            [str(j + 1), cons, str(coverage[j]), f"{conservation[j]:.3f}"] + vals))
    return "\n".join(lines) + "\n"


def consensus_sequence(aligned_seqs: list[str]) -> str:
    """Most-frequent non-gap residue per column ('-' if a column is all gaps)."""
    if not aligned_seqs:
        return ""
    ncol = len(aligned_seqs[0])
    out = []
    for j in range(ncol):
        counts = Counter(s[j].upper() for s in aligned_seqs if s[j] in _BACKGROUND)
        out.append(counts.most_common(1)[0][0] if counts else "-")
    return "".join(out)
