"""MSA mutual information covariance (CoeViz-style MI with APC correction).

Reference
---------
Dunn, S.D., Wahl, L.M. & Gloor, G.B. (2008) Bioinformatics 24, 333–340.
"""
from __future__ import annotations
import math
from collections import Counter

_AAS = frozenset("ACDEFGHIKLMNPQRSTVWY")
_MAX_COLS = 500   # hard limit to keep computation tractable


def calc_msa_mutual_information(
    aligned_seqs: list[str],
) -> tuple[list[list[float]], list[list[float]]]:
    """Pairwise mutual information with Average Product Correction (APC).

    Parameters
    ----------
    aligned_seqs:
        List of gap-containing aligned sequences of equal length ('-' for gaps).
        At least 4 sequences are required for a meaningful result.

    Returns
    -------
    (mi_raw, mi_apc):
        Both are *n_col × n_col* 2-D lists of floats (bits).
        *mi_apc* has the APC background removed; use this for display.
        Returns zero matrices for inputs that are too small or exceed the
        column limit.

    Notes
    -----
    The Average Product Correction removes the systematic background signal
    that arises from differential column entropy and phylogenetic effects
    (Dunn et al. 2008).  High MI-APC scores indicate columns that covary
    beyond what is expected from their individual entropies — a signature of
    functional or structural coupling.
    """
    n_seq = len(aligned_seqs)
    n_col = len(aligned_seqs[0]) if aligned_seqs else 0

    zero = [[0.0] * n_col for _ in range(n_col)]

    if n_seq < 4 or n_col < 2 or n_col > _MAX_COLS:
        return zero, [row[:] for row in zero]

    # --- extract columns (list-of-lists, gaps kept for indexing) ---
    cols: list[list[str]] = [[s[j] if j < len(s) else "-" for s in aligned_seqs]
                              for j in range(n_col)]

    # --- per-column marginal Shannon entropy (gaps excluded) ---
    def _H(counter: Counter, total: int) -> float:
        if total == 0:
            return 0.0
        return -sum((c / total) * math.log2(c / total) for c in counter.values() if c > 0)

    H: list[float] = []
    col_counts: list[tuple[Counter, int]] = []
    for col in cols:
        cnt = Counter(aa for aa in col if aa in _AAS)
        tot = sum(cnt.values())
        H.append(_H(cnt, tot))
        col_counts.append((cnt, tot))

    # --- pairwise MI ---
    mi_raw: list[list[float]] = [[0.0] * n_col for _ in range(n_col)]
    for i in range(n_col):
        if H[i] == 0.0:
            continue
        for j in range(i + 1, n_col):
            if H[j] == 0.0:
                continue
            joint: Counter = Counter()
            total = 0
            for a, b in zip(cols[i], cols[j]):
                if a in _AAS and b in _AAS:
                    joint[(a, b)] += 1
                    total += 1
            if total < 4:
                continue
            H_ij = _H(joint, total)
            val = max(0.0, H[i] + H[j] - H_ij)
            mi_raw[i][j] = val
            mi_raw[j][i] = val

    # --- APC: MI_APC(i,j) = MI(i,j) − mean_row_i × mean_row_j / mean_total ---
    n = n_col
    # row mean excludes self-comparison (diagonal is 0)
    row_means = [sum(mi_raw[i]) / (n - 1) for i in range(n)]
    # mean over all off-diagonal elements (each counted twice in the sum)
    flat_sum = sum(sum(row) for row in mi_raw)
    total_mean = flat_sum / (n * (n - 1)) if n > 1 else 0.0

    mi_apc: list[list[float]] = [[0.0] * n_col for _ in range(n_col)]
    if total_mean > 0.0:
        for i in range(n_col):
            for j in range(i + 1, n_col):
                apc = row_means[i] * row_means[j] / total_mean
                val = max(0.0, mi_raw[i][j] - apc)
                mi_apc[i][j] = val
                mi_apc[j][i] = val

    return mi_raw, mi_apc
