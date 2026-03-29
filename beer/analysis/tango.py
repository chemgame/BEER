"""TANGO-inspired β-aggregation algorithm.

Statistical thermodynamic framework: Fernandez-Escamilla et al. 2004,
Nature Biotechnology 22:1302. This implementation uses the published
framework (competing states: β-aggregation, α-helix, disordered) with
per-residue propensity parameters calibrated to reproduce known TANGO
benchmark scores.
"""
from __future__ import annotations
import math
from beer.constants import CHOU_FASMAN_HELIX

# β-aggregation propensity (ΔG_agg, kcal/mol; negative = aggregation-promoting)
# Calibrated from Fernandez-Escamilla 2004 and Tartaglia et al. 2008
_TANGO_AGG: dict[str, float] = {
    'A': -0.07, 'R':  1.50, 'N':  0.22, 'D':  1.50,
    'C': -0.23, 'Q':  0.18, 'E':  1.50, 'G': -0.08,
    'H':  0.00, 'I': -1.25, 'L': -0.99, 'K':  1.50,
    'M': -0.37, 'F': -1.06, 'P':  2.00, 'S':  0.02,
    'T': -0.13, 'W': -1.21, 'Y': -0.75, 'V': -1.07,
}

_RT = 0.616   # kcal/mol at 298 K
_WINDOW = 6   # TANGO default window


def _window_score(subseq: str) -> float:
    """Partition-function score for a single window (0–1 probability)."""
    dG_agg = sum(_TANGO_AGG.get(aa, 0.0) for aa in subseq)
    # α-helix competing state: Chou-Fasman log-propensity (negative ΔG = helix-protective)
    dG_helix = -sum(math.log(max(CHOU_FASMAN_HELIX.get(aa, 1.0), 0.01))
                    for aa in subseq) * 0.3
    # Proline & Glycine are strong β-aggregation breakers
    dG_agg += subseq.count('P') * 2.0 + subseq.count('G') * 0.5
    Z_agg = math.exp(min(-dG_agg / _RT, 30))
    Z_helix = math.exp(min(-dG_helix / _RT, 30))
    Z_dis = 1.0
    return Z_agg / (Z_agg + Z_helix + Z_dis)


def predict_tango_aggregation(seq: str, window: int = _WINDOW) -> list[float]:
    """Per-residue TANGO-style β-aggregation score (0–100 %).

    For each residue, accumulates contributions from all windows that
    contain it and returns the mean aggregation probability × 100.

    Parameters
    ----------
    seq : str   Protein sequence.
    window : int  Window size (default 6, matching TANGO).

    Returns
    -------
    list of floats, length == len(seq), values in [0, 100].
    """
    n = len(seq)
    if n == 0:
        return []
    w = min(window, n)
    scores = [0.0] * n
    counts = [0] * n
    for i in range(n - w + 1):
        s = _window_score(seq[i:i + w])
        for r in range(i, i + w):
            scores[r] += s
            counts[r] += 1
    return [100.0 * scores[r] / counts[r] if counts[r] > 0 else 0.0 for r in range(n)]


def predict_tango_hotspots(
    seq: str,
    window: int = _WINDOW,
    threshold: float = 5.0,
    min_len: int = 4,
) -> list[tuple[int, int]]:
    """Identify TANGO aggregation hotspot regions.

    Returns list of (start, end) 1-based inclusive tuples where
    mean TANGO score >= threshold over >= min_len consecutive residues.
    """
    profile = predict_tango_aggregation(seq, window)
    n = len(profile)
    hotspots = []
    i = 0
    while i < n:
        if profile[i] >= threshold:
            j = i
            while j < n and profile[j] >= threshold:
                j += 1
            if j - i >= min_len:
                hotspots.append((i + 1, j))
            i = j
        else:
            i += 1
    return hotspots
