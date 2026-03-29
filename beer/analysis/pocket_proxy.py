"""Sequence-based binding pocket proxy scorer.

Uses a sliding window to score each residue's potential to be part of a
binding pocket, based on:
  - Local amphipathicity (hydrophobic + charged neighbours)
  - Low local complexity (pocket residues tend to have defined structure)
  - High local conservation proxy (uncommon amino acid combinations)
  - Burial propensity (ASA proxy using Janin scale)

This is NOT a substitute for structure-based pocket detection (e.g.,
fpocket, SiteMap).  It is a fast sequence-level proxy for annotating
potentially functional surface patches.

Reference: Loosely inspired by ConCavity (Capra et al. 2009) sequence
features, without phylogenetic conservation.
"""
from __future__ import annotations
import numpy as np

# Janin burial scale (1979): higher = more buried
_BURIAL = {
    "A": 0.30, "C": 0.90, "D": 0.10, "E": 0.10, "F": 0.60,
    "G": 0.20, "H": 0.40, "I": 0.60, "K": 0.10, "L": 0.45,
    "M": 0.40, "N": 0.10, "P": 0.20, "Q": 0.10, "R": 0.10,
    "S": 0.20, "T": 0.20, "V": 0.50, "W": 0.55, "Y": 0.35,
}

# Kyte-Doolittle hydrophobicity
_KD = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}

_CHARGED  = set("DEKRH")
_AROMATIC = set("FWY")


def calc_pocket_proxy_score(seq: str, window: int = 9) -> np.ndarray:
    """Return per-residue pocket proxy score (0-1, higher = more pocket-like).

    Parameters
    ----------
    seq:    Protein sequence.
    window: Sliding window radius (default 9 → 9-residue windows).
    """
    L = len(seq)
    raw = np.zeros(L)
    half = window // 2

    for i in range(L):
        lo = max(0, i - half)
        hi = min(L, i + half + 1)
        chunk = seq[lo:hi]

        # 1. Burial propensity of central residue
        burial = _BURIAL.get(seq[i], 0.2)

        # 2. Local amphipathicity: mix of hydrophobic + charged in window
        n_hphob  = sum(1 for aa in chunk if _KD.get(aa, 0) > 1.5)
        n_charge = sum(1 for aa in chunk if aa in _CHARGED)
        amphip = min(1.0, (n_hphob + n_charge) / max(len(chunk), 1))

        # 3. Aromatic contribution (active-site proxy)
        n_arom = sum(1 for aa in chunk if aa in _AROMATIC) / max(len(chunk), 1)

        # 4. Diversity of window (Shannon entropy proxy)
        from collections import Counter
        counts = Counter(chunk)
        total  = len(chunk)
        entropy = -sum((c / total) * np.log2(c / total) for c in counts.values())
        norm_entropy = entropy / np.log2(max(total, 2))

        # Combine: weighted sum
        score = (0.35 * burial + 0.30 * amphip + 0.20 * n_arom + 0.15 * norm_entropy)
        raw[i] = score

    # Normalise to [0, 1]
    mn, mx = raw.min(), raw.max()
    if mx > mn:
        raw = (raw - mn) / (mx - mn)
    return raw


def find_pocket_regions(score: np.ndarray, threshold: float = 0.65,
                        min_len: int = 5) -> list[tuple[int, int]]:
    """Return list of (start, end) 0-based regions above threshold."""
    regions = []
    in_region = False
    start = 0
    for i, s in enumerate(score):
        if s >= threshold and not in_region:
            in_region = True
            start = i
        elif s < threshold and in_region:
            in_region = False
            if i - start >= min_len:
                regions.append((start, i - 1))
    if in_region and len(score) - start >= min_len:
        regions.append((start, len(score) - 1))
    return regions
