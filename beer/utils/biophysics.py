"""Biophysical computation utilities."""
from __future__ import annotations
import math
from math import log2

from beer.constants import (
    KYTE_DOOLITTLE,
    DEFAULT_PKA,
    STICKER_ALL,
)


def calc_net_charge(seq: str, pH: float = 7.0, pka: dict = None) -> float:
    """Henderson-Hasselbalch net charge."""
    p = pka or DEFAULT_PKA
    net = 1 / (1 + 10 ** (pH - p['NTERM'])) - 1 / (1 + 10 ** (p['CTERM'] - pH))
    for aa in seq:
        if aa in ('D', 'E', 'C', 'Y'):
            net -= 1 / (1 + 10 ** (p[aa] - pH))
        elif aa in ('K', 'R', 'H'):
            net += 1 / (1 + 10 ** (pH - p[aa]))
    return net


def sliding_window_hydrophobicity(seq: str, window_size: int = 9) -> list:
    """Kyte-Doolittle sliding window average."""
    if window_size > len(seq):
        return [sum(KYTE_DOOLITTLE[aa] for aa in seq) / len(seq)]
    return [
        sum(KYTE_DOOLITTLE[aa] for aa in seq[i:i + window_size]) / window_size
        for i in range(len(seq) - window_size + 1)
    ]


def calc_shannon_entropy(seq: str) -> float:
    """Sequence compositional entropy in bits. Max = log2(20) ≈ 4.32."""
    n = len(seq)
    counts = {}
    for aa in seq:
        counts[aa] = counts.get(aa, 0) + 1
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def sliding_window_ncpr(seq: str, window_size: int = 9) -> list:
    """Net charge per residue in a sliding window (K,R positive; D,E negative)."""
    pos = set("KR")
    neg = set("DE")
    if window_size > len(seq):
        p = sum(1 for aa in seq if aa in pos)
        n = sum(1 for aa in seq if aa in neg)
        return [(p - n) / len(seq)]
    return [
        (sum(1 for aa in seq[i:i + window_size] if aa in pos) -
         sum(1 for aa in seq[i:i + window_size] if aa in neg)) / window_size
        for i in range(len(seq) - window_size + 1)
    ]


def sliding_window_entropy(seq: str, window_size: int = 9) -> list:
    """Shannon entropy in a sliding window."""
    if window_size > len(seq):
        return [calc_shannon_entropy(seq)]
    return [
        calc_shannon_entropy(seq[i:i + window_size])
        for i in range(len(seq) - window_size + 1)
    ]


def calc_kappa(seq: str) -> float:
    """Charge patterning parameter (Das & Pappu 2013). Range [0, 1].
    0 = well-mixed charges, 1 = fully segregated."""
    pos_aa = set("KR")
    neg_aa = set("DE")
    blob_sz = 5
    pos_n = sum(1 for aa in seq if aa in pos_aa)
    neg_n = sum(1 for aa in seq if aa in neg_aa)
    if pos_n == 0 or neg_n == 0:
        return 0.0
    n_blobs = len(seq) // blob_sz
    if n_blobs < 2:
        return 0.0
    fcr_pos = pos_n / len(seq)
    fcr_neg = neg_n / len(seq)

    def _delta(s):
        nb = len(s) // blob_sz
        if nb == 0:
            return 0.0
        total = 0.0
        for i in range(nb):
            bl = s[i * blob_sz:(i + 1) * blob_sz]
            fp = sum(1 for a in bl if a in pos_aa) / len(bl)
            fn = sum(1 for a in bl if a in neg_aa) / len(bl)
            total += (fp - fcr_pos) ** 2 + (fn - fcr_neg) ** 2
        return total / nb

    delta = _delta(seq)
    neutral_n = len(seq) - pos_n - neg_n
    seg1 = 'K' * pos_n + 'D' * neg_n + 'G' * neutral_n
    seg2 = 'D' * neg_n + 'K' * pos_n + 'G' * neutral_n
    delta_max = max(_delta(seg1), _delta(seg2))
    return 0.0 if delta_max == 0 else min(1.0, delta / delta_max)


def calc_omega(seq: str) -> float:
    """Patterning of sticker residues (FWYKRDE) vs spacers (Das et al. 2015).
    Range [0, 1]. 0 = evenly distributed, 1 = fully clustered."""
    blob_sz = 5
    sticker_n = sum(1 for aa in seq if aa in STICKER_ALL)
    if sticker_n == 0 or sticker_n == len(seq):
        return 0.0
    n_blobs = len(seq) // blob_sz
    if n_blobs < 2:
        return 0.0
    f_stick = sticker_n / len(seq)

    def _delta(s):
        nb = len(s) // blob_sz
        if nb == 0:
            return 0.0
        total = 0.0
        for i in range(nb):
            bl = s[i * blob_sz:(i + 1) * blob_sz]
            fs = sum(1 for a in bl if a in STICKER_ALL) / len(bl)
            total += (fs - f_stick) ** 2
        return total / nb

    delta = _delta(seq)
    spacer_n = len(seq) - sticker_n
    seg1 = 'F' * sticker_n + 'G' * spacer_n
    seg2 = 'G' * spacer_n + 'F' * sticker_n
    delta_max = max(_delta(seg1), _delta(seg2))
    return 0.0 if delta_max == 0 else min(1.0, delta / delta_max)


def count_pairs(seq: str, set_a: set, set_b: set, window: int = 4) -> int:
    """Count unique (i,j) residue pairs where i in set_a, j in set_b, |i-j| <= window."""
    n = len(seq)
    pairs = set()
    for i in range(n):
        if seq[i] in set_a:
            for j in range(max(0, i - window), min(n, i + window + 1)):
                if j != i and seq[j] in set_b:
                    pairs.add((min(i, j), max(i, j)))
    return len(pairs)


def fraction_low_complexity(seq: str, window_size: int = 12,
                            threshold: float = 2.0) -> float:
    """Fraction of residues covered by at least one window with entropy < threshold."""
    if len(seq) < window_size:
        return 1.0 if calc_shannon_entropy(seq) < threshold else 0.0
    covered = [False] * len(seq)
    for i in range(len(seq) - window_size + 1):
        if calc_shannon_entropy(seq[i:i + window_size]) < threshold:
            for j in range(i, i + window_size):
                covered[j] = True
    return sum(covered) / len(seq)


def sticker_spacing_stats(seq: str) -> dict:
    """Return mean/min/max residue spacing between consecutive sticker residues."""
    positions = [i for i, aa in enumerate(seq) if aa in STICKER_ALL]
    if len(positions) < 2:
        return {"mean": None, "min": None, "max": None}
    gaps = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
    return {
        "mean": sum(gaps) / len(gaps),
        "min":  min(gaps),
        "max":  max(gaps),
    }
