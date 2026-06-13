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
    # Merge over defaults so a partial custom pKa dict (missing some residues)
    # doesn't raise KeyError — user-supplied values override, rest fall back.
    p = DEFAULT_PKA if not pka else {**DEFAULT_PKA, **pka}
    net = 1 / (1 + 10 ** (pH - p['NTERM'])) - 1 / (1 + 10 ** (p['CTERM'] - pH))
    for aa in seq:
        if aa in ('D', 'E', 'C', 'Y'):
            net -= 1 / (1 + 10 ** (p[aa] - pH))
        elif aa in ('K', 'R', 'H'):
            net += 1 / (1 + 10 ** (pH - p[aa]))
    return net


def calc_isoelectric_point(seq: str, pka: dict = None) -> float:
    """Isoelectric point consistent with calc_net_charge.

    Finds the pH in [0.0, 14.0] where Henderson-Hasselbalch net charge crosses
    zero via bisection (tolerance 1e-4).  Because net charge is strictly
    monotonically decreasing with pH, standard bisection is guaranteed to
    converge.

    Parameters
    ----------
    seq:
        Amino-acid sequence (single-letter codes, uppercase).
    pka:
        pKa table.  Defaults to ``DEFAULT_PKA`` (same default as
        ``calc_net_charge``), ensuring the two functions are always consistent.

    Returns
    -------
    float
        pH at which net charge == 0 (isoelectric point).
    """
    p = pka or DEFAULT_PKA
    lo, hi = 0.0, 14.0
    while (hi - lo) > 1e-4:
        mid = (lo + hi) / 2.0
        if calc_net_charge(seq, mid, p) > 0.0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def sliding_window_hydrophobicity(seq: str, window_size: int = 9, scale: dict = None) -> list:
    """Sliding-window hydrophobicity average.

    Parameters
    ----------
    scale : dict, optional
        Per-residue hydrophobicity values. Defaults to Kyte-Doolittle.
    """
    _scale = scale if scale is not None else KYTE_DOOLITTLE
    n = len(seq)
    if n == 0:
        return []
    if window_size > n:
        avg = sum(_scale.get(aa, 0.0) for aa in seq) / n
        return [avg] * n
    return [
        sum(_scale.get(aa, 0.0) for aa in seq[i:i + window_size]) / window_size
        for i in range(n - window_size + 1)
    ]


def calc_shannon_entropy(seq: str) -> float:
    """Sequence compositional entropy in bits. Max = log2(20) ≈ 4.32."""
    n = len(seq)
    if n == 0:
        return 0.0
    counts = {}
    for aa in seq:
        counts[aa] = counts.get(aa, 0) + 1
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def sliding_window_ncpr(seq: str, window_size: int = 9) -> list:
    """Net charge per residue in a sliding window (K,R positive; D,E negative)."""
    pos = set("KR")
    neg = set("DE")
    n = len(seq)
    if n == 0:
        return []
    if window_size > n:
        p = sum(1 for aa in seq if aa in pos)
        m = sum(1 for aa in seq if aa in neg)
        avg = (p - m) / n
        return [avg] * n
    return [
        (sum(1 for aa in seq[i:i + window_size] if aa in pos) -
         sum(1 for aa in seq[i:i + window_size] if aa in neg)) / window_size
        for i in range(n - window_size + 1)
    ]


def sliding_window_entropy(seq: str, window_size: int = 9) -> list:
    """Shannon entropy in a sliding window."""
    n = len(seq)
    if n == 0:
        return []
    if window_size > n:
        val = calc_shannon_entropy(seq)
        return [val] * n
    return [
        calc_shannon_entropy(seq[i:i + window_size])
        for i in range(n - window_size + 1)
    ]


def calc_kappa(seq: str) -> float:
    """Charge patterning parameter κ (Das & Pappu 2013 PNAS 110:13392). Range [0, 1].
    0 = well-mixed charges, 1 = maximally charge-segregated.

    Matches the localCIDER reference implementation exactly:
      σ = NCPR² / FCR  (sequence level and per sliding blob)
      δ = mean over sliding blobs of (σ_seq − σ_blob)²
      averaged over blob sizes 5 and 6 (optimal per the paper)
      κ = δ / δ_max  where δ_max is found by exhaustive neutral-placement search
    Returns 0.0 when FCR == 0 (no charged residues).
    """
    pos_aa = set("KR")
    neg_aa = set("DE")
    n = len(seq)
    charges = [1 if a in pos_aa else (-1 if a in neg_aa else 0) for a in seq]
    pos_n = charges.count(1)
    neg_n = charges.count(-1)
    neut_n = n - pos_n - neg_n
    fcr = (pos_n + neg_n) / n
    if fcr == 0.0:
        return 0.0
    ncpr = (pos_n - neg_n) / n
    sigma_seq = ncpr * ncpr / fcr

    def _delta(ch: list) -> float:
        N = len(ch)
        def _form(bsz: int) -> float:
            nb = N - bsz + 1
            if nb <= 0:
                return 0.0
            total = 0.0
            for i in range(nb):
                blob = ch[i:i + bsz]
                bp = sum(1 for c in blob if c > 0)
                bn = sum(1 for c in blob if c < 0)
                bfcr = (bp + bn) / bsz
                bncpr = (bp - bn) / bsz
                bsig = bncpr * bncpr / bfcr if bfcr > 0 else 0.0
                total += (sigma_seq - bsig) ** 2
            return total / nb
        return (_form(5) + _form(6)) / 2.0

    delta = _delta(charges)

    P = [1] * pos_n
    M = [-1] * neg_n
    dmax = 0.0

    if pos_n == 0 or neg_n == 0:
        # Single charge type: slide charge block through neutral positions.
        cb = P if pos_n > 0 else M
        nch = len(cb)
        for start in range(n - nch + 1):
            d = _delta([0] * start + cb + [0] * (n - nch - start))
            if d > dmax:
                dmax = d
    elif neut_n == 0:
        # No neutrals: slide the smaller charge block through the larger.
        if pos_n >= neg_n:
            for start in range(n - neg_n + 1):
                d = _delta([1] * start + M + [1] * (pos_n - start))
                if d > dmax:
                    dmax = d
        else:
            for start in range(n - pos_n + 1):
                d = _delta([-1] * start + P + [-1] * (neg_n - start))
                if d > dmax:
                    dmax = d
    elif neut_n >= 18:
        # Heuristic (localCIDER): try placing 0-6 neutrals at start/middle/end.
        # Both P-before-M and M-before-P arrangements to find true δ_max.
        cap = min(7, neut_n + 1)
        for sn in range(cap):
            for en in range(min(cap, neut_n - sn + 1)):
                mn = neut_n - sn - en
                for arrangement in ([0]*sn + P + [0]*mn + M + [0]*en,
                                    [0]*sn + M + [0]*mn + P + [0]*en):
                    d = _delta(arrangement)
                    if d > dmax:
                        dmax = d
    else:
        # Full exhaustive search over all ways to split neutrals into start/mid/end.
        # Both P-before-M and M-before-P arrangements.
        for mn in range(neut_n + 1):
            for sn in range(neut_n - mn + 1):
                en = neut_n - sn - mn
                for arrangement in ([0]*sn + P + [0]*mn + M + [0]*en,
                                    [0]*sn + M + [0]*mn + P + [0]*en):
                    d = _delta(arrangement)
                    if d > dmax:
                        dmax = d

    return 0.0 if dmax == 0.0 else min(1.0, delta / dmax)


def calc_omega(seq: str) -> float:
    """Sticker patterning index: clustering of sticker residues (F,W,Y,K,R,D,E) relative to spacers.
    Range [0, 1]. 0 = evenly distributed, 1 = fully clustered.

    Uses the blob-based delta formalism (blob size 5) from Das & Pappu (2013)
    PNAS 110:13392, applied to sticker residues (F,W,Y,K,R,D,E) instead of
    charged residues. Note: this is NOT the Omega metric of Das et al. (2015)
    PNAS 112:8183, which is defined over Pro + charged residues.
    """
    blob_sz = 5  # blob-based delta formalism: Das & Pappu (2013) PNAS 110:13392
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
    """Fraction of residues covered by at least one window with Shannon entropy < threshold.

    The default threshold of 2.0 bits and window of 12 residues are heuristic
    choices not from a specific publication. They are used here only as a
    summary statistic complementing the Shannon entropy value reported directly.
    For rigorous low-complexity analysis use SEG (Wootton & Federhen 1993,
    Comput. Chem. 17:149) or the Shannon entropy profile.
    """
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


def calc_polyx_stretches(seq: str, min_length: int = 4) -> list:
    """Detect homopolymeric runs of identical residues.

    Returns a list of dicts sorted by length descending, then start ascending.
    Each dict: {aa, start_1based, end_1based, length}.
    """
    stretches = []
    n = len(seq)
    i = 0
    while i < n:
        j = i + 1
        while j < n and seq[j] == seq[i]:
            j += 1
        run_len = j - i
        if run_len >= min_length:
            stretches.append({
                "aa": seq[i],
                "start_1based": i + 1,
                "end_1based": j,
                "length": run_len,
            })
        i = j
    stretches.sort(key=lambda x: (-x["length"], x["start_1based"]))
    return stretches


def calc_plaac_score(seq: str, window: int = 41) -> dict:
    """Per-residue PLAAC log-odds score (Lancaster et al. 2014).

    Uses yeast prion-like domain (FG domain) frequencies vs SwissProt background.
    Returns {profile, max_score, mean_score}.

    Interpretation: regions with consistently positive scores are candidate
    prion-like domains; refer to Lancaster et al. (2014) for interpretation.
    No hard threshold is applied here — binary classification of prion-like
    regions is not performed because the appropriate threshold depends on
    context (see Lancaster et al. 2014 Cell Reports).
    """
    # Yeast FG-domain (prion-like) frequencies — Lancaster et al. 2014, Table S1
    FG_FREQ: dict[str, float] = {
        "A": 0.059, "C": 0.003, "D": 0.019, "E": 0.020, "F": 0.040,
        "G": 0.098, "H": 0.014, "I": 0.023, "K": 0.043, "L": 0.046,
        "M": 0.013, "N": 0.120, "P": 0.044, "Q": 0.133, "R": 0.020,
        "S": 0.131, "T": 0.052, "V": 0.023, "W": 0.005, "Y": 0.091,
    }
    # SwissProt background frequencies
    BG_FREQ: dict[str, float] = {
        "A": 0.070, "C": 0.023, "D": 0.053, "E": 0.063, "F": 0.039,
        "G": 0.068, "H": 0.023, "I": 0.053, "K": 0.058, "L": 0.099,
        "M": 0.025, "N": 0.045, "P": 0.049, "Q": 0.037, "R": 0.051,
        "S": 0.072, "T": 0.057, "V": 0.064, "W": 0.013, "Y": 0.032,
    }

    # Per-residue log-odds: natural logarithm (base e), matching Lancaster et al. 2014
    log_odds: dict[str, float] = {}
    for aa in "ACDEFGHIKLMNPQRSTVWY":
        fg = FG_FREQ.get(aa, 1e-6)
        bg = BG_FREQ.get(aa, 1e-6)
        log_odds[aa] = math.log(fg / bg)  # ln(fg/bg)

    n = len(seq)
    # Sliding window average (half-window padding at edges)
    profile: list[float] = []
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        scores = [log_odds.get(seq[j], 0.0) for j in range(lo, hi)]
        profile.append(sum(scores) / len(scores))

    return {
        "profile": profile,
        "max_score": max(profile) if profile else 0.0,
        "mean_score": sum(profile) / len(profile) if profile else 0.0,
    }
