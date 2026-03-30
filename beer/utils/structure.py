"""Structure prediction utilities."""
from __future__ import annotations
import math
import re
from collections import Counter

from beer.constants import (
    KYTE_DOOLITTLE,
    DISORDER_PROPENSITY,
    COILED_COIL_PROPENSITY,
    LINEAR_MOTIFS,
    LARKS_AROMATIC,
    LARKS_LC,
    STICKER_ALL,
)


_DPROP_MIN = min(DISORDER_PROPENSITY.values())   # W = -0.884
_DPROP_MAX = max(DISORDER_PROPENSITY.values())   # P =  0.987
_DPROP_SPAN = _DPROP_MAX - _DPROP_MIN


def _classical_disorder_profile(seq: str, window: int = 9) -> list:
    """Classical sliding-window disorder propensity, normalised to 0-1.

    Normalisation uses the global min/max of the DISORDER_PROPENSITY scale
    (W = -0.884, P = 0.987) so that absolute scores are comparable across
    sequences.  Per-sequence normalisation would make a poly-W sequence look
    as disordered as a poly-P sequence, which is incorrect.
    """
    raw = [DISORDER_PROPENSITY.get(aa, 0.0) for aa in seq]
    norm = [(v - _DPROP_MIN) / _DPROP_SPAN for v in raw]
    n = len(norm)
    if n < window:
        return norm
    half = window // 2
    smoothed = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        smoothed.append(sum(norm[lo:hi]) / (hi - lo))
    return smoothed


def calc_disorder_profile(
    seq: str,
    window: int = 9,
    embedder=None,
    head: dict | None = None,
) -> list:
    """Per-residue disorder propensity, normalised to 0-1.

    When *embedder* is provided, is available, and *head* contains trained
    logistic-regression weights (``coef`` array of shape ``(1, D)`` and
    optional ``intercept``), ESM2 embeddings are used to compute per-residue
    disorder scores via a linear probe + sigmoid.  Falls back to the
    classical sliding-window approach otherwise.

    Parameters
    ----------
    seq:
        Protein sequence in single-letter uppercase code.
    window:
        Window size for the classical fallback (default 9).
    embedder:
        Optional :class:`beer.embeddings.base.SequenceEmbedder` instance.
    head:
        Optional dict with keys ``"coef"`` (numpy array, shape ``(1, D)``)
        and ``"intercept"`` (float or array, default 0.0).
    """
    # Priority 1: ESM2 probe (most accurate)
    if embedder is not None and embedder.is_available() and head is not None:
        emb = embedder.embed(seq)
        if emb is not None and len(emb) == len(seq):
            import numpy as np
            coef = head.get("coef")
            intercept = head.get("intercept", 0.0)
            if coef is not None and coef.shape[-1] == emb.shape[-1]:
                logits = emb @ coef.T + intercept
                scores = 1.0 / (1.0 + np.exp(-logits.ravel()))
                return scores.tolist()
    # Priority 2: metapredict (DL-based, installed as core dependency)
    try:
        import metapredict as meta
        scores = meta.predict_disorder(seq)
        return [float(s) for s in scores]
    except Exception:
        pass
    # Priority 3: classical sliding-window fallback
    return _classical_disorder_profile(seq, window)


def predict_tm_helices(seq: str, window: int = 19, threshold: float = 1.6,
                       min_len: int = 17, max_len: int = 25) -> list:
    """Predict TM helices via Kyte-Doolittle sliding window.

    Algorithm:
    1. Compute per-window KD average (only full-length windows; no partial edges).
    2. Mark every residue covered by at least one above-threshold window as TM.
    3. Collect contiguous marked regions; keep those within [min_len, max_len].
       Overlong regions are split by finding the single highest-scoring window.
    4. Assign topology using the inside-positive rule (von Heijne):
       the side flanking with more K/R is cytoplasmic.

    Returns list of dicts: {start (0-based), end (0-based inclusive), score, orientation}.

    References
    ----------
    Kyte, J. & Doolittle, R.F. (1982) J. Mol. Biol. 157:105-132.
        Window size 19 and threshold 1.6 for TM helix detection.
    von Heijne, G. (1986) EMBO J. 5:3021-3027.
        Inside-positive rule: the cytoplasmic flanking region is enriched in K/R.
    """
    n = len(seq)
    if n < window:
        return []

    # Step 1 — per-window average KD (strictly full windows only)
    win_scores = [
        sum(KYTE_DOOLITTLE.get(seq[j], 0.0) for j in range(i, i + window)) / window
        for i in range(n - window + 1)
    ]

    # Step 2 — per-residue TM mask: residue r is TM if any window covering it scores >= threshold
    tm_mask = [False] * n
    for i, score in enumerate(win_scores):
        if score >= threshold:
            for r in range(i, i + window):
                tm_mask[r] = True

    # Step 3 — collect contiguous TM regions
    helices = []
    i = 0
    while i < n:
        if tm_mask[i]:
            j = i
            while j < n and tm_mask[j]:
                j += 1
            span = j - i
            if min_len <= span <= max_len:
                seg_score = sum(KYTE_DOOLITTLE.get(seq[r], 0.0) for r in range(i, j)) / span
                helices.append({"start": i, "end": j - 1, "score": round(seg_score, 3)})
            elif span > max_len:
                # Too long — pick the best single window inside the region
                best_i = max(range(i, j - window + 1),
                             key=lambda k: win_scores[k] if k < len(win_scores) else -999)
                best_s = win_scores[best_i] if best_i < len(win_scores) else 0.0
                helices.append({"start": best_i, "end": best_i + window - 1,
                                 "score": round(best_s, 3)})
            i = j
        else:
            i += 1

    # Step 4 — inside-positive rule (von Heijne)
    pos = set("KR")
    flank = 15
    for h in helices:
        s, e = h["start"], h["end"]
        n_pos = sum(1 for aa in seq[max(0, s - flank):s] if aa in pos)
        c_pos = sum(1 for aa in seq[e + 1:min(n, e + 1 + flank)] if aa in pos)
        # More K/R on C-terminal side → C-term is cytoplasmic → N-term is extracellular → out→in
        h["orientation"] = "out\u2192in" if c_pos >= n_pos else "in\u2192out"
    return helices


def detect_larks(seq: str, window: int = 7, min_arom: int = 1,
                 min_lc_frac: float = 0.50, max_entropy: float = 1.8) -> list:
    """Detect LARKS (Low-complexity Aromatic-Rich Kinked Segments).

    A LARKS is a short window (6–8 residues) that:
      • Contains >= 1 aromatic residue (F/W/Y)
      • Has >= 50 % low-complexity residues (G/A/S/T/N/Q)
      • Has Shannon entropy < 1.8 bits

    Returns list of dicts: {start, end, seq, n_arom, lc_frac, entropy}
    """
    n = len(seq)
    hits = []
    for i in range(n - window + 1):
        w = seq[i:i + window]
        n_arom = sum(1 for aa in w if aa in LARKS_AROMATIC)
        if n_arom < min_arom:
            continue
        lc_frac = sum(1 for aa in w if aa in LARKS_LC) / window
        if lc_frac < min_lc_frac:
            continue
        # Shannon entropy of window
        cnt = Counter(w)
        H = -sum((v / window) * math.log2(v / window) for v in cnt.values())
        if H >= max_entropy:
            continue
        # Merge overlapping hits
        span = (i, i + window - 1)
        overlap = False
        for prev in hits:
            if prev["start"] <= span[1] and span[0] <= prev["end"]:
                overlap = True
                break
        if not overlap:
            hits.append({"start": i, "end": i + window - 1,
                         "seq": w, "n_arom": n_arom,
                         "lc_frac": round(lc_frac, 3),
                         "entropy": round(H, 3)})
    return hits


def predict_coiled_coil(seq: str, window: int = 28) -> list:
    """Heptad-periodicity coiled-coil scoring.

    Uses a 28-residue (4-heptad) sliding window with position-weighted
    propensity scores.  Positions a and d of each heptad (the hydrophobic
    core positions) receive weight 0.20; flanking positions receive 0.05–0.10.
    Propensity values are from the MTIDK database compiled by Lupas et al.
    (1991), as tabulated in Berger et al. (1995).

    Returns list of per-residue scores normalised to [0, 1].

    References
    ----------
    Lupas, A., Van Dyke, M. & Stock, J. (1991) Science 252:1162-1164.
    Berger, B. et al. (1995) Proc. Natl. Acad. Sci. USA 92:8259-8263.
    """
    n = len(seq)
    scores = [0.0] * n
    if n < window:
        return scores

    # Heptad positions: a=0, b=1, c=2, d=3, e=4, f=5, g=6
    # Weight positions a and d (0, 3) most heavily
    pos_weights = [0.20, 0.05, 0.05, 0.20, 0.10, 0.10, 0.10,
                   0.20, 0.05, 0.05, 0.20, 0.10, 0.10, 0.10,
                   0.20, 0.05, 0.05, 0.20, 0.10, 0.10, 0.10,
                   0.20, 0.05, 0.05, 0.20, 0.10, 0.10, 0.10]

    win_scores = []
    for i in range(n - window + 1):
        w = seq[i:i + window]
        s = sum(COILED_COIL_PROPENSITY.get(aa, 1.0) * pw
                for aa, pw in zip(w, pos_weights))
        win_scores.append(s)

    # Distribute window scores to residues
    counts = [0] * n
    for i, ws in enumerate(win_scores):
        for r in range(i, i + window):
            scores[r] += ws
            counts[r] += 1
    for i in range(n):
        if counts[i] > 0:
            scores[i] = scores[i] / counts[i]

    # Normalise to 0–1 range (typical score ~0.9 for real coiled coils)
    mx = max(scores) if max(scores) > 0 else 1.0
    scores = [s / mx for s in scores]
    return scores


def scan_linear_motifs(seq: str) -> list:
    """Scan sequence against the built-in LINEAR_MOTIFS library.

    Returns list of dicts: {name, description, start, end, match}
    """
    hits = []
    for name, pattern, description in LINEAR_MOTIFS:
        for m in re.finditer(pattern, seq):
            hits.append({
                "name": name,
                "description": description,
                "start": m.start(),
                "end": m.end() - 1,
                "match": m.group(),
            })
    hits.sort(key=lambda h: h["start"])
    return hits
