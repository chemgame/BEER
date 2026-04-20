"""Structure prediction utilities."""
from __future__ import annotations
import math
import re
from collections import Counter

from beer.constants import (
    KYTE_DOOLITTLE,
    DISORDER_PROPENSITY,
    COILS_MTIDK,
    COILS_BG_LOG,
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
    """Predict TM helices using TMHMM 2.0 (primary) or KD sliding-window (fallback).

    Primary: TMHMM 2.0 Viterbi HMM (Krogh et al. 2001), bundled locally — no
    internet required.  Accurately handles charged functional residues in TM
    helices (proton pumps, GPCRs).

    Fallback: Kyte-Doolittle sliding-window (Kyte & Doolittle 1982) used only
    if the TMHMM model file cannot be loaded.

    Returns list of dicts: {start (0-based), end (0-based inclusive), score,
    orientation, source}.
    """
    try:
        from beer.analysis.tmhmm_local import predict_tm_helices as _tmhmm
        return _tmhmm(seq)
    except Exception:
        pass  # fall through to KD-window

    # KD sliding-window fallback
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
    """COILS algorithm — heptad-register log-odds coiled-coil scorer.

    Implements the COILS method (Lupas et al. 1991 Science 252:1162;
    Lupas 1996 Methods Enzymol. 266:513) using the MTIDK 20×7 position-weight
    matrix.  For each window of length *window* (default 28 = 4 heptads) all
    seven heptad phase registers are tried and the maximum log-odds score is
    retained.  The log-odds are converted to P(CC) via a calibrated sigmoid:

        P(CC) = 1 / (1 + exp(−1.25 × (log_odds − 2.5)))

    Calibration: GCN4 leucine zipper log_odds ≈ 4.3 → P ≈ 0.90;
    (LEALELK)₄ synthetic log_odds ≈ 6.1 → P ≈ 0.99;
    random protein log_odds ≈ 0 → P ≈ 0.04.

    Returns list of per-residue P(CC) scores (float, 0–1), length == len(seq).

    References
    ----------
    Lupas, A., Van Dyke, M. & Stock, J. (1991) Science 252:1162-1164.
    Lupas, A. (1996) Methods Enzymol. 266:513-525.
    """
    n = len(seq)
    scores = [0.0] * n
    if n < window:
        return scores

    # Pre-compute background log-score per heptad position (sum over window).
    # Each window covers (window // 7) full heptads + remainder positions.
    # Background = sum of COILS_BG_LOG[pos % 7] for pos in range(window).
    bg_total = sum(COILS_BG_LOG[p % 7] for p in range(window))

    win_pcc: list[float] = []
    for i in range(n - window + 1):
        w = seq[i:i + window]
        best_log_score = -1e9
        for phase in range(7):
            s = sum(
                math.log(max(COILS_MTIDK.get(aa, (1.0,)*7)[(phase + j) % 7], 1e-9))
                for j, aa in enumerate(w)
            )
            if s > best_log_score:
                best_log_score = s
        log_odds = best_log_score - bg_total
        pcc = 1.0 / (1.0 + math.exp(-1.25 * (log_odds - 2.5)))
        win_pcc.append(pcc)

    # Distribute window P(CC) to residues (average over all covering windows)
    counts = [0] * n
    for i, pcc in enumerate(win_pcc):
        for r in range(i, i + window):
            scores[r] += pcc
            counts[r] += 1
    for i in range(n):
        if counts[i] > 0:
            scores[i] = scores[i] / counts[i]
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
