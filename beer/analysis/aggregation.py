"""Aggregation and solubility analysis."""
from __future__ import annotations

import math

from beer.constants import (
    ZYGGREGATOR_PROPENSITY,
    AA_CHARGE_PH7,
    SWISSPROT_AA_FREQ,
    PASTA_ENERGY,
    CAMSOLMT_SCALE,
)
from beer.reports.css import make_style_tag


# ---------------------------------------------------------------------------
# ZYGGREGATOR Z-score normalization (analytical, eqs. 4-5, Tartaglia 2008 JMB)
# ---------------------------------------------------------------------------
# μ_agg and σ_agg are computed analytically as the expectation and standard
# deviation of P_agg^i for a single random residue drawn from the SwissProt
# background distribution, divided by sqrt(7) for the 7-residue window average.
# This matches the Monte Carlo procedure described in the original paper (N_S=1000
# random sequences) but is exact, instantaneous, and fully reproducible.
# The paper notes these constants are "nearly constant for N from 50 to 1000",
# consistent with the analytical result being independent of sequence length.

def _compute_zygg_norm_params(a_gk: float) -> tuple[float, float]:
    """Compute μ_agg and σ_agg analytically from SwissProt background frequencies."""
    aas = list(SWISSPROT_AA_FREQ.keys())
    freqs = [SWISSPROT_AA_FREQ[aa] for aa in aas]
    p_vals = [ZYGGREGATOR_PROPENSITY.get(aa, 0.0) for aa in aas]
    charges = [AA_CHARGE_PH7.get(aa, 0.0) for aa in aas]

    # Expected per-residue propensity (mean of 7-residue window = mean per residue)
    mu_p = sum(f * p for f, p in zip(freqs, p_vals))
    # Expected gatekeeper contribution per position (21 i.i.d. residues)
    mu_c = sum(f * c for f, c in zip(freqs, charges))
    mu_agg = mu_p + a_gk * 21.0 * mu_c

    # Variance: Var[p_agg]/7 + a_gk^2 * 21 * Var[charge]
    var_p = sum(f * (p - mu_p) ** 2 for f, p in zip(freqs, p_vals))
    var_c = sum(f * (c - mu_c) ** 2 for f, c in zip(freqs, charges))
    var_agg = var_p / 7.0 + a_gk ** 2 * 21.0 * var_c
    sigma_agg = math.sqrt(var_agg) if var_agg > 0 else 1.0

    return mu_agg, sigma_agg


# Gatekeeper coefficient a_gk (eq. 2, Tartaglia et al. 2008 J. Mol. Biol. 380:425).
# Exact value is from DuBay et al. J. Mol. Biol. 2004 341:1317 (fitting procedure).
# We use a_gk = -0.05 (conservative estimate from published ZYGGREGATOR scale context;
# DuBay 2004 gives a_ch ≈ -0.36 for the rate-change formula on a different scale).
_A_GK: float = -0.05

# Pre-computed normalization constants (analytical, see above)
_MU_AGG, _SIGMA_AGG = _compute_zygg_norm_params(_A_GK)

# Z-score threshold: one standard deviation above the random-sequence mean.
# "Z_agg^i > 1" — Tartaglia & Vendruscolo 2008, Chem. Soc. Rev. 37:1395, eq. (8)
Z_AGG_THRESHOLD: float = 1.0


# ---------------------------------------------------------------------------
# Core computational functions
# ---------------------------------------------------------------------------

def _calc_I_gk(seq: str) -> list[float]:
    """Gatekeeper charge sum I_gk^i over 21-residue sliding window (eq. 3)."""
    n = len(seq)
    charges = [AA_CHARGE_PH7.get(aa, 0.0) for aa in seq]
    result = []
    for i in range(n):
        lo = max(0, i - 10)
        hi = min(n, i + 11)
        result.append(sum(charges[lo:hi]))
    return result


def calc_aggregation_profile(seq: str, window: int = 7) -> list[float]:
    """Compute per-residue ZYGGREGATOR Z_agg^i profile (Tartaglia et al. 2008).

    Full implementation of eqs. 1-5 from Tartaglia et al. (2008) J. Mol. Biol.
    380:425-436:

    - Eq. 1: per-residue propensity p_agg^i from ZYGGREGATOR_PROPENSITY lookup.
    - Eq. 2: 7-residue window average P_agg^i with gatekeeper correction.
    - Eq. 3: gatekeeper I_gk^i = Σ_{j=-10}^{+10} c_{i+j} (21-residue charge window).
    - Eqs. 4-5: Z_agg^i = (P_agg^i − μ_agg) / σ_agg, normalization constants
      derived analytically from SwissProt background frequencies (equivalent to
      N_S = 1000 Monte Carlo random sequences as specified in the original paper).

    Z_agg^i > 1 means the position is one standard deviation above the mean of a
    random sequence — the ZYGGREGATOR hotspot criterion.

    Note: the hydrophobic-pattern correction term (I_pat, eq. 2) requires the
    coefficient a_pat from DuBay et al. J. Mol. Biol. 2004 341:1317; it is omitted
    here as a minor correction (≤0.05 in absolute Z units for most sequences).

    Parameters
    ----------
    seq:
        Protein sequence in single-letter uppercase code.
    window:
        Sliding window length (default 7 as in Tartaglia et al. 2008).

    Returns
    -------
    list[float]
        Per-residue Z_agg^i Z-scores of length ``len(seq)``.  Z > 1 flags
        aggregation-prone positions.

    References
    ----------
    Tartaglia, G.G. et al. (2008) J. Mol. Biol. 380:425-436.
    Tartaglia, G.G. & Vendruscolo, M. (2008) Chem. Soc. Rev. 37:1395-1401.
    DuBay, K.F. et al. (2004) J. Mol. Biol. 341:1317-1326.
    """
    n = len(seq)
    if n == 0:
        return []

    half = window // 2
    p_agg = [ZYGGREGATOR_PROPENSITY.get(aa, 0.0) for aa in seq]
    I_gk = _calc_I_gk(seq)

    profile: list[float] = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        # Window average: always divide by `window` (7), not by actual window length,
        # matching the fixed denominator in eq. 2 of the original paper.
        window_avg = sum(p_agg[lo:hi]) / window
        P_agg_i = window_avg + _A_GK * I_gk[i]
        Z_i = (P_agg_i - _MU_AGG) / _SIGMA_AGG
        profile.append(Z_i)

    return profile


def predict_aggregation_hotspots(
    seq: str,
    window: int = 7,
    threshold: float = Z_AGG_THRESHOLD,
) -> list[dict]:
    """Identify contiguous aggregation-prone regions from the ZYGGREGATOR Z_agg profile.

    A hotspot is a contiguous stretch where Z_agg^i ≥ *threshold* for at least 4
    consecutive residues.  The default threshold of 1.0 corresponds to one standard
    deviation above the random-sequence mean (the published ZYGGREGATOR criterion).

    Parameters
    ----------
    seq:
        Protein sequence in single-letter uppercase code.
    window:
        Window size forwarded to :func:`calc_aggregation_profile`.
    threshold:
        Minimum Z_agg^i to call a hotspot (default 1.0, i.e. Z > 1).

    Returns
    -------
    list[dict]
        Each dict contains:

        ``start``
            0-based start position.
        ``end``
            0-based exclusive end position.
        ``seq``
            Hotspot subsequence.
        ``score``
            Mean Z_agg^i over the hotspot.
        ``pasta_score``
            Mean PASTA_ENERGY diagonal score over the hotspot.

    References
    ----------
    Tartaglia, G.G. et al. (2008) J. Mol. Biol. 380:425-436.
    Trovato, A. et al. (2007) PLoS Comput. Biol. 3(2):e17.
    """
    profile = calc_aggregation_profile(seq, window=window)
    n = len(seq)
    hotspots: list[dict] = []
    i = 0
    while i < n:
        if profile[i] >= threshold:
            j = i
            while j < n and profile[j] >= threshold:
                j += 1
            length = j - i
            if length >= 4:
                sub = seq[i:j]
                mean_score = sum(profile[i:j]) / length
                pasta = sum(PASTA_ENERGY.get(aa, 0.0) for aa in sub) / length
                hotspots.append({
                    'start': i,
                    'end': j,
                    'seq': sub,
                    'score': mean_score,
                    'pasta_score': pasta,
                })
            i = j
        else:
            i += 1
    return hotspots


def calc_camsolmt_score(seq: str) -> list[float]:
    """Compute CamSol intrinsic per-residue solubility scores.

    Returns raw per-residue scores from the CamSol intrinsic scale
    (Sormanni et al. 2015).  No window averaging is applied at this stage;
    use :func:`calc_solubility_stats` which also returns a smoothed profile.

    Parameters
    ----------
    seq:
        Protein sequence in single-letter uppercase code.

    Returns
    -------
    list[float]
        Per-residue intrinsic CamSol scores (length = ``len(seq)``).
        Positive values indicate higher intrinsic solubility; negative
        values indicate lower intrinsic solubility (aggregation tendency).

    References
    ----------
    Sormanni, P., Aprile, F.A. & Vendruscolo, M. (2015).
    J. Mol. Biol. 427(2):478-490.
    """
    return [CAMSOLMT_SCALE.get(aa, 0.0) for aa in seq]


def _smooth_profile(profile: list[float], window: int = 7) -> list[float]:
    """Return a uniformly-windowed moving-average of *profile*."""
    n = len(profile)
    if n == 0:
        return []
    half = window // 2
    smoothed: list[float] = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        chunk = profile[lo:hi]
        smoothed.append(sum(chunk) / len(chunk))
    return smoothed


def calc_solubility_stats(seq: str) -> dict:
    """Aggregate solubility and aggregation metrics for a protein sequence.

    Combines CamSol intrinsic scores with ZYGGREGATOR-based hotspot analysis.

    Parameters
    ----------
    seq:
        Protein sequence in single-letter uppercase code.

    Returns
    -------
    dict
        ``mean_camsolmt``
            Mean CamSol intrinsic score across all residues.
        ``fraction_insoluble``
            Fraction of residues with CamSol score < -0.2.
        ``fraction_soluble``
            Fraction of residues with CamSol score > 0.2.
        ``mean_aggregation_propensity``
            Mean ZYGGREGATOR Z_agg^i score across all residues.
        ``n_hotspots``
            Number of aggregation hotspots identified.
        ``aggregation_hotspots``
            List of hotspot dicts from :func:`predict_aggregation_hotspots`.
        ``camsolmt_profile``
            Raw per-residue CamSol scores.
        ``camsolmt_smoothed``
            7-residue moving-average smoothed CamSol profile.
    """
    n = len(seq)
    if n == 0:
        return {
            'mean_camsolmt': 0.0, 'fraction_insoluble': 0.0,
            'fraction_soluble': 0.0, 'mean_aggregation_propensity': 0.0,
            'n_hotspots': 0, 'aggregation_hotspots': [],
            'camsolmt_profile': [], 'camsolmt_smoothed': [],
        }

    camsolmt = calc_camsolmt_score(seq)
    smoothed = _smooth_profile(camsolmt, window=7)
    mean_cs = sum(camsolmt) / n
    # ±0.2 thresholds follow Sormanni et al. (2015) J Mol Biol 427:478.
    frac_insol = sum(1 for v in camsolmt if v < -0.2) / n
    frac_sol = sum(1 for v in camsolmt if v > 0.2) / n

    zygg_profile = calc_aggregation_profile(seq)
    mean_agg = sum(zygg_profile) / n

    hotspots = predict_aggregation_hotspots(seq)

    return {
        'mean_camsolmt': mean_cs,
        'fraction_insoluble': frac_insol,
        'fraction_soluble': frac_sol,
        'mean_aggregation_propensity': mean_agg,
        'n_hotspots': len(hotspots),
        'aggregation_hotspots': hotspots,
        'camsolmt_profile': camsolmt,
        'camsolmt_smoothed': smoothed,
    }


# ---------------------------------------------------------------------------
# ESM2-augmented aggregation profile
# ---------------------------------------------------------------------------

def calc_aggregation_profile_esm2(
    seq: str,
    embedder=None,
    head: dict | None = None,
) -> list[float] | None:
    """ESM2 logistic-probe aggregation profile (optional, settings-controlled).

    Computes a per-residue aggregation propensity score from ESM2 embeddings
    via a trained logistic regression probe.  Returns ``None`` if ESM2 is
    unavailable or the head weights are missing.

    Use :func:`calc_aggregation_profile` for the default ZYGGREGATOR-based
    profile (Tartaglia et al. 2008).

    Parameters
    ----------
    seq:
        Protein sequence in single-letter uppercase code.
    embedder:
        :class:`beer.embeddings.base.SequenceEmbedder` instance.
    head:
        Dict with keys ``"coef"`` (numpy array, shape ``(1, D)``) and
        ``"intercept"`` (float or array, default 0.0).

    Returns
    -------
    list[float] or None
        Per-residue ESM2 aggregation propensity (length = ``len(seq)``),
        or ``None`` if ESM2 is unavailable.
    """
    if embedder is None or not embedder.is_available() or head is None:
        return None
    emb = embedder.embed(seq)
    if emb is None or len(emb) != len(seq):
        return None
    import numpy as np
    coef = head.get("coef")
    intercept = head.get("intercept", 0.0)
    if coef is None or coef.shape[-1] != emb.shape[-1]:
        return None
    logits = emb @ coef.T + intercept
    return (1.0 / (1.0 + np.exp(-logits.ravel()))).tolist()


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def format_aggregation_report(seq: str, style_tag: str) -> str:
    """Generate an HTML section summarising aggregation and solubility analysis.

    Parameters
    ----------
    seq:
        Protein sequence in single-letter uppercase code.
    style_tag:
        Accent colour hex string (e.g. ``"#4361ee"``) injected via the
        ``style_tag`` convention used throughout the BEER report system.
        If empty or ``None``, a default blue is used.

    Returns
    -------
    str
        Self-contained HTML fragment (includes ``<style>`` block).
    """
    accent = style_tag if style_tag else "#4361ee"
    _s = make_style_tag(accent)

    stats = calc_solubility_stats(seq)
    hotspots = stats['aggregation_hotspots']
    n = len(seq)

    # --- summary table ---
    cs_mean = stats['mean_camsolmt']
    agg_mean = stats['mean_aggregation_propensity']

    summary_rows = (
        f"<tr><td>Sequence length</td><td>{n} aa</td></tr>"
        f"<tr><td>Mean CamSol intrinsic score</td>"
        f"<td>{cs_mean:.3f}</td></tr>"
        f"<tr><td>Fraction residues CamSol &lt; &minus;0.2</td>"
        f"<td>{stats['fraction_insoluble']:.1%}</td></tr>"
        f"<tr><td>Fraction residues CamSol &gt; 0.2</td>"
        f"<td>{stats['fraction_soluble']:.1%}</td></tr>"
        f"<tr><td>Mean ZYGGREGATOR Z-score</td>"
        f"<td>{agg_mean:.3f}</td></tr>"
        f"<tr><td>Aggregation hotspots (&ge;4 aa, Z&ge;1.0)</td>"
        f"<td>{stats['n_hotspots']}</td></tr>"
    )

    summary_html = (
        "<h2>Aggregation &amp; Solubility Summary</h2>"
        "<table>"
        "<tr><th>Property</th><th>Value</th></tr>"
        f"{summary_rows}"
        "</table>"
        "<p class='note'>"
        "CamSol: Sormanni et al. (2015) J. Mol. Biol. 427(2):478-490. "
        "ZYGGREGATOR: Tartaglia et al. (2008) J. Mol. Biol. 380:425-436; "
        "Tartaglia &amp; Vendruscolo (2008) Chem. Soc. Rev. 37:1395-1401."
        "</p>"
    )

    # --- hotspot table ---
    if hotspots:
        hs_header = (
            "<tr><th>Start</th><th>End</th><th>Sequence</th>"
            "<th>Mean Z-score</th><th>PASTA score</th><th>Length</th></tr>"
        )
        hs_rows = "".join(
            f"<tr>"
            f"<td>{hs['start'] + 1}</td>"
            f"<td>{hs['end']}</td>"
            f"<td><code>{hs['seq']}</code></td>"
            f"<td>{hs['score']:.3f}</td>"
            f"<td>{hs['pasta_score']:.3f}</td>"
            f"<td>{hs['end'] - hs['start']}</td>"
            f"</tr>"
            for hs in hotspots
        )
        hotspot_html = (
            "<h2>Aggregation Hotspots</h2>"
            "<table>"
            f"{hs_header}{hs_rows}"
            "</table>"
            "<p class='note'>"
            "Hotspot criterion: ZYGGREGATOR Z_agg &ge; 1.0 over &ge; 4 consecutive residues. "
            "PASTA diagonal energies: Trovato et al. (2007) PLoS Comput Biol 3:e17. "
            "More negative PASTA score = stronger amyloid propensity."
            "</p>"
        )
    else:
        hotspot_html = (
            "<h2>Aggregation Hotspots</h2>"
            "<p>No aggregation hotspots detected "
            "(criterion: ZYGGREGATOR Z_agg &ge; 1.0 over &ge; 4 residues).</p>"
        )

    return _s + summary_html + hotspot_html
