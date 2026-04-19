"""Aggregation and solubility analysis."""
from __future__ import annotations

from beer.constants import (
    ZYGGREGATOR_PROPENSITY,
    PASTA_ENERGY,
    CAMSOLMT_SCALE,
)
from beer.reports.css import make_style_tag


# ---------------------------------------------------------------------------
# Core computational functions
# ---------------------------------------------------------------------------

def calc_aggregation_profile(seq: str, window: int = 6) -> list[float]:
    """Compute per-residue ZYGGREGATOR-style beta-aggregation propensity profile.

    A sliding window of length *window* is centred at each residue; the score
    for that position is the arithmetic mean of ZYGGREGATOR_PROPENSITY values
    over the window.  At sequence edges the window is truncated to the
    available residues (partial windows are acceptable).

    Parameters
    ----------
    seq:
        Protein sequence in single-letter uppercase code.
    window:
        Sliding window length (default 6 as in Tartaglia & Vendruscolo 2008).

    Returns
    -------
    list[float]
        Per-residue profile of length ``len(seq)``.

    References
    ----------
    Tartaglia, G.G. & Vendruscolo, M. (2008) Chem. Biol. 15(9):1008-1018.
    """
    n = len(seq)
    if n == 0:
        return []

    half = window // 2
    profile: list[float] = []

    for i in range(n):
        # Anchor the window at the left edge of the half-window, then extend
        # right by exactly `window` residues; clamp at the C-terminus by
        # pulling the left boundary back so the window stays full-length.
        lo = max(0, i - half)
        hi = min(n, lo + window)
        if hi == n:
            lo = max(0, hi - window)
        sub = seq[lo:hi]
        vals = [ZYGGREGATOR_PROPENSITY.get(aa, 0.0) for aa in sub]
        profile.append(sum(vals) / len(vals) if vals else 0.0)

    return profile


def predict_aggregation_hotspots(
    seq: str,
    window: int = 6,
    threshold: float = 0.3,
) -> list[dict]:
    """Identify contiguous aggregation-prone regions using ZYGGREGATOR profile.

    A hotspot is a contiguous stretch where the sliding-window aggregation
    profile (see :func:`calc_aggregation_profile`) equals or exceeds *threshold*
    for at least 4 consecutive residues.  For each hotspot, a PASTA energy score
    is also computed as the mean of PASTA_ENERGY values over the hotspot
    sequence (lower = more amyloidogenic).

    Parameters
    ----------
    seq:
        Protein sequence in single-letter uppercase code.
    window:
        Window size forwarded to :func:`calc_aggregation_profile`.
    threshold:
        Minimum mean propensity to call a hotspot (default 1.0).

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
            Mean ZYGGREGATOR propensity over the hotspot.
        ``pasta_score``
            Mean PASTA_ENERGY diagonal score over the hotspot.

    References
    ----------
    Tartaglia, G.G. & Vendruscolo, M. (2008) Chem. Biol. 15(9):1008-1018.
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
            Mean ZYGGREGATOR propensity (no windowing) across all residues.
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

    zygg_raw = [ZYGGREGATOR_PROPENSITY.get(aa, 0.0) for aa in seq]
    mean_agg = sum(zygg_raw) / n

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
    profile (Tartaglia & Vendruscolo 2008).

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
        f"<tr><td>Mean ZYGGREGATOR propensity</td>"
        f"<td>{agg_mean:.3f}</td></tr>"
        f"<tr><td>Aggregation hotspots (&ge;4 aa, propensity &ge;0.3)</td>"
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
        "ZYGGREGATOR: Tartaglia &amp; Vendruscolo (2008) Chem. Biol. 15:1008-1018."
        "</p>"
    )

    # --- hotspot table ---
    if hotspots:
        hs_header = (
            "<tr><th>Start</th><th>End</th><th>Sequence</th>"
            "<th>ZYGGREGATOR score</th><th>PASTA score</th><th>Length</th></tr>"
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
            "PASTA diagonal energies: Trovato et al. 2007 PLoS Comput Biol 3:e17. "
            "More negative PASTA score = stronger amyloid propensity."
            "</p>"
        )
    else:
        hotspot_html = (
            "<h2>Aggregation Hotspots</h2>"
            "<p>No aggregation hotspots detected "
            "(threshold: mean ZYGGREGATOR &ge; 0.3 over &ge; 4 residues).</p>"
        )

    return _s + summary_html + hotspot_html
