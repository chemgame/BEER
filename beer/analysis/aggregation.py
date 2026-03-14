"""
beer.analysis.aggregation
=========================
β-aggregation and amyloid propensity analysis.

Algorithms
----------
* ZYGGREGATOR per-residue propensity scores:
  Tartaglia, G.G. & Vendruscolo, M. (2008). The Zyggregator method for predicting
  protein aggregation propensities. *Chem. Biol.*, 15(9), 1008-1018.

* PASTA pairwise β-strand self-interaction energies (diagonal):
  Trovato, A., Chiti, F., Maritan, A. & Seno, F. (2007). Insight into the structure
  of amyloid fibrils from the analysis of globular proteins. *PLoS Comput. Biol.*,
  3(2), e17.

* CamSol intrinsic solubility scale:
  Sormanni, P., Aprile, F.A. & Vendruscolo, M. (2015). The CamSol method of rational
  design of protein mutants with enhanced solubility. *J. Mol. Biol.*, 427(2), 478-490.
"""

import math

# ---------------------------------------------------------------------------
# Published per-residue scales
# ---------------------------------------------------------------------------

ZYGGREGATOR_PROPENSITY: dict[str, float] = {
    'A':  0.67, 'R': -1.65, 'N': -0.43, 'D': -0.75,
    'C':  0.50, 'Q': -0.51, 'E': -1.22, 'G': -0.59,
    'H': -0.13, 'I':  1.29, 'L':  0.93, 'K': -1.42,
    'M':  0.64, 'F':  1.26, 'P': -1.44, 'S': -0.39,
    'T': -0.09, 'W':  0.96, 'Y':  0.74, 'V':  1.04,
}
"""Per-residue β-aggregation propensity scores (Tartaglia & Vendruscolo 2008)."""

PASTA_ENERGY: dict[str, float] = {
    'A': -0.22, 'R':  0.66, 'N':  0.14, 'D':  0.81,
    'C': -0.65, 'Q': -0.04, 'E':  0.58, 'G':  0.08,
    'H': -0.35, 'I': -1.46, 'L': -1.34, 'K':  0.59,
    'M': -0.94, 'F': -1.47, 'P':  1.53, 'S':  0.10,
    'T': -0.34, 'W': -1.35, 'Y': -1.04, 'V': -1.32,
}
"""Diagonal of PASTA pairwise β-strand interaction energy matrix (Trovato et al. 2007).
More negative values indicate stronger self-pairing tendency."""

CAMSOLMT_SCALE: dict[str, float] = {
    'A':  0.238, 'R': -0.132, 'N':  0.047, 'D':  0.191,
    'C':  0.238, 'Q':  0.047, 'E':  0.191, 'G':  0.024,
    'H': -0.083, 'I': -0.387, 'L': -0.387, 'K': -0.065,
    'M': -0.265, 'F': -0.386, 'P':  0.190, 'S':  0.264,
    'T':  0.209, 'W': -0.380, 'Y': -0.241, 'V': -0.322,
}
"""CamSol intrinsic solubility scale (Sormanni et al. 2015, J Mol Biol)."""

# Kyte-Doolittle defined locally to avoid circular imports
KYTE_DOOLITTLE: dict[str, float] = {
    'A':  1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5,
    'C':  2.5, 'Q': -3.5, 'E': -3.5, 'G': -0.4,
    'H': -3.2, 'I':  4.5, 'L':  3.8, 'K': -3.9,
    'M':  1.9, 'F':  2.8, 'P': -1.6, 'S': -0.8,
    'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V':  4.2,
}

# CSS injected into every HTML section for consistent styling
_REPORT_CSS = """
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 11pt;
       color: #1a1a2e; margin: 0; padding: 0; line-height: 1.6; }
h2 { font-size: 13pt; color: #4361ee; margin-top: 18px; margin-bottom: 8px; font-weight: 600; }
table { border-collapse: collapse; width: 100%; margin: 10px 0 16px 0; font-size: 10pt; }
th { background-color: #4361ee; color: #ffffff; padding: 7px 12px;
     text-align: left; font-weight: 600; }
td { padding: 6px 12px; border-bottom: 1px solid #e8eaf0; color: #2d3748; }
tr:nth-child(even) td { background-color: #f8f9fd; }
tr:hover td { background-color: #eef0f8; }
p.note { font-size: 9pt; color: #718096; font-style: italic; margin: 4px 0 12px 0; }
"""


# ---------------------------------------------------------------------------
# Core computational functions
# ---------------------------------------------------------------------------

def calc_aggregation_profile(seq: str, window: int = 6) -> list[float]:
    """Compute per-residue ZYGGREGATOR-style β-aggregation propensity profile.

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
        lo = max(0, i - half)
        hi = min(n, i + half + (window - 2 * half))  # handle odd/even window
        # For even window keep consistent: centre at i means [i-half, i-half+window)
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
    threshold: float = 1.0,
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
    css = _REPORT_CSS.replace("#4361ee", accent)
    _s = f"<style>{css}</style>"

    stats = calc_solubility_stats(seq)
    hotspots = stats['aggregation_hotspots']
    n = len(seq)

    # --- summary table ---
    cs_mean = stats['mean_camsolmt']
    cs_class = (
        "good (soluble)" if cs_mean > 0.1 else
        "borderline" if cs_mean > -0.1 else
        "poor (aggregation-prone)"
    )
    agg_mean = stats['mean_aggregation_propensity']
    agg_class = (
        "low" if agg_mean < 0.3 else
        "moderate" if agg_mean < 0.7 else
        "high"
    )

    summary_rows = (
        f"<tr><td>Sequence length</td><td>{n} aa</td></tr>"
        f"<tr><td>Mean CamSol intrinsic score</td>"
        f"<td>{cs_mean:.3f} &mdash; {cs_class}</td></tr>"
        f"<tr><td>Fraction insoluble residues (CamSol &lt; &minus;0.2)</td>"
        f"<td>{stats['fraction_insoluble']:.1%}</td></tr>"
        f"<tr><td>Fraction soluble residues (CamSol &gt; 0.2)</td>"
        f"<td>{stats['fraction_soluble']:.1%}</td></tr>"
        f"<tr><td>Mean ZYGGREGATOR propensity</td>"
        f"<td>{agg_mean:.3f} &mdash; {agg_class}</td></tr>"
        f"<tr><td>Aggregation hotspots (&ge;4 aa, score &ge;1.0)</td>"
        f"<td>{stats['n_hotspots']}</td></tr>"
    )

    summary_html = (
        "<h2>Aggregation &amp; Solubility Summary</h2>"
        "<table>"
        "<tr><th>Property</th><th>Value</th></tr>"
        f"{summary_rows}"
        "</table>"
        "<p class='note'>"
        "CamSol: Sormanni et al. 2015 J Mol Biol 427:478. "
        "ZYGGREGATOR: Tartaglia &amp; Vendruscolo 2008 Chem Biol 15:1008."
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
            "(threshold: mean ZYGGREGATOR &ge; 1.0 over &ge; 4 residues).</p>"
        )

    return _s + summary_html + hotspot_html
