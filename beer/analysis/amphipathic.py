"""Amphipathic helix analysis using the Eisenberg hydrophobic moment."""
from __future__ import annotations
import math

from beer.constants import EISENBERG_SCALE
from beer.reports.css import make_style_tag


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _hydrophobic_moment_window(
    sub: str,
    angle_deg: float = 100.0,
) -> float:
    """Compute muH for a single window sub-sequence.

    muH = sqrt( [Sum Hi*sin(i*delta)]^2 + [Sum Hi*cos(i*delta)]^2 ) / n

    where delta is the inter-residue angle (100 degrees for alpha-helix, 160 degrees for beta-strand),
    i is 0-indexed position within the window, and n = len(sub).

    Parameters
    ----------
    sub:
        Window sub-sequence.
    angle_deg:
        Angular increment per residue in degrees.

    Returns
    -------
    float
        Normalised hydrophobic moment (divided by window length).
    """
    n = len(sub)
    if n == 0:
        return 0.0
    delta = math.radians(angle_deg)
    sin_sum = 0.0
    cos_sum = 0.0
    for i, aa in enumerate(sub):
        h = EISENBERG_SCALE.get(aa, 0.0)
        angle = i * delta
        sin_sum += h * math.sin(angle)
        cos_sum += h * math.cos(angle)
    return math.sqrt(sin_sum ** 2 + cos_sum ** 2) / n


def calc_hydrophobic_moment(
    seq: str,
    window: int = 11,
    angle_deg: float = 100.0,
) -> list[float]:
    """Compute the Eisenberg hydrophobic moment profile across a sequence.

    For each residue position, the moment is computed for the window centred
    at that position.  Edge positions use smaller windows (minimum 5 residues).

    Parameters
    ----------
    seq:
        Protein sequence in single-letter uppercase code.
    window:
        Number of residues in each window (default 11).
    angle_deg:
        Angular increment per residue in degrees.
        Use 100.0 for alpha-helix; 160.0 for beta-strand.

    Returns
    -------
    list[float]
        Per-residue muH values; length == ``len(seq)``.

    References
    ----------
    Eisenberg, D., Weiss, R.M. & Terwilliger, T.C. (1984)
    Proc. Natl. Acad. Sci. USA 81:140-144.
    """
    n = len(seq)
    if n == 0:
        return []

    half = window // 2
    profile: list[float] = []

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        # Ensure minimum window of 5 residues at edges
        if hi - lo < 5:
            if i < half:
                hi = min(n, lo + 5)
            else:
                lo = max(0, hi - 5)
        sub = seq[lo:hi]
        profile.append(_hydrophobic_moment_window(sub, angle_deg))

    return profile


def calc_hydrophobic_moment_profile(
    seq: str,
    angle_deg: float = 100.0,
    window: int = 11,
) -> list[float]:
    """Alias for :func:`calc_hydrophobic_moment` with argument order matching
    the public API signature (angle_deg before window).

    Parameters
    ----------
    seq:
        Protein sequence in single-letter uppercase code.
    angle_deg:
        Angular increment per residue (default 100.0 for alpha-helix).
    window:
        Sliding window length.

    Returns
    -------
    list[float]
        Full per-residue muH profile.
    """
    return calc_hydrophobic_moment(seq, window=window, angle_deg=angle_deg)


def predict_amphipathic_helices(
    seq: str,
    window: int = 11,
    moment_threshold: float = 0.35,
    min_length: int = 7,
) -> list[dict]:
    """Identify amphipathic helices using the Eisenberg hydrophobic moment.

    A candidate helix is a contiguous run of residues where:

    1. The sliding-window muH (alpha-helix, delta = 100 degrees) >= *moment_threshold*.
    2. The mean Eisenberg hydrophobicity of the same window is between
       -0.5 and 1.5 (not purely hydrophobic, not purely hydrophilic).

    Parameters
    ----------
    seq:
        Protein sequence in single-letter uppercase code.
    window:
        Window size for hydrophobic moment calculation.
    moment_threshold:
        Minimum mean muH to flag a region as amphipathic (default 0.35).
    min_length:
        Minimum contiguous run length (default 7 aa).

    Returns
    -------
    list[dict]
        Each dict contains:

        ``start``
            0-based start position.
        ``end``
            0-based exclusive end position.
        ``seq``
            Sub-sequence.
        ``mean_moment``
            Mean muH over the region.
        ``mean_hydrophobicity``
            Mean Eisenberg hydrophobicity over the region.
        ``type``
            ``"membrane-binding"`` if mean hydrophobicity > 0.2,
            else ``"amphipathic"``.

    References
    ----------
    Eisenberg, D., Weiss, R.M. & Terwilliger, T.C. (1984)
    Proc. Natl. Acad. Sci. USA 81:140-144.
    """
    moment_profile = calc_hydrophobic_moment(seq, window=window, angle_deg=100.0)
    hydro_vals = [EISENBERG_SCALE.get(aa, 0.0) for aa in seq]
    n = len(seq)
    helices: list[dict] = []
    i = 0

    while i < n:
        mu = moment_profile[i]
        h = hydro_vals[i]
        if mu >= moment_threshold and -0.5 <= h <= 1.5:
            j = i
            while j < n:
                mu_j = moment_profile[j]
                h_j = hydro_vals[j]
                if mu_j >= moment_threshold and -0.5 <= h_j <= 1.5:
                    j += 1
                else:
                    break
            length = j - i
            if length >= min_length:
                sub = seq[i:j]
                mean_mu = sum(moment_profile[i:j]) / length
                mean_h = sum(hydro_vals[i:j]) / length
                helix_type = "membrane-binding" if mean_h > 0.2 else "amphipathic"
                helices.append({
                    'start': i,
                    'end': j,
                    'seq': sub,
                    'mean_moment': round(mean_mu, 4),
                    'mean_hydrophobicity': round(mean_h, 4),
                    'type': helix_type,
                })
            i = j
        else:
            i += 1

    return helices


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def format_amphipathic_report(seq: str, style_tag: str) -> str:
    """Generate HTML section for amphipathic helix analysis.

    Reports mean muH for alpha-helix (delta = 100 degrees) and beta-strand (delta = 160 degrees) angles,
    a summary statistics table, and a table of predicted amphipathic helices.

    Parameters
    ----------
    seq:
        Protein sequence in single-letter uppercase code.
    style_tag:
        Accent colour hex string (e.g. ``"#4361ee"``).

    Returns
    -------
    str
        Self-contained HTML fragment (includes ``<style>`` block).
    """
    accent = style_tag if style_tag else "#4361ee"
    _s = make_style_tag(accent)

    n = len(seq)
    if n == 0:
        return _s + "<h2>Amphipathic Helix Analysis</h2><p>Empty sequence.</p>"

    mu_helix = calc_hydrophobic_moment(seq, window=11, angle_deg=100.0)
    mu_strand = calc_hydrophobic_moment(seq, window=11, angle_deg=160.0)
    mean_mu_helix = sum(mu_helix) / n
    mean_mu_strand = sum(mu_strand) / n
    max_mu_helix = max(mu_helix)
    max_mu_strand = max(mu_strand)

    mean_h = sum(EISENBERG_SCALE.get(aa, 0.0) for aa in seq) / n

    helices = predict_amphipathic_helices(seq)
    n_helices = len(helices)

    # Amphipathicity class
    if mean_mu_helix >= 0.35:
        amph_class = "high (strong amphipathic character)"
    elif mean_mu_helix >= 0.20:
        amph_class = "moderate"
    else:
        amph_class = "low"

    summary_rows = (
        f"<tr><td>Mean &mu;H (&#945;-helix, &delta;=100&deg;)</td>"
        f"<td>{mean_mu_helix:.4f} &mdash; {amph_class}</td></tr>"
        f"<tr><td>Max &mu;H (&#945;-helix)</td><td>{max_mu_helix:.4f}</td></tr>"
        f"<tr><td>Mean &mu;H (&beta;-strand, &delta;=160&deg;)</td>"
        f"<td>{mean_mu_strand:.4f}</td></tr>"
        f"<tr><td>Max &mu;H (&beta;-strand)</td><td>{max_mu_strand:.4f}</td></tr>"
        f"<tr><td>Mean Eisenberg hydrophobicity</td><td>{mean_h:.4f}</td></tr>"
        f"<tr><td>Predicted amphipathic regions</td><td>{n_helices}</td></tr>"
    )

    summary_html = (
        "<h2>Amphipathic Helix Analysis (Eisenberg 1984)</h2>"
        "<table>"
        "<tr><th>Property</th><th>Value</th></tr>"
        f"{summary_rows}"
        "</table>"
        "<p class='note'>"
        "Hydrophobic moment: Eisenberg, Weiss &amp; Terwilliger (1984) "
        "Proc. Natl. Acad. Sci. USA 81:140. "
        "Window = 11 aa; &mu;H threshold = 0.35; min region = 7 aa."
        "</p>"
    )

    if helices:
        helix_header = (
            "<tr><th>Start</th><th>End</th><th>Sequence</th>"
            "<th>Mean &mu;H</th><th>Mean H</th><th>Type</th></tr>"
        )
        helix_rows = "".join(
            f"<tr>"
            f"<td>{h['start'] + 1}</td>"
            f"<td>{h['end']}</td>"
            f"<td><code>{h['seq']}</code></td>"
            f"<td>{h['mean_moment']:.4f}</td>"
            f"<td>{h['mean_hydrophobicity']:.4f}</td>"
            f"<td>{h['type']}</td>"
            f"</tr>"
            for h in helices
        )
        helix_html = (
            "<h2>Predicted Amphipathic Regions</h2>"
            "<table>"
            f"{helix_header}{helix_rows}"
            "</table>"
        )
    else:
        helix_html = (
            "<h2>Predicted Amphipathic Regions</h2>"
            "<p>No amphipathic regions detected "
            "(&mu;H &ge; 0.35, hydrophobicity &minus;0.5 to 1.5, &ge; 7 aa).</p>"
        )

    return _s + summary_html + helix_html
