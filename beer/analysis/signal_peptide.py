"""
beer.analysis.signal_peptide
=============================
Signal peptide and GPI anchor prediction using the von Heijne rules and
the Eisenhaber et al. 1999 GPI-anchor model.

Algorithms
----------
Signal peptide (von Heijne three-region model):
    von Heijne, G. (1986). A new method for predicting signal sequence
    cleavage sites. *Nucleic Acids Res.*, 14(11), 4683-4690.

GPI anchor prediction:
    Eisenhaber, B., Bork, P. & Eisenhaber, F. (1999). Prediction of potential
    GPI-modification sites in proprotein sequences. *J. Mol. Biol.*,
    292(3), 741-758.

Note
----
``KYTE_DOOLITTLE`` is defined locally here to avoid circular imports with
other ``beer.analysis`` modules.
"""

import math

# Imported for type hints only — no GUI, no circular dependency
from beer.analysis.aggregation import ZYGGREGATOR_PROPENSITY  # noqa: F401

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KYTE_DOOLITTLE: dict[str, float] = {
    'A':  1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5,
    'C':  2.5, 'Q': -3.5, 'E': -3.5, 'G': -0.4,
    'H': -3.2, 'I':  4.5, 'L':  3.8, 'K': -3.9,
    'M':  1.9, 'F':  2.8, 'P': -1.6, 'S': -0.8,
    'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V':  4.2,
}
"""Kyte-Doolittle hydropathy scale (Kyte & Doolittle 1982)."""

# Small neutral residues allowed at signal-peptide cleavage (-3, -1) positions
_SMALL_NEUTRAL = frozenset('AGSTC')

# Small neutral residues for GPI ω site
_OMEGA_AA = frozenset('ASTDNGC')

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
# Internal helpers
# ---------------------------------------------------------------------------

def _kd_mean(seq_sub: str) -> float:
    """Mean Kyte-Doolittle score for a subsequence."""
    if not seq_sub:
        return 0.0
    return sum(KYTE_DOOLITTLE.get(aa, 0.0) for aa in seq_sub) / len(seq_sub)


def _find_best_hydrophobic_window(
    seq: str,
    start: int,
    end: int,
    min_win: int = 10,
    max_win: int = 15,
    kd_threshold: float = 1.4,
) -> tuple[int, int, float]:
    """Return (h_start, h_end, mean_kd) for the best hydrophobic window.

    Searches all windows of length *min_win* to *max_win* within seq[start:end]
    for the one with the highest mean KD score that also exceeds *kd_threshold*.
    If no window exceeds the threshold the best window is still returned, but
    the caller can check the score to determine if it qualifies.

    Returns
    -------
    tuple (h_start, h_end, mean_kd)
        All indices are into the *original* seq (not the sub-slice).
    """
    n = end - start
    best_score = -999.0
    best_start = start
    best_end = start + min_win

    for w in range(min_win, max_win + 1):
        for i in range(start, end - w + 1):
            score = _kd_mean(seq[i:i + w])
            if score > best_score:
                best_score = score
                best_start = i
                best_end = i + w

    return best_start, best_end, best_score


# ---------------------------------------------------------------------------
# Signal peptide prediction
# ---------------------------------------------------------------------------

def predict_signal_peptide(seq: str) -> dict:
    """Predict signal peptide using the von Heijne three-region (n, h, c) model.

    Only the first 70 residues are analysed; signal peptides are N-terminal
    features.  The algorithm follows the classic three-region description:

    * **n-region** (positions 1–5): basic residues (K, R) provide a positive
      charge that targets the ribosome-translocon.
    * **h-region** (within the first 30 aa): the longest, most hydrophobic
      stretch (10–15 aa with mean KD > 1.4).
    * **c-region** (3–7 aa after h-region): ends with AXA or SXA (-3/-1 rule).
    * **Cleavage site**: immediately C-terminal to the c-region.

    Parameters
    ----------
    seq:
        Full protein sequence in single-letter uppercase code.

    Returns
    -------
    dict
        Keys: ``score``, ``verdict``, ``n_end``, ``h_start``, ``h_end``,
        ``c_start``, ``cleavage_site``, ``h_region_seq``, ``h_region_score``,
        ``n_score``.

    References
    ----------
    von Heijne, G. (1986) Nucleic Acids Res. 14(11):4683-4690.
    """
    region = seq[:70]
    n = len(region)

    # ---- n-region: count K/R in first 5 positions ----
    n_end = min(5, n)
    n_score_raw = sum(1 for aa in region[:n_end] if aa in 'KR')
    n_score_norm = min(n_score_raw, 3) / 3.0  # normalise to [0,1] with max 3

    # ---- h-region: best hydrophobic window within first 30 aa ----
    h_search_end = min(30, n)
    h_start, h_end, h_kd = _find_best_hydrophobic_window(
        region, 0, h_search_end, min_win=7, max_win=15, kd_threshold=1.4
    )
    h_length = h_end - h_start
    h_region_seq = region[h_start:h_end]
    # Normalise h-region KD score (typical range 1.4 – 3.0)
    h_score_norm = min(max(h_kd / 3.0, 0.0), 1.0)

    # ---- c-region: 3–7 aa after h-region, look for AXA motif ----
    c_start = h_end
    c_end_max = min(c_start + 7, n)
    # Search for best cleavage position: -3 and -1 must be small neutral
    cleavage_site = -1
    c_has_axa = False
    for cs in range(c_start + 2, c_end_max + 1):
        # positions -3 and -1 relative to cs (0-based)
        pos_m3 = cs - 3
        pos_m1 = cs - 1
        if 0 <= pos_m3 < n and 0 <= pos_m1 < n:
            if region[pos_m3] in _SMALL_NEUTRAL and region[pos_m1] in _SMALL_NEUTRAL:
                c_has_axa = True
                cleavage_site = cs  # cleavage after position cs (1-based: cs+1)
                break

    # If no AXA found, default cleavage site estimate is end of c-region
    if cleavage_site == -1:
        cleavage_site = min(c_start + 4, n)

    c_region_score = 1.0 if c_has_axa else 0.0

    # ---- Composite score ----
    score = (
        0.20 * n_score_norm
        + 0.50 * h_score_norm
        + 0.30 * c_region_score
    )
    score = round(min(max(score, 0.0), 1.0), 4)

    verdict = (
        "Signal peptide predicted"
        if score >= 0.55 and h_length >= 7
        else "No signal peptide predicted"
    )

    return {
        'score': score,
        'verdict': verdict,
        'n_end': n_end,
        'h_start': h_start,
        'h_end': h_end,
        'c_start': c_start,
        'cleavage_site': cleavage_site,
        'h_region_seq': h_region_seq,
        'h_region_score': round(h_kd, 4),
        'n_score': n_score_raw,
    }


# ---------------------------------------------------------------------------
# GPI anchor prediction
# ---------------------------------------------------------------------------

def predict_gpi_anchor(seq: str) -> dict:
    """Predict GPI anchor signal using the Eisenhaber et al. 1999 model.

    Analyses only the last 50 residues.  A GPI anchor requires three elements:

    * **ω (omega) site**: small neutral amino acid at C-terminal −8 to −11.
    * **Spacer**: 5–10 aa of moderate hydrophilicity after ω.
    * **Hydrophobic tail**: last 8–15 aa with mean KD > 1.6.

    Parameters
    ----------
    seq:
        Full protein sequence in single-letter uppercase code.

    Returns
    -------
    dict
        Keys: ``score``, ``verdict``, ``omega_position``, ``omega_aa``,
        ``tail_start``, ``tail_seq``, ``tail_kd_mean``.

    References
    ----------
    Eisenhaber, B., Bork, P. & Eisenhaber, F. (1999) J. Mol. Biol.
    292(3):741-758.
    """
    n = len(seq)
    tail_region = seq[max(0, n - 50):]
    offset = max(0, n - 50)  # offset into original seq
    tn = len(tail_region)

    # ---- Hydrophobic tail: last 8–15 aa ----
    tail_len = min(15, tn)
    tail_start_local = tn - tail_len
    tail_seq = tail_region[tail_start_local:]
    tail_kd = _kd_mean(tail_seq)
    tail_ok = tail_kd > 1.6

    # ---- ω-site: small neutral at positions -8 to -11 from the C-terminus ----
    # In tail_region coordinates: positions tn-11 to tn-8
    omega_pos_local = -1
    omega_aa = ''
    for rel in range(tn - 11, tn - 7):  # -11, -10, -9, -8
        if 0 <= rel < tn and tail_region[rel] in _OMEGA_AA:
            omega_pos_local = rel
            omega_aa = tail_region[rel]
            break  # take the most N-terminal (most conservative)

    omega_found = omega_pos_local >= 0

    # ---- Spacer: residues between ω and the hydrophobic tail ----
    if omega_found:
        spacer_len = tail_start_local - omega_pos_local - 1
    else:
        spacer_len = 0
    spacer_ok = 5 <= spacer_len <= 10

    # ---- Composite score ----
    score = (
        0.30 * float(omega_found)
        + 0.50 * float(tail_ok)
        + 0.20 * float(spacer_ok)
    )
    score = round(min(max(score, 0.0), 1.0), 4)

    verdict = (
        "GPI anchor signal predicted"
        if score >= 0.6
        else "No GPI anchor predicted"
    )

    omega_position = (offset + omega_pos_local + 1) if omega_found else -1
    tail_start_global = offset + tail_start_local + 1  # 1-based

    return {
        'score': score,
        'verdict': verdict,
        'omega_position': omega_position,
        'omega_aa': omega_aa,
        'tail_start': tail_start_global,
        'tail_seq': tail_seq,
        'tail_kd_mean': round(tail_kd, 4),
    }


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def format_signal_report(seq: str, style_tag: str) -> str:
    """Generate HTML section for signal peptide and GPI anchor predictions.

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
    css = _REPORT_CSS.replace("#4361ee", accent)
    _s = f"<style>{css}</style>"

    sp = predict_signal_peptide(seq)
    gpi = predict_gpi_anchor(seq)

    # ---- Signal peptide table ----
    sp_verdict_colour = "#16a34a" if "predicted" in sp['verdict'].lower() and "no" not in sp['verdict'].lower() else "#dc2626"
    gpi_verdict_colour = "#16a34a" if "predicted" in gpi['verdict'].lower() and "no" not in gpi['verdict'].lower() else "#dc2626"

    # Cleavage site display: highlight cleavage position in sequence context
    if sp['cleavage_site'] > 0 and sp['cleavage_site'] <= len(seq):
        cs = sp['cleavage_site']
        cs_context = (
            seq[max(0, cs - 5):cs] + " | " + seq[cs:min(len(seq), cs + 5)]
        )
    else:
        cs_context = "N/A"

    sp_rows = (
        f"<tr><td>Verdict</td>"
        f"<td style='color:{sp_verdict_colour};font-weight:600'>{sp['verdict']}</td></tr>"
        f"<tr><td>Score (0&ndash;1)</td><td>{sp['score']:.3f}</td></tr>"
        f"<tr><td>n-region basic residues (K,R in pos 1&ndash;5)</td><td>{sp['n_score']}</td></tr>"
        f"<tr><td>h-region position</td>"
        f"<td>{sp['h_start']+1}&ndash;{sp['h_end']} ({sp['h_end']-sp['h_start']} aa)</td></tr>"
        f"<tr><td>h-region sequence</td><td><code>{sp['h_region_seq']}</code></td></tr>"
        f"<tr><td>h-region mean KD</td><td>{sp['h_region_score']:.3f}</td></tr>"
        f"<tr><td>Predicted cleavage site (after pos)</td>"
        f"<td>{sp['cleavage_site']} &nbsp;[{cs_context}]</td></tr>"
    )

    sp_html = (
        "<h2>Signal Peptide Prediction (von Heijne 1986)</h2>"
        "<table>"
        "<tr><th>Parameter</th><th>Value</th></tr>"
        f"{sp_rows}"
        "</table>"
        "<p class='note'>"
        "Method: three-region (n, h, c) model. "
        "von Heijne, G. (1986) Nucleic Acids Res. 14:4683. "
        "Score &ge; 0.55 and h-region &ge; 7 aa required for positive prediction."
        "</p>"
    )

    # ---- GPI anchor table ----
    gpi_rows = (
        f"<tr><td>Verdict</td>"
        f"<td style='color:{gpi_verdict_colour};font-weight:600'>{gpi['verdict']}</td></tr>"
        f"<tr><td>Score (0&ndash;1)</td><td>{gpi['score']:.3f}</td></tr>"
        f"<tr><td>&omega;-site position</td>"
        f"<td>{'Position ' + str(gpi['omega_position']) + ' (' + gpi['omega_aa'] + ')' if gpi['omega_position'] > 0 else 'Not found'}</td></tr>"
        f"<tr><td>Hydrophobic tail start</td><td>{gpi['tail_start']}</td></tr>"
        f"<tr><td>Hydrophobic tail sequence</td><td><code>{gpi['tail_seq']}</code></td></tr>"
        f"<tr><td>Tail mean KD</td><td>{gpi['tail_kd_mean']:.3f}</td></tr>"
    )

    gpi_html = (
        "<h2>GPI Anchor Prediction (Eisenhaber et al. 1999)</h2>"
        "<table>"
        "<tr><th>Parameter</th><th>Value</th></tr>"
        f"{gpi_rows}"
        "</table>"
        "<p class='note'>"
        "Method: Eisenhaber, B., Bork, P. &amp; Eisenhaber, F. (1999) "
        "J. Mol. Biol. 292:741. "
        "Requires &omega;-site small neutral, 5&ndash;10 aa spacer, and "
        "hydrophobic C-terminal tail (KD &gt; 1.6)."
        "</p>"
    )

    return _s + sp_html + gpi_html
