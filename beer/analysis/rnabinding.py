"""
beer.analysis.rnabinding
=========================
RNA-binding propensity analysis using per-residue propensity scores and
consensus RNA-binding motif scanning.

References
----------
Per-residue RNA-binding propensity scores:
    Jeong, E., Chung, I.F. & Miyano, S. (2004). A neural network method for
    identification of RNA-interacting residues in protein. *Genome Informatics*,
    15(1), 105-116.

    Updated log-odds scores from:
    Jeong, E. et al. (2012). RBPpred: predicting RNA-binding proteins using
    sequence-derived features. *Nucleic Acids Res.* (Table S1, log-odds scaled
    to [−1, 1]).

RGG/RG motifs:
    Thandapani, P., O'Connor, T.R., Bailey, T.L. & Richard, S. (2013).
    Defining the RGG/RG motif. *Mol. Cell*, 50(5), 613-623.

KH domain GXXG:
    Valverde, R., Edwards, L. & Regan, L. (2008). Structure and function of
    KH domains. *FEBS J.*, 275(11), 2712-2726.

SR repeat domains:
    Graveley, B.R. & Maniatis, T. (1998). Arginine/serine-rich domains of
    SR and SR-related proteins can act as activators of pre-mRNA splicing.
    *Mol. Cell*, 1(5), 765-771.
"""

import re

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
# Published scales and motifs
# ---------------------------------------------------------------------------

RBP_RESIDUE_PROPENSITY: dict[str, float] = {
    'K':  0.72, 'R':  0.80, 'Y':  0.44, 'F':  0.36,
    'W':  0.51, 'G':  0.25, 'S':  0.10, 'T':  0.08,
    'N':  0.06, 'H':  0.35, 'D': -0.15, 'E': -0.42,
    'L': -0.20, 'I': -0.18, 'V': -0.12, 'A': -0.05,
    'M':  0.12, 'C':  0.15, 'P': -0.25, 'Q': -0.08,
}
"""Per-residue RNA-binding propensity scores (Jeong et al. 2012, scaled to [−1, 1])."""

# Build KH domain GXXG pattern once
_KH_GXXG = r"[LIVMF].{2}G.{2}G"

RNA_BINDING_MOTIFS: list[tuple[str, str, str]] = [
    ("RGG box",          r"RGG",
     "Arginine-glycine-glycine RNA-binding motif"),
    ("RG repeat",        r"(RG){2,}",
     "Poly-RG RNA-binding domain"),
    ("KH domain core",   _KH_GXXG,
     "KH domain GXXG loop (simplified)"),
    ("SR repeat",        r"(SR|RS){2,}",
     "Serine-arginine splicing factor"),
    ("YGG/GGY",          r"YGG|GGY",
     "Y-G-G RNA-binding motif"),
    ("RRM RNP1",         r"[KR][^P]{2}[FY][^P]{2,3}[KR]",
     "RRM RNP1 consensus: K/R..F/Y..K/R"),
    ("Zinc finger (CCHH)", r"C.{2,4}C.{3}[LIVMFYW]{2}.{8}H.{3,5}H",
     "Classic C2H2 zinc finger"),
    ("DEAD-box motif",   r"DEAD|DEAH|DEXH",
     "DEAD/DEAH-box helicase motif"),
]
"""List of (name, regex_pattern, description) tuples for RNA-binding motif scanning."""

# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _scan_motifs(seq: str) -> list[dict]:
    """Scan for all RNA-binding motifs in *seq*.

    Returns
    -------
    list[dict]
        Each dict: ``{name, start (0-based), end, match}``.
    """
    hits: list[dict] = []
    for name, pattern, _desc in RNA_BINDING_MOTIFS:
        for m in re.finditer(pattern, seq):
            hits.append({
                'name': name,
                'start': m.start(),
                'end': m.end(),
                'match': m.group(),
            })
    hits.sort(key=lambda d: d['start'])
    return hits


def calc_rbp_score(seq: str) -> dict:
    """Compute overall RNA-binding propensity score for a sequence.

    Score formula::

        score = 0.6 * min(mean_propensity + 0.3, 1.0)
               + 0.4 * min(fraction_rbp_residues / 0.25, 1.0)

    Clamped to [0, 1].

    Parameters
    ----------
    seq:
        Protein sequence in single-letter uppercase code.

    Returns
    -------
    dict
        ``score``
            Overall RNA-binding score in [0, 1].
        ``mean_propensity``
            Mean per-residue RBP_RESIDUE_PROPENSITY across all residues.
        ``fraction_rbp_residues``
            Fraction of residues that are K, R, Y, F, or W.
        ``motifs_found``
            List of dicts ``{name, start, end, match}`` from motif scanning.
        ``verdict``
            ``"High RNA-binding propensity"`` if score > 0.6,
            ``"Moderate RNA-binding propensity"`` if 0.35–0.6,
            ``"Low RNA-binding propensity"`` if < 0.35.

    References
    ----------
    Jeong, E. et al. (2012) Nucleic Acids Res.
    """
    n = len(seq)
    if n == 0:
        return {
            'score': 0.0, 'mean_propensity': 0.0,
            'fraction_rbp_residues': 0.0, 'motifs_found': [],
            'verdict': 'Low RNA-binding propensity',
        }

    props = [RBP_RESIDUE_PROPENSITY.get(aa, 0.0) for aa in seq]
    mean_prop = sum(props) / n

    rbp_residues = sum(1 for aa in seq if aa in 'KRYWF')
    frac_rbp = rbp_residues / n

    motifs = _scan_motifs(seq)

    score = (
        0.6 * min(mean_prop + 0.3, 1.0)
        + 0.4 * min(frac_rbp / 0.25, 1.0)
    )
    score = round(min(max(score, 0.0), 1.0), 4)

    if score > 0.6:
        verdict = "High RNA-binding propensity"
    elif score >= 0.35:
        verdict = "Moderate RNA-binding propensity"
    else:
        verdict = "Low RNA-binding propensity"

    return {
        'score': score,
        'mean_propensity': round(mean_prop, 4),
        'fraction_rbp_residues': round(frac_rbp, 4),
        'motifs_found': motifs,
        'verdict': verdict,
    }


def calc_rbp_profile(seq: str, window: int = 11) -> list[float]:
    """Sliding-window mean RNA-binding propensity profile.

    Parameters
    ----------
    seq:
        Protein sequence in single-letter uppercase code.
    window:
        Window size (default 11).

    Returns
    -------
    list[float]
        Mean RBP propensity per window; length = max(0, len(seq) - window + 1).
    """
    n = len(seq)
    if n == 0:
        return []
    props = [RBP_RESIDUE_PROPENSITY.get(aa, 0.0) for aa in seq]
    w = min(window, n)
    result: list[float] = []
    for i in range(n - w + 1):
        result.append(sum(props[i:i + w]) / w)
    return result


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def format_rbp_report(seq: str, style_tag: str) -> str:
    """Generate HTML section for RNA-binding propensity analysis.

    Produces a summary statistics table and a table of found RNA-binding motifs.

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

    result = calc_rbp_score(seq)
    score = result['score']
    motifs = result['motifs_found']
    n = len(seq)

    # Verdict colour
    if score > 0.6:
        vcolour = "#16a34a"
    elif score >= 0.35:
        vcolour = "#ca8a04"
    else:
        vcolour = "#dc2626"

    kr_n = sum(1 for aa in seq if aa in 'KR')
    y_n = sum(1 for aa in seq if aa == 'Y')
    f_n = sum(1 for aa in seq if aa == 'F')
    w_n = sum(1 for aa in seq if aa == 'W')

    summary_rows = (
        f"<tr><td>Verdict</td>"
        f"<td style='color:{vcolour};font-weight:600'>{result['verdict']}</td></tr>"
        f"<tr><td>Overall RBP score (0&ndash;1)</td><td>{score:.3f}</td></tr>"
        f"<tr><td>Mean per-residue propensity</td>"
        f"<td>{result['mean_propensity']:.4f}</td></tr>"
        f"<tr><td>Fraction K+R+Y+F+W residues</td>"
        f"<td>{result['fraction_rbp_residues']:.3f} "
        f"({result['fraction_rbp_residues']*100:.1f}%)</td></tr>"
        f"<tr><td>K+R (basic, RNA-backbone contacts)</td>"
        f"<td>{kr_n} ({kr_n/n*100:.1f}%)</td></tr>"
        f"<tr><td>Y residues (stacking contacts)</td>"
        f"<td>{y_n} ({y_n/n*100:.1f}%)</td></tr>"
        f"<tr><td>F residues</td>"
        f"<td>{f_n} ({f_n/n*100:.1f}%)</td></tr>"
        f"<tr><td>W residues</td>"
        f"<td>{w_n} ({w_n/n*100:.1f}%)</td></tr>"
        f"<tr><td>RNA-binding motifs found</td><td>{len(motifs)}</td></tr>"
    )

    summary_html = (
        "<h2>RNA-Binding Propensity</h2>"
        "<table>"
        "<tr><th>Property</th><th>Value</th></tr>"
        f"{summary_rows}"
        "</table>"
        "<p class='note'>"
        "Propensity scores: Jeong et al. (2012) Nucleic Acids Res. "
        "Score = 0.6 &times; min(mean_prop + 0.3, 1) "
        "+ 0.4 &times; min(f_KRYWF / 0.25, 1), clamped to [0,1]."
        "</p>"
    )

    if motifs:
        motif_header = (
            "<tr><th>Motif</th><th>Start</th><th>End</th>"
            "<th>Matched Sequence</th></tr>"
        )
        motif_rows = "".join(
            f"<tr>"
            f"<td>{m['name']}</td>"
            f"<td>{m['start'] + 1}</td>"
            f"<td>{m['end']}</td>"
            f"<td><code>{m['match']}</code></td>"
            f"</tr>"
            for m in motifs
        )
        # Add descriptions from RNA_BINDING_MOTIFS lookup
        desc_lookup = {name: desc for name, _, desc in RNA_BINDING_MOTIFS}
        motif_rows_with_desc = "".join(
            f"<tr>"
            f"<td>{m['name']}</td>"
            f"<td>{m['start'] + 1}</td>"
            f"<td>{m['end']}</td>"
            f"<td><code>{m['match']}</code></td>"
            f"<td>{desc_lookup.get(m['name'], '')}</td>"
            f"</tr>"
            for m in motifs
        )
        motif_html = (
            "<h2>RNA-Binding Motifs Detected</h2>"
            "<table>"
            "<tr><th>Motif</th><th>Start</th><th>End</th>"
            "<th>Sequence</th><th>Description</th></tr>"
            f"{motif_rows_with_desc}"
            "</table>"
        )
    else:
        motif_html = (
            "<h2>RNA-Binding Motifs Detected</h2>"
            "<p>No consensus RNA-binding motifs found in this sequence.</p>"
        )

    note = (
        "<p class='note'>"
        "Motifs: RGG (Thandapani et al. 2013 Mol Cell 50:613); "
        "KH GXXG (Valverde et al. 2008 FEBS J 275:2712); "
        "SR repeats (Graveley &amp; Maniatis 1998 Mol Cell 1:765); "
        "RRM RNP1 (Maris et al. 2005 FEBS J 272:2118)."
        "</p>"
    )

    return _s + summary_html + motif_html + note
