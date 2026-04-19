"""RNA-binding propensity analysis."""
from __future__ import annotations
import re

from beer.constants import RBP_RESIDUE_PROPENSITY, RNA_BINDING_MOTIFS
from beer.reports.css import make_style_tag


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
    """Return RNA-binding residue composition and motif hits.

    Reports per-residue propensity (Jeong et al. 2012) and motif scanning.
    No composite score is computed — such scores require validation against
    experimental RBP data.

    Parameters
    ----------
    seq:
        Protein sequence in single-letter uppercase code.

    Returns
    -------
    dict
        ``mean_propensity``
            Mean per-residue RBP_RESIDUE_PROPENSITY across all residues.
        ``fraction_rbp_residues``
            Fraction of residues that are K, R, Y, F, or W.
        ``motifs_found``
            List of dicts ``{name, start, end, match}`` from motif scanning.

    References
    ----------
    Jeong, E. et al. (2012) Nucleic Acids Res.
    """
    n = len(seq)
    if n == 0:
        return {
            'mean_propensity': 0.0,
            'fraction_rbp_residues': 0.0,
            'motifs_found': [],
        }

    props = [RBP_RESIDUE_PROPENSITY.get(aa, 0.0) for aa in seq]
    mean_prop = sum(props) / n
    rbp_residues = sum(1 for aa in seq if aa in 'KRYWF')
    frac_rbp = rbp_residues / n
    motifs = _scan_motifs(seq)

    return {
        'mean_propensity': round(mean_prop, 4),
        'fraction_rbp_residues': round(frac_rbp, 4),
        'motifs_found': motifs,
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
    _s = make_style_tag(accent)

    result = calc_rbp_score(seq)
    motifs = result['motifs_found']
    n = len(seq)

    kr_n = sum(1 for aa in seq if aa in 'KR')
    y_n = sum(1 for aa in seq if aa == 'Y')
    f_n = sum(1 for aa in seq if aa == 'F')
    w_n = sum(1 for aa in seq if aa == 'W')

    summary_rows = (
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
        "<h2>RNA-Binding Composition &amp; Motifs</h2>"
        "<table>"
        "<tr><th>Property</th><th>Value</th></tr>"
        f"{summary_rows}"
        "</table>"
        "<p class='note'>"
        "Per-residue propensity scores are from Jeong et al. (2012) Nucleic Acids Res. "
        "Motif annotations are based on consensus sequences from published literature. "
        "No validated composite RBP score is computed."
        "</p>"
    )

    if motifs:
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
