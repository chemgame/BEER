"""RNA-binding propensity analysis."""
from __future__ import annotations
import re

from beer.constants import (
    RBP_RESIDUE_PROPENSITY,
    RNA_BINDING_MOTIFS,
    CHOU_FASMAN_HELIX,
    VDW_VOLUME,
    KYTE_DOOLITTLE,
)
from beer.reports.css import make_style_tag

# catRAPID weights (Bellucci et al. 2011 Nat Methods 8:444; Agostini et al. 2013)
_W_SP  = 0.0169   # secondary structure propensity
_W_HP  = 0.0117   # hydrophobicity (inverted KD, normalised to [-1,+1])
_W_VDW = 0.0283   # van der Waals volume


def _catrapid_residue(aa: str) -> float:
    """catRAPID per-residue propensity ω(i) = c₁·SP + c₂·HP + c₃·vdW."""
    sp  = CHOU_FASMAN_HELIX.get(aa, 1.0)
    hp  = -KYTE_DOOLITTLE.get(aa, 0.0) / 4.5  # invert & normalise KD to [-1,+1]
    vdw = VDW_VOLUME.get(aa, 0.5)
    return _W_SP * sp + _W_HP * hp + _W_VDW * vdw


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
    """Return RNA-binding residue composition, catRAPID composite score, and motif hits.

    Primary score: catRAPID-style composite (Bellucci et al. 2011 Nat Methods 8:444)
    combining secondary structure propensity, inverse hydrophobicity, and van der
    Waals volume.  The mean per-residue catRAPID score is reported as the main RNA-
    binding propensity index.  The Jeong et al. 2012 mean propensity is also
    retained for reference.

    Parameters
    ----------
    seq:
        Protein sequence in single-letter uppercase code.

    Returns
    -------
    dict
        ``catrapid_score``
            Mean per-residue catRAPID propensity ω̄ (higher = more likely RBP).
        ``mean_propensity``
            Mean per-residue Jeong et al. 2012 propensity (retained for reference).
        ``fraction_rbp_residues``
            Fraction of residues that are K, R, Y, F, or W.
        ``motifs_found``
            List of dicts ``{name, start, end, match}`` from motif scanning.

    References
    ----------
    Bellucci et al. (2011) Nat Methods 8:444. Agostini et al. (2013) Structure 21:1987.
    Jeong, E. et al. (2012) Nucleic Acids Res.
    """
    n = len(seq)
    if n == 0:
        return {
            'catrapid_score': 0.0,
            'mean_propensity': 0.0,
            'fraction_rbp_residues': 0.0,
            'motifs_found': [],
        }

    catrapid_vals = [_catrapid_residue(aa) for aa in seq]
    catrapid_mean = sum(catrapid_vals) / n

    props = [RBP_RESIDUE_PROPENSITY.get(aa, 0.0) for aa in seq]
    mean_prop = sum(props) / n
    rbp_residues = sum(1 for aa in seq if aa in 'KRYWF')
    frac_rbp = rbp_residues / n
    motifs = _scan_motifs(seq)

    return {
        'catrapid_score': round(catrapid_mean, 4),
        'mean_propensity': round(mean_prop, 4),
        'fraction_rbp_residues': round(frac_rbp, 4),
        'motifs_found': motifs,
    }


def calc_rbp_profile(seq: str, window: int = 11) -> list[float]:
    """Sliding-window catRAPID per-residue RNA-binding propensity profile.

    Uses the catRAPID composite ω(i) = c₁·SP + c₂·HP + c₃·vdW (Bellucci et al.
    2011 Nat Methods 8:444) rather than the Jeong et al. 2012 propensity.

    Parameters
    ----------
    seq:
        Protein sequence in single-letter uppercase code.
    window:
        Window size (default 11).

    Returns
    -------
    list[float]
        Mean catRAPID propensity per window; length = max(0, len(seq) - window + 1).
    """
    n = len(seq)
    if n == 0:
        return []
    vals = [_catrapid_residue(aa) for aa in seq]
    w = min(window, n)
    result: list[float] = []
    for i in range(n - w + 1):
        result.append(round(sum(vals[i:i + w]) / w, 4))
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

    cr = result['catrapid_score']
    cr_label = "high" if cr > 0.020 else ("moderate" if cr > 0.015 else "low")
    summary_rows = (
        f"<tr><td><b>catRAPID score &omega;&#772;</b></td>"
        f"<td><b>{cr:.4f}</b> &mdash; {cr_label} RNA-binding propensity</td></tr>"
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
        "<h2>RNA-Binding Propensity (catRAPID) &amp; Motifs</h2>"
        "<table>"
        "<tr><th>Property</th><th>Value</th></tr>"
        f"{summary_rows}"
        "</table>"
        "<p class='note'>"
        "catRAPID composite: &omega;(i) = 0.0169&sdot;SP + 0.0117&sdot;HP + 0.0283&sdot;vdW "
        "where SP = Chou-Fasman helix propensity, HP = inverse KD hydrophobicity, "
        "vdW = van der Waals contact volume. "
        "Bellucci et al. (2011) Nat Methods 8:444; Agostini et al. (2013) Structure 21:1987. "
        "Profile uses 11-residue sliding window. "
        "Motif annotations from published literature."
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
