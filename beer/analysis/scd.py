"""
beer.analysis.scd
==================
Sequence Charge Decoration (SCD) and related charge-patterning metrics for
intrinsically disordered proteins (IDPs).

The SCD metric quantifies the patterning of charged residues along the
sequence.  Positive SCD values indicate same-sign charge clustering
(polyelectrolyte-like behaviour), while negative values indicate
alternating charge patterns characteristic of well-mixed polyampholytes.

References
----------
SCD definition:
    Sawle, L. & Ghosh, K. (2015). A theoretical method to compute sequence
    dependent configurational properties in charged polymers and proteins.
    *J. Chem. Phys.*, 143(8), 085101.

Charge segregation / κ:
    Das, R.K. & Pappu, R.V. (2013). Conformations of intrinsically disordered
    proteins are influenced by linear sequence distributions of oppositely
    charged residues. *Proc. Natl. Acad. Sci. USA*, 110(33), 13392-13397.
"""

import math

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
# Charge assignment
# ---------------------------------------------------------------------------

def _charge_vector(seq: str) -> list[int]:
    """Assign +1 (K, R), -1 (D, E), 0 (all others) to each position."""
    result: list[int] = []
    for aa in seq:
        if aa in 'KR':
            result.append(1)
        elif aa in 'DE':
            result.append(-1)
        else:
            result.append(0)
    return result


# ---------------------------------------------------------------------------
# SCD
# ---------------------------------------------------------------------------

def calc_scd(seq: str) -> float:
    """Compute Sequence Charge Decoration (SCD).

    SCD = (1/N) · Σ_{i<j} σ_i · σ_j · |i−j|^0.5

    where σ_i = +1 for K/R, σ_i = −1 for D/E, σ_i = 0 otherwise, and N is
    the sequence length.

    Parameters
    ----------
    seq:
        Protein sequence in single-letter uppercase code.

    Returns
    -------
    float
        SCD value.  Positive: same-sign charge clustering; negative: mixed
        polyampholyte-like patterning (Sawle & Ghosh 2015).

    References
    ----------
    Sawle, L. & Ghosh, K. (2015) J. Chem. Phys. 143:085101.
    """
    n = len(seq)
    if n < 2:
        return 0.0
    sigma = _charge_vector(seq)
    total = 0.0
    for i in range(n):
        si = sigma[i]
        if si == 0:
            continue
        for j in range(i + 1, n):
            sj = sigma[j]
            if sj == 0:
                continue
            total += si * sj * math.sqrt(j - i)
    return total / n


def calc_scd_profile(seq: str, window: int = 20) -> list[float]:
    """Sliding-window SCD profile.

    Computes SCD for each successive window of length *window* along the
    sequence.  Returns one value per window start position.

    Parameters
    ----------
    seq:
        Protein sequence in single-letter uppercase code.
    window:
        Window length (default 20).

    Returns
    -------
    list[float]
        SCD values; length = max(0, len(seq) - window + 1).
    """
    n = len(seq)
    if n < window:
        return [calc_scd(seq)] if n > 0 else []
    return [calc_scd(seq[i:i + window]) for i in range(n - window + 1)]


# ---------------------------------------------------------------------------
# Charge segregation score
# ---------------------------------------------------------------------------

def calc_charge_segregation_score(seq: str) -> float:
    """Compute charge segregation score within a ±5 residue neighbourhood.

    Score = (n_same_sign_pairs − n_opposite_sign_pairs) / total_charged_pairs

    where pairs are all (i, j) with |i−j| ≤ 5 where both σ_i ≠ 0 and σ_j ≠ 0.

    Parameters
    ----------
    seq:
        Protein sequence in single-letter uppercase code.

    Returns
    -------
    float
        Score in [−1, +1].  +1 = fully charge-segregated (polyelectrolyte);
        −1 = perfectly mixed (polyampholyte).  Returns 0.0 for sequences with
        < 2 charged residues.

    References
    ----------
    Das, R.K. & Pappu, R.V. (2013) Proc. Natl. Acad. Sci. USA 110:13392.
    """
    sigma = _charge_vector(seq)
    n = len(sigma)
    n_same = 0
    n_opp = 0
    for i in range(n):
        if sigma[i] == 0:
            continue
        for j in range(i + 1, min(i + 6, n)):
            if sigma[j] == 0:
                continue
            if sigma[i] * sigma[j] > 0:
                n_same += 1
            else:
                n_opp += 1
    total = n_same + n_opp
    if total == 0:
        return 0.0
    return (n_same - n_opp) / total


# ---------------------------------------------------------------------------
# Block length statistics
# ---------------------------------------------------------------------------

def calc_mean_block_length(seq: str, residue_set: set) -> float:
    """Mean length of contiguous blocks of residues in *residue_set*.

    Parameters
    ----------
    seq:
        Protein sequence in single-letter uppercase code.
    residue_set:
        Set of single-letter amino acid codes defining the block type.

    Returns
    -------
    float
        Mean block length.  Returns 0.0 if no residues of the given type exist.
    """
    blocks: list[int] = []
    current = 0
    for aa in seq:
        if aa in residue_set:
            current += 1
        else:
            if current > 0:
                blocks.append(current)
                current = 0
    if current > 0:
        blocks.append(current)
    return sum(blocks) / len(blocks) if blocks else 0.0


def calc_pos_neg_block_lengths(seq: str) -> dict:
    """Compute block length statistics for positive (K, R) and negative (D, E) residues.

    Parameters
    ----------
    seq:
        Protein sequence in single-letter uppercase code.

    Returns
    -------
    dict
        ``mean_pos_block``
            Mean length of contiguous K/R runs.
        ``mean_neg_block``
            Mean length of contiguous D/E runs.
        ``max_pos_block``
            Longest K/R run.
        ``max_neg_block``
            Longest D/E run.
    """
    def _block_lengths(s: str, rs: set) -> list[int]:
        blocks: list[int] = []
        cur = 0
        for aa in s:
            if aa in rs:
                cur += 1
            else:
                if cur > 0:
                    blocks.append(cur)
                    cur = 0
        if cur > 0:
            blocks.append(cur)
        return blocks

    pos_blocks = _block_lengths(seq, {'K', 'R'})
    neg_blocks = _block_lengths(seq, {'D', 'E'})

    return {
        'mean_pos_block': round(sum(pos_blocks) / len(pos_blocks), 3) if pos_blocks else 0.0,
        'mean_neg_block': round(sum(neg_blocks) / len(neg_blocks), 3) if neg_blocks else 0.0,
        'max_pos_block': max(pos_blocks) if pos_blocks else 0,
        'max_neg_block': max(neg_blocks) if neg_blocks else 0,
    }


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def _scd_interpretation(scd: float) -> str:
    """Return a human-readable interpretation of the SCD value."""
    if scd < -1.0:
        return "well-mixed polyampholyte"
    elif scd < 0.0:
        return "mixed"
    elif scd < 1.0:
        return "mildly segregated"
    else:
        return "strongly charge-segregated"


def format_scd_report(seq: str, style_tag: str) -> str:
    """Generate HTML section for SCD and charge-patterning analysis.

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

    n = len(seq)
    if n == 0:
        return _s + "<h2>Charge Patterning (SCD)</h2><p>Empty sequence.</p>"

    scd = calc_scd(seq)
    interp = _scd_interpretation(scd)
    seg_score = calc_charge_segregation_score(seq)
    blocks = calc_pos_neg_block_lengths(seq)

    sigma = _charge_vector(seq)
    n_pos = sum(1 for s in sigma if s > 0)
    n_neg = sum(1 for s in sigma if s < 0)
    fcr = (n_pos + n_neg) / n if n > 0 else 0.0
    ncpr = (n_pos - n_neg) / n if n > 0 else 0.0

    seg_interp = (
        "strongly segregated (+)" if seg_score > 0.5 else
        "moderately segregated" if seg_score > 0.2 else
        "well mixed" if seg_score > -0.2 else
        "alternating / polyampholyte-like"
    )

    rows = (
        f"<tr><td>Sequence Charge Decoration (SCD)</td>"
        f"<td>{scd:.4f} &mdash; {interp}</td></tr>"
        f"<tr><td>Charge Segregation Score (&#177;5 aa)</td>"
        f"<td>{seg_score:.4f} &mdash; {seg_interp}</td></tr>"
        f"<tr><td>Positive residues (K, R)</td><td>{n_pos} ({n_pos/n*100:.1f}%)</td></tr>"
        f"<tr><td>Negative residues (D, E)</td><td>{n_neg} ({n_neg/n*100:.1f}%)</td></tr>"
        f"<tr><td>FCR (fraction charged)</td><td>{fcr:.3f}</td></tr>"
        f"<tr><td>NCPR (net charge per residue)</td><td>{ncpr:+.3f}</td></tr>"
        f"<tr><td>Mean positive block length (K, R)</td>"
        f"<td>{blocks['mean_pos_block']:.2f}</td></tr>"
        f"<tr><td>Mean negative block length (D, E)</td>"
        f"<td>{blocks['mean_neg_block']:.2f}</td></tr>"
        f"<tr><td>Max positive block (K, R)</td><td>{blocks['max_pos_block']}</td></tr>"
        f"<tr><td>Max negative block (D, E)</td><td>{blocks['max_neg_block']}</td></tr>"
    )

    html = (
        "<h2>Sequence Charge Decoration (SCD)</h2>"
        "<table>"
        "<tr><th>Property</th><th>Value</th></tr>"
        f"{rows}"
        "</table>"
        "<p class='note'>"
        "SCD: Sawle &amp; Ghosh (2015) J. Chem. Phys. 143:085101. "
        "Interpretation: SCD &lt; &minus;1 = well-mixed polyampholyte; "
        "&minus;1 to 0 = mixed; 0 to 1 = mildly segregated; "
        "&gt; 1 = strongly charge-segregated."
        "</p>"
    )

    return _s + html
