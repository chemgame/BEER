"""Proteolytic cleavage site prediction for protein sequences."""
from __future__ import annotations

import re

from beer.reports.css import make_style_tag

# ---------------------------------------------------------------------------
# Monoisotopic residue masses (Da)
# ---------------------------------------------------------------------------

_RESIDUE_MASS: dict[str, float] = {
    "A": 71.037,
    "R": 156.101,
    "N": 114.043,
    "D": 115.027,
    "C": 103.009,
    "E": 129.043,
    "Q": 128.059,
    "G": 57.021,
    "H": 137.059,
    "I": 113.084,
    "L": 113.084,
    "K": 128.095,
    "M": 131.040,
    "F": 147.068,
    "P": 97.053,
    "S": 87.032,
    "T": 101.048,
    "W": 186.079,
    "Y": 163.063,
    "V": 99.068,
}

_WATER_MASS = 18.011

# ---------------------------------------------------------------------------
# Enzyme cleavage rules
# ---------------------------------------------------------------------------

# Each rule is either:
#   ("after",  regex_pattern)  – cleavage occurs AFTER the matched residue
#   ("before", regex_pattern)  – cleavage occurs BEFORE the matched residue
#
# Positions returned are always 1-based.

_ENZYME_RULES: dict[str, tuple[str, str]] = {
    "Trypsin":              ("after",  r"[KR](?!P)"),
    "Chymotrypsin (high)":  ("after",  r"[FYW](?!P)"),
    "Chymotrypsin (low)":   ("after",  r"[FYWML](?!P)"),
    "Lys-C":                ("after",  r"K"),
    "Asp-N":                ("before", r"D"),
    "Glu-C (pH 8)":         ("after",  r"E"),
    "Glu-C (pH 4)":         ("after",  r"[ED]"),
    "CNBr":                 ("after",  r"M"),
    "Arg-C":                ("after",  r"R(?!P)"),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calc_proteolytic_sites(seq: str) -> dict[str, list[int]]:
    """Predict proteolytic cleavage sites for common enzymes.

    Parameters
    ----------
    seq:
        Protein sequence in single-letter uppercase code.

    Returns
    -------
    dict
        Keys are enzyme names; values are sorted lists of 1-based cleavage
        positions.  A position *p* means cleavage occurs after residue *p*
        (i.e. between residues *p* and *p+1*).  For Asp-N the position is
        the 1-based index of the D residue itself (cleavage before D).
    """
    seq = seq.upper()
    result: dict[str, list[int]] = {}

    for enzyme, (mode, pattern) in _ENZYME_RULES.items():
        positions: list[int] = []
        for m in re.finditer(pattern, seq):
            if mode == "after":
                # Position of the matched residue (1-based)
                pos = m.start() + 1
                # Only record if the cleavage is not at the very end
                if pos < len(seq):
                    positions.append(pos)
            else:  # "before"
                # The D is at m.start(); cleavage is before it.
                # Return the 1-based index of D (= position after which we
                # cut in the preceding fragment).
                pos = m.start() + 1
                # Skip cleavage at position 1 (nothing before it)
                if pos > 1:
                    positions.append(pos - 1)
        result[enzyme] = sorted(positions)

    return result


def calc_peptide_masses(seq: str, sites: list[int]) -> list[dict]:
    """Compute peptide fragments and their monoisotopic masses.

    Parameters
    ----------
    seq:
        Full protein sequence (uppercase single-letter code).
    sites:
        Sorted list of 1-based cleavage positions (after which cleavage occurs).

    Returns
    -------
    list[dict]
        Each dict has keys:

        ``start_1based``   – 1-based start position of the peptide.
        ``end_1based``     – 1-based end position of the peptide.
        ``peptide_seq``    – Amino acid sequence of the peptide.
        ``mass_da``        – Monoisotopic mass (Da) = sum of residue masses + water.
    """
    seq = seq.upper()
    n = len(seq)
    if n == 0:
        return []

    # Build boundary list: split points are 0-based start indices of each
    # peptide fragment.
    boundaries: list[int] = [0]
    for s in sorted(sites):
        if 1 <= s < n:
            boundaries.append(s)  # 0-based start of next peptide = site (1-based after = 0-based next)
    boundaries.append(n)

    peptides: list[dict] = []
    for i in range(len(boundaries) - 1):
        start0 = boundaries[i]
        end0 = boundaries[i + 1]
        pep = seq[start0:end0]
        if not pep:
            continue
        mass = sum(_RESIDUE_MASS.get(aa, 0.0) for aa in pep) + _WATER_MASS
        peptides.append({
            "start_1based": start0 + 1,
            "end_1based": end0,
            "peptide_seq": pep,
            "mass_da": round(mass, 4),
        })

    return peptides


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def format_proteolysis_report(seq: str, accent_color: str) -> str:
    """Generate an HTML proteolysis report for a protein sequence.

    Parameters
    ----------
    seq:
        Protein sequence in single-letter uppercase code.
    accent_color:
        Hex colour string used to style the report (e.g. ``"#4361ee"``).

    Returns
    -------
    str
        Self-contained HTML fragment (no ``<html>`` / ``<body>`` wrapper).
    """
    seq = seq.upper()
    n = len(seq)

    all_sites = calc_proteolytic_sites(seq)

    # ------------------------------------------------------------------ #
    # Build per-enzyme summary statistics
    # ------------------------------------------------------------------ #
    summary_rows: list[dict] = []
    for enzyme, sites in all_sites.items():
        peptides = calc_peptide_masses(seq, sites)
        lengths = [p["end_1based"] - p["start_1based"] + 1 for p in peptides]
        n_cuts = len(sites)
        n_peps = len(peptides)
        min_len = min(lengths) if lengths else 0
        max_len = max(lengths) if lengths else 0
        avg_len = round(sum(lengths) / len(lengths), 1) if lengths else 0.0
        summary_rows.append({
            "enzyme":   enzyme,
            "n_cuts":   n_cuts,
            "n_peps":   n_peps,
            "min_len":  min_len,
            "max_len":  max_len,
            "avg_len":  avg_len,
        })

    # ------------------------------------------------------------------ #
    # HTML helpers
    # ------------------------------------------------------------------ #
    def _th(*headers: str) -> str:
        return "<tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>"

    def _td(*cells) -> str:
        return "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"

    def _trunc(s: str, n: int = 20) -> str:
        return s if len(s) <= n else s[:n] + "..."

    # ------------------------------------------------------------------ #
    # Summary table
    # ------------------------------------------------------------------ #
    summary_header = _th("Enzyme", "# Cuts", "# Peptides",
                         "Min length (aa)", "Max length (aa)", "Avg length (aa)")
    summary_body = "\n".join(
        _td(r["enzyme"], r["n_cuts"], r["n_peps"],
            r["min_len"], r["max_len"], r["avg_len"])
        for r in summary_rows
    )
    summary_table = (
        f"<table>\n<thead>{summary_header}</thead>\n"
        f"<tbody>\n{summary_body}\n</tbody>\n</table>"
    )

    # ------------------------------------------------------------------ #
    # Trypsin detailed peptide table
    # ------------------------------------------------------------------ #
    trypsin_sites = all_sites.get("Trypsin", [])
    trypsin_peptides = calc_peptide_masses(seq, trypsin_sites)

    trypsin_header = _th("Start", "End", "Sequence", "Mass (Da)")
    trypsin_body = "\n".join(
        _td(p["start_1based"], p["end_1based"],
            f'<code>{_trunc(p["peptide_seq"])}</code>',
            f'{p["mass_da"]:.4f}')
        for p in trypsin_peptides
    )
    trypsin_table = (
        f"<table>\n<thead>{trypsin_header}</thead>\n"
        f"<tbody>\n{trypsin_body}\n</tbody>\n</table>"
    )

    # ------------------------------------------------------------------ #
    # Assemble HTML
    # ------------------------------------------------------------------ #
    style = make_style_tag(accent_color)

    html = f"""{style}
<h2>Proteolytic Cleavage Analysis</h2>
<p>Sequence length: <strong>{n}</strong> aa</p>

<h3>Summary — All Enzymes</h3>
{summary_table}

<h3>Trypsin Peptides (detailed)</h3>
{trypsin_table}

<p class="note">
Trypsin rule: cleaves after K/R, not before P (Keil 1992).
Masses: monoisotopic residue masses + H&#8322;O.
</p>
"""
    return html
