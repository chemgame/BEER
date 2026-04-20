"""Kinase-site phosphorylation prediction using consensus position-weight matrices.

Implements NetPhos-inspired scanning of Ser, Thr, and Tyr residues with
position-weight matrices derived from experimentally verified phosphorylation
sites. Each phospho-acceptor residue is scored in a ±7-residue window;
sites scoring above threshold are reported per kinase.

References
----------
Blom, N., Gammeltoft, S. & Brunak, S. (1999) J. Mol. Biol. 294:1351-1362.
Blom, N. et al. (2004) Proteomics 4:1633-1649. (NetPhos 2.0)
"""
from __future__ import annotations

import math

from beer.reports.css import make_style_tag


# ---------------------------------------------------------------------------
# Per-kinase consensus definitions
# Window of 15 residues centred on the phospho-acceptor (position 7, 0-based).
# Each entry: (name, acceptors, pwm, threshold, reference)
#   pwm: dict mapping position_offset → {aa: log_odds}
#        Only key positions are stored; missing positions contribute 0.
# ---------------------------------------------------------------------------

# Consensus logos (log-odds relative to background frequency 0.05 per aa):
#   Positive = enriched; negative = depleted.
# Values derived from known phosphosite data as described in:
#   Scansite 4.0 (Obenauer et al. 2003 NAR 31:3635) and PhosphoSitePlus.

_LOG05 = math.log(0.05)   # background log-frequency for 20-aa uniform prior


def _lo(freq: float) -> float:
    """Log-odds relative to uniform amino acid background."""
    return math.log(max(freq, 1e-9)) - _LOG05


# PKA (cAMP-dependent protein kinase A): R[R/K]x[S/T]
# Flanking positions enriched for basic residues at -3,-2; hydrophobic at +1.
_PKA_PWM: dict[int, dict[str, float]] = {
    -3: {aa: _lo(v) for aa, v in {
        'R': 0.40, 'K': 0.25, 'H': 0.08,
        'D': 0.005, 'E': 0.005, 'P': 0.01,
    }.items()},
    -2: {aa: _lo(v) for aa, v in {
        'R': 0.35, 'K': 0.20, 'Q': 0.10,
        'D': 0.01, 'E': 0.01,
    }.items()},
    -1: {aa: _lo(v) for aa, v in {
        'A': 0.12, 'V': 0.10, 'L': 0.10, 'I': 0.09,
        'G': 0.08, 'F': 0.07, 'M': 0.07, 'S': 0.06, 'T': 0.06,
        'P': 0.01, 'R': 0.03, 'K': 0.03,
    }.items()},
    +1: {aa: _lo(v) for aa, v in {
        'F': 0.15, 'L': 0.14, 'V': 0.12, 'I': 0.11, 'Y': 0.10,
        'M': 0.08, 'A': 0.07, 'W': 0.05,
        'P': 0.01, 'R': 0.02, 'K': 0.02,
    }.items()},
}

# PKC (protein kinase C): [S/T]x[R/K] or basic + hydrophobic context
# Enriched for basic at +2/+3; hydrophobic at −1.
_PKC_PWM: dict[int, dict[str, float]] = {
    -1: {aa: _lo(v) for aa, v in {
        'F': 0.13, 'L': 0.12, 'V': 0.11, 'I': 0.10, 'M': 0.09,
        'A': 0.08, 'R': 0.08, 'K': 0.06,
        'P': 0.005, 'D': 0.01, 'E': 0.01,
    }.items()},
    +1: {aa: _lo(v) for aa, v in {
        'R': 0.20, 'K': 0.18, 'Q': 0.10, 'H': 0.08,
        'D': 0.01, 'E': 0.01, 'P': 0.01,
    }.items()},
    +2: {aa: _lo(v) for aa, v in {
        'R': 0.25, 'K': 0.20, 'H': 0.10,
        'D': 0.01, 'E': 0.01,
    }.items()},
}

# CK2 (casein kinase 2): [S/T]xxE/D (negative flanking at +3)
_CK2_PWM: dict[int, dict[str, float]] = {
    +1: {aa: _lo(v) for aa, v in {
        'E': 0.20, 'D': 0.15, 'N': 0.08, 'Q': 0.07,
        'R': 0.01, 'K': 0.01,
    }.items()},
    +2: {aa: _lo(v) for aa, v in {
        'E': 0.18, 'D': 0.14, 'N': 0.08, 'Q': 0.07,
        'R': 0.01, 'K': 0.01,
    }.items()},
    +3: {aa: _lo(v) for aa, v in {
        'E': 0.30, 'D': 0.20, 'N': 0.08,
        'R': 0.005, 'K': 0.005,
    }.items()},
}

# Src/Tyr-kinase: YxxΦ or DFGxxY substrate motif
# Enriched for hydrophobic at +3 (Φ positions).
_SRC_TYR_PWM: dict[int, dict[str, float]] = {
    -3: {aa: _lo(v) for aa, v in {
        'E': 0.20, 'D': 0.15, 'N': 0.10, 'Q': 0.08,
        'R': 0.02, 'K': 0.02,
    }.items()},
    -1: {aa: _lo(v) for aa, v in {
        'E': 0.18, 'D': 0.14, 'G': 0.12,
        'R': 0.01, 'K': 0.01,
    }.items()},
    +1: {aa: _lo(v) for aa, v in {
        'E': 0.15, 'Q': 0.12, 'D': 0.10, 'N': 0.08,
        'R': 0.02, 'K': 0.02,
    }.items()},
    +3: {aa: _lo(v) for aa, v in {
        'F': 0.15, 'L': 0.14, 'I': 0.12, 'V': 0.12, 'M': 0.10,
        'Y': 0.08, 'W': 0.06,
        'P': 0.01, 'D': 0.01, 'E': 0.01,
    }.items()},
}

_KINASE_DEFS: list[tuple[str, str, dict[int, dict[str, float]], float]] = [
    ("PKA",     "ST",  _PKA_PWM,     1.0),
    ("PKC",     "ST",  _PKC_PWM,     0.8),
    ("CK2",     "ST",  _CK2_PWM,     1.0),
    ("Src/Tyr", "Y",   _SRC_TYR_PWM, 0.8),
]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_site(seq: str, pos: int, pwm: dict[int, dict[str, float]]) -> float:
    """Sum log-odds contributions from context residues around *pos*."""
    score = 0.0
    n = len(seq)
    for offset, col in pwm.items():
        j = pos + offset
        if 0 <= j < n:
            score += col.get(seq[j], 0.0)
    return score


def predict_phosphorylation(seq: str) -> dict[str, list[dict]]:
    """Predict kinase phosphorylation sites using consensus PWMs.

    Scans every Ser/Thr/Tyr for each kinase PWM; returns sites above
    the kinase-specific threshold.

    Parameters
    ----------
    seq:
        Protein sequence in single-letter uppercase code.

    Returns
    -------
    dict
        Keys are kinase names; values are lists of dicts with keys
        ``position`` (1-based), ``residue``, ``score``.
    """
    seq = seq.upper()
    result: dict[str, list[dict]] = {}
    for kinase, acceptors, pwm, threshold in _KINASE_DEFS:
        hits = []
        for i, aa in enumerate(seq):
            if aa in acceptors:
                score = _score_site(seq, i, pwm)
                if score >= threshold:
                    hits.append({
                        "position": i + 1,
                        "residue": aa,
                        "score": round(score, 3),
                    })
        result[kinase] = hits
    return result


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def format_phospho_report(seq: str, style_tag: str) -> str:
    """Generate HTML section for phosphorylation site predictions.

    Parameters
    ----------
    seq:
        Protein sequence in single-letter uppercase code.
    style_tag:
        Accent colour hex string (e.g. ``"#4361ee"``).
    """
    accent = style_tag if style_tag else "#4361ee"
    _s = make_style_tag(accent)

    predictions = predict_phosphorylation(seq)
    total = sum(len(v) for v in predictions.values())

    summary_rows = "".join(
        f"<tr><td>{kinase}</td><td>{len(hits)}</td></tr>"
        for kinase, hits in predictions.items()
    )
    summary_html = (
        "<h2>Phosphorylation Sites (NetPhos-style PWM)</h2>"
        "<table>"
        "<tr><th>Kinase</th><th>Sites predicted</th></tr>"
        f"{summary_rows}"
        "<tr><td><b>Total</b></td><td><b>" + str(total) + "</b></td></tr>"
        "</table>"
    )

    detail_parts = []
    for kinase, hits in predictions.items():
        if not hits:
            detail_parts.append(
                f"<h3>{kinase}</h3><p>No sites above threshold.</p>"
            )
            continue
        rows = "".join(
            f"<tr><td>{h['position']}</td><td>{h['residue']}</td>"
            f"<td>{h['score']:.3f}</td></tr>"
            for h in hits
        )
        detail_parts.append(
            f"<h3>{kinase}</h3>"
            "<table>"
            "<tr><th>Position</th><th>Residue</th><th>PWM score</th></tr>"
            f"{rows}"
            "</table>"
        )

    detail_html = "".join(detail_parts)

    note = (
        "<p class='note'>"
        "NetPhos-style position-weight matrices for PKA (R[R/K]x[S/T]), "
        "PKC ([S/T]x[R/K]), CK2 ([S/T]xxE/D), and Src/Tyr kinase (YxxΦ). "
        "PWM log-odds relative to uniform amino acid background (Blom et al. 1999 "
        "J. Mol. Biol. 294:1351). Scores &ge; threshold are reported; all sites require "
        "experimental validation."
        "</p>"
    )

    return _s + summary_html + detail_html + note
