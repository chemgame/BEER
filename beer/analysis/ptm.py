"""Post-translational modification (PTM) site prediction."""
from __future__ import annotations
import re

from beer.reports.css import make_style_tag


# Confidence badge colours
_CONF_COLOUR = {
    "high":   "#16a34a",
    "medium": "#ca8a04",
    "low":    "#dc2626",
}

# ---------------------------------------------------------------------------
# PTM scanner helpers
# ---------------------------------------------------------------------------


def _context(seq: str, pos: int, flank: int = 3) -> str:
    """Return the sequence context +/-flank around *pos* (0-based)."""
    lo = max(0, pos - flank)
    hi = min(len(seq), pos + flank + 1)
    left = seq[lo:pos]
    right = seq[pos + 1:hi]
    centre = seq[pos]
    # Pad to fixed width for alignment
    pad_l = " " * (flank - (pos - lo))
    pad_r = " " * (flank - (hi - pos - 1))
    return f"{pad_l}{left}[{centre}]{right}{pad_r}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scan_ptm_sites(seq: str) -> list[dict]:
    """Scan a protein sequence for putative PTM sites using consensus rules.

    Parameters
    ----------
    seq:
        Protein sequence in single-letter uppercase code.

    Returns
    -------
    list[dict]
        Each dict contains:

        ``type``
            PTM type string.
        ``position_1based``
            Residue position (1-based) of the modified residue.
        ``context``
            Sequence context (+-3 residues) centred on the modified residue.
        ``description``
            Short human-readable description.
        ``confidence``
            ``"high"``, ``"medium"``, or ``"low"``.
    """
    results: list[dict] = []
    n = len(seq)

    def _add(ptm_type: str, pos0: int, desc: str, conf: str) -> None:
        results.append({
            'type': ptm_type,
            'position_1based': pos0 + 1,
            'context': _context(seq, pos0),
            'description': desc,
            'confidence': conf,
        })

    # ------------------------------------------------------------------
    # (a) N-linked glycosylation: N[^P][ST]
    # ------------------------------------------------------------------
    for m in re.finditer(r'N[^P][ST]', seq):
        _add(
            'N-linked glycosylation',
            m.start(),
            f"NxS/T sequon (x={seq[m.start()+1]}, not Pro); "
            f"acceptor Asn at position {m.start()+1}",
            'high',
        )

    # ------------------------------------------------------------------
    # (b) O-linked glycosylation (mucin-type): cluster >=3 S/T in 5 aa
    # ------------------------------------------------------------------
    _nglyc_positions = {m.start() for m in re.finditer(r'N[^P][ST]', seq)}
    reported_o = set()
    for i in range(n - 4):
        window5 = seq[i:i + 5]
        st_count = sum(1 for aa in window5 if aa in 'ST')
        if st_count >= 3:
            # Report each S/T in this window that hasn't been flagged yet
            for j in range(i, min(i + 5, n)):
                if seq[j] in 'ST' and j not in reported_o:
                    reported_o.add(j)
                    _add(
                        'O-linked glycosylation',
                        j,
                        f"Mucin-type cluster: {st_count} S/T within 5-aa window "
                        f"at positions {i+1}-{i+5}",
                        'medium',
                    )

    # ------------------------------------------------------------------
    # (c) CK2 phosphorylation: [ST]xx[DE]
    # ------------------------------------------------------------------
    for m in re.finditer(r'[ST].{2}[DE]', seq):
        _add(
            'Phosphoserine/Thr (CK2)',
            m.start(),
            f"CK2 motif [S/T]xx[D/E]: phospho-{seq[m.start()]} at {m.start()+1}, "
            f"acidic at +3 ({seq[m.start()+3]})",
            'medium',
        )

    # ------------------------------------------------------------------
    # (d) PKA phosphorylation: R[^P][^P][ST]
    # ------------------------------------------------------------------
    for m in re.finditer(r'R[^P][^P][ST]', seq):
        st_pos = m.start() + 3
        _add(
            'Phosphoserine/Thr (PKA)',
            st_pos,
            f"PKA motif R..S/T: phospho-{seq[st_pos]} at {st_pos+1}",
            'medium',
        )

    # ------------------------------------------------------------------
    # (e) Ubiquitination (PSIKXE): [LVIMF]K.[DE]
    # ------------------------------------------------------------------
    for m in re.finditer(r'[LVIMF]K.[DE]', seq):
        k_pos = m.start() + 1
        _add(
            'Ubiquitination (\u03a8KXE)',
            k_pos,
            f"\u03a8KXE ubiquitination motif: K at {k_pos+1}, "
            f"\u03a8={seq[m.start()]}, X={seq[m.start()+2]}, E/D={seq[m.start()+3]}",
            'medium',
        )

    # ------------------------------------------------------------------
    # (g) SUMOylation: [VILMF]K.E
    # ------------------------------------------------------------------
    for m in re.finditer(r'[VILMF]K.E', seq):
        k_pos = m.start() + 1
        _add(
            'SUMOylation (\u03a8KxE)',
            k_pos,
            f"\u03a8KxE SUMOylation motif: K at {k_pos+1}, "
            f"\u03a8={seq[m.start()]}, x={seq[m.start()+2]}",
            'medium',
        )

    # ------------------------------------------------------------------
    # (h) N-terminal acetylation (NatA)
    # ------------------------------------------------------------------
    if n >= 2:
        aa1, aa2 = seq[0], seq[1]
        # NatA: initiator Met removed when followed by small aa (A/C/G/P/S/T/V)
        # Then acetylates the new N-terminus.
        if aa1 == 'M' and aa2 in 'ACGPSTV' and aa2 != 'P':
            _add(
                'N-terminal acetylation (NatA)',
                1,  # new N-terminus after Met removal
                f"NatA: Met(1) removed (followed by {aa2}); "
                f"new N-terminal {aa2} at position 2 is acetylated",
                'medium',
            )
        elif aa1 in 'SATGC':
            _add(
                'N-terminal acetylation (NatA)',
                0,
                f"NatA: N-terminal {aa1} at position 1 may be acetylated "
                f"(no preceding Met removal)",
                'medium',
            )

    # ------------------------------------------------------------------
    # (i) Arginine methylation: RGG, RG, GR
    # ------------------------------------------------------------------
    for m in re.finditer(r'RGG', seq):
        _add(
            'Arg methylation (RGG)',
            m.start(),
            f"RGG box: R at {m.start()+1} (PRMT4/PRMT5 substrate motif)",
            'medium',
        )
    for m in re.finditer(r'(?<!G)RG(?!G)', seq):
        _add(
            'Arg methylation (RG)',
            m.start(),
            f"RG motif: R at {m.start()+1}",
            'medium',
        )
    for m in re.finditer(r'GR', seq):
        _add(
            'Arg methylation (GR)',
            m.start() + 1,
            f"GR motif: R at {m.start()+2}",
            'medium',
        )

    # Sort by position, then type
    results.sort(key=lambda d: (d['position_1based'], d['type']))
    return results


def summarize_ptm_sites(sites: list[dict]) -> dict:
    """Count PTM sites by type.

    Parameters
    ----------
    sites:
        Output of :func:`scan_ptm_sites`.

    Returns
    -------
    dict
        Mapping ``{ptm_type: count, ..., "total": N}``.
    """
    counts: dict[str, int] = {}
    for site in sites:
        t = site['type']
        counts[t] = counts.get(t, 0) + 1
    counts['total'] = len(sites)
    return counts


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def format_ptm_report(seq: str, style_tag: str) -> str:
    """Generate an HTML section summarising predicted PTM sites.

    Produces a table grouped by PTM type with columns for position, sequence
    context, confidence, and a description.

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

    sites = scan_ptm_sites(seq)
    summary = summarize_ptm_sites(sites)
    total = summary.pop('total', 0)

    if not sites:
        return (
            _s
            + "<h2>Post-Translational Modifications</h2>"
            + "<p>No consensus PTM sites detected in this sequence.</p>"
        )

    # Group by type maintaining insertion order
    by_type: dict[str, list[dict]] = {}
    for site in sites:
        by_type.setdefault(site['type'], []).append(site)

    # Summary table
    summary_rows = "".join(
        f"<tr><td>{ptm_type}</td><td>{count}</td></tr>"
        for ptm_type, count in sorted(summary.items())
    )
    summary_html = (
        "<h2>Post-Translational Modifications</h2>"
        "<table>"
        "<tr><th>PTM Type</th><th>Count</th></tr>"
        f"{summary_rows}"
        f"<tr><td><strong>Total</strong></td><td><strong>{total}</strong></td></tr>"
        "</table>"
    )

    # Per-type sub-tables
    detail_parts: list[str] = []
    header = (
        "<tr><th>PTM Type</th><th>Position</th><th>Context (&#177;3 aa)</th>"
        "<th>Confidence</th><th>Description</th></tr>"
    )
    for ptm_type, type_sites in by_type.items():
        rows = "".join(
            "<tr>"
            f"<td>{site['type']}</td>"
            f"<td>{site['position_1based']}</td>"
            f"<td><code>{site['context'].strip()}</code></td>"
            f"<td style='color:{_CONF_COLOUR.get(site['confidence'], '#000')};font-weight:600'>"
            f"{site['confidence'].capitalize()}</td>"
            f"<td>{site['description']}</td>"
            "</tr>"
            for site in type_sites
        )
        detail_parts.append(
            f"<h3>{ptm_type}</h3>"
            "<table>"
            f"{header}{rows}"
            "</table>"
        )

    detail_html = "".join(detail_parts)

    note = (
        "<p class='note'>"
        "Predictions are sequence-based consensus motif scans only. "
        "Experimental validation is required for confirmation. "
        "Confidence: High = published sequon; Medium = established motif; "
        "Low = contextual inference."
        "</p>"
    )

    return _s + summary_html + detail_html + note
