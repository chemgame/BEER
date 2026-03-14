"""
beer.analysis.ptm
=================
Post-translational modification (PTM) prediction using consensus sequence rules.

Each scanner implements a well-characterised consensus motif from primary
literature.  Predictions are *in silico* sequence-based only and do not
substitute for experimental validation.

References
----------
N-glycosylation sequon:
    Apweiler, R., Hermjakob, H. & Sharon, N. (1999). On the frequency of
    protein glycosylation, as deduced from analysis of the SWISS-PROT database.
    *Biochim. Biophys. Acta*, 1473(1), 4-8.

CK2 phosphorylation:
    Meggio, F. & Pinna, L.A. (2003). One-thousand-and-one substrates of protein
    kinase CK2? *FASEB J.*, 17(3), 349-368.

PKA phosphorylation:
    Kennelly, P.J. & Krebs, E.G. (1991). Consensus sequences as substrate
    specificity determinants for protein kinases and protein phosphatases.
    *J. Biol. Chem.*, 266(24), 15555-15558.

Ubiquitination / SUMOylation ΨKXE:
    Rodriguez, M.S., Desterro, J.M., Lain, S., Midgley, C.A., Lane, D.P. &
    Hay, R.T. (1999). SUMO-1 modification activates the transcriptional response
    of p53. *EMBO J.*, 18(22), 6455-6461.

N-terminal acetylation (NatA):
    Arnesen, T. et al. (2009). Proteomics analyses reveal the evolutionary
    conservation and divergence of N-terminal acetyltransferases from yeast and
    humans. *Proc. Natl. Acad. Sci. USA*, 106(20), 8157-8162.

Arginine methylation RGG:
    Thandapani, P., O'Connor, T.R., Bailey, T.L. & Richard, S. (2013).
    Defining the RGG/RG motif. *Mol. Cell*, 50(5), 613-623.

Palmitoylation:
    Fukata, M. & Bhaskara, M.V. (2004). Protein palmitoylation in neuronal
    development and synaptic plasticity. *Nat. Rev. Neurosci.*, 5(5), 423-432.
"""

import re

# ---------------------------------------------------------------------------
# CSS shared across all BEER HTML reports
# ---------------------------------------------------------------------------

_REPORT_CSS = """
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 11pt;
       color: #1a1a2e; margin: 0; padding: 0; line-height: 1.6; }
h2 { font-size: 13pt; color: #4361ee; margin-top: 18px; margin-bottom: 8px; font-weight: 600; }
h3 { font-size: 11pt; color: #4361ee; margin-top: 14px; margin-bottom: 4px; font-weight: 600; }
table { border-collapse: collapse; width: 100%; margin: 10px 0 16px 0; font-size: 10pt; }
th { background-color: #4361ee; color: #ffffff; padding: 7px 12px;
     text-align: left; font-weight: 600; }
td { padding: 6px 12px; border-bottom: 1px solid #e8eaf0; color: #2d3748; }
tr:nth-child(even) td { background-color: #f8f9fd; }
tr:hover td { background-color: #eef0f8; }
p.note { font-size: 9pt; color: #718096; font-style: italic; margin: 4px 0 12px 0; }
"""

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
    """Return the sequence context ±flank around *pos* (0-based)."""
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
            Sequence context (±3 residues) centred on the modified residue.
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
    # (b) O-linked glycosylation (mucin-type): cluster ≥3 S/T in 5 aa
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
    # (e) Phosphotyrosine (EGFR-like): Y near [DE] within ±3
    # ------------------------------------------------------------------
    for i in range(n):
        if seq[i] == 'Y':
            lo = max(0, i - 3)
            hi = min(n, i + 4)
            context_win = seq[lo:i] + seq[i + 1:hi]
            if any(aa in 'DE' for aa in context_win):
                _add(
                    'Phosphotyrosine (EGFR-like)',
                    i,
                    f"Y at {i+1} flanked by D/E within ±3 residues "
                    f"(acidic context promotes EGFR-family phosphorylation)",
                    'low',
                )

    # ------------------------------------------------------------------
    # (f) Ubiquitination (ΨKXE): [LVIMF]K.[DE]
    # ------------------------------------------------------------------
    for m in re.finditer(r'[LVIMF]K.[DE]', seq):
        k_pos = m.start() + 1
        _add(
            'Ubiquitination (ΨKXE)',
            k_pos,
            f"ΨKXE ubiquitination motif: K at {k_pos+1}, "
            f"Ψ={seq[m.start()]}, X={seq[m.start()+2]}, E/D={seq[m.start()+3]}",
            'medium',
        )

    # ------------------------------------------------------------------
    # (g) SUMOylation: [VILMF]K.E
    # ------------------------------------------------------------------
    for m in re.finditer(r'[VILMF]K.E', seq):
        k_pos = m.start() + 1
        _add(
            'SUMOylation (ΨKxE)',
            k_pos,
            f"ΨKxE SUMOylation motif: K at {k_pos+1}, "
            f"Ψ={seq[m.start()]}, x={seq[m.start()+2]}",
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
    # (i) Lysine acetylation (internal): KxxK or GKxx
    # ------------------------------------------------------------------
    for m in re.finditer(r'K.{2}K', seq):
        # report the first K
        _add(
            'Lys acetylation (internal)',
            m.start(),
            f"KxxK motif: K at {m.start()+1} in context KxxK",
            'low',
        )
    for m in re.finditer(r'GK.{2}', seq):
        k_pos = m.start() + 1
        _add(
            'Lys acetylation (internal)',
            k_pos,
            f"GKxx motif: K at {k_pos+1} preceded by Gly",
            'low',
        )

    # ------------------------------------------------------------------
    # (j) Arginine methylation: RGG, RG, GR
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

    # ------------------------------------------------------------------
    # (k) Palmitoylation (DHHC target): C preceded by K/R within 3 positions
    # ------------------------------------------------------------------
    for i in range(n):
        if seq[i] == 'C':
            lo = max(0, i - 3)
            upstream = seq[lo:i]
            if any(aa in 'KR' for aa in upstream):
                _add(
                    'Palmitoylation (DHHC)',
                    i,
                    f"C at {i+1} preceded by basic residue within 3 positions "
                    f"(DHHC acyltransferase target context)",
                    'low',
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
    css = _REPORT_CSS.replace("#4361ee", accent)
    _s = f"<style>{css}</style>"

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
