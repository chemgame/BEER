#!/usr/bin/env python3
"""
BEER - Biochemical Estimator & Explorer of Residues

Requirements:
  pip install biopython matplotlib PyQt5 mplcursors
"""

import sys, math, os, base64, json, csv, subprocess, re, difflib, importlib.util
import warnings
from io import BytesIO, StringIO
from math import log2
import urllib.request
import numpy as np

# All sub-modules are inlined below; flags always True
_HAS_AGGREGATION = True
_HAS_PTM         = True
_HAS_SIGNAL      = True
_HAS_AMPHIPATHIC = True
_HAS_SCD         = True
_HAS_RBP         = True
_HAS_TANDEM      = True
_HAS_NEW_GRAPHS  = True
_HAS_ELM         = True
_HAS_DISPROT     = True
_HAS_PHASEPDB    = True
_HAS_PHI_PSI     = True

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QTextBrowser,
    QFileDialog, QTabWidget, QMessageBox, QTableWidget, QTableWidgetItem,
    QCheckBox, QStatusBar, QComboBox, QFormLayout,
    QSplitter, QScrollArea, QFrame, QDialog, QDialogButtonBox,
    QSpinBox, QProgressDialog, QAbstractItemView,
    QListWidget, QListWidgetItem, QTreeWidget, QTreeWidgetItem, QStackedWidget,
    QInputDialog,
)
from PyQt5.QtGui import QFont, QKeySequence
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import QShortcut
from PyQt5.QtPrintSupport import QPrinter
try:
    from PyQt5.QtWebEngineWidgets import QWebEngineView
    _WEBENGINE_AVAILABLE = True
except ImportError:
    _WEBENGINE_AVAILABLE = False

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
from matplotlib.patches import Patch, Rectangle
import matplotlib.pyplot as plt
plt.style.use("default")
import mplcursors

# Suppress known harmless warnings from matplotlib and Qt font fallback
warnings.filterwarnings("ignore", message="Setting the 'color' property will override")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

from Bio.SeqUtils.ProtParam import ProteinAnalysis as BPProteinAnalysis
from Bio import SeqIO
from Bio.PDB import PDBParser, PDBIO, Select as PDBSelect, PPBuilder
from Bio.PDB.Polypeptide import is_aa
from Bio.Blast import NCBIWWW, NCBIXML
from Bio.SeqUtils import seq1

# ===========================================================================
# INLINED SUB-MODULES (single-file distribution)
# ===========================================================================
import urllib.error  # noqa (used by network workers)
from collections import Counter  # noqa (used by tandem_repeats and new_graphs)
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.patches import FancyArrowPatch, Arc
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# Shared HTML report CSS
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


# --- beer.analysis.aggregation ---

# ---------------------------------------------------------------------------
# Published per-residue scales
# ---------------------------------------------------------------------------

ZYGGREGATOR_PROPENSITY: dict[str, float] = {
    'A':  0.67, 'R': -1.65, 'N': -0.43, 'D': -0.75,
    'C':  0.50, 'Q': -0.51, 'E': -1.22, 'G': -0.59,
    'H': -0.13, 'I':  1.29, 'L':  0.93, 'K': -1.42,
    'M':  0.64, 'F':  1.26, 'P': -1.44, 'S': -0.39,
    'T': -0.09, 'W':  0.96, 'Y':  0.74, 'V':  1.04,
}
"""Per-residue β-aggregation propensity scores (Tartaglia & Vendruscolo 2008)."""

PASTA_ENERGY: dict[str, float] = {
    'A': -0.22, 'R':  0.66, 'N':  0.14, 'D':  0.81,
    'C': -0.65, 'Q': -0.04, 'E':  0.58, 'G':  0.08,
    'H': -0.35, 'I': -1.46, 'L': -1.34, 'K':  0.59,
    'M': -0.94, 'F': -1.47, 'P':  1.53, 'S':  0.10,
    'T': -0.34, 'W': -1.35, 'Y': -1.04, 'V': -1.32,
}
"""Diagonal of PASTA pairwise β-strand interaction energy matrix (Trovato et al. 2007).
More negative values indicate stronger self-pairing tendency."""

CAMSOLMT_SCALE: dict[str, float] = {
    'A':  0.238, 'R': -0.132, 'N':  0.047, 'D':  0.191,
    'C':  0.238, 'Q':  0.047, 'E':  0.191, 'G':  0.024,
    'H': -0.083, 'I': -0.387, 'L': -0.387, 'K': -0.065,
    'M': -0.265, 'F': -0.386, 'P':  0.190, 'S':  0.264,
    'T':  0.209, 'W': -0.380, 'Y': -0.241, 'V': -0.322,
}
"""CamSol intrinsic solubility scale (Sormanni et al. 2015, J Mol Biol)."""

# Kyte-Doolittle defined locally to avoid circular imports
KYTE_DOOLITTLE: dict[str, float] = {
    'A':  1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5,
    'C':  2.5, 'Q': -3.5, 'E': -3.5, 'G': -0.4,
    'H': -3.2, 'I':  4.5, 'L':  3.8, 'K': -3.9,
    'M':  1.9, 'F':  2.8, 'P': -1.6, 'S': -0.8,
    'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V':  4.2,
}

# CSS injected into every HTML section for consistent styling



# ---------------------------------------------------------------------------
# Core computational functions
# ---------------------------------------------------------------------------

def calc_aggregation_profile(seq: str, window: int = 6) -> list[float]:
    """Compute per-residue ZYGGREGATOR-style β-aggregation propensity profile.

    A sliding window of length *window* is centred at each residue; the score
    for that position is the arithmetic mean of ZYGGREGATOR_PROPENSITY values
    over the window.  At sequence edges the window is truncated to the
    available residues (partial windows are acceptable).

    Parameters
    ----------
    seq:
        Protein sequence in single-letter uppercase code.
    window:
        Sliding window length (default 6 as in Tartaglia & Vendruscolo 2008).

    Returns
    -------
    list[float]
        Per-residue profile of length ``len(seq)``.

    References
    ----------
    Tartaglia, G.G. & Vendruscolo, M. (2008) Chem. Biol. 15(9):1008-1018.
    """
    n = len(seq)
    if n == 0:
        return []

    half = window // 2
    profile: list[float] = []

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + (window - 2 * half))  # handle odd/even window
        # For even window keep consistent: centre at i means [i-half, i-half+window)
        lo = max(0, i - half)
        hi = min(n, lo + window)
        if hi == n:
            lo = max(0, hi - window)
        sub = seq[lo:hi]
        vals = [ZYGGREGATOR_PROPENSITY.get(aa, 0.0) for aa in sub]
        profile.append(sum(vals) / len(vals) if vals else 0.0)

    return profile


def predict_aggregation_hotspots(
    seq: str,
    window: int = 6,
    threshold: float = 1.0,
) -> list[dict]:
    """Identify contiguous aggregation-prone regions using ZYGGREGATOR profile.

    A hotspot is a contiguous stretch where the sliding-window aggregation
    profile (see :func:`calc_aggregation_profile`) equals or exceeds *threshold*
    for at least 4 consecutive residues.  For each hotspot, a PASTA energy score
    is also computed as the mean of PASTA_ENERGY values over the hotspot
    sequence (lower = more amyloidogenic).

    Parameters
    ----------
    seq:
        Protein sequence in single-letter uppercase code.
    window:
        Window size forwarded to :func:`calc_aggregation_profile`.
    threshold:
        Minimum mean propensity to call a hotspot (default 1.0).

    Returns
    -------
    list[dict]
        Each dict contains:

        ``start``
            0-based start position.
        ``end``
            0-based exclusive end position.
        ``seq``
            Hotspot subsequence.
        ``score``
            Mean ZYGGREGATOR propensity over the hotspot.
        ``pasta_score``
            Mean PASTA_ENERGY diagonal score over the hotspot.

    References
    ----------
    Tartaglia, G.G. & Vendruscolo, M. (2008) Chem. Biol. 15(9):1008-1018.
    Trovato, A. et al. (2007) PLoS Comput. Biol. 3(2):e17.
    """
    profile = calc_aggregation_profile(seq, window=window)
    n = len(seq)
    hotspots: list[dict] = []
    i = 0
    while i < n:
        if profile[i] >= threshold:
            j = i
            while j < n and profile[j] >= threshold:
                j += 1
            length = j - i
            if length >= 4:
                sub = seq[i:j]
                mean_score = sum(profile[i:j]) / length
                pasta = sum(PASTA_ENERGY.get(aa, 0.0) for aa in sub) / length
                hotspots.append({
                    'start': i,
                    'end': j,
                    'seq': sub,
                    'score': mean_score,
                    'pasta_score': pasta,
                })
            i = j
        else:
            i += 1
    return hotspots


def calc_camsolmt_score(seq: str) -> list[float]:
    """Compute CamSol intrinsic per-residue solubility scores.

    Returns raw per-residue scores from the CamSol intrinsic scale
    (Sormanni et al. 2015).  No window averaging is applied at this stage;
    use :func:`calc_solubility_stats` which also returns a smoothed profile.

    Parameters
    ----------
    seq:
        Protein sequence in single-letter uppercase code.

    Returns
    -------
    list[float]
        Per-residue intrinsic CamSol scores (length = ``len(seq)``).
        Positive values indicate higher intrinsic solubility; negative
        values indicate lower intrinsic solubility (aggregation tendency).

    References
    ----------
    Sormanni, P., Aprile, F.A. & Vendruscolo, M. (2015).
    J. Mol. Biol. 427(2):478-490.
    """
    return [CAMSOLMT_SCALE.get(aa, 0.0) for aa in seq]


def _smooth_profile(profile: list[float], window: int = 7) -> list[float]:
    """Return a uniformly-windowed moving-average of *profile*."""
    n = len(profile)
    if n == 0:
        return []
    half = window // 2
    smoothed: list[float] = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        chunk = profile[lo:hi]
        smoothed.append(sum(chunk) / len(chunk))
    return smoothed


def calc_solubility_stats(seq: str) -> dict:
    """Aggregate solubility and aggregation metrics for a protein sequence.

    Combines CamSol intrinsic scores with ZYGGREGATOR-based hotspot analysis.

    Parameters
    ----------
    seq:
        Protein sequence in single-letter uppercase code.

    Returns
    -------
    dict
        ``mean_camsolmt``
            Mean CamSol intrinsic score across all residues.
        ``fraction_insoluble``
            Fraction of residues with CamSol score < -0.2.
        ``fraction_soluble``
            Fraction of residues with CamSol score > 0.2.
        ``mean_aggregation_propensity``
            Mean ZYGGREGATOR propensity (no windowing) across all residues.
        ``n_hotspots``
            Number of aggregation hotspots identified.
        ``aggregation_hotspots``
            List of hotspot dicts from :func:`predict_aggregation_hotspots`.
        ``camsolmt_profile``
            Raw per-residue CamSol scores.
        ``camsolmt_smoothed``
            7-residue moving-average smoothed CamSol profile.
    """
    n = len(seq)
    if n == 0:
        return {
            'mean_camsolmt': 0.0, 'fraction_insoluble': 0.0,
            'fraction_soluble': 0.0, 'mean_aggregation_propensity': 0.0,
            'n_hotspots': 0, 'aggregation_hotspots': [],
            'camsolmt_profile': [], 'camsolmt_smoothed': [],
        }

    camsolmt = calc_camsolmt_score(seq)
    smoothed = _smooth_profile(camsolmt, window=7)
    mean_cs = sum(camsolmt) / n
    frac_insol = sum(1 for v in camsolmt if v < -0.2) / n
    frac_sol = sum(1 for v in camsolmt if v > 0.2) / n

    zygg_raw = [ZYGGREGATOR_PROPENSITY.get(aa, 0.0) for aa in seq]
    mean_agg = sum(zygg_raw) / n

    hotspots = predict_aggregation_hotspots(seq)

    return {
        'mean_camsolmt': mean_cs,
        'fraction_insoluble': frac_insol,
        'fraction_soluble': frac_sol,
        'mean_aggregation_propensity': mean_agg,
        'n_hotspots': len(hotspots),
        'aggregation_hotspots': hotspots,
        'camsolmt_profile': camsolmt,
        'camsolmt_smoothed': smoothed,
    }


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def format_aggregation_report(seq: str, style_tag: str) -> str:
    """Generate an HTML section summarising aggregation and solubility analysis.

    Parameters
    ----------
    seq:
        Protein sequence in single-letter uppercase code.
    style_tag:
        Accent colour hex string (e.g. ``"#4361ee"``) injected via the
        ``style_tag`` convention used throughout the BEER report system.
        If empty or ``None``, a default blue is used.

    Returns
    -------
    str
        Self-contained HTML fragment (includes ``<style>`` block).
    """
    accent = style_tag if style_tag else "#4361ee"
    css = _REPORT_CSS.replace("#4361ee", accent)
    _s = f"<style>{css}</style>"

    stats = calc_solubility_stats(seq)
    hotspots = stats['aggregation_hotspots']
    n = len(seq)

    # --- summary table ---
    cs_mean = stats['mean_camsolmt']
    cs_class = (
        "good (soluble)" if cs_mean > 0.1 else
        "borderline" if cs_mean > -0.1 else
        "poor (aggregation-prone)"
    )
    agg_mean = stats['mean_aggregation_propensity']
    agg_class = (
        "low" if agg_mean < 0.3 else
        "moderate" if agg_mean < 0.7 else
        "high"
    )

    summary_rows = (
        f"<tr><td>Sequence length</td><td>{n} aa</td></tr>"
        f"<tr><td>Mean CamSol intrinsic score</td>"
        f"<td>{cs_mean:.3f} &mdash; {cs_class}</td></tr>"
        f"<tr><td>Fraction insoluble residues (CamSol &lt; &minus;0.2)</td>"
        f"<td>{stats['fraction_insoluble']:.1%}</td></tr>"
        f"<tr><td>Fraction soluble residues (CamSol &gt; 0.2)</td>"
        f"<td>{stats['fraction_soluble']:.1%}</td></tr>"
        f"<tr><td>Mean ZYGGREGATOR propensity</td>"
        f"<td>{agg_mean:.3f} &mdash; {agg_class}</td></tr>"
        f"<tr><td>Aggregation hotspots (&ge;4 aa, score &ge;1.0)</td>"
        f"<td>{stats['n_hotspots']}</td></tr>"
    )

    summary_html = (
        "<h2>Aggregation &amp; Solubility Summary</h2>"
        "<table>"
        "<tr><th>Property</th><th>Value</th></tr>"
        f"{summary_rows}"
        "</table>"
        "<p class='note'>"
        "CamSol: Sormanni et al. 2015 J Mol Biol 427:478. "
        "ZYGGREGATOR: Tartaglia &amp; Vendruscolo 2008 Chem Biol 15:1008."
        "</p>"
    )

    # --- hotspot table ---
    if hotspots:
        hs_header = (
            "<tr><th>Start</th><th>End</th><th>Sequence</th>"
            "<th>ZYGGREGATOR score</th><th>PASTA score</th><th>Length</th></tr>"
        )
        hs_rows = "".join(
            f"<tr>"
            f"<td>{hs['start'] + 1}</td>"
            f"<td>{hs['end']}</td>"
            f"<td><code>{hs['seq']}</code></td>"
            f"<td>{hs['score']:.3f}</td>"
            f"<td>{hs['pasta_score']:.3f}</td>"
            f"<td>{hs['end'] - hs['start']}</td>"
            f"</tr>"
            for hs in hotspots
        )
        hotspot_html = (
            "<h2>Aggregation Hotspots</h2>"
            "<table>"
            f"{hs_header}{hs_rows}"
            "</table>"
            "<p class='note'>"
            "PASTA diagonal energies: Trovato et al. 2007 PLoS Comput Biol 3:e17. "
            "More negative PASTA score = stronger amyloid propensity."
            "</p>"
        )
    else:
        hotspot_html = (
            "<h2>Aggregation Hotspots</h2>"
            "<p>No aggregation hotspots detected "
            "(threshold: mean ZYGGREGATOR &ge; 1.0 over &ge; 4 residues).</p>"
        )

    return _s + summary_html + hotspot_html


# --- beer.analysis.ptm ---

# ---------------------------------------------------------------------------
# CSS shared across all BEER HTML reports
# ---------------------------------------------------------------------------



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


# --- beer.analysis.signal_peptide ---

# Imported for type hints only — no GUI, no circular dependency

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


# --- beer.analysis.amphipathic ---

# ---------------------------------------------------------------------------
# Published hydrophobicity scale
# ---------------------------------------------------------------------------

EISENBERG_SCALE: dict[str, float] = {
    'A':  0.620, 'R': -2.530, 'N': -0.780, 'D': -0.900,
    'C':  0.290, 'Q': -0.850, 'E': -0.740, 'G':  0.480,
    'H': -0.400, 'I':  1.380, 'L':  1.060, 'K': -1.500,
    'M':  0.640, 'F':  1.190, 'P':  0.120, 'S': -0.180,
    'T': -0.050, 'W':  0.810, 'Y':  0.260, 'V':  1.080,
}
"""Eisenberg normalised consensus hydrophobicity scale (Eisenberg et al. 1984 PNAS)."""



# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _hydrophobic_moment_window(
    sub: str,
    angle_deg: float = 100.0,
) -> float:
    """Compute μH for a single window sub-sequence.

    μH = sqrt( [Σ Hi·sin(i·δ)]² + [Σ Hi·cos(i·δ)]² ) / n

    where δ is the inter-residue angle (100° for α-helix, 160° for β-strand),
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
        Use 100.0 for α-helix; 160.0 for β-strand.

    Returns
    -------
    list[float]
        Per-residue μH values; length == ``len(seq)``.

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
        Angular increment per residue (default 100.0 for α-helix).
    window:
        Sliding window length.

    Returns
    -------
    list[float]
        Full per-residue μH profile.
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

    1. The sliding-window μH (α-helix, δ = 100°) ≥ *moment_threshold*.
    2. The mean Eisenberg hydrophobicity of the same window is between
       −0.5 and 1.5 (not purely hydrophobic, not purely hydrophilic).

    Parameters
    ----------
    seq:
        Protein sequence in single-letter uppercase code.
    window:
        Window size for hydrophobic moment calculation.
    moment_threshold:
        Minimum mean μH to flag a region as amphipathic (default 0.35).
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
            Mean μH over the region.
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

    Reports mean μH for α-helix (δ = 100°) and β-strand (δ = 160°) angles,
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
    css = _REPORT_CSS.replace("#4361ee", accent)
    _s = f"<style>{css}</style>"

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


# --- beer.analysis.scd ---



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


# --- beer.analysis.rnabinding ---



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


# --- beer.analysis.tandem_repeats ---



# ---------------------------------------------------------------------------
# Tandem repeat detection
# ---------------------------------------------------------------------------

def find_tandem_repeats(
    seq: str,
    min_unit: int = 2,
    max_unit: int = 8,
    min_copies: int = 2,
) -> list[dict]:
    """Find exact tandem repeats by k-mer extension.

    For every starting position and every unit length in [*min_unit*, *max_unit*],
    the repeat unit ``seq[i:i+k]`` is extended greedily as long as the next
    k-mer matches exactly.  Overlapping candidates are resolved by keeping the
    longest repeat (by total covered length).

    Parameters
    ----------
    seq:
        Protein sequence in single-letter uppercase code.
    min_unit:
        Minimum repeat unit length (default 2).
    max_unit:
        Maximum repeat unit length (default 8).
    min_copies:
        Minimum number of consecutive copies (default 2).

    Returns
    -------
    list[dict]
        Each dict:

        ``start_1based``
            1-based start position of the repeat.
        ``end_1based``
            1-based end position (inclusive).
        ``unit``
            Repeat unit sequence.
        ``unit_length``
            Length of the repeat unit.
        ``n_copies``
            Number of exact consecutive copies.
        ``total_length``
            Total length covered (unit_length * n_copies).

    Sorted by *total_length* descending.  Overlapping repeats (same positions)
    are deduplicated; only the longest non-overlapping set is returned.
    """
    n = len(seq)
    candidates: list[dict] = []

    for k in range(min_unit, max_unit + 1):
        i = 0
        while i <= n - k * min_copies:
            unit = seq[i:i + k]
            copies = 1
            j = i + k
            while j + k <= n and seq[j:j + k] == unit:
                copies += 1
                j += k
            if copies >= min_copies:
                candidates.append({
                    'start_1based': i + 1,
                    'end_1based': i + k * copies,
                    'unit': unit,
                    'unit_length': k,
                    'n_copies': copies,
                    'total_length': k * copies,
                })
                # advance past this repeat to avoid partial overlaps at same k
                i += k * copies
            else:
                i += 1

    # Sort by total length descending
    candidates.sort(key=lambda d: d['total_length'], reverse=True)

    # Deduplicate: remove intervals that are fully contained in already-kept repeats
    kept: list[dict] = []
    used_positions: set[int] = set()
    for cand in candidates:
        s0 = cand['start_1based'] - 1  # 0-based
        e0 = cand['end_1based']        # 0-based exclusive
        pos_set = set(range(s0, e0))
        # keep if it doesn't overlap >50% with already-kept intervals
        if not pos_set.intersection(used_positions):
            kept.append(cand)
            used_positions.update(pos_set)

    kept.sort(key=lambda d: d['total_length'], reverse=True)
    return kept


# ---------------------------------------------------------------------------
# Direct repeat detection
# ---------------------------------------------------------------------------

def find_direct_repeats(
    seq: str,
    min_length: int = 4,
    max_gap: int = 30,
) -> list[dict]:
    """Find pairs of identical subsequences separated by ≤ *max_gap* residues.

    Uses an O(n²) approach: for each position *i*, look ahead to positions
    *j* within ``[i + min_length, i + min_length + max_gap]`` and check
    whether ``seq[i:i+L]`` == ``seq[j:j+L]`` for increasing L.

    Parameters
    ----------
    seq:
        Protein sequence in single-letter uppercase code.
    min_length:
        Minimum repeat length to report (default 4).
    max_gap:
        Maximum gap (in residues) between the two copies (default 30).

    Returns
    -------
    list[dict]
        Each dict:

        ``seq1_start_1based``
            1-based start of the first copy.
        ``seq2_start_1based``
            1-based start of the second copy.
        ``repeat_seq``
            Shared repeat sequence.
        ``length``
            Length of the shared sequence.
        ``gap``
            Number of residues between the two copies.
    """
    n = len(seq)
    results: list[dict] = []

    for i in range(n - min_length):
        for j in range(i + min_length, min(i + min_length + max_gap + 1, n - min_length + 1)):
            gap = j - i - min_length  # gap after end of first copy
            # Actually gap = j - (i + L); we'll compute for the actual matched length
            # Extend match
            L = 0
            max_L = min(n - i, n - j)
            while L < max_L and seq[i + L] == seq[j + L]:
                L += 1
            if L >= min_length:
                actual_gap = j - (i + L)
                if actual_gap < 0:
                    # copies overlap — skip
                    continue
                if actual_gap > max_gap:
                    continue
                results.append({
                    'seq1_start_1based': i + 1,
                    'seq2_start_1based': j + 1,
                    'repeat_seq': seq[i:i + L],
                    'length': L,
                    'gap': actual_gap,
                })

    # Remove duplicates (same seq2_start, same repeat)
    seen: set[tuple] = set()
    unique: list[dict] = []
    for r in results:
        key = (r['seq1_start_1based'], r['seq2_start_1based'], r['repeat_seq'])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    unique.sort(key=lambda d: d['length'], reverse=True)
    return unique


# ---------------------------------------------------------------------------
# Compositional (low-complexity) repeat detection
# ---------------------------------------------------------------------------

def find_compositional_repeats(
    seq: str,
    window: int = 10,
    step: int = 5,
    n_top: int = 3,
) -> list[dict]:
    """Find windows with unusually low compositional diversity.

    A window is flagged as low-complexity (LC) if the fraction contributed by
    the top-*n_top* most frequent amino acids exceeds 0.70.

    Parameters
    ----------
    seq:
        Protein sequence in single-letter uppercase code.
    window:
        Window length (default 10).
    step:
        Step size between window starts (default 5).
    n_top:
        Number of most-frequent amino acids to consider (default 3).

    Returns
    -------
    list[dict]
        Each dict:

        ``start``
            0-based start of window.
        ``end``
            0-based exclusive end.
        ``dominant_aa``
            The single most frequent amino acid in the window.
        ``fraction``
            Combined fraction of the top-*n_top* amino acids.
        ``seq_window``
            The window sub-sequence.
    """
    n = len(seq)
    results: list[dict] = []

    for i in range(0, n - window + 1, step):
        sub = seq[i:i + window]
        counts = Counter(sub)
        top_counts = sorted(counts.values(), reverse=True)[:n_top]
        top_fraction = sum(top_counts) / window
        if top_fraction > 0.70:
            dominant_aa = counts.most_common(1)[0][0]
            results.append({
                'start': i,
                'end': i + window,
                'dominant_aa': dominant_aa,
                'fraction': round(top_fraction, 3),
                'seq_window': sub,
            })

    # Merge overlapping LC windows
    if not results:
        return results

    merged: list[dict] = [results[0]]
    for r in results[1:]:
        prev = merged[-1]
        if r['start'] <= prev['end']:
            # Extend previous window
            merged[-1] = {
                'start': prev['start'],
                'end': r['end'],
                'dominant_aa': prev['dominant_aa'],
                'fraction': max(prev['fraction'], r['fraction']),
                'seq_window': seq[prev['start']:r['end']],
            }
        else:
            merged.append(r)

    return merged


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------

def calc_repeat_stats(seq: str) -> dict:
    """Compute comprehensive repeat statistics for a protein sequence.

    Parameters
    ----------
    seq:
        Protein sequence in single-letter uppercase code.

    Returns
    -------
    dict
        ``n_tandem_repeats``
            Total number of non-overlapping tandem repeats.
        ``total_tandem_coverage``
            Fraction of sequence covered by tandem repeats.
        ``n_direct_repeats``
            Number of direct-repeat pairs (capped at reporting first 100).
        ``largest_tandem_unit``
            Repeat unit of the longest tandem repeat.
        ``largest_tandem_copies``
            Copy number of the longest tandem repeat.
        ``tandem_repeats``
            Full list from :func:`find_tandem_repeats`.
        ``direct_repeats``
            First 10 entries from :func:`find_direct_repeats`.
    """
    n = len(seq)
    tandem = find_tandem_repeats(seq)
    direct = find_direct_repeats(seq)

    # Coverage
    covered: set[int] = set()
    for tr in tandem:
        covered.update(range(tr['start_1based'] - 1, tr['end_1based']))
    coverage = len(covered) / n if n > 0 else 0.0

    largest = tandem[0] if tandem else None

    return {
        'n_tandem_repeats': len(tandem),
        'total_tandem_coverage': round(coverage, 4),
        'n_direct_repeats': len(direct),
        'largest_tandem_unit': largest['unit'] if largest else '',
        'largest_tandem_copies': largest['n_copies'] if largest else 0,
        'tandem_repeats': tandem,
        'direct_repeats': direct[:10],
    }


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def format_repeats_report(seq: str, style_tag: str) -> str:
    """Generate HTML section for tandem and direct repeat analysis.

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

    stats = calc_repeat_stats(seq)
    lc = find_compositional_repeats(seq)

    summary_rows = (
        f"<tr><td>Tandem repeats</td><td>{stats['n_tandem_repeats']}</td></tr>"
        f"<tr><td>Tandem repeat coverage</td>"
        f"<td>{stats['total_tandem_coverage']:.1%}</td></tr>"
        f"<tr><td>Largest tandem unit</td>"
        f"<td><code>{stats['largest_tandem_unit'] or 'N/A'}</code> "
        f"({stats['largest_tandem_copies']} copies)</td></tr>"
        f"<tr><td>Direct repeats (&ge;4 aa, gap &le;30)</td>"
        f"<td>{stats['n_direct_repeats']}</td></tr>"
        f"<tr><td>Low-complexity windows</td><td>{len(lc)}</td></tr>"
    )

    summary_html = (
        "<h2>Sequence Repeats</h2>"
        "<table>"
        "<tr><th>Property</th><th>Value</th></tr>"
        f"{summary_rows}"
        "</table>"
    )

    # Tandem repeat table
    if stats['tandem_repeats']:
        tr_header = (
            "<tr><th>Start</th><th>End</th><th>Unit</th>"
            "<th>Unit length</th><th>Copies</th><th>Total length</th></tr>"
        )
        tr_rows = "".join(
            f"<tr>"
            f"<td>{tr['start_1based']}</td>"
            f"<td>{tr['end_1based']}</td>"
            f"<td><code>{tr['unit']}</code></td>"
            f"<td>{tr['unit_length']}</td>"
            f"<td>{tr['n_copies']}</td>"
            f"<td>{tr['total_length']}</td>"
            f"</tr>"
            for tr in stats['tandem_repeats']
        )
        tandem_html = (
            "<h2>Tandem Repeats</h2>"
            "<table>"
            f"{tr_header}{tr_rows}"
            "</table>"
            "<p class='note'>"
            "Exact k-mer extension algorithm; unit lengths 2&ndash;8; "
            "minimum 2 copies; non-overlapping."
            "</p>"
        )
    else:
        tandem_html = (
            "<h2>Tandem Repeats</h2>"
            "<p>No exact tandem repeats found "
            "(unit 2&ndash;8 aa, &ge;2 copies).</p>"
        )

    # Direct repeat table (first 10)
    if stats['direct_repeats']:
        dr_header = (
            "<tr><th>Copy 1 start</th><th>Copy 2 start</th>"
            "<th>Sequence</th><th>Length</th><th>Gap</th></tr>"
        )
        dr_rows = "".join(
            f"<tr>"
            f"<td>{dr['seq1_start_1based']}</td>"
            f"<td>{dr['seq2_start_1based']}</td>"
            f"<td><code>{dr['repeat_seq']}</code></td>"
            f"<td>{dr['length']}</td>"
            f"<td>{dr['gap']}</td>"
            f"</tr>"
            for dr in stats['direct_repeats']
        )
        direct_html = (
            "<h2>Direct Repeats (top 10 by length)</h2>"
            "<table>"
            f"{dr_header}{dr_rows}"
            "</table>"
            "<p class='note'>"
            "Direct repeats: identical sequence pairs separated by &le;30 residues, "
            "minimum match length 4 aa."
            "</p>"
        )
    else:
        direct_html = (
            "<h2>Direct Repeats</h2>"
            "<p>No direct repeats found (&ge;4 aa, gap &le;30).</p>"
        )

    # LC table
    if lc:
        lc_header = (
            "<tr><th>Start</th><th>End</th>"
            "<th>Dominant AA</th><th>Top-3 fraction</th><th>Sequence</th></tr>"
        )
        lc_rows = "".join(
            f"<tr>"
            f"<td>{w['start'] + 1}</td>"
            f"<td>{w['end']}</td>"
            f"<td>{w['dominant_aa']}</td>"
            f"<td>{w['fraction']:.1%}</td>"
            f"<td><code>{w['seq_window']}</code></td>"
            f"</tr>"
            for w in lc
        )
        lc_html = (
            "<h2>Low-Complexity Windows</h2>"
            "<table>"
            f"{lc_header}{lc_rows}"
            "</table>"
            "<p class='note'>"
            "Window = 10 aa; step = 5; flagged when top-3 AAs &gt; 70% of window. "
            "Wootton &amp; Federhen (1993) Comput. Chem. 17:149."
            "</p>"
        )
    else:
        lc_html = (
            "<h2>Low-Complexity Windows</h2>"
            "<p>No low-complexity windows detected.</p>"
        )

    return _s + summary_html + tandem_html + direct_html + lc_html

format_tandem_repeats_report = format_repeats_report  # alias expected by beer.py


# --- beer.graphs.new_graphs (RBP bug fixed) ---

# ---------------------------------------------------------------------------
# Module-level constants (no imports from beer.py)
# ---------------------------------------------------------------------------
AGGREGATION_THRESHOLD = 1.0
SOLUBILITY_NEUTRAL = 0.0
HM_THRESHOLD = 0.35
RBP_THRESHOLD = 0.3

PTM_COLORS = {
    "phospho": "#1f77b4",
    "glycosylation": "#2ca02c",
    "ubiquitination": "#ff7f0e",
    "sumo": "#9467bd",
    "acetylation": "#17becf",
    "methylation": "#e377c2",
    "palmitoylation": "#8c564b",
}

MW_STANDARDS_KDA = [10, 15, 20, 25, 37, 50, 75, 100, 150, 250]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _apply_font_sizes(ax, label_font: int, tick_font: int) -> None:
    """Apply consistent label and tick font sizes to an Axes object."""
    ax.xaxis.label.set_fontsize(label_font)
    ax.yaxis.label.set_fontsize(label_font)
    ax.title.set_fontsize(label_font)
    ax.tick_params(axis="both", labelsize=tick_font)


def _residue_x(seq: str) -> np.ndarray:
    """Return 1-based residue position array for a sequence string."""
    return np.arange(1, len(seq) + 1, dtype=float)


# ---------------------------------------------------------------------------
# 1. Aggregation profile
# ---------------------------------------------------------------------------

def create_aggregation_profile_figure(
    seq: str,
    aggregation_profile: list,
    hotspots: list,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Line plot of per-residue β-aggregation propensity (Zyggregator).

    Parameters
    ----------
    seq:
        Protein sequence string (used for x-axis length).
    aggregation_profile:
        Per-residue aggregation score (same length as seq).
    hotspots:
        List of (start, end) tuples (1-based, inclusive) marking hotspot
        regions to shade in red.
    label_font, tick_font:
        Font sizes for axis labels/titles and tick labels.

    Returns
    -------
    matplotlib.figure.Figure
    """
    x = _residue_x(seq)
    y = np.asarray(aggregation_profile, dtype=float)

    fig = Figure(figsize=(10, 4), tight_layout=True)
    ax = fig.add_subplot(111)

    ax.plot(x, y, color="steelblue", linewidth=1.5, label="Aggregation propensity")

    # Fill orange where y > threshold
    ax.fill_between(x, AGGREGATION_THRESHOLD, y, where=(y > AGGREGATION_THRESHOLD),
                    interpolate=True, color="orange", alpha=0.6,
                    label=f"Above threshold ({AGGREGATION_THRESHOLD})")

    # Dashed threshold line
    ax.axhline(AGGREGATION_THRESHOLD, color="black", linestyle="--", linewidth=1.0,
               label=f"Threshold ({AGGREGATION_THRESHOLD})")

    # Red hotspot rectangles
    y_min, y_max = ax.get_ylim()
    for idx, hs in enumerate(hotspots):
        start, end = hs[0], hs[1]
        label_hs = "Hotspot" if idx == 0 else "_nolegend_"
        rect = Rectangle(
            (start - 0.5, y_min),
            (end - start + 1),
            y_max - y_min,
            linewidth=0,
            edgecolor="none",
            facecolor="red",
            alpha=0.25,
            label=label_hs,
            zorder=0,
        )
        ax.add_patch(rect)

    ax.set_xlabel("Residue Position", fontsize=label_font)
    ax.set_ylabel("β-Aggregation Propensity", fontsize=label_font)
    ax.set_title("β-Aggregation Propensity Profile (Zyggregator)", fontsize=label_font)
    ax.set_xlim(x[0], x[-1])
    ax.tick_params(axis="both", labelsize=tick_font)
    ax.legend(fontsize=tick_font, loc="upper right")

    return fig


# ---------------------------------------------------------------------------
# 2. CamSol solubility profile
# ---------------------------------------------------------------------------

def create_solubility_profile_figure(
    seq: str,
    camsolmt_profile: list,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Per-residue CamSol intrinsic solubility profile.

    Parameters
    ----------
    seq:
        Protein sequence string.
    camsolmt_profile:
        Per-residue CamSol scores (same length as seq).
    label_font, tick_font:
        Font sizes.

    Returns
    -------
    matplotlib.figure.Figure
    """
    x = _residue_x(seq)
    y = np.asarray(camsolmt_profile, dtype=float)

    fig = Figure(figsize=(10, 4), tight_layout=True)
    ax = fig.add_subplot(111)

    ax.plot(x, y, color="black", linewidth=1.2, label="CamSol score")

    ax.fill_between(x, 0, y, where=(y >= 0), interpolate=True,
                    color="green", alpha=0.5, label="Soluble (>0)")
    ax.fill_between(x, 0, y, where=(y < 0), interpolate=True,
                    color="red", alpha=0.5, label="Insoluble (<0)")

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)

    ax.set_xlabel("Residue Position", fontsize=label_font)
    ax.set_ylabel("CamSol Score", fontsize=label_font)
    ax.set_title("CamSol Intrinsic Solubility Profile", fontsize=label_font)
    ax.set_xlim(x[0], x[-1])
    ax.tick_params(axis="both", labelsize=tick_font)
    ax.legend(fontsize=tick_font, loc="upper right")

    return fig


# ---------------------------------------------------------------------------
# 3. Hydrophobic moment profile
# ---------------------------------------------------------------------------

def create_hydrophobic_moment_figure(
    seq: str,
    moment_alpha: list,
    moment_beta: list,
    amphipathic_regions: list,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Hydrophobic moment profile for α-helix and β-strand windows.

    Parameters
    ----------
    seq:
        Protein sequence string.
    moment_alpha:
        μH values for α-helix (δ=100°) per residue position.
    moment_beta:
        μH values for β-strand (δ=160°) per residue position.
    amphipathic_regions:
        List of (start, end) tuples (1-based) marking amphipathic regions.
    label_font, tick_font:
        Font sizes.

    Returns
    -------
    matplotlib.figure.Figure
    """
    x = _residue_x(seq)
    ya = np.asarray(moment_alpha, dtype=float)
    yb = np.asarray(moment_beta, dtype=float)

    fig = Figure(figsize=(10, 4), tight_layout=True)
    ax = fig.add_subplot(111)

    ax.plot(x, ya, color="blue", linewidth=1.5,
            label="μH (α-helix, δ=100°)")
    ax.plot(x, yb, color="red", linewidth=1.5,
            label="μH (β-strand, δ=160°)")

    ax.axhline(HM_THRESHOLD, color="grey", linestyle="--", linewidth=1.0,
               label=f"Threshold ({HM_THRESHOLD})")

    # Green rectangles for amphipathic regions
    y_min, y_max = 0, max(np.max(ya), np.max(yb)) * 1.15 + 0.1
    for idx, region in enumerate(amphipathic_regions):
        start, end = region[0], region[1]
        label_amp = "Amphipathic" if idx == 0 else "_nolegend_"
        rect = Rectangle(
            (start - 0.5, 0),
            (end - start + 1),
            y_max,
            linewidth=0,
            facecolor="green",
            alpha=0.20,
            label=label_amp,
            zorder=0,
        )
        ax.add_patch(rect)

    ax.set_xlabel("Residue Position", fontsize=label_font)
    ax.set_ylabel("μH", fontsize=label_font)
    ax.set_title("Hydrophobic Moment Profile", fontsize=label_font)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="both", labelsize=tick_font)
    ax.legend(fontsize=tick_font, loc="upper right")

    return fig


# ---------------------------------------------------------------------------
# 4. pI / MW 2D gel map
# ---------------------------------------------------------------------------

def create_pI_MW_gel_figure(
    proteins_data: list,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """2D scatter plot of pI vs log10(MW) — SDS-PAGE proxy.

    Parameters
    ----------
    proteins_data:
        List of dicts with keys: ``name``, ``pI``, ``mol_weight`` (Da),
        and optional ``color``.
    label_font, tick_font:
        Font sizes.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = Figure(figsize=(9, 6), tight_layout=True)
    ax = fig.add_subplot(111)

    single = len(proteins_data) == 1

    # MW standard dashed lines
    for mw_kda in MW_STANDARDS_KDA:
        log_mw = math.log10(mw_kda * 1000)
        ax.axhline(log_mw, color="lightgrey", linestyle="--", linewidth=0.8, zorder=0)
        ax.text(13.8, log_mw, f"{mw_kda} kDa", va="center", ha="right",
                fontsize=max(tick_font - 3, 7), color="grey")

    # Vertical reference lines
    ax.axvline(7.0, color="grey", linestyle="--", linewidth=0.8, zorder=0)
    ax.axvline(4.0, color="lightblue", linestyle=":", linewidth=0.8, zorder=0)
    ax.axvline(10.0, color="lightcoral", linestyle=":", linewidth=0.8, zorder=0)

    scatter_artists = []
    for pdata in proteins_data:
        pI_val = float(pdata["pI"])
        mw_val = float(pdata["mol_weight"])
        log_mw = math.log10(mw_val)
        color = pdata.get("color", "steelblue")
        marker = "*" if single else "o"
        ms = 18 if single else 10
        sc = ax.scatter(pI_val, log_mw, color=color, marker=marker,
                        s=ms**2, zorder=5, edgecolors="black", linewidths=0.5)
        ax.annotate(
            pdata["name"],
            xy=(pI_val, log_mw),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=max(tick_font - 2, 8),
        )
        scatter_artists.append(sc)

    # mplcursors tooltip showing name, pI, MW
    if scatter_artists:
        try:
            cursor = mplcursors.cursor(scatter_artists, hover=True)

            @cursor.connect("add")
            def on_add(sel):
                idx = sel.index
                if idx < len(proteins_data):
                    p = proteins_data[idx]
                    sel.annotation.set_text(
                        f"{p['name']}\npI={p['pI']:.2f}\nMW={p['mol_weight']:.0f} Da"
                    )
        except Exception:
            pass  # mplcursors may not function outside a GUI event loop

    # Y-axis ticks: show kDa labels at standard positions
    std_log_ticks = [math.log10(mw * 1000) for mw in MW_STANDARDS_KDA]
    ax.set_yticks(std_log_ticks)
    ax.set_yticklabels([f"{mw} kDa" for mw in MW_STANDARDS_KDA], fontsize=tick_font)

    ax.set_xlabel("Isoelectric Point (pI)", fontsize=label_font)
    ax.set_ylabel("Molecular Weight", fontsize=label_font)
    ax.set_title("pI / MW 2D Map (SDS-PAGE Proxy)", fontsize=label_font)
    ax.set_xlim(0, 14)
    ax.tick_params(axis="x", labelsize=tick_font)

    return fig


# ---------------------------------------------------------------------------
# 5. PTM profile (lollipop)
# ---------------------------------------------------------------------------

def create_ptm_profile_figure(
    seq: str,
    ptm_sites: list,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Lollipop/stem plot of predicted PTM sites.

    Parameters
    ----------
    seq:
        Protein sequence string.
    ptm_sites:
        List of dicts: {``position`` (1-based int), ``ptm_type`` (str)}.
        PTM types: phospho, glycosylation, ubiquitination, sumo,
        acetylation, methylation, palmitoylation.
    label_font, tick_font:
        Font sizes.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Collect all PTM types present
    ptm_types_present = []
    for site in ptm_sites:
        pt = site.get("ptm_type", "unknown")
        if pt not in ptm_types_present:
            ptm_types_present.append(pt)

    # Assign y-index per PTM type
    ptm_y_map = {pt: i for i, pt in enumerate(ptm_types_present)}
    n_types = max(len(ptm_types_present), 1)

    fig = Figure(figsize=(12, max(3, n_types * 1.2 + 1.5)), tight_layout=True)
    ax = fig.add_subplot(111)

    for site in ptm_sites:
        pos = site.get("position", 1)
        pt = site.get("ptm_type", "unknown")
        y_idx = ptm_y_map.get(pt, 0)
        color = PTM_COLORS.get(pt, "#7f7f7f")

        # Vertical stem
        ax.plot([pos, pos], [y_idx - 0.35, y_idx], color=color,
                linewidth=1.0, zorder=2)
        # Dot
        ax.scatter([pos], [y_idx], color=color, s=60, zorder=3,
                   edgecolors="black", linewidths=0.4)

    # Y-axis labels = PTM types
    ax.set_yticks(list(ptm_y_map.values()))
    ax.set_yticklabels(list(ptm_y_map.keys()), fontsize=tick_font)
    ax.set_ylim(-0.8, n_types - 0.2)

    ax.set_xlabel("Residue Position", fontsize=label_font)
    ax.set_ylabel("PTM Type", fontsize=label_font)
    ax.set_title("Post-Translational Modification Sites (Predicted)", fontsize=label_font)
    ax.set_xlim(0, len(seq) + 1)
    ax.tick_params(axis="x", labelsize=tick_font)

    # Legend patches
    legend_patches = [
        mpatches.Patch(color=PTM_COLORS.get(pt, "#7f7f7f"), label=pt)
        for pt in ptm_types_present
    ]
    if legend_patches:
        ax.legend(handles=legend_patches, fontsize=tick_font,
                  loc="upper right", title="PTM Type",
                  title_fontsize=tick_font)

    return fig


# ---------------------------------------------------------------------------
# 6. RNA-binding propensity profile
# ---------------------------------------------------------------------------

def create_rbp_profile_figure(
    seq: str,
    rbp_profile: list,
    rbp_motifs: list,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Sliding-window RNA-binding propensity profile.

    Parameters
    ----------
    seq:
        Protein sequence string.
    rbp_profile:
        Per-residue RBP propensity scores (same length as seq).
    rbp_motifs:
        List of dicts: {``start``, ``end``, ``motif_name``, optional ``color``}.
    label_font, tick_font:
        Font sizes.

    Returns
    -------
    matplotlib.figure.Figure
    """
    x = _residue_x(seq)
    y = np.asarray(rbp_profile, dtype=float)
    if len(x) != len(y):  # sliding window produces shorter array
        x = x[:len(y)]

    fig = Figure(figsize=(10, 4), tight_layout=True)
    ax = fig.add_subplot(111)

    ax.plot(x, y, color="teal", linewidth=1.4, label="RBP propensity")

    ax.fill_between(x, RBP_THRESHOLD, y, where=(y > RBP_THRESHOLD),
                    interpolate=True, color="teal", alpha=0.4,
                    label=f"Above threshold ({RBP_THRESHOLD})")

    ax.axhline(RBP_THRESHOLD, color="black", linestyle="--", linewidth=1.0,
               label=f"Threshold ({RBP_THRESHOLD})")

    # RBP motif colored spans
    motif_colors = [
        "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
        "#911eb4", "#42d4f4", "#f032e6",
    ]
    for i, motif in enumerate(rbp_motifs):
        start = motif.get("start", 1)
        end = motif.get("end", start)
        mcolor = motif.get("color", motif_colors[i % len(motif_colors)])
        mname = motif.get("motif_name", f"Motif {i+1}")
        ax.axvspan(start - 0.5, end + 0.5, color=mcolor, alpha=0.25,
                   label=mname, zorder=0)

    ax.set_xlabel("Residue Position", fontsize=label_font)
    ax.set_ylabel("RBP Propensity", fontsize=label_font)
    ax.set_title("RNA-Binding Propensity Profile", fontsize=label_font)
    ax.set_xlim(x[0], x[-1])
    ax.tick_params(axis="both", labelsize=tick_font)
    ax.legend(fontsize=tick_font, loc="upper right")

    return fig


# ---------------------------------------------------------------------------
# 7. Truncation series analysis (2×3 multi-panel)
# ---------------------------------------------------------------------------

def create_truncation_series_figure(
    truncation_data: dict,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Multi-panel (2×3) figure showing how 6 properties change with truncation.

    Parameters
    ----------
    truncation_data:
        Dict with keys ``n_trunc`` and ``c_trunc``, each a list of dicts:
        {``pct``, ``pI``, ``gravy``, ``fcr``, ``ncpr``, ``net_charge_7``,
        ``disorder_frac``}.
    label_font, tick_font:
        Font sizes.

    Returns
    -------
    matplotlib.figure.Figure
    """
    panels = [
        ("pI", "pI"),
        ("gravy", "GRAVY"),
        ("fcr", "FCR"),
        ("ncpr", "NCPR"),
        ("net_charge_7", "Net Charge (pH 7)"),
        ("disorder_frac", "Disorder Fraction"),
    ]

    n_trunc = truncation_data.get("n_trunc", [])
    c_trunc = truncation_data.get("c_trunc", [])

    def extract(data_list, key):
        return (
            [d.get("pct", 0) for d in data_list],
            [d.get(key, 0) for d in data_list],
        )

    fig = Figure(figsize=(14, 8), tight_layout=True)
    fig.suptitle("Truncation Series Analysis", fontsize=label_font + 2)

    for panel_idx, (key, ylabel) in enumerate(panels):
        ax = fig.add_subplot(2, 3, panel_idx + 1)

        nx, ny = extract(n_trunc, key)
        cx, cy = extract(c_trunc, key)

        if nx:
            ax.plot(nx, ny, color="blue", linewidth=1.5, marker="o",
                    markersize=4, label="N-terminal")
        if cx:
            ax.plot(cx, cy, color="red", linewidth=1.5, marker="s",
                    markersize=4, label="C-terminal")

        ax.set_xlabel("Truncation (%)", fontsize=label_font - 2)
        ax.set_ylabel(ylabel, fontsize=label_font - 2)
        ax.tick_params(axis="both", labelsize=tick_font - 1)
        ax.set_xlim(0, 90)

        if panel_idx == 0:
            ax.legend(fontsize=tick_font - 2, loc="best")

    return fig


# ---------------------------------------------------------------------------
# 8. SCD profile
# ---------------------------------------------------------------------------

def create_scd_profile_figure(
    seq: str,
    scd_profile: list,
    window: int,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Sliding-window SCD (Sequence Charge Decoration) profile.

    Parameters
    ----------
    seq:
        Protein sequence string.
    scd_profile:
        Per-window SCD values.  Length may be <= len(seq).
    window:
        Window size used for the SCD calculation (for title display).
    label_font, tick_font:
        Font sizes.

    Returns
    -------
    matplotlib.figure.Figure
    """
    x = np.arange(1, len(scd_profile) + 1, dtype=float)
    y = np.asarray(scd_profile, dtype=float)

    fig = Figure(figsize=(10, 4), tight_layout=True)
    ax = fig.add_subplot(111)

    ax.plot(x, y, color="black", linewidth=1.2)

    ax.fill_between(x, 0, y, where=(y > 0), interpolate=True,
                    color="red", alpha=0.5, label="Segregated charges (+)")
    ax.fill_between(x, 0, y, where=(y < 0), interpolate=True,
                    color="blue", alpha=0.5, label="Mixed charges (−)")

    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)

    ax.set_xlabel("Residue Position", fontsize=label_font)
    ax.set_ylabel("SCD", fontsize=label_font)
    ax.set_title(
        f"SCD Profile (Sequence Charge Decoration, window={window})",
        fontsize=label_font,
    )
    ax.set_xlim(x[0], x[-1])
    ax.tick_params(axis="both", labelsize=tick_font)
    ax.legend(fontsize=tick_font, loc="upper right")

    return fig


# ---------------------------------------------------------------------------
# 9. Ramachandran plot
# ---------------------------------------------------------------------------

def create_ramachandran_figure(
    phi_psi_data: list,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Classical Ramachandran plot coloured by secondary structure.

    Parameters
    ----------
    phi_psi_data:
        List of dicts: {``phi``, ``psi``, ``resname``, ``resnum``, ``ss``}.
        ``ss`` values: 'H' = helix, 'E' = sheet, 'C' = coil.
    label_font, tick_font:
        Font sizes.

    Returns
    -------
    matplotlib.figure.Figure
    """
    SS_COLORS = {"H": "#1f77b4", "E": "#d62728", "C": "#aaaaaa"}
    SS_LABELS = {"H": "α-Helix", "E": "β-Sheet", "C": "Coil"}

    fig = Figure(figsize=(7, 6), tight_layout=True)
    ax = fig.add_subplot(111)

    # ---- Allowed/generous regions as filled rectangles ----
    # Core α-helix
    ax.add_patch(Rectangle((-80, -60), 32, 40, color="#555555", alpha=0.25,
                            zorder=0, label="_helix_region"))
    # Core β-sheet
    ax.add_patch(Rectangle((-150, 90), 60, 70, color="#888888", alpha=0.20,
                            zorder=0, label="_sheet_region"))
    # Left-handed helix (positive phi)
    ax.add_patch(Rectangle((40, 20), 40, 40, color="#bbbbbb", alpha=0.20,
                            zorder=0, label="_lh_region"))

    # Crosshairs
    ax.axhline(0, color="grey", linewidth=0.6, linestyle="-")
    ax.axvline(0, color="grey", linewidth=0.6, linestyle="-")

    # Plot residues
    plotted_ss = set()
    for res in phi_psi_data:
        phi = res.get("phi")
        psi = res.get("psi")
        ss = res.get("ss", "C")
        if phi is None or psi is None:
            continue
        color = SS_COLORS.get(ss, "#aaaaaa")
        label = SS_LABELS.get(ss, ss) if ss not in plotted_ss else "_nolegend_"
        plotted_ss.add(ss)
        ax.scatter(phi, psi, color=color, s=12, alpha=0.7,
                   edgecolors="none", label=label, zorder=3)

    # Allowed region annotation patches for legend
    helix_patch = mpatches.Patch(color="#555555", alpha=0.4, label="Core α-helix region")
    sheet_patch = mpatches.Patch(color="#888888", alpha=0.35, label="Core β-sheet region")
    lh_patch = mpatches.Patch(color="#bbbbbb", alpha=0.35, label="Left-handed helix")

    # Build legend
    handles, labels = ax.get_legend_handles_labels()
    # Filter out internal labels
    visible = [(h, l) for h, l in zip(handles, labels) if not l.startswith("_")]
    visible_h, visible_l = zip(*visible) if visible else ([], [])
    ax.legend(
        list(visible_h) + [helix_patch, sheet_patch, lh_patch],
        list(visible_l) + ["Core α-helix", "Core β-sheet", "Left-handed helix"],
        fontsize=tick_font,
        loc="upper right",
        markerscale=1.5,
    )

    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xlabel("φ (°)", fontsize=label_font)
    ax.set_ylabel("ψ (°)", fontsize=label_font)
    ax.set_title("Ramachandran Plot (from PDB structure)", fontsize=label_font)
    ax.tick_params(axis="both", labelsize=tick_font)
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks(range(-180, 181, 60))

    return fig


# ---------------------------------------------------------------------------
# 10. Residue contact network
# ---------------------------------------------------------------------------

def _circular_layout(n: int) -> np.ndarray:
    """Return (n, 2) xy positions for n nodes evenly spaced on a unit circle."""
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _spring_layout(adj: np.ndarray, n_iter: int = 200,
                   seed: int = 42) -> np.ndarray:
    """Fruchterman-Reingold spring layout (pure numpy).

    Parameters
    ----------
    adj:
        (n, n) boolean adjacency matrix.
    n_iter:
        Number of iterations.

    Returns
    -------
    pos : np.ndarray of shape (n, 2)
    """
    n = adj.shape[0]
    rng = np.random.default_rng(seed)
    pos = rng.random((n, 2)) * 2 - 1  # uniform in [-1, 1]

    k = 1.0 / math.sqrt(n) if n > 0 else 1.0
    t = 0.1  # initial temperature

    for _ in range(n_iter):
        delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (n,n,2)
        dist = np.linalg.norm(delta, axis=2)  # (n,n)
        np.fill_diagonal(dist, 1e-9)

        # Repulsive forces
        rep = k**2 / dist  # (n, n)
        rep_force = np.einsum("ij,ijk->ik", rep / dist, delta)  # (n, 2)

        # Attractive forces (only connected pairs)
        att = (dist**2 / k) * adj
        att_force = np.einsum("ij,ijk->ik", att / dist, -delta)  # (n, 2)

        displacement = rep_force + att_force
        # Cap displacement by temperature
        disp_len = np.linalg.norm(displacement, axis=1, keepdims=True)
        disp_len = np.maximum(disp_len, 1e-9)
        pos += displacement / disp_len * np.minimum(disp_len, t)

        t *= 0.95  # cool down

    # Normalise to [-1, 1]
    pos -= pos.min(axis=0)
    pos /= pos.max(axis=0) + 1e-9
    pos = pos * 2 - 1

    return pos


def create_contact_network_figure(
    seq: str,
    dist_matrix: np.ndarray,
    cutoff_angstrom: float = 8.0,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Residue contact network derived from Cα distance matrix.

    Parameters
    ----------
    seq:
        Protein sequence string.
    dist_matrix:
        n×n numpy array of Cα pairwise distances (Å).
    cutoff_angstrom:
        Distance cutoff for defining contacts.
    label_font, tick_font:
        Font sizes.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = len(seq)
    dist_matrix = np.asarray(dist_matrix, dtype=float)

    # Build adjacency (ignore self-contacts and sequence neighbours ±1)
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 2, n):
            if dist_matrix[i, j] <= cutoff_angstrom:
                adj[i, j] = True
                adj[j, i] = True

    degree = adj.sum(axis=1).astype(float)  # (n,)
    max_deg = degree.max() if degree.max() > 0 else 1.0

    large_protein = n > 100
    top_n = 30

    if large_protein:
        # Show only top-30 by degree
        top_idx = np.argsort(degree)[-top_n:]
        sub_n = len(top_idx)
        idx_map = {old: new for new, old in enumerate(top_idx)}
        sub_degree = degree[top_idx]
        sub_adj = adj[np.ix_(top_idx, top_idx)]
        sub_seq = [seq[i] for i in top_idx]
        pos = _circular_layout(sub_n)
        node_degrees = sub_degree
        plot_adj = sub_adj
        residue_labels = [f"{top_idx[i]+1}" for i in range(sub_n)]
    else:
        pos = _spring_layout(adj.astype(float))
        node_degrees = degree
        plot_adj = adj
        sub_n = n
        residue_labels = [str(i + 1) for i in range(n)]

    # Degree centrality: normalise by (n-1)
    norm_n = (n - 1) if n > 1 else 1
    centrality = node_degrees / norm_n

    fig = Figure(figsize=(9, 8), tight_layout=True)
    ax = fig.add_subplot(111)

    cmap = cm.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=0, vmax=centrality.max() if centrality.max() > 0 else 1)

    # Draw edges
    for i in range(sub_n):
        for j in range(i + 1, sub_n):
            if plot_adj[i, j]:
                # Get original indices for distance lookup
                if large_protein:
                    orig_i = top_idx[i]
                    orig_j = top_idx[j]
                else:
                    orig_i, orig_j = i, j
                d = dist_matrix[orig_i, orig_j]
                lw = max(0.2, 2.0 / (d + 1e-9) * cutoff_angstrom)
                ax.plot(
                    [pos[i, 0], pos[j, 0]],
                    [pos[i, 1], pos[j, 1]],
                    color="lightgrey",
                    linewidth=min(lw, 2.0),
                    zorder=1,
                )

    # Draw nodes
    node_sizes = 50 + (node_degrees / max_deg) * 250
    scatter = ax.scatter(
        pos[:, 0], pos[:, 1],
        c=centrality, cmap="viridis", norm=norm,
        s=node_sizes, zorder=3, edgecolors="black", linewidths=0.4,
    )

    # Node labels (only if small protein)
    if not large_protein and n <= 50:
        for i in range(sub_n):
            ax.text(pos[i, 0], pos[i, 1], residue_labels[i],
                    ha="center", va="center",
                    fontsize=max(tick_font - 4, 6), zorder=4)

    # Colorbar
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Degree Centrality", fontsize=label_font - 2)
    cbar.ax.tick_params(labelsize=tick_font - 2)

    ax.set_aspect("equal")
    ax.axis("off")
    subtitle = ""
    if large_protein:
        subtitle = f" — Top {top_n} residues by degree"
    ax.set_title(
        f"Residue Contact Network (Cα ≤ {cutoff_angstrom} Å){subtitle}",
        fontsize=label_font,
    )

    return fig


# ---------------------------------------------------------------------------
# 11. MSA conservation profile
# ---------------------------------------------------------------------------

def _shannon_entropy(column_chars: list) -> float:
    """Shannon entropy (bits) for a list of characters (including gaps)."""
    total = len(column_chars)
    if total == 0:
        return 0.0
    counts = Counter(column_chars)
    probs = [c / total for c in counts.values()]
    h = -sum(p * math.log2(p) for p in probs if p > 0)
    return h


def create_msa_conservation_figure(
    sequences: list,
    names: list,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Per-column conservation bar chart from a multiple sequence alignment.

    Parameters
    ----------
    sequences:
        List of equal-length strings (aligned sequences; gaps = '-').
    names:
        Sequence names (same length as sequences; used in tooltip/legend).
    label_font, tick_font:
        Font sizes.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not sequences:
        fig = Figure(figsize=(10, 4))
        ax = fig.add_subplot(111)
        ax.set_title("MSA Conservation Profile (no data)", fontsize=label_font)
        return fig

    n_seq = len(sequences)
    aln_len = len(sequences[0])
    positions = np.arange(1, aln_len + 1)

    conservation = np.zeros(aln_len)
    dominant_aa = [""] * aln_len

    for col in range(aln_len):
        chars = [sequences[s][col] for s in range(n_seq)
                 if col < len(sequences[s])]
        h = _shannon_entropy(chars)
        n_distinct = len(set(chars))
        if n_distinct > 1:
            conservation[col] = max(0.0, 1.0 - h / math.log2(n_distinct))
        else:
            conservation[col] = 1.0  # perfectly conserved

        # Most common character (excluding gaps for dominant AA display)
        non_gap = [c for c in chars if c != "-"]
        if non_gap:
            dominant_aa[col] = Counter(non_gap).most_common(1)[0][0]
        else:
            dominant_aa[col] = "-"

    fig = Figure(figsize=(max(10, aln_len * 0.15), 4), tight_layout=True)
    ax = fig.add_subplot(111)

    ax.bar(positions, conservation, color="steelblue", width=0.8, zorder=2)

    # Dominant AA text overlay where conservation > 0.7
    for col in range(aln_len):
        if conservation[col] > 0.7 and dominant_aa[col] not in ("-", ""):
            ax.text(
                positions[col], conservation[col] + 0.02,
                dominant_aa[col],
                ha="center", va="bottom",
                fontsize=max(tick_font - 4, 6),
                color="darkblue",
            )

    ax.axhline(0.7, color="red", linestyle="--", linewidth=0.8,
               label="Conservation = 0.7")

    ax.set_xlabel("Alignment Position", fontsize=label_font)
    ax.set_ylabel("Conservation", fontsize=label_font)
    ax.set_title("MSA Conservation Profile", fontsize=label_font)
    ax.set_xlim(0.5, aln_len + 0.5)
    ax.set_ylim(0, 1.15)
    ax.tick_params(axis="both", labelsize=tick_font)
    ax.legend(fontsize=tick_font, loc="upper right")

    return fig


# ---------------------------------------------------------------------------
# 12. Protein complex MW bar chart
# ---------------------------------------------------------------------------

def create_complex_mw_figure(
    chains_data: list,
    stoichiometry_str: str,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Bar chart of individual chain MWs and the assembled complex MW.

    Parameters
    ----------
    chains_data:
        List of dicts: {``chain_id`` (str), ``mol_weight`` (Da), ``color`` (optional str)}.
    stoichiometry_str:
        Stoichiometry string, e.g. ``"A2B1"``.  Used for title/annotation.
    label_font, tick_font:
        Font sizes.

    Returns
    -------
    matplotlib.figure.Figure
    """

    # Parse stoichiometry string like "A2B1" → {A: 2, B: 1}
    stoich_map: dict = {}
    for match in re.finditer(r"([A-Za-z]+)(\d*)", stoichiometry_str):
        chain_id = match.group(1)
        count_str = match.group(2)
        count = int(count_str) if count_str else 1
        if chain_id:
            stoich_map[chain_id] = count

    # Build labels and heights
    chain_ids = [c["chain_id"] for c in chains_data]
    chain_mws = [float(c["mol_weight"]) for c in chains_data]
    default_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    ]
    chain_colors = [
        c.get("color", default_colors[i % len(default_colors)])
        for i, c in enumerate(chains_data)
    ]

    # Total complex MW = sum(n_copies * mw) for each chain
    total_mw = 0.0
    for chain in chains_data:
        cid = chain["chain_id"]
        n_copies = stoich_map.get(cid, 1)
        total_mw += n_copies * float(chain["mol_weight"])

    bar_labels = chain_ids + ["Complex"]
    bar_heights = chain_mws + [total_mw]
    bar_colors_all = chain_colors + ["#333333"]

    fig = Figure(figsize=(max(6, len(bar_labels) * 1.2 + 2), 5), tight_layout=True)
    ax = fig.add_subplot(111)

    bars = ax.bar(
        range(len(bar_labels)),
        [mw / 1000 for mw in bar_heights],  # convert to kDa
        color=bar_colors_all,
        edgecolor="black",
        linewidth=0.6,
    )

    # Annotate bars with values
    for bar_obj, height_kda in zip(bars, [h / 1000 for h in bar_heights]):
        ax.text(
            bar_obj.get_x() + bar_obj.get_width() / 2,
            bar_obj.get_height() + 0.5,
            f"{height_kda:.1f} kDa",
            ha="center", va="bottom",
            fontsize=max(tick_font - 2, 8),
        )

    # Total complex annotation
    ax.annotate(
        f"Total Complex: {total_mw/1000:.1f} kDa\n(Stoichiometry: {stoichiometry_str})",
        xy=(len(bar_labels) - 1, total_mw / 1000),
        xytext=(-60, 20),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=tick_font,
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="grey"),
    )

    ax.set_xticks(range(len(bar_labels)))
    ax.set_xticklabels(bar_labels, fontsize=tick_font)
    ax.set_ylabel("Molecular Weight (kDa)", fontsize=label_font)
    ax.set_title("Protein Complex Mass Composition", fontsize=label_font)
    ax.tick_params(axis="y", labelsize=tick_font)

    return fig


# --- beer.network.elm ---

_ELM_BASE_URL = "https://elm.eu.org/instances.json"
_TIMEOUT_SECONDS = 30


class ELMWorker(QThread):
    """Query ELM database for a sequence to find experimentally validated linear motifs.

    Uses ELM REST API: https://elm.eu.org/
    Endpoint: GET https://elm.eu.org/instances.json?q={uniprot_id}

    Signals
    -------
    finished(list):
        Emitted on success with a list of dicts, each containing:
        - elm_identifier (str)
        - start (int)
        - end (int)
        - logic (str)          "positive" | "negative" | "neutral"
        - toGo (str)           GO term(s)
        - primary_reference_pmed_id (str)
        - accession (str)      copy of the queried accession
    error(str):
        Emitted if the query fails, with a human-readable error message.
    progress(str):
        Emitted with status updates during the query.
    """

    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, accession: str, seq: str = ""):
        """Initialise the worker.

        Parameters
        ----------
        accession:
            UniProt accession (e.g. ``"P04637"``).  Will be stripped and
            upper-cased before use.
        seq:
            Protein sequence string (currently unused; reserved for future
            sequence-based search support).
        """
        super().__init__()
        self.accession = accession.strip().upper()
        self.seq = seq

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Fetch ELM instances for the stored accession.

        Builds the query URL, performs an HTTP GET request, parses the JSON
        response, and emits ``finished`` with the list of motif dicts.
        On any error (network, HTTP, JSON), emits ``error`` with a
        descriptive message.
        """
        if not self.accession:
            self.error.emit("ELM query failed: no accession provided.")
            return

        url = f"{_ELM_BASE_URL}?q={self.accession}"
        self.progress.emit(f"Querying ELM database for {self.accession}…")

        try:
            req = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "BEER-biophysics/1.0 (scientific software)",
                },
            )
            with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as response:
                raw = response.read().decode("utf-8")

        except urllib.error.HTTPError as exc:
            self.error.emit(
                f"ELM HTTP error {exc.code} for accession '{self.accession}': {exc.reason}"
            )
            return
        except urllib.error.URLError as exc:
            self.error.emit(
                f"ELM network error for accession '{self.accession}': {exc.reason}"
            )
            return
        except Exception as exc:
            self.error.emit(f"ELM query failed for '{self.accession}': {exc}")
            return

        # Parse JSON
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            self.error.emit(
                f"ELM returned invalid JSON for '{self.accession}': {exc}"
            )
            return

        # Extract instances list — ELM may return the list directly or nested
        if isinstance(data, list):
            raw_instances = data
        elif isinstance(data, dict):
            # Top-level key may be "instances" or the accession itself
            raw_instances = (
                data.get("instances")
                or data.get("Instances")
                or []
            )
            # Some ELM responses wrap results per-accession
            if not isinstance(raw_instances, list):
                raw_instances = []
        else:
            raw_instances = []

        if not raw_instances:
            self.progress.emit(
                f"No ELM instances found for accession '{self.accession}'."
            )
            self.finished.emit([])
            return

        instances: list = []
        for item in raw_instances:
            if not isinstance(item, dict):
                continue
            instances.append(
                {
                    "elm_identifier": item.get("elm_identifier", item.get("elm_type", "")),
                    "start": _safe_int(item.get("start", item.get("Start", 0))),
                    "end": _safe_int(item.get("end", item.get("End", 0))),
                    "logic": item.get("logic", item.get("Logic", "")),
                    "toGo": item.get("toGo", item.get("to_go", "")),
                    "primary_reference_pmed_id": item.get(
                        "primary_reference_pmed_id",
                        item.get("pmed_id", ""),
                    ),
                    "accession": self.accession,
                    # Preserve any extra fields returned by ELM
                    "raw": item,
                }
            )

        self.progress.emit(
            f"ELM: found {len(instances)} instance(s) for '{self.accession}'."
        )
        self.finished.emit(instances)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _safe_int(value, default: int = 0) -> int:
    """Convert *value* to int, returning *default* on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# --- beer.network.disprot ---

_DISPROT_API_BASE = "https://disprot.org/api"
_TIMEOUT_SECONDS = 30


class DisPRotWorker(QThread):
    """Query DisProt database for disorder annotations of a UniProt accession.

    REST API: GET https://disprot.org/api/{uniprot_id}

    Signals
    -------
    finished(dict):
        Emitted on success (including the case of a valid 404 / not found).
        Dict keys:
        - found (bool)              False if the protein is not in DisProt
        - disprot_id (str)
        - protein_name (str)
        - accession (str)           Echo of the queried accession
        - sequence_length (int)     Total protein length (0 if unknown)
        - regions (list)            List of dicts: {start, end, type}
        - n_disordered_aa (int)     Total disordered residues (may overlap)
        - fraction_disordered (float)  n_disordered_aa / sequence_length
    error(str):
        Emitted on network/parse errors with a human-readable message.
    """

    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, accession: str):
        """Initialise the worker.

        Parameters
        ----------
        accession:
            UniProt accession (e.g. ``"P04637"``).  Stripped and upper-cased.
        """
        super().__init__()
        self.accession = accession.strip().upper()

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Fetch DisProt disorder annotations for the stored accession.

        On success emits ``finished`` with a populated result dict.
        If the protein is not found in DisProt, emits ``finished`` with
        ``{"found": False}``.
        On network / parse errors emits ``error``.
        """
        if not self.accession:
            self.error.emit("DisProt query failed: no accession provided.")
            return

        url = f"{_DISPROT_API_BASE}/{self.accession}"

        try:
            req = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "BEER-biophysics/1.0 (scientific software)",
                },
            )
            with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as response:
                raw = response.read().decode("utf-8")

        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                # Protein not in DisProt — not an error, just not found
                self.finished.emit({"found": False, "accession": self.accession})
                return
            self.error.emit(
                f"DisProt HTTP error {exc.code} for '{self.accession}': {exc.reason}"
            )
            return
        except urllib.error.URLError as exc:
            self.error.emit(
                f"DisProt network error for '{self.accession}': {exc.reason}"
            )
            return
        except Exception as exc:
            self.error.emit(f"DisProt query failed for '{self.accession}': {exc}")
            return

        # Parse JSON
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            self.error.emit(
                f"DisProt returned invalid JSON for '{self.accession}': {exc}"
            )
            return

        if not data:
            self.finished.emit({"found": False, "accession": self.accession})
            return

        # Extract fields — DisProt API may use slightly different key names
        disprot_id = data.get("disprot_id", data.get("id", ""))
        protein_name = data.get("protein_name", data.get("name", ""))
        sequence_length = int(data.get("length", data.get("sequence_length", 0)))

        # Parse disorder regions
        raw_regions = data.get("regions", [])
        regions: list = []
        for reg in raw_regions:
            if not isinstance(reg, dict):
                continue
            start = _safe_int(reg.get("start", reg.get("Start", 0)))
            end = _safe_int(reg.get("end", reg.get("End", 0)))
            rtype = (
                reg.get("type_name", reg.get("type", reg.get("term", "disorder")))
            )
            if isinstance(rtype, dict):
                rtype = rtype.get("name", rtype.get("term", "disorder"))
            regions.append({"start": start, "end": end, "type": str(rtype)})

        # Count disordered residues using a coverage set to handle overlaps
        disordered_positions: set = set()
        for reg in regions:
            for pos in range(reg["start"], reg["end"] + 1):
                disordered_positions.add(pos)

        n_disordered_aa = len(disordered_positions)
        fraction_disordered = (
            n_disordered_aa / sequence_length if sequence_length > 0 else 0.0
        )

        result = {
            "found": True,
            "disprot_id": disprot_id,
            "protein_name": protein_name,
            "accession": self.accession,
            "sequence_length": sequence_length,
            "regions": regions,
            "n_disordered_aa": n_disordered_aa,
            "fraction_disordered": fraction_disordered,
        }

        self.finished.emit(result)



# _safe_int helper (shared by network workers)
def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

# --- beer.network.phasepdb ---

_PHASEPDB_BASE = "https://phasepdb.org/api/protein"
_TIMEOUT_SECONDS = 10


class PhaSepDBWorker(QThread):
    """Query PhaSepDB for phase separation data.

    Uses the PhaSepDB REST API.  If the queried accession is not in the
    database, emits ``finished({"found": False})`` rather than an error.

    Signals
    -------
    finished(dict):
        Emitted when the query completes (success or not-found).
        Keys on success:
        - found (bool)            True
        - source (str)            "PhaSepDB"
        - accession (str)         Echo of queried accession
        - gene_name (str)
        - protein_name (str)
        - category (str)          e.g. "scaffold", "client", "regulator"
        - evidence_type (str)     e.g. "in vitro", "in vivo", "prediction"
        - organism (str)
        - references (list)       List of dicts: {pmid, title}
        Keys when not found:
        - found (bool)            False
        - accession (str)
    error(str):
        Emitted on genuine network / parse failures.
    """

    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, accession: str):
        """Initialise the worker.

        Parameters
        ----------
        accession:
            UniProt accession (e.g. ``"P04637"``).  Stripped and upper-cased.
        """
        super().__init__()
        self.accession = accession.strip().upper()

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Try PhaSepDB and emit results.

        Performs a single GET request to PhaSepDB.  Emits ``finished`` with
        ``found=True`` on success, ``found=False`` on HTTP 404 / empty
        response, or ``error`` on genuine failures (network errors, malformed
        JSON, etc.).
        """
        if not self.accession:
            self.error.emit("PhaSepDB query failed: no accession provided.")
            return

        url = f"{_PHASEPDB_BASE}/{self.accession}"

        try:
            req = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "BEER-biophysics/1.0 (scientific software)",
                },
            )
            with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as response:
                raw = response.read().decode("utf-8")

        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                # Protein not in PhaSepDB — this is a valid outcome
                self.finished.emit({"found": False, "accession": self.accession})
                return
            self.error.emit(
                f"PhaSepDB HTTP error {exc.code} for '{self.accession}': {exc.reason}"
            )
            return
        except urllib.error.URLError as exc:
            self.error.emit(
                f"PhaSepDB network error for '{self.accession}': {exc.reason}"
            )
            return
        except Exception as exc:
            self.error.emit(f"PhaSepDB query failed for '{self.accession}': {exc}")
            return

        # Parse JSON
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            self.error.emit(
                f"PhaSepDB returned invalid JSON for '{self.accession}': {exc}"
            )
            return

        # Handle empty or missing data
        if not data:
            self.finished.emit({"found": False, "accession": self.accession})
            return

        # PhaSepDB may return a list (multiple entries) or a single object
        if isinstance(data, list):
            if len(data) == 0:
                self.finished.emit({"found": False, "accession": self.accession})
                return
            # Take the first match (most relevant)
            entry = data[0]
        elif isinstance(data, dict):
            entry = data
        else:
            self.finished.emit({"found": False, "accession": self.accession})
            return

        # Extract fields robustly
        references = entry.get("references", entry.get("refs", []))
        if not isinstance(references, list):
            references = []

        parsed_refs: list = []
        for ref in references:
            if isinstance(ref, dict):
                parsed_refs.append(
                    {
                        "pmid": str(ref.get("pmid", ref.get("PubMed", ""))),
                        "title": ref.get("title", ref.get("Title", "")),
                    }
                )
            elif isinstance(ref, str):
                parsed_refs.append({"pmid": ref, "title": ""})

        result = {
            "found": True,
            "source": "PhaSepDB",
            "accession": self.accession,
            "gene_name": entry.get("gene_name", entry.get("gene", "")),
            "protein_name": entry.get(
                "protein_name", entry.get("name", entry.get("protein", ""))
            ),
            "category": entry.get("category", entry.get("Category", "")),
            "evidence_type": entry.get(
                "evidence_type", entry.get("evidence", entry.get("Evidence", ""))
            ),
            "organism": entry.get("organism", entry.get("Organism", "")),
            "references": parsed_refs,
        }

        self.finished.emit(result)


# --- beer.io.pdb: extract_phi_psi ---
def extract_phi_psi(pdb_str: str) -> list:
    """Extract φ/ψ dihedral angles for each residue from a PDB string.

    Returns list of dicts: {phi, psi, resname, resnum, chain_id, ss}
    ss: 'H' (helix) or 'E' (sheet) or 'C' (coil) - estimated from phi/psi values.
    Uses Biopython's PPBuilder.
    """
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("x", StringIO(pdb_str))
    ppb = PPBuilder()
    result = []
    for model in struct:
        for pp in ppb.build_peptides(model):
            phi_psi = pp.get_phi_psi_list()
            for i, (residue, (phi, psi)) in enumerate(zip(pp, phi_psi)):
                if phi is None or psi is None:
                    continue
                phi_deg = math.degrees(phi)
                psi_deg = math.degrees(psi)
                # Estimate secondary structure from dihedral angles
                if -160 <= phi_deg <= -45 and -60 <= psi_deg <= 50:
                    ss = 'H'
                elif phi_deg <= -90 and (psi_deg >= 90 or psi_deg <= -150):
                    ss = 'E'
                else:
                    ss = 'C'
                result.append({
                    "phi": phi_deg,
                    "psi": psi_deg,
                    "resname": residue.get_resname(),
                    "resnum": residue.get_id()[1],
                    "chain_id": residue.get_parent().get_id(),
                    "ss": ss,
                })
        break  # first model only
    return result

_extract_phi_psi = extract_phi_psi  # alias used by beer.py


# --- Constants ---

VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
KYTE_DOOLITTLE = {
    'A': 1.8,  'R': -4.5, 'N': -3.5, 'D': -3.5,
    'C': 2.5,  'Q': -3.5, 'E': -3.5, 'G': -0.4,
    'H': -3.2, 'I': 4.5,  'L': 3.8,  'K': -3.9,
    'M': 1.9,  'F': 2.8,  'P': -1.6, 'S': -0.8,
    'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

DEFAULT_PKA = {
    'NTERM': 9.69, 'CTERM': 2.34,
    'D': 3.90, 'E': 4.07, 'C': 8.18, 'Y': 10.46,
    'H': 6.04, 'K': 10.54, 'R': 12.48
}

# Named colours for the Settings colour picker (name → hex)
NAMED_COLORS = {
    "Royal Blue":   "#4361ee",
    "Magenta":      "#f72585",
    "Emerald":      "#43aa8b",
    "Purple":       "#7209b7",
    "Tangerine":    "#f3722c",
    "Steel Blue":   "#277da1",
    "Cyan Green":   "#06d6a0",
    "Slate":        "#2d3748",
    "Coral":        "#ff6b6b",
    "Gold":         "#f59e0b",
    "Teal":         "#0d9488",
    "Crimson":      "#dc2626",
    "Indigo":       "#4f46e5",
    "Sky Blue":     "#0ea5e9",
    "Forest Green": "#16a34a",
    "Olive":        "#84cc16",
    "Rose":         "#f43f5e",
    "Violet":       "#8b5cf6",
    "Navy":         "#1e40af",
    "Ochre":        "#ca8a04",
    "Charcoal":     "#374151",
    "Salmon":       "#fb923c",
    "Mint":         "#34d399",
    "Lavender":     "#a78bfa",
}

# Colormaps available in Settings
NAMED_COLORMAPS = [
    "coolwarm", "RdBu", "RdYlBu", "RdYlGn", "Spectral",
    "viridis", "plasma", "inferno", "magma", "cividis", "turbo",
    "PiYG", "PRGn", "BrBG", "puOr",
    "Blues", "Reds", "Greens", "Purples", "Oranges",
    "YlOrRd", "YlGnBu", "GnBu", "OrRd",
    "hot", "copper", "cool", "autumn", "winter", "spring", "summer",
    "twilight", "twilight_shifted", "hsv",
]

# Disorder/order residue classification (Uversky)
DISORDER_PROMOTING = set("AEGKPQRS")
ORDER_PROMOTING    = set("CFHILMVWY")

# Sticker residue sets for phase separation analysis
STICKER_AROMATIC      = set("FWY")
STICKER_ELECTROSTATIC = set("KRDE")
STICKER_ALL           = STICKER_AROMATIC | STICKER_ELECTROSTATIC

# Prion-like domain composition residues (PLAAC/Lancaster)
PRION_LIKE = set("NQSGY")

# LARKS (Low-complexity Aromatic-Rich Kinked Segments) residues
LARKS_AROMATIC = set("FWY")
LARKS_LC       = set("GASTNQ")  # low-complexity residues for LARKS windows

# Coiled-coil heptad periodicity (Lupas/Berger matrix — simplified mean propensity)
COILED_COIL_PROPENSITY = {
    'A': 1.29, 'R': 0.96, 'N': 0.90, 'D': 0.72, 'C': 0.77,
    'Q': 1.23, 'E': 1.44, 'G': 0.56, 'H': 0.92, 'I': 0.98,
    'L': 1.36, 'K': 1.17, 'M': 1.16, 'F': 0.73, 'P': 0.40,
    'S': 0.82, 'T': 0.89, 'W': 0.72, 'Y': 0.70, 'V': 1.09,
}

# Linear motif regex library (name, pattern, description)
LINEAR_MOTIFS = [
    ("NLS (basic)", r"[KR]{3,}|[KR]{2}.{1,3}[KR]{2,}", "Nuclear localisation signal (basic cluster)"),
    ("NES (hydrophobic)", r"L.{2,3}[LIVMF].{2,3}[LIVMF].{1}[LIVMF]", "Nuclear export signal (leucine-rich)"),
    ("PPII / PxxP", r"P.{1,2}P", "SH3-binding proline-rich motif (PxxP)"),
    ("14-3-3 (mode 1)", r"R.{2}[ST]", "14-3-3 binding: RSxS/RSxT consensus"),
    ("RGG box", r"RGG", "RGG repeat — RNA-binding / LLPS driver"),
    ("FG repeat", r"FG|GF", "Phe-Gly nucleoporin repeat"),
    ("KFERQ (autophagy)", r"[KQRE][LVIF][DEQ][EQ][RQKIVLF]", "Chaperone-mediated autophagy targeting (KFERQ-like)"),
    ("ER retention (KDEL)", r"KDEL|HDEL|RDEL", "ER retention signal"),
    ("RxxS/T (PKA)", r"R.{1,2}[ST]", "PKA consensus phosphorylation site (RxxS/T)"),
    ("SxIP (EB1)", r"[ST].IP", "Microtubule plus-end tracking via EB1"),
    ("WW domain ligand", r"PP.Y|P.{1,2}P", "WW-domain binding (PPxY / PxxP)"),
    ("Caspase-3 cleavage", r"DEVD|DMQD", "Caspase-3/7 cleavage site (DxxD)"),
    ("Glycosylation (N-linked)", r"N[^P][ST]", "N-linked glycosylation sequon (NxS/T, x≠P)"),
    ("SUMOylation", r"[VILMF]K.E", "SUMOylation consensus (ΨKxE)"),
    ("Phospho (CK2)", r"[ST].{2}[DE]", "CK2 phosphorylation consensus (S/TxxE/D)"),
]

# Chou-Fasman helix and sheet propensities (Chou & Fasman 1978)
CHOU_FASMAN_HELIX = {
    'A': 1.42, 'R': 0.98, 'N': 0.67, 'D': 1.01, 'C': 0.70,
    'Q': 1.11, 'E': 1.51, 'G': 0.57, 'H': 1.00, 'I': 1.08,
    'L': 1.21, 'K': 1.16, 'M': 1.45, 'F': 1.13, 'P': 0.57,
    'S': 0.77, 'T': 0.83, 'W': 1.08, 'Y': 0.69, 'V': 1.06,
}
CHOU_FASMAN_SHEET = {
    'A': 0.83, 'R': 0.93, 'N': 0.89, 'D': 0.54, 'C': 1.19,
    'Q': 1.10, 'E': 0.37, 'G': 0.75, 'H': 0.87, 'I': 1.60,
    'L': 1.30, 'K': 0.74, 'M': 1.05, 'F': 1.38, 'P': 0.55,
    'S': 0.75, 'T': 1.19, 'W': 1.37, 'Y': 1.47, 'V': 1.70,
}
# IUPred-inspired per-residue disorder propensity (higher = more disordered)
DISORDER_PROPENSITY = {
    'A':  0.060, 'R': -0.260, 'N':  0.007, 'D':  0.192, 'C': -0.020,
    'Q': -0.091, 'E':  0.736, 'G':  0.166, 'H': -0.303, 'I': -0.486,
    'L': -0.326, 'K':  0.586, 'M': -0.397, 'F': -0.697, 'P':  0.987,
    'S':  0.341, 'T':  0.059, 'W': -0.884, 'Y': -0.510, 'V': -0.386,
}
# Residue colours for the colour-coded sequence viewer
_AA_COLOURS = {
    # Hydrophobic (orange)
    **{aa: "#e06c00" for aa in "ACILMV"},
    # Aromatic (dark amber)
    **{aa: "#c47600" for aa in "FWY"},
    # Positive (blue)
    **{aa: "#2563eb" for aa in "KRH"},
    # Negative (red)
    **{aa: "#dc2626" for aa in "DE"},
    # Polar uncharged (teal)
    **{aa: "#0d9488" for aa in "NQST"},
    # Special (purple)
    **{aa: "#7c3aed" for aa in "GP"},
}

REPORT_SECTIONS = [
    "Composition",
    "Properties",
    "Hydrophobicity",
    "Charge",
    "Aromatic & \u03c0",
    "Low Complexity",
    "Disorder",
    "Secondary Structure",
    "Repeat Motifs",
    "Sticker & Spacer",
    "TM Helices",
    "Phase Separation",
    "Linear Motifs",
    # --- New sections ---
    "\u03b2-Aggregation & Solubility",
    "PTM Sites",
    "Signal Peptide & GPI",
    "Amphipathic Helices",
    "Charge Decoration (SCD)",
    "RNA Binding",
    "Tandem Repeats",
]

GRAPH_TITLES = [
    "Amino Acid Composition (Bar)",
    "Amino Acid Composition (Pie)",
    "Hydrophobicity Profile",
    "Net Charge vs pH",
    "Bead Model (Hydrophobicity)",
    "Bead Model (Charge)",
    "Sticker Map",
    "Local Charge Profile",
    "Local Complexity",
    "Cation\u2013\u03c0 Map",
    "Isoelectric Focus",
    "Secondary Structure",
    "Helical Wheel",
    "Charge Decoration",
    "Linear Sequence Map",
    "Disorder Profile",
    "TM Topology",
    "pLDDT Profile",
    "Distance Map",
    "Domain Architecture",
    "Uversky Phase Plot",
    "Coiled-Coil Profile",
    "Saturation Mutagenesis",
    # --- New graphs ---
    "\u03b2-Aggregation Profile",
    "Solubility Profile",
    "Hydrophobic Moment",
    "PTM Map",
    "RNA-Binding Profile",
    "SCD Profile",
    "pI / MW Map",
    "Truncation Series",
    "Ramachandran Plot",
    "Residue Contact Network",
    "MSA Conservation",
    "Complex Mass",
]

# Graph categories for the tree browser (order matters; every GRAPH_TITLES entry must appear here)
GRAPH_CATEGORIES = [
    ("Composition", [
        "Amino Acid Composition (Bar)",
        "Amino Acid Composition (Pie)",
    ]),
    ("Profiles", [
        "Hydrophobicity Profile",
        "Local Charge Profile",
        "Local Complexity",
        "Disorder Profile",
        "Coiled-Coil Profile",
        "Linear Sequence Map",
        "Secondary Structure",
    ]),
    ("Charge & \u03c0-Interactions", [
        "Net Charge vs pH",
        "Isoelectric Focus",
        "Charge Decoration",
        "Cation\u2013\u03c0 Map",
    ]),
    ("Structure & Folding", [
        "Bead Model (Hydrophobicity)",
        "Bead Model (Charge)",
        "Sticker Map",
        "Helical Wheel",
        "TM Topology",
    ]),
    ("Phase Separation / IDP", [
        "Uversky Phase Plot",
        "Saturation Mutagenesis",
    ]),
    ("AlphaFold / Structural", [
        "pLDDT Profile",
        "Distance Map",
        "Domain Architecture",
        "Ramachandran Plot",
        "Residue Contact Network",
    ]),
    ("Aggregation & Solubility", [
        "\u03b2-Aggregation Profile",
        "Solubility Profile",
        "Hydrophobic Moment",
    ]),
    ("New Features", [
        "PTM Map",
        "RNA-Binding Profile",
        "SCD Profile",
        "pI / MW Map",
        "Truncation Series",
        "MSA Conservation",
        "Complex Mass",
    ]),
]

LIGHT_THEME_CSS = """
 QWidget {
     background-color: #f5f6fa;
     color: #1a1a2e;
     font-family: 'Segoe UI', 'Inter', Arial, sans-serif;
     font-size: 12px;
 }
 QMainWindow { background-color: #f5f6fa; }
 QLineEdit, QTextEdit, QTextBrowser {
     background-color: #ffffff;
     color: #1a1a2e;
     border: 1px solid #d0d4e0;
     border-radius: 4px;
     padding: 4px 6px;
     selection-background-color: #4361ee;
 }
 QPushButton {
     background-color: #4361ee;
     color: #ffffff;
     border: none;
     border-radius: 5px;
     padding: 6px 14px;
     font-weight: 600;
     letter-spacing: 0.3px;
 }
 QPushButton:hover { background-color: #3451d1; }
 QPushButton:pressed { background-color: #2940b8; }
 QPushButton:disabled { background-color: #b0b8cc; color: #f0f0f0; }
 QTabWidget::pane { border: 1px solid #d0d4e0; border-radius: 4px; background: #ffffff; }
 QTabBar::tab {
     background: #e8eaf0;
     color: #4a5568;
     padding: 8px 16px;
     border-top-left-radius: 5px;
     border-top-right-radius: 5px;
     margin-right: 2px;
     font-weight: 500;
 }
 QTabBar::tab:selected { background: #4361ee; color: #ffffff; }
 QTabBar::tab:hover:!selected { background: #d0d4e8; }
 QTableWidget {
     background-color: #ffffff;
     gridline-color: #e8eaf0;
     border: 1px solid #d0d4e0;
     border-radius: 4px;
     alternate-background-color: #f8f9fd;
 }
 QHeaderView::section {
     background-color: #4361ee;
     color: #ffffff;
     padding: 6px 10px;
     border: none;
     font-weight: 600;
 }
 QComboBox {
     background-color: #ffffff;
     border: 1px solid #d0d4e0;
     border-radius: 4px;
     padding: 4px 8px;
 }
 QComboBox::drop-down { border: none; }
 QLabel { color: #2d3748; font-weight: 500; }
 QCheckBox { color: #2d3748; spacing: 6px; }
 QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #d0d4e0; border-radius: 3px; }
 QCheckBox::indicator:checked { background-color: #4361ee; border-color: #4361ee; }
 QScrollBar:vertical { background: #f0f0f5; width: 10px; border-radius: 5px; }
 QScrollBar::handle:vertical { background: #c0c4d0; border-radius: 5px; min-height: 30px; }
 QStatusBar { background-color: #4361ee; color: #ffffff; font-size: 11px; }
 /* --- Left navigation sidebar --- */
 QListWidget#nav_bar {
     background-color: #e4e8f4;
     border: none;
     border-right: 1px solid #c8cede;
     padding: 8px 0;
     font-size: 11px;
     font-weight: 500;
     outline: 0;
 }
 QListWidget#nav_bar::item {
     padding: 11px 10px;
     color: #4a5568;
     border-left: 3px solid transparent;
 }
 QListWidget#nav_bar::item:selected {
     background-color: #dce3f8;
     color: #4361ee;
     border-left: 3px solid #4361ee;
     font-weight: 700;
 }
 QListWidget#nav_bar::item:hover:!selected { background-color: #d4d9ec; }
 QFrame#nav_sep { color: #c8cede; max-width: 1px; }
 /* --- Graph tree & report nav --- */
 QTreeWidget#graph_tree, QListWidget#report_nav {
     background-color: #f0f2fa;
     border: none;
     border-right: 1px solid #d0d4e0;
     font-size: 11px;
     outline: 0;
 }
 QTreeWidget#graph_tree::item { padding: 5px 6px; color: #4a5568; }
 QTreeWidget#graph_tree::item:selected { background-color: #4361ee; color: #ffffff; border-radius: 3px; }
 QTreeWidget#graph_tree::branch { background-color: #f0f2fa; }
 QListWidget#report_nav::item { padding: 8px 10px; color: #4a5568; }
 QListWidget#report_nav::item:selected { background-color: #4361ee; color: #ffffff; }
 QListWidget#report_nav::item:hover:!selected { background-color: #dce3f8; }
"""

DARK_THEME_CSS = """
 QWidget {
     background-color: #1a1a2e;
     color: #e2e8f0;
     font-family: 'Segoe UI', 'Inter', Arial, sans-serif;
     font-size: 12px;
 }
 QMainWindow { background-color: #1a1a2e; }
 QLineEdit, QTextEdit, QTextBrowser {
     background-color: #16213e;
     color: #e2e8f0;
     border: 1px solid #2d3561;
     border-radius: 4px;
     padding: 4px 6px;
     selection-background-color: #4cc9f0;
 }
 QPushButton {
     background-color: #4cc9f0;
     color: #1a1a2e;
     border: none;
     border-radius: 5px;
     padding: 6px 14px;
     font-weight: 600;
     letter-spacing: 0.3px;
 }
 QPushButton:hover { background-color: #3ab7dd; }
 QPushButton:pressed { background-color: #28a4c9; }
 QPushButton:disabled { background-color: #2d3561; color: #6b7280; }
 QTabWidget::pane { border: 1px solid #2d3561; border-radius: 4px; background: #16213e; }
 QTabBar::tab {
     background: #0f3460;
     color: #94a3b8;
     padding: 8px 16px;
     border-top-left-radius: 5px;
     border-top-right-radius: 5px;
     margin-right: 2px;
     font-weight: 500;
 }
 QTabBar::tab:selected { background: #4cc9f0; color: #1a1a2e; }
 QTabBar::tab:hover:!selected { background: #1a3a5c; }
 QTableWidget {
     background-color: #16213e;
     gridline-color: #2d3561;
     border: 1px solid #2d3561;
     border-radius: 4px;
     alternate-background-color: #1e2a4a;
 }
 QHeaderView::section {
     background-color: #0f3460;
     color: #4cc9f0;
     padding: 6px 10px;
     border: none;
     font-weight: 600;
 }
 QComboBox {
     background-color: #16213e;
     color: #e2e8f0;
     border: 1px solid #2d3561;
     border-radius: 4px;
     padding: 4px 8px;
 }
 QComboBox::drop-down { border: none; }
 QLabel { color: #94a3b8; font-weight: 500; }
 QCheckBox { color: #94a3b8; spacing: 6px; }
 QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #2d3561; border-radius: 3px; }
 QCheckBox::indicator:checked { background-color: #4cc9f0; border-color: #4cc9f0; }
 QScrollBar:vertical { background: #16213e; width: 10px; border-radius: 5px; }
 QScrollBar::handle:vertical { background: #2d3561; border-radius: 5px; min-height: 30px; }
 QStatusBar { background-color: #0f3460; color: #4cc9f0; font-size: 11px; }
 /* --- Left navigation sidebar --- */
 QListWidget#nav_bar {
     background-color: #0f3460;
     border: none;
     border-right: 1px solid #1a3a5c;
     padding: 8px 0;
     font-size: 11px;
     font-weight: 500;
     outline: 0;
 }
 QListWidget#nav_bar::item {
     padding: 11px 10px;
     color: #94a3b8;
     border-left: 3px solid transparent;
 }
 QListWidget#nav_bar::item:selected {
     background-color: #1a3a5c;
     color: #4cc9f0;
     border-left: 3px solid #4cc9f0;
     font-weight: 700;
 }
 QListWidget#nav_bar::item:hover:!selected { background-color: #1a3a5c; color: #e2e8f0; }
 QFrame#nav_sep { color: #1a3a5c; max-width: 1px; }
 /* --- Graph tree & report nav --- */
 QTreeWidget#graph_tree, QListWidget#report_nav {
     background-color: #16213e;
     border: none;
     border-right: 1px solid #2d3561;
     font-size: 11px;
     outline: 0;
 }
 QTreeWidget#graph_tree::item { padding: 5px 6px; color: #94a3b8; }
 QTreeWidget#graph_tree::item:selected { background-color: #4cc9f0; color: #1a1a2e; border-radius: 3px; }
 QTreeWidget#graph_tree::branch { background-color: #16213e; }
 QListWidget#report_nav::item { padding: 8px 10px; color: #94a3b8; }
 QListWidget#report_nav::item:selected { background-color: #4cc9f0; color: #1a1a2e; }
 QListWidget#report_nav::item:hover:!selected { background-color: #1a3a5c; }
"""

# --- HTML/PDF styling ---
REPORT_CSS = """
body {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 11pt;
    color: #1a1a2e;
    margin: 0;
    padding: 0;
    line-height: 1.6;
}
h1 { font-size: 18pt; color: #1a1a2e; border-bottom: 2px solid #4361ee; padding-bottom: 6px; margin-top: 20px; }
h2 { font-size: 13pt; color: #4361ee; margin-top: 18px; margin-bottom: 8px; font-weight: 600; }
table {
    border-collapse: collapse;
    width: 100%;
    margin: 10px 0 16px 0;
    font-size: 10pt;
}
th {
    background-color: #4361ee;
    color: #ffffff;
    padding: 7px 12px;
    text-align: left;
    font-weight: 600;
}
td {
    padding: 6px 12px;
    border-bottom: 1px solid #e8eaf0;
    color: #2d3748;
}
tr:nth-child(even) td { background-color: #f8f9fd; }
tr:hover td { background-color: #eef0f8; }
p.note {
    font-size: 9pt;
    color: #718096;
    font-style: italic;
    margin: 4px 0 12px 0;
}
pre.sequence {
    font-family: 'Courier New', Courier, monospace;
    font-size: 10pt;
    background: #f8f9fd;
    border: 1px solid #e8eaf0;
    border-radius: 4px;
    padding: 10px 14px;
    line-height: 1.8;
    color: #1a1a2e;
    white-space: pre;
}
"""

# --- Helpers ---

def clean_sequence(seq: str) -> str:
    return seq.strip().replace(" ", "").upper()

def is_valid_protein(seq: str) -> bool:
    return all(aa in VALID_AMINO_ACIDS for aa in seq)

def calc_net_charge(seq: str, pH: float = 7.0, pka: dict = None) -> float:
    """Henderson-Hasselbalch net charge."""
    p = pka or DEFAULT_PKA
    net = 1/(1+10**(pH - p['NTERM'])) - 1/(1+10**(p['CTERM'] - pH))
    for aa in seq:
        if aa in ('D', 'E', 'C', 'Y'):
            net -= 1/(1+10**(p[aa] - pH))
        elif aa in ('K', 'R', 'H'):
            net += 1/(1+10**(pH - p[aa]))
    return net

def sliding_window_hydrophobicity(seq: str, window_size: int = 9) -> list:
    """Kyte-Doolittle sliding window average."""
    if window_size > len(seq):
        return [sum(KYTE_DOOLITTLE[aa] for aa in seq) / len(seq)]
    return [
        sum(KYTE_DOOLITTLE[aa] for aa in seq[i:i+window_size]) / window_size
        for i in range(len(seq) - window_size + 1)
    ]

def calc_shannon_entropy(seq: str) -> float:
    """Sequence compositional entropy in bits. Max = log2(20) ≈ 4.32."""
    n = len(seq)
    counts = {}
    for aa in seq:
        counts[aa] = counts.get(aa, 0) + 1
    return -sum((c/n) * math.log2(c/n) for c in counts.values())

def sliding_window_ncpr(seq: str, window_size: int = 9) -> list:
    """Net charge per residue in a sliding window (K,R positive; D,E negative)."""
    pos = set("KR")
    neg = set("DE")
    if window_size > len(seq):
        p = sum(1 for aa in seq if aa in pos)
        n = sum(1 for aa in seq if aa in neg)
        return [(p - n) / len(seq)]
    return [
        (sum(1 for aa in seq[i:i+window_size] if aa in pos) -
         sum(1 for aa in seq[i:i+window_size] if aa in neg)) / window_size
        for i in range(len(seq) - window_size + 1)
    ]

def sliding_window_entropy(seq: str, window_size: int = 9) -> list:
    """Shannon entropy in a sliding window."""
    if window_size > len(seq):
        return [calc_shannon_entropy(seq)]
    return [
        calc_shannon_entropy(seq[i:i+window_size])
        for i in range(len(seq) - window_size + 1)
    ]

def calc_kappa(seq: str) -> float:
    """Charge patterning parameter (Das & Pappu 2013). Range [0, 1].
    0 = well-mixed charges, 1 = fully segregated."""
    pos_aa   = set("KR")
    neg_aa   = set("DE")
    blob_sz  = 5
    pos_n    = sum(1 for aa in seq if aa in pos_aa)
    neg_n    = sum(1 for aa in seq if aa in neg_aa)
    if pos_n == 0 or neg_n == 0:
        return 0.0
    n_blobs = len(seq) // blob_sz
    if n_blobs < 2:
        return 0.0
    fcr_pos = pos_n / len(seq)
    fcr_neg = neg_n / len(seq)

    def _delta(s):
        nb = len(s) // blob_sz
        if nb == 0:
            return 0.0
        total = 0.0
        for i in range(nb):
            bl = s[i*blob_sz:(i+1)*blob_sz]
            fp = sum(1 for a in bl if a in pos_aa) / len(bl)
            fn = sum(1 for a in bl if a in neg_aa) / len(bl)
            total += (fp - fcr_pos)**2 + (fn - fcr_neg)**2
        return total / nb

    delta     = _delta(seq)
    neutral_n = len(seq) - pos_n - neg_n
    seg1      = 'K'*pos_n + 'D'*neg_n + 'G'*neutral_n
    seg2      = 'D'*neg_n + 'K'*pos_n + 'G'*neutral_n
    delta_max = max(_delta(seg1), _delta(seg2))
    return 0.0 if delta_max == 0 else min(1.0, delta / delta_max)

def calc_omega(seq: str) -> float:
    """Patterning of sticker residues (FWYKRDE) vs spacers (Das et al. 2015).
    Range [0, 1]. 0 = evenly distributed, 1 = fully clustered."""
    blob_sz     = 5
    sticker_n   = sum(1 for aa in seq if aa in STICKER_ALL)
    if sticker_n == 0 or sticker_n == len(seq):
        return 0.0
    n_blobs = len(seq) // blob_sz
    if n_blobs < 2:
        return 0.0
    f_stick = sticker_n / len(seq)

    def _delta(s):
        nb = len(s) // blob_sz
        if nb == 0:
            return 0.0
        total = 0.0
        for i in range(nb):
            bl  = s[i*blob_sz:(i+1)*blob_sz]
            fs  = sum(1 for a in bl if a in STICKER_ALL) / len(bl)
            total += (fs - f_stick)**2
        return total / nb

    delta     = _delta(seq)
    spacer_n  = len(seq) - sticker_n
    seg1      = 'F'*sticker_n + 'G'*spacer_n
    seg2      = 'G'*spacer_n  + 'F'*sticker_n
    delta_max = max(_delta(seg1), _delta(seg2))
    return 0.0 if delta_max == 0 else min(1.0, delta / delta_max)

def count_pairs(seq: str, set_a: set, set_b: set, window: int = 4) -> int:
    """Count unique (i,j) residue pairs where i in set_a, j in set_b, |i-j| <= window."""
    n      = len(seq)
    pairs  = set()
    for i in range(n):
        if seq[i] in set_a:
            for j in range(max(0, i-window), min(n, i+window+1)):
                if j != i and seq[j] in set_b:
                    pairs.add((min(i, j), max(i, j)))
    return len(pairs)

def fraction_low_complexity(seq: str, window_size: int = 12,
                            threshold: float = 2.0) -> float:
    """Fraction of residues covered by at least one window with entropy < threshold."""
    if len(seq) < window_size:
        return 1.0 if calc_shannon_entropy(seq) < threshold else 0.0
    covered = [False] * len(seq)
    for i in range(len(seq) - window_size + 1):
        if calc_shannon_entropy(seq[i:i+window_size]) < threshold:
            for j in range(i, i+window_size):
                covered[j] = True
    return sum(covered) / len(seq)

def sticker_spacing_stats(seq: str) -> dict:
    """Return mean/min/max residue spacing between consecutive sticker residues."""
    positions = [i for i, aa in enumerate(seq) if aa in STICKER_ALL]
    if len(positions) < 2:
        return {"mean": None, "min": None, "max": None}
    gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
    return {
        "mean": sum(gaps) / len(gaps),
        "min":  min(gaps),
        "max":  max(gaps),
    }

# --- Sequence display helper ---

def format_sequence_block(seq: str, name: str = "", width: int = 60, group: int = 10) -> str:
    """Format sequence in UniProt style: groups of 10 residues, position numbers every 10."""
    lines = []
    if name:
        lines.append(f">{name}")
    for i in range(0, len(seq), width):
        chunk = seq[i:i+width]
        groups = "  ".join(chunk[j:j+group] for j in range(0, len(chunk), group))
        pos = str(i + 1).rjust(6)
        lines.append(f"{pos}  {groups}")
    return "\n".join(lines)


def _pub_style_ax(ax, title="", xlabel="", ylabel="", grid=True, despine=True,
                  title_size=13, label_size=11, tick_size=10):
    """Apply publication-quality styling to a matplotlib Axes object."""
    if title:
        ax.set_title(title, fontsize=title_size, fontweight="bold", pad=10, color="#1a1a2e")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=label_size, labelpad=6, color="#2d3748")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=label_size, labelpad=6, color="#2d3748")
    ax.tick_params(labelsize=tick_size, length=4, width=0.8, colors="#4a5568")
    if despine:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.spines["left"].set_color("#c0c4d0")
        ax.spines["bottom"].set_color("#c0c4d0")
    if grid:
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, color="#c8cdd8")
        ax.set_axisbelow(True)
    ax.set_facecolor("#fafbff")


def calc_chou_fasman_profile(seq: str) -> tuple:
    """Return (helix_list, sheet_list) of per-residue Chou-Fasman propensities."""
    helix = [CHOU_FASMAN_HELIX.get(aa, 1.0) for aa in seq]
    sheet = [CHOU_FASMAN_SHEET.get(aa, 1.0) for aa in seq]
    return helix, sheet


def calc_disorder_profile(seq: str, window: int = 9) -> list:
    """Sliding-window disorder propensity, normalised to 0-1."""
    raw = [DISORDER_PROPENSITY.get(aa, 0.0) for aa in seq]
    mn, mx = min(raw), max(raw)
    span = mx - mn if mx != mn else 1.0
    norm = [(v - mn) / span for v in raw]
    n = len(norm)
    if n < window:
        return norm
    half = window // 2
    smoothed = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        smoothed.append(sum(norm[lo:hi]) / (hi - lo))
    return smoothed


# --- Transmembrane / structural helpers ---

def predict_tm_helices(seq: str, window: int = 19, threshold: float = 1.6,
                       min_len: int = 17, max_len: int = 25) -> list:
    """Predict TM helices via Kyte-Doolittle sliding window.

    Algorithm:
    1. Compute per-window KD average (only full-length windows; no partial edges).
    2. Mark every residue covered by at least one above-threshold window as TM.
    3. Collect contiguous marked regions; keep those within [min_len, max_len].
       Overlong regions are split by finding the single highest-scoring window.
    4. Assign topology using the inside-positive rule (von Heijne):
       the side flanking with more K/R is cytoplasmic.

    Returns list of dicts: {start (0-based), end (0-based inclusive), score, orientation}.
    """
    n = len(seq)
    if n < window:
        return []

    # Step 1 — per-window average KD (strictly full windows only)
    win_scores = [
        sum(KYTE_DOOLITTLE.get(seq[j], 0.0) for j in range(i, i + window)) / window
        for i in range(n - window + 1)
    ]

    # Step 2 — per-residue TM mask: residue r is TM if any window covering it scores ≥ threshold
    tm_mask = [False] * n
    for i, score in enumerate(win_scores):
        if score >= threshold:
            for r in range(i, i + window):
                tm_mask[r] = True

    # Step 3 — collect contiguous TM regions
    helices = []
    i = 0
    while i < n:
        if tm_mask[i]:
            j = i
            while j < n and tm_mask[j]:
                j += 1
            span = j - i
            if min_len <= span <= max_len:
                seg_score = sum(KYTE_DOOLITTLE.get(seq[r], 0.0) for r in range(i, j)) / span
                helices.append({"start": i, "end": j - 1, "score": round(seg_score, 3)})
            elif span > max_len:
                # Too long — pick the best single window inside the region
                best_i = max(range(i, j - window + 1),
                             key=lambda k: win_scores[k] if k < len(win_scores) else -999)
                best_s = win_scores[best_i] if best_i < len(win_scores) else 0.0
                helices.append({"start": best_i, "end": best_i + window - 1,
                                 "score": round(best_s, 3)})
            i = j
        else:
            i += 1

    # Step 4 — inside-positive rule (von Heijne)
    pos   = set("KR")
    flank = 15
    for h in helices:
        s, e  = h["start"], h["end"]
        n_pos = sum(1 for aa in seq[max(0, s - flank):s] if aa in pos)
        c_pos = sum(1 for aa in seq[e + 1:min(n, e + 1 + flank)] if aa in pos)
        # More K/R on C-terminal side → C-term is cytoplasmic → N-term is extracellular → out→in
        h["orientation"] = "out\u2192in" if c_pos >= n_pos else "in\u2192out"
    return helices


def compute_ca_distance_matrix(pdb_str: str) -> np.ndarray:
    """Return symmetric Cα pairwise distance matrix (Å) from a PDB string.

    Only the first chain of the first model is used so that the matrix length
    matches the single-chain pLDDT array returned by extract_plddt_from_pdb().
    """
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("af", StringIO(pdb_str))
    coords = []
    for model in struct:
        first_chain = next(iter(model), None)
        if first_chain is None:
            break
        for res in first_chain:
            if is_aa(res, standard=True) and res.has_id("CA"):
                coords.append(res["CA"].get_vector().get_array())
        break
    if not coords:
        return np.array([])
    ca   = np.array(coords, dtype=float)
    diff = ca[:, np.newaxis, :] - ca[np.newaxis, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def extract_plddt_from_pdb(pdb_str: str) -> list:
    """Extract per-residue pLDDT / B-factor confidence scores from a PDB string.

    Only the first chain of the first model is read.  This matches the single-chain
    output produced by AlphaFold and avoids silently concatenating scores from
    multiple chains when a multi-chain PDB is passed.
    """
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("af", StringIO(pdb_str))
    scores = []
    for model in struct:
        first_chain = next(iter(model), None)
        if first_chain is None:
            break
        for res in first_chain:
            if is_aa(res, standard=True):
                for atom in res:
                    if atom.get_name() == "CA":
                        scores.append(atom.get_bfactor())
                        break
        break
    return scores


def detect_larks(seq: str, window: int = 7, min_arom: int = 1,
                 min_lc_frac: float = 0.50, max_entropy: float = 1.8) -> list:
    """Detect LARKS (Low-complexity Aromatic-Rich Kinked Segments).

    A LARKS is a short window (6–8 residues) that:
      • Contains ≥ 1 aromatic residue (F/W/Y)
      • Has ≥ 50 % low-complexity residues (G/A/S/T/N/Q)
      • Has Shannon entropy < 1.8 bits

    Returns list of dicts: {start, end, seq, n_arom, lc_frac, entropy}
    """
    n = len(seq)
    hits = []
    seen = set()
    for i in range(n - window + 1):
        w = seq[i:i + window]
        n_arom = sum(1 for aa in w if aa in LARKS_AROMATIC)
        if n_arom < min_arom:
            continue
        lc_frac = sum(1 for aa in w if aa in LARKS_LC) / window
        if lc_frac < min_lc_frac:
            continue
        # Shannon entropy of window
        cnt = Counter(w)
        H = -sum((v / window) * math.log2(v / window) for v in cnt.values())
        if H >= max_entropy:
            continue
        # Merge overlapping hits
        span = (i, i + window - 1)
        overlap = False
        for prev in hits:
            if prev["start"] <= span[1] and span[0] <= prev["end"]:
                overlap = True
                break
        if not overlap:
            hits.append({"start": i, "end": i + window - 1,
                         "seq": w, "n_arom": n_arom,
                         "lc_frac": round(lc_frac, 3),
                         "entropy": round(H, 3)})
    return hits


def calc_llps_score(seq: str, fcr: float, ncpr: float, arom_f: float,
                    prion_score: float, omega: float, disorder_f: float,
                    larks: list) -> dict:
    """Compute a composite LLPS propensity score (0–1).

    Weighted combination of sequence features associated with
    liquid-liquid phase separation:
      w1=0.25  Aromatic fraction (pi–pi driving force)
      w2=0.20  Prion-like score (compositional complexity)
      w3=0.15  Disorder-promoting fraction
      w4=0.15  FCR (charge fraction — enables electrostatic valency)
      w5=0.15  Omega (sticker clustering)
      w6=0.10  LARKS density (per-100-aa)
      penalty  0.10 * |NCPR| (high charge asymmetry suppresses LLPS)
    """
    n = len(seq)
    if n == 0:
        return {"score": 0.0, "verdict": "Low LLPS propensity", "components": {}}
    # Normalise each feature to [0,1]
    arom_norm    = min(arom_f / 0.15, 1.0)           # 15 % arom → full score
    prion_norm   = min(prion_score / 0.40, 1.0)       # 40 % prion-like → full
    disord_norm  = min(disorder_f / 0.60, 1.0)        # 60 % disorder → full
    fcr_norm     = min(fcr / 0.30, 1.0)               # 30 % charged → full
    omega_norm   = omega                               # already 0–1
    ncpr_penalty = min(abs(ncpr) / 0.30, 1.0)         # high |NCPR| → penalty
    larks_norm   = min(len(larks) / (n / 100.0) / 5.0, 1.0)  # 5 LARKS/100aa → full

    score = (
        0.25 * arom_norm
        + 0.20 * prion_norm
        + 0.15 * disord_norm
        + 0.15 * fcr_norm
        + 0.15 * omega_norm
        + 0.10 * larks_norm
        - 0.10 * ncpr_penalty   # penalise strongly asymmetric charge
    )
    score = max(0.0, min(1.0, score))

    if score >= 0.55:
        verdict = "High LLPS propensity"
    elif score >= 0.30:
        verdict = "Moderate LLPS propensity"
    else:
        verdict = "Low LLPS propensity"

    return {
        "score": round(score, 3),
        "verdict": verdict,
        "components": {
            "Aromatic fraction": round(arom_norm, 3),
            "Prion-like score": round(prion_norm, 3),
            "Disorder fraction": round(disord_norm, 3),
            "FCR": round(fcr_norm, 3),
            "Omega (sticker clustering)": round(omega_norm, 3),
            "LARKS density": round(larks_norm, 3),
            "|NCPR| penalty": round(ncpr_penalty, 3),
        },
    }


def predict_coiled_coil(seq: str, window: int = 28) -> list:
    """Heptad-periodicity coiled-coil scoring (Lupas-inspired).

    Uses a 28-residue (4-heptad) sliding window with position-weighted
    propensity scores for the a/d positions of the heptad (which are the
    hydrophobic core positions).

    Returns list of per-residue scores (0 = no coiled-coil propensity).
    """
    n = len(seq)
    scores = [0.0] * n
    if n < window:
        return scores

    # Heptad positions: a=0, b=1, c=2, d=3, e=4, f=5, g=6
    # Weight positions a and d (0, 3) most heavily
    pos_weights = [0.20, 0.05, 0.05, 0.20, 0.10, 0.10, 0.10,
                   0.20, 0.05, 0.05, 0.20, 0.10, 0.10, 0.10,
                   0.20, 0.05, 0.05, 0.20, 0.10, 0.10, 0.10,
                   0.20, 0.05, 0.05, 0.20, 0.10, 0.10, 0.10]

    win_scores = []
    for i in range(n - window + 1):
        w = seq[i:i + window]
        s = sum(COILED_COIL_PROPENSITY.get(aa, 1.0) * pw
                for aa, pw in zip(w, pos_weights))
        win_scores.append(s)

    # Distribute window scores to residues
    counts = [0] * n
    for i, ws in enumerate(win_scores):
        for r in range(i, i + window):
            scores[r] += ws
            counts[r] += 1
    for i in range(n):
        if counts[i] > 0:
            scores[i] = scores[i] / counts[i]

    # Normalise to 0–1 range (typical score ~0.9 for real coiled coils)
    mx = max(scores) if max(scores) > 0 else 1.0
    scores = [s / mx for s in scores]
    return scores


def scan_linear_motifs(seq: str) -> list:
    """Scan sequence against the built-in LINEAR_MOTIFS library.

    Returns list of dicts: {name, description, start, end, match}
    """
    hits = []
    for name, pattern, description in LINEAR_MOTIFS:
        for m in re.finditer(pattern, seq):
            hits.append({
                "name": name,
                "description": description,
                "start": m.start(),
                "end": m.end() - 1,
                "match": m.group(),
            })
    hits.sort(key=lambda h: h["start"])
    return hits


# --- Analysis ---

class AnalysisTools:
    @staticmethod
    def analyze_sequence(seq: str, pH_value: float = 7.0, window_size: int = 9,
                         use_reducing: bool = False, pka: dict = None) -> dict:
        pa          = BPProteinAnalysis(seq)
        aa_counts   = pa.count_amino_acids()
        seq_length  = len(seq)
        aa_freq     = {aa: count/seq_length*100 for aa, count in aa_counts.items()}
        mol_weight  = pa.molecular_weight()
        iso_point   = pa.isoelectric_point()
        gravy       = pa.gravy()
        instability = pa.instability_index()
        aromaticity = pa.aromaticity()
        net_charge_7  = calc_net_charge(seq, 7.0, pka)
        net_charge_pH = calc_net_charge(seq, pH_value, pka)

        n_cystine  = 0 if use_reducing else seq.count("C") // 2
        extinction = 5500*seq.count("W") + 1490*seq.count("Y") + 125*n_cystine

        # --- Charge features ---
        pos_n   = sum(aa_counts.get(k, 0) for k in "KR")
        neg_n   = sum(aa_counts.get(k, 0) for k in "DE")
        fcr     = (pos_n + neg_n) / seq_length
        ncpr    = (pos_n - neg_n) / seq_length
        kappa   = calc_kappa(seq)
        ch_asym = (pos_n / neg_n) if neg_n > 0 else float('inf')

        kappa_interp = (
            "well-mixed"        if kappa < 0.2 else
            "moderately patterned" if kappa < 0.5 else
            "strongly segregated"
        )

        # --- Aromatic & π features ---
        n_tyr  = aa_counts.get("Y", 0)
        n_phe  = aa_counts.get("F", 0)
        n_trp  = aa_counts.get("W", 0)
        arom_n = n_tyr + n_phe + n_trp
        arom_f = arom_n / seq_length
        cation_pi_n = count_pairs(seq, set("KR"), STICKER_AROMATIC, window=4)
        pi_pi_n     = count_pairs(seq, STICKER_AROMATIC, STICKER_AROMATIC, window=4)

        # --- Low complexity features ---
        entropy      = calc_shannon_entropy(seq)
        entropy_norm = entropy / math.log2(20)
        unique_aa    = sum(1 for v in aa_counts.values() if v > 0)
        prion_score  = sum(aa_counts.get(k, 0) for k in PRION_LIKE) / seq_length
        lc_frac      = fraction_low_complexity(seq, window_size=12, threshold=2.0)

        # --- Disorder features ---
        disorder_f   = sum(aa_counts.get(k, 0) for k in DISORDER_PROMOTING) / seq_length
        order_f      = sum(aa_counts.get(k, 0) for k in ORDER_PROMOTING) / seq_length
        aliphatic_idx = (
            aa_counts.get("A", 0)
            + 2.9 * aa_counts.get("V", 0)
            + 3.9 * (aa_counts.get("I", 0) + aa_counts.get("L", 0))
        ) / seq_length * 100
        omega = calc_omega(seq)
        omega_interp = (
            "evenly distributed" if omega < 0.2 else
            "moderately clustered" if omega < 0.5 else
            "strongly clustered"
        )

        # --- Repeat motifs ---
        rgg_n  = seq.count("RGG")
        fg_n   = seq.count("FG")
        yg_n   = seq.count("YG") + seq.count("GY")
        sr_n   = seq.count("SR") + seq.count("RS")
        qn_n   = seq.count("QN") + seq.count("NQ")

        # --- Sticker & spacer ---
        sticker_arom_n  = arom_n
        sticker_elec_n  = sum(aa_counts.get(k, 0) for k in "KRDE")
        sticker_total_n = sum(1 for aa in seq if aa in STICKER_ALL)
        sticker_frac    = sticker_total_n / seq_length
        spacing         = sticker_spacing_stats(seq)
        _fmt_spacing    = lambda v: f"{v:.1f}" if v is not None else "N/A"

        # --- Hydrophobicity features ---
        hydro_vals    = [KYTE_DOOLITTLE[aa] for aa in seq]
        avg_kd        = sum(hydro_vals) / seq_length
        n_hydrophobic = sum(1 for v in hydro_vals if v > 0)
        n_hydrophilic = sum(1 for v in hydro_vals if v < 0)
        n_neutral_kd  = seq_length - n_hydrophobic - n_hydrophilic
        pct_hydro     = n_hydrophobic / seq_length * 100
        pct_hydrophil = n_hydrophilic / seq_length * 100

        # --- HTML sections (styled) ---
        _style = f"<style>{REPORT_CSS}</style>"
        extra_charge = (
            f"<tr><td>Net Charge (pH {pH_value:.1f})</td><td>{net_charge_pH:.2f}</td></tr>"
            if abs(pH_value - 7.0) >= 1e-6 else ""
        )

        sorted_aas = sorted(aa_counts, key=lambda aa: aa_freq[aa], reverse=True)
        comp_html = _style + (
            "<h2>Composition</h2>"
            "<table>"
            "<tr><th>Amino Acid</th><th>Count</th><th>Frequency (%)</th></tr>"
            + "".join(
                f"<tr><td>{aa}</td><td>{aa_counts[aa]}</td><td>{aa_freq[aa]:.2f}%</td></tr>"
                for aa in sorted_aas
            )
            + "</table>"
        )

        bio_html = _style + f"""
        <h2>Properties</h2>
        <table>
          <tr><th>Property</th><th>Value</th></tr>
          <tr><td>Sequence Length</td><td>{seq_length} aa</td></tr>
          <tr><td>Molecular Weight</td><td>{mol_weight:.2f} Da</td></tr>
          <tr><td>Isoelectric Point (pI)</td><td>{iso_point:.2f}</td></tr>
          <tr><td>Net Charge (pH 7.0)</td><td>{net_charge_7:.2f}</td></tr>
          {extra_charge}
          <tr><td>Extinction Coeff. (280 nm)</td><td>{extinction} M&#8315;&#185;cm&#8315;&#185;</td></tr>
          <tr><td>GRAVY Score</td><td>{gravy:.3f}</td></tr>
          <tr><td>Instability Index</td><td>{instability:.2f}</td></tr>
          <tr><td>Aromaticity</td><td>{aromaticity:.3f}</td></tr>
        </table>
        """

        hydro_html = _style + f"""
        <h2>Hydrophobicity</h2>
        <table>
          <tr><th>Property</th><th>Value</th></tr>
          <tr><td>GRAVY Score (Kyte-Doolittle)</td><td>{gravy:.4f}</td></tr>
          <tr><td>Average hydrophobicity per residue</td><td>{avg_kd:.4f}</td></tr>
          <tr><td>Hydrophobic residues (KD &gt; 0)</td><td>{n_hydrophobic} ({pct_hydro:.1f}%)</td></tr>
          <tr><td>Hydrophilic residues (KD &lt; 0)</td><td>{n_hydrophilic} ({pct_hydrophil:.1f}%)</td></tr>
          <tr><td>Neutral residues (KD = 0)</td><td>{n_neutral_kd} ({n_neutral_kd/seq_length*100:.1f}%)</td></tr>
        </table>
        <h2>Kyte-Doolittle Values by Residue</h2>
        <table>
          <tr><th>Amino Acid</th><th>Count</th><th>KD Score</th><th>Contribution</th></tr>
          {"".join(
            f"<tr><td>{aa}</td><td>{aa_counts.get(aa,0)}</td>"
            f"<td>{KYTE_DOOLITTLE[aa]:+.1f}</td>"
            f"<td>{KYTE_DOOLITTLE[aa]*aa_counts.get(aa,0)/seq_length:+.4f}</td></tr>"
            for aa in sorted(KYTE_DOOLITTLE, key=lambda x: KYTE_DOOLITTLE[x], reverse=True)
          )}
        </table>
        <p class="note">GRAVY (Grand Average of Hydropathicity): positive = hydrophobic, negative = hydrophilic (Kyte &amp; Doolittle 1982)</p>
        """

        charge_html = _style + f"""
        <h2>Charge</h2>
        <table>
          <tr><th>Property</th><th>Value</th></tr>
          <tr><td>Positive residues (K, R)</td><td>{pos_n}</td></tr>
          <tr><td>Negative residues (D, E)</td><td>{neg_n}</td></tr>
          <tr><td>FCR (fraction charged)</td><td>{fcr:.3f}</td></tr>
          <tr><td>NCPR (net charge/residue)</td><td>{ncpr:+.3f}</td></tr>
          <tr><td>Charge asymmetry (pos/neg)</td><td>{"%.2f" % ch_asym if neg_n > 0 else "&#8734; (no neg.)"}</td></tr>
          <tr><td>Kappa (&kappa;)</td><td>{kappa:.3f} &mdash; {kappa_interp}</td></tr>
        </table>
        <p class="note">&kappa;: 0 = well-mixed, 1 = fully segregated (Das &amp; Pappu 2013)</p>
        """

        aromatic_html = _style + f"""
        <h2>Aromatic &amp; &pi;-Interactions</h2>
        <table>
          <tr><th>Property</th><th>Value</th></tr>
          <tr><td>Aromatic fraction (F+W+Y)</td><td>{arom_f:.3f} ({arom_n} residues)</td></tr>
          <tr><td>Tyr (Y)</td><td>{n_tyr} ({n_tyr/seq_length*100:.1f}%)</td></tr>
          <tr><td>Phe (F)</td><td>{n_phe} ({n_phe/seq_length*100:.1f}%)</td></tr>
          <tr><td>Trp (W)</td><td>{n_trp} ({n_trp/seq_length*100:.1f}%)</td></tr>
          <tr><td>Cation&ndash;&pi; pairs (K/R &harr; F/W/Y, &plusmn;4)</td><td>{cation_pi_n}</td></tr>
          <tr><td>&pi;&ndash;&pi; pairs (F/W/Y &harr; F/W/Y, &plusmn;4)</td><td>{pi_pi_n}</td></tr>
        </table>
        """

        lc_html = _style + f"""
        <h2>Low Complexity</h2>
        <table>
          <tr><th>Property</th><th>Value</th></tr>
          <tr><td>Shannon entropy</td><td>{entropy:.3f} bits (max 4.32)</td></tr>
          <tr><td>Normalized entropy</td><td>{entropy_norm:.3f}</td></tr>
          <tr><td>Unique amino acids</td><td>{unique_aa} / 20</td></tr>
          <tr><td>Prion-like score (N,Q,S,G,Y)</td><td>{prion_score:.3f}</td></tr>
          <tr><td>LC fraction (w=12, H&lt;2.0 bits)</td><td>{lc_frac:.3f}</td></tr>
        </table>
        <p class="note">Prion-like score: fraction of N,Q,S,G,Y (Lancaster &amp; Bhatt)</p>
        """

        disorder_html = _style + f"""
        <h2>Disorder &amp; Flexibility</h2>
        <table>
          <tr><th>Property</th><th>Value</th></tr>
          <tr><td>Disorder-promoting fraction (A,E,G,K,P,Q,R,S)</td><td>{disorder_f:.3f}</td></tr>
          <tr><td>Order-promoting fraction (C,F,H,I,L,M,V,W,Y)</td><td>{order_f:.3f}</td></tr>
          <tr><td>Aliphatic index</td><td>{aliphatic_idx:.1f}</td></tr>
          <tr><td>Omega (&Omega;)</td><td>{omega:.3f} &mdash; {omega_interp}</td></tr>
        </table>
        <p class="note">Disorder/order: Uversky. Aliphatic index: Ikai 1980. &Omega;: 0 = even, 1 = clustered (Das et al. 2015)</p>
        """

        repeats_html = _style + f"""
        <h2>Repeat Motifs</h2>
        <table>
          <tr><th>Motif</th><th>Count</th></tr>
          <tr><td>RGG</td><td>{rgg_n}</td></tr>
          <tr><td>FG</td><td>{fg_n}</td></tr>
          <tr><td>YG + GY</td><td>{yg_n}</td></tr>
          <tr><td>SR + RS</td><td>{sr_n}</td></tr>
          <tr><td>QN + NQ</td><td>{qn_n}</td></tr>
        </table>
        """

        sticker_html = _style + f"""
        <h2>Sticker &amp; Spacer</h2>
        <table>
          <tr><th>Property</th><th>Value</th></tr>
          <tr><td>Total stickers (F,W,Y,K,R,D,E)</td><td>{sticker_total_n} ({sticker_frac*100:.1f}%)</td></tr>
          <tr><td>Aromatic stickers (F,W,Y)</td><td>{sticker_arom_n}</td></tr>
          <tr><td>Electrostatic stickers (K,R,D,E)</td><td>{sticker_elec_n}</td></tr>
          <tr><td>Mean sticker spacing</td><td>{_fmt_spacing(spacing["mean"])} residues</td></tr>
          <tr><td>Min sticker spacing</td><td>{_fmt_spacing(spacing["min"])} residues</td></tr>
          <tr><td>Max sticker spacing</td><td>{_fmt_spacing(spacing["max"])} residues</td></tr>
        </table>
        <p class="note">Sticker-and-spacer model: Mittag &amp; Pappu</p>
        """

        # --- Transmembrane helix prediction ---
        tm_helices   = predict_tm_helices(seq)
        n_tm         = len(tm_helices)
        tm_rows = "".join(
            f"<tr><td>{i}</td><td>{h['start']+1}</td><td>{h['end']+1}</td>"
            f"<td>{h['end']-h['start']+1}</td><td>{h['score']:.3f}</td>"
            f"<td>{h['orientation']}</td></tr>"
            for i, h in enumerate(tm_helices, 1)
        )
        tm_body = (
            tm_rows
            if tm_helices
            else "<tr><td colspan='6'><em>No TM helices predicted</em></td></tr>"
        )
        tm_html = _style + f"""
        <h2>Transmembrane Helices</h2>
        <table>
          <tr><th>#</th><th>Start</th><th>End</th>
              <th>Length</th><th>Avg KD Score</th><th>Orientation</th></tr>
          {tm_body}
        </table>
        <p class="note">Kyte-Doolittle sliding window (w=19, threshold=1.6).
        Orientation by inside-positive rule (von Heijne): out&rarr;in = N-term extracellular.</p>
        """

        # --- Chou-Fasman secondary structure propensity ---
        cf_helix_arr, cf_sheet_arr = calc_chou_fasman_profile(seq)
        mean_helix = sum(cf_helix_arr) / seq_length
        mean_sheet = sum(cf_sheet_arr) / seq_length
        n_high_helix = sum(1 for v in cf_helix_arr if v > 1.0)
        n_high_sheet = sum(1 for v in cf_sheet_arr if v > 1.0)

        # --- IUPred-style disorder ---
        disorder_scores = calc_disorder_profile(seq)
        mean_disorder = sum(disorder_scores) / seq_length
        disordered_frac = sum(1 for v in disorder_scores if v > 0.5) / seq_length

        ss_html = _style + f"""
        <h2>Secondary Structure Propensity (Chou-Fasman)</h2>
        <table>
          <tr><th>Property</th><th>Value</th></tr>
          <tr><td>Mean helix propensity (P&alpha;)</td><td>{mean_helix:.3f}</td></tr>
          <tr><td>Mean sheet propensity (P&beta;)</td><td>{mean_sheet:.3f}</td></tr>
          <tr><td>Helix-forming residues (P&alpha; &gt; 1.0)</td><td>{n_high_helix} ({n_high_helix/seq_length*100:.1f}%)</td></tr>
          <tr><td>Sheet-forming residues (P&beta; &gt; 1.0)</td><td>{n_high_sheet} ({n_high_sheet/seq_length*100:.1f}%)</td></tr>
          <tr><td>Mean disorder score (IUPred-inspired)</td><td>{mean_disorder:.3f}</td></tr>
          <tr><td>Disordered fraction (score &gt; 0.5)</td><td>{disordered_frac:.3f} ({disordered_frac*100:.1f}%)</td></tr>
        </table>
        <p class="note">Chou-Fasman: P &gt; 1.0 = propensity for helix/sheet. Disorder: IUPred-inspired per-residue score (0=ordered, 1=disordered).</p>
        """

        # --- LARKS & LLPS ---
        larks = detect_larks(seq)
        llps  = calc_llps_score(seq, fcr, ncpr, arom_f, prion_score, omega, disorder_f, larks)
        larks_rows = "".join(
            f"<tr><td>{h['start']+1}–{h['end']+1}</td><td>{h['seq']}</td>"
            f"<td>{h['n_arom']}</td><td>{h['lc_frac']:.2f}</td><td>{h['entropy']:.2f}</td></tr>"
            for h in larks
        ) or "<tr><td colspan='5'><em>No LARKS detected</em></td></tr>"
        comp_rows = "".join(
            f"<tr><td>{k}</td><td>{v:.3f}</td></tr>"
            for k, v in llps["components"].items()
        )
        verdict_color = (
            "#16a34a" if "High" in llps["verdict"] else
            "#ca8a04" if "Moderate" in llps["verdict"] else
            "#dc2626"
        )
        phase_html = _style + f"""
        <h2>Phase Separation Propensity</h2>
        <table>
          <tr><th>Property</th><th>Value</th></tr>
          <tr><td><b>Composite LLPS Score</b></td>
              <td><b style="color:{verdict_color};">{llps['score']:.3f} — {llps['verdict']}</b></td></tr>
        </table>
        <h3>Score Components</h3>
        <table>
          <tr><th>Component (normalised)</th><th>Value</th></tr>
          {comp_rows}
        </table>
        <h3>LARKS (Low-complexity Aromatic-Rich Kinked Segments)</h3>
        <table>
          <tr><th>Position</th><th>Sequence</th><th>Aromatics</th><th>LC Frac</th><th>Entropy (bits)</th></tr>
          {larks_rows}
        </table>
        <p class="note">LLPS score: weighted combination of aromatic fraction, prion-like score,
        disorder fraction, FCR, Omega, LARKS density, and |NCPR| penalty.
        LARKS: 7-residue windows with ≥1 aromatic, ≥50% LC residues, entropy &lt;1.8 bits (Hughes et al. 2018).</p>
        """

        # --- Coiled-coil prediction ---
        cc_profile = predict_coiled_coil(seq)
        cc_threshold = 0.50
        cc_regions = []
        i = 0
        while i < seq_length:
            if cc_profile[i] >= cc_threshold:
                j = i
                while j < seq_length and cc_profile[j] >= cc_threshold:
                    j += 1
                if j - i >= 7:
                    cc_regions.append((i + 1, j))
                i = j
            else:
                i += 1
        n_cc_res = sum(e - s + 1 for s, e in cc_regions)
        cc_frac  = n_cc_res / seq_length if seq_length > 0 else 0.0

        # --- Linear motif scan ---
        motifs = scan_linear_motifs(seq)
        motif_rows = "".join(
            f"<tr><td>{m['name']}</td><td>{m['start']+1}–{m['end']+1}</td>"
            f"<td><tt>{m['match']}</tt></td><td>{m['description']}</td></tr>"
            for m in motifs
        ) or "<tr><td colspan='4'><em>No motifs found</em></td></tr>"
        motifs_html = _style + f"""
        <h2>Linear Motif Scan</h2>
        <p>Scanned against {len(LINEAR_MOTIFS)} built-in motif patterns.</p>
        <table>
          <tr><th>Motif</th><th>Position</th><th>Match</th><th>Description</th></tr>
          {motif_rows}
        </table>
        <p class="note">Pattern library includes NLS, NES, PxxP, 14-3-3, RGG, FG, KFERQ, KDEL,
        N-glycosylation, SUMOylation, CK2 phosphorylation, caspase cleavage, WW-domain, SxIP, PKA sites.
        Matches are regex-based and require experimental validation.</p>
        """

        # ── New feature sections ─────────────────────────────────────────────
        if _HAS_AGGREGATION:
            aggr_html  = format_aggregation_report(seq, _style)
            solub_stats = calc_solubility_stats(seq)
        else:
            aggr_html   = _style + "<h2>β-Aggregation & Solubility</h2><p>Module not available. Run: pip install beer-biophys</p>"
            solub_stats = {}

        if _HAS_PTM:
            ptm_html = format_ptm_report(seq, _style)
            ptm_sites = scan_ptm_sites(seq)
        else:
            ptm_html  = _style + "<h2>PTM Sites</h2><p>Module not available.</p>"
            ptm_sites = []

        if _HAS_SIGNAL:
            signal_html = format_signal_report(seq, _style)
            sp_result   = predict_signal_peptide(seq)
            gpi_result  = predict_gpi_anchor(seq)
        else:
            signal_html = _style + "<h2>Signal Peptide & GPI</h2><p>Module not available.</p>"
            sp_result   = {}
            gpi_result  = {}

        if _HAS_AMPHIPATHIC:
            amph_html    = format_amphipathic_report(seq, _style)
            moment_alpha = calc_hydrophobic_moment_profile(seq, angle_deg=100.0)
            moment_beta  = calc_hydrophobic_moment_profile(seq, angle_deg=160.0)
            amph_regions = predict_amphipathic_helices(seq)
        else:
            amph_html    = _style + "<h2>Amphipathic Helices</h2><p>Module not available.</p>"
            moment_alpha = []
            moment_beta  = []
            amph_regions = []

        if _HAS_SCD:
            scd_val   = calc_scd(seq)
            scd_profile_data = calc_scd_profile(seq, window=20)
            scd_blocks = calc_pos_neg_block_lengths(seq)
            scd_html  = format_scd_report(seq, _style)
        else:
            scd_val   = 0.0
            scd_profile_data = []
            scd_blocks = {}
            scd_html  = _style + "<h2>Charge Decoration (SCD)</h2><p>Module not available.</p>"

        if _HAS_RBP:
            rbp_result  = calc_rbp_score(seq)
            rbp_profile_data = calc_rbp_profile(seq)
            rbp_html    = format_rbp_report(seq, _style)
        else:
            rbp_result  = {}
            rbp_profile_data = []
            rbp_html    = _style + "<h2>RNA Binding</h2><p>Module not available.</p>"

        if _HAS_TANDEM:
            tandem_html = format_tandem_repeats_report(seq, _style)
            tandem_stats = calc_repeat_stats(seq)
        else:
            tandem_html  = _style + "<h2>Tandem Repeats</h2><p>Module not available.</p>"
            tandem_stats = {}

        return {
            "report_sections": {
                "Composition":        comp_html,
                "Properties":         bio_html,
                "Hydrophobicity":     hydro_html,
                "Charge":             charge_html,
                "Aromatic & \u03c0":  aromatic_html,
                "Low Complexity":     lc_html,
                "Disorder":           disorder_html,
                "Secondary Structure": ss_html,
                "Repeat Motifs":      repeats_html,
                "Sticker & Spacer":   sticker_html,
                "TM Helices":         tm_html,
                "Phase Separation":   phase_html,
                "Linear Motifs":      motifs_html,
                # New sections
                "\u03b2-Aggregation & Solubility": aggr_html,
                "PTM Sites":          ptm_html,
                "Signal Peptide & GPI": signal_html,
                "Amphipathic Helices": amph_html,
                "Charge Decoration (SCD)": scd_html,
                "RNA Binding":        rbp_html,
                "Tandem Repeats":     tandem_html,
            },
            "tm_helices":      tm_helices,
            "aa_counts":       aa_counts,
            "aa_freq":         aa_freq,
            "hydro_profile":   sliding_window_hydrophobicity(seq, window_size),
            "ncpr_profile":    sliding_window_ncpr(seq, window_size),
            "entropy_profile": sliding_window_entropy(seq, window_size),
            "cf_helix":        cf_helix_arr,
            "cf_sheet":        cf_sheet_arr,
            "disorder_scores": disorder_scores,
            "window_size":     window_size,
            "seq":             seq,
            "mol_weight":      mol_weight,
            "iso_point":       iso_point,
            "net_charge_7":    net_charge_7,
            "extinction":      extinction,
            "gravy":           gravy,
            "instability":     instability,
            "aromaticity":     aromaticity,
            "fcr":             fcr,
            "ncpr":            ncpr,
            "arom_f":          arom_f,
            "prion_score":     prion_score,
            "omega":           omega,
            "disorder_f":      disorder_f,
            "larks":           larks,
            "llps":            llps,
            "cc_profile":      cc_profile,
            "motifs":          motifs,
            # New feature data
            "solub_stats":     solub_stats,
            "ptm_sites":       ptm_sites,
            "sp_result":       sp_result,
            "gpi_result":      gpi_result,
            "moment_alpha":    moment_alpha,
            "moment_beta":     moment_beta,
            "amph_regions":    amph_regions,
            "scd":             scd_val,
            "scd_profile":     scd_profile_data,
            "scd_blocks":      scd_blocks,
            "rbp":             rbp_result,
            "rbp_profile":     rbp_profile_data,
            "tandem_stats":    tandem_stats,
        }

# --- Graphing ---

# Colour palette for publication-quality graphs
_PALETTE = ["#4361ee", "#f72585", "#4cc9f0", "#7209b7", "#3a0ca3",
            "#f3722c", "#43aa8b", "#277da1", "#577590", "#90be6d"]
_ACCENT  = "#4361ee"
_FILL    = "#4361ee"
_NEG_COL = "#f72585"
_POS_COL = "#4361ee"
_NEUT_COL = "#adb5bd"


class GraphingTools:

    @staticmethod
    def create_amino_acid_composition_figure(aa_counts, aa_freq, label_font=14, tick_font=12):
        fig = Figure(figsize=(9, 4.5), dpi=120)
        fig.set_facecolor("#ffffff")
        ax  = fig.add_subplot(111)
        aas  = sorted(aa_counts)
        cnts = [aa_counts[a] for a in aas]
        colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(aas))]
        bars = ax.bar(aas, cnts, color=colors, width=0.65, zorder=3,
                      edgecolor="white", linewidth=0.5)
        _pub_style_ax(ax,
                      title="Amino Acid Composition",
                      xlabel="Amino Acid",
                      ylabel="Count",
                      grid=True,
                      title_size=label_font+1,
                      label_size=label_font-1,
                      tick_size=tick_font-1)
        max_cnt = max(cnts) if cnts else 1
        for bar, a in zip(bars, aas):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + max_cnt*0.01,
                    f"{aa_freq[a]:.1f}%", ha="center", va="bottom",
                    fontsize=tick_font-3, color="#4a5568", fontweight="500")
        fig.tight_layout(pad=1.5)
        mplcursors.cursor(ax)
        return fig

    @staticmethod
    def create_amino_acid_composition_pie_figure(aa_counts, label_font=14):
        # Filter out zero-count amino acids
        items  = [(aa, cnt) for aa, cnt in aa_counts.items() if cnt > 0]
        labels = [x[0] for x in items]
        values = [x[1] for x in items]
        colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(labels))]

        fig = Figure(figsize=(7, 5), dpi=120)
        fig.set_facecolor("#ffffff")
        ax  = fig.add_subplot(111)
        wedges, texts, autotexts = ax.pie(
            values, labels=labels, colors=colors,
            autopct="%1.1f%%", startangle=140,
            pctdistance=0.82,
            wedgeprops=dict(linewidth=0.8, edgecolor="white")
        )
        for t in texts:
            t.set_fontsize(label_font - 3)
            t.set_color("#2d3748")
        for at in autotexts:
            at.set_fontsize(label_font - 4)
            at.set_color("#ffffff")
            at.set_fontweight("bold")
        ax.set_title("Amino Acid Composition", fontsize=label_font+1,
                     fontweight="bold", color="#1a1a2e", pad=12)
        fig.tight_layout(pad=1.5)
        mplcursors.cursor(ax)
        return fig

    @staticmethod
    def create_hydrophobicity_figure(hydro_profile, window_size, label_font=14, tick_font=12):
        fig = Figure(figsize=(9, 4), dpi=120)
        fig.set_facecolor("#ffffff")
        ax  = fig.add_subplot(111)
        xs  = list(range(1, len(hydro_profile)+1))
        # Colour fill above/below zero
        ax.fill_between(xs, hydro_profile, 0,
                        where=[v >= 0 for v in hydro_profile],
                        alpha=0.18, color=_ACCENT, interpolate=True)
        ax.fill_between(xs, hydro_profile, 0,
                        where=[v < 0 for v in hydro_profile],
                        alpha=0.18, color=_NEG_COL, interpolate=True)
        ax.plot(xs, hydro_profile, color=_ACCENT, linewidth=1.8,
                marker="o", markersize=3.5, markerfacecolor=_ACCENT,
                markeredgewidth=0, zorder=4)
        ax.axhline(0, color="#888", linewidth=0.8, linestyle="--", zorder=3)
        _pub_style_ax(ax,
                      title=f"Hydrophobicity Profile  (window = {window_size})",
                      xlabel="Residue Position",
                      ylabel="Kyte-Doolittle Score",
                      grid=True,
                      title_size=label_font+1,
                      label_size=label_font-1,
                      tick_size=tick_font-1)
        fig.tight_layout(pad=1.5)
        mplcursors.cursor(ax)
        return fig

    @staticmethod
    def create_net_charge_vs_pH_figure(seq, label_font=14, tick_font=12, pka=None):
        fig = Figure(figsize=(8, 4), dpi=120)
        fig.set_facecolor("#ffffff")
        ax  = fig.add_subplot(111)
        phs  = [i/10 for i in range(141)]
        nets = [calc_net_charge(seq, p, pka) for p in phs]
        ax.plot(phs, nets, color=_ACCENT, linewidth=2.0, zorder=4)
        ax.fill_between(phs, nets, 0,
                        where=[v >= 0 for v in nets],
                        alpha=0.12, color=_POS_COL, interpolate=True)
        ax.fill_between(phs, nets, 0,
                        where=[v < 0 for v in nets],
                        alpha=0.12, color=_NEG_COL, interpolate=True)
        ax.axhline(0, color="#888", linewidth=0.8, linestyle="--", zorder=3)
        # Mark pI
        pI = min(zip(nets, phs), key=lambda x: abs(x[0]))[1]
        ax.axvline(pI, color="#f72585", linewidth=1.0, linestyle=":", alpha=0.8)
        ax.text(pI + 0.2, max(nets)*0.85, f"pI ≈ {pI:.1f}",
                fontsize=tick_font - 1, color="#f72585")
        _pub_style_ax(ax,
                      title="Net Charge vs. pH",
                      xlabel="pH",
                      ylabel="Net Charge",
                      grid=True,
                      title_size=label_font+1,
                      label_size=label_font-1,
                      tick_size=tick_font-1)
        ax.set_xlim(0, 14)
        fig.tight_layout(pad=1.5)
        mplcursors.cursor(ax)
        return fig

    @staticmethod
    def create_bead_model_hydrophobicity_figure(seq, show_labels, label_font=14, tick_font=12, cmap="coolwarm"):
        n    = len(seq)
        w    = max(10, min(20, 0.28 * n))
        fig  = Figure(figsize=(w, 2.4), dpi=120)
        fig.set_facecolor("#ffffff")
        ax   = fig.add_subplot(111)
        xs   = list(range(1, n+1))
        vals = [KYTE_DOOLITTLE[aa] for aa in seq]
        sc   = ax.scatter(xs, [1]*n, c=vals, cmap=cmap,
                          s=220, linewidths=0.5, edgecolors="white",
                          vmin=-4.5, vmax=4.5, zorder=4)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.7, aspect=15, pad=0.02)
        cbar.set_label("Hydrophobicity", fontsize=tick_font-1, color="#4a5568")
        cbar.ax.tick_params(labelsize=tick_font-2, colors="#4a5568")
        ax.set_yticks([])
        ax.set_xlim(0, n + 1)
        ax.set_ylim(0.5, 1.5)
        _pub_style_ax(ax,
                      title="Bead Model — Hydrophobicity",
                      xlabel="Residue Position",
                      grid=False,
                      despine=False,
                      title_size=label_font,
                      label_size=label_font-2,
                      tick_size=tick_font-2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        if show_labels and n <= 60:
            for i, aa in enumerate(seq):
                ax.text(xs[i], 1, aa, ha="center", va="center",
                        fontsize=max(5, label_font-5), color="white",
                        fontweight="bold")
        # Position ruler every 10
        ticks = [i for i in range(10, n+1, 10)]
        ax.set_xticks(ticks)
        ax.tick_params(labelsize=tick_font-2)
        fig.tight_layout(pad=1.2)
        mplcursors.cursor(ax)
        return fig

    @staticmethod
    def create_bead_model_charge_figure(seq, show_labels, label_font=14, tick_font=12):
        n    = len(seq)
        w    = max(10, min(20, 0.28 * n))
        fig  = Figure(figsize=(w, 2.4), dpi=120)
        fig.set_facecolor("#ffffff")
        ax   = fig.add_subplot(111)
        xs   = list(range(1, n+1))
        pos_c  = "#4361ee"
        neg_c  = "#f72585"
        neu_c  = "#adb5bd"
        his_c  = "#4cc9f0"
        cols = []
        for aa in seq:
            if aa in "KR":
                cols.append(pos_c)
            elif aa in "DE":
                cols.append(neg_c)
            elif aa == "H":
                cols.append(his_c)
            else:
                cols.append(neu_c)
        ax.scatter(xs, [1]*n, c=cols, s=220, linewidths=0.5,
                   edgecolors="white", zorder=4)
        ax.legend(handles=[
            Patch(color=pos_c, label="Positive (K, R)"),
            Patch(color=neg_c, label="Negative (D, E)"),
            Patch(color=his_c, label="His (H)"),
            Patch(color=neu_c, label="Neutral"),
        ], loc="upper right", fontsize=max(7, label_font-5),
           framealpha=0.85, edgecolor="#d0d4e0")
        ax.set_yticks([])
        ax.set_xlim(0, n + 1)
        ax.set_ylim(0.5, 1.5)
        _pub_style_ax(ax,
                      title="Bead Model — Charge",
                      xlabel="Residue Position",
                      grid=False,
                      despine=False,
                      title_size=label_font,
                      label_size=label_font-2,
                      tick_size=tick_font-2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        if show_labels and n <= 60:
            for i, aa in enumerate(seq):
                ax.text(xs[i], 1, aa, ha="center", va="center",
                        fontsize=max(5, label_font-5), color="white",
                        fontweight="bold")
        ticks = [i for i in range(10, n+1, 10)]
        ax.set_xticks(ticks)
        ax.tick_params(labelsize=tick_font-2)
        fig.tight_layout(pad=1.2)
        mplcursors.cursor(ax)
        return fig

    @staticmethod
    def create_sticker_map(seq, show_labels, label_font=14, tick_font=12):
        n    = len(seq)
        w    = max(10, min(20, 0.28 * n))
        fig  = Figure(figsize=(w, 2.4), dpi=120)
        fig.set_facecolor("#ffffff")
        ax   = fig.add_subplot(111)
        xs   = list(range(1, n+1))
        arom_c  = "#f59e0b"   # amber
        basic_c = "#4361ee"   # blue
        acid_c  = "#f72585"   # pink-red
        space_c = "#e2e8f0"   # light gray
        cols = []
        for aa in seq:
            if aa in STICKER_AROMATIC:
                cols.append(arom_c)
            elif aa in "KR":
                cols.append(basic_c)
            elif aa in "DE":
                cols.append(acid_c)
            else:
                cols.append(space_c)
        ax.scatter(xs, [1]*n, c=cols, s=220, linewidths=0.5,
                   edgecolors="white", zorder=4)
        ax.legend(handles=[
            Patch(color=arom_c,  label="Aromatic (F,W,Y)"),
            Patch(color=basic_c, label="Basic (K,R)"),
            Patch(color=acid_c,  label="Acidic (D,E)"),
            Patch(color=space_c, label="Spacer"),
        ], loc="upper right", fontsize=max(7, label_font-5),
           framealpha=0.85, edgecolor="#d0d4e0")
        ax.set_yticks([])
        ax.set_xlim(0, n + 1)
        ax.set_ylim(0.5, 1.5)
        _pub_style_ax(ax,
                      title="Sticker Map",
                      xlabel="Residue Position",
                      grid=False,
                      despine=False,
                      title_size=label_font,
                      label_size=label_font-2,
                      tick_size=tick_font-2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        if show_labels and n <= 60:
            for i, aa in enumerate(seq):
                tc = "#ffffff" if aa in STICKER_ALL else "#6b7280"
                ax.text(xs[i], 1, aa, ha="center", va="center",
                        fontsize=max(5, label_font-5), color=tc,
                        fontweight="bold")
        ticks = [i for i in range(10, n+1, 10)]
        ax.set_xticks(ticks)
        ax.tick_params(labelsize=tick_font-2)
        fig.tight_layout(pad=1.2)
        mplcursors.cursor(ax)
        return fig

    @staticmethod
    def create_local_charge_figure(ncpr_profile, window_size, label_font=14, tick_font=12):
        fig = Figure(figsize=(9, 4), dpi=120)
        fig.set_facecolor("#ffffff")
        ax  = fig.add_subplot(111)
        xs  = list(range(1, len(ncpr_profile)+1))
        ax.fill_between(xs, ncpr_profile, 0,
                        where=[v >= 0 for v in ncpr_profile],
                        alpha=0.18, color=_POS_COL, interpolate=True)
        ax.fill_between(xs, ncpr_profile, 0,
                        where=[v < 0 for v in ncpr_profile],
                        alpha=0.18, color=_NEG_COL, interpolate=True)
        ax.plot(xs, ncpr_profile, color=_ACCENT, linewidth=1.8,
                marker="o", markersize=3.5, markeredgewidth=0, zorder=4)
        ax.axhline(0, color="#888", linewidth=0.8, linestyle="--", zorder=3)
        _pub_style_ax(ax,
                      title=f"Local Charge Profile  (window = {window_size})",
                      xlabel="Residue Position",
                      ylabel="NCPR",
                      grid=True,
                      title_size=label_font+1,
                      label_size=label_font-1,
                      tick_size=tick_font-1)
        fig.tight_layout(pad=1.5)
        mplcursors.cursor(ax)
        return fig

    @staticmethod
    def create_local_complexity_figure(entropy_profile, window_size, label_font=14, tick_font=12):
        fig = Figure(figsize=(9, 4), dpi=120)
        fig.set_facecolor("#ffffff")
        ax  = fig.add_subplot(111)
        xs  = list(range(1, len(entropy_profile)+1))
        ax.fill_between(xs, entropy_profile, 2.0,
                        where=[v < 2.0 for v in entropy_profile],
                        alpha=0.18, color=_NEG_COL, interpolate=True,
                        label="Low complexity region")
        ax.plot(xs, entropy_profile, color=_ACCENT, linewidth=1.8,
                marker="o", markersize=3.5, markeredgewidth=0, zorder=4,
                label="Entropy")
        ax.axhline(2.0, color="#f72585", linewidth=1.2, linestyle="--",
                   zorder=3, label="LC threshold (2.0 bits)")
        _pub_style_ax(ax,
                      title=f"Local Complexity  (window = {window_size})",
                      xlabel="Residue Position",
                      ylabel="Shannon Entropy (bits)",
                      grid=True,
                      title_size=label_font+1,
                      label_size=label_font-1,
                      tick_size=tick_font-1)
        ax.legend(fontsize=tick_font-2, framealpha=0.85, edgecolor="#d0d4e0")
        fig.tight_layout(pad=1.5)
        mplcursors.cursor(ax)
        return fig

    @staticmethod
    def create_cation_pi_map(seq, label_font=14, tick_font=12):
        n      = len(seq)
        window = 8
        mat    = np.zeros((n, n))
        arom   = STICKER_AROMATIC
        basic  = set("KR")
        for i in range(n):
            if seq[i] in basic or seq[i] in arom:
                for j in range(max(0, i-window), min(n, i+window+1)):
                    if j != i:
                        is_cp = (seq[i] in basic and seq[j] in arom) or \
                                (seq[i] in arom  and seq[j] in basic)
                        if is_cp:
                            mat[i, j] = 1.0 / abs(i - j)
        fig = Figure(figsize=(6, 5), dpi=120)
        fig.set_facecolor("#ffffff")
        ax  = fig.add_subplot(111)
        ax.set_facecolor("#fafbff")
        im  = ax.imshow(mat, cmap="YlOrRd", aspect="auto", origin="upper",
                        interpolation="nearest")
        cbar = fig.colorbar(im, ax=ax, shrink=0.85, aspect=20, pad=0.02)
        cbar.set_label("Proximity (1 / distance)", fontsize=tick_font-1, color="#4a5568")
        cbar.ax.tick_params(labelsize=tick_font-2, colors="#4a5568")
        _pub_style_ax(ax,
                      title="Cation\u2013\u03c0 Proximity Map",
                      xlabel="Residue Position",
                      ylabel="Residue Position",
                      grid=False,
                      title_size=label_font+1,
                      label_size=label_font-1,
                      tick_size=tick_font-1)
        fig.tight_layout(pad=1.5)
        return fig

    @staticmethod
    def create_isoelectric_focus(seq, label_font=14, tick_font=12, pka=None):
        """Enhanced isoelectric focusing simulation."""
        fig = Figure(figsize=(9, 4.5), dpi=120)
        fig.set_facecolor("#ffffff")
        ax  = fig.add_subplot(111)
        phs  = [i / 20 for i in range(281)]   # 0 to 14, step 0.05
        nets = [calc_net_charge(seq, p, pka) for p in phs]
        ax.plot(phs, nets, color=_ACCENT, linewidth=2.2, zorder=5)
        ax.fill_between(phs, nets, 0,
                        where=[v >= 0 for v in nets],
                        alpha=0.15, color=_POS_COL, interpolate=True, label="Positive region")
        ax.fill_between(phs, nets, 0,
                        where=[v < 0 for v in nets],
                        alpha=0.15, color=_NEG_COL, interpolate=True, label="Negative region")
        ax.axhline(0, color="#888", linewidth=0.8, linestyle="--", zorder=3)
        # Locate pI
        pI_idx = min(range(len(nets)), key=lambda i: abs(nets[i]))
        pI     = phs[pI_idx]
        ax.axvline(pI, color="#f72585", linewidth=1.8, linestyle="-", alpha=0.9, zorder=4)
        y_top = max(nets) if max(nets) > 0 else 1
        ax.annotate(f"  pI = {pI:.2f}", xy=(pI, 0),
                    xytext=(pI + 0.5, y_top * 0.65),
                    fontsize=tick_font, color="#f72585", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#f72585", lw=1.2))
        # Physiological pH 7.4
        ch74 = calc_net_charge(seq, 7.4, pka)
        ax.axvline(7.4, color="#43aa8b", linewidth=1.0, linestyle=":", alpha=0.8, zorder=3)
        y_bot = min(nets) if min(nets) < 0 else -1
        ax.text(7.55, y_bot * 0.65,
                f"pH 7.4\n({ch74:+.1f})", fontsize=tick_font - 2, color="#43aa8b")
        _pub_style_ax(ax,
                      title="Isoelectric Focusing Simulation",
                      xlabel="pH", ylabel="Net Charge",
                      grid=True, title_size=label_font + 1,
                      label_size=label_font - 1, tick_size=tick_font - 1)
        ax.set_xlim(0, 14)
        ax.legend(fontsize=tick_font - 2, framealpha=0.85, edgecolor="#d0d4e0", loc="upper right")
        fig.tight_layout(pad=1.5)
        mplcursors.cursor(ax)
        return fig

    @staticmethod
    def create_secondary_structure(cf_helix, cf_sheet, label_font=14, tick_font=12):
        """Chou-Fasman helix (top) and sheet (bottom) propensity — two stacked subplots."""
        n  = len(cf_helix)
        xs = list(range(1, n + 1))
        fig = Figure(figsize=(9, 6), dpi=120)
        fig.set_facecolor("#ffffff")
        fig.suptitle("Secondary Structure Propensity", fontsize=label_font,
                     fontweight="bold", color="#1a1a2e", y=0.98)
        ax1, ax2 = fig.subplots(2, 1, sharex=True)
        fig.subplots_adjust(hspace=0.35, left=0.10, right=0.97, top=0.92, bottom=0.10)

        for ax, vals, col, ylabel in [
            (ax1, cf_helix, "#4361ee", "Helix P\u03b1"),
            (ax2, cf_sheet, "#f72585", "Sheet P\u03b2"),
        ]:
            ax.set_facecolor("#fafbff")
            ax.plot(xs, vals, color=col, linewidth=1.6, zorder=4)
            ax.fill_between(xs, vals, 1.0,
                            where=[v > 1.0 for v in vals],
                            alpha=0.22, color=col, interpolate=True,
                            label="Above neutral")
            ax.fill_between(xs, vals, 1.0,
                            where=[v <= 1.0 for v in vals],
                            alpha=0.08, color=col, interpolate=True)
            ax.axhline(1.0, color="#888", linewidth=0.9, linestyle="--",
                       zorder=3, label="Neutral (1.0)")
            _pub_style_ax(ax, ylabel=ylabel, grid=True,
                          label_size=label_font - 2, tick_size=tick_font - 2)
            ylo = min(min(vals) - 0.05, 0.45)
            yhi = max(max(vals) + 0.05, 1.75)
            ax.set_ylim(ylo, yhi)
            ax.legend(fontsize=tick_font - 3, framealpha=0.85,
                      edgecolor="#d0d4e0", loc="upper right")

        ax2.set_xlabel("Residue Position", fontsize=label_font - 2,
                       labelpad=4, color="#2d3748")
        mplcursors.cursor(ax1)
        mplcursors.cursor(ax2)
        return fig

    @staticmethod
    def create_helical_wheel(seq, label_font=14):
        """Helical wheel projection using Cartesian layout with connecting lines.
        First ≤18 residues placed at 100° per step, starting from the top (North),
        proceeding clockwise.  Dot colour = Kyte-Doolittle hydrophobicity; text
        colour chosen by luminance for contrast.
        """
        seg = seq[:18]
        n   = len(seg)
        fig = Figure(figsize=(6.0, 6.0), dpi=120)
        fig.set_facecolor("#ffffff")
        ax  = fig.add_axes([0.06, 0.06, 0.78, 0.84])          # leave room for colourbar
        ax.set_facecolor("#fafbff")
        ax.set_aspect("equal")
        ax.axis("off")

        cmap    = plt.get_cmap("RdYlBu_r")
        kd_min, kd_max = -4.5, 4.5
        norm    = Normalize(vmin=kd_min, vmax=kd_max)

        DOT_R   = 0.13          # radius of each residue circle (data-space)
        RING_R  = 1.0           # distance from centre to each residue

        # Convert step index → Cartesian (start at top, clockwise)
        def _pos(i):
            theta = math.radians(90.0 - i * 100.0)   # 90° = top; −100° per residue
            return math.cos(theta) * RING_R, math.sin(theta) * RING_R

        xs = [_pos(i)[0] for i in range(n)]
        ys = [_pos(i)[1] for i in range(n)]

        # Draw connecting lines between sequential residues (behind dots)
        for i in range(n - 1):
            ax.plot([xs[i], xs[i + 1]], [ys[i], ys[i + 1]],
                    color="#b0bac8", linewidth=1.2, zorder=2, solid_capstyle="round")

        # Draw dots + labels
        for i, aa in enumerate(seg):
            kd  = KYTE_DOOLITTLE.get(aa, 0.0)
            col = cmap(norm(kd))
            r, g, b, _ = col
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            txt_col = "#1a1a2e" if lum > 0.45 else "white"

            circle = plt.Circle((xs[i], ys[i]), DOT_R,
                                 color=col, zorder=4,
                                 linewidth=1.2, edgecolor="#718096")
            ax.add_patch(circle)
            ax.text(xs[i], ys[i], aa,
                    ha="center", va="center",
                    fontsize=label_font - 2, fontweight="bold",
                    color=txt_col, zorder=5)
            # Residue number outside the dot
            nx = xs[i] * (1.0 + (DOT_R + 0.07) / RING_R)
            ny = ys[i] * (1.0 + (DOT_R + 0.07) / RING_R)
            ax.text(nx, ny, str(i + 1),
                    ha="center", va="center",
                    fontsize=max(6, label_font - 6), color="#718096", zorder=5)

        pad = RING_R + DOT_R + 0.28
        ax.set_xlim(-pad, pad)
        ax.set_ylim(-pad, pad)
        ax.set_title(f"Helical Wheel  (residues 1\u2013{n})",
                     fontsize=label_font, fontweight="bold", color="#1a1a2e", pad=10)

        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cax = fig.add_axes([0.87, 0.12, 0.03, 0.68])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("Hydrophobicity (KD)", fontsize=label_font - 4, color="#4a5568")
        cbar.ax.tick_params(labelsize=label_font - 5, colors="#4a5568")
        return fig

    @staticmethod
    def create_charge_decoration(fcr, ncpr, label_font=14, tick_font=12):
        """Das-Pappu FCR vs |NCPR| phase diagram."""
        abs_ncpr = abs(ncpr)
        fig = Figure(figsize=(6, 5.5), dpi=120)
        fig.set_facecolor("#ffffff")
        ax  = fig.add_subplot(111)
        ax.set_facecolor("#fafbff")
        # Feasibility boundary |NCPR| ≤ FCR
        fcr_arr = np.linspace(0, 0.72, 200)
        ax.plot(fcr_arr, fcr_arr, color="#c0c4d0", linewidth=1.5,
                linestyle="--", zorder=2, label="|NCPR| = FCR boundary")
        ax.fill_between(fcr_arr, 0, fcr_arr, alpha=0.04, color="#4361ee")
        # Phase regions
        ax.axvspan(0,    0.25, alpha=0.09, color="#43aa8b")
        ax.axvspan(0.25, 0.35, alpha=0.09, color="#f3722c")
        ax.axvspan(0.35, 0.72, alpha=0.09, color="#f72585")
        ax.axvline(0.25, color="#888", linewidth=0.7, linestyle=":", zorder=3)
        ax.axvline(0.35, color="#888", linewidth=0.7, linestyle=":", zorder=3)
        # Region labels
        ax.text(0.12, 0.62, "Globule /\nCollapsed", ha="center", fontsize=tick_font - 3,
                color="#43aa8b", fontweight="600")
        ax.text(0.30, 0.62, "Coil /\nTadpole", ha="center", fontsize=tick_font - 3,
                color="#f3722c", fontweight="600")
        ax.text(0.54, 0.62, "Strong\nPolyelectrolyte", ha="center", fontsize=tick_font - 3,
                color="#f72585", fontweight="600")
        # Protein point
        ax.scatter([fcr], [abs_ncpr], marker="*", s=380, color="#f72585",
                   zorder=10, edgecolors="white", linewidths=0.8, label="This protein")
        ax.annotate(f"  ({fcr:.2f}, {abs_ncpr:.2f})", xy=(fcr, abs_ncpr),
                    fontsize=tick_font - 1, color="#f72585", va="bottom")
        _pub_style_ax(ax,
                      title="Charge Decoration Phase Diagram (Das-Pappu)",
                      xlabel="FCR  (Fraction of Charged Residues)",
                      ylabel="|NCPR|  (|Net Charge Per Residue|)",
                      grid=True, title_size=label_font + 1,
                      label_size=label_font - 1, tick_size=tick_font - 1)
        ax.set_xlim(0, 0.72)
        ax.set_ylim(0, 0.72)
        ax.legend(fontsize=tick_font - 3, framealpha=0.85, edgecolor="#d0d4e0",
                  loc="upper left")
        fig.tight_layout(pad=1.5)
        mplcursors.cursor(ax)
        return fig

    @staticmethod
    def create_linear_sequence_map(seq, hydro_profile, ncpr_profile,
                                   disorder_scores, cf_helix,
                                   label_font=14, tick_font=12):
        """4-track linear sequence map."""
        n      = len(seq)
        xs_win = list(range(1, len(hydro_profile) + 1))
        xs_all = list(range(1, n + 1))
        fig    = Figure(figsize=(10, 7), dpi=120)
        fig.set_facecolor("#ffffff")
        axs = fig.subplots(4, 1, sharex=False)
        fig.subplots_adjust(hspace=0.55, left=0.10, right=0.97, top=0.93, bottom=0.07)
        fig.suptitle("Linear Sequence Map", fontsize=label_font + 1,
                     fontweight="bold", color="#1a1a2e")

        def _track(ax, xs, ys, col, zero, ylabel_txt, fill_above=True, fill_below=True):
            if fill_above:
                ax.fill_between(xs, ys, zero,
                                where=[v >= zero for v in ys],
                                alpha=0.28, color=col, interpolate=True)
            if fill_below:
                ax.fill_between(xs, ys, zero,
                                where=[v < zero for v in ys],
                                alpha=0.28, color="#f72585", interpolate=True)
            ax.plot(xs, ys, color=col, linewidth=1.3)
            ax.axhline(zero, color="#aaa", linewidth=0.6, linestyle="--")
            ax.set_ylabel(ylabel_txt, fontsize=tick_font - 2, color="#4a5568")
            ax.tick_params(labelsize=tick_font - 3, length=3)
            for sp in ["top", "right"]:
                ax.spines[sp].set_visible(False)
            ax.set_facecolor("#fafbff")
            ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, color="#c8cdd8")
            ax.set_axisbelow(True)

        _track(axs[0], xs_win, hydro_profile, "#4361ee",  0, "Hydrophobicity")
        _track(axs[1], xs_win, ncpr_profile,  "#7209b7",  0, "NCPR")
        _track(axs[2], xs_all, disorder_scores, "#f3722c", 0.5, "Disorder",
               fill_above=True, fill_below=False)
        _track(axs[3], xs_all, cf_helix,       "#43aa8b",  1.0, "Helix P\u03b1",
               fill_above=True, fill_below=False)
        axs[3].set_xlabel("Residue Position", fontsize=tick_font - 1, color="#4a5568")
        return fig

    @staticmethod
    def create_disorder_profile(disorder_scores, label_font=14, tick_font=12):
        """IUPred-style per-residue disorder score plot."""
        n  = len(disorder_scores)
        xs = list(range(1, n + 1))
        fig = Figure(figsize=(9, 4), dpi=120)
        fig.set_facecolor("#ffffff")
        ax  = fig.add_subplot(111)
        ax.fill_between(xs, disorder_scores, 0.5,
                        where=[v > 0.5 for v in disorder_scores],
                        alpha=0.28, color="#f3722c", interpolate=True,
                        label="Disordered (> 0.5)")
        ax.fill_between(xs, disorder_scores, 0.5,
                        where=[v <= 0.5 for v in disorder_scores],
                        alpha=0.12, color="#4361ee", interpolate=True,
                        label="Ordered (\u2264 0.5)")
        ax.plot(xs, disorder_scores, color="#f3722c", linewidth=1.8,
                marker="o", markersize=3.0, markeredgewidth=0, zorder=4)
        ax.axhline(0.5, color="#888", linewidth=1.0, linestyle="--",
                   zorder=3, label="Threshold (0.5)")
        _pub_style_ax(ax,
                      title="Disorder Profile (IUPred-inspired)",
                      xlabel="Residue Position", ylabel="Disorder Score",
                      grid=True, title_size=label_font + 1,
                      label_size=label_font - 1, tick_size=tick_font - 1)
        ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=tick_font - 2, framealpha=0.85, edgecolor="#d0d4e0")
        fig.tight_layout(pad=1.5)
        mplcursors.cursor(ax)
        return fig


    @staticmethod
    def create_tm_topology_figure(seq: str, helices: list, label_font=14, tick_font=12):
        """Simplified transmembrane topology diagram (snake-plot style)."""
        n   = len(seq)
        fig = Figure(figsize=(max(9, n * 0.06), 4.5), dpi=120)
        fig.set_facecolor("#ffffff")
        ax  = fig.add_subplot(111)
        ax.set_facecolor("#fafbff")
        # Membrane band
        ax.axhspan(-0.5, 0.5, alpha=0.12, color="#f59e0b")
        ax.text(2, 0.65,  "Extracellular", fontsize=tick_font - 2, color="#6b7280", style="italic")
        ax.text(2, -0.85, "Cytoplasmic",   fontsize=tick_font - 2, color="#6b7280", style="italic")
        # Draw backbone segments and TM rectangles
        side     = 1   # +1 = extracellular, -1 = cytoplasmic
        prev_end = 0
        for h in helices:
            s, e = h["start"], h["end"]
            y    = side * 1.15
            if s > prev_end:
                ax.plot([prev_end + 1, s], [y, y],
                        color="#4361ee", linewidth=1.8, solid_capstyle="round", zorder=3)
            rect = Rectangle((s + 1, -0.5), e - s, 1.0,
                              color="#4361ee", alpha=0.75, zorder=4, linewidth=0)
            ax.add_patch(rect)
            mid = (s + e) / 2 + 1
            ax.text(mid, 0, f"{s+1}–{e+1}",
                    ha="center", va="center",
                    fontsize=max(5, tick_font - 5), color="white", fontweight="bold", zorder=5)
            side     = -side
            prev_end = e
        # Final loop
        y = side * 1.15
        ax.plot([prev_end + 1, n], [y, y],
                color="#4361ee", linewidth=1.8, solid_capstyle="round", zorder=3)
        _pub_style_ax(ax,
                      title=f"TM Topology  ({len(helices)} predicted helix/es)",
                      xlabel="Residue Position", ylabel="",
                      grid=False, title_size=label_font + 1,
                      label_size=label_font - 1, tick_size=tick_font - 1)
        ax.set_xlim(0, n + 2)
        ax.set_ylim(-1.6, 1.8)
        ax.set_yticks([])
        ax.legend(handles=[Patch(color="#f59e0b", alpha=0.3, label="Membrane"),
                            Patch(color="#4361ee", alpha=0.75, label="TM helix")],
                  fontsize=tick_font - 3, framealpha=0.85, edgecolor="#d0d4e0",
                  loc="upper right")
        fig.tight_layout(pad=1.5)
        return fig

    @staticmethod
    def create_plddt_figure(plddt: list, label_font=14, tick_font=12):
        """Per-residue pLDDT confidence score with coloured confidence zones."""
        n   = len(plddt)
        xs  = list(range(1, n + 1))
        fig = Figure(figsize=(9, 4), dpi=120)
        fig.set_facecolor("#ffffff")
        ax  = fig.add_subplot(111)
        # Confidence zone bands
        ax.axhspan(90, 100, alpha=0.07, color="#0053D6")
        ax.axhspan(70,  90, alpha=0.07, color="#65CBF3")
        ax.axhspan(50,  70, alpha=0.07, color="#FFDB13")
        ax.axhspan( 0,  50, alpha=0.07, color="#FF7D45")
        cmap = plt.get_cmap("RdYlBu")
        norm = mcolors.Normalize(vmin=0, vmax=100)
        for i in range(n - 1):
            ax.plot([xs[i], xs[i + 1]], [plddt[i], plddt[i + 1]],
                    color=cmap(norm((plddt[i] + plddt[i + 1]) / 2)),
                    linewidth=1.8, zorder=4, solid_capstyle="round")
        for thresh, col, lbl in [
            (90, "#0053D6", ">90 Very high"),
            (70, "#65CBF3", "70–90 Confident"),
            (50, "#FFDB13", "50–70 Low"),
        ]:
            ax.axhline(thresh, color=col, linewidth=0.8, linestyle="--", alpha=0.8)
        _pub_style_ax(ax,
                      title="pLDDT / B-factor Confidence",
                      xlabel="Residue Position", ylabel="pLDDT Score",
                      grid=False, title_size=label_font + 1,
                      label_size=label_font - 1, tick_size=tick_font - 1)
        ax.set_ylim(0, 100)
        ax.set_xlim(1, n)
        ax.legend(handles=[
            Patch(color="#0053D6", alpha=0.5, label=">90  Very high"),
            Patch(color="#65CBF3", alpha=0.5, label="70–90  Confident"),
            Patch(color="#FFDB13", alpha=0.5, label="50–70  Low"),
            Patch(color="#FF7D45", alpha=0.5, label="<50  Very low"),
        ], fontsize=tick_font - 3, framealpha=0.85, edgecolor="#d0d4e0", loc="lower right")
        fig.tight_layout(pad=1.5)
        mplcursors.cursor(ax)
        return fig

    @staticmethod
    def create_distance_map_figure(dist_matrix: np.ndarray, label_font=14, tick_font=12):
        """Cα pairwise distance heatmap from a loaded PDB structure.
        Cells ≤8 Å are highlighted as contacts."""
        n   = dist_matrix.shape[0]
        fig = Figure(figsize=(6.5, 5.5), dpi=120)
        fig.set_facecolor("#ffffff")
        ax  = fig.add_subplot(111)
        ax.set_facecolor("#fafbff")
        im   = ax.imshow(dist_matrix, cmap="viridis_r", aspect="auto",
                         origin="upper", interpolation="nearest",
                         vmin=0, vmax=min(40, dist_matrix.max()))
        cbar = fig.colorbar(im, ax=ax, shrink=0.85, aspect=20, pad=0.02)
        cbar.set_label("Cα distance (Å)", fontsize=tick_font - 1, color="#4a5568")
        cbar.ax.tick_params(labelsize=tick_font - 2, colors="#4a5568")
        # Overlay 8 Å contact threshold as a contour
        ax.contour(dist_matrix, levels=[8.0], colors=["#f72585"],
                   linewidths=[0.6], alpha=0.7)
        _pub_style_ax(ax,
                      title=f"Cα Distance Map  ({n} residues)  — pink contour = 8 Å contact",
                      xlabel="Residue Position", ylabel="Residue Position",
                      grid=False, title_size=label_font,
                      label_size=label_font - 1, tick_size=tick_font - 1)
        fig.tight_layout(pad=1.5)
        return fig

    @staticmethod
    def create_domain_architecture_figure(seq_len: int, domains: list,
                                          seq: str = "",
                                          disorder_scores=None,
                                          tm_helices=None,
                                          label_font=14, tick_font=12):
        """Multi-track domain architecture figure.

        Tracks (top→bottom, only added when data is available):
          • Pfam/InterPro domains  — when *domains* is non-empty
          • Disordered regions     — when *disorder_scores* provided
          • Low-complexity regions — when *seq* provided
          • TM helices             — when *tm_helices* is non-empty
        """
        # ---- build per-residue LC mask ----------------------------------------
        def _lc_mask(s, window=12, thr=2.0):
            n = len(s)
            covered = [False] * n
            for i in range(n - min(window, n) + 1):
                win = s[i:i + window]
                counts = {}
                for aa in win:
                    counts[aa] = counts.get(aa, 0) + 1
                L = len(win)
                ent = -sum((c / L) * log2(c / L) for c in counts.values() if c > 0)
                if ent < thr:
                    for j in range(i, min(i + window, n)):
                        covered[j] = True
            return covered

        # ---- collect run-length-encoded regions (1-based inclusive) -----------
        def _runs(bools):
            segs, in_seg, start = [], False, 0
            for i, v in enumerate(bools):
                if v and not in_seg:
                    in_seg, start = True, i + 1      # 1-based start
                elif not v and in_seg:
                    segs.append((start, i))           # i is 1-based end (exclusive→inclusive)
                    in_seg = False
            if in_seg:
                segs.append((start, len(bools)))
            return segs

        # ---- build track list -------------------------------------------------
        # Each entry: (label, colour, regions, pfam_meta)
        # regions: list of (s1, e1) 1-based inclusive
        # pfam_meta: list of domain dicts (only for Pfam track), else None
        tracks = []

        if domains:
            pfam_regions = [(d["start"], d["end"]) for d in domains]
            tracks.append(("Pfam Domains", None, pfam_regions, domains))

        if disorder_scores is not None and len(disorder_scores) == seq_len:
            dis_runs = _runs([v > 0.5 for v in disorder_scores])
            tracks.append(("Disorder", "#f3722c", dis_runs, None))

        if seq and len(seq) == seq_len:
            lc_runs = _runs(_lc_mask(seq))
            tracks.append(("Low Complexity", "#a8dadc", lc_runs, None))

        if tm_helices:
            # predict_tm_helices returns 0-based start/end (inclusive)
            tm_regions = [(h["start"] + 1, h["end"] + 1) for h in tm_helices]
            tracks.append(("TM Helices", "#6a4c93", tm_regions, None))

        # Ensure at least one track (backbone only) so the graph always renders
        if not tracks:
            tracks.append(("Sequence", "#94a3b8", [], None))

        n_tracks = len(tracks)
        fig_h    = max(2.5, 1.2 + n_tracks * 1.0)
        fig = Figure(figsize=(10, fig_h), dpi=120)
        fig.set_facecolor("#ffffff")
        ax  = fig.add_subplot(111)
        ax.set_facecolor("#fafbff")

        # y positions: top track at y = n_tracks-1, bottom track at y = 0
        track_ys      = list(range(n_tracks - 1, -1, -1))
        half          = 0.32
        legend_patches = []

        for tidx, (label, colour, regions, meta) in enumerate(tracks):
            ty = track_ys[tidx]
            # Backbone line
            ax.plot([1, seq_len], [ty, ty], color="#cbd5e0", linewidth=2.5,
                    solid_capstyle="round", zorder=2)
            # N/C terminus only on topmost track
            if tidx == 0:
                ax.text(1,        ty + half + 0.06, "N",
                        ha="center", fontsize=tick_font - 3, color="#718096")
                ax.text(seq_len,  ty + half + 0.06, "C",
                        ha="center", fontsize=tick_font - 3, color="#718096")

            if meta is not None:
                # Pfam track: each domain gets a distinct palette colour
                for i, (dom, (s, e)) in enumerate(zip(meta, regions)):
                    col  = _PALETTE[i % len(_PALETTE)]
                    w    = e - s + 1          # inclusive width
                    rect = Rectangle((s, ty - half), w, 2 * half,
                                     color=col, alpha=0.85, zorder=4, linewidth=0)
                    ax.add_patch(rect)
                    mid = (s + e) / 2.0
                    ax.text(mid, ty, dom["name"][:14],
                            ha="center", va="center",
                            fontsize=max(5, tick_font - 5), color="white",
                            fontweight="bold", zorder=5)
                    legend_patches.append(Patch(color=col, label=dom["name"]))
            else:
                for (s, e) in regions:
                    w    = e - s + 1          # inclusive width
                    rect = Rectangle((s, ty - half), w, 2 * half,
                                     color=colour, alpha=0.80, zorder=4, linewidth=0)
                    ax.add_patch(rect)
                if regions:
                    legend_patches.append(Patch(color=colour, label=label))

        # Y-axis tick labels as track names (lets matplotlib manage placement)
        ax.set_yticks(track_ys)
        ax.set_yticklabels([t[0] for t in tracks],
                           fontsize=max(6, tick_font - 3), color="#4a5568",
                           fontweight="600")
        ax.tick_params(axis="y", length=0, pad=6)

        _pub_style_ax(ax,
                      title="Domain Architecture",
                      xlabel="Residue Position", ylabel="",
                      grid=False, title_size=label_font + 1,
                      label_size=label_font - 1, tick_size=tick_font - 1)
        ax.set_xlim(1, seq_len)
        ax.set_ylim(-0.7, n_tracks - 0.3)
        # Keep ytick labels (override _pub_style_ax which may have cleared them)
        ax.set_yticks(track_ys)
        ax.set_yticklabels([t[0] for t in tracks],
                           fontsize=max(6, tick_font - 3), color="#4a5568",
                           fontweight="600")
        ax.tick_params(axis="y", length=0, pad=6)

        if legend_patches:
            ax.legend(handles=legend_patches,
                      fontsize=max(6, tick_font - 4), framealpha=0.85,
                      edgecolor="#d0d4e0", loc="upper right",
                      ncol=max(1, len(legend_patches) // 6))
        fig.tight_layout(pad=1.5)
        return fig

    @staticmethod
    def create_uversky_phase_plot(seq: str, label_font=14, tick_font=12):
        """Uversky charge-hydrophobicity phase diagram.

        The sequence is plotted as a single point in the
        mean |net charge| vs mean hydrophobicity space.
        Phase boundary from Uversky et al. 2000:
          <H> = 2.785 * <|R|> + 0.446
        Region above boundary → IDP/disordered
        Region below boundary → compact/folded
        """
        n = len(seq)
        if n == 0:
            fig = Figure(figsize=(6, 5), dpi=120)
            return fig

        pos_n = sum(1 for aa in seq if aa in "KR")
        neg_n = sum(1 for aa in seq if aa in "DE")
        mean_charge = abs(pos_n - neg_n) / n
        mean_hydro  = sum(KYTE_DOOLITTLE.get(aa, 0.0) for aa in seq) / n
        # Normalise hydrophobicity to 0–1 scale (KD range: -4.5 to 4.5)
        h_norm = (mean_hydro + 4.5) / 9.0

        fig = Figure(figsize=(6, 5), dpi=120)
        fig.set_facecolor("#ffffff")
        ax  = fig.add_subplot(111)
        ax.set_facecolor("#fafbff")

        # Draw phase boundary line: H_norm = 2.785 * |R| + 0.446
        r_vals = np.linspace(0, 0.5, 200)
        h_boundary = 2.785 * r_vals + 0.446
        # Clip to [0, 1]
        h_boundary = np.clip(h_boundary, 0, 1)
        ax.plot(r_vals, h_boundary, color="#374151", linewidth=1.8,
                linestyle="--", label="Uversky boundary", zorder=3)

        # Shade regions
        ax.fill_between(r_vals, h_boundary, 1.0, alpha=0.10,
                        color="#4361ee", label="Ordered / compact")
        ax.fill_between(r_vals, 0, h_boundary, alpha=0.10,
                        color="#f72585", label="Disordered / IDP")

        # Annotate regions
        ax.text(0.05, 0.75, "Ordered / Folded", fontsize=tick_font - 3,
                color="#4361ee", alpha=0.8, style="italic")
        ax.text(0.25, 0.15, "Disordered / IDP", fontsize=tick_font - 3,
                color="#f72585", alpha=0.8, style="italic")

        # Plot the sequence point
        region = "IDP" if h_norm < (2.785 * mean_charge + 0.446) else "Ordered"
        pt_color = "#f72585" if region == "IDP" else "#4361ee"
        ax.scatter([mean_charge], [h_norm], color=pt_color, s=120, zorder=5,
                   edgecolors="white", linewidths=1.2)
        ax.annotate(f"  ({mean_charge:.3f}, {h_norm:.3f})\n  → {region}",
                    xy=(mean_charge, h_norm),
                    fontsize=tick_font - 3, color=pt_color,
                    xytext=(mean_charge + 0.02, h_norm + 0.04))

        _pub_style_ax(ax,
                      title="Uversky Charge–Hydrophobicity Phase Plot",
                      xlabel="Mean |Net Charge| per residue",
                      ylabel="Mean Hydrophobicity (normalised 0–1)",
                      grid=True, title_size=label_font + 1,
                      label_size=label_font - 1, tick_size=tick_font - 1)
        ax.set_xlim(0, 0.5)
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=tick_font - 3, framealpha=0.85,
                  edgecolor="#d0d4e0", loc="upper right")
        fig.tight_layout(pad=1.5)
        return fig

    @staticmethod
    def create_coiled_coil_profile(cc_profile: list, label_font=14, tick_font=12):
        """Per-residue coiled-coil propensity profile."""
        n  = len(cc_profile)
        xs = list(range(1, n + 1))

        fig = Figure(figsize=(9, 4), dpi=120)
        fig.set_facecolor("#ffffff")
        ax  = fig.add_subplot(111)
        ax.set_facecolor("#fafbff")

        ax.plot(xs, cc_profile, color=_ACCENT, linewidth=1.4, zorder=3)
        ax.fill_between(xs, cc_profile, 0.5,
                        where=[v > 0.5 for v in cc_profile],
                        alpha=0.22, color=_ACCENT, zorder=2,
                        label="Coiled-coil region (>0.50)")
        ax.axhline(0.5, color="#374151", linestyle="--", linewidth=0.9,
                   alpha=0.7, label="Threshold 0.50", zorder=4)
        _pub_style_ax(ax,
                      title="Coiled-Coil Propensity Profile",
                      xlabel="Residue Position",
                      ylabel="Coiled-Coil Score",
                      grid=True, title_size=label_font + 1,
                      label_size=label_font - 1, tick_size=tick_font - 1)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=tick_font - 3, framealpha=0.85,
                  edgecolor="#d0d4e0", loc="upper right")
        fig.tight_layout(pad=1.5)
        mplcursors.cursor(ax)
        return fig

    @staticmethod
    def create_saturation_mutagenesis_figure(seq: str, label_font=14, tick_font=12):
        """In silico saturation mutagenesis heatmap.

        For every position and every amino acid substitution, computes a
        combined perturbation score:
            Δscore = |ΔGRAVY| + |ΔNCPR|
        where ΔGRAVY and ΔNCPR are the changes in global GRAVY and NCPR
        relative to the wild-type sequence.

        The heatmap rows = 20 amino acids (sorted by hydrophobicity),
        columns = sequence positions.  Wild-type residue cells are marked
        with a dot.
        """
        n   = len(seq)
        if n == 0 or n > 500:
            # Too long for a meaningful dense heatmap — return placeholder
            fig = Figure(figsize=(9, 5), dpi=120)
            ax  = fig.add_subplot(111)
            ax.text(0.5, 0.5,
                    "Sequence too long for saturation mutagenesis\n(limit: 500 aa)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=label_font - 2, color="#718096")
            ax.set_axis_off()
            return fig

        AAS_BY_HYDRO = list("RNDQEKSHPYTGACMWFLIV")  # approx. hydrophobicity order (polar→hydrophobic)
        wt_gravy = sum(KYTE_DOOLITTLE.get(aa, 0.0) for aa in seq) / n
        pos_n    = sum(1 for aa in seq if aa in "KR")
        neg_n    = sum(1 for aa in seq if aa in "DE")
        wt_ncpr  = (pos_n - neg_n) / n

        mat = np.zeros((20, n), dtype=float)
        for col_i, pos in enumerate(range(n)):
            wt_aa = seq[pos]
            wt_kd = KYTE_DOOLITTLE.get(wt_aa, 0.0)
            wt_is_pos = 1 if wt_aa in "KR" else 0
            wt_is_neg = 1 if wt_aa in "DE" else 0
            for row_i, mut_aa in enumerate(AAS_BY_HYDRO):
                if mut_aa == wt_aa:
                    mat[row_i, col_i] = 0.0
                    continue
                mut_kd  = KYTE_DOOLITTLE.get(mut_aa, 0.0)
                delta_gravy = (mut_kd - wt_kd) / n
                mut_is_pos  = 1 if mut_aa in "KR" else 0
                mut_is_neg  = 1 if mut_aa in "DE" else 0
                delta_ncpr  = ((mut_is_pos - wt_is_pos) - (mut_is_neg - wt_is_neg)) / n
                mat[row_i, col_i] = abs(delta_gravy) + abs(delta_ncpr)

        fig = Figure(figsize=(max(9, n * 0.08 + 1), 5.5), dpi=120)
        fig.set_facecolor("#ffffff")
        ax  = fig.add_subplot(111)

        im = ax.imshow(mat, aspect="auto", cmap="hot_r", origin="upper",
                       interpolation="nearest",
                       vmin=0, vmax=np.percentile(mat[mat > 0], 95) if mat.max() > 0 else 1)
        cbar = fig.colorbar(im, ax=ax, shrink=0.85, aspect=20, pad=0.02)
        cbar.set_label("|ΔGRAVY| + |ΔNCPR|", fontsize=tick_font - 1, color="#4a5568")
        cbar.ax.tick_params(labelsize=tick_font - 2, colors="#4a5568")

        # Mark wild-type positions with a small dot
        for col_i, wt_aa in enumerate(seq):
            if wt_aa in AAS_BY_HYDRO:
                row_i = AAS_BY_HYDRO.index(wt_aa)
                ax.plot(col_i, row_i, "w.", markersize=3, alpha=0.8)

        ax.set_yticks(range(20))
        ax.set_yticklabels(AAS_BY_HYDRO, fontsize=max(6, tick_font - 4))
        ax.set_xlabel("Residue Position", fontsize=label_font - 1, color="#4a5568")
        ax.set_ylabel("Substitution", fontsize=label_font - 1, color="#4a5568")
        ax.set_title("Saturation Mutagenesis  (|ΔGRAVY| + |ΔNCPR|)",
                     fontsize=label_font, fontweight="bold", color="#1a1a2e", pad=8)
        ax.tick_params(axis="x", labelsize=tick_font - 2)
        fig.tight_layout(pad=1.5)
        return fig

    # ── New graph wrappers ────────────────────────────────────────────────────

    @staticmethod
    def create_aggregation_profile_figure(seq, aggregation_profile, hotspots,
                                          label_font=14, tick_font=12):
        if _HAS_NEW_GRAPHS:
            return create_aggregation_profile_figure(
                seq, aggregation_profile, hotspots,
                label_font=label_font, tick_font=tick_font)
        fig = Figure(figsize=(9, 3), dpi=120)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "beer.graphs.new_graphs not installed",
                ha="center", va="center", transform=ax.transAxes, color="#718096")
        return fig

    @staticmethod
    def create_solubility_profile_figure(seq, camsolmt_profile,
                                         label_font=14, tick_font=12):
        if _HAS_NEW_GRAPHS:
            return create_solubility_profile_figure(
                seq, camsolmt_profile, label_font=label_font, tick_font=tick_font)
        fig = Figure(figsize=(9, 3), dpi=120)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "beer.graphs.new_graphs not installed",
                ha="center", va="center", transform=ax.transAxes, color="#718096")
        return fig

    @staticmethod
    def create_hydrophobic_moment_figure(seq, moment_alpha, moment_beta,
                                         amphipathic_regions,
                                         label_font=14, tick_font=12):
        if _HAS_NEW_GRAPHS:
            return create_hydrophobic_moment_figure(
                seq, moment_alpha, moment_beta, amphipathic_regions,
                label_font=label_font, tick_font=tick_font)
        fig = Figure(figsize=(9, 3), dpi=120)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "beer.graphs.new_graphs not installed",
                ha="center", va="center", transform=ax.transAxes, color="#718096")
        return fig

    @staticmethod
    def create_pI_MW_gel_figure(proteins_data, label_font=14, tick_font=12):
        if _HAS_NEW_GRAPHS:
            return create_pI_MW_gel_figure(
                proteins_data, label_font=label_font, tick_font=tick_font)
        fig = Figure(figsize=(7, 5), dpi=120)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "beer.graphs.new_graphs not installed",
                ha="center", va="center", transform=ax.transAxes, color="#718096")
        return fig

    @staticmethod
    def create_ptm_profile_figure(seq, ptm_sites, label_font=14, tick_font=12):
        if _HAS_NEW_GRAPHS:
            return create_ptm_profile_figure(
                seq, ptm_sites, label_font=label_font, tick_font=tick_font)
        fig = Figure(figsize=(9, 4), dpi=120)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "beer.graphs.new_graphs not installed",
                ha="center", va="center", transform=ax.transAxes, color="#718096")
        return fig

    @staticmethod
    def create_rbp_profile_figure(seq, rbp_profile, rbp_motifs,
                                   label_font=14, tick_font=12):
        if _HAS_NEW_GRAPHS:
            return create_rbp_profile_figure(
                seq, rbp_profile, rbp_motifs,
                label_font=label_font, tick_font=tick_font)
        fig = Figure(figsize=(9, 3), dpi=120)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "beer.graphs.new_graphs not installed",
                ha="center", va="center", transform=ax.transAxes, color="#718096")
        return fig

    @staticmethod
    def create_scd_profile_figure(seq, scd_profile, window,
                                   label_font=14, tick_font=12):
        if _HAS_NEW_GRAPHS:
            return create_scd_profile_figure(
                seq, scd_profile, window,
                label_font=label_font, tick_font=tick_font)
        fig = Figure(figsize=(9, 3), dpi=120)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "beer.graphs.new_graphs not installed",
                ha="center", va="center", transform=ax.transAxes, color="#718096")
        return fig

    @staticmethod
    def create_truncation_series_figure(truncation_data,
                                         label_font=14, tick_font=12):
        if _HAS_NEW_GRAPHS:
            return create_truncation_series_figure(
                truncation_data, label_font=label_font, tick_font=tick_font)
        fig = Figure(figsize=(12, 8), dpi=120)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "beer.graphs.new_graphs not installed",
                ha="center", va="center", transform=ax.transAxes, color="#718096")
        return fig

    @staticmethod
    def create_ramachandran_figure(phi_psi_data, label_font=14, tick_font=12):
        if _HAS_NEW_GRAPHS:
            return create_ramachandran_figure(
                phi_psi_data, label_font=label_font, tick_font=tick_font)
        fig = Figure(figsize=(6, 6), dpi=120)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "beer.graphs.new_graphs not installed\nor no PDB structure loaded",
                ha="center", va="center", transform=ax.transAxes, color="#718096")
        ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
        ax.axhline(0, color="#aaa", lw=0.5); ax.axvline(0, color="#aaa", lw=0.5)
        ax.set_xlabel("φ (°)"); ax.set_ylabel("ψ (°)")
        ax.set_title("Ramachandran Plot")
        return fig

    @staticmethod
    def create_contact_network_figure(seq, dist_matrix, cutoff=8.0,
                                       label_font=14, tick_font=12):
        if _HAS_NEW_GRAPHS:
            return create_contact_network_figure(
                seq, dist_matrix, cutoff_angstrom=cutoff,
                label_font=label_font, tick_font=tick_font)
        fig = Figure(figsize=(7, 7), dpi=120)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "beer.graphs.new_graphs not installed\nor no PDB structure loaded",
                ha="center", va="center", transform=ax.transAxes, color="#718096")
        return fig

    @staticmethod
    def create_msa_conservation_figure(sequences, names,
                                        label_font=14, tick_font=12):
        if _HAS_NEW_GRAPHS:
            return create_msa_conservation_figure(
                sequences, names, label_font=label_font, tick_font=tick_font)
        fig = Figure(figsize=(10, 4), dpi=120)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "beer.graphs.new_graphs not installed",
                ha="center", va="center", transform=ax.transAxes, color="#718096")
        return fig

    @staticmethod
    def create_complex_mw_figure(chains_data, stoichiometry_str,
                                   label_font=14, tick_font=12):
        if _HAS_NEW_GRAPHS:
            return create_complex_mw_figure(
                chains_data, stoichiometry_str,
                label_font=label_font, tick_font=tick_font)
        fig = Figure(figsize=(7, 5), dpi=120)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "beer.graphs.new_graphs not installed",
                ha="center", va="center", transform=ax.transAxes, color="#718096")
        return fig


# --- Export ---

class ExportTools:
    @staticmethod
    def _generate_full_html(analysis_data, seq_name=""):
        """Generate HTML for the PDF report (text/tables only – graphs are saved separately)."""
        if not analysis_data or "report_sections" not in analysis_data:
            return "<p>No analysis data available.</p>"

        header_name = seq_name or "Protein Sequence"
        seq   = analysis_data.get("seq", "")
        seq_block = format_sequence_block(seq, name=seq_name)
        # Escape for HTML pre block
        seq_block_html = seq_block.replace("&", "&amp;").replace("<", "&lt;")

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
{REPORT_CSS}
@page {{ margin: 18mm 20mm; }}
.page-break {{ page-break-after: always; }}
.seq-block {{
    font-family: 'Courier New', monospace;
    font-size: 9.5pt;
    background: #f8f9fd;
    border: 1px solid #e8eaf0;
    border-radius: 4px;
    padding: 10px 14px;
    line-height: 2.0;
    color: #1a1a2e;
    white-space: pre;
    margin-bottom: 14px;
    overflow-x: auto;
}}
</style>
</head><body>
<h1>BEER Analysis Report</h1>
<p style="color:#718096;font-size:10pt;">
  Sequence: <strong>{header_name}</strong>
  &nbsp;&bull;&nbsp; Length: <strong>{len(seq)} aa</strong>
</p>
<h2 style="margin-top:16px;">Sequence</h2>
<div class="seq-block">{seq_block_html}</div>
<div class="page-break"></div>
"""
        for sec in REPORT_SECTIONS:
            content = analysis_data["report_sections"].get(sec, "")
            # Strip inline <style> blocks already embedded per-section to avoid duplication
            content = re.sub(r"<style>[^<]*</style>", "", content, flags=re.DOTALL)
            html += content + "\n"

        html += "</body></html>"
        return html

    @staticmethod
    def export_pdf(analysis_data, file_name, parent, seq_name=""):
        try:
            printer = QPrinter(QPrinter.HighResolution)
            printer.setOutputFormat(QPrinter.PdfFormat)
            printer.setOutputFileName(file_name)
            browser = QTextBrowser()
            browser.setHtml(ExportTools._generate_full_html(analysis_data, seq_name))
            browser.document().print_(printer)
            QMessageBox.information(parent, "Success", f"PDF exported to {file_name}")
        except Exception as e:
            QMessageBox.warning(parent, "Export Failed", f"PDF export error: {e}")

# --- PDB import ---

def import_pdb_sequence(file_name: str) -> dict:
    """Extract one-letter sequences for each chain in a PDB."""
    try:
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("pdb", file_name)
        chains = {}
        model  = next(struct.get_models())
        for chain in model:
            seq = ""
            for res in chain:
                if is_aa(res, standard=True):
                    try:
                        seq += seq1(res.get_resname())
                    except KeyError:
                        continue
            if seq:
                chains[chain.id] = seq
        return chains
    except Exception as e:
        raise RuntimeError(f"PDB parse error: {e}") from e

def extract_chain_structures(pdb_str: str) -> dict:
    """Return per-chain structure dicts keyed by chain ID.

    Each value is ``{"pdb_str": str, "plddt": list, "dist_matrix": ndarray}``.
    Uses PDBIO to write each chain individually so that pLDDT extraction and
    distance-matrix computation always operate on a single, correctly-sized chain.
    """
    class _ChainSelect(PDBSelect):
        def __init__(self, cid):
            self._cid = cid
        def accept_chain(self, chain):
            return chain.id == self._cid

    parser = PDBParser(QUIET=True)
    struct  = parser.get_structure("x", StringIO(pdb_str))
    result  = {}
    model   = next(struct.get_models())
    io      = PDBIO()
    io.set_structure(struct)
    for chain in model:
        buf = StringIO()
        io.save(buf, _ChainSelect(chain.id))
        chain_pdb = buf.getvalue()
        if not chain_pdb.strip():
            continue
        plddt = extract_plddt_from_pdb(chain_pdb)
        dm    = compute_ca_distance_matrix(chain_pdb)
        result[chain.id] = {"pdb_str": chain_pdb, "plddt": plddt, "dist_matrix": dm}
    return result


# --- Batch stats helper ---

def _calc_batch_stats(seq: str, data: dict) -> tuple:
    """Return (hydro%, hydrophil%, pos%, neg%, neu%) for a sequence."""
    length = len(seq)
    hydro  = sum(1 for aa in seq if KYTE_DOOLITTLE.get(aa, 0.0) > 0) / length * 100
    pos    = sum(data["aa_counts"].get(k, 0) for k in ("K", "R", "H")) / length * 100
    neg    = sum(data["aa_counts"].get(k, 0) for k in ("D", "E")) / length * 100
    neu    = 100 - (pos + neg)
    return hydro, 100 - hydro, pos, neg, neu

# --- Worker thread ---

class AnalysisWorker(QThread):
    """Non-blocking analysis in a QThread.  Emits finished(dict) or error(str)."""
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, seq, pH, window_size, use_reducing, pka):
        super().__init__()
        self.seq          = seq
        self.pH           = pH
        self.window_size  = window_size
        self.use_reducing = use_reducing
        self.pka          = pka

    def run(self):
        try:
            data = AnalysisTools.analyze_sequence(
                self.seq, self.pH, self.window_size,
                self.use_reducing, self.pka
            )
            self.finished.emit(data)
        except Exception as exc:
            self.error.emit(str(exc))


# --- AlphaFold worker ---

class AlphaFoldWorker(QThread):
    """Fetch AlphaFold predicted structure for a UniProt accession.
    Emits finished(dict) with keys: pdb_str, plddt, dist_matrix, accession."""
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, accession: str):
        super().__init__()
        self.accession = accession.strip().upper()

    def run(self):
        try:
            self.progress.emit(f"Querying AlphaFold for {self.accession}…")
            meta_url = f"https://alphafold.ebi.ac.uk/api/prediction/{self.accession}"
            req = urllib.request.Request(meta_url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                meta = json.loads(r.read().decode())
            if not meta:
                self.error.emit(f"No AlphaFold prediction found for {self.accession}.")
                return
            pdb_url = meta[0]["pdbUrl"]
            self.progress.emit("Downloading PDB structure…")
            with urllib.request.urlopen(pdb_url, timeout=60) as r:
                pdb_str = r.read().decode()
            self.progress.emit("Extracting pLDDT and distance matrix…")
            plddt       = extract_plddt_from_pdb(pdb_str)
            dist_matrix = compute_ca_distance_matrix(pdb_str)
            self.finished.emit({
                "pdb_str":     pdb_str,
                "plddt":       plddt,
                "dist_matrix": dist_matrix,
                "accession":   self.accession,
            })
        except Exception as exc:
            self.error.emit(f"AlphaFold fetch failed: {exc}")


# --- Pfam worker ---

class PfamWorker(QThread):
    """Fetch Pfam domain annotations for a UniProt accession via InterPro REST API."""
    finished = pyqtSignal(list)
    error    = pyqtSignal(str)

    def __init__(self, accession: str):
        super().__init__()
        self.accession = accession.strip().upper()

    def run(self):
        try:
            url = (
                f"https://www.ebi.ac.uk/interpro/api/entry/pfam"
                f"/protein/uniprot/{self.accession}/?page_size=100"
            )
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read().decode())
            domains = []
            for result in data.get("results", []):
                meta = result.get("metadata", {})
                raw_name = meta.get("name", meta.get("accession", "Unknown"))
                name = raw_name.get("name", "") if isinstance(raw_name, dict) else str(raw_name)
                acc  = meta.get("accession", "")
                for prot in result.get("proteins", []):
                    for loc in prot.get("entry_protein_locations", []):
                        for frag in loc.get("fragments", []):
                            domains.append({
                                "name":      name,
                                "accession": acc,
                                "start":     frag["start"],
                                "end":       frag["end"],
                            })
            domains.sort(key=lambda d: d["start"])
            self.finished.emit(domains)
        except Exception as exc:
            self.error.emit(f"Pfam fetch failed: {exc}")


# --- BLAST worker ---

class BlastWorker(QThread):
    """Run NCBI blastp and return top hits. Can take 1-3 minutes."""
    finished = pyqtSignal(list)
    error    = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, seq: str, database: str = "nr", hitlist_size: int = 20):
        super().__init__()
        self.seq          = seq
        self.database     = database
        self.hitlist_size = hitlist_size

    def run(self):
        try:
            self.progress.emit("Submitting BLAST search (this may take 1–3 min)…")
            result_handle = NCBIWWW.qblast(
                "blastp", self.database, self.seq,
                hitlist_size=self.hitlist_size,
            )
            self.progress.emit("Parsing BLAST results…")
            blast_record = NCBIXML.read(result_handle)
            hits = []
            for aln in blast_record.alignments[:self.hitlist_size]:
                hsp = aln.hsps[0]
                hits.append({
                    "accession": aln.accession,
                    "title":     aln.title[:100],
                    "length":    aln.length,
                    "score":     hsp.score,
                    "e_value":   hsp.expect,
                    "identity":  hsp.identities / hsp.align_length * 100,
                    "subject":   hsp.sbjct.replace("-", ""),
                })
            self.finished.emit(hits)
        except ImportError:
            self.error.emit("Bio.Blast not available — install biopython.")
        except Exception as exc:
            self.error.emit(f"BLAST failed: {exc}")


# --- Mutation dialog ---

class MutationDialog(QDialog):
    """Simple dialog: pick a position and a replacement amino acid."""
    def __init__(self, seq: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Mutate Residue")
        self.setMinimumWidth(320)
        self._seq = seq
        layout = QFormLayout(self)
        layout.setSpacing(10)

        self.pos_spin = QSpinBox()
        self.pos_spin.setRange(1, len(seq))
        self.pos_spin.setValue(1)
        self.pos_spin.valueChanged.connect(self._update_current)
        layout.addRow("Position (1-based):", self.pos_spin)

        self.current_lbl = QLabel(seq[0] if seq else "?")
        self.current_lbl.setStyleSheet(
            "font-weight:700; color:#4361ee; font-family:monospace; font-size:14pt;"
        )
        layout.addRow("Current residue:", self.current_lbl)

        self.aa_combo = QComboBox()
        self.aa_combo.addItems(sorted(VALID_AMINO_ACIDS))
        layout.addRow("Replace with:", self.aa_combo)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)

    def _update_current(self, pos):
        aa = self._seq[pos - 1] if 0 <= pos - 1 < len(self._seq) else "?"
        self.current_lbl.setText(aa)

    def get_mutation(self):
        """Returns (position_0based, new_aa)."""
        return self.pos_spin.value() - 1, self.aa_combo.currentText()


# --- Figure Composer dialog ---

class _FigureComposerDialog(QDialog):
    """Dialog to build a multi-panel figure from existing graph canvases."""
    _LAYOUTS = ["1\u00d71", "1\u00d72", "2\u00d71", "2\u00d72", "2\u00d73", "3\u00d72", "3\u00d73"]

    def __init__(self, available_titles: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Figure Composer")
        self.setMinimumSize(560, 400)
        layout = QVBoxLayout(self)

        top = QHBoxLayout()
        top.addWidget(QLabel("Layout:"))
        self.layout_combo = QComboBox()
        self.layout_combo.addItems(self._LAYOUTS)
        self.layout_combo.setCurrentText("2\u00d72")
        self.layout_combo.currentTextChanged.connect(self._rebuild_slots)
        top.addWidget(self.layout_combo)
        top.addStretch()
        layout.addLayout(top)

        self._available = ["— None —"] + available_titles
        self._slots_frame = QWidget()
        self._slots_grid  = QGridLayout(self._slots_frame)
        layout.addWidget(self._slots_frame)

        self._slot_combos: list = []
        self._rebuild_slots(self.layout_combo.currentText())

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _rebuild_slots(self, layout_str: str):
        try:
            nr, nc = [int(x) for x in layout_str.split("\u00d7")]
        except Exception:
            nr, nc = 2, 2
        # Clear old
        for c in self._slot_combos:
            c.setParent(None)
        self._slot_combos.clear()
        for i in range(nr * nc):
            cb = QComboBox()
            cb.addItems(self._available)
            self._slot_combos.append(cb)
            self._slots_grid.addWidget(QLabel(f"Panel {chr(ord('A')+i)}:"), i // nc, (i % nc) * 2)
            self._slots_grid.addWidget(cb, i // nc, (i % nc) * 2 + 1)

    def get_composition(self):
        layout_str = self.layout_combo.currentText()
        titles = []
        for cb in self._slot_combos:
            t = cb.currentText()
            titles.append(None if t == "— None —" else t)
        return layout_str, titles


# --- Navigation sidebar widget ---

class NavTabWidget(QWidget):
    """Left-sidebar navigation that is a drop-in replacement for QTabWidget.
    Implements the subset of QTabWidget API used in this app."""

    _NAV_ICONS = {
        "Analysis":            "🧪",
        "Graphs":              "📊",
        "Structure":           "🔬",
        "BLAST":               "🔍",
        "Compare":             "⚖",
        "Multichain Analysis": "📋",
        "Truncation":          "✂",
        "MSA":                 "🔀",
        "Complex":             "⚛",
        "Settings":            "⚙",
        "Help":                "❓",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.nav_list = QListWidget()
        self.nav_list.setObjectName("nav_bar")
        self.nav_list.setFixedWidth(152)
        self.nav_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(self.nav_list)

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setFrameShadow(QFrame.Plain)
        sep.setObjectName("nav_sep")
        outer.addWidget(sep)

        self.stack = QStackedWidget()
        outer.addWidget(self.stack, 1)

        self.nav_list.currentRowChanged.connect(self.stack.setCurrentIndex)

    def addTab(self, widget: QWidget, name: str) -> int:
        icon = self._NAV_ICONS.get(name, "▸")
        item = QListWidgetItem(f"  {icon}  {name}")
        self.nav_list.addItem(item)
        idx = self.stack.addWidget(widget)
        if self.nav_list.count() == 1:
            self.nav_list.setCurrentRow(0)
        return idx

    def setCurrentIndex(self, idx: int):
        self.nav_list.setCurrentRow(idx)

    def currentIndex(self) -> int:
        return self.nav_list.currentRow()

    def currentWidget(self) -> QWidget:
        return self.stack.currentWidget()

    def widget(self, idx: int) -> QWidget:
        return self.stack.widget(idx)

    def count(self) -> int:
        return self.stack.count()


# --- Main GUI ---

class ProteinAnalyzerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BEER - Biochemical Estimator & Explorer of Residues")
        self.resize(1200, 900)
        self.setStyleSheet(LIGHT_THEME_CSS)

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

        # State
        self.analysis_data       = None
        self.batch_data          = []
        self.default_window_size = 9
        self.default_pH          = 7.0
        self.show_bead_labels    = True
        self.label_font_size     = 14
        self.tick_font_size      = 12
        self.marker_size         = 10
        self.colormap            = "coolwarm"
        self.graph_color         = "#4361ee"
        self.show_heading        = True
        self.show_grid           = True
        self.default_graph_format = "PNG"
        self.enable_tooltips     = True
        self.use_reducing        = False
        self.custom_pka          = None
        self.sequence_name       = ""
        self.app_font_size       = 12
        self._tooltips: dict     = {}
        self.transparent_bg      = True
        self._analysis_worker    = None
        self._history: list      = []   # list of (name, seq)

        # --- New state for AlphaFold / Pfam / BLAST ---
        self.current_accession   = ""   # last successfully fetched UniProt accession
        self.alphafold_data      = None # dict: pdb_str, plddt, dist_matrix, accession
        self.batch_struct        = {}   # maps batch rec_id -> per-chain struct dict
        self.pfam_domains        = []   # list of domain dicts from Pfam
        self._alphafold_worker   = None
        self._pfam_worker        = None
        self._blast_worker       = None

        # --- New state for extended features ---
        self._elm_worker         = None
        self._disprot_worker     = None
        self._phasepdb_worker    = None
        self.elm_data            = []   # list of ELM instances
        self.disprot_data        = {}   # DisProt disorder regions
        self.phasepdb_data       = {}   # PhaSepDB lookup result
        self._msa_sequences      = []   # list of aligned sequences
        self._msa_names          = []   # corresponding names
        self._plugins            = []   # loaded plugin modules
        self._load_plugins()

        self.check_dependencies()
        self.main_tabs = NavTabWidget()
        self.setCentralWidget(self.main_tabs)
        self.init_analysis_tab()
        self.init_graphs_tab()
        self.init_structure_tab()
        self.init_blast_tab()
        self.init_batch_tab()
        self.init_comparison_tab()
        self.init_truncation_tab()
        self.init_msa_tab()
        self.init_complex_tab()
        self.init_settings_tab()
        self.init_help_tab()
        self._setup_shortcuts()

    # --- Tooltip helpers ---

    def _set_tooltip(self, widget, text: str):
        """Register a tooltip; only shown when tooltips are enabled."""
        self._tooltips[widget] = text
        if self.enable_tooltips:
            widget.setToolTip(text)

    def _apply_tooltips(self):
        for widget, text in self._tooltips.items():
            widget.setToolTip(text if self.enable_tooltips else "")

    # --- Layout helpers ---

    def _clear_layout(self, layout):
        for i in reversed(range(layout.count())):
            item = layout.itemAt(i)
            w = item.widget() if item else None
            if w:
                w.setParent(None)

    def _replace_graph(self, title: str, fig):
        """Swap graph canvas in the named tab."""
        tab, vb = self.graph_tabs[title]
        self._clear_layout(vb)
        canvas = FigureCanvas(fig)
        vb.addWidget(NavigationToolbar2QT(canvas, self))
        vb.addWidget(canvas)
        btn = QPushButton("Save Graph")
        btn.clicked.connect(lambda _, t=title: self.save_graph(t))
        vb.addWidget(btn, alignment=Qt.AlignRight)

    def _find_canvas(self, vb) -> FigureCanvas | None:
        for i in range(vb.count()):
            item = vb.itemAt(i)
            if item:
                w = item.widget()
                if isinstance(w, FigureCanvas):
                    return w
        return None

    # --- Batch helpers ---

    def _populate_batch_row(self, rec_id: str, seq: str, data: dict):
        hydro, hydrophil, pos, neg, neu = _calc_batch_stats(seq, data)
        row = self.batch_table.rowCount()
        self.batch_table.insertRow(row)
        for col, val in enumerate([
            rec_id, str(len(seq)), f"{data['mol_weight']:.2f}",
            f"{data['net_charge_7']:.2f}",
            f"{hydro:.1f}%", f"{hydrophil:.1f}%",
            f"{pos:.1f}%", f"{neg:.1f}%", f"{neu:.1f}%",
        ]):
            self.batch_table.setItem(row, col, QTableWidgetItem(val))

    def _populate_chain_combo(self):
        self.chain_combo.clear()
        for rec_id, _, _ in self.batch_data:
            self.chain_combo.addItem(rec_id)
        self.chain_combo.setEnabled(bool(self.batch_data))
        if self.batch_data:
            self.chain_combo.setCurrentIndex(0)

    def _load_batch(self, entries: list):
        """Analyze and load a list of (id, seq) pairs into the batch table."""
        self.batch_data.clear()
        self.batch_table.setRowCount(0)
        # Reset structure state; callers that bring structure (import_pdb,
        # fetch_accession PDB branch, _on_alphafold_finished) re-populate these
        # after calling _load_batch.
        self.batch_struct   = {}
        self.alphafold_data = None
        self.save_pdb_btn.setEnabled(False)
        for rec_id, seq in entries:
            if not is_valid_protein(seq):
                continue
            data = AnalysisTools.analyze_sequence(
                seq, 7.0, self.default_window_size, self.use_reducing, self.custom_pka
            )
            self.batch_data.append((rec_id, seq, data))
            self._populate_batch_row(rec_id, seq, data)
        self._populate_chain_combo()

    # --- Tab builders ---

    def init_analysis_tab(self):
        container = QWidget()
        outer     = QVBoxLayout(container)
        outer.setContentsMargins(6, 6, 6, 4)
        outer.setSpacing(6)
        self.main_tabs.addTab(container, "Analysis")

        # ---- toolbar row 1 ----
        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)
        self.import_fasta_btn = QPushButton("Import FASTA")
        self.import_fasta_btn.clicked.connect(self.import_fasta)
        self.import_pdb_btn = QPushButton("Import PDB")
        self.import_pdb_btn.clicked.connect(self.import_pdb)
        self.analyze_btn = QPushButton("Analyze  [Ctrl+↵]")
        self.analyze_btn.clicked.connect(self.on_analyze)
        self.save_pdf_btn = QPushButton("Export PDF")
        self.save_pdf_btn.clicked.connect(self.export_pdf)
        self.mutate_btn = QPushButton("Mutate…")
        self.mutate_btn.clicked.connect(self.open_mutation_dialog)
        self.session_save_btn = QPushButton("Save Session")
        self.session_save_btn.clicked.connect(self.session_save)
        self.session_load_btn = QPushButton("Load Session")
        self.session_load_btn.clicked.connect(self.session_load)
        self.figure_composer_btn = QPushButton("Figure Composer")
        self.figure_composer_btn.clicked.connect(self.open_figure_composer)
        self._set_tooltip(self.figure_composer_btn,
                          "Compose a multi-panel publication figure from any combination of graphs.")
        for w in (self.import_fasta_btn, self.import_pdb_btn, self.analyze_btn,
                  self.save_pdf_btn, self.mutate_btn,
                  self.session_save_btn, self.session_load_btn,
                  self.figure_composer_btn):
            w.setMinimumHeight(32)
            toolbar.addWidget(w)
        toolbar.addStretch()
        outer.addLayout(toolbar)

        # ---- toolbar row 2: UniProt/NCBI fetch + history ----
        tb2 = QHBoxLayout()
        tb2.setSpacing(6)
        tb2.addWidget(QLabel("Fetch:"))
        self.accession_input = QLineEdit()
        self.accession_input.setPlaceholderText("UniProt ID or PDB ID (e.g. P04637, 1ABC)")
        self.accession_input.setMaximumWidth(200)
        tb2.addWidget(self.accession_input)
        fetch_btn = QPushButton("Fetch")
        fetch_btn.setMinimumHeight(28)
        fetch_btn.clicked.connect(self.fetch_accession)
        tb2.addWidget(fetch_btn)
        tb2.addSpacing(16)
        self.fetch_af_btn = QPushButton("Fetch AlphaFold")
        self.fetch_af_btn.setMinimumHeight(28)
        self.fetch_af_btn.setEnabled(False)
        self.fetch_af_btn.clicked.connect(self.fetch_alphafold)
        self._set_tooltip(self.fetch_af_btn,
                          "Fetch AlphaFold predicted structure (requires a UniProt accession).")
        tb2.addWidget(self.fetch_af_btn)
        self.fetch_pfam_btn = QPushButton("Fetch Pfam")
        self.fetch_pfam_btn.setMinimumHeight(28)
        self.fetch_pfam_btn.setEnabled(False)
        self.fetch_pfam_btn.clicked.connect(self.fetch_pfam)
        self._set_tooltip(self.fetch_pfam_btn,
                          "Fetch Pfam domain annotations from InterPro (requires a UniProt accession).")
        tb2.addWidget(self.fetch_pfam_btn)

        self.fetch_elm_btn = QPushButton("Fetch ELM")
        self.fetch_elm_btn.setMinimumHeight(28)
        self.fetch_elm_btn.setEnabled(False)
        self.fetch_elm_btn.clicked.connect(self.fetch_elm)
        self._set_tooltip(self.fetch_elm_btn,
                          "Fetch experimentally validated linear motifs from ELM database.")
        tb2.addWidget(self.fetch_elm_btn)

        self.fetch_disprot_btn = QPushButton("DisProt")
        self.fetch_disprot_btn.setMinimumHeight(28)
        self.fetch_disprot_btn.setEnabled(False)
        self.fetch_disprot_btn.clicked.connect(self.fetch_disprot)
        self._set_tooltip(self.fetch_disprot_btn,
                          "Fetch experimentally validated disorder regions from DisProt.")
        tb2.addWidget(self.fetch_disprot_btn)

        self.fetch_phasepdb_btn = QPushButton("PhaSepDB")
        self.fetch_phasepdb_btn.setMinimumHeight(28)
        self.fetch_phasepdb_btn.setEnabled(False)
        self.fetch_phasepdb_btn.clicked.connect(self.fetch_phasepdb)
        self._set_tooltip(self.fetch_phasepdb_btn,
                          "Check if protein is in PhaSepDB (phase separation database).")
        tb2.addWidget(self.fetch_phasepdb_btn)

        tb2.addSpacing(20)
        tb2.addWidget(QLabel("History:"))
        self.history_combo = QComboBox()
        self.history_combo.setMinimumWidth(200)
        self.history_combo.addItem("— recent sequences —")
        self.history_combo.currentIndexChanged.connect(self._on_history_selected)
        tb2.addWidget(self.history_combo)
        tb2.addStretch()
        outer.addLayout(tb2)

        # ---- splitter: left input panel | right results panel ----
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(4)

        # Left panel: sequence input + chain selector + sequence viewer
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 4, 0)
        left_layout.setSpacing(5)

        seq_label = QLabel("Protein Sequence:")
        seq_label.setStyleSheet("font-weight:600; color:#4361ee;")
        left_layout.addWidget(seq_label)

        self.seq_text = QTextEdit()
        self.seq_text.setPlaceholderText("Paste a protein sequence here, or use Import…")
        self.seq_text.setFont(QFont("Courier New", 10))
        self.seq_text.setMaximumHeight(160)
        left_layout.addWidget(self.seq_text)

        chain_row = QHBoxLayout()
        chain_lbl = QLabel("Chain:")
        chain_lbl.setStyleSheet("font-weight:600;")
        self.chain_combo = QComboBox()
        self.chain_combo.setEnabled(False)
        self.chain_combo.currentTextChanged.connect(self.on_chain_selected)
        chain_row.addWidget(chain_lbl)
        chain_row.addWidget(self.chain_combo, 1)
        left_layout.addLayout(chain_row)

        # Sequence viewer (UniProt style) + motif search
        sv_hdr = QHBoxLayout()
        seq_view_label = QLabel("Sequence Viewer:")
        seq_view_label.setStyleSheet("font-weight:600; color:#4361ee; margin-top:4px;")
        sv_hdr.addWidget(seq_view_label)
        sv_hdr.addStretch()
        sv_hdr.addWidget(QLabel("Search:"))
        self.motif_input = QLineEdit()
        self.motif_input.setPlaceholderText("motif / regex")
        self.motif_input.setMaximumWidth(130)
        sv_hdr.addWidget(self.motif_input)
        hl_btn = QPushButton("Highlight")
        hl_btn.setMinimumHeight(26)
        hl_btn.clicked.connect(self.highlight_motif)
        sv_hdr.addWidget(hl_btn)
        clr_btn = QPushButton("Clear")
        clr_btn.setMinimumHeight(26)
        clr_btn.clicked.connect(self.clear_motif_highlight)
        sv_hdr.addWidget(clr_btn)
        left_layout.addLayout(sv_hdr)
        self.seq_viewer = QTextBrowser()
        self.seq_viewer.setFont(QFont("Courier New", 10))
        self.seq_viewer.setStyleSheet(
            "QTextBrowser { background:#f8f9fd; border:1px solid #e8eaf0;"
            " border-radius:4px; padding:6px; }"
        )
        left_layout.addWidget(self.seq_viewer, 1)

        splitter.addWidget(left)

        # Right panel: section list + content stack
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 0, 0, 0)
        right_layout.setSpacing(0)

        report_panel = QWidget()
        report_h     = QHBoxLayout(report_panel)
        report_h.setContentsMargins(0, 0, 0, 0)
        report_h.setSpacing(0)

        self.report_section_list = QListWidget()
        self.report_section_list.setObjectName("report_nav")
        self.report_section_list.setFixedWidth(152)
        self.report_section_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        report_h.addWidget(self.report_section_list)

        rsep = QFrame()
        rsep.setFrameShape(QFrame.VLine)
        rsep.setFrameShadow(QFrame.Plain)
        rsep.setObjectName("nav_sep")
        report_h.addWidget(rsep)

        self.report_stack = QStackedWidget()
        report_h.addWidget(self.report_stack, 1)

        right_layout.addWidget(report_panel, 1)

        self.report_section_tabs = {}
        for sec in REPORT_SECTIONS:
            self.report_section_list.addItem(QListWidgetItem(sec))

            tab = QWidget()
            vb  = QVBoxLayout(tab)
            vb.setContentsMargins(4, 4, 4, 4)
            btn_row = QHBoxLayout()
            btn_row.setSpacing(4)
            if sec == "Composition":
                for lbl, mode in [("A–Z", "alpha"), ("By Freq", "composition"),
                                   ("Hydro ↑", "hydro_inc"), ("Hydro ↓", "hydro_dec")]:
                    b = QPushButton(lbl)
                    b.setMaximumWidth(90)
                    b.setMinimumHeight(26)
                    b.clicked.connect(lambda _, m=mode: self.sort_composition(m))
                    btn_row.addWidget(b)
            btn_row.addStretch()
            copy_btn = QPushButton("Copy Table")
            copy_btn.setMaximumWidth(100)
            copy_btn.setMinimumHeight(26)
            copy_btn.clicked.connect(lambda _, s=sec: self._copy_section(s))
            btn_row.addWidget(copy_btn)
            vb.addLayout(btn_row)
            browser = QTextBrowser()
            vb.addWidget(browser)
            self.report_stack.addWidget(tab)
            self.report_section_tabs[sec] = browser

        self.report_section_list.currentRowChanged.connect(
            self.report_stack.setCurrentIndex)
        self.report_section_list.setCurrentRow(0)

        splitter.addWidget(right)
        splitter.setSizes([400, 700])
        outer.addWidget(splitter, 1)

    def init_graphs_tab(self):
        container = QWidget()
        outer     = QHBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.main_tabs.addTab(container, "Graphs")

        # ── Left: category tree ──────────────────────────────────────────────
        self.graph_tree = QTreeWidget()
        self.graph_tree.setObjectName("graph_tree")
        self.graph_tree.setHeaderHidden(True)
        self.graph_tree.setFixedWidth(186)
        self.graph_tree.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graph_tree.setIndentation(14)
        outer.addWidget(self.graph_tree)

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setFrameShadow(QFrame.Plain)
        sep.setObjectName("nav_sep")
        outer.addWidget(sep)

        # ── Right: canvas stack + toolbar ───────────────────────────────────
        right = QWidget()
        right_v = QVBoxLayout(right)
        right_v.setContentsMargins(4, 4, 4, 4)
        right_v.setSpacing(4)
        outer.addWidget(right, 1)

        save_all = QPushButton("Save All Graphs")
        save_all.setMaximumWidth(160)
        save_all.clicked.connect(self.save_all_graphs)
        right_v.addWidget(save_all, alignment=Qt.AlignRight)

        self.graph_stack = QStackedWidget()
        right_v.addWidget(self.graph_stack, 1)

        # ── Populate tree and stack ──────────────────────────────────────────
        self.graph_tabs = {}
        self._graph_title_to_stack_idx: dict = {}
        bold_font = QFont()
        bold_font.setBold(True)
        bold_font.setPointSize(10)

        for category, titles in GRAPH_CATEGORIES:
            cat_item = QTreeWidgetItem([f"  {category}"])
            cat_item.setFont(0, bold_font)
            cat_item.setFlags(cat_item.flags() & ~Qt.ItemIsSelectable)
            self.graph_tree.addTopLevelItem(cat_item)

            for title in titles:
                leaf = QTreeWidgetItem([f"  {title}"])
                leaf.setData(0, Qt.UserRole, title)
                cat_item.addChild(leaf)

                panel = QWidget()
                vb    = QVBoxLayout(panel)
                vb.setContentsMargins(4, 4, 4, 4)
                ph = QLabel(f"Run analysis to generate:\n{title}")
                ph.setAlignment(Qt.AlignCenter)
                ph.setStyleSheet("color:#718096; font-style:italic;")
                vb.addWidget(ph)
                save_btn = QPushButton("Save Graph")
                save_btn.setMaximumWidth(120)
                save_btn.clicked.connect(lambda _, t=title: self.save_graph(t))
                vb.addWidget(save_btn, alignment=Qt.AlignRight)

                idx = self.graph_stack.addWidget(panel)
                self.graph_tabs[title] = (panel, vb)
                self._graph_title_to_stack_idx[title] = idx

            cat_item.setExpanded(True)

        self.graph_tree.itemClicked.connect(self._on_graph_tree_clicked)
        # Select first graph
        first_cat = self.graph_tree.topLevelItem(0)
        if first_cat and first_cat.childCount():
            first_leaf = first_cat.child(0)
            self.graph_tree.setCurrentItem(first_leaf)
            self.graph_stack.setCurrentIndex(0)

    def init_structure_tab(self):
        """Tab for interactive 3D structure viewer (PDB upload, RCSB PDB fetch, or AlphaFold fetch)."""
        container = QWidget()
        layout    = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        self.main_tabs.addTab(container, "Structure")

        info_row = QHBoxLayout()
        self.af_status_lbl = QLabel("No structure loaded.  Import a PDB file, fetch a PDB ID, or fetch AlphaFold.")
        self.af_status_lbl.setStyleSheet("color:#718096; font-style:italic;")
        info_row.addWidget(self.af_status_lbl, 1)
        self.save_pdb_btn = QPushButton("Save PDB")
        self.save_pdb_btn.setEnabled(False)
        self.save_pdb_btn.clicked.connect(self._save_pdb)
        info_row.addWidget(self.save_pdb_btn)
        layout.addLayout(info_row)

        if _WEBENGINE_AVAILABLE:
            self.structure_viewer = QWebEngineView()
            self.structure_viewer.setMinimumHeight(500)
            layout.addWidget(self.structure_viewer, 1)
            # Colour-mode buttons
            btn_row = QHBoxLayout()
            for label, js in [
                ("Color: pLDDT",       "colorByPLDDT()"),
                ("Color: Residue Type","colorByResidue()"),
                ("Color: Chain",       "colorByChain()"),
                ("Cartoon / Sphere",   "toggleStyle()"),
            ]:
                b = QPushButton(label)
                b.setMinimumHeight(28)
                b.clicked.connect(lambda _, call=js: self.structure_viewer.page().runJavaScript(call))
                btn_row.addWidget(b)
            btn_row.addStretch()
            layout.addLayout(btn_row)
        else:
            msg = QLabel(
                "PyQtWebEngine is not installed.\n"
                "Install it with:  pip install PyQtWebEngine\n\n"
                "You can still save the PDB file and open it in PyMOL, UCSF ChimeraX, or 3Dmol.csb.pitt.edu."
            )
            msg.setAlignment(Qt.AlignCenter)
            msg.setStyleSheet("color:#718096; font-size:11pt;")
            layout.addWidget(msg, 1)
            self.structure_viewer = None

    _3DMOL_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  html, body {{ margin:0; padding:0; overflow:hidden; background:#1a1a2e; width:100%; height:100%; }}
  #vp {{ width:100%; height:100vh; position:relative; }}
</style>
</head><body>
<div id="vp"></div>
<script src="https://3dmol.org/build/3Dmol-min.js"></script>
<script>
var viewer = null;
var pdbData = {pdb_json};
var cartoonMode = true;

function init() {{
    viewer = $3Dmol.createViewer("vp", {{backgroundColor:"#1a1a2e", antialias:true}});
    viewer.addModel(pdbData, "pdb");
    colorByPLDDT();
    viewer.zoomTo();
    viewer.render();
}}

function colorByPLDDT() {{
    if (!viewer) return;
    viewer.setStyle({{}}, {{cartoon:{{colorscheme:{{prop:"b",gradient:"rwb",min:0,max:100}}}}}});
    viewer.render();
}}

function colorByResidue() {{
    if (!viewer) return;
    viewer.setStyle({{}}, {{cartoon:{{colorscheme:"amino"}}}});
    viewer.render();
}}

function colorByChain() {{
    if (!viewer) return;
    viewer.setStyle({{}}, {{cartoon:{{colorscheme:"chain"}}}});
    viewer.render();
}}

function toggleStyle() {{
    if (!viewer) return;
    cartoonMode = !cartoonMode;
    var scheme = {{prop:"b",gradient:"rwb",min:0,max:100}};
    if (cartoonMode) {{
        viewer.setStyle({{}}, {{cartoon:{{colorscheme:scheme}}}});
    }} else {{
        viewer.setStyle({{}}, {{sphere:{{colorscheme:scheme,radius:0.5}}}});
    }}
    viewer.render();
}}

window.addEventListener("load", init);
</script>
</body></html>"""

    def _load_structure_viewer(self, pdb_str: str):
        """Load PDB string into the 3D viewer widget."""
        if not _WEBENGINE_AVAILABLE or self.structure_viewer is None:
            return
        pdb_json = json.dumps(pdb_str)
        html     = self._3DMOL_HTML.format(pdb_json=pdb_json)
        self.structure_viewer.setHtml(html)

    def _save_pdb(self):
        if not self.alphafold_data:
            return
        fn, _ = QFileDialog.getSaveFileName(self, "Save PDB", "", "PDB Files (*.pdb)")
        if fn:
            if not fn.lower().endswith(".pdb"):
                fn += ".pdb"
            try:
                with open(fn, "w") as f:
                    f.write(self.alphafold_data["pdb_str"])
                self.statusBar.showMessage(f"PDB saved: {fn}", 3000)
            except OSError as e:
                QMessageBox.critical(self, "Save Failed", str(e))

    def init_blast_tab(self):
        """Tab for NCBI BLAST search of the current sequence."""
        container = QWidget()
        layout    = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        self.main_tabs.addTab(container, "BLAST")

        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("Database:"))
        self.blast_db_combo = QComboBox()
        self.blast_db_combo.addItems(["nr", "swissprot", "pdb", "refseq_protein"])
        self.blast_db_combo.setMaximumWidth(140)
        ctrl_row.addWidget(self.blast_db_combo)
        ctrl_row.addSpacing(12)
        ctrl_row.addWidget(QLabel("Max hits:"))
        self.blast_hits_spin = QSpinBox()
        self.blast_hits_spin.setRange(5, 100)
        self.blast_hits_spin.setValue(20)
        self.blast_hits_spin.setMaximumWidth(70)
        ctrl_row.addWidget(self.blast_hits_spin)
        ctrl_row.addSpacing(12)
        self.blast_run_btn = QPushButton("BLAST Current Sequence")
        self.blast_run_btn.setMinimumHeight(30)
        self.blast_run_btn.clicked.connect(self.run_blast)
        ctrl_row.addWidget(self.blast_run_btn)
        ctrl_row.addStretch()
        layout.addLayout(ctrl_row)

        self.blast_status_lbl = QLabel("Ready.  Run analysis first, then click 'BLAST Current Sequence'.")
        self.blast_status_lbl.setStyleSheet("color:#718096; font-style:italic;")
        layout.addWidget(self.blast_status_lbl)

        self.blast_table = QTableWidget()
        self.blast_table.setAlternatingRowColors(True)
        self.blast_table.setColumnCount(7)
        self.blast_table.setHorizontalHeaderLabels(
            ["Accession", "Description", "Length", "Score", "E-value", "% Identity", "Load"]
        )
        self.blast_table.horizontalHeader().setStretchLastSection(False)
        self.blast_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.blast_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.blast_table, 1)

    def init_comparison_tab(self):
        container = QWidget()
        layout    = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        self.main_tabs.addTab(container, "Compare")

        # Two sequence inputs side by side
        inputs = QSplitter(Qt.Horizontal)
        for attr, lbl in [("compare_seq_a", "Sequence A"), ("compare_seq_b", "Sequence B")]:
            w = QWidget()
            v = QVBoxLayout(w)
            v.setContentsMargins(0, 0, 0, 0)
            v.addWidget(QLabel(lbl))
            te = QTextEdit()
            te.setPlaceholderText(f"Paste {lbl} here…")
            te.setFont(QFont("Courier New", 10))
            te.setMaximumHeight(120)
            v.addWidget(te)
            setattr(self, attr, te)
            inputs.addWidget(w)
        layout.addWidget(inputs)

        cmp_btn = QPushButton("Compare Sequences")
        cmp_btn.setMinimumHeight(32)
        cmp_btn.clicked.connect(self.do_compare)
        layout.addWidget(cmp_btn, alignment=Qt.AlignLeft)

        self.compare_table = QTableWidget()
        self.compare_table.setAlternatingRowColors(True)
        self.compare_table.setColumnCount(3)
        self.compare_table.setHorizontalHeaderLabels(["Property", "Sequence A", "Sequence B"])
        self.compare_table.horizontalHeader().setStretchLastSection(True)
        self.compare_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        layout.addWidget(self.compare_table, 1)

    def init_batch_tab(self):
        container = QWidget()
        layout    = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        self.main_tabs.addTab(container, "Multichain Analysis")

        btn_row = QHBoxLayout()
        self.batch_export_csv_btn = QPushButton("Export CSV")
        self.batch_export_csv_btn.setMinimumHeight(30)
        self.batch_export_csv_btn.clicked.connect(self.export_batch_csv)
        self.batch_export_json_btn = QPushButton("Export JSON")
        self.batch_export_json_btn.setMinimumHeight(30)
        self.batch_export_json_btn.clicked.connect(self.export_batch_json)
        btn_row.addWidget(self.batch_export_csv_btn)
        btn_row.addWidget(self.batch_export_json_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.batch_table = QTableWidget()
        self.batch_table.setAlternatingRowColors(True)
        self.batch_table.setColumnCount(9)
        self.batch_table.setHorizontalHeaderLabels([
            "ID", "Length", "MW (Da)", "Net Charge",
            "% Hydro", "% Hydrophil", "% +Charged", "% -Charged", "% Neutral",
        ])
        self.batch_table.horizontalHeader().setStretchLastSection(True)
        self.batch_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.batch_table.cellDoubleClicked.connect(self.show_batch_details)
        layout.addWidget(self.batch_table, 1)

    def init_settings_tab(self):
        container = QWidget()
        self.main_tabs.addTab(container, "Settings")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        inner = QWidget()
        scroll.setWidget(inner)
        outer = QVBoxLayout(container)
        outer.addWidget(scroll)

        layout = QVBoxLayout(inner)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(14)

        def _section(title):
            lbl = QLabel(title)
            lbl.setStyleSheet(
                "font-size:11pt; font-weight:700; color:#4361ee;"
                " border-bottom:1px solid #d0d4e0; padding-bottom:4px; margin-top:8px;"
            )
            layout.addWidget(lbl)

        form = QFormLayout()
        form.setHorizontalSpacing(20)
        form.setVerticalSpacing(8)
        form.setLabelAlignment(Qt.AlignRight)

        _section("Analysis Parameters")
        self.ph_input = QLineEdit(str(self.default_pH))
        self._set_tooltip(self.ph_input, "Sets the pH value used for net-charge calculations.")
        form.addRow("Default pH:", self.ph_input)

        self.window_size_input = QLineEdit(str(self.default_window_size))
        self._set_tooltip(self.window_size_input, "Length of sliding window for hydrophobicity profiles.")
        form.addRow("Sliding Window Size:", self.window_size_input)

        self.pka_input = QLineEdit("")
        self.pka_input.setPlaceholderText("e.g. 9.69,2.34,3.90,4.07,8.18,10.46,6.04,10.54,12.48")
        self._set_tooltip(self.pka_input, "Leave blank for defaults. Provide nine comma-separated numbers.")
        form.addRow("Override pKa (N,C,D,E,C,Y,H,K,R):", self.pka_input)

        self.reducing_checkbox = QCheckBox("Assume reducing conditions (Cys not in disulphide)")
        form.addRow("", self.reducing_checkbox)
        layout.addLayout(form)

        form2 = QFormLayout()
        form2.setHorizontalSpacing(20)
        form2.setVerticalSpacing(8)
        form2.setLabelAlignment(Qt.AlignRight)
        _section("Sequence Display")
        self.seq_name_input = QLineEdit(self.sequence_name)
        self.seq_name_input.setPlaceholderText("Leave blank to use FASTA/PDB name automatically")
        self._set_tooltip(self.seq_name_input,
                          "Override the sequence display name. Leave blank to use the file name.")
        form2.addRow("Sequence Name:", self.seq_name_input)
        layout.addLayout(form2)

        form3 = QFormLayout()
        form3.setHorizontalSpacing(20)
        form3.setVerticalSpacing(8)
        form3.setLabelAlignment(Qt.AlignRight)
        _section("Graph Appearance")
        self.label_font_input = QLineEdit(str(self.label_font_size))
        form3.addRow("Label Font Size:", self.label_font_input)

        self.tick_font_input = QLineEdit(str(self.tick_font_size))
        form3.addRow("Tick Font Size:", self.tick_font_input)

        self.marker_size_input = QLineEdit(str(self.marker_size))
        self._set_tooltip(self.marker_size_input, "Size of data markers in line and scatter graphs.")
        form3.addRow("Marker Size:", self.marker_size_input)

        self.graph_format_combo = QComboBox()
        self.graph_format_combo.addItems(["PNG", "SVG", "PDF"])
        self._set_tooltip(self.graph_format_combo, "Default file format when saving graphs.")
        form3.addRow("Default Graph Format:", self.graph_format_combo)

        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(NAMED_COLORMAPS)
        self.colormap_combo.setCurrentText(self.colormap)
        self._set_tooltip(self.colormap_combo, "Colour map for the bead hydrophobicity model.")
        form3.addRow("Bead Colormap:", self.colormap_combo)

        self.graph_color_combo = QComboBox()
        self.graph_color_combo.addItems(list(NAMED_COLORS.keys()))
        # Select the name that matches the current hex
        _rev = {v: k for k, v in NAMED_COLORS.items()}
        self.graph_color_combo.setCurrentText(_rev.get(self.graph_color, "Royal Blue"))
        self._set_tooltip(self.graph_color_combo, "Accent colour for line and bar graphs.")
        form3.addRow("Graph Accent Colour:", self.graph_color_combo)

        self.heading_checkbox = QCheckBox("Show Graph Titles")
        self.heading_checkbox.setChecked(self.show_heading)
        form3.addRow("", self.heading_checkbox)

        self.grid_checkbox = QCheckBox("Show Grid")
        self.grid_checkbox.setChecked(self.show_grid)
        form3.addRow("", self.grid_checkbox)

        self.label_checkbox = QCheckBox("Show residue labels on bead models (≤60 aa)")
        self.label_checkbox.setChecked(self.show_bead_labels)
        form3.addRow("", self.label_checkbox)

        self.transparent_bg_checkbox = QCheckBox("Transparent background on PNG/SVG export")
        self.transparent_bg_checkbox.setChecked(self.transparent_bg)
        form3.addRow("", self.transparent_bg_checkbox)
        layout.addLayout(form3)

        form4 = QFormLayout()
        form4.setHorizontalSpacing(20)
        form4.setVerticalSpacing(8)
        form4.setLabelAlignment(Qt.AlignRight)
        _section("Interface")
        self.app_font_size_input = QLineEdit(str(self.app_font_size))
        self.app_font_size_input.setPlaceholderText("e.g. 12")
        self._set_tooltip(self.app_font_size_input,
                          "Global application font size in points (requires Apply Settings).")
        form4.addRow("UI Font Size (pt):", self.app_font_size_input)

        self.theme_toggle = QCheckBox("Dark Theme")
        self._set_tooltip(self.theme_toggle, "Toggle between light and dark application themes.")
        self.theme_toggle.stateChanged.connect(self.toggle_theme)
        self.theme_toggle.setChecked(False)
        form4.addRow("", self.theme_toggle)

        self.tooltips_checkbox = QCheckBox("Enable Tooltips")
        self.tooltips_checkbox.setChecked(self.enable_tooltips)
        form4.addRow("", self.tooltips_checkbox)
        layout.addLayout(form4)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        apply_btn = QPushButton("Apply Settings")
        apply_btn.setMinimumHeight(34)
        apply_btn.clicked.connect(self.apply_settings)
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.setMinimumHeight(34)
        reset_btn.clicked.connect(self.reset_defaults)
        btn_row.addWidget(apply_btn)
        btn_row.addWidget(reset_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)
        layout.addStretch()

    def init_help_tab(self):
        container = QWidget()
        outer_v   = QVBoxLayout(container)
        outer_v.setContentsMargins(0, 0, 0, 0)
        outer_v.setSpacing(0)
        self.main_tabs.addTab(container, "Help")

        # Two-panel layout: section list on left, content on right
        help_h = QHBoxLayout()
        help_h.setContentsMargins(0, 0, 0, 0)
        help_h.setSpacing(0)
        outer_v.addLayout(help_h)

        help_nav = QListWidget()
        help_nav.setObjectName("report_nav")
        help_nav.setFixedWidth(172)
        help_nav.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        help_h.addWidget(help_nav)

        sep = QFrame(); sep.setFrameShape(QFrame.VLine)
        sep.setFrameShadow(QFrame.Plain); sep.setObjectName("nav_sep")
        help_h.addWidget(sep)

        help_stack = QStackedWidget()
        help_h.addWidget(help_stack, 1)

        _HELP_SECTIONS = [
            ("Getting Started", """
<h1>Getting Started</h1>
<h2>Input methods</h2>
<ul>
  <li><b>Paste sequence</b> — type or paste a bare amino-acid string (ACDEFG…) or FASTA block into the sequence box and click <b>Analyze [Ctrl+Enter]</b>.</li>
  <li><b>Import FASTA</b> — load a .fa / .fasta file (single or multi-sequence).</li>
  <li><b>Import PDB</b> — extract sequence(s) from a local PDB file.</li>
  <li><b>Fetch</b> — enter a <b>UniProt ID</b> (e.g. <tt>P04637</tt>) or a 4-character <b>PDB ID</b> (e.g. <tt>1ABC</tt>) and click <b>Fetch</b>.
    UniProt IDs automatically set the accession for <b>Fetch AlphaFold</b>, <b>Fetch Pfam</b>, and other databases, and trigger analysis immediately.
    PDB IDs download all chains and the coordinate file from RCSB; structural graphs (Ramachandran, distance map, etc.) are shown immediately.</li>
</ul>
<h2>Navigation</h2>
<p>Use the <b>left sidebar</b> to switch between sections. Keyboard shortcuts:</p>
<table>
  <tr><th>Shortcut</th><th>Action</th></tr>
  <tr><td>Ctrl+Enter</td><td>Run analysis</td></tr>
  <tr><td>Ctrl+G</td><td>Jump to Graphs</td></tr>
  <tr><td>Ctrl+E</td><td>Export PDF report</td></tr>
  <tr><td>Ctrl+S</td><td>Save session</td></tr>
  <tr><td>Ctrl+O</td><td>Load session</td></tr>
  <tr><td>Ctrl+F</td><td>Focus motif search</td></tr>
</table>
"""),
            ("Sequence Analysis", """
<h1>Sequence Analysis</h1>
<h2>Composition</h2>
<p>Counts and percentage frequencies of each of the 20 standard amino acids. Use the sort buttons (A–Z, By Freq, Hydro ↑/↓) to reorder the table and the matching bar chart.</p>
<h2>Properties</h2>
<ul>
  <li><b>Molecular Weight</b> — approximate monoisotopic mass (Da).</li>
  <li><b>Isoelectric Point (pI)</b> — pH at which net charge = 0 (Henderson-Hasselbalch).</li>
  <li><b>Extinction Coefficient</b> — absorbance at 280 nm per M per cm; uses W (5500), Y (1490), C–C (125 in oxidising conditions).</li>
  <li><b>GRAVY Score</b> — grand average of hydropathicity; positive = hydrophobic (Kyte &amp; Doolittle 1982).</li>
  <li><b>Instability Index</b> — &lt;40 suggests stable protein in vitro (Guruprasad et al. 1990).</li>
  <li><b>Aromaticity</b> — fraction of F, W, Y residues.</li>
</ul>
<h2>Hydrophobicity</h2>
<p>Per-residue Kyte-Doolittle values plus fraction of hydrophobic (KD &gt; 0) vs hydrophilic (KD &lt; 0) residues.</p>
<h2>Charge</h2>
<ul>
  <li><b>FCR</b> — fraction of charged residues (K, R, D, E).</li>
  <li><b>NCPR</b> — net charge per residue = (pos − neg) / length.</li>
  <li><b>Kappa (κ)</b> — charge patterning: 0 = well-mixed, 1 = fully segregated (Das &amp; Pappu 2013).</li>
  <li><b>Charge asymmetry</b> — ratio of positive to negative residues.</li>
</ul>
<h2>Aromatic &amp; π-Interactions</h2>
<ul>
  <li><b>Aromatic fraction</b> — (F+W+Y)/length; π–π stacking drives many condensates.</li>
  <li><b>Cation–π pairs</b> — K or R within ±4 positions of F/W/Y.</li>
  <li><b>π–π pairs</b> — F/W/Y within ±4 positions of another aromatic.</li>
</ul>
<h2>Low Complexity</h2>
<ul>
  <li><b>Shannon entropy</b> — compositional complexity in bits; max ≈ 4.32 (all 20 AAs equal).</li>
  <li><b>Prion-like score</b> — fraction of N, Q, S, G, Y; enriched in yeast prion domains (PLAAC / Lancaster &amp; Bhatt).</li>
  <li><b>LC fraction</b> — fraction covered by windows (w=12) with entropy &lt; 2.0 bits.</li>
</ul>
<h2>Disorder</h2>
<ul>
  <li><b>Disorder-promoting fraction</b> — A, E, G, K, P, Q, R, S (Uversky classification).</li>
  <li><b>Order-promoting fraction</b> — C, F, H, I, L, M, V, W, Y.</li>
  <li><b>Aliphatic index</b> — (A + 2.9V + 3.9(I+L)) / length × 100 (Ikai 1980).</li>
  <li><b>Omega (Ω)</b> — patterning of sticker residues; 0 = even, 1 = clustered (Das et al. 2015).</li>
</ul>
<h2>Secondary Structure (Chou-Fasman)</h2>
<p>Per-residue helix (Pα) and sheet (Pβ) propensities. Pα or Pβ &gt; 1.0 favours that element.
The per-residue disorder score is an IUPred-inspired propensity (0 = ordered, 1 = disordered).</p>
<h2>Repeat Motifs</h2>
<ul>
  <li><b>RGG</b> — Arg-Gly-Gly; key driver in FUS, hnRNP family.</li>
  <li><b>FG</b> — Phe-Gly; hallmark of nucleoporin IDRs.</li>
  <li><b>SR/RS</b> — Ser-Arg; splicing factor signature.</li>
  <li><b>QN/NQ</b> — Gln-Asn; yeast prion signature.</li>
</ul>
<h2>Sticker &amp; Spacer</h2>
<p>Stickers = F, W, Y, K, R, D, E — residues mediating specific interactions.
Spacers = all others. Mean/min/max gaps between consecutive stickers (Mittag &amp; Pappu model).</p>
<h2>Phase Separation</h2>
<p>Composite LLPS score (0–1) combining aromatic fraction, prion-like score, disorder fraction, FCR, Omega, LARKS density, and |NCPR| penalty. LARKS are short amyloid-like segments associated with condensate formation (see <b>Phase Separation</b> help page for details).</p>
<h2>Linear Motifs</h2>
<p>Regex scan against 15 built-in functional short linear motif patterns (SLiMs): NLS, NES, PxxP, 14-3-3, RGG, FG, KFERQ, KDEL, PKA, SxIP, WW ligand, caspase-3, N-glycosylation, SUMOylation, CK2 sites. All matches require experimental validation.</p>
"""),
            ("Transmembrane Helices", """
<h1>Transmembrane Helix Prediction</h1>
<p>Available in the <b>TM Helices</b> report section and the <b>TM Topology</b> graph
after running analysis. No external server required — purely sequence-based.</p>
<h2>Algorithm</h2>
<ol>
  <li>For every position where a full <b>19-residue window</b> fits, the Kyte-Doolittle average is computed (no partial-edge windows — avoids false positives at sequence termini).</li>
  <li>Each residue covered by at least one window scoring ≥ <b>1.6</b> is marked as a TM candidate.</li>
  <li>Contiguous TM-marked segments of length <b>17–25 aa</b> are retained as helices.</li>
  <li>Segments longer than 25 aa are trimmed to the single best-scoring 19-residue window within that region.</li>
  <li><b>Inside-positive rule (von Heijne)</b> — the 15-residue flanks on each side are scanned for K/R; the side with more positives is cytoplasmic.
    <ul>
      <li><b>out→in</b>: N-terminus extracellular, C-terminus cytoplasmic.</li>
      <li><b>in→out</b>: N-terminus cytoplasmic, C-terminus extracellular.</li>
    </ul>
  </li>
</ol>
<h2>Domain Architecture — TM track</h2>
<p>After analysis the <b>Domain Architecture</b> graph automatically shows a TM Helices track
(purple blocks) without requiring AlphaFold or Pfam data.</p>
<h2>TM Topology graph</h2>
<p>A simplified snake-plot. The yellow band represents the membrane. Blue rectangles are TM helices
labelled with their residue range. Loops are drawn above (extracellular) or below (cytoplasmic)
the band according to the predicted topology.</p>
<p class="note">This is a heuristic predictor for first-pass screening.
For high-accuracy results use TMHMM, Phobius, or DeepTMHMM.</p>
"""),
            ("AlphaFold & 3D Structure", """
<h1>3D Structure &amp; AlphaFold Integration</h1>
<p>Structure data can come from three sources:</p>
<ul>
  <li><b>Import PDB</b> — load a local .pdb file directly.</li>
  <li><b>Fetch PDB ID</b> — enter a 4-character RCSB PDB code in the accession field; sequences and
      the coordinate file are both downloaded automatically.</li>
  <li><b>Fetch AlphaFold</b> — requires a UniProt accession.  Fetch it with the <b>Fetch</b> button first,
      then click <b>Fetch AlphaFold</b> to download the EBI AlphaFold2 predicted structure.</li>
</ul>
<p>For every structure source, Cα coordinates are extracted per chain and used to compute:</p>
<ul>
  <li>Per-residue <b>pLDDT / B-factor</b> scores.</li>
  <li>Cα pairwise <b>distance matrix</b>.</li>
</ul>
<h2>pLDDT Profile graph</h2>
<p>Per-residue confidence score (0–100) plotted with four coloured confidence bands:</p>
<table>
  <tr><th>Colour</th><th>Range</th><th>Meaning</th></tr>
  <tr><td>Blue</td><td>&gt;90</td><td>Very high confidence</td></tr>
  <tr><td>Cyan</td><td>70–90</td><td>Confident</td></tr>
  <tr><td>Yellow</td><td>50–70</td><td>Low confidence</td></tr>
  <tr><td>Orange</td><td>&lt;50</td><td>Very low / disordered region</td></tr>
</table>
<h2>Distance Map graph</h2>
<p>Symmetric Cα–Cα pairwise distance heatmap (viridis palette, 0–40 Å). The pink contour marks
the <b>8 Å contact threshold</b> — residue pairs inside this contour are in physical contact.</p>
<h2>3D Structure viewer</h2>
<p>The <b>Structure</b> section hosts an interactive 3D viewer powered by
<a href="https://3dmol.csb.pitt.edu">3Dmol.js</a>. Requires <b>PyQtWebEngine</b>:</p>
<pre>pip install PyQtWebEngine</pre>
<p>If not installed, the PDB can be saved locally and opened in PyMOL, UCSF ChimeraX, or the
web viewer at <tt>3dmol.csb.pitt.edu</tt>.</p>
<p>Colour modes available in the Structure section:</p>
<ul>
  <li><b>pLDDT</b> — red (low) → white → blue (high).</li>
  <li><b>Residue Type</b> — amino-acid colour scheme.</li>
  <li><b>Chain</b> — each chain a different colour.</li>
  <li><b>Cartoon / Sphere</b> — toggle representation.</li>
</ul>
"""),
            ("Pfam Domains", """
<h1>Pfam Domain Annotations</h1>
<p>Requires an internet connection and a valid UniProt accession. Click <b>Fetch Pfam</b>
after loading a UniProt accession. Not available for PDB IDs.</p>
<h2>Data source</h2>
<p>Queries the <b>EMBL-EBI InterPro REST API</b> for all Pfam-family entries associated
with the given UniProt protein. Results include domain name, accession, and start/end residue positions.</p>
<h2>Domain Architecture graph (multi-track)</h2>
<p>The Domain Architecture graph is always shown after analysis and displays up to four tracks:</p>
<table>
  <tr><th>Track</th><th>Colour</th><th>Source</th></tr>
  <tr><td>Pfam Domains</td><td>BEER palette</td><td>Fetch Pfam (requires UniProt)</td></tr>
  <tr><td>Disorder</td><td>Orange</td><td>IUPred-inspired score &gt; 0.5 (always available)</td></tr>
  <tr><td>Low Complexity</td><td>Teal</td><td>Shannon entropy &lt; 2.0 bits (always available)</td></tr>
  <tr><td>TM Helices</td><td>Purple</td><td>KD sliding window prediction (always available)</td></tr>
</table>
<p class="note">Only Pfam entries are shown in the Pfam track. Tracks without data are omitted automatically.</p>
"""),
            ("BLAST Search", """
<h1>BLAST Integration</h1>
<p>The <b>BLAST</b> section submits the currently analysed sequence to NCBI via the
<tt>Bio.Blast.NCBIWWW</tt> interface and displays the top hits. Requires internet access
and can take <b>1–3 minutes</b>.</p>
<h2>Controls</h2>
<ul>
  <li><b>Database</b> — nr (non-redundant), swissprot, pdb, refseq_protein.</li>
  <li><b>Max hits</b> — number of alignments to retrieve (5–100).</li>
  <li><b>BLAST Current Sequence</b> — submits blastp with the sequence from the last analysis.</li>
</ul>
<h2>Results table</h2>
<table>
  <tr><th>Column</th><th>Meaning</th></tr>
  <tr><td>Accession</td><td>NCBI accession of the hit</td></tr>
  <tr><td>Description</td><td>Truncated title of the hit (first 80 chars)</td></tr>
  <tr><td>Length</td><td>Subject sequence length (aa)</td></tr>
  <tr><td>Score</td><td>Bit score of top HSP</td></tr>
  <tr><td>E-value</td><td>Expect value of top HSP</td></tr>
  <tr><td>% Identity</td><td>Percent identical residues in the aligned region</td></tr>
  <tr><td>Load</td><td>Loads the subject sequence into Analysis and re-runs it</td></tr>
</table>
<p class="note">BLAST uses the public NCBI servers. Do not submit large numbers of queries
in rapid succession. For batch analyses use NCBI standalone BLAST locally.</p>
"""),
            ("Graphs Reference", """
<h1>Graphs Reference</h1>
<p>All graphs are in the <b>Graphs</b> section. Use the category tree on the left to navigate.
Each graph has a <b>Save Graph</b> button; <b>Save All Graphs</b> exports every graph
to a chosen directory in the format configured in Settings.</p>
<h2>Composition</h2>
<ul>
  <li><b>AA Composition (Bar / Pie)</b> — amino acid counts and frequencies. Bar sort order matches the report buttons.</li>
</ul>
<h2>Profiles</h2>
<ul>
  <li><b>Hydrophobicity Profile</b> — Kyte-Doolittle sliding-window average (window set in Settings).</li>
  <li><b>Local Charge Profile</b> — sliding-window NCPR.</li>
  <li><b>Local Complexity</b> — sliding-window Shannon entropy; dashed line = LC threshold (2.0 bits).</li>
  <li><b>Disorder Profile</b> — IUPred-inspired per-residue score; orange fill = disordered (&gt;0.5).</li>
  <li><b>Linear Sequence Map</b> — four-track overview: hydrophobicity, NCPR, disorder, helix Pα.</li>
  <li><b>Secondary Structure</b> — two-panel Chou-Fasman: top = helix Pα, bottom = sheet Pβ. Fill above 1.0 = propensity for that structure.</li>
</ul>
<h2>Charge &amp; π-Interactions</h2>
<ul>
  <li><b>Net Charge vs pH</b> — Henderson-Hasselbalch charge curve 0–14; pI marked.</li>
  <li><b>Isoelectric Focus</b> — enhanced charge curve with physiological pH 7.4 annotation.</li>
  <li><b>Charge Decoration</b> — Das-Pappu FCR vs |NCPR| phase diagram.</li>
  <li><b>Cation–π Map</b> — proximity heat map (1/distance) for K/R ↔ F/W/Y pairs within ±8 residues.</li>
</ul>
<h2>Structure &amp; Folding</h2>
<ul>
  <li><b>Bead Model (Hydrophobicity)</b> — per-residue KD score; colourmap selectable in Settings.</li>
  <li><b>Bead Model (Charge)</b> — K/R blue, D/E red, H cyan, neutral grey.</li>
  <li><b>Sticker Map</b> — aromatic (amber), basic (blue), acidic (pink), spacer (grey).</li>
  <li><b>Helical Wheel</b> — Cartesian projection of first 18 residues at 100°/step; connecting lines between sequential residues; KD coloured with luminance-contrast labels.</li>
  <li><b>TM Topology</b> — snake-plot of predicted transmembrane helices (see Transmembrane Helices).</li>
</ul>
<h2>Structural Graphs</h2>
<ul>
  <li><b>pLDDT Profile</b> — per-residue B-factor confidence (0–100). Available after any structure is loaded
      (PDB upload, RCSB PDB ID fetch, or AlphaFold fetch).</li>
  <li><b>Cα Distance Map</b> — pairwise distance heatmap with 8 Å contact contour. Available after any structure is loaded.</li>
  <li><b>Ramachandran Plot</b> — φ/ψ dihedral angles coloured by secondary structure. Available after any structure is loaded.</li>
  <li><b>Residue Contact Network</b> — graph of residues within 8 Å contact distance. Available after any structure is loaded.</li>
  <li><b>Domain Architecture</b> — multi-track: Pfam domains, Disorder, Low Complexity, TM Helices.
      Always shown after analysis; Pfam track appears after Fetch Pfam.</li>
</ul>
<h2>Phase Separation / IDP</h2>
<ul>
  <li><b>Uversky Phase Plot</b> — mean |net charge| vs mean normalised hydrophobicity; Uversky boundary line separates IDP from ordered proteins.</li>
  <li><b>Coiled-Coil Profile</b> — per-residue heptad-periodicity score; fill above 0.50 = predicted coiled-coil region.</li>
  <li><b>Saturation Mutagenesis</b> — 20×n heatmap of |ΔGRAVY| + |ΔNCPR| for all single-residue substitutions; white dot = wild type. Available for sequences ≤500 aa.</li>
</ul>
"""),
            ("Phase Separation", """
<h1>Phase Separation &amp; IDP Analysis</h1>
<h2>Composite LLPS Score</h2>
<p>A weighted combination of seven sequence features predicting liquid-liquid phase separation (LLPS) propensity (score 0–1):</p>
<table>
  <tr><th>Feature</th><th>Weight</th><th>Rationale</th></tr>
  <tr><td>Aromatic fraction (F+W+Y)</td><td>0.25</td><td>π–π stacking is the primary LLPS driving force</td></tr>
  <tr><td>Prion-like score (N,Q,S,G,Y)</td><td>0.20</td><td>Prion-like domains enrich most condensates</td></tr>
  <tr><td>Disorder fraction</td><td>0.15</td><td>Intrinsic disorder enables multivalent contacts</td></tr>
  <tr><td>FCR</td><td>0.15</td><td>Charged residues provide electrostatic valency</td></tr>
  <tr><td>Omega (sticker clustering)</td><td>0.15</td><td>Clustered stickers lower saturation concentration</td></tr>
  <tr><td>LARKS density</td><td>0.10</td><td>Short amyloid-like segments stabilise condensates</td></tr>
  <tr><td>|NCPR| penalty</td><td>−0.10</td><td>Strongly asymmetric charge suppresses LLPS</td></tr>
</table>
<p>Score &ge;0.55 = High · 0.30–0.55 = Moderate · &lt;0.30 = Low</p>
<h2>LARKS Detection</h2>
<p>LARKS (Low-complexity Aromatic-Rich Kinked Segments) are short amyloid-like segments
found in many IDP condensates (e.g. FUS, hnRNPA1). A 7-residue window qualifies as a LARKS if:</p>
<ul>
  <li>≥ 1 aromatic residue (F/W/Y)</li>
  <li>≥ 50% low-complexity residues (G/A/S/T/N/Q)</li>
  <li>Shannon entropy &lt; 1.8 bits</li>
</ul>
<p class="note">Reference: Hughes et al. (2018) <i>Science</i> 359, 698–701.</p>
<h2>Uversky Phase Plot</h2>
<p>The sequence is plotted in mean |net charge| vs normalised mean hydrophobicity space.
The dashed boundary (Uversky 2000) separates compact/ordered proteins (above line) from IDPs (below line):</p>
<pre>&lt;H&gt; = 2.785 × &lt;|R|&gt; + 0.446</pre>
<p class="note">Reference: Uversky, Gillespie &amp; Fink (2000) <i>Proteins</i> 41, 415–427.</p>
<h2>Coiled-Coil Prediction</h2>
<p>Heptad-periodicity scoring using a 28-residue (4-heptad) sliding window. Positions a and d (hydrophobic core)
are weighted most heavily. The per-residue score is normalised to 0–1; regions above 0.50 are predicted
coiled-coil segments. Requires ≥7 consecutive residues above threshold.</p>
<h2>Saturation Mutagenesis Heatmap</h2>
<p>Every residue is mutated to all 20 amino acids in silico. Heatmap colour encodes |ΔGRAVY| + |ΔNCPR|.
Hot spots mark positions where substitutions most strongly perturb global properties.
Wild-type residues shown as white dots. Available for sequences ≤ 500 aa.</p>
"""),
            ("Linear Motifs", """
<h1>Linear Motif Scanner</h1>
<p>Sequence is scanned against a built-in library of 15 functional short linear motifs (SLiMs)
using regular-expression matching.</p>
<table>
  <tr><th>Motif Class</th><th>Biological Role</th></tr>
  <tr><td>NLS (basic)</td><td>Nuclear import</td></tr>
  <tr><td>NES (hydrophobic)</td><td>Nuclear export (CRM1)</td></tr>
  <tr><td>PxxP / PPII</td><td>SH3 domain binding</td></tr>
  <tr><td>14-3-3 (mode 1)</td><td>Phosphoserine binding</td></tr>
  <tr><td>RGG box</td><td>RNA binding / LLPS driver</td></tr>
  <tr><td>FG repeat</td><td>Nucleoporin IDR signature</td></tr>
  <tr><td>KFERQ-like</td><td>Chaperone-mediated autophagy</td></tr>
  <tr><td>KDEL/HDEL/RDEL</td><td>ER retention signal</td></tr>
  <tr><td>RxxS/T (PKA)</td><td>PKA phosphorylation consensus</td></tr>
  <tr><td>SxIP</td><td>EB1-dependent microtubule tracking</td></tr>
  <tr><td>WW domain ligand</td><td>PPxY / PxxP interaction</td></tr>
  <tr><td>Caspase-3 site</td><td>Apoptotic cleavage (DEVD)</td></tr>
  <tr><td>N-glycosylation</td><td>NxS/T sequon (x≠P)</td></tr>
  <tr><td>SUMOylation</td><td>ΨKxE consensus</td></tr>
  <tr><td>CK2 site</td><td>Casein kinase II (S/TxxE/D)</td></tr>
</table>
<p class="note">All motifs are regex-based consensus patterns and require experimental validation.
For comprehensive SLiM prediction use ELM (elm.eu.org) or SLiMFinder.</p>
"""),
            ("Multichain & Compare", """
<h1>Multichain Analysis</h1>
<p>When a multi-FASTA file or PDB with multiple chains is imported, all sequences are
analysed in bulk and shown in the <b>Multichain Analysis</b> table. Double-click any row
to load that sequence into the Analysis section with full results.</p>
<p>Export the table to <b>CSV</b> or <b>JSON</b> for downstream processing.</p>
<h1>Compare</h1>
<p>Paste two sequences (or FASTA entries) into the side-by-side inputs and click
<b>Compare Sequences</b>. A property table shows both values side-by-side:
length, MW, pI, GRAVY, FCR, NCPR, net charge, instability, aromaticity, and extinction coefficient.</p>
"""),
            ("Settings & Session", """
<h1>Settings</h1>
<h2>Analysis Parameters</h2>
<ul>
  <li><b>Default pH</b> — pH for net-charge calculations (0–14).</li>
  <li><b>Sliding Window Size</b> — window width for hydrophobicity, NCPR, and entropy profiles.</li>
  <li><b>Override pKa</b> — nine comma-separated values: N-term, C-term, D, E, C, Y, H, K, R.
      Leave blank to use the built-in defaults.</li>
  <li><b>Reducing conditions</b> — if checked, Cys residues are not counted as disulphide pairs for extinction coefficient.</li>
</ul>
<h2>Graph Appearance</h2>
<ul>
  <li><b>Label / Tick Font Size</b> — point size of axis titles and tick labels.</li>
  <li><b>Default Graph Format</b> — PNG, SVG, or PDF for Save Graph and Save All Graphs.</li>
  <li><b>Bead Colormap</b> — 30+ matplotlib colourmap choices for the Bead Hydrophobicity model.</li>
  <li><b>Graph Accent Colour</b> — 24 named colours for the primary line/fill of most graphs.
      Applying this setting immediately re-renders all graphs.</li>
  <li><b>Transparent background</b> — export graphs with a transparent background (PNG/SVG only). On by default.</li>
</ul>
<h2>Interface</h2>
<ul>
  <li><b>UI Font Size</b> — global font size in points (8–24).</li>
  <li><b>Dark Theme</b> — toggles between light and dark colour themes.</li>
  <li><b>Enable Tooltips</b> — show hover tooltips on Settings widgets. On by default.</li>
</ul>
<h2>Reset to Defaults</h2>
<p>Restores all settings to their factory defaults and re-applies them immediately.</p>
<h1>Sessions</h1>
<p>Use <b>Save Session</b> / <b>Load Session</b> (or Ctrl+S / Ctrl+O) to persist the current
sequence, name, pH, window size, pKa overrides, reducing conditions, font sizes, and
transparency setting in a <tt>.beer</tt> JSON file.</p>
"""),
        ]

        for section_name, html_body in _HELP_SECTIONS:
            help_nav.addItem(QListWidgetItem(section_name))
            page   = QWidget()
            page_v = QVBoxLayout(page)
            page_v.setContentsMargins(0, 0, 0, 0)
            browser = QTextBrowser()
            browser.setOpenExternalLinks(True)
            full_html = (
                f"<style>{REPORT_CSS} body{{padding:12px;}}</style>"
                + html_body
            )
            browser.setHtml(full_html)
            page_v.addWidget(browser)
            help_stack.addWidget(page)

        help_nav.currentRowChanged.connect(help_stack.setCurrentIndex)
        help_nav.setCurrentRow(0)

    # --- Import ---

    def import_fasta(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open FASTA File", "", "FASTA Files (*.fa *.fasta)")
        if not file_name:
            return
        try:
            records = list(SeqIO.parse(file_name, "fasta"))
        except Exception as e:
            QMessageBox.critical(self, "FASTA Error", f"Failed to parse file: {e}")
            return
        if not records:
            QMessageBox.warning(self, "No Records", "No sequences found.")
            return
        entries = [(rec.id, clean_sequence(str(rec.seq))) for rec in records]
        self._load_batch(entries)
        # Set default name to first record id (unless user overrode it)
        if not self.sequence_name:
            self.sequence_name = entries[0][0] if entries else ""

    def import_pdb(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open PDB File", "", "PDB Files (*.pdb)")
        if not file_name:
            return
        try:
            chains = import_pdb_sequence(file_name)
        except RuntimeError as e:
            QMessageBox.critical(self, "PDB Error", str(e))
            return
        if not chains:
            QMessageBox.warning(self, "No Chains", "No valid chains found.")
            return
        try:
            with open(file_name, "r") as fh:
                pdb_str = fh.read()
        except OSError:
            pdb_str = None
        pdb_base = os.path.splitext(os.path.basename(file_name))[0]
        entries  = [(f"{pdb_base}_{cid}", seq) for cid, seq in chains.items()]
        self._load_batch(entries)  # resets batch_struct / alphafold_data
        # Compute and store per-chain structure data so Ramachandran plot,
        # distance map, pLDDT profile and 3D viewer work for uploaded PDBs.
        if pdb_str:
            chain_structs = extract_chain_structures(pdb_str)
            for cid_letter, struct in chain_structs.items():
                rec_id = f"{pdb_base}_{cid_letter}"
                self.batch_struct[rec_id] = struct
            # Load the first chain into the 3D viewer and analysis view.
            first_id = entries[0][0]
            if first_id in self.batch_struct:
                self.alphafold_data = self.batch_struct[first_id]
                self._load_structure_viewer(self.alphafold_data["pdb_str"])
                self.save_pdb_btn.setEnabled(True)
        self.sequence_name = entries[0][0] if entries else pdb_base
        # Auto-populate the sequence viewer and run graphs immediately.
        # _load_batch already analysed every chain; use the first chain's data.
        if self.batch_data:
            first_id, first_seq, first_data = self.batch_data[0]
            self.seq_text.setPlainText(first_seq)
            self.analysis_data = first_data
            self._update_seq_viewer()
            self.update_graph_tabs()
        n_chains = len(self.batch_data)
        chain_word = "chain" if n_chains == 1 else "chains"
        self.statusBar.showMessage(
            f"Loaded {os.path.basename(file_name)}  —  {n_chains} {chain_word}", 4000)

    # --- Analysis ---

    def sort_composition(self, mode: str):
        if not self.analysis_data:
            return
        counts = self.analysis_data["aa_counts"]
        freq   = self.analysis_data["aa_freq"]
        items  = list(counts.items())
        if mode == "alpha":
            items.sort(key=lambda x: x[0])
        elif mode == "composition":
            items.sort(key=lambda x: freq[x[0]], reverse=True)
        elif mode == "hydro_inc":
            items.sort(key=lambda x: KYTE_DOOLITTLE[x[0]])
        else:
            items.sort(key=lambda x: KYTE_DOOLITTLE[x[0]], reverse=True)

        html = (
            f"<style>{REPORT_CSS}</style>"
            "<h2>Composition</h2>"
            "<table>"
            "<tr><th>Amino Acid</th><th>Count</th><th>Frequency (%)</th></tr>"
            + "".join(
                f"<tr><td>{aa}</td><td>{cnt}</td><td>{freq[aa]:.2f}%</td></tr>"
                for aa, cnt in items
            )
            + "</table>"
        )
        self.report_section_tabs["Composition"].setHtml(html)

        # Refresh bar chart to match sort order
        fig = GraphingTools.create_amino_acid_composition_figure(
            dict(items), {aa: freq[aa] for aa, _ in items},
            label_font=self.label_font_size, tick_font=self.tick_font_size
        )
        self._replace_graph("Amino Acid Composition (Bar)", fig)

    @staticmethod
    def _parse_pasted_text(raw: str):
        """Parse pasted text into a list of (id, sequence) pairs.

        Accepts:
        - FASTA format (lines starting with '>'):
            >name
            SEQUENCE...
        - Multiple bare sequences on separate lines (each line = one sequence)
        - A single sequence (no newlines / single line)

        Returns list of (id, seq) tuples with only valid protein sequences.
        """
        raw = raw.strip()
        if not raw:
            return []

        entries = []

        if ">" in raw:
            # FASTA format
            current_id  = None
            current_seq = []
            for line in raw.splitlines():
                line = line.strip()
                if line.startswith(">"):
                    if current_id is not None and current_seq:
                        entries.append((current_id, "".join(current_seq).upper()))
                    current_id  = line[1:].split()[0] or f"seq{len(entries)+1}"
                    current_seq = []
                elif line:
                    current_seq.append(line.replace(" ", "").replace("\t", ""))
            if current_id is not None and current_seq:
                entries.append((current_id, "".join(current_seq).upper()))
        else:
            # One or more bare sequences, one per non-empty line
            lines = [ln.strip().replace(" ", "").upper()
                     for ln in raw.splitlines() if ln.strip()]
            if len(lines) == 1:
                entries = [("Sequence", lines[0])]
            else:
                for i, ln in enumerate(lines, 1):
                    entries.append((f"Seq{i}", ln))

        # Validate and filter
        valid = [(rid, seq) for rid, seq in entries if seq and is_valid_protein(seq)]
        return valid

    def on_analyze(self):
        raw = self.seq_text.toPlainText()
        if not raw.strip():
            QMessageBox.warning(self, "Input", "Enter or paste a sequence.")
            return

        try:
            pH = float(self.ph_input.text())
        except ValueError:
            pH = 7.0

        entries = self._parse_pasted_text(raw)
        if not entries:
            QMessageBox.warning(
                self, "Invalid Input",
                "No valid protein sequences found.\n"
                "Ensure sequences contain only standard amino acid letters (ACDEFGHIKLMNPQRSTVWY)."
            )
            return

        if len(entries) > 1:
            # Multiple sequences → load as batch and show first
            self._load_batch(entries)
            if not self.sequence_name:
                self.sequence_name = entries[0][0]
            self.statusBar.showMessage(
                f"Loaded {len(entries)} sequences into Multichain Analysis", 4000
            )
            QMessageBox.information(
                self, "Batch Loaded",
                f"{len(entries)} sequences detected and loaded.\n"
                "Use the Multichain Analysis tab or the chain selector to navigate."
            )
            return

        # Single sequence → run in worker thread
        seq = entries[0][1]
        if not self.sequence_name:
            self.sequence_name = entries[0][0]

        # Clear chain combo when manually typing
        if not self.batch_data:
            self.chain_combo.clear()
            self.chain_combo.setEnabled(False)

        self.analyze_btn.setEnabled(False)
        self.statusBar.showMessage("Analyzing…")

        self._analysis_worker = AnalysisWorker(
            seq, pH, self.default_window_size, self.use_reducing, self.custom_pka
        )
        self._analysis_worker.finished.connect(self._on_worker_finished)
        self._analysis_worker.error.connect(self._on_worker_error)
        self._analysis_worker.start()

    def _update_seq_viewer(self, highlight_pattern: str = ""):
        """Refresh the sequence viewer panel with colour-coded residues (UniProt style)."""
        if not self.analysis_data:
            return
        seq  = self.analysis_data["seq"]
        name = self.sequence_name or ""
        text = format_sequence_block(seq, name=name)

        # Pre-compile highlight pattern
        hl_re = None
        if highlight_pattern:
            try:
                hl_re = re.compile(highlight_pattern.upper())
            except re.error:
                hl_re = None

        def _colour_residues(chunk: str) -> str:
            """Wrap each residue letter in a coloured span, with optional highlight."""
            parts = []
            for aa in chunk:
                if aa == " ":
                    parts.append("&nbsp;")
                    continue
                col = _AA_COLOURS.get(aa, "#1a1a2e")
                parts.append(f'<span style="color:{col};">{aa}</span>')
            result = "".join(parts)
            # Now apply motif highlighting on top (background colour)
            if hl_re:
                # Rebuild plain text chunk for regex matching, then wrap matches
                positions = [m.span() for m in hl_re.finditer(chunk.replace(" ", ""))]
                if positions:
                    # Re-colourise with highlight bg
                    idx = 0
                    new_parts = []
                    for aa in chunk:
                        if aa == " ":
                            new_parts.append("&nbsp;")
                            continue
                        col = _AA_COLOURS.get(aa, "#1a1a2e")
                        # Check if this plain-text index falls in a match
                        in_match = any(lo <= idx < hi for lo, hi in positions)
                        if in_match:
                            new_parts.append(
                                f'<span style="color:{col};background:#fef08a;'
                                f'border-radius:2px;">{aa}</span>'
                            )
                        else:
                            new_parts.append(f'<span style="color:{col};">{aa}</span>')
                        idx += 1
                    result = "".join(new_parts)
            return result

        lines      = text.split("\n")
        html_lines = []
        for ln in lines:
            if ln.startswith(">"):
                html_lines.append(
                    f'<span style="color:#4361ee;font-weight:700;">{ln}</span>'
                )
            elif ln and ln.lstrip()[0:1].isdigit():
                parts = ln.split("  ", 1)
                if len(parts) == 2:
                    pos_str, seq_str = parts
                    coloured = _colour_residues(seq_str)
                    html_lines.append(
                        f'<span style="color:#718096;">{pos_str}</span>'
                        f'&nbsp;&nbsp;{coloured}'
                    )
                else:
                    html_lines.append(_colour_residues(ln))
            else:
                html_lines.append(f'<span style="color:#1a1a2e;">{ln}</span>')
        html = (
            '<style>body{font-family:"Courier New",monospace;font-size:10pt;'
            'background:#f8f9fd;padding:8px;line-height:2.0;}</style>'
            + "<br>".join(html_lines)
        )
        self.seq_viewer.setHtml(html)

    def update_graph_tabs(self):
        if not self.analysis_data:
            return
        seq  = self.analysis_data["seq"]
        lf   = self.label_font_size
        tf   = self.tick_font_size
        figs = {
            "Amino Acid Composition (Bar)": GraphingTools.create_amino_acid_composition_figure(
                self.analysis_data["aa_counts"], self.analysis_data["aa_freq"],
                label_font=lf, tick_font=tf),
            "Amino Acid Composition (Pie)": GraphingTools.create_amino_acid_composition_pie_figure(
                self.analysis_data["aa_counts"], label_font=lf),
            "Hydrophobicity Profile": GraphingTools.create_hydrophobicity_figure(
                self.analysis_data["hydro_profile"], self.analysis_data["window_size"],
                label_font=lf, tick_font=tf),
            "Net Charge vs pH": GraphingTools.create_net_charge_vs_pH_figure(
                seq, label_font=lf, tick_font=tf, pka=self.custom_pka),
            "Bead Model (Hydrophobicity)": GraphingTools.create_bead_model_hydrophobicity_figure(
                seq, self.show_bead_labels, label_font=lf, tick_font=tf, cmap=self.colormap),
            "Bead Model (Charge)": GraphingTools.create_bead_model_charge_figure(
                seq, self.show_bead_labels, label_font=lf, tick_font=tf),
            "Sticker Map": GraphingTools.create_sticker_map(
                seq, self.show_bead_labels, label_font=lf, tick_font=tf),
            "Local Charge Profile": GraphingTools.create_local_charge_figure(
                self.analysis_data["ncpr_profile"], self.analysis_data["window_size"],
                label_font=lf, tick_font=tf),
            "Local Complexity": GraphingTools.create_local_complexity_figure(
                self.analysis_data["entropy_profile"], self.analysis_data["window_size"],
                label_font=lf, tick_font=tf),
            "Cation\u2013\u03c0 Map": GraphingTools.create_cation_pi_map(
                seq, label_font=lf, tick_font=tf),
            "Isoelectric Focus": GraphingTools.create_isoelectric_focus(
                seq, label_font=lf, tick_font=tf, pka=self.custom_pka),
            "Secondary Structure": GraphingTools.create_secondary_structure(
                self.analysis_data["cf_helix"], self.analysis_data["cf_sheet"],
                label_font=lf, tick_font=tf),
            "Helical Wheel": GraphingTools.create_helical_wheel(
                seq, label_font=lf),
            "Charge Decoration": GraphingTools.create_charge_decoration(
                self.analysis_data["fcr"], self.analysis_data["ncpr"],
                label_font=lf, tick_font=tf),
            "Linear Sequence Map": GraphingTools.create_linear_sequence_map(
                seq,
                self.analysis_data["hydro_profile"],
                self.analysis_data["ncpr_profile"],
                self.analysis_data["disorder_scores"],
                self.analysis_data["cf_helix"],
                label_font=lf, tick_font=tf),
            "Disorder Profile": GraphingTools.create_disorder_profile(
                self.analysis_data["disorder_scores"], label_font=lf, tick_font=tf),
        }

        # TM Topology is always available after analysis
        figs["TM Topology"] = GraphingTools.create_tm_topology_figure(
            seq, self.analysis_data.get("tm_helices", []),
            label_font=lf, tick_font=tf)

        # Phase separation / IDP graphs
        figs["Uversky Phase Plot"] = GraphingTools.create_uversky_phase_plot(
            seq, label_font=lf, tick_font=tf)
        if self.analysis_data.get("cc_profile"):
            figs["Coiled-Coil Profile"] = GraphingTools.create_coiled_coil_profile(
                self.analysis_data["cc_profile"], label_font=lf, tick_font=tf)
        figs["Saturation Mutagenesis"] = GraphingTools.create_saturation_mutagenesis_figure(
            seq, label_font=lf, tick_font=tf)

        # Structure-dependent graphs — available from any structure source
        # (PDB upload, RCSB PDB fetch, or AlphaFold fetch).
        if self.alphafold_data:
            plddt = self.alphafold_data.get("plddt")
            if plddt and len(plddt) == len(seq):
                figs["pLDDT Profile"] = GraphingTools.create_plddt_figure(
                    plddt, label_font=lf, tick_font=tf)
            dm = self.alphafold_data.get("dist_matrix")
            # Guard: dist_matrix must be square and match the analysed sequence length.
            # A mismatch occurs when the FASTA and PDB chain have different residue
            # counts (e.g. disordered tails absent from the structure).
            if dm is not None and dm.ndim == 2 and dm.shape[0] == len(seq) and dm.shape[0] > 0:
                figs["Distance Map"] = GraphingTools.create_distance_map_figure(
                    dm, label_font=lf, tick_font=tf)

        # Domain architecture — always rendered; shows all available tracks
        figs["Domain Architecture"] = GraphingTools.create_domain_architecture_figure(
            len(seq), self.pfam_domains,
            seq=seq,
            disorder_scores=self.analysis_data.get("disorder_scores"),
            tm_helices=self.analysis_data.get("tm_helices"),
            label_font=lf, tick_font=tf)

        # ── New feature graphs ────────────────────────────────────────────────
        # β-Aggregation & Solubility
        if _HAS_AGGREGATION:
            aggr_profile = calc_aggregation_profile(seq)
            hotspots     = predict_aggregation_hotspots(seq)
            figs["\u03b2-Aggregation Profile"] = GraphingTools.create_aggregation_profile_figure(
                seq, aggr_profile, hotspots, label_font=lf, tick_font=tf)
            camsolmt = calc_camsolmt_score(seq)
            figs["Solubility Profile"] = GraphingTools.create_solubility_profile_figure(
                seq, camsolmt, label_font=lf, tick_font=tf)

        if _HAS_AMPHIPATHIC:
            figs["Hydrophobic Moment"] = GraphingTools.create_hydrophobic_moment_figure(
                seq,
                self.analysis_data.get("moment_alpha", []),
                self.analysis_data.get("moment_beta", []),
                self.analysis_data.get("amph_regions", []),
                label_font=lf, tick_font=tf)

        if _HAS_PTM:
            figs["PTM Map"] = GraphingTools.create_ptm_profile_figure(
                seq, self.analysis_data.get("ptm_sites", []),
                label_font=lf, tick_font=tf)

        if _HAS_RBP:
            figs["RNA-Binding Profile"] = GraphingTools.create_rbp_profile_figure(
                seq,
                self.analysis_data.get("rbp_profile", []),
                self.analysis_data.get("rbp", {}).get("motifs_found", []),
                label_font=lf, tick_font=tf)

        if _HAS_SCD:
            figs["SCD Profile"] = GraphingTools.create_scd_profile_figure(
                seq, self.analysis_data.get("scd_profile", []),
                window=20, label_font=lf, tick_font=tf)

        # pI / MW Map — always available
        figs["pI / MW Map"] = GraphingTools.create_pI_MW_gel_figure(
            [{"name": self.sequence_name or "Protein",
              "pI":   self.analysis_data["iso_point"],
              "mol_weight": self.analysis_data["mol_weight"]}],
            label_font=lf, tick_font=tf)

        # Ramachandran plot — any loaded PDB structure (upload, RCSB fetch, AlphaFold)
        if self.alphafold_data and _HAS_PHI_PSI:
            phi_psi = _extract_phi_psi(self.alphafold_data["pdb_str"])
            figs["Ramachandran Plot"] = GraphingTools.create_ramachandran_figure(
                phi_psi, label_font=lf, tick_font=tf)

        # Residue contact network — requires distance matrix matching sequence length
        if self.alphafold_data:
            dm = self.alphafold_data.get("dist_matrix")
            if (dm is not None and dm.ndim == 2
                    and dm.shape[0] == len(seq) and dm.shape[0] > 0):
                figs["Residue Contact Network"] = GraphingTools.create_contact_network_figure(
                    seq, dm, label_font=lf, tick_font=tf)

        # MSA Conservation (requires MSA data)
        if self._msa_sequences:
            figs["MSA Conservation"] = GraphingTools.create_msa_conservation_figure(
                self._msa_sequences, self._msa_names,
                label_font=lf, tick_font=tf)

        # Apply global heading/grid/colour overrides
        for title, fig in figs.items():
            if title not in self.graph_tabs:
                continue  # skip graphs not in the tree (e.g. plugin graphs added later)
            if fig.axes:
                ax = fig.axes[0]
                if not self.show_heading:
                    ax.set_title("")
                ax.grid(self.show_grid)
            self._replace_graph(title, fig)

    def show_batch_details(self, row, _):
        sid = self.batch_table.item(row, 0).text()
        for cid, seq, data in self.batch_data:
            if cid == sid:
                self.seq_text.setPlainText(seq)
                self.analysis_data  = data
                self.sequence_name  = cid
                for sec, browser in self.report_section_tabs.items():
                    browser.setHtml(data["report_sections"][sec])
                self._restore_chain_structure(cid)
                self._update_seq_viewer()
                self.update_graph_tabs()
                return

    # --- Graph tree handler ---

    def _on_graph_tree_clicked(self, item: QTreeWidgetItem, _col: int):
        title = item.data(0, Qt.UserRole)
        if title and title in self._graph_title_to_stack_idx:
            self.graph_stack.setCurrentIndex(self._graph_title_to_stack_idx[title])

    # --- Export ---

    def export_pdf(self):
        if not self.analysis_data:
            QMessageBox.warning(self, "No Data", "Run analysis first.")
            return
        fn, _ = QFileDialog.getSaveFileName(self, "Export PDF", "", "PDF Files (*.pdf)")
        if fn:
            ExportTools.export_pdf(
                self.analysis_data, fn, self,
                seq_name=self.sequence_name
            )

    def save_graph(self, title: str):
        tab, vb = self.graph_tabs[title]
        canvas  = self._find_canvas(vb)
        if not canvas:
            QMessageBox.warning(self, "No Graph", "Graph not available.")
            return
        ext  = self.default_graph_format.lower()
        fn, _ = QFileDialog.getSaveFileName(
            self, "Save Graph", "", f"{self.default_graph_format} Files (*.{ext})"
        )
        if fn:
            if not fn.lower().endswith(f".{ext}"):
                fn += f".{ext}"
            use_transparent = self.transparent_bg and ext in ("png", "svg")
            canvas.figure.savefig(
                fn, format=ext, dpi=200, bbox_inches="tight",
                facecolor="none" if use_transparent else "white",
                transparent=use_transparent
            )
            QMessageBox.information(self, "Saved", f"{title} → {fn}")

    def save_all_graphs(self):
        d = QFileDialog.getExistingDirectory(self, "Select Directory")
        if not d:
            return
        ext = self.default_graph_format.lower()
        try:
            use_transparent = self.transparent_bg and ext in ("png", "svg")
            for title, (tab, vb) in self.graph_tabs.items():
                canvas = self._find_canvas(vb)
                if canvas:
                    safe = re.sub(r'[^\w\-]', '_', title)
                    path = os.path.join(d, safe + f".{ext}")
                    canvas.figure.savefig(
                        path, format=ext, dpi=200, bbox_inches="tight",
                        facecolor="none" if use_transparent else "white",
                        transparent=use_transparent,
                    )
            QMessageBox.information(self, "Saved", "All graphs exported.")
        except OSError as e:
            QMessageBox.critical(self, "Export Error", f"Could not save graphs: {e}")

    def export_batch_csv(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv)")
        if not fn or not self.batch_data:
            return
        try:
            with open(fn, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "ID", "Length", "MW (Da)", "Net Charge (pH 7)",
                    "% Hydro", "% Hydrophil", "% +Charged", "% -Charged", "% Neutral",
                ])
                for cid, seq, data in self.batch_data:
                    hydro, hydrophil, pos, neg, neu = _calc_batch_stats(seq, data)
                    writer.writerow([
                        cid, str(len(seq)), f"{data['mol_weight']:.2f}",
                        f"{data['net_charge_7']:.2f}",
                        f"{hydro:.1f}%", f"{hydrophil:.1f}%",
                        f"{pos:.1f}%", f"{neg:.1f}%", f"{neu:.1f}%",
                    ])
            QMessageBox.information(self, "Saved", "Batch CSV exported.")
        except OSError as e:
            QMessageBox.critical(self, "Export Error", f"Could not write file: {e}")

    def export_batch_json(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Save JSON", "", "JSON Files (*.json)")
        if not fn or not self.batch_data:
            return
        try:
            out = []
            for cid, seq, data in self.batch_data:
                hydro, hydrophil, pos, neg, neu = _calc_batch_stats(seq, data)
                out.append({
                    "ID":                cid,
                    "Length":            len(seq),
                    "MW (Da)":           f"{data['mol_weight']:.2f}",
                    "Net Charge (pH 7)": f"{data['net_charge_7']:.2f}",
                    "% Hydro":           f"{hydro:.1f}%",
                    "% Hydrophil":       f"{hydrophil:.1f}%",
                    "% +Charged":        f"{pos:.1f}%",
                    "% -Charged":        f"{neg:.1f}%",
                    "% Neutral":         f"{neu:.1f}%",
                })
            with open(fn, "w") as f:
                json.dump(out, f, indent=2)
            QMessageBox.information(self, "Saved", "Batch JSON exported.")
        except OSError as e:
            QMessageBox.critical(self, "Export Error", f"Could not write file: {e}")

    # --- Settings ---

    def toggle_theme(self):
        if self.theme_toggle.isChecked():
            self.setStyleSheet(DARK_THEME_CSS)
            plt.style.use("dark_background")
        else:
            self.setStyleSheet(LIGHT_THEME_CSS)
            plt.style.use("default")
        label = "Dark" if self.theme_toggle.isChecked() else "Light"
        self.statusBar.showMessage(f"{label} theme activated", 2000)

    def apply_settings(self):
        try:
            self.default_window_size = int(self.window_size_input.text())
        except (ValueError, TypeError):
            pass
        try:
            self.default_pH = float(self.ph_input.text())
        except (ValueError, TypeError):
            pass
        self.show_bead_labels = self.label_checkbox.isChecked()
        self.transparent_bg   = self.transparent_bg_checkbox.isChecked()
        self.colormap         = self.colormap_combo.currentText()
        try:
            self.label_font_size = int(self.label_font_input.text())
            self.tick_font_size  = int(self.tick_font_input.text())
            self.marker_size     = int(self.marker_size_input.text())
        except (ValueError, TypeError):
            pass
        self.graph_color = NAMED_COLORS.get(self.graph_color_combo.currentText(), "#4361ee")
        # Propagate accent colour to module-level graph constants
        global _ACCENT, _FILL, _POS_COL
        _ACCENT  = self.graph_color
        _FILL    = self.graph_color
        _POS_COL = self.graph_color
        self.show_heading         = self.heading_checkbox.isChecked()
        self.show_grid            = self.grid_checkbox.isChecked()
        self.default_graph_format = self.graph_format_combo.currentText()
        self.use_reducing         = self.reducing_checkbox.isChecked()

        # Sequence name override
        name_override = self.seq_name_input.text().strip()
        if name_override:
            self.sequence_name = name_override

        # Global UI font size
        try:
            fs = int(self.app_font_size_input.text())
            if 8 <= fs <= 24:
                self.app_font_size = fs
                app_inst = QApplication.instance()
                if app_inst:
                    f = app_inst.font()
                    f.setPointSize(fs)
                    app_inst.setFont(f)
        except (ValueError, TypeError):
            pass

        raw_pka = [p.strip() for p in self.pka_input.text().split(",") if p.strip()]
        self.custom_pka = None
        if len(raw_pka) == 9:
            try:
                vals = list(map(float, raw_pka))
                self.custom_pka = {
                    'NTERM': vals[0], 'CTERM': vals[1], 'D': vals[2], 'E': vals[3],
                    'C': vals[4], 'Y': vals[5], 'H': vals[6], 'K': vals[7], 'R': vals[8],
                }
            except ValueError:
                QMessageBox.warning(self, "pKa list",
                                    "Custom pKa list could not be parsed – using defaults.")

        self.enable_tooltips = self.tooltips_checkbox.isChecked()
        self._apply_tooltips()

        if self.theme_toggle.isChecked():
            self.setStyleSheet(DARK_THEME_CSS)
        else:
            self.setStyleSheet(LIGHT_THEME_CSS)

        if self.analysis_data:
            for sec, browser in self.report_section_tabs.items():
                browser.setHtml(self.analysis_data["report_sections"][sec])
            self._update_seq_viewer()
            self.update_graph_tabs()
        self.statusBar.showMessage("Settings applied", 2000)

    def reset_defaults(self):
        self.window_size_input.setText("9")
        self.ph_input.setText("7.0")
        self.pka_input.setText("")
        self.reducing_checkbox.setChecked(False)
        self.label_checkbox.setChecked(True)
        self.colormap_combo.setCurrentText("coolwarm")
        self.label_font_input.setText("14")
        self.tick_font_input.setText("12")
        self.marker_size_input.setText("10")
        self.graph_color_combo.setCurrentText("Royal Blue")
        self.graph_format_combo.setCurrentText("PNG")
        self.heading_checkbox.setChecked(True)
        self.grid_checkbox.setChecked(True)
        self.transparent_bg_checkbox.setChecked(True)
        self.app_font_size_input.setText("12")
        self.theme_toggle.setChecked(False)
        self.tooltips_checkbox.setChecked(True)
        self.apply_settings()

    # --- Chain selection ---

    def _restore_chain_structure(self, cid: str):
        """Update alphafold_data and the 3D viewer for the given chain ID.

        When structure data was loaded per-chain (PDB upload / RCSB PDB fetch),
        batch_struct maps each rec_id to its own struct dict so every chain gets
        its own Ramachandran plot, distance map, pLDDT profile and 3D view.

        When batch_struct is empty (pure sequence mode, e.g. UniProt accession
        with a separately fetched AlphaFold model), alphafold_data is left
        untouched so the single fetched structure continues to be displayed.
        """
        if not self.batch_struct:
            # Pure-sequence batch — AlphaFold data (if any) was fetched explicitly
            # for the current single sequence; leave it intact.
            return
        struct = self.batch_struct.get(cid)
        self.alphafold_data = struct
        if struct:
            self._load_structure_viewer(struct["pdb_str"])
            self.save_pdb_btn.setEnabled(True)
        else:
            self.save_pdb_btn.setEnabled(False)

    def on_chain_selected(self, text: str):
        for cid, seq, data in self.batch_data:
            if cid == text:
                self.seq_text.setPlainText(seq)
                self.analysis_data = data
                self.sequence_name = cid
                for sec, browser in self.report_section_tabs.items():
                    browser.setHtml(data["report_sections"][sec])
                self._restore_chain_structure(cid)
                self._update_seq_viewer()
                self.update_graph_tabs()
                break

    # --- Keyboard shortcuts ---

    def _setup_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+Return"), self, self.on_analyze)
        QShortcut(QKeySequence("Ctrl+E"),      self, self.export_pdf)
        QShortcut(QKeySequence("Ctrl+G"),      self,
                  lambda: self.main_tabs.setCurrentIndex(1))
        QShortcut(QKeySequence("Ctrl+S"),      self, self.session_save)
        QShortcut(QKeySequence("Ctrl+O"),      self, self.session_load)
        QShortcut(QKeySequence("Ctrl+F"),      self,
                  lambda: self.motif_input.setFocus())

    # --- Worker callbacks ---

    def _on_worker_finished(self, data: dict):
        seq  = data["seq"]
        self._run_plugins(seq, data)
        self.analysis_data = data
        self._add_to_history(self.sequence_name, seq)
        for sec, browser in self.report_section_tabs.items():
            if sec in data["report_sections"]:
                browser.setHtml(data["report_sections"][sec])
        self._update_seq_viewer()
        self.update_graph_tabs()
        self.analyze_btn.setEnabled(True)
        self.statusBar.showMessage(
            f"Analysis complete  |  {len(seq)} aa  |  {self.sequence_name}", 4000
        )

    def _on_worker_error(self, msg: str):
        self.analyze_btn.setEnabled(True)
        self.statusBar.showMessage("Analysis failed", 3000)
        QMessageBox.critical(self, "Analysis Error", msg)

    # --- History ---

    def _add_to_history(self, name: str, seq: str):
        # Avoid duplicates by sequence
        self._history = [(n, s) for n, s in self._history if s != seq]
        self._history.insert(0, (name or "Sequence", seq))
        self._history = self._history[:10]
        # Rebuild combo
        self.history_combo.blockSignals(True)
        self.history_combo.clear()
        self.history_combo.addItem("— recent sequences —")
        for n, _ in self._history:
            self.history_combo.addItem(n)
        self.history_combo.setCurrentIndex(0)
        self.history_combo.blockSignals(False)

    def _on_history_selected(self, idx: int):
        if idx <= 0:
            return
        name, seq = self._history[idx - 1]
        self.seq_text.setPlainText(seq)
        self.sequence_name = name
        self.history_combo.setCurrentIndex(0)
        self.on_analyze()

    # --- Accession fetch ---

    def fetch_accession(self):
        acc = self.accession_input.text().strip()
        if not acc:
            QMessageBox.warning(self, "Fetch", "Enter a UniProt ID or PDB ID.")
            return
        # Detect PDB ID: exactly 4 alphanumeric chars, first char is a digit
        is_pdb = (len(acc) == 4 and acc[0].isdigit() and acc.isalnum())
        self.statusBar.showMessage(f"Fetching {acc}…")
        try:
            if is_pdb:
                raw = self._fetch_pdb_fasta(acc)
            else:
                url = f"https://rest.uniprot.org/uniprotkb/{acc}.fasta"
                with urllib.request.urlopen(url, timeout=15) as resp:
                    raw = resp.read().decode()
        except Exception as e:
            self.statusBar.showMessage("Fetch failed", 3000)
            QMessageBox.warning(self, "Fetch Failed", str(e))
            return
        entries = self._parse_pasted_text(raw)
        if not entries:
            QMessageBox.warning(self, "Fetch", "No valid protein sequence returned.")
            return
        if is_pdb:
            # Use the first field of the RCSB FASTA header (e.g. "4HHB_1") as chain label,
            # then load ALL chains into the multichain table exactly as import_pdb() does.
            tagged = [(rid.split("|")[0], seq) for rid, seq in entries]
            self._load_batch(tagged)  # resets batch_struct / alphafold_data
            rid, seq = tagged[0]
            # Also fetch the actual PDB coordinate file so structure-dependent
            # graphs (Ramachandran, distance map, pLDDT, 3D viewer) are available.
            self.statusBar.showMessage(f"Downloading PDB structure for {acc.upper()}…")
            try:
                pdb_str = self._fetch_pdb_structure(acc)
                chain_structs = extract_chain_structures(pdb_str)
                # Map RCSB chain letter → tagged rec_id  (tagged labels are "4HHB_1", "4HHB_2"…)
                # The FASTA headers tell us which chain letters correspond to each entity.
                # We build the mapping by matching tagged index to chain order.
                chain_letters = list(chain_structs.keys())
                for i, (rec_id, _) in enumerate(tagged):
                    # RCSB FASTA entity i may group multiple identical chains; we
                    # associate it with the first chain letter for that entity index.
                    if i < len(chain_letters):
                        self.batch_struct[rec_id] = chain_structs[chain_letters[i]]
                if tagged[0][0] in self.batch_struct:
                    self.alphafold_data = self.batch_struct[tagged[0][0]]
                    self._load_structure_viewer(self.alphafold_data["pdb_str"])
                    self.save_pdb_btn.setEnabled(True)
            except Exception:
                pass  # Structure fetch is best-effort; sequences are already loaded
        else:
            rid, seq = entries[0]
        self.seq_text.setPlainText(seq)
        self.sequence_name = rid
        # Store accession; AlphaFold/Pfam/ELM/DisProt/PhaSepDB need a UniProt ID
        self.current_accession = acc if not is_pdb else ""
        self.fetch_af_btn.setEnabled(True)
        self.fetch_pfam_btn.setEnabled(True)
        self.fetch_elm_btn.setEnabled(not is_pdb)
        self.fetch_disprot_btn.setEnabled(not is_pdb)
        self.fetch_phasepdb_btn.setEnabled(not is_pdb)
        self.accession_input.clear()
        src = "PDB" if is_pdb else "UniProt"
        msg = f"Fetched {rid} from {src}  ({len(seq)} aa)"
        if is_pdb and len(entries) > 1:
            msg += f"  \u2014 {len(entries)} chains loaded"
        self.statusBar.showMessage(msg, 4000)

        if is_pdb:
            # Analysis was done inside _load_batch; activate the first chain immediately.
            if self.batch_data:
                _, _, first_data = self.batch_data[0]
                self.analysis_data = first_data
                self._update_seq_viewer()
                self.update_graph_tabs()
        else:
            # UniProt single sequence: start the analysis worker.
            self.on_analyze()

    def _fetch_pdb_fasta(self, pdb_id: str) -> str:
        """Fetch FASTA sequence(s) from RCSB PDB for a given 4-char PDB ID."""
        url = f"https://www.rcsb.org/fasta/entry/{pdb_id.upper()}"
        req = urllib.request.Request(url, headers={"User-Agent": "BEER/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode()

    def _fetch_pdb_structure(self, pdb_id: str) -> str:
        """Download the PDB coordinate file from RCSB for a given 4-char PDB ID."""
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        req = urllib.request.Request(url, headers={"User-Agent": "BEER/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode()

    # --- AlphaFold ---

    def fetch_alphafold(self):
        acc = self.current_accession
        if not acc:
            acc, ok = QInputDialog.getText(
                self, "AlphaFold — UniProt ID Required",
                "Enter a UniProt accession (e.g. P04637):"
            )
            if not ok or not acc.strip():
                return
            acc = acc.strip()
            self.current_accession = acc
        if self._alphafold_worker and self._alphafold_worker.isRunning():
            return
        self.fetch_af_btn.setEnabled(False)
        self._alphafold_worker = AlphaFoldWorker(acc)
        self._alphafold_worker.progress.connect(
            lambda msg: self.statusBar.showMessage(msg))
        self._alphafold_worker.finished.connect(self._on_alphafold_finished)
        self._alphafold_worker.error.connect(self._on_alphafold_error)
        self._alphafold_worker.start()

    def _on_alphafold_finished(self, data: dict):
        self.alphafold_data = data
        # Register in batch_struct so the structure persists when the user switches
        # chains and then switches back to this sequence.
        if self.sequence_name:
            self.batch_struct[self.sequence_name] = data
        self.fetch_af_btn.setEnabled(True)
        self.save_pdb_btn.setEnabled(True)
        n_res = len(data.get("plddt", []))
        mean_plddt = (sum(data["plddt"]) / n_res) if n_res else 0
        self.af_status_lbl.setText(
            f"Loaded AlphaFold structure for {data['accession']}  "
            f"({n_res} residues, mean pLDDT = {mean_plddt:.1f})"
        )
        self.af_status_lbl.setStyleSheet("color:#43aa8b; font-weight:600;")
        self._load_structure_viewer(data["pdb_str"])
        if self.analysis_data:
            self.update_graph_tabs()
        self.statusBar.showMessage(
            f"AlphaFold structure loaded  ({data['accession']})", 4000)

    def _on_alphafold_error(self, msg: str):
        self.fetch_af_btn.setEnabled(True)
        self.statusBar.showMessage("AlphaFold fetch failed", 3000)
        QMessageBox.warning(self, "AlphaFold Error", msg)

    # --- Pfam ---

    def fetch_pfam(self):
        acc = self.current_accession
        if not acc:
            acc, ok = QInputDialog.getText(
                self, "Pfam — UniProt ID Required",
                "Enter a UniProt accession (e.g. P04637):"
            )
            if not ok or not acc.strip():
                return
            acc = acc.strip()
            self.current_accession = acc
        if self._pfam_worker and self._pfam_worker.isRunning():
            return
        self.fetch_pfam_btn.setEnabled(False)
        self.statusBar.showMessage(f"Fetching Pfam domains for {acc}…")
        self._pfam_worker = PfamWorker(acc)
        self._pfam_worker.finished.connect(self._on_pfam_finished)
        self._pfam_worker.error.connect(self._on_pfam_error)
        self._pfam_worker.start()

    def _on_pfam_finished(self, domains: list):
        self.pfam_domains = domains
        self.fetch_pfam_btn.setEnabled(True)
        if not domains:
            self.statusBar.showMessage("No Pfam domains found.", 3000)
            QMessageBox.information(self, "Pfam", "No Pfam domain annotations found.")
            return
        if self.analysis_data:
            self.update_graph_tabs()
        self.statusBar.showMessage(
            f"Loaded {len(domains)} Pfam domain(s).", 4000)

    def _on_pfam_error(self, msg: str):
        self.fetch_pfam_btn.setEnabled(True)
        self.statusBar.showMessage("Pfam fetch failed", 3000)
        QMessageBox.warning(self, "Pfam Error", msg)

    # --- BLAST ---

    def run_blast(self):
        if not self.analysis_data:
            QMessageBox.warning(self, "BLAST", "Run analysis first.")
            return
        if self._blast_worker and self._blast_worker.isRunning():
            QMessageBox.information(self, "BLAST", "A BLAST search is already running.")
            return
        seq = self.analysis_data["seq"]
        db  = self.blast_db_combo.currentText()
        n   = self.blast_hits_spin.value()
        self.blast_run_btn.setEnabled(False)
        self.blast_table.setRowCount(0)
        self._blast_worker = BlastWorker(seq, database=db, hitlist_size=n)
        self._blast_worker.progress.connect(
            lambda msg: self.blast_status_lbl.setText(msg))
        self._blast_worker.finished.connect(self._on_blast_finished)
        self._blast_worker.error.connect(self._on_blast_error)
        self._blast_worker.start()

    def _on_blast_finished(self, hits: list):
        self.blast_run_btn.setEnabled(True)
        self.blast_status_lbl.setText(f"{len(hits)} hit(s) returned.")
        self.blast_table.setRowCount(0)
        for hit in hits:
            row = self.blast_table.rowCount()
            self.blast_table.insertRow(row)
            self.blast_table.setItem(row, 0, QTableWidgetItem(hit["accession"]))
            self.blast_table.setItem(row, 1, QTableWidgetItem(hit["title"][:80]))
            self.blast_table.setItem(row, 2, QTableWidgetItem(str(hit["length"])))
            self.blast_table.setItem(row, 3, QTableWidgetItem(f"{hit['score']:.0f}"))
            self.blast_table.setItem(row, 4, QTableWidgetItem(f"{hit['e_value']:.2e}"))
            self.blast_table.setItem(row, 5, QTableWidgetItem(f"{hit['identity']:.1f}%"))
            load_btn = QPushButton("Load")
            load_btn.clicked.connect(
                lambda _, h=hit: self._load_blast_hit(h))
            self.blast_table.setCellWidget(row, 6, load_btn)
        self.blast_table.resizeColumnsToContents()
        self.statusBar.showMessage(f"BLAST complete — {len(hits)} hits", 4000)

    def _on_blast_error(self, msg: str):
        self.blast_run_btn.setEnabled(True)
        self.blast_status_lbl.setText(f"Error: {msg}")
        QMessageBox.warning(self, "BLAST Error", msg)

    def _load_blast_hit(self, hit: dict):
        seq = hit.get("subject", "")
        if not seq or not is_valid_protein(seq):
            QMessageBox.warning(self, "Load Hit", "Subject sequence is not a valid protein.")
            return
        self.seq_text.setPlainText(seq)
        self.sequence_name = hit["accession"]
        self.main_tabs.setCurrentIndex(0)
        self.on_analyze()

    # --- Mutation tool ---

    def open_mutation_dialog(self):
        if not self.analysis_data:
            QMessageBox.warning(self, "Mutate", "Run analysis first.")
            return
        seq = self.analysis_data["seq"]
        dlg = MutationDialog(seq, self)
        if dlg.exec_() != QDialog.Accepted:
            return
        pos, new_aa = dlg.get_mutation()
        mutated = seq[:pos] + new_aa + seq[pos + 1:]
        old_aa  = seq[pos]
        self.seq_text.setPlainText(mutated)
        self.statusBar.showMessage(
            f"Mutated position {pos+1}: {old_aa} → {new_aa}", 3000
        )
        self.on_analyze()

    # --- Motif search ---

    def highlight_motif(self):
        if not self.analysis_data:
            return
        pattern = self.motif_input.text().strip()
        if not pattern:
            self._update_seq_viewer()
            return
        seq = self.analysis_data["seq"]
        try:
            matches = list(re.finditer(pattern.upper(), seq))
        except re.error as e:
            self.statusBar.showMessage(f"Invalid regex: {e}", 3000)
            return
        self._update_seq_viewer(highlight_pattern=pattern)
        self.statusBar.showMessage(
            f"{len(matches)} match(es) found for '{pattern}'", 3000
        )

    def clear_motif_highlight(self):
        self.motif_input.clear()
        self._update_seq_viewer()

    # --- Sequence comparison ---

    def do_compare(self):
        raw_a = self.compare_seq_a.toPlainText()
        raw_b = self.compare_seq_b.toPlainText()
        entries_a = self._parse_pasted_text(raw_a)
        entries_b = self._parse_pasted_text(raw_b)
        if not entries_a or not entries_b:
            QMessageBox.warning(self, "Compare",
                                "Both Sequence A and B must be valid protein sequences.")
            return
        seq_a = entries_a[0][1]
        seq_b = entries_b[0][1]
        da = AnalysisTools.analyze_sequence(seq_a)
        db = AnalysisTools.analyze_sequence(seq_b)

        props = [
            ("Length (aa)",           str(len(seq_a)),           str(len(seq_b))),
            ("Molecular Weight (Da)",  f"{da['mol_weight']:.2f}", f"{db['mol_weight']:.2f}"),
            ("Isoelectric Point (pI)", f"{da['iso_point']:.2f}",  f"{db['iso_point']:.2f}"),
            ("GRAVY Score",            f"{da['gravy']:.3f}",      f"{db['gravy']:.3f}"),
            ("FCR",                    f"{da['fcr']:.3f}",        f"{db['fcr']:.3f}"),
            ("NCPR",                   f"{da['ncpr']:+.3f}",      f"{db['ncpr']:+.3f}"),
            ("Net Charge (pH 7)",      f"{da['net_charge_7']:.2f}", f"{db['net_charge_7']:.2f}"),
            ("Instability Index",      f"{da['instability']:.2f}", f"{db['instability']:.2f}"),
            ("Aromaticity",            f"{da['aromaticity']:.3f}", f"{db['aromaticity']:.3f}"),
            ("Extinction Coeff.",      str(da['extinction']),     str(db['extinction'])),
        ]
        self.compare_table.setRowCount(len(props))
        for row, (prop, va, vb) in enumerate(props):
            self.compare_table.setItem(row, 0, QTableWidgetItem(prop))
            self.compare_table.setItem(row, 1, QTableWidgetItem(va))
            self.compare_table.setItem(row, 2, QTableWidgetItem(vb))
        self.compare_table.resizeColumnsToContents()
        self.statusBar.showMessage("Comparison complete", 3000)

    # --- Copy table to clipboard ---

    def _copy_section(self, sec: str):
        browser = self.report_section_tabs.get(sec)
        if not browser:
            return
        text = browser.toPlainText()
        QApplication.clipboard().setText(text)
        self.statusBar.showMessage(f"'{sec}' copied to clipboard", 2000)

    # --- Session save / load ---

    def session_save(self):
        fn, _ = QFileDialog.getSaveFileName(
            self, "Save Session", "", "BEER Session Files (*.beer)"
        )
        if not fn:
            return
        if not fn.endswith(".beer"):
            fn += ".beer"
        state = {
            "seq":         self.seq_text.toPlainText(),
            "seq_name":    self.sequence_name,
            "pH":          self.default_pH,
            "window_size": self.default_window_size,
            "use_reducing": self.use_reducing,
            "custom_pka":  self.custom_pka,
            "app_font_size": self.app_font_size,
            "label_font_size": self.label_font_size,
            "tick_font_size":  self.tick_font_size,
            "colormap":     self.colormap,
            "transparent_bg": self.transparent_bg,
        }
        try:
            with open(fn, "w") as f:
                json.dump(state, f, indent=2)
            self.statusBar.showMessage(f"Session saved: {fn}", 3000)
        except OSError as e:
            QMessageBox.critical(self, "Save Failed", str(e))

    def session_load(self):
        fn, _ = QFileDialog.getOpenFileName(
            self, "Load Session", "", "BEER Session Files (*.beer)"
        )
        if not fn:
            return
        try:
            with open(fn) as f:
                state = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Load Failed", str(e))
            return
        seq = state.get("seq", "")
        if seq:
            self.seq_text.setPlainText(seq)
        self.sequence_name    = state.get("seq_name", "")
        self.default_pH       = state.get("pH", 7.0)
        self.default_window_size = state.get("window_size", 9)
        self.use_reducing     = state.get("use_reducing", False)
        self.custom_pka       = state.get("custom_pka", None)
        self.transparent_bg   = state.get("transparent_bg", False)
        # Update settings UI widgets
        self.ph_input.setText(str(self.default_pH))
        self.window_size_input.setText(str(self.default_window_size))
        self.transparent_bg_checkbox.setChecked(self.transparent_bg)
        self.statusBar.showMessage(f"Session loaded: {fn}", 3000)
        if seq:
            self.on_analyze()

    # ── New tabs ─────────────────────────────────────────────────────────────

    def init_truncation_tab(self):
        """Tab for N/C terminal truncation series analysis."""
        container = QWidget()
        layout    = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        self.main_tabs.addTab(container, "Truncation")

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Step (%):"))
        self.trunc_step_spin = QSpinBox()
        self.trunc_step_spin.setRange(5, 25)
        self.trunc_step_spin.setValue(10)
        self.trunc_step_spin.setMaximumWidth(70)
        ctrl.addWidget(self.trunc_step_spin)
        ctrl.addSpacing(10)
        self.trunc_nterm_cb = QCheckBox("N-terminal")
        self.trunc_nterm_cb.setChecked(True)
        self.trunc_cterm_cb = QCheckBox("C-terminal")
        self.trunc_cterm_cb.setChecked(True)
        ctrl.addWidget(self.trunc_nterm_cb)
        ctrl.addWidget(self.trunc_cterm_cb)
        ctrl.addSpacing(10)
        run_trunc_btn = QPushButton("Run Truncation Series")
        run_trunc_btn.setMinimumHeight(30)
        run_trunc_btn.clicked.connect(self.run_truncation_series)
        ctrl.addWidget(run_trunc_btn)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self.trunc_status_lbl = QLabel("Run analysis first, then click 'Run Truncation Series'.")
        self.trunc_status_lbl.setStyleSheet("color:#718096; font-style:italic;")
        layout.addWidget(self.trunc_status_lbl)

        self.trunc_table = QTableWidget()
        self.trunc_table.setAlternatingRowColors(True)
        self.trunc_table.setColumnCount(8)
        self.trunc_table.setHorizontalHeaderLabels([
            "Type", "Trunc%", "Remaining aa", "MW (Da)", "pI", "GRAVY", "FCR", "NCPR"])
        self.trunc_table.horizontalHeader().setStretchLastSection(True)
        self.trunc_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        layout.addWidget(self.trunc_table, 1)

    def init_msa_tab(self):
        """Tab for Multiple Sequence Alignment + conservation analysis."""
        container = QWidget()
        layout    = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        self.main_tabs.addTab(container, "MSA")

        ctrl = QHBoxLayout()
        self.msa_aligned_cb = QCheckBox("Input is pre-aligned (gaps as '-')")
        self.msa_aligned_cb.setChecked(False)
        ctrl.addWidget(self.msa_aligned_cb)
        ctrl.addSpacing(16)
        run_msa_btn = QPushButton("Align & Show Conservation")
        run_msa_btn.setMinimumHeight(30)
        run_msa_btn.clicked.connect(self.run_msa)
        ctrl.addWidget(run_msa_btn)
        clear_msa_btn = QPushButton("Clear")
        clear_msa_btn.setMinimumHeight(30)
        clear_msa_btn.clicked.connect(self._clear_msa)
        ctrl.addWidget(clear_msa_btn)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        splitter = QSplitter(Qt.Horizontal)
        left_w = QWidget()
        left_v = QVBoxLayout(left_w)
        left_v.addWidget(QLabel("Paste multi-FASTA sequences here (≥2 sequences):"))
        self.msa_input = QTextEdit()
        self.msa_input.setPlaceholderText(">seq1\nACDEFG...\n>seq2\nACDEFG...")
        self.msa_input.setFont(QFont("Courier New", 9))
        left_v.addWidget(self.msa_input)
        splitter.addWidget(left_w)

        right_w = QWidget()
        right_v = QVBoxLayout(right_w)
        right_v.addWidget(QLabel("Alignment preview:"))
        self.msa_viewer = QTextBrowser()
        self.msa_viewer.setFont(QFont("Courier New", 9))
        right_v.addWidget(self.msa_viewer)
        splitter.addWidget(right_w)
        layout.addWidget(splitter, 1)

    def init_complex_tab(self):
        """Tab for protein complex stoichiometry calculations."""
        container = QWidget()
        layout    = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        self.main_tabs.addTab(container, "Complex")

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Stoichiometry (e.g. A2B1):"))
        self.complex_stoich_input = QLineEdit()
        self.complex_stoich_input.setPlaceholderText("A2B1")
        self.complex_stoich_input.setMaximumWidth(120)
        ctrl.addWidget(self.complex_stoich_input)
        ctrl.addSpacing(10)
        run_complex_btn = QPushButton("Calculate Complex")
        run_complex_btn.setMinimumHeight(30)
        run_complex_btn.clicked.connect(self.run_complex_calc)
        ctrl.addWidget(run_complex_btn)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        splitter = QSplitter(Qt.Horizontal)
        left_w = QWidget()
        left_v = QVBoxLayout(left_w)
        left_v.addWidget(QLabel("Paste chain sequences (multi-FASTA, chain ID in header):"))
        self.complex_input = QTextEdit()
        self.complex_input.setPlaceholderText(">ChainA\nACDEFG...\n>ChainB\nACDEFG...")
        self.complex_input.setFont(QFont("Courier New", 9))
        left_v.addWidget(self.complex_input)
        splitter.addWidget(left_w)

        right_w = QWidget()
        right_v = QVBoxLayout(right_w)
        self.complex_result_browser = QTextBrowser()
        right_v.addWidget(QLabel("Results:"))
        right_v.addWidget(self.complex_result_browser)
        splitter.addWidget(right_w)
        layout.addWidget(splitter, 1)

    # ── New method callbacks ──────────────────────────────────────────────────

    def run_truncation_series(self):
        if not self.analysis_data:
            QMessageBox.warning(self, "Truncation", "Run analysis first.")
            return
        seq    = self.analysis_data["seq"]
        step   = self.trunc_step_spin.value()
        do_n   = self.trunc_nterm_cb.isChecked()
        do_c   = self.trunc_cterm_cb.isChecked()
        n      = len(seq)
        rows   = []
        for pct in range(step, 100, step):
            n_rem = max(5, int(n * (1 - pct / 100)))
            if do_n:
                trunc_seq = seq[n - n_rem:]
                if is_valid_protein(trunc_seq) and len(trunc_seq) >= 5:
                    d = AnalysisTools.analyze_sequence(
                        trunc_seq, self.default_pH,
                        self.default_window_size, self.use_reducing, self.custom_pka)
                    rows.append(("N-term", pct, len(trunc_seq), d))
            if do_c:
                trunc_seq = seq[:n_rem]
                if is_valid_protein(trunc_seq) and len(trunc_seq) >= 5:
                    d = AnalysisTools.analyze_sequence(
                        trunc_seq, self.default_pH,
                        self.default_window_size, self.use_reducing, self.custom_pka)
                    rows.append(("C-term", pct, len(trunc_seq), d))
        self.trunc_table.setRowCount(0)
        for ttype, pct, rem, d in rows:
            row = self.trunc_table.rowCount()
            self.trunc_table.insertRow(row)
            for col, val in enumerate([
                ttype, f"{pct}%", str(rem),
                f"{d['mol_weight']:.2f}", f"{d['iso_point']:.2f}",
                f"{d['gravy']:.3f}", f"{d['fcr']:.3f}", f"{d['ncpr']:+.3f}",
            ]):
                self.trunc_table.setItem(row, col, QTableWidgetItem(val))
        self.trunc_table.resizeColumnsToContents()
        # Also generate the truncation graph
        if rows and _HAS_NEW_GRAPHS:
            n_data = [{"pct": r[1], "pI": r[3]["iso_point"], "gravy": r[3]["gravy"],
                       "fcr": r[3]["fcr"], "ncpr": r[3]["ncpr"],
                       "net_charge_7": r[3]["net_charge_7"],
                       "disorder_frac": r[3]["disorder_f"]}
                      for r in rows if r[0] == "N-term"]
            c_data = [{"pct": r[1], "pI": r[3]["iso_point"], "gravy": r[3]["gravy"],
                       "fcr": r[3]["fcr"], "ncpr": r[3]["ncpr"],
                       "net_charge_7": r[3]["net_charge_7"],
                       "disorder_frac": r[3]["disorder_f"]}
                      for r in rows if r[0] == "C-term"]
            fig = GraphingTools.create_truncation_series_figure(
                {"n_trunc": n_data, "c_trunc": c_data},
                label_font=self.label_font_size, tick_font=self.tick_font_size)
            self._replace_graph("Truncation Series", fig)
        self.trunc_status_lbl.setText(f"Computed {len(rows)} truncation variants.")
        self.statusBar.showMessage("Truncation series complete.", 3000)

    def run_msa(self):
        raw = self.msa_input.toPlainText().strip()
        if not raw:
            QMessageBox.warning(self, "MSA", "Paste multi-FASTA sequences first.")
            return
        entries = self._parse_pasted_text(raw)
        if len(entries) < 2:
            QMessageBox.warning(self, "MSA", "Need at least 2 sequences.")
            return
        names = [e[0] for e in entries]
        seqs  = [e[1] for e in entries]
        pre_aligned = self.msa_aligned_cb.isChecked()
        if not pre_aligned:
            # Simple pairwise progressive alignment using difflib
            def _align_pair(s1, s2):
                sm = difflib.SequenceMatcher(None, s1, s2)
                a1, a2 = [], []
                for op, i1, i2, j1, j2 in sm.get_opcodes():
                    if op == "equal":
                        a1.append(s1[i1:i2]); a2.append(s2[j1:j2])
                    elif op == "replace":
                        m = max(i2-i1, j2-j1)
                        a1.append(s1[i1:i2].ljust(m, "-")); a2.append(s2[j1:j2].ljust(m, "-"))
                    elif op == "insert":
                        a1.append("-" * (j2-j1)); a2.append(s2[j1:j2])
                    elif op == "delete":
                        a1.append(s1[i1:i2]); a2.append("-" * (i2-i1))
                return "".join(a1), "".join(a2)
            aligned = [seqs[0]]
            for i in range(1, len(seqs)):
                _, a2 = _align_pair(aligned[0], seqs[i])
                aligned.append(a2)
        else:
            maxlen = max(len(s) for s in seqs)
            aligned = [s.ljust(maxlen, "-") for s in seqs]

        self._msa_sequences = aligned
        self._msa_names     = names
        # Display alignment preview
        preview_lines = []
        for name, aln_seq in zip(names, aligned):
            preview_lines.append(f"<b>{name[:20]}</b>  <tt>{aln_seq[:80]}{'…' if len(aln_seq)>80 else ''}</tt>")
        self.msa_viewer.setHtml("<br>".join(preview_lines))
        # Generate conservation graph
        if _HAS_NEW_GRAPHS:
            fig = GraphingTools.create_msa_conservation_figure(
                aligned, names,
                label_font=self.label_font_size, tick_font=self.tick_font_size)
            self._replace_graph("MSA Conservation", fig)
        self.statusBar.showMessage(
            f"MSA: {len(aligned)} sequences, {len(aligned[0])} alignment columns", 3000)

    def _clear_msa(self):
        self._msa_sequences = []
        self._msa_names     = []
        self.msa_input.clear()
        self.msa_viewer.clear()
        self.statusBar.showMessage("MSA cleared.", 2000)

    def run_complex_calc(self):
        raw   = self.complex_input.toPlainText().strip()
        stoich = self.complex_stoich_input.text().strip() or "A1"
        entries = self._parse_pasted_text(raw)
        if not entries:
            QMessageBox.warning(self, "Complex", "Paste at least one chain sequence.")
            return
        # Parse stoichiometry: e.g. "A2B1" → {A:2, B:1}
        stoich_map = {}
        for m in re.finditer(r"([A-Za-z]+)(\d*)", stoich):
            chain_id = m.group(1).upper()
            count    = int(m.group(2)) if m.group(2) else 1
            if chain_id:
                stoich_map[chain_id] = count
        chain_data = {e[0].split()[0].upper(): e[1] for e in entries}
        # Compute properties using the top-level BPProteinAnalysis import
        _BPA = BPProteinAnalysis
        lines  = ["<h2>Chain Properties</h2><table><tr><th>Chain</th>"
                  "<th>n Copies</th><th>Length (aa)</th><th>MW (Da)</th>"
                  "<th>pI</th><th>Ext.Coeff.</th></tr>"]
        total_mw  = 0.0
        total_ext = 0
        chains_fig_data = []
        for cid, seq in chain_data.items():
            copies = stoich_map.get(cid, 1)
            pa  = _BPA(seq)
            mw  = pa.molecular_weight()
            pi  = pa.isoelectric_point()
            ext = 5500 * seq.count("W") + 1490 * seq.count("Y") + 125 * (seq.count("C")//2)
            total_mw  += mw  * copies
            total_ext += ext * copies
            lines.append(
                f"<tr><td>{cid}</td><td>{copies}</td><td>{len(seq)}</td>"
                f"<td>{mw:.2f}</td><td>{pi:.2f}</td><td>{ext}</td></tr>"
            )
            chains_fig_data.append({"chain_id": cid, "mol_weight": mw * copies})
        lines.append("</table>")
        lines.append(f"<h2>Complex Totals (stoichiometry: {stoich})</h2>"
                     f"<table><tr><th>Property</th><th>Value</th></tr>"
                     f"<tr><td>Total MW</td><td>{total_mw:.2f} Da ({total_mw/1000:.2f} kDa)</td></tr>"
                     f"<tr><td>Combined Ext. Coeff. (280 nm)</td><td>{total_ext} M⁻¹cm⁻¹</td></tr>"
                     f"</table>")
        self.complex_result_browser.setHtml("".join(lines))
        # Generate complex mass graph
        if _HAS_NEW_GRAPHS and chains_fig_data:
            fig = GraphingTools.create_complex_mw_figure(
                chains_fig_data, stoich,
                label_font=self.label_font_size, tick_font=self.tick_font_size)
            self._replace_graph("Complex Mass", fig)
        self.statusBar.showMessage("Complex calculation done.", 3000)

    # ── ELM / DisProt / PhaSepDB callbacks ────────────────────────────────────

    def fetch_elm(self):
        if not _HAS_ELM:
            QMessageBox.information(self, "ELM",
                "ELM module not available.\nInstall with: pip install beer-biophys")
            return
        acc = self.current_accession
        if not acc:
            QMessageBox.warning(self, "ELM", "Fetch a UniProt accession first.")
            return
        if self._elm_worker and self._elm_worker.isRunning():
            return
        self.fetch_elm_btn.setEnabled(False)
        self.statusBar.showMessage(f"Fetching ELM instances for {acc}…")
        seq = self.analysis_data["seq"] if self.analysis_data else ""
        self._elm_worker = ELMWorker(acc, seq)
        self._elm_worker.finished.connect(self._on_elm_finished)
        self._elm_worker.error.connect(self._on_elm_error)
        self._elm_worker.start()

    def _on_elm_finished(self, instances: list):
        self.elm_data = instances
        self.fetch_elm_btn.setEnabled(True)
        n = len(instances)
        if n == 0:
            QMessageBox.information(self, "ELM",
                "No ELM instances found for this accession.")
        else:
            # Show in a popup summary
            lines = ["<h2>ELM Instances</h2><table>"
                     "<tr><th>ELM Class</th><th>Start</th><th>End</th><th>Logic</th></tr>"]
            for inst in instances[:50]:
                lines.append(
                    f"<tr><td>{inst.get('elm_identifier','?')}</td>"
                    f"<td>{inst.get('start','?')}</td><td>{inst.get('end','?')}</td>"
                    f"<td>{inst.get('logic','?')}</td></tr>")
            lines.append("</table>")
            dlg = QDialog(self); dlg.setWindowTitle("ELM Instances")
            dlg.resize(600, 400)
            vb = QVBoxLayout(dlg)
            br = QTextBrowser(); br.setHtml("".join(lines))
            vb.addWidget(br)
            btns = QDialogButtonBox(QDialogButtonBox.Close)
            btns.rejected.connect(dlg.reject); vb.addWidget(btns)
            dlg.exec_()
        self.statusBar.showMessage(f"ELM: {n} instance(s) found.", 3000)

    def _on_elm_error(self, msg: str):
        self.fetch_elm_btn.setEnabled(True)
        QMessageBox.warning(self, "ELM Error", msg)

    def fetch_disprot(self):
        if not _HAS_DISPROT:
            QMessageBox.information(self, "DisProt",
                "DisProt module not available.\nInstall with: pip install beer-biophys")
            return
        acc = self.current_accession
        if not acc:
            QMessageBox.warning(self, "DisProt", "Fetch a UniProt accession first.")
            return
        if self._disprot_worker and self._disprot_worker.isRunning():
            return
        self.fetch_disprot_btn.setEnabled(False)
        self.statusBar.showMessage(f"Fetching DisProt annotations for {acc}…")
        self._disprot_worker = DisPRotWorker(acc)
        self._disprot_worker.finished.connect(self._on_disprot_finished)
        self._disprot_worker.error.connect(self._on_disprot_error)
        self._disprot_worker.start()

    def _on_disprot_finished(self, data: dict):
        self.disprot_data = data
        self.fetch_disprot_btn.setEnabled(True)
        regions = data.get("regions", [])
        n = len(regions)
        if n == 0:
            QMessageBox.information(self, "DisProt",
                "This protein is not in DisProt, or has no annotated disorder regions.")
        else:
            frac = data.get("fraction_disordered", 0)
            lines = [f"<h2>DisProt: {data.get('disprot_id','?')}</h2>"
                     f"<p>{data.get('protein_name','')}</p>"
                     f"<p>Fraction disordered: {frac:.3f}</p>"
                     "<table><tr><th>Start</th><th>End</th><th>Type</th></tr>"]
            for r in regions:
                lines.append(
                    f"<tr><td>{r['start']}</td><td>{r['end']}</td>"
                    f"<td>{r.get('type','IDR')}</td></tr>")
            lines.append("</table>")
            dlg = QDialog(self); dlg.setWindowTitle("DisProt Disorder Regions")
            dlg.resize(500, 350)
            vb = QVBoxLayout(dlg)
            br = QTextBrowser(); br.setHtml("".join(lines))
            vb.addWidget(br)
            btns = QDialogButtonBox(QDialogButtonBox.Close)
            btns.rejected.connect(dlg.reject); vb.addWidget(btns)
            dlg.exec_()
        self.statusBar.showMessage(f"DisProt: {n} disorder region(s).", 3000)

    def _on_disprot_error(self, msg: str):
        self.fetch_disprot_btn.setEnabled(True)
        self.statusBar.showMessage("DisProt fetch failed.", 2000)
        QMessageBox.warning(self, "DisProt Error", msg)

    def fetch_phasepdb(self):
        if not _HAS_PHASEPDB:
            QMessageBox.information(self, "PhaSepDB",
                "PhaSepDB module not available.\nInstall with: pip install beer-biophys")
            return
        acc = self.current_accession
        if not acc:
            QMessageBox.warning(self, "PhaSepDB", "Fetch a UniProt accession first.")
            return
        if self._phasepdb_worker and self._phasepdb_worker.isRunning():
            return
        self.fetch_phasepdb_btn.setEnabled(False)
        self.statusBar.showMessage(f"Checking PhaSepDB for {acc}…")
        self._phasepdb_worker = PhaSepDBWorker(acc)
        self._phasepdb_worker.finished.connect(self._on_phasepdb_finished)
        self._phasepdb_worker.error.connect(self._on_phasepdb_error)
        self._phasepdb_worker.start()

    def _on_phasepdb_finished(self, data: dict):
        self.phasepdb_data = data
        self.fetch_phasepdb_btn.setEnabled(True)
        if not data.get("found"):
            QMessageBox.information(self, "PhaSepDB",
                "This protein was not found in PhaSepDB (phase separation database).\n"
                "This does not rule out phase-separation capacity.")
        else:
            msg = (f"<h2>PhaSepDB Hit</h2>"
                   f"<p>Source: {data.get('source','PhaSepDB')}</p>"
                   f"<p>Category: <b>{data.get('category','?')}</b></p>"
                   f"<p>Evidence type: {data.get('evidence_type','?')}</p>")
            dlg = QDialog(self); dlg.setWindowTitle("PhaSepDB")
            dlg.resize(400, 200)
            vb = QVBoxLayout(dlg)
            br = QTextBrowser(); br.setHtml(msg)
            vb.addWidget(br)
            btns = QDialogButtonBox(QDialogButtonBox.Close)
            btns.rejected.connect(dlg.reject); vb.addWidget(btns)
            dlg.exec_()
        self.statusBar.showMessage(
            "PhaSepDB: found" if data.get("found") else "PhaSepDB: not found", 3000)

    def _on_phasepdb_error(self, msg: str):
        self.fetch_phasepdb_btn.setEnabled(True)
        QMessageBox.warning(self, "PhaSepDB Error", msg)

    # ── Figure Composer ───────────────────────────────────────────────────────

    def open_figure_composer(self):
        """Open the Figure Composer dialog to build a multi-panel publication figure."""
        available = list(self.graph_tabs.keys())
        dlg = _FigureComposerDialog(available, self)
        if dlg.exec_() != QDialog.Accepted:
            return
        layout_str, selected_titles = dlg.get_composition()
        # Parse layout e.g. "2×2" → (rows, cols)
        try:
            rows_s, cols_s = layout_str.split("\u00d7")
            nrows, ncols = int(rows_s), int(cols_s)
        except Exception:
            nrows, ncols = 1, 1
        total = nrows * ncols
        titles = (selected_titles + [None] * total)[:total]
        fig_out = Figure(figsize=(ncols * 6, nrows * 4.5), dpi=150)
        fig_out.set_facecolor("#ffffff")
        for i, title in enumerate(titles):
            ax_sub = fig_out.add_subplot(nrows, ncols, i + 1)
            ax_sub.set_visible(False)
            if title and title in self.graph_tabs:
                _, vb = self.graph_tabs[title]
                canvas = self._find_canvas(vb)
                if canvas:
                    src_fig = canvas.figure
                    if src_fig.axes:
                        src_ax  = src_fig.axes[0]
                        # Copy the axis into composite figure using rasterization
                        ax_sub.set_visible(True)
                        buf = BytesIO()
                        src_fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                        buf.seek(0)
                        img_arr = plt.imread(buf)
                        ax_sub.imshow(img_arr, aspect="auto")
                        ax_sub.axis("off")
                        lbl = chr(ord("A") + i)
                        ax_sub.text(-0.05, 1.05, lbl, transform=ax_sub.transAxes,
                                    fontsize=14, fontweight="bold", va="top")
        fig_out.tight_layout(pad=1.0)
        # Save dialog
        ext  = self.default_graph_format.lower()
        fn, _ = QFileDialog.getSaveFileName(
            self, "Save Composed Figure", "",
            f"{self.default_graph_format} Files (*.{ext})")
        if fn:
            if not fn.lower().endswith(f".{ext}"):
                fn += f".{ext}"
            use_transparent = self.transparent_bg and ext in ("png", "svg")
            fig_out.savefig(fn, format=ext, dpi=200, bbox_inches="tight",
                            transparent=use_transparent,
                            facecolor="none" if use_transparent else "white")
            QMessageBox.information(self, "Saved", f"Composed figure saved to:\n{fn}")
        plt.close(fig_out)

    # ── Plugin system ─────────────────────────────────────────────────────────

    def _load_plugins(self):
        """Scan ~/.beer/plugins/ for .py files and load valid BEER plugins."""
        plugin_dir = os.path.expanduser("~/.beer/plugins")
        if not os.path.isdir(plugin_dir):
            return
        for fname in os.listdir(plugin_dir):
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(plugin_dir, fname)
            try:
                spec   = importlib.util.spec_from_file_location(fname[:-3], fpath)
                mod    = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, "PLUGIN_NAME") and hasattr(mod, "analyze"):
                    self._plugins.append(mod)
            except Exception as e:
                print(f"[BEER] Plugin load error ({fname}): {e}", file=sys.stderr)

    def _run_plugins(self, seq: str, data: dict):
        """Call each loaded plugin's analyze(seq, data) → html, inject into report."""
        for plugin in self._plugins:
            try:
                html = plugin.analyze(seq, data)
                sec_name = getattr(plugin, "PLUGIN_NAME", "Plugin")
                data["report_sections"][sec_name] = html
                if sec_name not in REPORT_SECTIONS:
                    REPORT_SECTIONS.append(sec_name)
                    # Add to UI section list
                    self.report_section_list.addItem(QListWidgetItem(sec_name))
                    tab = QWidget()
                    vb  = QVBoxLayout(tab)
                    vb.setContentsMargins(4, 4, 4, 4)
                    browser = QTextBrowser()
                    vb.addWidget(browser)
                    self.report_stack.addWidget(tab)
                    self.report_section_tabs[sec_name] = browser
            except Exception as e:
                print(f"[BEER] Plugin runtime error ({plugin.PLUGIN_NAME}): {e}",
                      file=sys.stderr)

    # --- Dependency check ---

    def check_dependencies(self):
        missing = []
        for pkg in ("Bio", "matplotlib", "PyQt5", "mplcursors"):
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg)
        if missing:
            resp = QMessageBox.question(
                self, "Missing Dependencies",
                "These packages are missing: " + ", ".join(missing) + "\nInstall now?",
                QMessageBox.Yes | QMessageBox.No
            )
            if resp == QMessageBox.Yes:
                result = subprocess.call([sys.executable, "-m", "pip", "install"] + missing)
                if result != 0:
                    QMessageBox.warning(self, "Install Failed",
                                        "Some packages could not be installed.")
        if not _WEBENGINE_AVAILABLE:
            self.statusBar.showMessage(
                "Tip: install PyQtWebEngine (pip install PyQtWebEngine) for the 3D structure viewer.",
                8000
            )


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    w   = ProteinAnalyzerGUI()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
