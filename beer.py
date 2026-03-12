#!/usr/bin/env python3
"""
BEER - Biochemical Estimator & Explorer of Residues

Requirements:
  pip install biopython matplotlib PyQt5 mplcursors
"""

import sys, math, os, base64, json, csv, subprocess
from io import BytesIO
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QTextBrowser,
    QFileDialog, QTabWidget, QMessageBox, QTableWidget, QTableWidgetItem,
    QCheckBox, QStatusBar, QComboBox, QFormLayout,
    QSplitter, QScrollArea, QFrame
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from PyQt5.QtPrintSupport import QPrinter

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
plt.style.use("default")
import mplcursors

from Bio.SeqUtils.ProtParam import ProteinAnalysis as BPProteinAnalysis
from Bio import SeqIO
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1

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

# Normalization ranges for radar chart; adjust for atypical proteins
RADAR_RANGES = {
    "Mol Weight":  (5000, 150000),
    "pI":          (4, 11),
    "GRAVY":       (-2.5, 2.5),
    "Instability": (20, 80),
    "Aromaticity": (0, 0.2),
}

# Disorder/order residue classification (Uversky)
DISORDER_PROMOTING = set("AEGKPQRS")
ORDER_PROMOTING    = set("CFHILMVWY")

# Sticker residue sets for phase separation analysis
STICKER_AROMATIC      = set("FWY")
STICKER_ELECTROSTATIC = set("KRDE")
STICKER_ALL           = STICKER_AROMATIC | STICKER_ELECTROSTATIC

# Prion-like domain composition residues (PLAAC/Lancaster)
PRION_LIKE = set("NQSGY")

REPORT_SECTIONS = [
    "Overview",
    "Composition",
    "Properties",
    "Charge",
    "Aromatic & \u03c0",
    "Low Complexity",
    "Disorder",
    "Repeat Motifs",
    "Sticker & Spacer",
]

GRAPH_TITLES = [
    "Amino Acid Composition (Bar)",
    "Amino Acid Composition (Pie)",
    "Hydrophobicity Profile",
    "Net Charge vs pH",
    "Bead Model (Hydrophobicity)",
    "Bead Model (Charge)",
    "Properties Radar Chart",
    "Sticker Map",
    "Local Charge Profile",
    "Local Complexity",
    "Cation\u2013\u03c0 Map",
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
        solubility    = AnalysisTools.predict_solubility(seq)

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

        # --- HTML sections (styled) ---
        _style = f"<style>{REPORT_CSS}</style>"
        extra = (
            f"<tr><td>Net Charge (pH {pH_value:.1f})</td><td>{net_charge_pH:.2f}</td></tr>"
            if abs(pH_value - 7.0) >= 1e-6 else ""
        )
        overview_html = _style + f"""
        <h2>Overview</h2>
        <table>
          <tr><th>Property</th><th>Value</th></tr>
          <tr><td>Sequence Length</td><td>{seq_length} aa</td></tr>
          <tr><td>Net Charge (pH 7.0)</td><td>{net_charge_7:.2f}</td></tr>
          {extra}
          <tr><td>Solubility Prediction</td><td>{solubility}</td></tr>
        </table>
        """

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
          <tr><td>Molecular Weight</td><td>{mol_weight:.2f} Da</td></tr>
          <tr><td>Isoelectric Point (pI)</td><td>{iso_point:.2f}</td></tr>
          <tr><td>Extinction Coeff. (280 nm)</td><td>{extinction} M&#8315;&#185;cm&#8315;&#185;</td></tr>
          <tr><td>GRAVY Score</td><td>{gravy:.3f}</td></tr>
          <tr><td>Instability Index</td><td>{instability:.2f}</td></tr>
          <tr><td>Aromaticity</td><td>{aromaticity:.3f}</td></tr>
        </table>
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

        return {
            "report_sections": {
                "Overview":        overview_html,
                "Composition":     comp_html,
                "Properties":      bio_html,
                "Charge":          charge_html,
                "Aromatic & \u03c0": aromatic_html,
                "Low Complexity":  lc_html,
                "Disorder":        disorder_html,
                "Repeat Motifs":   repeats_html,
                "Sticker & Spacer": sticker_html,
            },
            "aa_counts":      aa_counts,
            "aa_freq":        aa_freq,
            "hydro_profile":  sliding_window_hydrophobicity(seq, window_size),
            "ncpr_profile":   sliding_window_ncpr(seq, window_size),
            "entropy_profile": sliding_window_entropy(seq, window_size),
            "window_size":    window_size,
            "seq":            seq,
            "mol_weight":     mol_weight,
            "iso_point":      iso_point,
            "net_charge_7":   net_charge_7,
            "extinction":     extinction,
            "gravy":          gravy,
            "instability":    instability,
            "aromaticity":    aromaticity,
        }

    @staticmethod
    def predict_solubility(seq: str) -> str:
        avg_hydro = sum(KYTE_DOOLITTLE[aa] for aa in seq) / len(seq)
        return "Likely soluble" if avg_hydro < 0.5 else "Low solubility predicted"

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
    def create_radar_chart_figure(data, label_font=14):
        props  = list(RADAR_RANGES.keys())
        vals   = [data["mol_weight"], data["iso_point"], data["gravy"],
                  data["instability"], data["aromaticity"]]
        norm   = []
        for p, v in zip(props, vals):
            mn, mx = RADAR_RANGES[p]
            norm.append(max(0, min(1, (v - mn) / (mx - mn))))
        norm   += norm[:1]
        angles  = [n/len(props)*2*math.pi for n in range(len(props))] + [0]
        fig = Figure(figsize=(5.5, 5), dpi=120)
        fig.set_facecolor("#ffffff")
        ax  = fig.add_subplot(111, polar=True)
        ax.set_facecolor("#fafbff")
        ax.plot(angles, norm, color=_ACCENT, linewidth=2.0, zorder=4)
        ax.fill(angles, norm, color=_ACCENT, alpha=0.22, zorder=3)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(props, fontsize=label_font-3, color="#2d3748")
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["25%", "50%", "75%", "100%"],
                            fontsize=label_font-5, color="#718096")
        ax.tick_params(pad=6)
        ax.set_title("Properties Radar Chart", fontsize=label_font,
                     fontweight="bold", color="#1a1a2e", pad=14)
        ax.grid(color="#c8cdd8", linewidth=0.6, linestyle="--")
        ax.spines["polar"].set_color("#d0d4e0")
        fig.tight_layout(pad=2.0)
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

# --- Export ---

class ExportTools:
    @staticmethod
    def _generate_full_html(analysis_data, graph_tabs, seq_name=""):
        if not analysis_data or "report_sections" not in analysis_data:
            return "<p>No analysis data available.</p>"

        header_name = seq_name or "Protein Sequence"
        seq_block = format_sequence_block(
            analysis_data.get("seq", ""), name=seq_name
        ).replace("\n", "<br>").replace(" ", "&nbsp;")

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
}}
.graph-section {{ margin: 14px 0 24px 0; text-align: center; }}
.graph-section img {{ max-width: 100%; height: auto; border: 1px solid #e8eaf0; border-radius: 4px; }}
</style>
</head><body>
<h1>BEER Analysis Report</h1>
<p style="color:#718096;font-size:10pt;">Sequence: <strong>{header_name}</strong>
&nbsp;&bull;&nbsp; Length: <strong>{len(analysis_data.get("seq",""))} aa</strong></p>
<h2 style="margin-top:16px;">Sequence</h2>
<div class="seq-block">{seq_block}</div>
<div class="page-break"></div>
"""
        for sec in REPORT_SECTIONS:
            html += analysis_data["report_sections"].get(sec, "")

        html += '<div class="page-break"></div><h1>Graphs</h1>\n'

        for title, (tab, vbox) in graph_tabs.items():
            for i in range(vbox.count()):
                item = vbox.itemAt(i)
                w    = item.widget() if item else None
                if w and hasattr(w, "figure"):
                    buf = BytesIO()
                    w.figure.savefig(buf, format="png", dpi=200,
                                     bbox_inches="tight",
                                     facecolor="white")
                    buf.seek(0)
                    b64 = base64.b64encode(buf.read()).decode()
                    html += (
                        f'<h2>{title}</h2>'
                        '<div class="graph-section">'
                        f'<img src="data:image/png;base64,{b64}" />'
                        "</div>"
                        '<div class="page-break"></div>\n'
                    )
                    break

        html += "</body></html>"
        return html

    @staticmethod
    def export_report_text(analysis_data, file_name):
        try:
            with open(file_name, "w") as f:
                for sec, content in analysis_data["report_sections"].items():
                    text = content.replace("<br>", "\n")
                    for tag in ("<h2>", "</h2>", "<table", "</table>", "<tr>", "</tr>"):
                        text = text.replace(tag, "\n")
                    text = text.replace("<th>", "").replace("</th>", "\t")
                    text = text.replace("<td>", "").replace("</td>", "\t")
                    f.write(f"==== {sec} ====\n{text}\n\n")
            return True, f"Report saved to {file_name}"
        except Exception as e:
            return False, f"Save error: {e}"

    @staticmethod
    def export_pdf(analysis_data, graph_tabs, file_name, parent, seq_name=""):
        try:
            printer = QPrinter(QPrinter.HighResolution)
            printer.setOutputFormat(QPrinter.PdfFormat)
            printer.setOutputFileName(file_name)
            browser = QTextBrowser()
            browser.setHtml(ExportTools._generate_full_html(analysis_data, graph_tabs, seq_name))
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

# --- Batch stats helper ---

def _calc_batch_stats(seq: str, data: dict) -> tuple:
    """Return (hydro%, hydrophil%, pos%, neg%, neu%) for a sequence."""
    length = len(seq)
    hydro  = sum(1 for aa in seq if KYTE_DOOLITTLE[aa] > 0) / length * 100
    pos    = sum(data["aa_counts"].get(k, 0) for k in ("K", "R", "H")) / length * 100
    neg    = sum(data["aa_counts"].get(k, 0) for k in ("D", "E")) / length * 100
    neu    = 100 - (pos + neg)
    return hydro, 100 - hydro, pos, neg, neu

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
        self.enable_tooltips     = False
        self.use_reducing        = False
        self.custom_pka          = None
        self.sequence_name       = ""   # display name for current sequence
        self._tooltips: dict     = {}  # widget -> tooltip text

        self.check_dependencies()
        self.main_tabs = QTabWidget()
        self.setCentralWidget(self.main_tabs)
        self.init_analysis_tab()
        self.init_graphs_tab()
        self.init_batch_tab()
        self.init_settings_tab()
        self.init_help_tab()

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

        # ---- toolbar row ----
        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)
        self.import_fasta_btn = QPushButton("  Import FASTA")
        self.import_fasta_btn.clicked.connect(self.import_fasta)
        self.import_pdb_btn = QPushButton("  Import PDB")
        self.import_pdb_btn.clicked.connect(self.import_pdb)
        self.analyze_btn = QPushButton("  Analyze")
        self.analyze_btn.clicked.connect(self.on_analyze)
        self.save_txt_btn = QPushButton("  Save Report")
        self.save_txt_btn.clicked.connect(self.save_report)
        self.save_pdf_btn = QPushButton("  Export PDF")
        self.save_pdf_btn.clicked.connect(self.export_pdf)
        for w in (self.import_fasta_btn, self.import_pdb_btn, self.analyze_btn,
                  self.save_txt_btn, self.save_pdf_btn):
            w.setMinimumHeight(32)
            toolbar.addWidget(w)
        toolbar.addStretch()
        outer.addLayout(toolbar)

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

        # Sequence viewer (UniProt style)
        seq_view_label = QLabel("Sequence Viewer:")
        seq_view_label.setStyleSheet("font-weight:600; color:#4361ee; margin-top:4px;")
        left_layout.addWidget(seq_view_label)
        self.seq_viewer = QTextBrowser()
        self.seq_viewer.setFont(QFont("Courier New", 10))
        self.seq_viewer.setStyleSheet(
            "QTextBrowser { background:#f8f9fd; border:1px solid #e8eaf0;"
            " border-radius:4px; padding:6px; }"
        )
        left_layout.addWidget(self.seq_viewer, 1)

        splitter.addWidget(left)

        # Right panel: report tabs
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 0, 0, 0)
        right_layout.setSpacing(0)

        self.report_tabs = QTabWidget()
        right_layout.addWidget(self.report_tabs)
        self.report_section_tabs = {}
        for sec in REPORT_SECTIONS:
            tab = QWidget()
            vb  = QVBoxLayout(tab)
            vb.setContentsMargins(4, 4, 4, 4)
            if sec == "Composition":
                sort_row = QHBoxLayout()
                sort_row.setSpacing(4)
                for lbl, mode in [("A–Z", "alpha"), ("By Freq", "composition"),
                                   ("Hydro ↑", "hydro_inc"), ("Hydro ↓", "hydro_dec")]:
                    btn = QPushButton(lbl)
                    btn.setMaximumWidth(90)
                    btn.setMinimumHeight(28)
                    btn.clicked.connect(lambda _, m=mode: self.sort_composition(m))
                    sort_row.addWidget(btn)
                sort_row.addStretch()
                vb.addLayout(sort_row)
            browser = QTextBrowser()
            vb.addWidget(browser)
            self.report_tabs.addTab(tab, sec)
            self.report_section_tabs[sec] = browser

        splitter.addWidget(right)
        splitter.setSizes([400, 700])
        outer.addWidget(splitter, 1)

    def init_graphs_tab(self):
        container = QWidget()
        layout    = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        self.main_tabs.addTab(container, "Graphs")

        self.graphs_subtabs = QTabWidget()
        layout.addWidget(self.graphs_subtabs, 1)

        self.graph_tabs = {}
        for title in GRAPH_TITLES:
            tab = QWidget()
            vb  = QVBoxLayout(tab)
            vb.setContentsMargins(4, 4, 4, 4)
            placeholder = QLabel(f"Run analysis to generate:  {title}")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("color:#718096; font-style:italic;")
            vb.addWidget(placeholder)
            btn = QPushButton("Save Graph")
            btn.setMaximumWidth(120)
            btn.clicked.connect(lambda _, t=title: self.save_graph(t))
            vb.addWidget(btn, alignment=Qt.AlignRight)
            self.graphs_subtabs.addTab(tab, title)
            self.graph_tabs[title] = (tab, vb)

        save_all = QPushButton("Save All Graphs")
        save_all.setMinimumHeight(30)
        save_all.clicked.connect(self.save_all_graphs)
        layout.addWidget(save_all, alignment=Qt.AlignRight)

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
        self.colormap_combo.addItems([
            "coolwarm", "viridis", "plasma", "inferno", "magma",
            "cividis", "Spectral", "RdBu", "PiYG", "PRGn",
            "hot", "copper", "cool", "autumn", "hsv",
        ])
        self._set_tooltip(self.colormap_combo, "Colour map for bead hydrophobicity model.")
        form3.addRow("Bead Colormap:", self.colormap_combo)

        self.graph_color_combo = QComboBox()
        self.graph_color_combo.addItems([
            "#4361ee", "#f72585", "#43aa8b", "#7209b7",
            "#f3722c", "#277da1", "#06d6a0", "#2d3748",
        ])
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
        layout.addLayout(form3)

        form4 = QFormLayout()
        form4.setHorizontalSpacing(20)
        form4.setVerticalSpacing(8)
        form4.setLabelAlignment(Qt.AlignRight)
        _section("Interface")
        self.theme_toggle = QCheckBox("Dark Theme")
        self._set_tooltip(self.theme_toggle, "Toggle between light and dark application themes.")
        self.theme_toggle.stateChanged.connect(self.toggle_theme)
        self.theme_toggle.setChecked(False)
        form4.addRow("", self.theme_toggle)

        self.tooltips_checkbox = QCheckBox("Enable Tooltips")
        self.tooltips_checkbox.setChecked(False)
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
        layout    = QVBoxLayout()
        container = QWidget()
        container.setLayout(layout)
        self.main_tabs.addTab(container, "Help")
        b = QTextBrowser()
        b.setHtml("""
        <h1>BEER Help &amp; Definitions</h1>

        <h2>Protein Sequence</h2>
        <p>Enter a single-letter amino acid sequence or import from FASTA/PDB.</p>

        <h2>Overview</h2>
        <ul>
          <li><b>Sequence Length:</b> Number of residues.</li>
          <li><b>Sequence:</b> The raw amino acid string.</li>
          <li><b>Net Charge:</b> At pH 7.0 (and custom pH if specified).</li>
          <li><b>Solubility Prediction:</b> Based on average hydrophobicity.</li>
        </ul>

        <h2>Composition</h2>
        <p>Counts and percentage frequencies of each residue.</p>

        <h2>Properties</h2>
        <ul>
          <li><b>Molecular Weight:</b> Approx. mass in Daltons.</li>
          <li><b>Isoelectric Point (pI):</b> pH with zero net charge.</li>
          <li><b>Extinction Coeff.:</b> Absorbance at 280 nm per M per cm.</li>
          <li><b>GRAVY Score:</b> Grand average of hydropathicity (higher = hydrophobic).</li>
          <li><b>Instability Index:</b> &lt;40 suggests stable protein.</li>
          <li><b>Aromaticity:</b> Fraction of F, W, Y residues.</li>
        </ul>

        <h2>Charge</h2>
        <ul>
          <li><b>FCR:</b> Fraction of charged residues (K,R,D,E).</li>
          <li><b>NCPR:</b> Net charge per residue = (K+R &minus; D+E) / length.</li>
          <li><b>Kappa (&kappa;):</b> Charge patterning, 0 = well-mixed, 1 = fully segregated (Das &amp; Pappu 2013).</li>
          <li><b>Charge asymmetry:</b> Ratio of positive to negative residues.</li>
        </ul>

        <h2>Aromatic &amp; &pi;</h2>
        <ul>
          <li><b>Aromatic fraction:</b> (F+W+Y)/length &mdash; &pi;&ndash;&pi; stacking drives many condensates.</li>
          <li><b>Cation&ndash;&pi; pairs:</b> K/R within &plusmn;4 positions of F/W/Y.</li>
          <li><b>&pi;&ndash;&pi; pairs:</b> F/W/Y within &plusmn;4 positions of another F/W/Y.</li>
        </ul>

        <h2>Low Complexity</h2>
        <ul>
          <li><b>Shannon entropy:</b> Compositional complexity in bits; max = log&#8322;(20) &asymp; 4.32.</li>
          <li><b>Prion-like score:</b> Fraction of N,Q,S,G,Y &mdash; enriched in yeast prion domains (PLAAC).</li>
          <li><b>LC fraction:</b> Fraction of sequence covered by windows with entropy &lt; 2.0 bits.</li>
        </ul>

        <h2>Disorder</h2>
        <ul>
          <li><b>Disorder-promoting fraction:</b> A,E,G,K,P,Q,R,S (Uversky classification).</li>
          <li><b>Order-promoting fraction:</b> C,F,H,I,L,M,V,W,Y.</li>
          <li><b>Aliphatic index:</b> (A + 2.9V + 3.9(I+L)) / length &times; 100 (Ikai 1980).</li>
          <li><b>Omega (&Omega;):</b> Patterning of sticker residues; 0 = even, 1 = clustered (Das et al. 2015).</li>
        </ul>

        <h2>Repeat Motifs</h2>
        <ul>
          <li><b>RGG:</b> Arg-Gly-Gly &mdash; major driver in FUS, hnRNP family.</li>
          <li><b>FG:</b> Phe-Gly &mdash; hallmark of nucleoporin IDRs.</li>
          <li><b>YG/GY:</b> Tyr-Gly variants.</li>
          <li><b>SR/RS:</b> Ser-Arg &mdash; splicing factor signature.</li>
          <li><b>QN/NQ:</b> Gln-Asn &mdash; yeast prion signature.</li>
        </ul>

        <h2>Sticker &amp; Spacer</h2>
        <ul>
          <li><b>Stickers:</b> F,W,Y,K,R,D,E &mdash; residues mediating specific interactions.</li>
          <li><b>Spacers:</b> all other residues providing chain flexibility and valency.</li>
          <li><b>Spacing stats:</b> Mean/min/max gap between consecutive stickers (Mittag &amp; Pappu).</li>
        </ul>

        <h2>Graphs</h2>
        <ul>
          <li><b>Bar/Pie Charts:</b> Amino acid composition.</li>
          <li><b>Hydrophobicity Profile:</b> Sliding-window Kyte-Doolittle average.</li>
          <li><b>Net Charge vs pH:</b> Charge curve from pH 0 to 14.</li>
          <li><b>Bead Models:</b> Per-residue hydrophobicity or charge.</li>
          <li><b>Radar Chart:</b> Normalized physiochemical properties.</li>
          <li><b>Sticker Map:</b> Per-residue sticker identity (aromatic/basic/acidic/spacer).</li>
          <li><b>Local Charge Profile:</b> Sliding-window NCPR showing charge block structure.</li>
          <li><b>Local Complexity:</b> Sliding-window Shannon entropy; red dashed line = LC threshold.</li>
          <li><b>Cation&ndash;&pi; Map:</b> Proximity heat map of K/R vs F/W/Y pairs along the sequence.</li>
        </ul>

        <h2>Batch Analysis</h2>
        <p>Import multi-FASTA or PDB to analyze multiple sequences; select one for detail.</p>

        <h2>Settings</h2>
        <p>Adjust window size, pH, fonts, colormap, theme, and display options.</p>
        """)
        layout.addWidget(b)

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
        pdb_base = os.path.splitext(os.path.basename(file_name))[0]
        entries  = [(f"{pdb_base}_{cid}", seq) for cid, seq in chains.items()]
        self._load_batch([(cid, seq) for cid, seq in chains.items()])
        if not self.sequence_name:
            self.sequence_name = entries[0][0] if entries else pdb_base

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

    def on_analyze(self):
        seq = clean_sequence(self.seq_text.toPlainText())
        if not seq:
            QMessageBox.warning(self, "Input", "Enter sequence.")
            return
        if not is_valid_protein(seq):
            QMessageBox.warning(self, "Input", "Invalid residues.")
            return
        try:
            pH = float(self.ph_input.text())
        except ValueError:
            pH = 7.0

        self.analysis_data = AnalysisTools.analyze_sequence(
            seq, pH, self.default_window_size, self.use_reducing, self.custom_pka
        )
        # Default name when typing raw sequence
        if not self.sequence_name:
            self.sequence_name = "Sequence"

        # Disable chain combo only when no batch is loaded
        if not self.batch_data:
            self.chain_combo.clear()
            self.chain_combo.setEnabled(False)

        for sec, browser in self.report_section_tabs.items():
            browser.setHtml(self.analysis_data["report_sections"][sec])
        self._update_seq_viewer()
        self.update_graph_tabs()
        self.statusBar.showMessage(
            f"Analysis complete  |  {len(seq)} aa  |  {self.sequence_name}", 4000
        )
        QMessageBox.information(self, "Done", "Analysis complete.")

    def _update_seq_viewer(self):
        """Refresh the sequence viewer panel with UniProt-style formatted output."""
        if not self.analysis_data:
            return
        seq  = self.analysis_data["seq"]
        name = self.sequence_name or ""
        text = format_sequence_block(seq, name=name)
        # Colour position numbers
        lines  = text.split("\n")
        html_lines = []
        for ln in lines:
            if ln.startswith(">"):
                html_lines.append(
                    f'<span style="color:#4361ee;font-weight:700;">{ln}</span>'
                )
            elif ln and ln.lstrip()[0:1].isdigit():
                # position number + sequence groups
                parts = ln.split("  ", 1)
                if len(parts) == 2:
                    pos_str, seq_str = parts
                    html_lines.append(
                        f'<span style="color:#718096;">{pos_str}</span>'
                        f'  <span style="color:#1a1a2e;">{seq_str}</span>'
                    )
                else:
                    html_lines.append(f'<span style="color:#1a1a2e;">{ln}</span>')
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
            "Properties Radar Chart": GraphingTools.create_radar_chart_figure(
                self.analysis_data, label_font=lf),
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
        }

        # Apply global heading/grid/colour overrides
        for title, fig in figs.items():
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
                self._update_seq_viewer()
                self.update_graph_tabs()
                return

    # --- Export ---

    def save_report(self):
        if not self.analysis_data:
            QMessageBox.warning(self, "No Data", "Run analysis first.")
            return
        fn, _ = QFileDialog.getSaveFileName(self, "Save Report", "", "Text Files (*.txt)")
        if fn:
            ok, msg = ExportTools.export_report_text(self.analysis_data, fn)
            QMessageBox.information(self, "Save", msg)

    def export_pdf(self):
        if not self.analysis_data:
            QMessageBox.warning(self, "No Data", "Run analysis first.")
            return
        fn, _ = QFileDialog.getSaveFileName(self, "Export PDF", "", "PDF Files (*.pdf)")
        if fn:
            ExportTools.export_pdf(
                self.analysis_data, self.graph_tabs, fn, self,
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
            canvas.figure.savefig(fn, format=ext, dpi=200,
                                   bbox_inches="tight", facecolor="white")
            QMessageBox.information(self, "Saved", f"{title} → {fn}")

    def save_all_graphs(self):
        d = QFileDialog.getExistingDirectory(self, "Select Directory")
        if not d:
            return
        ext = self.default_graph_format.lower()
        try:
            for title, (tab, vb) in self.graph_tabs.items():
                canvas = self._find_canvas(vb)
                if canvas:
                    path = os.path.join(d, title.replace(" ", "_") + f".{ext}")
                    canvas.figure.savefig(path, format=ext)
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
        self.colormap         = self.colormap_combo.currentText()
        try:
            self.label_font_size = int(self.label_font_input.text())
            self.tick_font_size  = int(self.tick_font_input.text())
            self.marker_size     = int(self.marker_size_input.text())
        except (ValueError, TypeError):
            pass
        self.graph_color          = self.graph_color_combo.currentText()
        self.show_heading         = self.heading_checkbox.isChecked()
        self.show_grid            = self.grid_checkbox.isChecked()
        self.default_graph_format = self.graph_format_combo.currentText()
        self.use_reducing         = self.reducing_checkbox.isChecked()

        # Sequence name override
        name_override = self.seq_name_input.text().strip()
        if name_override:
            self.sequence_name = name_override

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
        self.label_checkbox.setChecked(True)
        self.colormap_combo.setCurrentText("coolwarm")
        self.label_font_input.setText("14")
        self.tick_font_input.setText("12")
        self.marker_size_input.setText("10")
        self.graph_color_combo.setCurrentText("#4361ee")
        self.graph_format_combo.setCurrentText("PNG")
        self.tooltips_checkbox.setChecked(False)
        self.heading_checkbox.setChecked(True)
        self.grid_checkbox.setChecked(True)
        self.apply_settings()

    # --- Chain selection ---

    def on_chain_selected(self, text: str):
        for cid, seq, data in self.batch_data:
            if cid == text:
                self.seq_text.setPlainText(seq)
                self.analysis_data = data
                self.sequence_name = cid
                for sec, browser in self.report_section_tabs.items():
                    browser.setHtml(data["report_sections"][sec])
                self._update_seq_viewer()
                self.update_graph_tabs()
                break

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


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    w   = ProteinAnalyzerGUI()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
