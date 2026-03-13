#!/usr/bin/env python3
"""
BEER - Biochemical Estimator & Explorer of Residues

Requirements:
  pip install biopython matplotlib PyQt5 mplcursors
"""

import sys, math, os, base64, json, csv, subprocess, re
from io import BytesIO, StringIO
import urllib.request
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QTextBrowser,
    QFileDialog, QTabWidget, QMessageBox, QTableWidget, QTableWidgetItem,
    QCheckBox, QStatusBar, QComboBox, QFormLayout,
    QSplitter, QScrollArea, QFrame, QDialog, QDialogButtonBox,
    QSpinBox, QProgressDialog, QAbstractItemView,
    QListWidget, QListWidgetItem, QTreeWidget, QTreeWidgetItem, QStackedWidget,
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
        "Properties Radar Chart",
        "Helical Wheel",
        "TM Topology",
    ]),
    ("AlphaFold / Structural", [
        "pLDDT Profile",
        "Distance Map",
        "Domain Architecture",
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
                       min_len: int = 15, max_len: int = 35) -> list:
    """Predict TM helices using Kyte-Doolittle sliding window (TMHMM-heuristic).
    Returns list of dicts: {start(0-based), end(0-based inclusive), score, orientation}."""
    n    = len(seq)
    half = window // 2
    scores = [
        sum(KYTE_DOOLITTLE[seq[j]] for j in range(max(0, i - half), min(n, i + half + 1)))
        / (min(n, i + half + 1) - max(0, i - half))
        for i in range(n)
    ]
    helices = []
    i = 0
    while i < n:
        if scores[i] >= threshold:
            j = i
            while j < n and scores[j] >= threshold:
                j += 1
            span = j - i
            if min_len <= span <= max_len:
                helices.append({
                    "start": i, "end": j - 1,
                    "score": round(sum(scores[i:j]) / span, 3),
                })
            i = j
        else:
            i += 1
    # Inside-positive rule (von Heijne): cytoplasmic loops are K/R-enriched
    pos   = set("KR")
    flank = 15
    for h in helices:
        s, e  = h["start"], h["end"]
        n_pos = sum(1 for aa in seq[max(0, s - flank):s] if aa in pos)
        c_pos = sum(1 for aa in seq[e + 1:min(n, e + 1 + flank)] if aa in pos)
        h["orientation"] = "out\u2192in" if c_pos >= n_pos else "in\u2192out"
    return helices


def compute_ca_distance_matrix(pdb_str: str) -> np.ndarray:
    """Return symmetric Cα pairwise distance matrix (Å) from a PDB string."""
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("af", StringIO(pdb_str))
    coords = []
    for model in struct:
        for chain in model:
            for res in chain:
                if is_aa(res, standard=True) and res.has_id("CA"):
                    coords.append(res["CA"].get_vector().get_array())
        break
    if not coords:
        return np.array([])
    ca   = np.array(coords, dtype=float)
    diff = ca[:, np.newaxis, :] - ca[np.newaxis, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def extract_plddt_from_pdb(pdb_str: str) -> list:
    """Extract per-residue pLDDT scores (stored in B-factor column) from AlphaFold PDB."""
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("af", StringIO(pdb_str))
    scores = []
    for model in struct:
        for chain in model:
            for res in chain:
                if is_aa(res, standard=True):
                    for atom in res:
                        if atom.get_name() == "CA":
                            scores.append(atom.get_bfactor())
                            break
        break
    return scores


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
        }

    # predict_solubility removed – superseded by Hydrophobicity tab

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
        """Chou-Fasman per-residue helix and sheet propensity."""
        n  = len(cf_helix)
        xs = list(range(1, n + 1))
        fig = Figure(figsize=(9, 4), dpi=120)
        fig.set_facecolor("#ffffff")
        ax  = fig.add_subplot(111)
        ax.plot(xs, cf_helix, color="#4361ee", linewidth=1.6, label="Helix P\u03b1", zorder=4)
        ax.plot(xs, cf_sheet, color="#f72585", linewidth=1.6, label="Sheet P\u03b2", zorder=4)
        ax.fill_between(xs, cf_helix, 1.0,
                        where=[v > 1.0 for v in cf_helix],
                        alpha=0.20, color="#4361ee", interpolate=True)
        ax.fill_between(xs, cf_sheet, 1.0,
                        where=[v > 1.0 for v in cf_sheet],
                        alpha=0.20, color="#f72585", interpolate=True)
        ax.axhline(1.0, color="#888", linewidth=0.8, linestyle="--",
                   label="Neutral (1.0)", zorder=3)
        _pub_style_ax(ax,
                      title="Secondary Structure Propensity (Chou-Fasman)",
                      xlabel="Residue Position", ylabel="Propensity",
                      grid=True, title_size=label_font + 1,
                      label_size=label_font - 1, tick_size=tick_font - 1)
        ax.legend(fontsize=tick_font - 2, framealpha=0.85, edgecolor="#d0d4e0")
        fig.tight_layout(pad=1.5)
        mplcursors.cursor(ax)
        return fig

    @staticmethod
    def create_helical_wheel(seq, label_font=14):
        """Helical wheel projection: first ≤18 residues at 100° per residue."""
        seg = seq[:18]
        n   = len(seg)
        fig = Figure(figsize=(5.5, 5.5), dpi=120)
        fig.set_facecolor("#ffffff")
        ax  = fig.add_subplot(111, polar=True)
        ax.set_facecolor("#fafbff")
        cmap    = plt.get_cmap("RdYlBu_r")
        kd_min, kd_max = -4.5, 4.5
        for i, aa in enumerate(seg):
            ang  = math.radians(i * 100.0)
            kd   = KYTE_DOOLITTLE.get(aa, 0.0)
            nkd  = (kd - kd_min) / (kd_max - kd_min)
            col  = cmap(nkd)
            ax.scatter([ang], [1.0], s=520, c=[col], zorder=5,
                       edgecolors="white", linewidths=1.0)
            ax.text(ang, 1.0, aa, ha="center", va="center",
                    fontsize=label_font - 3, fontweight="bold",
                    color="white" if nkd > 0.4 else "#1a1a2e", zorder=6)
            ax.text(ang, 1.24, str(i + 1), ha="center", va="bottom",
                    fontsize=label_font - 5, color="#718096", zorder=6)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines["polar"].set_visible(False)
        ax.set_title(f"Helical Wheel  (residues 1\u2013{n})",
                     fontsize=label_font, fontweight="bold", color="#1a1a2e", pad=14)
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        sm  = ScalarMappable(cmap=cmap, norm=Normalize(vmin=kd_min, vmax=kd_max))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.65, pad=0.06, aspect=20)
        cbar.set_label("Hydrophobicity (KD)", fontsize=label_font - 4, color="#4a5568")
        cbar.ax.tick_params(labelsize=label_font - 5, colors="#4a5568")
        fig.tight_layout(pad=2.0)
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
        """Per-residue AlphaFold pLDDT confidence score with coloured confidence zones."""
        import matplotlib.colors as mcolors
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
                      title="AlphaFold pLDDT Confidence",
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
        """Cα pairwise distance heatmap from AlphaFold structure.
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
                                          label_font=14, tick_font=12):
        """Linear domain architecture ruler from Pfam/InterPro annotations."""
        fig = Figure(figsize=(9, max(2.5, 1.0 + len(domains) * 0.5)), dpi=120)
        fig.set_facecolor("#ffffff")
        ax  = fig.add_subplot(111)
        ax.set_facecolor("#fafbff")
        # Backbone
        ax.plot([1, seq_len], [0, 0], color="#94a3b8", linewidth=3,
                solid_capstyle="round", zorder=2)
        ax.text(1,        0.18, "N", ha="center", fontsize=tick_font - 2, color="#4a5568")
        ax.text(seq_len, 0.18, "C", ha="center", fontsize=tick_font - 2, color="#4a5568")
        for i, dom in enumerate(domains):
            col  = _PALETTE[i % len(_PALETTE)]
            s, e = dom["start"], dom["end"]
            rect = Rectangle((s, -0.25), e - s, 0.5,
                              color=col, alpha=0.85, zorder=4, linewidth=0)
            ax.add_patch(rect)
            mid = (s + e) / 2
            ax.text(mid, 0, dom["name"][:14],
                    ha="center", va="center",
                    fontsize=max(5, tick_font - 5), color="white",
                    fontweight="bold", zorder=5)
        _pub_style_ax(ax,
                      title="Domain Architecture (Pfam/InterPro)",
                      xlabel="Residue Position", ylabel="",
                      grid=False, title_size=label_font + 1,
                      label_size=label_font - 1, tick_size=tick_font - 1)
        ax.set_xlim(0, seq_len + 5)
        ax.set_ylim(-0.6, 0.6)
        ax.set_yticks([])
        if domains:
            ax.legend(
                handles=[Patch(color=_PALETTE[i % len(_PALETTE)], label=d["name"])
                         for i, d in enumerate(domains)],
                fontsize=max(6, tick_font - 4), framealpha=0.85,
                edgecolor="#d0d4e0", loc="upper right",
                ncol=max(1, len(domains) // 6)
            )
        fig.tight_layout(pad=1.5)
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
            import re as _re
            content = _re.sub(r"<style>[^<]*</style>", "", content, flags=_re.DOTALL)
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

# --- Batch stats helper ---

def _calc_batch_stats(seq: str, data: dict) -> tuple:
    """Return (hydro%, hydrophil%, pos%, neg%, neu%) for a sequence."""
    length = len(seq)
    hydro  = sum(1 for aa in seq if KYTE_DOOLITTLE[aa] > 0) / length * 100
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
                name = raw_name.get("name", raw_name) if isinstance(raw_name, dict) else raw_name
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
            from Bio.Blast import NCBIWWW, NCBIXML
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


# --- Navigation sidebar widget ---

class NavTabWidget(QWidget):
    """Left-sidebar navigation that is a drop-in replacement for QTabWidget.
    Implements the subset of QTabWidget API used in this app."""

    _NAV_ICONS = {
        "Analysis":            "🧪",
        "Graphs":              "📊",
        "Structure":           "🔬",
        "BLAST":               "🔍",
        "Compare":             "⚖\ufe0f",
        "Multichain Analysis": "📋",
        "Settings":            "⚙\ufe0f",
        "Help":                "❓",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.nav_list = QListWidget()
        self.nav_list.setObjectName("nav_bar")
        self.nav_list.setFixedWidth(136)
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
        self.enable_tooltips     = False
        self.use_reducing        = False
        self.custom_pka          = None
        self.sequence_name       = ""   # display name for current sequence
        self.app_font_size       = 12   # global UI font size (pt)
        self._tooltips: dict     = {}  # widget -> tooltip text
        self.transparent_bg      = False
        self._analysis_worker    = None
        self._history: list      = []   # list of (name, seq)

        # --- New state for AlphaFold / Pfam / BLAST ---
        self.current_accession   = ""   # last successfully fetched UniProt accession
        self.alphafold_data      = None # dict: pdb_str, plddt, dist_matrix, accession
        self.pfam_domains        = []   # list of domain dicts from Pfam
        self._alphafold_worker   = None
        self._pfam_worker        = None
        self._blast_worker       = None

        self.check_dependencies()
        self.main_tabs = NavTabWidget()
        self.setCentralWidget(self.main_tabs)
        self.init_analysis_tab()
        self.init_graphs_tab()
        self.init_structure_tab()
        self.init_blast_tab()
        self.init_batch_tab()
        self.init_comparison_tab()
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
        for w in (self.import_fasta_btn, self.import_pdb_btn, self.analyze_btn,
                  self.save_pdf_btn, self.mutate_btn,
                  self.session_save_btn, self.session_load_btn):
            w.setMinimumHeight(32)
            toolbar.addWidget(w)
        toolbar.addStretch()
        outer.addLayout(toolbar)

        # ---- toolbar row 2: UniProt/NCBI fetch + history ----
        tb2 = QHBoxLayout()
        tb2.setSpacing(6)
        tb2.addWidget(QLabel("Fetch accession:"))
        self.accession_input = QLineEdit()
        self.accession_input.setPlaceholderText("UniProt ID or NCBI accession")
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
        """Tab for 3D AlphaFold structure viewer and pLDDT info."""
        container = QWidget()
        layout    = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        self.main_tabs.addTab(container, "Structure")

        info_row = QHBoxLayout()
        self.af_status_lbl = QLabel("No structure loaded.  Use 'Fetch AlphaFold' after fetching a UniProt accession.")
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

        self.transparent_bg_checkbox = QCheckBox("Transparent background (PNG/SVG graph export)")
        self.transparent_bg_checkbox.setChecked(False)
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
  <li><b>Fetch accession</b> — enter a UniProt ID (e.g. <tt>P04637</tt>) or NCBI accession and click <b>Fetch</b>. This also enables the <b>Fetch AlphaFold</b> and <b>Fetch Pfam</b> buttons.</li>
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
"""),
            ("Transmembrane Helices", """
<h1>Transmembrane Helix Prediction</h1>
<p>Available in the <b>TM Helices</b> report section and the <b>TM Topology</b> graph
after running analysis. No external server required — prediction is purely sequence-based.</p>
<h2>Algorithm</h2>
<ol>
  <li>A <b>sliding window</b> (width = 19) of Kyte-Doolittle scores is computed at every position.</li>
  <li>Contiguous runs where the window score exceeds <b>1.6</b> are merged into candidate helices.</li>
  <li>Candidates with length outside <b>15–35 aa</b> are discarded.</li>
  <li><b>Inside-positive rule (von Heijne)</b> — the flanking 15 residues on each side are scanned
      for K and R. The side with more positively charged residues is assigned as cytoplasmic.
      <ul>
        <li><b>out→in</b>: N-terminus is extracellular, C-terminus is cytoplasmic.</li>
        <li><b>in→out</b>: N-terminus is cytoplasmic, C-terminus is extracellular.</li>
      </ul>
  </li>
</ol>
<h2>TM Topology graph</h2>
<p>A simplified snake-plot. The yellow band represents the membrane. Blue rectangles are TM helices
labelled with their residue range. Loops are drawn above (extracellular) or below (cytoplasmic)
the band according to the predicted topology.</p>
<p class="note">Note: This is a heuristic predictor suitable for a first-pass screen.
For high-accuracy results use TMHMM, Phobius, or DeepTMHMM.</p>
"""),
            ("AlphaFold & 3D Structure", """
<h1>AlphaFold Integration</h1>
<p>Requires an internet connection and a valid UniProt accession (fetch it with the
<b>Fetch</b> button in the Analysis toolbar first). Then click <b>Fetch AlphaFold</b>.</p>
<h2>What gets downloaded</h2>
<ul>
  <li>AlphaFold2 predicted PDB file from the EBI server.</li>
  <li>Per-residue <b>pLDDT</b> scores (stored in the B-factor column of the PDB).</li>
  <li>Cα pairwise <b>distance matrix</b> computed from the structure coordinates.</li>
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
after loading an accession.</p>
<h2>Data source</h2>
<p>Queries the <b>EMBL-EBI InterPro REST API</b> for all Pfam-family entries associated
with the given UniProt protein. Results include domain name, accession, and start/end residue
positions.</p>
<h2>Domain Architecture graph</h2>
<p>A linear ruler from N- to C-terminus. Each Pfam domain is drawn as a coloured box labelled
with a truncated domain name (hover for full label in the legend). Domains are coloured from the
BEER palette in the order they appear. Overlapping domains are all drawn at the same height.</p>
<p class="note">Only Pfam entries are shown. InterPro, SUPERFAMILY, PRINTS, and other databases
are excluded for clarity.</p>
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
<p>All graphs are accessible from the <b>Graphs</b> section. Use the category tree on the
left to navigate. Each graph has its own <b>Save Graph</b> button; <b>Save All Graphs</b>
exports the whole collection to a chosen directory.</p>
<h2>Composition</h2>
<ul>
  <li><b>Bar / Pie Chart</b> — amino acid counts and frequencies. Bar chart sort order matches
      the Composition report section buttons.</li>
</ul>
<h2>Profiles</h2>
<ul>
  <li><b>Hydrophobicity Profile</b> — Kyte-Doolittle sliding-window average (window set in Settings).</li>
  <li><b>Local Charge Profile</b> — sliding-window NCPR; shows charge blocks.</li>
  <li><b>Local Complexity</b> — sliding-window Shannon entropy; red dashed line = LC threshold (2.0 bits).</li>
  <li><b>Disorder Profile</b> — IUPred-inspired per-residue score; orange fill = disordered (&gt;0.5).</li>
  <li><b>Linear Sequence Map</b> — four-track overview: hydrophobicity, NCPR, disorder, helix Pα.</li>
  <li><b>Secondary Structure</b> — Chou-Fasman per-residue Pα (helix) and Pβ (sheet) propensities.</li>
</ul>
<h2>Charge &amp; π-Interactions</h2>
<ul>
  <li><b>Net Charge vs pH</b> — Henderson-Hasselbalch charge curve 0–14; pI marked.</li>
  <li><b>Isoelectric Focus</b> — enhanced version with physiological pH 7.4 annotation.</li>
  <li><b>Charge Decoration</b> — Das-Pappu FCR vs |NCPR| phase diagram; star = this protein.</li>
  <li><b>Cation–π Map</b> — proximity heat map (1/distance weight) for K/R ↔ F/W/Y pairs.</li>
</ul>
<h2>Structure &amp; Folding</h2>
<ul>
  <li><b>Bead Model (Hydrophobicity)</b> — per-residue KD score, coolwarm colourmap.</li>
  <li><b>Bead Model (Charge)</b> — K/R blue, D/E red, H cyan, neutral grey.</li>
  <li><b>Sticker Map</b> — aromatic (amber), basic (blue), acidic (pink), spacer (grey).</li>
  <li><b>Properties Radar Chart</b> — five normalised properties: MW, pI, GRAVY, instability, aromaticity.</li>
  <li><b>Helical Wheel</b> — projection of first 18 residues at 100° per step, KD coloured.</li>
  <li><b>TM Topology</b> — snake-plot of predicted transmembrane helices (see TM Helices section).</li>
</ul>
<h2>AlphaFold / Structural</h2>
<ul>
  <li><b>pLDDT Profile</b> — per-residue AlphaFold confidence (see AlphaFold section). Requires Fetch AlphaFold.</li>
  <li><b>Distance Map</b> — Cα pairwise distance heatmap with 8 Å contact contour. Requires Fetch AlphaFold.</li>
  <li><b>Domain Architecture</b> — linear Pfam domain map. Requires Fetch Pfam.</li>
</ul>
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
  <li><b>Default pH</b> — pH used for net-charge calculations (0–14).</li>
  <li><b>Sliding Window Size</b> — window width for hydrophobicity, NCPR, and entropy profiles.</li>
  <li><b>Override pKa</b> — custom pKa values (N-term, C-term, D, E, C, Y, H, K, R) as comma-separated numbers.</li>
  <li><b>Reducing conditions</b> — if checked, Cys residues are not counted as disulphide pairs for extinction coefficient.</li>
</ul>
<h2>Graph Appearance</h2>
<ul>
  <li><b>Label / Tick Font Size</b> — point size of axis titles and tick labels.</li>
  <li><b>Default Graph Format</b> — PNG, SVG, or PDF for Save Graph / Save All.</li>
  <li><b>Bead Colormap</b> — matplotlib colourmap for the Bead Hydrophobicity model.</li>
  <li><b>Graph Accent Colour</b> — primary line/fill colour for most graphs.</li>
  <li><b>Transparent background</b> — export graphs with alpha = 0 (PNG/SVG only).</li>
</ul>
<h2>Interface</h2>
<ul>
  <li><b>UI Font Size</b> — global application font size in points.</li>
  <li><b>Dark Theme</b> — toggles between light and dark colour themes.</li>
  <li><b>Enable Tooltips</b> — show tooltips on Settings widgets.</li>
</ul>
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

        # Structure-dependent graphs (only when AlphaFold data is loaded)
        if self.alphafold_data:
            if self.alphafold_data.get("plddt"):
                figs["pLDDT Profile"] = GraphingTools.create_plddt_figure(
                    self.alphafold_data["plddt"], label_font=lf, tick_font=tf)
            dm = self.alphafold_data.get("dist_matrix")
            if dm is not None and dm.size > 0:
                figs["Distance Map"] = GraphingTools.create_distance_map_figure(
                    dm, label_font=lf, tick_font=tf)

        # Domain architecture (only when Pfam data is loaded)
        if self.pfam_domains:
            figs["Domain Architecture"] = GraphingTools.create_domain_architecture_figure(
                len(seq), self.pfam_domains, label_font=lf, tick_font=tf)

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
        self.transparent_bg   = self.transparent_bg_checkbox.isChecked()
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
        self.app_font_size_input.setText("12")
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
        self.analysis_data = data
        self._add_to_history(self.sequence_name, seq)
        for sec, browser in self.report_section_tabs.items():
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

    # --- Accession fetch ---

    def fetch_accession(self):
        acc = self.accession_input.text().strip()
        if not acc:
            QMessageBox.warning(self, "Fetch", "Enter a UniProt accession.")
            return
        url = f"https://rest.uniprot.org/uniprotkb/{acc}.fasta"
        self.statusBar.showMessage(f"Fetching {acc}…")
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                raw = resp.read().decode()
        except Exception as e:
            self.statusBar.showMessage("Fetch failed", 3000)
            QMessageBox.warning(self, "Fetch Failed", str(e))
            return
        entries = self._parse_pasted_text(raw)
        if not entries:
            QMessageBox.warning(self, "Fetch", "No valid protein sequence returned.")
            return
        rid, seq = entries[0]
        self.seq_text.setPlainText(seq)
        self.sequence_name = rid
        # Store the raw accession for AlphaFold / Pfam lookups
        self.current_accession = acc
        self.fetch_af_btn.setEnabled(True)
        self.fetch_pfam_btn.setEnabled(True)
        self.accession_input.clear()
        self.statusBar.showMessage(f"Fetched {rid}  ({len(seq)} aa)", 3000)

    # --- AlphaFold ---

    def fetch_alphafold(self):
        acc = self.current_accession
        if not acc:
            QMessageBox.warning(self, "AlphaFold", "Fetch a UniProt accession first.")
            return
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
            QMessageBox.warning(self, "Pfam", "Fetch a UniProt accession first.")
            return
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
