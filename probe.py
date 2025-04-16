#!/usr/bin/env python3
"""
Protein Analyzer GUI Application with Professional Reporting, External Annotation Placeholder, and Advanced Settings

Features:
  - Input protein sequence manually (editable QTextEdit) or via FASTA import.
  - Compute properties: amino acid composition, molecular weight, pI, molar extinction coefficient, 
    GRAVY, instability index, aromaticity, secondary structure, net charge, and hydrophobicity profile.
  - Reports are displayed as HTML tables, including a new Annotation section (placeholder).
  - Embedded graphs with adjustable fonts and colormap, plus navigation toolbars.
  - Batch Analysis: import multi-FASTA files and view a summary table.
  - Settings tab allows adjustment of sliding window size, graph font sizes, colormap, bead labels, and theme.
  - Export options: save the report as text, export combined report (including graphs) to PDF, and print preview.
  - An “Apply Settings” button dynamically updates reports and graphs.
  
Author: Saumyak Mukherjee  
Contact: saumyak.mukherjee@biophys.mpg.de

Requirements:
  pip install biopython matplotlib PyQt5
"""

import sys, math, os, base64
from io import BytesIO

# PyQt5 Imports (note: QPrintPreviewDialog from QtPrintSupport)
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QTextEdit, QTextBrowser,
                             QFileDialog, QTabWidget, QMessageBox, QTableWidget, QTableWidgetItem,
                             QCheckBox, QStatusBar, QComboBox)
from PyQt5.QtCore import Qt
from PyQt5.QtPrintSupport import QPrintPreviewDialog, QPrinter

# Matplotlib imports; default style is "default" (light mode)
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
import matplotlib.pyplot as plt
plt.style.use("default")  # default light mode

# Biopython imports
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio import SeqIO

# ---------------------------
# Helper Functions
# ---------------------------
VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
KYTE_DOOLITTLE = {
    'A': 1.8,  'R': -4.5, 'N': -3.5, 'D': -3.5,
    'C': 2.5,  'Q': -3.5, 'E': -3.5, 'G': -0.4,
    'H': -3.2, 'I': 4.5,  'L': 3.8,  'K': -3.9,
    'M': 1.9,  'F': 2.8,  'P': -1.6, 'S': -0.8,
    'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

def clean_sequence(seq: str) -> str:
    return seq.strip().replace(" ", "").upper()

def is_valid_protein(seq: str) -> bool:
    return all(aa in VALID_AMINO_ACIDS for aa in seq)

def calc_net_charge(seq: str, pH: float = 7.0) -> float:
    pKa_nterm = 9.69
    pKa_cterm = 2.34
    pKa_side = {'D': 3.90, 'E': 4.07, 'C': 8.18, 'Y': 10.46,
                'H': 6.04, 'K': 10.54, 'R': 12.48}
    net_charge = 1.0/(1.0+10**(pH-pKa_nterm)) - 1.0/(1.0+10**(pKa_cterm-pH))
    for aa in seq:
        if aa in ('D', 'E', 'C', 'Y'):
            net_charge += -1.0/(1.0+10**(pKa_side[aa]-pH))
        elif aa in ('K', 'R', 'H'):
            net_charge += 1.0/(1.0+10**(pH-pKa_side[aa]))
    return net_charge

def sliding_window_hydrophobicity(seq: str, window_size: int = 9) -> list:
    profile = []
    if window_size > len(seq):
        return [sum(KYTE_DOOLITTLE[aa] for aa in seq)/len(seq)]
    for i in range(len(seq)-window_size+1):
        window = seq[i:i+window_size]
        profile.append(sum(KYTE_DOOLITTLE[aa] for aa in window)/window_size)
    return profile

def annotate_sequence(seq: str) -> str:
    # Placeholder for external annotation integration.
    return """
    <h2>Annotation</h2>
    <table border="1" cellpadding="5">
      <tr><td>External Annotation</td><td>No annotation available (feature under development)</td></tr>
    </table>
    """

def analyze_sequence(seq: str, pH_value: float = 7.0, window_size: int = 9) -> dict:
    pa = ProteinAnalysis(seq)
    aa_counts = pa.count_amino_acids()
    seq_length = len(seq)
    aa_freq = {aa: (aa_counts[aa]/seq_length*100) for aa in aa_counts}
    mol_weight = pa.molecular_weight()
    iso_point = pa.isoelectric_point()
    extinction_coeffs = pa.molar_extinction_coefficient()
    gravy = pa.gravy()
    instability = pa.instability_index()
    aromaticity = pa.aromaticity()
    sec_struct = pa.secondary_structure_fraction()  # (helix, turn, sheet)
    pos_count = aa_counts.get('K',0) + aa_counts.get('R',0) + aa_counts.get('H',0)
    neg_count = aa_counts.get('D',0) + aa_counts.get('E',0)
    net_charge_7 = calc_net_charge(seq, 7.0)
    net_charge_custom = calc_net_charge(seq, pH_value)
    hydro_profile = sliding_window_hydrophobicity(seq, window_size)
    if hydro_profile:
        hydro_avg = sum(hydro_profile)/len(hydro_profile)
        hydro_min = min(hydro_profile)
        hydro_max = max(hydro_profile)
    else:
        hydro_avg = hydro_min = hydro_max = 0.0

    # Build HTML tables for each section.
    overview_html = f"""
    <h2>Overview</h2>
    <table border="1" cellpadding="5">
      <tr><th>Property</th><th>Value</th></tr>
      <tr><td>Sequence Length</td><td>{seq_length} amino acids</td></tr>
      <tr><td>Sequence</td><td>{seq}</td></tr>
    </table>
    """
    comp_html = "<h2>Amino Acid Composition</h2><table border=\"1\" cellpadding=\"5\"><tr><th>Amino Acid</th><th>Count</th><th>Frequency (%)</th></tr>" \
                + "".join([f"<tr><td>{aa}</td><td>{aa_counts[aa]}</td><td>{aa_freq[aa]:.2f}</td></tr>" for aa in sorted(aa_counts)]) \
                + "</table>"
    bio_html = f"""
    <h2>Biochemical Properties</h2>
    <table border="1" cellpadding="5">
      <tr><td>Molecular Weight</td><td>{mol_weight:.2f} Da</td></tr>
      <tr><td>Isoelectric Point (pI)</td><td>{iso_point:.2f}</td></tr>
      <tr><td>Molar Extinction Coefficient</td><td>{extinction_coeffs} M<sup>-1</sup> cm<sup>-1</sup></td></tr>
      <tr><td>GRAVY Score</td><td>{gravy:.3f}</td></tr>
      <tr><td>Instability Index</td><td>{instability:.2f} &lt; 40 indicates stability</td></tr>
      <tr><td>Aromaticity</td><td>{aromaticity:.3f}</td></tr>
      <tr><td>Grouped Counts</td><td>Positive (K,R,H): {pos_count}, Negative (D,E): {neg_count}</td></tr>
    </table>
    """
    sec_html = f"""
    <h2>Secondary Structure</h2>
    <table border="1" cellpadding="5">
      <tr><td>Helix</td><td>{sec_struct[0]*100:.2f}%</td></tr>
      <tr><td>Turn</td><td>{sec_struct[1]*100:.2f}%</td></tr>
      <tr><td>Sheet</td><td>{sec_struct[2]*100:.2f}%</td></tr>
    </table>
    """
    if abs(pH_value - 7.0) < 1e-6:
        net_html = f"""
        <h2>Net Charge</h2>
        <table border="1" cellpadding="5">
          <tr><td>At pH 7.0</td><td>{net_charge_7:.2f}</td></tr>
        </table>
        """
    else:
        net_html = f"""
        <h2>Net Charge</h2>
        <table border="1" cellpadding="5">
          <tr><td>At pH 7.0</td><td>{net_charge_7:.2f}</td></tr>
          <tr><td>At pH {pH_value:.1f}</td><td>{net_charge_custom:.2f}</td></tr>
        </table>
        """
    hydro_html = f"""
    <h2>Hydrophobicity Profile</h2>
    <table border="1" cellpadding="5">
      <tr><td>Sliding Window Size</td><td>{window_size}</td></tr>
      <tr><td>Average</td><td>{hydro_avg:.3f}</td></tr>
      <tr><td>Minimum</td><td>{hydro_min:.3f}</td></tr>
      <tr><td>Maximum</td><td>{hydro_max:.3f}</td></tr>
    </table>
    """
    annotation_html = annotate_sequence(seq)
    
    report_sections = {
        "Overview": overview_html,
        "Amino Acid Composition": comp_html,
        "Biochemical Properties": bio_html,
        "Secondary Structure": sec_html,
        "Net Charge": net_html,
        "Hydrophobicity Profile": hydro_html,
        "Annotation": annotation_html
    }
    return {
        "report_sections": report_sections,
        "aa_counts": aa_counts,
        "aa_freq": aa_freq,
        "hydro_profile": hydro_profile,
        "window_size": window_size,
        "seq": seq,
        "mol_weight": mol_weight,
        "iso_point": iso_point,
        "net_charge_7": net_charge_7,
        "extinction_coeffs": extinction_coeffs
    }

# ---------------------------
# Plotting Functions (with adjustable fonts/colormap)
# ---------------------------
def create_amino_acid_composition_figure(aa_counts: dict, aa_freq: dict,
                                          label_font: int=14, tick_font: int=12) -> Figure:
    fig = Figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    amino_acids = sorted(aa_counts.keys())
    counts = [aa_counts[aa] for aa in amino_acids]
    ax.bar(amino_acids, counts, color="skyblue")
    ax.set_xlabel("Amino Acids", fontsize=label_font)
    ax.set_ylabel("Counts", fontsize=label_font)
    ax.set_title("Amino Acid Composition", fontsize=label_font+2)
    ax.tick_params(labelsize=tick_font)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    for i, aa in enumerate(amino_acids):
        ax.text(i, counts[i]+0.5, f"{aa_freq[aa]:.1f}%", ha="center", fontsize=10)
    return fig

def create_hydrophobicity_figure(hydro_profile: list, window_size: int,
                                 label_font: int=14, tick_font: int=12) -> Figure:
    fig = Figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    x_vals = list(range(1, len(hydro_profile)+1))
    ax.plot(x_vals, hydro_profile, marker="o", linestyle="-", color="magenta")
    ax.set_xlabel("Window Start Position", fontsize=label_font)
    ax.set_ylabel("Avg. Hydrophobicity", fontsize=label_font)
    ax.set_title(f"Hydrophobicity Profile (Window = {window_size})", fontsize=label_font+2)
    ax.tick_params(labelsize=tick_font)
    ax.grid(True)
    return fig

def create_net_charge_vs_pH_figure(seq: str,
                                   label_font: int=14, tick_font: int=12) -> Figure:
    fig = Figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    pH_vals = [i/10.0 for i in range(0,141)]
    net_charges = [calc_net_charge(seq, pH) for pH in pH_vals]
    ax.plot(pH_vals, net_charges, color="green")
    ax.set_xlabel("pH", fontsize=label_font)
    ax.set_ylabel("Net Charge", fontsize=label_font)
    ax.set_title("Net Charge vs pH", fontsize=label_font+2)
    ax.tick_params(labelsize=tick_font)
    ax.grid(True)
    return fig

def create_bead_model_hydrophobicity_figure(seq: str, show_labels: bool,
                                            label_font: int=14, tick_font: int=12,
                                            cmap: str="coolwarm") -> Figure:
    fig = Figure(figsize=(min(0.25*len(seq), 12), 2))
    ax = fig.add_subplot(111)
    positions = list(range(1, len(seq)+1))
    hydrophobicities = [KYTE_DOOLITTLE[aa] for aa in seq]
    sc = ax.scatter(positions, [1]*len(seq), c=hydrophobicities, cmap=cmap, s=200, edgecolors="k")
    cbar = fig.colorbar(sc, ax=ax, label="Hydrophobicity (Kyte-Doolittle)")
    cbar.ax.yaxis.label.set_fontsize(label_font)
    ax.set_xlabel("Residue Position", fontsize=label_font)
    ax.set_yticks([])
    ax.set_title("Bead Model: Hydrophobicity", fontsize=label_font+2)
    ax.tick_params(labelsize=tick_font)
    if show_labels and len(seq) <= 50:
        for i, aa in enumerate(seq):
            ax.text(positions[i], 1, aa, ha="center", va="center", color="white", fontsize=label_font-2)
    return fig

def create_bead_model_charge_figure(seq: str, show_labels: bool,
                                    label_font: int=14, tick_font: int=12) -> Figure:
    fig = Figure(figsize=(min(0.25*len(seq), 12), 2))
    ax = fig.add_subplot(111)
    positions = list(range(1, len(seq)+1))
    colors = []
    for aa in seq:
        if aa in ('K','R','H'):
            colors.append("blue")
        elif aa in ('D','E'):
            colors.append("red")
        else:
            colors.append("gray")
    ax.scatter(positions, [1]*len(seq), c=colors, s=200, edgecolors="k")
    ax.set_xlabel("Residue Position", fontsize=label_font)
    ax.set_yticks([])
    ax.set_title("Bead Model: Charge", fontsize=label_font+2)
    ax.tick_params(labelsize=tick_font)
    if show_labels and len(seq) <= 50:
        for i, aa in enumerate(seq):
            ax.text(positions[i], 1, aa, ha="center", va="center", color="white", fontsize=label_font-2)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="blue", edgecolor="k", label="Positive"),
                       Patch(facecolor="red", edgecolor="k", label="Negative"),
                       Patch(facecolor="gray", edgecolor="k", label="Neutral")]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=tick_font)
    return fig

# ---------------------------
# Main GUI Application
# ---------------------------
class ProteinAnalyzerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Protein Sequence Analyzer")
        self.resize(1200,900)
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.analysis_data = None
        # Default settings
        self.default_window_size = 9
        self.show_bead_labels = True
        self.current_theme = "Light"  # default now is Light mode
        self.label_font_size = 14
        self.tick_font_size = 12
        self.colormap = "coolwarm"
        
        self.main_tabs = QTabWidget()
        self.setCentralWidget(self.main_tabs)
        self.init_analysis_tab()
        self.init_graphs_tab()
        self.init_batch_tab()
        self.init_settings_tab()
        self.init_help_tab()
    
    # Analysis Tab
    def init_analysis_tab(self):
        layout = QVBoxLayout()
        container = QWidget()
        container.setLayout(layout)
        self.main_tabs.addTab(container, "Analysis")
        # Top row: Import and Analyze buttons
        top_layout = QHBoxLayout()
        self.import_button = QPushButton("Import FASTA File")
        self.import_button.setToolTip("Import a FASTA file")
        self.import_button.clicked.connect(self.import_fasta)
        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.setToolTip("Analyze the entered sequence")
        self.analyze_button.clicked.connect(self.on_analyze)
        top_layout.addWidget(self.import_button)
        top_layout.addWidget(self.analyze_button)
        layout.addLayout(top_layout)
        # pH input
        ph_layout = QHBoxLayout()
        ph_label = QLabel("pH for Net Charge:")
        self.ph_input = QLineEdit("7.0")
        ph_layout.addWidget(ph_label)
        ph_layout.addWidget(self.ph_input)
        layout.addLayout(ph_layout)
        # Export buttons row
        export_layout = QHBoxLayout()
        self.save_report_button = QPushButton("Save Report")
        self.save_report_button.clicked.connect(self.save_report)
        self.export_pdf_button = QPushButton("Export to PDF")
        self.export_pdf_button.clicked.connect(self.export_pdf)
        self.print_preview_button = QPushButton("Print Preview")
        self.print_preview_button.clicked.connect(self.print_preview)
        export_layout.addWidget(self.save_report_button)
        export_layout.addWidget(self.export_pdf_button)
        export_layout.addWidget(self.print_preview_button)
        layout.addLayout(export_layout)
        # Sequence input area (editable)
        seq_label = QLabel("Protein Sequence:")
        layout.addWidget(seq_label)
        self.seq_text = QTextEdit()
        self.seq_text.setPlaceholderText("Enter protein sequence (only standard amino acids)...")
        layout.addWidget(self.seq_text)
        # Report Tabs (including new Annotation tab)
        self.report_tabs = QTabWidget()
        layout.addWidget(self.report_tabs)
        self.report_section_tabs = {}
        for title in ["Overview", "Amino Acid Composition", "Biochemical Properties",
                      "Secondary Structure", "Net Charge", "Hydrophobicity Profile", "Annotation"]:
            tab = QWidget()
            vbox = QVBoxLayout(tab)
            browser = QTextBrowser()
            vbox.addWidget(browser)
            self.report_tabs.addTab(tab, title)
            self.report_section_tabs[title] = browser
    
    # Graphs Tab
    def init_graphs_tab(self):
        self.graphs_subtabs = QTabWidget()
        layout = QVBoxLayout()
        container = QWidget()
        container.setLayout(layout)
        self.main_tabs.addTab(container, "Graphs")
        layout.addWidget(self.graphs_subtabs)
        self.graph_tabs = {}
        for title in ["Amino Acid Composition", "Hydrophobicity Profile", "Net Charge vs pH",
                      "Bead Model (Hydrophobicity)", "Bead Model (Charge)"]:
            tab = QWidget()
            vbox = QVBoxLayout(tab)
            placeholder = QLabel(f"{title} graph will appear here after analysis.")
            placeholder.setAlignment(Qt.AlignCenter)
            vbox.addWidget(placeholder)
            btn = QPushButton("Save Graph")
            btn.clicked.connect(lambda _, t=title: self.save_graph(t))
            vbox.addWidget(btn, alignment=Qt.AlignRight)
            self.graphs_subtabs.addTab(tab, title)
            self.graph_tabs[title] = (tab, vbox)
    
    # Batch Analysis Tab
    def init_batch_tab(self):
        layout = QVBoxLayout()
        container = QWidget()
        container.setLayout(layout)
        self.main_tabs.addTab(container, "Batch Analysis")
        top_layout = QHBoxLayout()
        self.batch_import_button = QPushButton("Import Multi-FASTA File")
        self.batch_import_button.clicked.connect(self.import_batch)
        top_layout.addWidget(self.batch_import_button)
        layout.addLayout(top_layout)
        self.batch_table = QTableWidget()
        self.batch_table.setColumnCount(5)
        self.batch_table.setHorizontalHeaderLabels(["ID", "Length", "MW (Da)", "pI", "Net Charge (pH 7)"])
        self.batch_table.cellDoubleClicked.connect(self.show_batch_details)
        layout.addWidget(self.batch_table)
    
    # Settings Tab
    def init_settings_tab(self):
        layout = QVBoxLayout()
        container = QWidget()
        container.setLayout(layout)
        self.main_tabs.addTab(container, "Settings")
        # Theme toggle
        theme_layout = QHBoxLayout()
        theme_label = QLabel("Theme:")
        self.theme_toggle = QCheckBox("Dark Theme")
        self.theme_toggle.setChecked(False)  # default Light mode
        self.theme_toggle.stateChanged.connect(self.toggle_theme)
        theme_layout.addWidget(theme_label)
        theme_layout.addWidget(self.theme_toggle)
        layout.addLayout(theme_layout)
        # Sliding window size
        hw_layout = QHBoxLayout()
        hw_label = QLabel("Sliding Window Size:")
        self.window_size_input = QLineEdit(str(self.default_window_size))
        hw_layout.addWidget(hw_label)
        hw_layout.addWidget(self.window_size_input)
        layout.addLayout(hw_layout)
        # Bead label toggle
        self.label_checkbox = QCheckBox("Show residue labels in bead plots (if sequence ≤ 50)")
        self.label_checkbox.setChecked(self.show_bead_labels)
        layout.addWidget(self.label_checkbox)
        # Graph label font size
        label_font_layout = QHBoxLayout()
        label_font_label = QLabel("Graph Label Font Size:")
        self.label_font_input = QLineEdit("14")
        label_font_layout.addWidget(label_font_label)
        label_font_layout.addWidget(self.label_font_input)
        layout.addLayout(label_font_layout)
        # Graph tick font size
        tick_font_layout = QHBoxLayout()
        tick_font_label = QLabel("Graph Tick Font Size:")
        self.tick_font_input = QLineEdit("12")
        tick_font_layout.addWidget(tick_font_label)
        tick_font_layout.addWidget(self.tick_font_input)
        layout.addLayout(tick_font_layout)
        # Colormap selection
        colormap_layout = QHBoxLayout()
        colormap_label = QLabel("Colormap:")
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(["viridis", "plasma", "coolwarm", "inferno", "magma"])
        colormap_layout.addWidget(colormap_label)
        colormap_layout.addWidget(self.colormap_combo)
        layout.addLayout(colormap_layout)
        # Apply Settings button for dynamic update
        apply_btn = QPushButton("Apply Settings")
        apply_btn.clicked.connect(self.apply_settings)
        layout.addWidget(apply_btn)
        layout.addStretch()
    
    # Help Tab
    def init_help_tab(self):
        layout = QVBoxLayout()
        container = QWidget()
        container.setLayout(layout)
        self.main_tabs.addTab(container, "Help")
        help_browser = QTextBrowser()
        help_browser.setHtml("""
        <h1>Protein Sequence Analyzer - Help</h1>
        <p>This software analyzes protein sequences and provides detailed reports and graphs.</p>
        <h2>Calculated Properties</h2>
        <table border="1" cellpadding="5">
          <tr><th>Property</th><th>Description</th></tr>
          <tr><td>Amino Acid Composition</td><td>Counts and frequencies of the 20 standard amino acids.</td></tr>
          <tr><td>Molecular Weight</td><td>Mass of the protein in Daltons.</td></tr>
          <tr><td>Isoelectric Point (pI)</td><td>pH at which the protein carries no net charge.</td></tr>
          <tr><td>Molar Extinction Coefficient</td><td>Light absorption at 280 nm.</td></tr>
          <tr><td>GRAVY Score</td><td>Average hydrophobicity (higher = more hydrophobic).</td></tr>
          <tr><td>Instability Index</td><td>Estimation of in vitro stability (&lt;40 stable).</td></tr>
          <tr><td>Aromaticity</td><td>Frequency of aromatic residues.</td></tr>
          <tr><td>Secondary Structure</td><td>Predicted percentages of helices, turns, and sheets.</td></tr>
          <tr><td>Net Charge</td><td>Calculated at pH 7.0 and at the specified pH.</td></tr>
          <tr><td>Hydrophobicity Profile</td><td>Sliding-window analysis using the Kyte–Doolittle scale.</td></tr>
        </table>
        <h2>Graphical Outputs</h2>
        <ul>
          <li>Amino Acid Composition (Bar Chart)</li>
          <li>Hydrophobicity Profile (Line Chart)</li>
          <li>Net Charge vs pH</li>
          <li>Bead Models (by Hydrophobicity and by Charge)</li>
        </ul>
        <h2>Batch Analysis</h2>
        <p>Import a multi-FASTA file in the Batch Analysis tab to analyze multiple sequences. Double-click a row for details.</p>
        <h2>Settings</h2>
        <p>Customize parameters including theme, sliding window size, graph font sizes, colormap, and bead labels. Click "Apply Settings" to update reports and graphs dynamically.</p>
        <h2>Export Options</h2>
        <p>You can save the report as text, export to PDF (which includes graphs), or view a print preview.</p>
        <h2>Annotation</h2>
        <p>The Annotation section provides external annotation information (this is currently a placeholder).</p>
        <h2>Author</h2>
        <p>Developed by: <b>Saumyak Mukherjee</b><br>
        Contact: <a href="mailto:saumyak.mukherjee@biophys.mpg.de">saumyak.mukherjee@biophys.mpg.de</a></p>
        """)
        layout.addWidget(help_browser)
    
    # ----- Helper Methods for Analysis -----
    def import_fasta(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open FASTA File", "", "FASTA Files (*.fa *.fasta *.txt)")
        if file_name:
            try:
                with open(file_name, "r") as handle:
                    records = list(SeqIO.parse(handle, "fasta"))
                    if not records:
                        raise Exception("No records found.")
                    self.seq_text.setPlainText(str(records[0].seq))
            except Exception as e:
                QMessageBox.critical(self, "Import Error", f"Error reading FASTA file:\n{e}")
    
    def on_analyze(self):
        seq = clean_sequence(self.seq_text.toPlainText())
        if not seq:
            QMessageBox.warning(self, "Input Error", "Please enter a protein sequence.")
            return
        if not is_valid_protein(seq):
            QMessageBox.warning(self, "Input Error", "Sequence contains invalid amino acids.")
            return
        try:
            pH_val = float(self.ph_input.text().strip())
        except ValueError:
            pH_val = 7.0
        try:
            window_size = int(self.window_size_input.text().strip())
        except ValueError:
            window_size = self.default_window_size
        self.show_bead_labels = self.label_checkbox.isChecked()
        self.label_font_size = int(self.label_font_input.text().strip())
        self.tick_font_size = int(self.tick_font_input.text().strip())
        self.colormap = self.colormap_combo.currentText()
        
        self.analysis_data = analyze_sequence(seq, pH_val, window_size)
        report_sections = self.analysis_data["report_sections"]
        for title, browser in self.report_section_tabs.items():
            if title in report_sections:
                browser.setHtml(report_sections[title])
        self.update_graph_tabs()
        self.statusBar.showMessage("Analysis complete.", 3000)
        QMessageBox.information(self, "Analysis", "Protein analysis complete!")
    
    def update_graph_tabs(self):
        if not self.analysis_data:
            return
        seq = self.analysis_data["seq"]
        fig_aa = create_amino_acid_composition_figure(
            self.analysis_data["aa_counts"], self.analysis_data["aa_freq"],
            label_font=self.label_font_size, tick_font=self.tick_font_size)
        fig_hydro = create_hydrophobicity_figure(
            self.analysis_data["hydro_profile"], self.analysis_data["window_size"],
            label_font=self.label_font_size, tick_font=self.tick_font_size)
        fig_net = create_net_charge_vs_pH_figure(
            seq, label_font=self.label_font_size, tick_font=self.tick_font_size)
        fig_bead_hydro = create_bead_model_hydrophobicity_figure(
            seq, self.show_bead_labels, label_font=self.label_font_size, tick_font=self.tick_font_size,
            cmap=self.colormap)
        fig_bead_charge = create_bead_model_charge_figure(
            seq, self.show_bead_labels, label_font=self.label_font_size, tick_font=self.tick_font_size)
        figs = {
            "Amino Acid Composition": fig_aa,
            "Hydrophobicity Profile": fig_hydro,
            "Net Charge vs pH": fig_net,
            "Bead Model (Hydrophobicity)": fig_bead_hydro,
            "Bead Model (Charge)": fig_bead_charge
        }
        for title, fig in figs.items():
            tab, vbox = self.graph_tabs[title]
            for i in reversed(range(vbox.count())):
                widget = vbox.itemAt(i).widget()
                if widget is not None:
                    widget.setParent(None)
            canvas = FigureCanvas(fig)
            toolbar = NavigationToolbar2QT(canvas, self)
            vbox.addWidget(toolbar)
            vbox.addWidget(canvas)
            btn = QPushButton("Save Graph")
            btn.clicked.connect(lambda _, t=title: self.save_graph(t))
            vbox.addWidget(btn, alignment=Qt.AlignRight)
    
    def save_report(self):
        if not self.analysis_data:
            QMessageBox.warning(self, "No Report", "No report available.")
            return
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Report", "", "Text Files (*.txt)")
        if file_name:
            try:
                with open(file_name, "w") as f:
                    for section, content in self.analysis_data["report_sections"].items():
                        text = content.replace("<br>", "\n").replace("<h2>", "").replace("</h2>", "\n")
                        text = text.replace("<table border=\"1\" cellpadding=\"5\">", "").replace("</table>", "\n")
                        text = text.replace("<tr>", "").replace("</tr>", "\n")
                        text = text.replace("<th>", "").replace("</th>", "\t")
                        text = text.replace("<td>", "").replace("</td>", "\t")
                        f.write(f"==== {section} ====\n{text}\n\n")
                QMessageBox.information(self, "Success", f"Report saved to {file_name}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Could not save report: {e}")
    
    def export_pdf(self):
        if not self.analysis_data:
            QMessageBox.warning(self, "No Report", "No report to export.")
            return
        printer = QPrinter(QPrinter.HighResolution)
        file_name, _ = QFileDialog.getSaveFileName(self, "Export PDF", "", "PDF Files (*.pdf)")
        if file_name:
            printer.setOutputFormat(QPrinter.PdfFormat)
            printer.setOutputFileName(file_name)
            html_content = ""
            for section in ["Overview", "Amino Acid Composition", "Biochemical Properties",
                            "Secondary Structure", "Net Charge", "Hydrophobicity Profile", "Annotation"]:
                html_content += self.analysis_data["report_sections"][section]
            for title, (tab, vbox) in self.graph_tabs.items():
                for i in range(vbox.count()):
                    widget = vbox.itemAt(i).widget()
                    if isinstance(widget, FigureCanvas):
                        buf = BytesIO()
                        widget.figure.savefig(buf, format="png")
                        buf.seek(0)
                        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
                        html_content += f"<h2>{title}</h2><img src='data:image/png;base64,{img_base64}'><br>"
                        break
            temp_browser = QTextBrowser()
            temp_browser.setHtml(html_content)
            temp_browser.document().print_(printer)
            QMessageBox.information(self, "Success", f"PDF exported to {file_name}")
    
    def print_preview(self):
        if not self.analysis_data:
            QMessageBox.warning(self, "No Report", "No report for print preview.")
            return
        printer = QPrinter(QPrinter.HighResolution)
        preview = QPrintPreviewDialog(printer, self)
        html_content = ""
        for section in ["Overview", "Amino Acid Composition", "Biochemical Properties",
                        "Secondary Structure", "Net Charge", "Hydrophobicity Profile", "Annotation"]:
            html_content += self.analysis_data["report_sections"][section]
        for title, (tab, vbox) in self.graph_tabs.items():
            for i in range(vbox.count()):
                widget = vbox.itemAt(i).widget()
                if isinstance(widget, FigureCanvas):
                    buf = BytesIO()
                    widget.figure.savefig(buf, format="png")
                    buf.seek(0)
                    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
                    html_content += f"<h2>{title}</h2><img src='data:image/png;base64,{img_base64}'><br>"
                    break
        temp_browser = QTextBrowser()
        temp_browser.setHtml(html_content)
        preview.paintRequested.connect(lambda p: temp_browser.document().print_(p))
        preview.exec_()
    
    def save_graph(self, tab_title: str):
        tab, vbox = self.graph_tabs[tab_title]
        canvas = None
        for i in range(vbox.count()):
            widget = vbox.itemAt(i).widget()
            if isinstance(widget, FigureCanvas):
                canvas = widget
                break
        if canvas is None:
            QMessageBox.warning(self, "Save Error", "No graph found.")
            return
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Graph", "", "PNG Files (*.png);;JPEG Files (*.jpg)")
        if file_name:
            try:
                canvas.figure.savefig(file_name)
                QMessageBox.information(self, "Success", f"Graph saved to {file_name}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Could not save graph: {e}")
    
    def import_batch(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Import Multi-FASTA File", "", "FASTA Files (*.fa *.fasta *.txt)")
        if file_name:
            try:
                records = list(SeqIO.parse(file_name, "fasta"))
                self.batch_table.setRowCount(0)
                for record in records:
                    seq = clean_sequence(str(record.seq))
                    if not is_valid_protein(seq):
                        continue
                    data = analyze_sequence(seq, 7.0, self.default_window_size)
                    row = self.batch_table.rowCount()
                    self.batch_table.insertRow(row)
                    self.batch_table.setItem(row, 0, QTableWidgetItem(record.id))
                    self.batch_table.setItem(row, 1, QTableWidgetItem(str(len(seq))))
                    self.batch_table.setItem(row, 2, QTableWidgetItem(f"{data['mol_weight']:.2f}"))
                    self.batch_table.setItem(row, 3, QTableWidgetItem(f"{data['iso_point']:.2f}"))
                    self.batch_table.setItem(row, 4, QTableWidgetItem(f"{data['net_charge_7']:.2f}"))
                QMessageBox.information(self, "Batch Import", "Batch analysis loaded.")
            except Exception as e:
                QMessageBox.critical(self, "Batch Import Error", f"Error: {e}")
    
    def show_batch_details(self, row, column):
        seq_id = self.batch_table.item(row, 0).text()
        msg = f"Details for sequence: {seq_id}\n\n"
        msg += f"Length: {self.batch_table.item(row, 1).text()} amino acids\n"
        msg += f"Molecular Weight: {self.batch_table.item(row, 2).text()} Da\n"
        msg += f"Isoelectric Point: {self.batch_table.item(row, 3).text()}\n"
        msg += f"Net Charge (pH 7): {self.batch_table.item(row, 4).text()}\n"
        QMessageBox.information(self, "Batch Analysis Detail", msg)
    
    def toggle_theme(self):
        if self.theme_toggle.isChecked():
            self.current_theme = "Dark"
            self.setStyleSheet("""
            QWidget { background-color: #2b2b2b; color: #f0f0f0; font-family: Arial; font-size: 12px; }
            QLineEdit, QTextEdit, QTextBrowser { background-color: #3c3c3c; color: #f0f0f0; }
            QPushButton { background-color: #007acc; color: #f0f0f0; border: none; padding: 5px; }
            QPushButton:hover { background-color: #005f99; }
            QTabWidget::pane { border: 1px solid #444; }
            QTabBar::tab { background: #3c3c3c; padding: 10px; }
            QTabBar::tab:selected { background: #007acc; }
            QTableWidget { background-color: #3c3c3c; gridline-color: #555; }
            QHeaderView::section { background-color: #3c3c3c; }
            """)
            plt.style.use("dark_background")
        else:
            self.current_theme = "Light"
            self.setStyleSheet("""
            QWidget { background-color: #f0f0f0; color: #333; font-family: Arial; font-size: 12px; }
            QLineEdit, QTextEdit, QTextBrowser { background-color: #ffffff; color: #333; }
            QPushButton { background-color: #4CAF50; color: #fff; border: none; padding: 5px; }
            QPushButton:hover { background-color: #45a049; }
            QTabWidget::pane { border: 1px solid #ccc; }
            QTabBar::tab { background: #ffffff; padding: 10px; }
            QTabBar::tab:selected { background: #4CAF50; }
            QTableWidget { background-color: #ffffff; gridline-color: #ccc; }
            QHeaderView::section { background-color: #ffffff; }
            """)
            plt.style.use("default")
        self.statusBar.showMessage(f"{self.current_theme} theme activated", 3000)
    
    def apply_settings(self):
        try:
            window_size = int(self.window_size_input.text().strip())
        except ValueError:
            window_size = self.default_window_size
        self.show_bead_labels = self.label_checkbox.isChecked()
        self.label_font_size = int(self.label_font_input.text().strip())
        self.tick_font_size = int(self.tick_font_input.text().strip())
        self.colormap = self.colormap_combo.currentText()
        # Re-run analysis if a sequence exists so that graphs and report update with settings.
        seq_text = self.seq_text.toPlainText()
        seq = clean_sequence(seq_text)
        if seq and is_valid_protein(seq):
            try:
                pH_val = float(self.ph_input.text().strip())
            except ValueError:
                pH_val = 7.0
            self.analysis_data = analyze_sequence(seq, pH_val, window_size)
            report_sections = self.analysis_data["report_sections"]
            for title, browser in self.report_section_tabs.items():
                if title in report_sections:
                    browser.setHtml(report_sections[title])
            self.update_graph_tabs()
            self.statusBar.showMessage("Settings applied and reports refreshed.", 3000)
        else:
            self.statusBar.showMessage("Settings applied.", 3000)

def main():
    app = QApplication(sys.argv)
    window = ProteinAnalyzerGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
