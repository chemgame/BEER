#!/usr/bin/env python3
"""
Extended Protein Analyzer GUI Application (Refactored):
  - Removed UniProt annotation integration
  - Centralized REPORT_SECTIONS and GRAPH_TITLES constants
  - Eliminated redundant imports and code
  - Batch analysis on multi-FASTA / multichain PDB imports
  - Chain selection dialog for detailed analysis
  - Warning notes on Secondary Structure & Disorder tabs

Author: Saumyak Mukherjee (refactored)
Contact: saumyak.mukherjee@biophys.mpg.de

Requirements:
  pip install biopython matplotlib PyQt5 mplcursors
"""

import sys, math, os, base64, json, csv
from io import BytesIO

# PyQt5 Imports
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QDialog,
                             QLabel, QLineEdit, QPushButton, QTextEdit, QTextBrowser,
                             QFileDialog, QTabWidget, QMessageBox, QTableWidget, QTableWidgetItem,
                             QCheckBox, QStatusBar, QComboBox, QProgressBar, QInputDialog,
                             QFormLayout)

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtPrintSupport import QPrintPreviewDialog, QPrinter

# Matplotlib and mplcursors for interactive charts
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
import matplotlib.pyplot as plt
plt.style.use("default")
import mplcursors

# Biopython
from Bio.SeqUtils.ProtParam import ProteinAnalysis as BPProteinAnalysis
from Bio import SeqIO
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1

# ---------------------------
# Constants and Helper Functions
# ---------------------------
VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
KYTE_DOOLITTLE = {
    'A': 1.8,  'R': -4.5, 'N': -3.5, 'D': -3.5,
    'C': 2.5,  'Q': -3.5, 'E': -3.5, 'G': -0.4,
    'H': -3.2, 'I': 4.5,  'L': 3.8,  'K': -3.9,
    'M': 1.9,  'F': 2.8,  'P': -1.6, 'S': -0.8,
    'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

REPORT_SECTIONS = [
    "Overview",
    "Amino Acid Composition",
    "Biochemical Properties",
    "Secondary Structure",
    "Net Charge",
    "Hydrophobicity Profile",
    "Solubility",
    "Disorder"
]

GRAPH_TITLES = [
    "Amino Acid Composition (Bar)",
    "Amino Acid Composition (Pie)",
    "Hydrophobicity Profile",
    "Net Charge vs pH",
    "Bead Model (Hydrophobicity)",
    "Bead Model (Charge)",
    "Properties Radar Chart"
]

LIGHT_THEME_CSS = """
QWidget { background-color: #f0f0f0; color: #333; font-family: Arial; font-size: 12px; }
QLineEdit, QTextEdit, QTextBrowser { background-color: #ffffff; color: #333; }
QPushButton { background-color: #4CAF50; color: #fff; border: none; padding: 5px; }
QPushButton:hover { background-color: #45a049; }
QTabWidget::pane { border: 1px solid #ccc; }
QTabBar::tab { background: #ffffff; padding: 10px; }
QTabBar::tab:selected { background: #4CAF50; }
QTableWidget { background-color: #ffffff; gridline-color: #ccc; }
QHeaderView::section { background-color: #ffffff; }
QProgressBar { background-color: #cccccc; color: #333; border: none; text-align: center; }
QProgressBar::chunk { background-color: #4CAF50; }

"""

DARK_THEME_CSS = """
QWidget { background-color: #2b2b2b; color: #f0f0f0; font-family: Arial; font-size: 12px; }
QLineEdit, QTextEdit, QTextBrowser { background-color: #3c3c3c; color: #f0f0f0; }
QPushButton { background-color: #007acc; color: #f0f0f0; border: none; padding: 5px; }
QPushButton:hover { background-color: #005f99; }
QTabWidget::pane { border: 1px solid #444; }
QTabBar::tab { background: #3c3c3c; padding: 10px; }
QTabBar::tab:selected { background: #007acc; }
QTableWidget { background-color: #3c3c3c; gridline-color: #555; }
QHeaderView::section { background-color: #3c3c3c; }
QProgressBar { background-color: #555555; color: #f0f0f0; border: none; text-align: center; }
QProgressBar::chunk { background-color: #007acc; }

"""

def clean_sequence(seq: str) -> str:
    return seq.strip().replace(" ", "").upper()

def is_valid_protein(seq: str) -> bool:
    return all(aa in VALID_AMINO_ACIDS for aa in seq)

def calc_net_charge(seq: str, pH: float = 7.0) -> float:
    pKa_nterm = 9.69
    pKa_cterm = 2.34
    pKa_side = {'D': 3.90, 'E': 4.07, 'C': 8.18, 'Y': 10.46,
                'H': 6.04, 'K': 10.54, 'R': 12.48}
    net_charge = 1.0/(1.0 + 10**(pH - pKa_nterm)) - 1.0/(1.0 + 10**(pKa_cterm - pH))
    for aa in seq:
        if aa in ('D','E','C','Y'):
            net_charge += -1.0/(1.0 + 10**(pKa_side[aa] - pH))
        elif aa in ('K','R','H'):
            net_charge += 1.0/(1.0 + 10**(pH - pKa_side[aa]))
    return net_charge

def sliding_window_hydrophobicity(seq: str, window_size: int = 9) -> list:
    if window_size > len(seq):
        return [sum(KYTE_DOOLITTLE[aa] for aa in seq)/len(seq)]
    return [
        sum(KYTE_DOOLITTLE[aa] for aa in seq[i:i+window_size]) / window_size
        for i in range(len(seq) - window_size + 1)
    ]


# ---------------------------
# Analysis Tools Class
# ---------------------------
class AnalysisTools:
    @staticmethod
    def analyze_sequence(seq: str, pH_value: float = 7.0, window_size: int = 9) -> dict:
        pa = BPProteinAnalysis(seq)
        aa_counts = pa.count_amino_acids()
        seq_length = len(seq)
        aa_freq = {aa: (aa_counts[aa]/seq_length*100) for aa in aa_counts}
        mol_weight = pa.molecular_weight()
        iso_point = pa.isoelectric_point()
        extinction_coeffs = pa.molar_extinction_coefficient()
        gravy = pa.gravy()
        instability = pa.instability_index()
        aromaticity = pa.aromaticity()
        sec_struct = pa.secondary_structure_fraction()
        net_charge_7 = calc_net_charge(seq, 7.0)
        net_charge_custom = calc_net_charge(seq, pH_value)
        hydro_profile = sliding_window_hydrophobicity(seq, window_size)
        hydro_avg = sum(hydro_profile)/len(hydro_profile) if hydro_profile else 0.0
        hydro_min = min(hydro_profile) if hydro_profile else 0.0
        hydro_max = max(hydro_profile) if hydro_profile else 0.0

        solubility = AnalysisTools.predict_solubility(seq)
        disorder = AnalysisTools.predict_disorder(seq)

        # Build HTML sections with warning notes
        overview_html = f"""
        <h2>Overview</h2>
        <table border="1" cellpadding="5">
          <tr><th>Property</th><th>Value</th></tr>
          <tr><td>Sequence Length</td><td>{seq_length} aa</td></tr>
          <tr><td>Sequence</td><td>{seq}</td></tr>
        </table>
        """
        # Default sort: by composition value (frequency descending)
        sorted_aas = sorted(aa_counts, key=lambda aa: aa_freq[aa], reverse=True)
        comp_html = "<h2>Amino Acid Composition</h2><table border=\"1\" cellpadding=\"5\"><tr><th>Amino Acid</th><th>Count</th><th>Frequency (%)</th></tr>" \
                    + "".join(
                        f"<tr><td>{aa}</td>"
                        f"<td>{aa_counts[aa]}</td>"
                        f"<td>{aa_freq[aa]:.2f}%</td></tr>"
                      for aa in sorted_aas
                    ) \
                    + "</table>"
        bio_html = f"""
        <h2>Biochemical Properties</h2>
        <table border="1" cellpadding="5">
          <tr><td>Mol. Weight</td><td>{mol_weight:.2f} Da</td></tr>
          <tr><td>Isoelectric Point</td><td>{iso_point:.2f}</td></tr>
          <tr><td>Extinction Coeff.</td><td>{extinction_coeffs} M⁻¹cm⁻¹</td></tr>
          <tr><td>GRAVY</td><td>{gravy:.3f}</td></tr>
          <tr><td>Instability</td><td>{instability:.2f}</td></tr>
          <tr><td>Aromaticity</td><td>{aromaticity:.3f}</td></tr>
        </table>
        """
        sec_html = f"""
        <h2>Secondary Structure</h2>
        <p><b>Warning:</b> This is just a prediction. Do not trust blindly.</p>
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
          <tr><td>Window</td><td>{window_size}</td></tr>
          <tr><td>Avg</td><td>{hydro_avg:.3f}</td></tr>
          <tr><td>Min</td><td>{hydro_min:.3f}</td></tr>
          <tr><td>Max</td><td>{hydro_max:.3f}</td></tr>
        </table>
        """
        sol_html = f"""
        <h2>Solubility Prediction</h2>
        <p>{solubility}</p>
        """
        dis_html = f"""
        <h2>Disorder Prediction</h2>
        <p><b>Warning:</b> This is just a prediction. Do not trust blindly.</p>
        <p>{disorder}</p>
        """

        report_sections = {
            "Overview": overview_html,
            "Amino Acid Composition": comp_html,
            "Biochemical Properties": bio_html,
            "Secondary Structure": sec_html,
            "Net Charge": net_html,
            "Hydrophobicity Profile": hydro_html,
            "Solubility": sol_html,
            "Disorder": dis_html
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
            "extinction_coeffs": extinction_coeffs,
            "gravy": gravy,
            "instability": instability,
            "aromaticity": aromaticity
        }

    @staticmethod
    def predict_solubility(seq: str) -> str:
        avg_hydro = sum(KYTE_DOOLITTLE[aa] for aa in seq)/len(seq)
        return "Likely soluble" if avg_hydro < 0.5 else "Low solubility predicted"

    @staticmethod
    def predict_disorder(seq: str) -> str:
        disorder_pct = (seq.count('P') + seq.count('G')) / len(seq) * 100
        return f"Approximately {disorder_pct:.1f}% disordered regions"


# ---------------------------
# Graphing Tools Class
# ---------------------------
class GraphingTools:
    @staticmethod
    def create_amino_acid_composition_figure(aa_counts, aa_freq, label_font=14, tick_font=12):
        fig = Figure(figsize=(5,4))
        ax = fig.add_subplot(111)
        aas = sorted(aa_counts.keys())
        cnts = [aa_counts[a] for a in aas]
        ax.bar(aas, cnts)
        ax.set_xlabel("Amino Acids", fontsize=label_font)
        ax.set_ylabel("Counts", fontsize=label_font)
        ax.set_title("AA Composition (Bar)", fontsize=label_font+2)
        ax.tick_params(labelsize=tick_font)
        for i,a in enumerate(aas):
            ax.text(i, cnts[i]+0.5, f"{aa_freq[a]:.1f}%", ha="center", fontsize=10)
        mplcursors.cursor(ax)
        return fig

    @staticmethod
    def create_amino_acid_composition_pie_figure(aa_counts, label_font=14):
        fig = Figure(figsize=(5,4))
        ax = fig.add_subplot(111)
        ax.pie(list(aa_counts.values()), labels=list(aa_counts.keys()), autopct="%1.1f%%", startangle=140)
        ax.set_title("AA Composition (Pie)", fontsize=label_font+2)
        mplcursors.cursor(ax)
        return fig

    @staticmethod
    def create_hydrophobicity_figure(hydro_profile, window_size, label_font=14, tick_font=12):
        fig = Figure(figsize=(5,4))
        ax = fig.add_subplot(111)
        ax.plot(range(1, len(hydro_profile)+1), hydro_profile, marker="o")
        ax.set_xlabel("Window Start", fontsize=label_font)
        ax.set_ylabel("Hydrophobicity", fontsize=label_font)
        ax.set_title(f"Hydrophobicity (w={window_size})", fontsize=label_font+2)
        ax.tick_params(labelsize=tick_font)
        ax.grid(True)
        mplcursors.cursor(ax)
        return fig

    @staticmethod
    def create_net_charge_vs_pH_figure(seq, label_font=14, tick_font=12):
        fig = Figure(figsize=(5,4))
        ax = fig.add_subplot(111)
        phs = [i/10 for i in range(0,141)]
        nets = [calc_net_charge(seq,p) for p in phs]
        ax.plot(phs, nets)
        ax.set_xlabel("pH", fontsize=label_font)
        ax.set_ylabel("Net Charge", fontsize=label_font)
        ax.set_title("Net Charge vs pH", fontsize=label_font+2)
        ax.tick_params(labelsize=tick_font)
        ax.grid(True)
        mplcursors.cursor(ax)
        return fig

    @staticmethod
    def create_bead_model_hydrophobicity_figure(seq, show_labels, label_font=14, tick_font=12, cmap="coolwarm"):
        fig = Figure(figsize=(min(12,0.25*len(seq)),2))
        ax = fig.add_subplot(111)
        xs = list(range(1,len(seq)+1))
        vals = [KYTE_DOOLITTLE[aa] for aa in seq]
        sc = ax.scatter(xs, [1]*len(seq), c=vals, cmap=cmap, s=200)
        fig.colorbar(sc, ax=ax, label="Hydrophobicity")
        ax.set_yticks([])
        ax.set_xlabel("Residue", fontsize=label_font)
        ax.set_title("Bead Hydrophobicity", fontsize=label_font+2)
        if show_labels and len(seq)<=50:
            for i,aa in enumerate(seq):
                ax.text(xs[i],1,aa,ha="center",va="center",fontsize=label_font-2)
        mplcursors.cursor(ax)
        return fig

    @staticmethod
    def create_bead_model_charge_figure(seq, show_labels, label_font=14, tick_font=12):
        fig = Figure(figsize=(min(12,0.25*len(seq)),2))
        ax = fig.add_subplot(111)
        xs = list(range(1,len(seq)+1))
        cols = ["blue" if aa in "KRH" else "red" if aa in "DE" else "gray" for aa in seq]
        ax.scatter(xs, [1]*len(seq), c=cols, s=200)
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color="blue", label="Pos"),
            Patch(color="red", label="Neg"),
            Patch(color="gray", label="Neu")
        ], loc="upper right")
        ax.set_yticks([])
        ax.set_xlabel("Residue", fontsize=label_font)
        ax.set_title("Bead Charge", fontsize=label_font+2)
        if show_labels and len(seq)<=50:
            for i,aa in enumerate(seq):
                ax.text(xs[i],1,aa,ha="center",va="center",color="white",fontsize=label_font-2)
        mplcursors.cursor(ax)
        return fig

    @staticmethod
    def create_radar_chart_figure(data, label_font=14):
        props = ["Mol Weight","pI","GRAVY","Instability","Aromaticity"]
        vals = [data["mol_weight"],data["iso_point"],data["gravy"],data["instability"],data["aromaticity"]]
        ranges = {
            "Mol Weight":(5000,150000),"pI":(4,11),"GRAVY":(-2.5,2.5),
            "Instability":(20,80),"Aromaticity":(0,0.2)
        }
        norm=[] 
        for p,v in zip(props,vals):
            mn,mx = ranges[p]
            x = max(0,min(1,(v-mn)/(mx-mn)))
            norm.append(x)
        norm+=norm[:1]
        angles = [n/len(props)*2*math.pi for n in range(len(props))]+[0]
        fig = Figure(figsize=(5,4))
        ax = fig.add_subplot(111,polar=True)
        ax.plot(angles,norm)
        ax.fill(angles,norm,alpha=0.4)
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(props,fontsize=label_font-2)
        ax.set_title("Radar Chart",fontsize=label_font+2)
        mplcursors.cursor(ax)
        return fig


# ---------------------------
# Export Tools Class
# ---------------------------
class ExportTools:
    @staticmethod
    def _generate_full_html(analysis_data, graph_tabs):
        html = ""
        for sec in REPORT_SECTIONS:
            html += analysis_data["report_sections"][sec]
        for title,(tab,vbox) in graph_tabs.items():
            for i in range(vbox.count()):
                w = vbox.itemAt(i).widget()
                if getattr(w, "figure", None):
                    buf = BytesIO()
                    w.figure.savefig(buf, format="png")
                    buf.seek(0)
                    b64 = base64.b64encode(buf.read()).decode()
                    html += f"<h2>{title}</h2><img src='data:image/png;base64,{b64}'><br>"
                    break
        return html

    @staticmethod
    def export_report_text(analysis_data, file_name):
        try:
            with open(file_name, "w") as f:
                for sec, content in analysis_data["report_sections"].items():
                    text = content.replace("<br>", "\n")
                    text = text.replace("<h2>", "").replace("</h2>", "\n")
                    text = text.replace("<table", "").replace("</table>", "\n")
                    text = text.replace("<tr>", "").replace("</tr>", "\n")
                    text = text.replace("<th>", "").replace("</th>", "\t")
                    text = text.replace("<td>", "").replace("</td>", "\t")
                    f.write(f"==== {sec} ====\n{text}\n\n")
            return True, f"Report saved to {file_name}"
        except Exception as e:
            return False, f"Save error: {e}"

    @staticmethod
    def export_pdf(analysis_data, graph_tabs, file_name, parent):
        printer = QPrinter(QPrinter.HighResolution)
        printer.setOutputFormat(QPrinter.PdfFormat)
        printer.setOutputFileName(file_name)
        browser = QTextBrowser()
        browser.setHtml(ExportTools._generate_full_html(analysis_data, graph_tabs))
        browser.document().print_(printer)
        QMessageBox.information(parent, "Success", f"PDF exported to {file_name}")


# ---------------------------
# PDB Import Function
# ---------------------------
def import_pdb_sequence(file_name: str) -> dict:
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("pdb", file_name)
    model = next(struct.get_models())
    chains = {}
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


# ---------------------------
# Main GUI Application Class
# ---------------------------
class ProteinAnalyzerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PROBE - PROtein analyzer and Bioinformatics Estimator")
        self.resize(1200,900)
        #self.statusBar = QStatusBar(); self.setStatusBar(self.statusBar)
        #self.progress_bar = QProgressBar(); self.statusBar.addPermanentWidget(self.progress_bar)
        # ─── single status bar + progress bar ───
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)  # hidden until we show it
        self.statusBar.addPermanentWidget(self.progress_bar)
        self.analysis_data = None
        self.batch_data = []
        self.default_window_size = 9
        self.default_pH = 7.0
        self.show_bead_labels = True
        self.current_theme = "Light"
        self.label_font_size = 14
        self.tick_font_size = 12
        self.colormap = "coolwarm"
        self.marker_size = 10
        self.graph_color = "blue"
        self.show_heading = True
        self.show_grid = True
        self.main_tabs = QTabWidget(); self.setCentralWidget(self.main_tabs)
        self.init_analysis_tab()
        self.init_graphs_tab()
        self.init_batch_tab()
        self.init_settings_tab()
        self.init_help_tab()

    def reset_defaults(self):
        """Restore all user‑editable settings to factory defaults."""
        # text / numeric
        self.window_size_input.setText("9")
        self.ph_input.setText("7.0")
        self.label_font_input.setText("14")
        self.tick_font_input.setText("12")
        self.marker_size_input.setText("10")
    
        # checkboxes & combos
        self.label_checkbox.setChecked(True)
        self.heading_checkbox.setChecked(True)
        self.grid_checkbox.setChecked(True)
        self.colormap_combo.setCurrentText("coolwarm")
        self.graph_color_combo.setCurrentText("blue")
    
        # theme toggle (optional—leave as‑is)
    
        # commit to internal vars and refresh displays
        self.apply_settings()


    def init_analysis_tab(self):
        layout = QVBoxLayout()
        container = QWidget()
        container.setLayout(layout)
        self.main_tabs.addTab(container, "Analysis")

        # Top row: Import and Analyze buttons
        row = QHBoxLayout()
        self.import_fasta_btn = QPushButton("Import FASTA")
        self.import_fasta_btn.clicked.connect(self.import_fasta)
        self.import_pdb_btn = QPushButton("Import PDB")
        self.import_pdb_btn.clicked.connect(self.import_pdb)
        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.clicked.connect(self.on_analyze)
        for w in (self.import_fasta_btn, self.import_pdb_btn, self.analyze_btn):
            row.addWidget(w)
        layout.addLayout(row)

        # Export buttons
        exp_row = QHBoxLayout()
        self.save_txt_btn = QPushButton("Save Report")
        self.save_txt_btn.clicked.connect(self.save_report)
        self.save_pdf_btn = QPushButton("Export PDF")
        self.save_pdf_btn.clicked.connect(self.export_pdf)
        exp_row.addWidget(self.save_txt_btn)
        exp_row.addWidget(self.save_pdf_btn)
        layout.addLayout(exp_row)

        # Sequence entry
        layout.addWidget(QLabel("Protein Sequence:"))
        self.seq_text = QTextEdit()
        layout.addWidget(self.seq_text)

        # ─── Sorting buttons for Amino‑Acid Composition ───
        #sort_layout = QHBoxLayout()
        #self.sort_alpha_btn = QPushButton("Sort A→Z")
        #self.sort_alpha_btn.clicked.connect(lambda: self.sort_composition("alpha"))
        #self.sort_comp_btn = QPushButton("Sort by Composition")
        #self.sort_comp_btn.clicked.connect(lambda: self.sort_composition("composition"))
        #self.sort_hydro_inc_btn = QPushButton("Sort Hydro ↑")
        #self.sort_hydro_inc_btn.clicked.connect(lambda: self.sort_composition("hydro_inc"))
        #self.sort_hydro_dec_btn = QPushButton("Sort Hydro ↓")
        #self.sort_hydro_dec_btn.clicked.connect(lambda: self.sort_composition("hydro_dec"))
        #for btn in (self.sort_alpha_btn,
        #            self.sort_comp_btn,
        #            self.sort_hydro_inc_btn,
        #            self.sort_hydro_dec_btn):
        #    sort_layout.addWidget(btn)
        #layout.addLayout(sort_layout)

        self.report_tabs = QTabWidget()
        layout.addWidget(self.report_tabs)
        self.report_section_tabs = {}
        
        for sec in REPORT_SECTIONS:
            tab = QWidget()
            vb = QVBoxLayout(tab)
        
            # ─── Only for the AA Composition tab, add sort buttons ───
            if sec == "Amino Acid Composition":
                sort_layout = QHBoxLayout()
                self.sort_alpha_btn = QPushButton("Sort A→Z")
                self.sort_alpha_btn.clicked.connect(lambda: self.sort_composition("alpha"))
                self.sort_comp_btn = QPushButton("Sort by Composition")
                self.sort_comp_btn.clicked.connect(lambda: self.sort_composition("composition"))
                self.sort_hydro_inc_btn = QPushButton("Sort Hydro ↑")
                self.sort_hydro_inc_btn.clicked.connect(lambda: self.sort_composition("hydro_inc"))
                self.sort_hydro_dec_btn = QPushButton("Sort Hydro ↓")
                self.sort_hydro_dec_btn.clicked.connect(lambda: self.sort_composition("hydro_dec"))
                for btn in (
                    self.sort_alpha_btn,
                    self.sort_comp_btn,
                    self.sort_hydro_inc_btn,
                    self.sort_hydro_dec_btn
                ):
                    sort_layout.addWidget(btn)
                vb.addLayout(sort_layout)
        
            # the existing report browser
            browser = QTextBrowser()
            vb.addWidget(browser)
        
            self.report_tabs.addTab(tab, sec)
            self.report_section_tabs[sec] = browser


    def init_graphs_tab(self):
        self.graphs_subtabs = QTabWidget()
        layout = QVBoxLayout(); container=QWidget(); container.setLayout(layout)
        self.main_tabs.addTab(container, "Graphs"); layout.addWidget(self.graphs_subtabs)
        self.graph_tabs = {}
        for title in GRAPH_TITLES:
            tab=QWidget(); vb=QVBoxLayout(tab)
            placeholder=QLabel(f"{title} will appear here"); placeholder.setAlignment(Qt.AlignCenter)
            vb.addWidget(placeholder)
            btn=QPushButton("Save Graph"); btn.clicked.connect(lambda _,t=title: self.save_graph(t))
            vb.addWidget(btn, alignment=Qt.AlignRight)
            self.graphs_subtabs.addTab(tab, title)
            self.graph_tabs[title]=(tab,vb)
        save_all=QPushButton("Save All Graphs"); save_all.clicked.connect(self.save_all_graphs)
        layout.addWidget(save_all, alignment=Qt.AlignRight)

    def init_batch_tab(self):
        layout = QVBoxLayout()
        container = QWidget()
        container.setLayout(layout)
        self.main_tabs.addTab(container, "Batch Analysis")

        # Export buttons for batch data (removed the Batch Import button)
        top_layout = QHBoxLayout()
        self.batch_export_csv_btn = QPushButton("Export Batch CSV")
        self.batch_export_csv_btn.clicked.connect(self.export_batch_csv)
        self.batch_export_json_btn = QPushButton("Export Batch JSON")
        self.batch_export_json_btn.clicked.connect(self.export_batch_json)
        top_layout.addWidget(self.batch_export_csv_btn)
        top_layout.addWidget(self.batch_export_json_btn)
        layout.addLayout(top_layout)

        # Batch results table with updated columns
        self.batch_table = QTableWidget()
        self.batch_table.setColumnCount(9)
        self.batch_table.setHorizontalHeaderLabels([
            "ID",
            "Length",
            "MW (Da)",
            "Net Charge (pH 7)",
            "% Hydrophobic",
            "% Hydrophilic",
            "% +Charged",
            "% -Charged",
            "% Neutral"
        ])
        self.batch_table.cellDoubleClicked.connect(self.show_batch_details)
        layout.addWidget(self.batch_table)

    def init_settings_tab(self):
        layout = QVBoxLayout()
        container = QWidget()
        container.setLayout(layout)
        self.main_tabs.addTab(container, "Settings")

        # --- Theme toggle (always visible) ---
        th = QHBoxLayout()
        th.addWidget(QLabel("Theme:"))
        self.theme_toggle = QCheckBox("Dark Theme")
        self.theme_toggle.stateChanged.connect(self.toggle_theme)
        th.addWidget(self.theme_toggle)
        layout.addLayout(th)

        # --- Primary settings form ---
        form = QFormLayout()
        self.window_size_input = QLineEdit(str(self.default_window_size))
        form.addRow("Sliding Window Size:", self.window_size_input)

        self.ph_input = QLineEdit(str(self.default_pH))
        form.addRow("Default pH:", self.ph_input)

        self.label_checkbox = QCheckBox("Show bead labels")
        self.label_checkbox.setChecked(self.show_bead_labels)
        form.addRow("", self.label_checkbox)

        self.colormap_combo = QComboBox()
        # extended palette
        self.colormap_combo.addItems([
            "viridis", "plasma", "coolwarm", "inferno", "magma",
            "cividis", "Spectral", "hot", "copper", "cool", "autumn",
            "pink", "berlin", "vaniamo", "managua", "hsv"
        ])
        form.addRow("Colormap:", self.colormap_combo)

        layout.addLayout(form)

        # --- Advanced settings header ---
        adv_lbl = QLabel("Advanced Settings")
        adv_lbl.setStyleSheet("font-weight:bold; margin-top:12px;")
        layout.addWidget(adv_lbl)

        # --- Advanced settings form ---
        adv = QFormLayout()
        self.label_font_input = QLineEdit(str(self.label_font_size))
        adv.addRow("Label Font Size:", self.label_font_input)

        self.tick_font_input = QLineEdit(str(self.tick_font_size))
        adv.addRow("Tick Font Size:", self.tick_font_input)

        self.marker_size_input = QLineEdit("10")
        adv.addRow("Marker Size:", self.marker_size_input)

        self.graph_color_combo = QComboBox()
        self.graph_color_combo.addItems([
            "blue", "green", "red", "cyan", "magenta", "yellow", "black", "gray"
        ])
        adv.addRow("Graph Color:", self.graph_color_combo)

        self.heading_checkbox = QCheckBox("Show Graph Heading")
        self.heading_checkbox.setChecked(True)
        adv.addRow("", self.heading_checkbox)

        self.grid_checkbox = QCheckBox("Show Grid")
        self.grid_checkbox.setChecked(True)
        adv.addRow("", self.grid_checkbox)

        layout.addLayout(adv)

        # Apply and stretch
        btn = QPushButton("Apply Settings")
        btn.clicked.connect(self.apply_settings)
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self.reset_defaults)
        layout.addWidget(btn)
        layout.addWidget(reset_btn)

    def init_help_tab(self):
        layout=QVBoxLayout(); c=QWidget(); c.setLayout(layout)
        self.main_tabs.addTab(c,"Help")
        b = QTextBrowser()
        b.setHtml("""
        <h1>PROBE Help &amp; Definitions</h1>
        <h2>Protein Sequence</h2>
        <p>Single-letter amino acid codes (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y).</p>
        <h2>Amino Acid Composition</h2>
        <p>Counts and percentage frequencies of each residue.</p>
        <h2>Biochemical Properties</h2>
        <ul>
          <li><b>Molecular Weight:</b> Approximate mass in Daltons.</li>
          <li><b>Isoelectric Point (pI):</b> pH at which net charge = 0.</li>
          <li><b>Extinction Coefficient:</b> Absorbance at 280 nm per M per cm.</li>
          <li><b>GRAVY Score:</b> Grand average of hydropathicity (positive = hydrophobic).</li>
          <li><b>Instability Index:</b> &lt;40 suggests stable protein.</li>
          <li><b>Aromaticity:</b> Fraction of F, W, Y residues.</li>
        </ul>
        <h2>Secondary Structure</h2>
        <p><b>Warning:</b> Prediction only. Do not trust blindly.</p>
        <h2>Net Charge</h2>
        <p>Calculated at chosen pH using standard pKa values.</p>
        <h2>Hydrophobicity Profile</h2>
        <p>Sliding‐window average hydrophobicity (default window = 9).</p>
        <h2>Solubility Prediction</h2>
        <p>Based on average hydrophobicity: lower → more soluble.</p>
        <h2>Disorder Prediction</h2>
        <p><b>Warning:</b> Prediction only. Do not trust blindly.</p>
        <h2>Graphs</h2>
        <ul>
          <li><b>Bar Chart:</b> Residue counts.</li>
          <li><b>Pie Chart:</b> Composition percentages.</li>
          <li><b>Hydrophobicity:</b> Windowed profile.</li>
          <li><b>Net Charge vs pH:</b> Charge curve 0–14 pH.</li>
          <li><b>Bead Models:</b> Hydrophobicity/charge per residue.</li>
          <li><b>Radar Chart:</b> Normalized property comparison.</li>
        </ul>
        <h2>Batch Analysis</h2>
        <p>Import multi‐FASTA or PDB chains; select one for detailed view.</p>
        <h2>Settings</h2>
        <p>Adjust window size, fonts, colormap, and theme.</p>
        """)
        layout.addWidget(b)

    def import_fasta(self):
        # Open a multi‐FASTA and batch‐analyze each record
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open FASTA File", "", "FASTA Files (*.fa *.fasta *.txt)"
        )
        if not file_name:
            return

        records = list(SeqIO.parse(file_name, "fasta"))
        if not records:
            QMessageBox.warning(self, "No Records", "No sequences found in the file.")
            return

        # Clear previous batch data
        self.batch_data = []
        self.batch_table.setRowCount(0)

        # Analyze and populate table
        for record in records:
            seq = clean_sequence(str(record.seq))
            if not is_valid_protein(seq):
                continue
            data = AnalysisTools.analyze_sequence(seq, 7.0, self.default_window_size)

            # Compute extra percentages
            length = len(seq)
            mw = data["mol_weight"]
            net = data["net_charge_7"]
            # hydrophobic vs hydrophilic
            hypo_count = sum(1 for aa in seq if KYTE_DOOLITTLE[aa] > 0)
            pct_hydrophobic = hypo_count / length * 100
            pct_hydrophilic = 100 - pct_hydrophobic
            # charged vs neutral
            counts = data["aa_counts"]
            pos = counts.get("K", 0) + counts.get("R", 0) + counts.get("H", 0)
            neg = counts.get("D", 0) + counts.get("E", 0)
            pct_pos = pos / length * 100
            pct_neg = neg / length * 100
            pct_neutral = 100 - (pct_pos + pct_neg)

            rec_id = record.id
            self.batch_data.append((rec_id, seq, data))

            row = self.batch_table.rowCount()
            self.batch_table.insertRow(row)
            self.batch_table.setItem(row, 0, QTableWidgetItem(rec_id))
            self.batch_table.setItem(row, 1, QTableWidgetItem(str(length)))
            self.batch_table.setItem(row, 2, QTableWidgetItem(f"{mw:.2f}"))
            self.batch_table.setItem(row, 3, QTableWidgetItem(f"{net:.2f}"))
            self.batch_table.setItem(row, 4, QTableWidgetItem(f"{pct_hydrophobic:.1f}%"))
            self.batch_table.setItem(row, 5, QTableWidgetItem(f"{pct_hydrophilic:.1f}%"))
            self.batch_table.setItem(row, 6, QTableWidgetItem(f"{pct_pos:.1f}%"))
            self.batch_table.setItem(row, 7, QTableWidgetItem(f"{pct_neg:.1f}%"))
            self.batch_table.setItem(row, 8, QTableWidgetItem(f"{pct_neutral:.1f}%"))

        # Let the user pick one sequence for detailed view
        ids = [entry[0] for entry in self.batch_data]
        choice, ok = QInputDialog.getItem(
            self, "Select Sequence", "Sequence ID:", ids, 0, False
        )
        if ok:
            for rec_id, seq, data in self.batch_data:
                if rec_id == choice:
                    self.seq_text.setPlainText(seq)
                    self.analysis_data = data
                    self.update_analysis_tabs()
                    break

        records = list(SeqIO.parse(file_name, "fasta"))
        if not records:
            self.progress_bar.hide()
            QMessageBox.warning(self, "No Records", "No sequences found in the file.")
            return
        
        # Prepare progress bar for batch processing
        total = len(records)
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        
        # Analyze and populate table with progress updates
        for i, record in enumerate(records, start=1):
            seq = clean_sequence(str(record.seq))
            if not is_valid_protein(seq):
                continue
            data = AnalysisTools.analyze_sequence(seq, 7.0, self.default_window_size)
            # ... (populate self.batch_table with results) ...
            self.progress_bar.setValue(i)
            QApplication.processEvents()  # refresh UI to update the bar&#8203;:contentReference[oaicite:3]{index=3}
        
        self.progress_bar.hide()


    def import_pdb(self):
        # Open a multi‐chain PDB and batch‐analyze each chain
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open PDB File", "", "PDB Files (*.pdb *.ent)"
        )
        if not file_name:
            return

        chains = import_pdb_sequence(file_name)
        if not chains:
            QMessageBox.warning(self, "No Chains", "No valid chains found in the PDB.")
            return

        # Clear previous batch data
        self.batch_data = []
        self.batch_table.setRowCount(0)

        # Analyze and populate table
        for chain_id, seq in chains.items():
            data = AnalysisTools.analyze_sequence(seq, 7.0, self.default_window_size)

            # Compute extra percentages
            length = len(seq)
            mw = data["mol_weight"]
            net = data["net_charge_7"]
            hypo_count = sum(1 for aa in seq if KYTE_DOOLITTLE[aa] > 0)
            pct_hydrophobic = hypo_count / length * 100
            pct_hydrophilic = 100 - pct_hydrophobic
            counts = data["aa_counts"]
            pos = counts.get("K", 0) + counts.get("R", 0) + counts.get("H", 0)
            neg = counts.get("D", 0) + counts.get("E", 0)
            pct_pos = pos / length * 100
            pct_neg = neg / length * 100
            pct_neutral = 100 - (pct_pos + pct_neg)

            self.batch_data.append((chain_id, seq, data))

            row = self.batch_table.rowCount()
            self.batch_table.insertRow(row)
            self.batch_table.setItem(row, 0, QTableWidgetItem(chain_id))
            self.batch_table.setItem(row, 1, QTableWidgetItem(str(length)))
            self.batch_table.setItem(row, 2, QTableWidgetItem(f"{mw:.2f}"))
            self.batch_table.setItem(row, 3, QTableWidgetItem(f"{net:.2f}"))
            self.batch_table.setItem(row, 4, QTableWidgetItem(f"{pct_hydrophobic:.1f}%"))
            self.batch_table.setItem(row, 5, QTableWidgetItem(f"{pct_hydrophilic:.1f}%"))
            self.batch_table.setItem(row, 6, QTableWidgetItem(f"{pct_pos:.1f}%"))
            self.batch_table.setItem(row, 7, QTableWidgetItem(f"{pct_neg:.1f}%"))
            self.batch_table.setItem(row, 8, QTableWidgetItem(f"{pct_neutral:.1f}%"))

        # Let the user pick one chain for detailed view
        ids = [entry[0] for entry in self.batch_data]
        choice, ok = QInputDialog.getItem(
            self, "Select Chain", "Chain ID:", ids, 0, False
        )
        if ok:
            for chain_id, seq, data in self.batch_data:
                if chain_id == choice:
                    self.seq_text.setPlainText(seq)
                    self.analysis_data = data
                    self.update_analysis_tabs()
                    break
        chains = import_pdb_sequence(file_name)
        if not chains:
            self.progress_bar.hide()
            QMessageBox.warning(self, "No Chains", "No valid chains found in the PDB.")
            return
        
        # Prepare progress bar for batch processing
        total = len(chains)
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        
        # Analyze each chain with progress updates
        for count, (chain_id, seq) in enumerate(chains.items(), start=1):
            data = AnalysisTools.analyze_sequence(seq, 7.0, self.default_window_size)
            # ... (populate self.batch_table with results) ...
            self.progress_bar.setValue(count)
            QApplication.processEvents()  # update UI after processing each chain&#8203;:contentReference[oaicite:5]{index=5}
        
        self.progress_bar.hide()

    def sort_composition(self, mode):
        aa_counts = self.analysis_data["aa_counts"]
        aa_freq    = self.analysis_data["aa_freq"]
        # Build list of (aa, count)
        items = list(aa_counts.items())

        if mode == "alpha":
            items.sort(key=lambda x: x[0])
        elif mode == "composition":
            items.sort(key=lambda x: aa_freq[x[0]], reverse=True)
        elif mode == "hydro_inc":
            items.sort(key=lambda x: KYTE_DOOLITTLE[x[0]])
        elif mode == "hydro_dec":
            items.sort(key=lambda x: KYTE_DOOLITTLE[x[0]], reverse=True)

        # Rebuild HTML table
        html = "<h2>Amino Acid Composition</h2>" \
               "<table border=\"1\" cellpadding=\"5\">" \
               "<tr><th>Amino Acid</th><th>Count</th><th>Frequency (%)</th></tr>"
        for aa, count in items:
            html += f"<tr><td>{aa}</td><td>{count}</td><td>{aa_freq[aa]:.2f}%</td></tr>"
        html += "</table>"

        # Update the report tab
        self.report_section_tabs["Amino Acid Composition"].setHtml(html)

        # Update the bar‐chart tab
        tab, vbox = self.graph_tabs["Amino Acid Composition (Bar)"]
        # clear old
        for i in reversed(range(vbox.count())):
            w = vbox.itemAt(i).widget()
            if w: w.setParent(None)
        # rebuild
        fig = GraphingTools.create_amino_acid_composition_figure(
            dict(items),
            {aa: aa_freq[aa] for aa, _ in items},
            label_font=self.label_font_size,
            tick_font=self.tick_font_size
        )
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar2QT(canvas, self)
        vbox.addWidget(toolbar)
        vbox.addWidget(canvas)
        btn = QPushButton("Save Graph")
        btn.clicked.connect(lambda _, t="Amino Acid Composition (Bar)": self.save_graph(t))
        vbox.addWidget(btn, alignment=Qt.AlignRight)

    def on_analyze(self):
        # ── show & initialize progress bar ──
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        QTimer.singleShot(100, lambda: self.progress_bar.setValue(20))
        QTimer.singleShot(300, lambda: self.progress_bar.setValue(50))
        QTimer.singleShot(500, lambda: self.progress_bar.setValue(80))
        QTimer.singleShot(700, lambda: self.progress_bar.setValue(100))

        seq=clean_sequence(self.seq_text.toPlainText())
        if not seq: QMessageBox.warning(self,"Input","Enter sequence."); return
        if not is_valid_protein(seq): QMessageBox.warning(self,"Input","Invalid residues."); return
        try:
            pH=float(self.ph_input.text())
        except Exception:
            pH=7.0
        self.analysis_data=AnalysisTools.analyze_sequence(seq,pH,self.default_window_size)
        self.update_analysis_tabs()
        QMessageBox.information(self,"Done","Analysis complete.")
        # hide it when done
        self.progress_bar.hide()

    def update_analysis_tabs(self):
        for sec,browser in self.report_section_tabs.items():
            browser.setHtml(self.analysis_data["report_sections"][sec])
        self.update_graph_tabs()

    def update_graph_tabs(self):
        if not self.analysis_data:
            return

        seq = self.analysis_data["seq"]
        figs = {
            "Amino Acid Composition (Bar)": GraphingTools.create_amino_acid_composition_figure(
                self.analysis_data["aa_counts"], self.analysis_data["aa_freq"],
                label_font=self.label_font_size, tick_font=self.tick_font_size),
            "Amino Acid Composition (Pie)": GraphingTools.create_amino_acid_composition_pie_figure(
                self.analysis_data["aa_counts"], label_font=self.label_font_size),
            "Hydrophobicity Profile": GraphingTools.create_hydrophobicity_figure(
                self.analysis_data["hydro_profile"], self.analysis_data["window_size"],
                label_font=self.label_font_size, tick_font=self.tick_font_size),
            "Net Charge vs pH": GraphingTools.create_net_charge_vs_pH_figure(
                seq, label_font=self.label_font_size, tick_font=self.tick_font_size),
            "Bead Model (Hydrophobicity)": GraphingTools.create_bead_model_hydrophobicity_figure(
                seq, self.show_bead_labels,
                label_font=self.label_font_size, tick_font=self.tick_font_size,
                cmap=self.colormap),
            "Bead Model (Charge)": GraphingTools.create_bead_model_charge_figure(
                seq, self.show_bead_labels,
                label_font=self.label_font_size, tick_font=self.tick_font_size),
            "Properties Radar Chart": GraphingTools.create_radar_chart_figure(
                self.analysis_data, label_font=self.label_font_size)
        }

        # ─── Apply user‐selected styling to each figure ───
        main_graphs = {
            "Amino Acid Composition (Bar)",
            "Hydrophobicity Profile",
            "Net Charge vs pH"
        }
        for title, fig in figs.items():
            if title not in main_graphs:
                continue
            ax = fig.axes[0]
            # Heading on/off
            ax.set_title(ax.get_title() if self.show_heading else "")
            # Grid on/off
            ax.grid(self.show_grid)
            # Apply color & marker size to line plots
            for line in ax.get_lines():
                line.set_color(self.graph_color)
                line.set_markersize(self.marker_size)
            # Apply color to bars (bar chart)
            for patch in getattr(ax, "patches", []):
                patch.set_facecolor(self.graph_color)

        # ─── Clear old widgets and re‐embed each canvas ───
        for title, fig in figs.items():
            tab, vbox = self.graph_tabs[title]
            # Remove existing widgets
            for i in reversed(range(vbox.count())):
                widget = vbox.itemAt(i).widget()
                if widget is not None:
                    widget.setParent(None)
            # Add toolbar and canvas
            canvas = FigureCanvas(fig)
            toolbar = NavigationToolbar2QT(canvas, self)
            vbox.addWidget(toolbar)
            vbox.addWidget(canvas)
            # Add Save Graph button
            btn = QPushButton("Save Graph")
            btn.clicked.connect(lambda _, t=title: self.save_graph(t))
            vbox.addWidget(btn, alignment=Qt.AlignRight)

    def show_batch_details(self, row, col):
        sid=self.batch_table.item(row,0).text()
        for cid,seq,data in self.batch_data:
            if cid==sid:
                self.seq_text.setPlainText(seq)
                self.analysis_data=data
                self.update_analysis_tabs()
                break

    def save_report(self):
        fn,_=QFileDialog.getSaveFileName(self,"Save Report","","Text Files (*.txt)")
        if fn:
            ok,msg=ExportTools.export_report_text(self.analysis_data,fn)
            QMessageBox.information(self,"Save",msg if ok else msg)

    def export_pdf(self):
        fn,_=QFileDialog.getSaveFileName(self,"Export PDF","","PDF Files (*.pdf)")
        if fn:
            ExportTools.export_pdf(self.analysis_data,self.graph_tabs,fn,self)

    def save_graph(self,title):
        tab,vbox=self.graph_tabs[title]
        canvas=None
        for i in range(vbox.count()):
            w=vbox.itemAt(i).widget()
            if isinstance(w,FigureCanvas): canvas=w; break
        if not canvas:
            QMessageBox.warning(self,"No Graph","Graph not available."); return
        fn,_=QFileDialog.getSaveFileName(self,"Save Graph","", "PNG Files (*.png)")
        if fn:
            canvas.figure.savefig(fn)
            QMessageBox.information(self,"Saved",f"{title} → {fn}")

    def save_all_graphs(self):
        d=QFileDialog.getExistingDirectory(self,"Select Dir")
        if not d: return
        for title,(tab,vbox) in self.graph_tabs.items():
            for i in range(vbox.count()):
                w=vbox.itemAt(i).widget()
                if isinstance(w,FigureCanvas):
                    path=os.path.join(d, title.replace(" ","_")+".png")
                    w.figure.savefig(path)
                    break
        QMessageBox.information(self,"Saved","All graphs exported.")

    def import_batch(self):
        # alias for import_fasta
        self.import_fasta()

    def export_batch_csv(self):
        fn,_=QFileDialog.getSaveFileName(self,"Save CSV","","CSV Files (*.csv)")
        if fn and self.batch_data:
            with open(fn,"w",newline="") as f:
                writer=csv.writer(f)
                writer.writerow(["ID","Length","MolWeight","pI","NetCharge"])
                for cid,seq,data in self.batch_data:
                    writer.writerow([cid,len(seq),f"{data['mol_weight']:.2f}",f"{data['iso_point']:.2f}",f"{data['net_charge_7']:.2f}"])
            QMessageBox.information(self,"Saved","Batch CSV exported.")

    def export_batch_json(self):
        fn,_=QFileDialog.getSaveFileName(self,"Save JSON","","JSON Files (*.json)")
        if fn and self.batch_data:
            out=[{"ID":cid,"Length":len(seq),"MolWeight":data["mol_weight"],"pI":data["iso_point"],"NetCharge":data["net_charge_7"]} for cid,seq,data in self.batch_data]
            with open(fn,"w") as f: json.dump(out,f,indent=2)
            QMessageBox.information(self,"Saved","Batch JSON exported.")

    def toggle_theme(self):
        if self.theme_toggle.isChecked():
            self.setStyleSheet(DARK_THEME_CSS); plt.style.use("dark_background")
        else:
            self.setStyleSheet(LIGHT_THEME_CSS); plt.style.use("default")
        self.statusBar.showMessage(f"{'Dark' if self.theme_toggle.isChecked() else 'Light'} theme activated",2000)

    def apply_settings(self):
        # Primary settings
        try:
            self.default_window_size = int(self.window_size_input.text())
        except ValueError:
            pass

        try:
            self.default_pH = float(self.ph_input.text())   # NEW
        except ValueError:
            pass

        self.show_bead_labels = self.label_checkbox.isChecked()
        self.colormap = self.colormap_combo.currentText()

        # Advanced settings
        try:
            self.label_font_size = int(self.label_font_input.text())
        except ValueError:
            pass
        try:
            self.tick_font_size = int(self.tick_font_input.text())
        except ValueError:
            pass
        try:
            self.marker_size = int(self.marker_size_input.text())
        except (ValueError, AttributeError):
            pass
        self.graph_color = self.graph_color_combo.currentText()
        self.show_heading = self.heading_checkbox.isChecked()
        self.show_grid = self.grid_checkbox.isChecked()

        # Refresh display if already analyzed
        if self.analysis_data:
            self.update_analysis_tabs()
            self.update_graph_tabs()
        self.statusBar.showMessage("Settings applied", 2000)

def main():
    app = QApplication(sys.argv)
    w = ProteinAnalyzerGUI()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
