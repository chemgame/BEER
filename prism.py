#!/usr/bin/env python3
"""
PRISM - Protein Residue Informatics & Sequence Metrics

Requirements:
  pip install biopython matplotlib PyQt5 mplcursors
"""

import sys, math, os, base64, json, csv
from io import BytesIO

# PyQt5 imports
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QTextBrowser,
    QFileDialog, QTabWidget, QMessageBox, QTableWidget, QTableWidgetItem,
    QCheckBox, QStatusBar, QComboBox, QProgressBar, QFormLayout
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtPrintSupport import QPrinter

# Matplotlib setup
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
# Constants & Helpers
# ---------------------------

VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
KYTE_DOOLITTLE = {
    'A': 1.8,  'R': -4.5, 'N': -3.5, 'D': -3.5,
    'C': 2.5,  'Q': -3.5, 'E': -3.5, 'G': -0.4,
    'H': -3.2, 'I': 4.5,  'L': 3.8,  'K': -3.9,
    'M': 1.9,  'F': 2.8,  'P': -1.6, 'S': -0.8,
    'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

# Only these three report tabs
REPORT_SECTIONS = [
    "Overview",
    "Composition",
    "Properties",
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

# Light / dark themes
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
    """Trim whitespace and uppercase."""
    return seq.strip().replace(" ", "").upper()

def is_valid_protein(seq: str) -> bool:
    """Check only standard amino acids."""
    return all(aa in VALID_AMINO_ACIDS for aa in seq)

def calc_net_charge(seq: str, pH: float = 7.0) -> float:
    """Henderson–Hasselbalch net charge."""
    pKa_nterm = 9.69; pKa_cterm = 2.34

    if hasattr(calc_net_charge, "pka_override") and calc_net_charge.pka_override:
        p = calc_net_charge.pka_override
        pKa_nterm, pKa_cterm = p['NTERM'], p['CTERM']
        pKa_side = {'D':p['D'],'E':p['E'],'C':p['C'],'Y':p['Y'],
                    'H':p['H'],'K':p['K'],'R':p['R']}
    else:
        pKa_nterm, pKa_cterm = 9.69, 2.34
        pKa_side = {'D':3.90,'E':4.07,'C':8.18,'Y':10.46,'H':6.04,'K':10.54,'R':12.48}


    net = 1/(1+10**(pH-pKa_nterm)) - 1/(1+10**(pKa_cterm-pH))
    for aa in seq:
        if aa in ('D','E','C','Y'):
            net -= 1/(1+10**(pKa_side[aa]-pH))
        elif aa in ('K','R','H'):
            net += 1/(1+10**(pH-pKa_side[aa]))
    return net

def sliding_window_hydrophobicity(seq: str, window_size: int=9) -> list:
    """Compute Kyte–Doolittle average over sliding windows."""
    if window_size > len(seq):
        return [sum(KYTE_DOOLITTLE[aa] for aa in seq)/len(seq)]
    return [
        sum(KYTE_DOOLITTLE[aa] for aa in seq[i:i+window_size]) / window_size
        for i in range(len(seq)-window_size+1)
    ]

# ---------------------------
# Analysis Tools
# ---------------------------
class AnalysisTools:
    @staticmethod
    def analyze_sequence(seq: str, pH_value: float=7.0, window_size: int=9) -> dict:
        """Compute all metrics and build HTML report sections."""
        pa = BPProteinAnalysis(seq)
        aa_counts     = pa.count_amino_acids()
        seq_length    = len(seq)
        aa_freq       = {aa:(count/seq_length*100) for aa,count in aa_counts.items()}
        mol_weight    = pa.molecular_weight()
        iso_point     = pa.isoelectric_point()
        gravy         = pa.gravy()
        instability   = pa.instability_index()
        aromaticity   = pa.aromaticity()
        net_charge_7  = calc_net_charge(seq, 7.0)
        net_charge_pH = calc_net_charge(seq, pH_value)
        solubility    = AnalysisTools.predict_solubility(seq)
        n_trp = seq.count("W")
        n_tyr = seq.count("Y")
        n_cys = seq.count("C")
        use_reducing = getattr(AnalysisTools, "use_reducing", False)
        if use_reducing:
            n_cystine = 0          # all Cys remain reduced
        else:
            n_cystine = n_cys // 2 # every pair forms a disulphide
        extinction = 5500*n_trp + 1490*n_tyr + 125*n_cystine        

        # Overview table with optional extra pH row
        extra_charge_row = (
            f"<tr><td>Net Charge (pH {pH_value:.1f})</td><td>{net_charge_pH:.2f}</td></tr>"
            if abs(pH_value-7.0) >= 1e-6 else ""
        )
        overview_html = f"""
        <h2>Overview</h2>
        <table border="1" cellpadding="5">
          <tr><th>Property</th><th>Value</th></tr>
          <tr><td>Sequence Length</td><td>{seq_length} aa</td></tr>
          <tr><td>Sequence</td><td>{seq}</td></tr>
          <tr><td>Net Charge (pH 7.0)</td><td>{net_charge_7:.2f}</td></tr>
          {extra_charge_row}
          <tr><td>Solubility Prediction</td><td>{solubility}</td></tr>
        </table>
        """

        # Composition table
        sorted_aas = sorted(aa_counts, key=lambda aa: aa_freq[aa], reverse=True)
        comp_html = """
        <h2>Composition</h2>
        <table border="1" cellpadding="5">
          <tr><th>Amino Acid</th><th>Count</th><th>Frequency (%)</th></tr>
        """ + "".join(
            f"<tr><td>{aa}</td><td>{aa_counts[aa]}</td><td>{aa_freq[aa]:.2f}%</td></tr>"
            for aa in sorted_aas
        ) + "</table>"

        # Properties table
        bio_html = f"""
        <h2>Properties</h2>
        <table border="1" cellpadding="5">
          <tr><td>Mol. Weight</td><td>{mol_weight:.2f} Da</td></tr>
          <tr><td>Isoelectric Point</td><td>{iso_point:.2f}</td></tr>
          <tr><td>Extinction Coeff.</td><td>{extinction} M⁻¹cm⁻¹</td></tr>
          <tr><td>GRAVY</td><td>{gravy:.3f}</td></tr>
          <tr><td>Instability</td><td>{instability:.2f}</td></tr>
          <tr><td>Aromaticity</td><td>{aromaticity:.3f}</td></tr>
        </table>
        """

        report_sections = {
            "Overview": overview_html,
            "Composition": comp_html,
            "Properties": bio_html,
        }
        return {
            "report_sections": report_sections,
            "aa_counts": aa_counts,
            "aa_freq": aa_freq,
            "hydro_profile": sliding_window_hydrophobicity(seq, window_size),
            "window_size": window_size,
            "seq": seq,
            "mol_weight": mol_weight,
            "iso_point": iso_point,
            "net_charge_7": net_charge_7,
            "extinction": extinction,
            "gravy": gravy,
            "instability": instability,
            "aromaticity": aromaticity
        }

    @staticmethod
    def predict_solubility(seq: str) -> str:
        """Simple hydrophobicity‐based solubility heuristic."""
        avg_hydro = sum(KYTE_DOOLITTLE[aa] for aa in seq) / len(seq)
        return "Likely soluble" if avg_hydro < 0.5 else "Low solubility predicted"

# ---------------------------
# Graphing Tools
# ---------------------------
class GraphingTools:
    """Generate matplotlib Figures for each graph type."""

    @staticmethod
    def create_amino_acid_composition_figure(aa_counts, aa_freq, label_font=14, tick_font=12):
        fig = Figure(figsize=(5,4))
        ax  = fig.add_subplot(111)
        aas = sorted(aa_counts)
        cnts= [aa_counts[a] for a in aas]
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
        ax  = fig.add_subplot(111)
        cmap   = plt.get_cmap("tab20")
        colors = [cmap(i) for i in range(len(aa_counts))]
        ax.pie(list(aa_counts.values()), labels=list(aa_counts.keys()),
               colors=colors, autopct="%1.1f%%", startangle=140)
        ax.set_title("AA Composition (Pie)", fontsize=label_font+2)
        mplcursors.cursor(ax)
        return fig

    @staticmethod
    def create_hydrophobicity_figure(hydro_profile, window_size, label_font=14, tick_font=12):
        fig = Figure(figsize=(5,4))
        ax  = fig.add_subplot(111)
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
        ax  = fig.add_subplot(111)
        phs = [i/10 for i in range(0,141)]
        nets= [calc_net_charge(seq,p) for p in phs]
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
        ax  = fig.add_subplot(111)
        xs  = list(range(1,len(seq)+1))
        vals= [KYTE_DOOLITTLE[aa] for aa in seq]
        sc  = ax.scatter(xs, [1]*len(seq), c=vals, cmap=cmap, s=200)
        fig.colorbar(sc, ax=ax, label="Hydrophobicity")
        ax.set_yticks([]); ax.set_xlabel("Residue", fontsize=label_font)
        ax.set_title("Bead Hydrophobicity", fontsize=label_font+2)
        if show_labels and len(seq)<=50:
            for i,aa in enumerate(seq):
                ax.text(xs[i],1,aa,ha="center",va="center",fontsize=label_font-2)
        mplcursors.cursor(ax)
        return fig

    @staticmethod
    def create_bead_model_charge_figure(seq, show_labels, label_font=14, tick_font=12):
        fig = Figure(figsize=(min(12,0.25*len(seq)),2))
        ax  = fig.add_subplot(111)
        xs  = list(range(1,len(seq)+1))
        cols= ["blue" if aa in "KRH" else "red" if aa in "DE" else "gray" for aa in seq]
        ax.scatter(xs, [1]*len(seq), c=cols, s=200)
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color="blue", label="Pos"),
            Patch(color="red",  label="Neg"),
            Patch(color="gray", label="Neu")
        ], loc="upper right")
        ax.set_yticks([]); ax.set_xlabel("Residue", fontsize=label_font)
        ax.set_title("Bead Charge", fontsize=label_font+2)
        if show_labels and len(seq)<=50:
            for i,aa in enumerate(seq):
                ax.text(xs[i],1,aa,ha="center",va="center",color="white",fontsize=label_font-2)
        mplcursors.cursor(ax)
        return fig

    @staticmethod
    def create_radar_chart_figure(data, label_font=14):
        props = ["Mol Weight","pI","GRAVY","Instability","Aromaticity"]
        vals  = [data["mol_weight"],data["iso_point"],data["gravy"],
                 data["instability"],data["aromaticity"]]
        ranges= {
            "Mol Weight":(5000,150000),"pI":(4,11),"GRAVY":(-2.5,2.5),
            "Instability":(20,80),"Aromaticity":(0,0.2)
        }
        norm=[] 
        for p,v in zip(props,vals):
            mn,mx= ranges[p]
            norm.append(max(0,min(1,(v-mn)/(mx-mn))))
        norm += norm[:1]
        angles= [n/len(props)*2*math.pi for n in range(len(props))] + [0]
        fig = Figure(figsize=(5,4))
        ax  = fig.add_subplot(111,polar=True)
        ax.plot(angles,norm); ax.fill(angles,norm,alpha=0.4)
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(props,fontsize=label_font-2)
        ax.set_title("Radar Chart",fontsize=label_font+2)
        mplcursors.cursor(ax)
        return fig

# ---------------------------
# Export Tools
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
                if hasattr(w, "figure"):
                    buf = BytesIO()
                    w.figure.savefig(buf, format="png")
                    buf.seek(0)
                    b64 = base64.b64encode(buf.read()).decode()
                    html += (
                       f"<h2>{title}</h2>"
                       "<div style='text-align:left;'>"
                       f"<img src='data:image/png;base64,{b64}' style='width:80%;'/>"
                       "</div>"
                       "<div style='page-break-after:always;'></div>"
                   )
                    break
        return html

    @staticmethod
    def export_report_text(analysis_data, file_name):
        """Plain‑text export of HTML report sections."""
        try:
            with open(file_name, "w") as f:
                for sec, content in analysis_data["report_sections"].items():
                    text = content.replace("<br>", "\n")
                    for tag in ("<h2>","</h2>","<table","</table>","<tr>","</tr>"):
                        text = text.replace(tag, "\n")
                    text = text.replace("<th>", "").replace("</th>", "\t")
                    text = text.replace("<td>", "").replace("</td>", "\t")
                    f.write(f"==== {sec} ====\n{text}\n\n")
            return True, f"Report saved to {file_name}"
        except Exception as e:
            return False, f"Save error: {e}"

    @staticmethod
    def export_pdf(analysis_data, graph_tabs, file_name, parent):
        """Render full HTML into a PDF via Qt printer."""
        printer = QPrinter(QPrinter.HighResolution)
        printer.setOutputFormat(QPrinter.PdfFormat)
        printer.setOutputFileName(file_name)
        browser = QTextBrowser()
        browser.setHtml(ExportTools._generate_full_html(analysis_data, graph_tabs))
        browser.document().print_(printer)
        QMessageBox.information(parent, "Success", f"PDF exported to {file_name}")

# ---------------------------
# PDB Import
# ---------------------------
def import_pdb_sequence(file_name: str) -> dict:
    """Extract one-letter sequences for each chain in a PDB."""
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

# ---------------------------
# Main GUI Application
# ---------------------------
class ProteinAnalyzerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PRISM - Protein Residue Informatics & Sequence Metrics")
        self.resize(1200, 900)

        # Default theme
        self.setStyleSheet(LIGHT_THEME_CSS)

        # Status bar & progress
        #self.statusBar   = QStatusBar(); self.setStatusBar(self.statusBar)
        #self.progress_bar= QProgressBar(); self.progress_bar.setRange(0,100)
        #self.progress_bar.setVisible(False)
        #self.statusBar.addPermanentWidget(self.progress_bar)
        # Status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

        # State
        self.analysis_data   = None
        self.batch_data      = []
        self.default_window_size = 9
        self.default_pH          = 7.0
        self.show_bead_labels    = True
        self.label_font_size     = 14
        self.tick_font_size      = 12
        self.marker_size         = 10
        self.colormap            = "coolwarm"
        self.graph_color         = "blue"
        self.show_heading        = True
        self.show_grid           = True
        self.default_graph_format = "PNG"
        self.enable_tooltips      = False

        # Main tabs
        self.check_dependencies()
        self.main_tabs = QTabWidget()
        self.setCentralWidget(self.main_tabs)
        self.init_analysis_tab()
        self.init_graphs_tab()
        self.init_batch_tab()
        self.init_settings_tab()
        self.init_help_tab()

    def init_analysis_tab(self):
        """Build the Analysis tab with import buttons, sequence box, chain dropdown, and report panes."""
        layout = QVBoxLayout(); container = QWidget(); container.setLayout(layout)
        self.main_tabs.addTab(container, "Analysis")

        # Import / Analyze row
        row = QHBoxLayout()
        self.import_fasta_btn = QPushButton("Import FASTA"); self.import_fasta_btn.clicked.connect(self.import_fasta)
        self.import_pdb_btn   = QPushButton("Import PDB");   self.import_pdb_btn.clicked.connect(self.import_pdb)
        self.analyze_btn      = QPushButton("Analyze");      self.analyze_btn.clicked.connect(self.on_analyze)
        for w in (self.import_fasta_btn, self.import_pdb_btn, self.analyze_btn):
            row.addWidget(w)
        layout.addLayout(row)

        # Export row
        exp = QHBoxLayout()
        self.save_txt_btn = QPushButton("Save Report"); self.save_txt_btn.clicked.connect(self.save_report)
        self.save_pdf_btn = QPushButton("Export PDF");   self.save_pdf_btn.clicked.connect(self.export_pdf)
        exp.addWidget(self.save_txt_btn); exp.addWidget(self.save_pdf_btn)
        layout.addLayout(exp)

        # Sequence input
        layout.addWidget(QLabel("Protein Sequence:"))
        self.seq_text = QTextEdit(); layout.addWidget(self.seq_text)

        # Chain selector
        layout.addWidget(QLabel("Select Chain:"))
        self.chain_combo = QComboBox(); self.chain_combo.setEnabled(False)
        self.chain_combo.currentTextChanged.connect(self.on_chain_selected)
        layout.addWidget(self.chain_combo)

        # Report section tabs
        self.report_tabs = QTabWidget(); layout.addWidget(self.report_tabs)
        self.report_section_tabs = {}
        for sec in REPORT_SECTIONS:
            tab = QWidget(); vb = QVBoxLayout(tab)
            if sec == "Composition":
                # Sort buttons for composition table
                sort_row = QHBoxLayout()
                for label, mode in [("A→Z","alpha"),("By Comp","composition"),("Hydro ↑","hydro_inc"),("Hydro ↓","hydro_dec")]:
                    btn = QPushButton(label)
                    btn.clicked.connect(lambda _,m=mode: self.sort_composition(m))
                    sort_row.addWidget(btn)
                vb.addLayout(sort_row)
            browser = QTextBrowser(); vb.addWidget(browser)
            self.report_tabs.addTab(tab, sec)
            self.report_section_tabs[sec] = browser

    def init_graphs_tab(self):
        """Set up empty graph placeholders."""
        self.graphs_subtabs = QTabWidget()
        layout = QVBoxLayout(); container = QWidget(); container.setLayout(layout)
        self.main_tabs.addTab(container, "Graphs")
        layout.addWidget(self.graphs_subtabs)
        self.graph_tabs = {}
        for title in GRAPH_TITLES:
            tab = QWidget(); vb = QVBoxLayout(tab)
            placeholder = QLabel(f"{title} will appear here")
            placeholder.setAlignment(Qt.AlignCenter)
            vb.addWidget(placeholder)
            btn = QPushButton("Save Graph"); btn.clicked.connect(lambda _,t=title: self.save_graph(t))
            vb.addWidget(btn, alignment=Qt.AlignRight)
            self.graphs_subtabs.addTab(tab, title)
            self.graph_tabs[title] = (tab, vb)
        save_all = QPushButton("Save All Graphs"); save_all.clicked.connect(self.save_all_graphs)
        layout.addWidget(save_all, alignment=Qt.AlignRight)

    def init_batch_tab(self):
        """Batch‐analysis table and export."""
        layout = QVBoxLayout(); container = QWidget(); container.setLayout(layout)
        self.main_tabs.addTab(container, "Multichain Analysis")
        btn_row = QHBoxLayout()
        self.batch_export_csv_btn  = QPushButton("Export CSV");  self.batch_export_csv_btn.clicked.connect(self.export_batch_csv)
        self.batch_export_json_btn = QPushButton("Export JSON"); self.batch_export_json_btn.clicked.connect(self.export_batch_json)
        btn_row.addWidget(self.batch_export_csv_btn); btn_row.addWidget(self.batch_export_json_btn)
        layout.addLayout(btn_row)
        self.batch_table = QTableWidget()
        self.batch_table.setColumnCount(9)
        self.batch_table.setHorizontalHeaderLabels([
            "ID","Length","MW (Da)","Net Charge (pH 7)",
            "% Hydro","% Hydrophil","% +Charged","% -Charged","% Neutral"
        ])
        self.batch_table.cellDoubleClicked.connect(self.show_batch_details)
        layout.addWidget(self.batch_table)

    def init_settings_tab(self):
        """Build the Settings tab: all settings in a single grouped form."""
        container = QWidget()
        self.main_tabs.addTab(container, "Settings")
        layout = QVBoxLayout(container)

        form = QFormLayout()

        # — Text inputs —
        self.ph_input = QLineEdit(str(self.default_pH))
        self.ph_input.setToolTip("Sets the pH value used for net‑charge calculations.")
        form.addRow("Default pH:", self.ph_input)

        self.window_size_input = QLineEdit(str(self.default_window_size))
        self.window_size_input.setToolTip("Length of sliding window for hydrophobicity profiles.")
        form.addRow("Sliding Window Size:", self.window_size_input)

        self.label_font_input = QLineEdit(str(self.label_font_size))
        form.addRow("Label Font Size:", self.label_font_input)

        self.tick_font_input = QLineEdit(str(self.tick_font_size))
        form.addRow("Tick Font Size:", self.tick_font_input)

        self.marker_size_input = QLineEdit(str(self.marker_size))
        self.marker_size_input.setToolTip("Size of data markers in line and scatter graphs.")
        form.addRow("Marker Size:", self.marker_size_input)

        self.pka_input = QLineEdit("")
        self.pka_input.setPlaceholderText("Custom pKa (N-term,C-term,D,E,C,Y,H,K,R)")
        self.pka_input.setToolTip("Leave blank for defaults. Provide nine comma-separated numbers.")
        form.addRow("Override pKa list:", self.pka_input)

        # — Default Graph Format —
        self.graph_format_combo = QComboBox()
        self.graph_format_combo.addItems(["PNG", "SVG", "PDF"])
        self.graph_format_combo.setToolTip("Default file format when saving individual or all graphs.")
        form.addRow("Default Graph Format:", self.graph_format_combo)

        # — Dropdowns —
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems([
            "viridis", "plasma", "coolwarm", "inferno", "magma",
            "cividis", "Spectral", "hot", "copper", "cool", "autumn",
            "pink", "berlin", "vaniamo", "managua", "hsv"
        ])
        self.colormap_combo.setToolTip("Color map for hydrophobicity and radar charts.")
        form.addRow("Colormap:", self.colormap_combo)

        self.graph_color_combo = QComboBox()
        self.graph_color_combo.addItems([
            "blue", "green", "red", "cyan",
            "magenta", "yellow", "black", "gray"
        ])
        self.graph_color_combo.setToolTip("Color for line and bar chart elements.")
        form.addRow("Graph Color:", self.graph_color_combo)

        # — Checkboxes —
        self.theme_toggle = QCheckBox("Dark Theme")
        self.theme_toggle.setToolTip("Toggle between light and dark application themes.")
        self.theme_toggle.stateChanged.connect(self.toggle_theme)
        self.theme_toggle.setChecked(False)
        form.addRow("", self.theme_toggle)

        self.tooltips_checkbox = QCheckBox("Enable Tooltips")
        self.tooltips_checkbox.setToolTip("Show or hide tooltips across the application.")
        self.tooltips_checkbox.setChecked(False)
        form.addRow("", self.tooltips_checkbox)

        self.label_checkbox = QCheckBox("Show bead labels")
        self.label_checkbox.setToolTip("Display residue labels on bead models when sequence is short.")
        self.label_checkbox.setChecked(self.show_bead_labels)
        form.addRow("", self.label_checkbox)

        self.heading_checkbox = QCheckBox("Show Graph Heading")
        self.heading_checkbox.setToolTip("Display titles above graphs.")
        self.heading_checkbox.setChecked(self.show_heading)
        form.addRow("", self.heading_checkbox)

        self.grid_checkbox = QCheckBox("Show Grid")
        self.grid_checkbox.setToolTip("Toggle grid lines on plots.")
        self.grid_checkbox.setChecked(self.show_grid)
        form.addRow("", self.grid_checkbox)

        self.reducing_checkbox = QCheckBox("Assume reducing conditions (Cys not in disulphide)")
        self.reducing_checkbox.setToolTip("If checked, Cys residues are counted as free thiols \
        rather than cystine when the 280-nm extinction coefficient is calculated.")
        form.addRow("", self.reducing_checkbox)        

        layout.addLayout(form)

        # — Apply & Reset Buttons —
        btn_row = QHBoxLayout()
        apply_btn = QPushButton("Apply Settings")
        apply_btn.clicked.connect(self.apply_settings)
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self.reset_defaults)
        btn_row.addWidget(apply_btn)
        btn_row.addWidget(reset_btn)
        layout.addLayout(btn_row)

    def init_help_tab(self):
        """Help text matching the three report tabs and graphs."""
        layout = QVBoxLayout(); container = QWidget(); container.setLayout(layout)
        self.main_tabs.addTab(container, "Help")
        b = QTextBrowser()
        b.setHtml("""
        <h1>PRISM Help &amp; Definitions</h1>

        <h2>Protein Sequence</h2>
        <p>Enter a single-letter amino acid sequence or import from FASTA/PDB.</p>

        <h2>Overview</h2>
        <ul>
          <li><b>Sequence Length:</b> Number of residues.</li>
          <li><b>Sequence:</b> The raw amino acid string.</li>
          <li><b>Net Charge:</b> At pH 7.0 (and custom pH if specified).</li>
          <li><b>Solubility Prediction:</b> Based on average hydrophobicity.</li>
        </ul>

        <h2>Composition</h2>
        <p>Counts and percentage frequencies of each residue.</p>

        <h2>Properties</h2>
        <ul>
          <li><b>Molecular Weight:</b> Approx. mass in Daltons.</li>
          <li><b>Isoelectric Point (pI):</b> pH with zero net charge.</li>
          <li><b>Extinction Coeff.:</b> Absorbance at 280 nm per M per cm.</li>
          <li><b>GRAVY Score:</b> Grand average of hydropathicity (higher = hydrophobic).</li>
          <li><b>Instability Index:</b> &lt;40 suggests stable protein.</li>
          <li><b>Aromaticity:</b> Fraction of F, W, Y residues.</li>
        </ul>

        <h2>Graphs</h2>
        <ul>
          <li><b>Bar/Pie Charts:</b> Amino acid composition.</li>
          <li><b>Hydrophobicity Profile:</b> Sliding-window average.</li>
          <li><b>Net Charge vs pH:</b> Charge curve from pH 0 to 14.</li>
          <li><b>Bead Models:</b> Per-residue hydrophobicity or charge.</li>
          <li><b>Radar Chart:</b> Normalized physiochemical properties.</li>
        </ul>

        <h2>Batch Analysis</h2>
        <p>Import multi-FASTA or PDB to analyze multiple sequences; select one for detail.</p>

        <h2>Settings</h2>
        <p>Adjust window size, pH, fonts, colormap, theme, and display options.</p>
        """)
        layout.addWidget(b)

    def import_fasta(self):
        """Batch-import from FASTA, populate table and chain dropdown."""
        file_name, _ = QFileDialog.getOpenFileName(self, "Open FASTA File", "", "FASTA Files (*.fa *.fasta)")
        if not file_name:
            return
        records = list(SeqIO.parse(file_name, "fasta"))
        if not records:
            QMessageBox.warning(self, "No Records", "No sequences found.")
            return

        # Clear previous data
        self.batch_data.clear()
        self.batch_table.setRowCount(0)

        # Analyze each record
        for rec in records:
            seq = clean_sequence(str(rec.seq))
            if not is_valid_protein(seq):
                continue
            data = AnalysisTools.analyze_sequence(seq, 7.0, self.default_window_size)
            self.batch_data.append((rec.id, seq, data))
            # Fill table row
            length = len(seq)
            mw     = data["mol_weight"]
            net7   = data["net_charge_7"]
            hydro  = sum(1 for aa in seq if KYTE_DOOLITTLE[aa]>0)/length*100
            pos    = sum(data["aa_counts"].get(k,0) for k in ("K","R","H"))/length*100
            neg    = sum(data["aa_counts"].get(k,0) for k in ("D","E"))/length*100
            neu    = 100-(pos+neg)
            row = self.batch_table.rowCount()
            self.batch_table.insertRow(row)
            for col,val in enumerate([
                rec.id, str(length), f"{mw:.2f}", f"{net7:.2f}",
                f"{hydro:.1f}%", f"{100-hydro:.1f}%",
                f"{pos:.1f}%", f"{neg:.1f}%", f"{neu:.1f}%"
            ]):
                self.batch_table.setItem(row, col, QTableWidgetItem(val))

        # Populate chain dropdown
        self.chain_combo.clear()
        for rec_id, _, _ in self.batch_data:
            self.chain_combo.addItem(rec_id)
        self.chain_combo.setEnabled(True)
        if self.chain_combo.count()>0:
            self.chain_combo.setCurrentIndex(0)

    def import_pdb(self):
        """Batch-import from PDB, populate table and chain dropdown."""
        file_name, _ = QFileDialog.getOpenFileName(self, "Open PDB File", "", "PDB Files (*.pdb)")
        if not file_name:
            return
        chains = import_pdb_sequence(file_name)
        if not chains:
            QMessageBox.warning(self, "No Chains", "No valid chains found.")
            return

        # Clear previous data
        self.batch_data.clear()
        self.batch_table.setRowCount(0)

        # Analyze each chain
        for cid, seq in chains.items():
            data = AnalysisTools.analyze_sequence(seq, 7.0, self.default_window_size)
            self.batch_data.append((cid, seq, data))
            # Fill table row
            length = len(seq)
            mw     = data["mol_weight"]
            net7   = data["net_charge_7"]
            hydro  = sum(1 for aa in seq if KYTE_DOOLITTLE[aa]>0)/length*100
            pos    = sum(data["aa_counts"].get(k,0) for k in ("K","R","H"))/length*100
            neg    = sum(data["aa_counts"].get(k,0) for k in ("D","E"))/length*100
            neu    = 100-(pos+neg)
            row = self.batch_table.rowCount()
            self.batch_table.insertRow(row)
            for col,val in enumerate([
                cid, str(length), f"{mw:.2f}", f"{net7:.2f}",
                f"{hydro:.1f}%", f"{100-hydro:.1f}%",
                f"{pos:.1f}%", f"{neg:.1f}%", f"{neu:.1f}%"
            ]):
                self.batch_table.setItem(row, col, QTableWidgetItem(val))

        # Populate chain dropdown
        self.chain_combo.clear()
        for cid, _, _ in self.batch_data:
            self.chain_combo.addItem(cid)
        self.chain_combo.setEnabled(True)
        if self.chain_combo.count()>0:
            self.chain_combo.setCurrentIndex(0)

    def sort_composition(self, mode):
        """Rebuild Composition HTML sorted by the given mode."""
        counts = self.analysis_data["aa_counts"]
        freq   = self.analysis_data["aa_freq"]
        items  = list(counts.items())
        if mode=="alpha":
            items.sort(key=lambda x: x[0])
        elif mode=="composition":
            items.sort(key=lambda x: freq[x[0]], reverse=True)
        elif mode=="hydro_inc":
            items.sort(key=lambda x: KYTE_DOOLITTLE[x[0]])
        else:  # hydro_dec
            items.sort(key=lambda x: KYTE_DOOLITTLE[x[0]], reverse=True)

        html = "<h2>Composition</h2><table border='1' cellpadding='5'>"
        html += "<tr><th>Amino Acid</th><th>Count</th><th>Frequency (%)</th></tr>"
        for aa, cnt in items:
            html += f"<tr><td>{aa}</td><td>{cnt}</td><td>{freq[aa]:.2f}%</td></tr>"
        html += "</table>"
        self.report_section_tabs["Composition"].setHtml(html)

        # Refresh the bar-chart to match new order
        tab,vb = self.graph_tabs["Amino Acid Composition (Bar)"]
        for i in reversed(range(vb.count())):
            w = vb.itemAt(i).widget()
            if w: w.setParent(None)
        fig = GraphingTools.create_amino_acid_composition_figure(
            dict(items), {aa:freq[aa] for aa,_ in items},
            label_font=self.label_font_size, tick_font=self.tick_font_size
        )
        canvas = FigureCanvas(fig)
        vb.addWidget(NavigationToolbar2QT(canvas, self))
        vb.addWidget(canvas)
        btn = QPushButton("Save Graph"); btn.clicked.connect(lambda _,t="Amino Acid Composition (Bar)": self.save_graph(t))
        vb.addWidget(btn, alignment=Qt.AlignRight)

    def on_analyze(self):
        """Analyze whatever is in the sequence box (manual or selected chain)."""
        #self.progress_bar.setRange(0,100); self.progress_bar.setValue(50); self.progress_bar.show()
        seq = clean_sequence(self.seq_text.toPlainText())
        if not seq:
            QMessageBox.warning(self, "Input", "Enter sequence."); return
        if not is_valid_protein(seq):
            QMessageBox.warning(self, "Input", "Invalid residues."); return
        try:
            pH = float(self.ph_input.text())
        except:
            pH = 7.0

        self.analysis_data = AnalysisTools.analyze_sequence(seq, pH, self.default_window_size)
        # If manual, disable chain combo
        if not any(seq == s for _,s,_ in self.batch_data):
            self.chain_combo.clear(); self.chain_combo.setEnabled(False)

        # Update all tabs
        for sec, browser in self.report_section_tabs.items():
            browser.setHtml(self.analysis_data["report_sections"][sec])
        self.update_graph_tabs()

        QMessageBox.information(self, "Done", "Analysis complete.")
        #self.progress_bar.hide()

    def update_graph_tabs(self):
        """Regenerate all main graphs with current styling."""
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
                seq, self.show_bead_labels, label_font=self.label_font_size,
                tick_font=self.tick_font_size, cmap=self.colormap),
            "Bead Model (Charge)": GraphingTools.create_bead_model_charge_figure(
                seq, self.show_bead_labels, label_font=self.label_font_size,
                tick_font=self.tick_font_size),
            "Properties Radar Chart": GraphingTools.create_radar_chart_figure(
                self.analysis_data, label_font=self.label_font_size)
        }

        # Apply user styling to main graphs
        main = {"Amino Acid Composition (Bar)","Hydrophobicity Profile","Net Charge vs pH"}
        for title, fig in figs.items():
            if title in main:
                ax = fig.axes[0]
                ax.set_title(ax.get_title() if self.show_heading else "")
                ax.grid(self.show_grid)
                for line in ax.get_lines():
                    line.set_color(self.graph_color); line.set_markersize(self.marker_size)
                for patch in getattr(ax, "patches", []):
                    patch.set_facecolor(self.graph_color)

        # Embed each canvas + toolbar
        for title, fig in figs.items():
            tab,vb = self.graph_tabs[title]
            for i in reversed(range(vb.count())):
                w = vb.itemAt(i).widget()
                if w: w.setParent(None)
            canvas = FigureCanvas(fig)
            vb.addWidget(NavigationToolbar2QT(canvas, self))
            vb.addWidget(canvas)
            btn = QPushButton("Save Graph"); btn.clicked.connect(lambda _,t=title: self.save_graph(t))
            vb.addWidget(btn, alignment=Qt.AlignRight)

    def show_batch_details(self, row, _):
        """Load selected batch sequence into Analysis view."""
        sid = self.batch_table.item(row,0).text()
        for cid, seq, data in self.batch_data:
            if cid == sid:
                self.seq_text.setPlainText(seq)
                self.analysis_data = data
                for sec,browser in self.report_section_tabs.items():
                    browser.setHtml(data["report_sections"][sec])
                self.update_graph_tabs()
                return

    def save_report(self):
        """Prompt to save text report."""
        fn,_ = QFileDialog.getSaveFileName(self, "Save Report", "", "Text Files (*.txt)")
        if fn:
            ok,msg = ExportTools.export_report_text(self.analysis_data, fn)
            QMessageBox.information(self, "Save", msg)

    def export_pdf(self):
        """Prompt to export full PDF."""
        fn,_ = QFileDialog.getSaveFileName(self, "Export PDF", "", "PDF Files (*.pdf)")
        if fn:
            ExportTools.export_pdf(self.analysis_data, self.graph_tabs, fn, self)

    def save_graph(self, title):
        """Save a single graph PNG."""
        tab,vb = self.graph_tabs[title]
        canvas = next((w for i in range(vb.count()) if isinstance((w:=vb.itemAt(i).widget()), FigureCanvas)), None)
        if not canvas:
            QMessageBox.warning(self, "No Graph", "Graph not available."); return
        ext  = self.default_graph_format.lower()
        fn,_ = QFileDialog.getSaveFileName(
            self, "Save Graph", "",
            f"{self.default_graph_format} Files (*.{ext})"
        )
        if fn:
            if not fn.lower().endswith(f".{ext}"):
                fn += f".{ext}"
            canvas.figure.savefig(fn, format=ext)            
            QMessageBox.information(self, "Saved", f"{title} → {fn}")

    def save_all_graphs(self):
        """Export all graphs at once."""
        d = QFileDialog.getExistingDirectory(self, "Select Directory")
        if not d:
            return
        ext = self.default_graph_format.lower()
        for title,(tab,vb) in self.graph_tabs.items():    
            canvas = next((w for i in range(vb.count()) if isinstance((w:=vb.itemAt(i).widget()), FigureCanvas)), None)
            if canvas:
                fname = title.replace(" ","_") + f".{ext}"
                path = os.path.join(d, fname)
                canvas.figure.savefig(path, format=ext)                
                canvas.figure.savefig(path)
        QMessageBox.information(self, "Saved", "All graphs exported.")

    def check_dependencies(self):
        """Notify if any required package is missing."""
        missing = []
        for pkg in ("Bio", "matplotlib", "PyQt5", "mplcursors"):
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg)
        if missing:
            resp = QMessageBox.question(
                self, "Missing Dependencies",
                "These packages are missing: " + ", ".join(missing) +
                "\nInstall now?",
                QMessageBox.Yes|QMessageBox.No
            )
            if resp == QMessageBox.Yes:
                import subprocess, sys
                subprocess.call([sys.executable, "-m", "pip", "install"] + missing)

    def on_chain_selected(self, text):
        """Switch to a batch‐imported sequence immediately."""
        for cid, seq, data in self.batch_data:
            if cid == text:
                self.seq_text.setPlainText(seq)
                self.analysis_data = data
                for sec,browser in self.report_section_tabs.items():
                    browser.setHtml(data["report_sections"][sec])
                self.update_graph_tabs()
                break

    def export_batch_csv(self):
        """Save batch data as CSV."""
        fn,_ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv)")
        if fn and self.batch_data:
            with open(fn, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "ID","Length","MW (Da)","Net Charge (pH 7)",
                    "% Hydro","% Hydrophil","% +Charged",
                    "% -Charged","% Neutral"
                ])
                for cid, seq, data in self.batch_data:
                    length = len(seq)
                    mw     = data["mol_weight"]
                    net7   = data["net_charge_7"]
                    hydro  = sum(1 for aa in seq if KYTE_DOOLITTLE[aa]>0)/length*100
                    hydrophil = 100 - hydro
                    pos    = sum(data["aa_counts"].get(k,0) for k in ("K","R","H"))/length*100
                    neg    = sum(data["aa_counts"].get(k,0) for k in ("D","E"))/length*100
                    neu    = 100 - (pos + neg)
                    writer.writerow([
                        cid,
                        str(length),
                        f"{mw:.2f}",
                        f"{net7:.2f}",
                        f"{hydro:.1f}%",
                        f"{hydrophil:.1f}%",
                        f"{pos:.1f}%",
                        f"{neg:.1f}%",
                        f"{neu:.1f}%"
                    ])
            QMessageBox.information(self, "Saved", "Batch CSV exported.")

    def export_batch_json(self):
        """Save batch data as JSON."""
        fn,_ = QFileDialog.getSaveFileName(self, "Save JSON", "", "JSON Files (*.json)")
        if fn and self.batch_data:
            out = []
            for cid, seq, data in self.batch_data:
                length = len(seq)
                mw     = data["mol_weight"]
                net7   = data["net_charge_7"]
                hydro  = sum(1 for aa in seq if KYTE_DOOLITTLE[aa]>0)/length*100
                hydrophil = 100 - hydro
                pos    = sum(data["aa_counts"].get(k,0) for k in ("K","R","H"))/length*100
                neg    = sum(data["aa_counts"].get(k,0) for k in ("D","E"))/length*100
                neu    = 100 - (pos + neg)
                out.append({
                    "ID": cid,
                    "Length": length,
                    "MW (Da)": f"{mw:.2f}",
                    "Net Charge (pH 7)": f"{net7:.2f}",
                    "% Hydro": f"{hydro:.1f}%",
                    "% Hydrophil": f"{hydrophil:.1f}%",
                    "% +Charged": f"{pos:.1f}%",
                    "% -Charged": f"{neg:.1f}%",
                    "% Neutral": f"{neu:.1f}%"
                })
            with open(fn, "w") as f:
                json.dump(out, f, indent=2)
            QMessageBox.information(self, "Saved", "Batch JSON exported.")

    def toggle_theme(self):
        """Switch between light/dark styles."""
        if self.theme_toggle.isChecked():
            self.setStyleSheet(DARK_THEME_CSS); plt.style.use("dark_background")
        else:
            self.setStyleSheet(LIGHT_THEME_CSS); plt.style.use("default")
        self.statusBar.showMessage(f"{'Dark' if self.theme_toggle.isChecked() else 'Light'} theme activated", 2000)

    def apply_settings(self):
        """Read settings from UI and re-render if needed."""
        try:
            self.default_window_size = int(self.window_size_input.text())
        except: pass
        try:
            self.default_pH = float(self.ph_input.text())
        except: pass
        self.show_bead_labels = self.label_checkbox.isChecked()
        self.colormap        = self.colormap_combo.currentText()
        try:
            self.label_font_size = int(self.label_font_input.text())
            self.tick_font_size  = int(self.tick_font_input.text())
            self.marker_size     = int(self.marker_size_input.text())
        except: pass
        self.graph_color = self.graph_color_combo.currentText()
        self.show_heading= self.heading_checkbox.isChecked()
        self.show_grid   = self.grid_checkbox.isChecked()
        self.show_grid   = self.grid_checkbox.isChecked()
        self.default_graph_format = self.graph_format_combo.currentText()
        self.use_reducing = self.reducing_checkbox.isChecked()
        AnalysisTools.use_reducing = self.use_reducing
        raw_pka = [p.strip() for p in self.pka_input.text().split(",") if p.strip()]
        self.custom_pka = None
        if len(raw_pka) == 9:
            try:
                vals = list(map(float, raw_pka))
                self.custom_pka = {
                    'NTERM': vals[0], 'CTERM': vals[1], 'D': vals[2], 'E': vals[3],
                    'C': vals[4], 'Y': vals[5], 'H': vals[6], 'K': vals[7], 'R': vals[8]
                }
            except ValueError:
                QMessageBox.warning(self, "pKa list",
                                    "Custom pKa list could not be parsed – using defaults.")

        calc_net_charge.pka_override = self.custom_pka

        self.enable_tooltips      = self.tooltips_checkbox.isChecked()

        if self.theme_toggle.isChecked():
            self.setStyleSheet(DARK_THEME_CSS)
        else:
            self.setStyleSheet(LIGHT_THEME_CSS)

        # disable tooltips globally if needed
        if not self.enable_tooltips:
            for w in QApplication.instance().allWidgets():
                w.setToolTip("")

        if self.analysis_data:
            # Refresh existing display
            for sec,browser in self.report_section_tabs.items():
                browser.setHtml(self.analysis_data["report_sections"][sec])
            self.update_graph_tabs()
        self.statusBar.showMessage("Settings applied", 2000)

    def reset_defaults(self):
        """Restore factory default settings."""
        self.window_size_input.setText("9")
        self.ph_input.setText("7.0")
        self.label_checkbox.setChecked(True)
        self.colormap_combo.setCurrentText("viridis")
        self.label_font_input.setText("14")
        self.tick_font_input.setText("12")
        self.marker_size_input.setText("10")
        self.graph_color_combo.setCurrentText("blue")
        self.graph_format_combo.setCurrentText("PNG")
        self.tooltips_checkbox.setChecked(False)
        self.heading_checkbox.setChecked(True)
        self.grid_checkbox.setChecked(True)
        self.apply_settings()

def main():
    app = QApplication(sys.argv)
    w   = ProteinAnalyzerGUI()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
