"""BEER main application window (PySide6)."""
from __future__ import annotations

import importlib
import importlib.util
import math, os, base64, json, csv, re, difflib, subprocess, sys
import urllib.request, urllib.error
import warnings

# All analysis modules are available in this refactored package
_HAS_AGGREGATION  = True
_HAS_AMPHIPATHIC  = True
_HAS_DISPROT      = True
_HAS_ELM          = True
_HAS_NEW_GRAPHS   = True
_HAS_PHASEPDB     = True
_HAS_PHI_PSI      = True
_HAS_PTM          = True
_HAS_RBP          = True
_HAS_SCD          = True

# Colourblind-safe accent colours (Paul Tol palette)
_CB_PALETTE = {
    "Tol Blue":   "#4477AA",
    "Tol Red":    "#CC6677",
    "Tol Green":  "#117733",
    "Tol Orange": "#DDAA33",
    "Tol Purple": "#AA4499",
    "Tol Cyan":   "#44AA99",
}

from io import BytesIO, StringIO

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT,
)
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.patches import FancyArrowPatch, Arc, Patch, Rectangle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
plt.style.use("default")
import mplcursors
import numpy as np
from collections import Counter

from Bio.SeqUtils.ProtParam import ProteinAnalysis as BPProteinAnalysis
from Bio import SeqIO
from Bio.PDB import PDBParser, PDBIO, PPBuilder
from Bio.PDB.Polypeptide import is_aa
from Bio.Blast import NCBIWWW, NCBIXML
from Bio.SeqUtils import seq1

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QTextBrowser,
    QFileDialog, QTabWidget, QMessageBox, QTableWidget, QTableWidgetItem,
    QCheckBox, QStatusBar, QComboBox, QFormLayout,
    QSplitter, QScrollArea, QFrame, QDialog, QDialogButtonBox,
    QSpinBox, QProgressDialog, QAbstractItemView,
    QListWidget, QListWidgetItem, QTreeWidget, QTreeWidgetItem, QStackedWidget,
    QInputDialog, QApplication, QDoubleSpinBox, QGroupBox, QMenu,
)
from PySide6.QtGui import QFont, QKeySequence, QAction, QShortcut, QImage
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtPrintSupport import QPrinter
try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    _WEBENGINE_AVAILABLE = True
except ImportError:
    _WEBENGINE_AVAILABLE = False

from beer.gui.themes import LIGHT_THEME_CSS, DARK_THEME_CSS
from beer import config as _config
from beer.gui.nav_widget import NavTabWidget
from beer.gui.dialogs import MutationDialog, _FigureComposerDialog
from beer.constants import (
    NAMED_COLORS, NAMED_COLORMAPS, GRAPH_TITLES, GRAPH_CATEGORIES,
    REPORT_SECTIONS, VALID_AMINO_ACIDS, _AA_COLOURS,
    KYTE_DOOLITTLE, DEFAULT_PKA, DISORDER_PROPENSITY, COILED_COIL_PROPENSITY,
    CHOU_FASMAN_HELIX, CHOU_FASMAN_SHEET, LINEAR_MOTIFS,
    STICKER_AROMATIC, STICKER_ELECTROSTATIC,
)
from beer.utils.sequence import clean_sequence, is_valid_protein, format_sequence_block
from beer.utils.biophysics import calc_net_charge
from beer.utils.pdb import import_pdb_sequence, extract_chain_structures, extract_phi_psi
from beer.analysis.core import AnalysisTools
from beer.analysis.aggregation import (
    calc_aggregation_profile, predict_aggregation_hotspots, calc_camsolmt_score,
)

# Alias used in graph generation block (original beer.py convention)
_extract_phi_psi = extract_phi_psi
from beer.network.workers import (
    AnalysisWorker, AlphaFoldWorker, PfamWorker, BlastWorker,
    ELMWorker, DisPRotWorker, PhaSepDBWorker,
)
from beer.graphs import (
    create_amino_acid_composition_figure, create_amino_acid_composition_pie_figure,
    create_hydrophobicity_figure, create_aggregation_profile_figure,
    create_solubility_profile_figure, create_scd_profile_figure,
    create_rbp_profile_figure, create_disorder_profile_figure,
    create_isoelectric_focus_figure,
    create_local_charge_figure, create_charge_decoration_figure,
    create_bead_model_hydrophobicity_figure, create_bead_model_charge_figure,
    create_helical_wheel_figure, create_tm_topology_figure,
    create_sticker_map_figure, create_hydrophobic_moment_figure,
    create_coiled_coil_profile_figure,
    create_linear_sequence_map_figure, create_ptm_profile_figure,
    create_domain_architecture_figure, create_cation_pi_map_figure,
    create_local_complexity_figure, create_ramachandran_figure,
    create_contact_network_figure, create_plddt_figure, create_distance_map_figure,
    create_msa_conservation_figure, create_complex_mw_figure,
    create_truncation_series_figure, create_pI_MW_gel_figure,
    create_saturation_mutagenesis_figure, create_uversky_phase_plot,
)
from beer.reports.css import REPORT_CSS
from beer.reports.sections import (
    format_aggregation_report, format_ptm_report, format_signal_report,
    format_amphipathic_report, format_scd_report, format_rbp_report,
    format_repeats_report,
)
from beer.io.export import ExportTools
from beer.io.session import save_session, load_session
from beer.embeddings.base import SequenceEmbedder


def _calc_batch_stats(seq: str, data: dict) -> tuple:
    """Return (hydro%, hydrophil%, pos%, neg%, neu%) for a sequence."""
    length = len(seq)
    hydro = sum(1 for aa in seq if KYTE_DOOLITTLE.get(aa, 0.0) > 0) / length * 100
    pos   = sum(data["aa_counts"].get(k, 0) for k in ("K", "R", "H")) / length * 100
    neg   = sum(data["aa_counts"].get(k, 0) for k in ("D", "E")) / length * 100
    neu   = 100 - (pos + neg)
    return hydro, 100 - hydro, pos, neg, neu

class ProteinAnalyzerGUI(QMainWindow):
    def __init__(self, embedder: "SequenceEmbedder | None" = None):
        super().__init__()
        self._embedder = embedder
        self.setWindowTitle("BEER - Biochemical Estimator & Explorer of Residues")
        self.resize(1200, 900)
        self.setStyleSheet(LIGHT_THEME_CSS)

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

        # State
        self.analysis_data       = None
        self.batch_data          = []
        _cfg = _config.load()
        self.default_window_size  = _cfg.get("window_size", 9)
        self.default_pH           = _cfg.get("ph", 7.0)
        self.use_reducing         = _cfg.get("use_reducing", False)
        self.custom_pka           = _cfg.get("custom_pka", None)
        self.colormap             = _cfg.get("colormap", "coolwarm")
        self.transparent_bg       = _cfg.get("transparent_bg", False)
        self.label_font_size      = _cfg.get("label_font_size", 14)
        self.tick_font_size       = _cfg.get("tick_font_size", 12)
        self.marker_size          = _cfg.get("marker_size", 10)
        self.show_bead_labels     = _cfg.get("show_bead_labels", True)
        self.graph_color          = NAMED_COLORS.get(_cfg.get("graph_color", "Royal Blue"), "#4361ee")
        self.show_heading         = _cfg.get("show_heading", True)
        self.show_grid            = _cfg.get("show_grid", True)
        self.default_graph_format = _cfg.get("graph_format", "PNG")
        self.app_font_size        = _cfg.get("app_font_size", 12)
        self.enable_tooltips      = _cfg.get("enable_tooltips", True)
        self.colorblind_safe      = _cfg.get("colorblind_safe", False)
        self._history             = _cfg.get("recent_sequences", [])
        self.sequence_name       = ""
        self._tooltips: dict     = {}
        self._analysis_worker    = None
        self._progress_dlg       = None

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

        self.setAcceptDrops(True)
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

        # Restore persisted theme
        if _cfg.get("theme_dark", False):
            self.theme_toggle.setChecked(True)
            self.setStyleSheet(DARK_THEME_CSS)
            plt.style.use("dark_background")

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
        canvas.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        canvas.customContextMenuRequested.connect(
            lambda pos, c=canvas: self._graph_context_menu(c, pos))
        vb.addWidget(NavigationToolbar2QT(canvas, self))
        vb.addWidget(canvas)
        btn = QPushButton("Save Graph")
        btn.clicked.connect(lambda _, t=title: self.save_graph(t))
        vb.addWidget(btn, alignment=Qt.AlignmentFlag.AlignRight)

    def _graph_context_menu(self, canvas, pos):
        menu = QMenu(self)
        copy_act = menu.addAction("Copy Figure to Clipboard")
        save_act = menu.addAction("Save Figure As\u2026")
        action = menu.exec(canvas.mapToGlobal(pos))
        if action == copy_act:
            buf = BytesIO()
            canvas.figure.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            img_data = buf.getvalue()
            qimg = QImage.fromData(img_data)
            QApplication.clipboard().setImage(qimg)
            self.statusBar.showMessage("Figure copied to clipboard.", 2000)
        elif action == save_act:
            ext  = self.default_graph_format.lower()
            fn, _ = QFileDialog.getSaveFileName(
                self, "Save Figure", "",
                f"{self.default_graph_format} Files (*.{ext})")
            if fn:
                if not fn.lower().endswith(f".{ext}"):
                    fn += f".{ext}"
                canvas.figure.savefig(fn, format=ext, dpi=200, bbox_inches="tight")
                self.statusBar.showMessage(f"Saved to {os.path.basename(fn)}", 2000)

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
            data = AnalysisTools.analyze_sequence(seq, 7.0, self.default_window_size, self.use_reducing, self.custom_pka, embedder=self._embedder)
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
        self.export_csv_btn = QPushButton("Export CSV")
        self.export_csv_btn.setToolTip("Export all scalar metrics to CSV")
        self.export_csv_btn.clicked.connect(self.export_metrics_csv)
        for w in (self.import_fasta_btn, self.import_pdb_btn, self.analyze_btn,
                  self.save_pdf_btn, self.mutate_btn,
                  self.session_save_btn, self.session_load_btn,
                  self.figure_composer_btn, self.export_csv_btn):
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
        splitter = QSplitter(Qt.Orientation.Horizontal)
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
        self.seq_text.setAcceptDrops(True)
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
        self.report_section_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        report_h.addWidget(self.report_section_list)

        rsep = QFrame()
        rsep.setFrameShape(QFrame.Shape.VLine)
        rsep.setFrameShadow(QFrame.Shadow.Plain)
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
        self.graph_tree.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.graph_tree.setIndentation(14)
        outer.addWidget(self.graph_tree)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFrameShadow(QFrame.Shadow.Plain)
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
        right_v.addWidget(save_all, alignment=Qt.AlignmentFlag.AlignRight)

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
            cat_item.setFlags(cat_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self.graph_tree.addTopLevelItem(cat_item)

            for title in titles:
                leaf = QTreeWidgetItem([f"  {title}"])
                leaf.setData(0, Qt.ItemDataRole.UserRole, title)
                cat_item.addChild(leaf)

                panel = QWidget()
                vb    = QVBoxLayout(panel)
                vb.setContentsMargins(4, 4, 4, 4)
                ph = QLabel(f"Run analysis to generate:\n{title}")
                ph.setAlignment(Qt.AlignmentFlag.AlignCenter)
                ph.setStyleSheet("color:#718096; font-style:italic;")
                vb.addWidget(ph)
                save_btn = QPushButton("Save Graph")
                save_btn.setMaximumWidth(120)
                save_btn.clicked.connect(lambda _, t=title: self.save_graph(t))
                vb.addWidget(save_btn, alignment=Qt.AlignmentFlag.AlignRight)

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
                "PySide6-WebEngine is not installed.\n"
                "Install it with:  pip install PySide6-WebEngine\n\n"
                "You can still save the PDB file and open it in PyMOL, UCSF ChimeraX, or 3Dmol.csb.pitt.edu."
            )
            msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
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
        self.blast_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.blast_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.blast_table, 1)

    def init_comparison_tab(self):
        container = QWidget()
        layout    = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        self.main_tabs.addTab(container, "Compare")

        # Two sequence inputs side by side
        inputs = QSplitter(Qt.Orientation.Horizontal)
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
        layout.addWidget(cmp_btn, alignment=Qt.AlignmentFlag.AlignLeft)

        self.compare_table = QTableWidget()
        self.compare_table.setAlternatingRowColors(True)
        self.compare_table.setColumnCount(3)
        self.compare_table.setHorizontalHeaderLabels(["Property", "Sequence A", "Sequence B"])
        self.compare_table.horizontalHeader().setStretchLastSection(True)
        self.compare_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
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
        self.batch_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.batch_table.cellDoubleClicked.connect(self.show_batch_details)
        layout.addWidget(self.batch_table, 1)

    def init_settings_tab(self):
        container = QWidget()
        self.main_tabs.addTab(container, "Settings")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
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
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

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
        form2.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
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
        form3.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
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
        form4.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
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

        self.colorblind_cb = QCheckBox("Colourblind-safe palette (Paul Tol)")
        self.colorblind_cb.setChecked(getattr(self, "colorblind_safe", False))
        self._set_tooltip(self.colorblind_cb,
                          "Use the Paul Tol colourblind-safe palette for all graphs.")
        form4.addRow("", self.colorblind_cb)
        layout.addLayout(form4)

        # ── ESM2 model selector ───────────────────────────────────────────
        form5 = QFormLayout()
        form5.setHorizontalSpacing(20)
        form5.setVerticalSpacing(8)
        form5.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        _section("ESM2 Embeddings (optional)")
        self.esm2_combo = QComboBox()
        self.esm2_combo.addItems([
            "esm2_t6_8M_UR50D",
            "esm2_t12_35M_UR50D",
            "esm2_t30_150M_UR50D",
            "esm2_t33_650M_UR50D",
        ])
        self.esm2_combo.setCurrentText("esm2_t6_8M_UR50D")
        self._set_tooltip(
            self.esm2_combo,
            "ESM2 model for per-residue embeddings (disorder, aggregation). "
            "Requires 'pip install fair-esm torch'. Larger models are more accurate but slower."
        )
        form5.addRow("ESM2 model:", self.esm2_combo)
        layout.addLayout(form5)

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
        help_nav.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        help_h.addWidget(help_nav)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFrameShadow(QFrame.Shadow.Plain); sep.setObjectName("nav_sep")
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
<h2>Secondary Structure</h2>
<p>Secondary structure prediction uses ESM2 embeddings for per-residue propensity scoring.
The per-residue disorder score is an ESM2-augmented propensity (0 = ordered, 1 = disordered).</p>
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
<p>LARKS are short amyloid-like segments associated with condensate formation (see <b>Phase Separation</b> help page for details). The Uversky phase plot and coiled-coil profile are available in the Graphs section.</p>
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
<pre>pip install PySide6-WebEngine</pre>
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
  <li><b>Linear Sequence Map</b> — four-track overview: hydrophobicity, NCPR, disorder, helix propensity.</li>
</ul>
<h2>Charge &amp; π-Interactions</h2>
<ul>
  <li><b>Isoelectric Focus</b> — Henderson-Hasselbalch charge curve 0–14; pI, physiological pH 7.4 charge, and positive/negative regions marked.</li>
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

        # Citation + Methods toolbar
        cite_bar = QHBoxLayout()

        cite_btn = QPushButton("Copy Citation (BibTeX)")
        cite_btn.setMinimumHeight(32)
        cite_btn.setToolTip("Copy BibTeX citation for BEER to clipboard")
        cite_btn.clicked.connect(self._copy_beer_citation)
        cite_bar.addWidget(cite_btn)

        methods_btn = QPushButton("Generate Methods Paragraph")
        methods_btn.setMinimumHeight(32)
        methods_btn.setToolTip("Auto-generate a methods paragraph for your paper")
        methods_btn.clicked.connect(self._generate_methods)
        cite_bar.addWidget(methods_btn)

        cite_bar.addStretch()
        outer_v.addLayout(cite_bar)

    def _copy_beer_citation(self):
        bibtex = """@article{mukherjee2025beer,
  title   = {BEER: A Desktop Application for Comprehensive Biophysical Analysis
             of Protein Sequences with ESM2-Augmented Predictors},
  author  = {Mukherjee, Saumyak},
  journal = {J. Chem. Inf. Model.},
  year    = {2025},
  note    = {Software article},
}"""
        QApplication.clipboard().setText(bibtex)
        self.statusBar.showMessage("BibTeX citation copied to clipboard.", 3000)

    def _generate_methods(self):
        """Auto-generate a methods paragraph based on current analysis settings."""
        if not self.analysis_data:
            QMessageBox.information(self, "Methods Generator",
                "Run an analysis first to generate a methods paragraph.")
            return
        seq   = self.analysis_data.get("seq", "")
        n_aa  = len(seq)
        esm2  = getattr(self._embedder, "model_name", None)
        esm2_str = (f" ESM2 embeddings (model: {esm2}) were used to augment "
                    f"disorder, aggregation, signal peptide, and PTM predictions.")  \
                   if esm2 else ""

        paragraph = (
            f"Biophysical sequence analysis was performed using BEER "
            f"(Biophysics Estimation and Evaluation Resource), a Python desktop "
            f"application developed for comprehensive protein characterisation. "
            f"The sequence ({n_aa} amino acids) was analysed to compute "
            f"molecular weight, isoelectric point (pI), net charge, GRAVY score, "
            f"fraction of charged residues (FCR), net charge per residue (NCPR), "
            f"charge patterning parameters \u03ba and \u03a9, sequence charge decoration (SCD), "
            f"intrinsic disorder profile, aggregation propensity, RNA-binding potential, "
            f"signal peptide probability, post-translational modification (PTM) sites, "
            f"sticker-spacer architecture, LARKS motifs, amphipathic helix detection, "
            f"tandem and compositional repeats, and hydrophobic moment.{esm2_str} "
            f"Analysis parameters: sliding window size = {self.default_window_size}, "
            f"pH = {self.default_pH:.1f}."
        )

        dlg = QDialog(self)
        dlg.setWindowTitle("Generated Methods Paragraph")
        dlg.resize(680, 300)
        vb = QVBoxLayout(dlg)
        lbl = QLabel("Copy and adapt this paragraph for your manuscript:")
        lbl.setWordWrap(True)
        vb.addWidget(lbl)
        te = QTextEdit()
        te.setPlainText(paragraph)
        te.setReadOnly(False)
        vb.addWidget(te)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        copy_btn2 = QPushButton("Copy to Clipboard")
        copy_btn2.clicked.connect(lambda: QApplication.clipboard().setText(te.toPlainText()))
        btns.addButton(copy_btn2, QDialogButtonBox.ButtonRole.ActionRole)
        btns.rejected.connect(dlg.reject)
        vb.addWidget(btns)
        dlg.exec()

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
        fig = create_amino_acid_composition_figure(
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

        self._progress_dlg = QProgressDialog("Running analysis…", "Cancel", 0, 0, self)
        self._progress_dlg.setWindowTitle("BEER Analysis")
        self._progress_dlg.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress_dlg.setMinimumDuration(500)
        self._progress_dlg.canceled.connect(self._cancel_analysis)
        self._progress_dlg.show()

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
            "Amino Acid Composition (Bar)": create_amino_acid_composition_figure(
                self.analysis_data["aa_counts"], self.analysis_data["aa_freq"],
                label_font=lf, tick_font=tf),
            "Amino Acid Composition (Pie)": create_amino_acid_composition_pie_figure(
                self.analysis_data["aa_counts"], label_font=lf),
            "Hydrophobicity Profile": create_hydrophobicity_figure(
                self.analysis_data["hydro_profile"], self.analysis_data["window_size"],
                label_font=lf, tick_font=tf),
            "Bead Model (Hydrophobicity)": create_bead_model_hydrophobicity_figure(
                seq, self.show_bead_labels, label_font=lf, tick_font=tf, cmap=self.colormap),
            "Bead Model (Charge)": create_bead_model_charge_figure(
                seq, self.show_bead_labels, label_font=lf, tick_font=tf),
            "Sticker Map": create_sticker_map_figure(
                seq, self.show_bead_labels, label_font=lf, tick_font=tf),
            "Local Charge Profile": create_local_charge_figure(
                self.analysis_data["ncpr_profile"], self.analysis_data["window_size"],
                label_font=lf, tick_font=tf),
            "Local Complexity": create_local_complexity_figure(
                self.analysis_data["entropy_profile"], self.analysis_data["window_size"],
                label_font=lf, tick_font=tf),
            "Cation\u2013\u03c0 Map": create_cation_pi_map_figure(
                seq, label_font=lf, tick_font=tf),
            "Isoelectric Focus": create_isoelectric_focus_figure(
                seq, label_font=lf, tick_font=tf, pka=self.custom_pka),
            "Helical Wheel": create_helical_wheel_figure(
                seq, label_font=lf),
            "Charge Decoration": create_charge_decoration_figure(
                self.analysis_data["fcr"], self.analysis_data["ncpr"],
                label_font=lf, tick_font=tf),
            "Linear Sequence Map": create_linear_sequence_map_figure(
                seq,
                self.analysis_data["hydro_profile"],
                self.analysis_data["ncpr_profile"],
                self.analysis_data["disorder_scores"],
                label_font=lf, tick_font=tf),
            "Disorder Profile": create_disorder_profile_figure(
                self.analysis_data["disorder_scores"], label_font=lf, tick_font=tf),
        }

        # TM Topology is always available after analysis
        figs["TM Topology"] = create_tm_topology_figure(
            seq, self.analysis_data.get("tm_helices", []),
            label_font=lf, tick_font=tf)

        # Phase separation / IDP graphs
        figs["Uversky Phase Plot"] = create_uversky_phase_plot(
            seq, label_font=lf, tick_font=tf)
        if self.analysis_data.get("cc_profile"):
            figs["Coiled-Coil Profile"] = create_coiled_coil_profile_figure(
                self.analysis_data["cc_profile"], label_font=lf, tick_font=tf)
        figs["Saturation Mutagenesis"] = create_saturation_mutagenesis_figure(
            seq, label_font=lf, tick_font=tf)

        # Structure-dependent graphs — available from any structure source
        # (PDB upload, RCSB PDB fetch, or AlphaFold fetch).
        if self.alphafold_data:
            plddt = self.alphafold_data.get("plddt")
            if plddt and len(plddt) == len(seq):
                figs["pLDDT Profile"] = create_plddt_figure(
                    plddt, label_font=lf, tick_font=tf)
            dm = self.alphafold_data.get("dist_matrix")
            # Guard: dist_matrix must be square and match the analysed sequence length.
            # A mismatch occurs when the FASTA and PDB chain have different residue
            # counts (e.g. disordered tails absent from the structure).
            if dm is not None and dm.ndim == 2 and dm.shape[0] == len(seq) and dm.shape[0] > 0:
                figs["Distance Map"] = create_distance_map_figure(
                    dm, label_font=lf, tick_font=tf)

        # Domain architecture — always rendered; shows all available tracks
        figs["Domain Architecture"] = create_domain_architecture_figure(
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
            figs["\u03b2-Aggregation Profile"] = create_aggregation_profile_figure(
                seq, aggr_profile, hotspots, label_font=lf, tick_font=tf)
            camsolmt = calc_camsolmt_score(seq)
            figs["Solubility Profile"] = create_solubility_profile_figure(
                seq, camsolmt, label_font=lf, tick_font=tf)

        if _HAS_AMPHIPATHIC:
            figs["Hydrophobic Moment"] = create_hydrophobic_moment_figure(
                seq,
                self.analysis_data.get("moment_alpha", []),
                self.analysis_data.get("moment_beta", []),
                self.analysis_data.get("amph_regions", []),
                label_font=lf, tick_font=tf)

        if _HAS_PTM:
            figs["PTM Map"] = create_ptm_profile_figure(
                seq, self.analysis_data.get("ptm_sites", []),
                label_font=lf, tick_font=tf)

        if _HAS_RBP:
            figs["RNA-Binding Profile"] = create_rbp_profile_figure(
                seq,
                self.analysis_data.get("rbp_profile", []),
                self.analysis_data.get("rbp", {}).get("motifs_found", []),
                label_font=lf, tick_font=tf)

        if _HAS_SCD:
            figs["SCD Profile"] = create_scd_profile_figure(
                seq, self.analysis_data.get("scd_profile", []),
                window=20, label_font=lf, tick_font=tf)

        # pI / MW Map — always available
        figs["pI / MW Map"] = create_pI_MW_gel_figure(
            [{"name": self.sequence_name or "Protein",
              "pI":   self.analysis_data["iso_point"],
              "mol_weight": self.analysis_data["mol_weight"]}],
            label_font=lf, tick_font=tf)

        # Ramachandran plot — any loaded PDB structure (upload, RCSB fetch, AlphaFold)
        if self.alphafold_data and _HAS_PHI_PSI:
            phi_psi = _extract_phi_psi(self.alphafold_data["pdb_str"])
            figs["Ramachandran Plot"] = create_ramachandran_figure(
                phi_psi, label_font=lf, tick_font=tf)

        # Residue contact network — requires distance matrix matching sequence length
        if self.alphafold_data:
            dm = self.alphafold_data.get("dist_matrix")
            if (dm is not None and dm.ndim == 2
                    and dm.shape[0] == len(seq) and dm.shape[0] > 0):
                figs["Residue Contact Network"] = create_contact_network_figure(
                    seq, dm, label_font=lf, tick_font=tf)

        # MSA Conservation (requires MSA data)
        if self._msa_sequences:
            figs["MSA Conservation"] = create_msa_conservation_figure(
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
        title = item.data(0, Qt.ItemDataRole.UserRole)
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

    def export_metrics_csv(self):
        """Export all scalar metrics from current analysis to CSV."""
        if not self.analysis_data:
            QMessageBox.warning(self, "Export", "Run analysis first.")
            return
        fn, _ = QFileDialog.getSaveFileName(
            self, "Export Metrics CSV",
            f"{self.sequence_name or 'metrics'}.csv",
            "CSV Files (*.csv)"
        )
        if not fn:
            return
        d = self.analysis_data
        rows = [
            ("Sequence Name",       self.sequence_name or ""),
            ("Length (aa)",         len(d.get("seq", ""))),
            ("Molecular Weight (Da)", f"{d.get('mol_weight', 0):.2f}"),
            ("Isoelectric Point",   f"{d.get('iso_point', 0):.2f}"),
            ("GRAVY",               f"{d.get('gravy', 0):.3f}"),
            ("Net Charge (pH 7)",   f"{d.get('net_charge_7', 0):.2f}"),
            ("FCR",                 f"{d.get('fcr', 0):.3f}"),
            ("NCPR",                f"{d.get('ncpr', 0):+.3f}"),
            ("Aromaticity",         f"{d.get('aromaticity', 0):.3f}"),
            ("Extinction Coeff. (reduced)", d.get('extinction', ('',''))[0] if isinstance(d.get('extinction'), tuple) else d.get('extinction', '')),
            ("Extinction Coeff. (non-reduced)", d.get('extinction', ('',''))[1] if isinstance(d.get('extinction'), tuple) else ''),
            ("Kappa (\u03ba)",      f"{d.get('kappa', 0):.4f}"),
            ("Omega (\u03a9)",      f"{d.get('omega', 0):.4f}"),
            ("SCD",                 f"{d.get('scd', 0):.3f}"),
            ("Fraction Disorder",   f"{d.get('disorder_f', 0):.3f}"),
            ("% Aggregation-prone", f"{d.get('solub_stats', {}).get('pct_aggregation_prone', 0):.1f}"),
            ("RNA-binding propensity", f"{d.get('rbp', {}).get('mean_propensity', 0):.3f}"),
            ("Signal peptide score", f"{d.get('sp_result', {}).get('score', 0):.3f}"),
        ]
        try:
            with open(fn, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Metric", "Value"])
                writer.writerows(rows)
            self.statusBar.showMessage(f"Metrics exported to {os.path.basename(fn)}", 3000)
        except OSError as e:
            QMessageBox.critical(self, "Export Error", str(e))

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
        # Propagate accent colour to graph style module
        import beer.graphs._style as _gstyle
        _gstyle._ACCENT  = self.graph_color
        _gstyle._FILL    = self.graph_color
        _gstyle._POS_COL = self.graph_color
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

        self.colorblind_safe = self.colorblind_cb.isChecked()
        if self.colorblind_safe:
            import beer.graphs._style as _gstyle
            _gstyle._ACCENT  = "#4477AA"
            _gstyle._FILL    = "#4477AA"
            _gstyle._POS_COL = "#4477AA"
            _gstyle._NEG_COL = "#CC6677"

        # Re-initialise ESM2 embedder if model changed
        new_esm2_model = self.esm2_combo.currentText()
        current_model = getattr(self._embedder, "model_name", None)
        if new_esm2_model != current_model:
            try:
                from beer.embeddings import get_embedder
                self._embedder = get_embedder(new_esm2_model)
            except Exception:
                pass

        if self.theme_toggle.isChecked():
            self.setStyleSheet(DARK_THEME_CSS)
        else:
            self.setStyleSheet(LIGHT_THEME_CSS)

        if self.analysis_data:
            for sec, browser in self.report_section_tabs.items():
                browser.setHtml(self.analysis_data["report_sections"][sec])
            self._update_seq_viewer()
            self.update_graph_tabs()

        # Persist settings to disk
        _config.save({
            "theme_dark":       self.theme_toggle.isChecked(),
            "window_size":      self.default_window_size,
            "ph":               self.default_pH,
            "use_reducing":     self.use_reducing,
            "custom_pka":       self.custom_pka,
            "colormap":         self.colormap,
            "graph_color":      self.graph_color_combo.currentText(),
            "label_font_size":  self.label_font_size,
            "tick_font_size":   self.tick_font_size,
            "marker_size":      self.marker_size,
            "show_bead_labels": self.show_bead_labels,
            "transparent_bg":   self.transparent_bg,
            "show_heading":     self.show_heading,
            "show_grid":        self.show_grid,
            "graph_format":     self.default_graph_format,
            "app_font_size":    self.app_font_size,
            "enable_tooltips":  self.enable_tooltips,
            "colorblind_safe":  getattr(self, "colorblind_safe", False),
            "esm2_model":       self.esm2_combo.currentText(),
            "recent_sequences": self._history,
        })
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
        self.colorblind_cb.setChecked(False)
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

    # --- Drag and drop ---

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() or event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event):
        md = event.mimeData()
        if md.hasUrls():
            for url in md.urls():
                fpath = url.toLocalFile()
                if fpath and os.path.isfile(fpath):
                    try:
                        with open(fpath, "r", errors="ignore") as fh:
                            raw = fh.read()
                        entries = self._parse_pasted_text(raw)
                        if entries:
                            if len(entries) == 1:
                                rid, seq = entries[0]
                                self.seq_text.setPlainText(seq)
                                self.sequence_name = rid
                                self.on_analyze()
                            else:
                                self._load_batch(entries)
                            event.acceptProposedAction()
                            return
                    except Exception:
                        pass
        elif md.hasText():
            raw = md.text()
            entries = self._parse_pasted_text(raw)
            if entries:
                rid, seq = entries[0]
                self.seq_text.setPlainText(seq)
                self.sequence_name = rid
                self.on_analyze()
                event.acceptProposedAction()
                return
        super().dropEvent(event)

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
        QShortcut(QKeySequence("Ctrl+/"),      self, self.show_shortcuts)

    def show_shortcuts(self):
        """Show keyboard shortcut reference overlay."""
        shortcuts = [
            ("Ctrl+Return", "Analyze sequence"),
            ("Ctrl+E",      "Export PDF report"),
            ("Ctrl+G",      "Switch to Graphs tab"),
            ("Ctrl+S",      "Save session"),
            ("Ctrl+O",      "Load session"),
            ("Ctrl+F",      "Focus motif search"),
            ("Ctrl+/",      "Show this shortcut reference"),
        ]
        lines = ["<h2>Keyboard Shortcuts</h2><table style='border-collapse:collapse;'>",
                 "<tr><th style='padding:4px 16px 4px 4px;text-align:left;'>Shortcut</th>"
                 "<th style='padding:4px;text-align:left;'>Action</th></tr>"]
        for key, action in shortcuts:
            lines.append(f"<tr><td style='padding:4px 16px 4px 4px;'>"
                         f"<kbd style='background:#f0f0f0;border:1px solid #ccc;"
                         f"border-radius:3px;padding:2px 6px;font-family:monospace;'>"
                         f"{key}</kbd></td>"
                         f"<td style='padding:4px;'>{action}</td></tr>")
        lines.append("</table>")
        dlg = QDialog(self)
        dlg.setWindowTitle("Keyboard Shortcuts")
        dlg.resize(420, 320)
        vb = QVBoxLayout(dlg)
        br = QTextBrowser()
        br.setHtml("".join(lines))
        vb.addWidget(br)
        btn = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btn.rejected.connect(dlg.reject)
        vb.addWidget(btn)
        dlg.exec()

    # --- Worker callbacks ---

    def _on_worker_finished(self, data: dict):
        if hasattr(self, "_progress_dlg") and self._progress_dlg:
            self._progress_dlg.close()
            self._progress_dlg = None
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
        if hasattr(self, "_progress_dlg") and self._progress_dlg:
            self._progress_dlg.close()
            self._progress_dlg = None
        self.analyze_btn.setEnabled(True)
        self.statusBar.showMessage("Analysis failed", 3000)
        QMessageBox.critical(self, "Analysis Error", msg)

    def _cancel_analysis(self):
        if self._analysis_worker and self._analysis_worker.isRunning():
            self._analysis_worker.terminate()
        self.analyze_btn.setEnabled(True)
        self.statusBar.showMessage("Analysis cancelled.", 2000)

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
        _config.set_value("recent_sequences", [(n, s) for n, s in self._history])

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
        if dlg.exec() != QDialog.Accepted:
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
            ("Aromaticity",            f"{da['aromaticity']:.3f}", f"{db['aromaticity']:.3f}"),
            ("Extinction Coeff.",      str(da['extinction']),     str(db['extinction'])),
        ]
        self.compare_table.setRowCount(len(props))
        for row, (prop, va, vb) in enumerate(props):
            self.compare_table.setItem(row, 0, QTableWidgetItem(prop))
            self.compare_table.setItem(row, 1, QTableWidgetItem(va))
            self.compare_table.setItem(row, 2, QTableWidgetItem(vb))
        self.compare_table.resizeColumnsToContents()

        # Profile overlays: disorder, hydrophobicity, aggregation side-by-side
        try:
            dis_a = da.get("disorder_scores", [])
            dis_b = db.get("disorder_scores", [])
            hyd_a = da.get("hydro_profile", [])
            hyd_b = db.get("hydro_profile", [])
            agg_a = da.get("aggr_profile_esm2", [])
            agg_b = db.get("aggr_profile_esm2", [])

            name_a = entries_a[0][0][:20] if entries_a[0][0] else "Sequence A"
            name_b = entries_b[0][0][:20] if entries_b[0][0] else "Sequence B"

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            fig.patch.set_facecolor("white")

            # Disorder overlay
            ax = axes[0]
            if dis_a:
                ax.plot(range(1, len(dis_a)+1), dis_a, label=name_a, color="#4361ee", linewidth=1.5)
            if dis_b:
                ax.plot(range(1, len(dis_b)+1), dis_b, label=name_b, color="#e63946", linewidth=1.5, linestyle="--")
            ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
            ax.set_xlabel("Residue"); ax.set_ylabel("Disorder score"); ax.set_title("Disorder")
            ax.legend(fontsize=9); ax.set_ylim(0, 1)

            # Hydrophobicity overlay
            ax = axes[1]
            if hyd_a:
                ax.plot(range(1, len(hyd_a)+1), hyd_a, label=name_a, color="#4361ee", linewidth=1.5)
            if hyd_b:
                ax.plot(range(1, len(hyd_b)+1), hyd_b, label=name_b, color="#e63946", linewidth=1.5, linestyle="--")
            ax.set_xlabel("Residue"); ax.set_ylabel("Hydrophobicity"); ax.set_title("Hydrophobicity")
            ax.legend(fontsize=9)

            # Aggregation overlay
            ax = axes[2]
            if agg_a:
                ax.plot(range(1, len(agg_a)+1), agg_a, label=name_a, color="#4361ee", linewidth=1.5)
            if agg_b:
                ax.plot(range(1, len(agg_b)+1), agg_b, label=name_b, color="#e63946", linewidth=1.5, linestyle="--")
            ax.set_xlabel("Residue"); ax.set_ylabel("Aggregation score"); ax.set_title("Aggregation")
            ax.legend(fontsize=9)

            fig.tight_layout(pad=2.0)
            self._replace_graph("Comparison Profiles", fig)
        except Exception:
            pass

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
        self.trunc_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
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

        splitter = QSplitter(Qt.Orientation.Horizontal)
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

        splitter = QSplitter(Qt.Orientation.Horizontal)
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
                    d = AnalysisTools.analyze_sequence(trunc_seq, self.default_pH, self.default_window_size, self.use_reducing, self.custom_pka, embedder=self._embedder)
                    rows.append(("N-term", pct, len(trunc_seq), d))
            if do_c:
                trunc_seq = seq[:n_rem]
                if is_valid_protein(trunc_seq) and len(trunc_seq) >= 5:
                    d = AnalysisTools.analyze_sequence(trunc_seq, self.default_pH, self.default_window_size, self.use_reducing, self.custom_pka, embedder=self._embedder)
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
            fig = create_truncation_series_figure(
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
            fig = create_msa_conservation_figure(
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
            fig = create_complex_mw_figure(
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
            dlg.exec()
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
            dlg.exec()
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
            dlg.exec()
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
        if dlg.exec() != QDialog.Accepted:
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
        for pkg in ("Bio", "matplotlib", "PySide6", "mplcursors"):
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg)
        if missing:
            resp = QMessageBox.question(
                self, "Missing Dependencies",
                "These packages are missing: " + ", ".join(missing) + "\nInstall now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if resp == QMessageBox.StandardButton.Yes:
                result = subprocess.call([sys.executable, "-m", "pip", "install"] + missing)
                if result != 0:
                    QMessageBox.warning(self, "Install Failed",
                                        "Some packages could not be installed.")
        if not _WEBENGINE_AVAILABLE:
            self.statusBar.showMessage(
                "Tip: install PyQtWebEngine (pip install PySide6-WebEngine) for the 3D structure viewer.",
                8000
            )


