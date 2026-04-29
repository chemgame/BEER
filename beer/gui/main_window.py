"""BEER main application window (PySide6)."""
from __future__ import annotations

import importlib
import importlib.util
import importlib.resources
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
    QInputDialog, QApplication, QDoubleSpinBox, QGroupBox, QMenu, QSlider,
    QColorDialog, QSizePolicy, QStyleFactory,
)
from PySide6.QtGui import QFont, QKeySequence, QAction, QShortcut, QImage, QIcon, QPixmap
from PySide6.QtCore import Qt, QThread, Signal, QObject, QEvent, QUrl
from PySide6.QtPrintSupport import QPrinter
try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    _WEBENGINE_AVAILABLE = True
except ImportError:
    _WEBENGINE_AVAILABLE = False

from beer.gui.themes import LIGHT_THEME_CSS, DARK_THEME_CSS
from beer import config as _config
from beer.gui.nav_widget import NavTabWidget
from beer.gui.dialogs import MutationDialog, _FigureComposerDialog, FormatChooserDialog
from beer.io.structure_formats import pdb_to_mmcif, pdb_to_gro, pdb_to_xyz
from beer.io.graph_data_export import get_graph_data
from beer.constants import (
    NAMED_COLORS, NAMED_COLORMAPS, GRAPH_TITLES, GRAPH_CATEGORIES,
    REPORT_SECTIONS, VALID_AMINO_ACIDS, _AA_COLOURS, BILSTM_PROFILE_TABS,
    KYTE_DOOLITTLE, DEFAULT_PKA, DISORDER_PROPENSITY,
    CHOU_FASMAN_HELIX, CHOU_FASMAN_SHEET, LINEAR_MOTIFS,
    STICKER_AROMATIC, STICKER_ELECTROSTATIC,
    HYDROPHOBICITY_SCALES,
)
from beer.utils.sequence import clean_sequence, is_valid_protein, format_sequence_block
from beer.utils.biophysics import calc_net_charge
from beer.utils.pdb import (
    import_pdb_sequence, extract_chain_structures, extract_phi_psi,
    import_mmcif_sequence, extract_chain_structures_mmcif,
)
from beer.analysis.core import AnalysisTools
from beer.analysis.aggregation import (
    calc_aggregation_profile, predict_aggregation_hotspots, calc_camsolmt_score,
)

# Alias used in graph generation block (original beer.py convention)
_extract_phi_psi = extract_phi_psi
from beer.network._http import fetch_uniprot_pdb_xrefs, fetch_rcsb_assembly_cif
from beer.network.workers import (
    AnalysisWorker, AlphaFoldWorker, PfamWorker, BlastWorker,
    ELMWorker, DisPRotWorker, PhaSepDBWorker,
    MobiDBWorker, UniProtVariantsWorker, IntActWorker,
    UniProtSequenceSearchWorker, UniProtFeaturesWorker,
    AISectionWorker,
)
from beer.graphs import (
    create_amino_acid_composition_figure,
    create_hydrophobicity_figure, create_aggregation_profile_figure,
    create_solubility_profile_figure, create_scd_profile_figure,
    create_rbp_profile_figure,
    create_isoelectric_focus_figure,
    create_local_charge_figure, create_charge_decoration_figure,
    create_helical_wheel_figure, create_tm_topology_figure,
    create_sticker_map_figure, create_hydrophobic_moment_figure,
    create_coiled_coil_profile_figure,
    create_linear_sequence_map_figure,
    create_domain_architecture_figure, create_cation_pi_map_figure,
    create_local_complexity_figure, create_ramachandran_figure,
    create_contact_network_figure, create_plddt_figure, create_sasa_figure,
    create_distance_map_figure, create_bead_model_ss_figure,
    create_msa_conservation_figure, create_complex_mw_figure,
    create_truncation_series_figure,
    create_saturation_mutagenesis_figure, create_uversky_phase_plot,
    create_annotation_track_figure, create_cleavage_map_figure,
    create_plaac_profile_figure,
    create_msa_covariance_figure,
    create_bilstm_profile_figure,
    create_bilstm_dual_track_figure,
    create_disorder_profile_figure,
    create_shd_profile_figure,
)
from beer.graphs.variant_map import create_alphafold_missense_figure
from beer.reports.css import REPORT_CSS, REPORT_CSS_DARK, get_report_css
from beer.reports.sections import (
    format_aggregation_report, format_signal_report,
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

# ---------------------------------------------------------------------------
# Report section help hints (module-level)
# ---------------------------------------------------------------------------
_SECTION_HINTS: dict = {
    "Composition":               "Amino-acid counts, frequencies, and physicochemical groupings.",
    "Properties":                "Global physicochemical parameters: MW, pI, GRAVY, aromaticity, extinction coefficient.",
    "Hydrophobicity":            "Kyte\u2013Doolittle hydropathy statistics and hydrophobic/hydrophilic fractions.",
    "Charge":                    "Net charge, FCR, NCPR, charge asymmetry (\u03ba/\u03a9) at the set pH.",
    "Aromatic & \u03c0":         "Aromatic fraction and cation\u2013\u03c0 / \u03c0\u2013\u03c0 pair counts.",
    "Low Complexity":            "Shannon entropy, prion-like score, LC fraction, PLAAC score, PolyX runs.",
    "Disorder":                  "AI Predictions disorder score (ESM2 650M → BiLSTM classifier, AUROC 0.9999); classical propensity fallback. Predicted IDR regions listed with residue ranges.",
    "Repeat Motifs":             "RGG/RG, FG, SR/RS, QN/NQ repeat motifs relevant to RNA-binding and phase separation.",
    "Sticker & Spacer":          "Aromatic/charged stickers and uncharged spacers \u2014 key determinants of condensate properties.",
    "TM Helices":                "Kyte\u2013Doolittle sliding-window TM helix prediction; inside-positive topology rule.",
    "LARKS":                     "Low-complexity Aromatic-Rich Kinked Segments \u2014 structural motifs associated with amyloid-like fibres (Hughes et al. 2018).",
    "Linear Motifs":             "Regex scan for 15+ short linear motifs: NLS, NES, PxxP, 14-3-3, KFERQ, KDEL, and more.",
    "\u03b2-Aggregation & Solubility": "ZYGGREGATOR \u03b2-aggregation hotspots and CamSol intrinsic solubility per residue.",
    "PTM Sites":                 "ESM2-predicted phosphorylation, ubiquitination, SUMOylation, glycosylation, and methylation sites.",
    "Signal Peptide & GPI":      "ESM2 signal-peptide probe (AUC 1.00); n/h/c-region annotation and GPI signal detection.",
    "Amphipathic Helices":       "Helical-wheel scan for amphipathic helices with hydrophobic moment \u2265 threshold.",
    "Charge Decoration (SCD)":   "Sequence Charge Decoration profile (Sawle & Ghosh 2015) and \u03ba/\u03a9 patterning metrics.",
    "RNA Binding":               "Per-residue RNA-binding propensity; RGG, RRM, KH, SR, DEAD-box, and zinc-finger motif hits.",
    "Tandem Repeats":            "Direct, tandem, and compositional repeat detection.",
    "Proteolytic Map":           "Predicted cleavage sites for 9 enzymes (Trypsin, Lys-C, Glu-C, Asp-N, Chymotrypsin, CNBr, Arg-C, Caspase-3, Proteinase K) with peptide masses.",
}

# Report section groups for collapsible tree (module-level)
_REPORT_SECTION_GROUPS: list = [
    ("Sequence Properties", ["Composition", "Properties", "Hydrophobicity", "Charge", "Aromatic & \u03c0"]),
    ("IDP & Phase Separation", ["Repeat Motifs", "Sticker & Spacer", "Charge Decoration (SCD)", "LARKS"]),
    ("Structure & Topology", ["Amphipathic Helices"]),
    ("Functional Sites", ["Linear Motifs", "Tandem Repeats", "Proteolytic Map",
                           "\u03b2-Aggregation & Solubility"]),
]

# ---------------------------------------------------------------------------

# AI Predictions head specs: (display_name, data_key, graph_title, auroc)
_AI_HEAD_SPECS: list[tuple[str, str, str, str]] = [
    ("Disorder",            "disorder_scores",          "Disorder Profile",       "0.9999"),
    ("Signal Peptide",      "sp_bilstm_profile",        "Signal Peptide Profile",      "0.9999"),
    ("Transmembrane",       "tm_bilstm_profile",        "Transmembrane Profile",       "0.992"),
    ("Intramembrane",       "intramem_bilstm_profile",  "Intramembrane Profile",       "—"),
    ("Coiled-Coil",         "cc_bilstm_profile",        "Coiled-Coil Profile",         "—"),
    ("DNA-Binding",         "dna_bilstm_profile",       "DNA-Binding Profile",         "0.998"),
    ("RNA Binding",         "rnabind_bilstm_profile",   "RNA Binding Profile",         "—"),
    ("Active Site",         "act_bilstm_profile",       "Active Site Profile",         "—"),
    ("Binding Site",        "bnd_bilstm_profile",       "Binding Site Profile",        "—"),
    ("Phosphorylation",     "phos_bilstm_profile",      "Phosphorylation Profile",     "—"),
    ("Low-Complexity",      "lcd_bilstm_profile",       "Low-Complexity Profile",      "—"),
    ("Zinc Finger",         "znf_bilstm_profile",       "Zinc Finger Profile",         "—"),
    ("Glycosylation",       "glyc_bilstm_profile",      "Glycosylation Profile",       "—"),
    ("Ubiquitination",      "ubiq_bilstm_profile",      "Ubiquitination Profile",      "—"),
    ("Methylation",         "meth_bilstm_profile",      "Methylation Profile",         "—"),
    ("Acetylation",         "acet_bilstm_profile",      "Acetylation Profile",         "—"),
    ("Lipidation",          "lipid_bilstm_profile",     "Lipidation Profile",          "—"),
    ("Disulfide Bond",      "disulf_bilstm_profile",    "Disulfide Bond Profile",      "—"),
    ("Functional Motif",    "motif_bilstm_profile",     "Functional Motif Profile",    "—"),
    ("Propeptide",          "prop_bilstm_profile",      "Propeptide Profile",          "—"),
    ("Repeat Region",       "rep_bilstm_profile",       "Repeat Region Profile",       "—"),
    ("Nucleotide-Binding",  "nucbind_bilstm_profile",   "Nucleotide-Binding Profile",  "—"),
    ("Transit Peptide",     "transit_bilstm_profile",   "Transit Peptide Profile",     "—"),
]

# graph_title → sec_key and data_key for lazy-trigger from the Graphs tab
_GRAPH_TITLE_TO_AI_SEC:      dict[str, str] = {gt: f"AI:{dn}" for dn, dk, gt, _ in _AI_HEAD_SPECS}
_GRAPH_TITLE_TO_AI_DATA_KEY: dict[str, str] = {gt: dk        for dn, dk, gt, _ in _AI_HEAD_SPECS}
# display_name → graph_title (for invalidation after compute)
_AI_DISPLAY_TO_GRAPH_TITLE:  dict[str, str] = {dn: gt        for dn, dk, gt, _ in _AI_HEAD_SPECS}
# display_name → _FEATURE_SCORE_KEYS label (names differ in two places)
_AI_DISPLAY_TO_FEATURE_LABEL: dict[str, str] = {
    "Disorder":           "Disorder",
    "Signal Peptide":     "Signal Peptide",
    "Transmembrane":      "Transmembrane",
    "Intramembrane":      "Intramembrane",
    "Coiled-Coil":        "Coiled-Coil",
    "DNA-Binding":        "DNA-Binding",
    "Active Site":        "Active Site",
    "Binding Site":       "Binding Site",
    "Phosphorylation":    "Phosphorylation",
    "Low-Complexity":     "Low Complexity",
    "Zinc Finger":        "Zinc Finger",
    "Glycosylation":      "Glycosylation",
    "Ubiquitination":     "Ubiquitination",
    "Methylation":        "Methylation",
    "Acetylation":        "Acetylation",
    "Lipidation":         "Lipidation",
    "Disulfide Bond":     "Disulfide Bond",
    "Functional Motif":   "Functional Motif",
    "Propeptide":         "Propeptide",
    "Repeat Region":      "Repeat Region",
    "RNA Binding":        "RNA-Binding",
    "Nucleotide-Binding": "Nucleotide-Binding",
    "Transit Peptide":    "Transit Peptide",
}

# ---------------------------------------------------------------------------

def _make_hsep() -> "QFrame":
    sep = QFrame()
    sep.setFrameShape(QFrame.Shape.HLine)
    sep.setFrameShadow(QFrame.Shadow.Sunken)
    return sep

# ---------------------------------------------------------------------------

class _BeerLinkFilter(QObject):
    """Viewport event filter that intercepts beer:// anchor clicks in QTextBrowser.

    PySide6 does not reliably emit anchorClicked for custom URL schemes; reading
    the anchor directly on MouseButtonRelease is the guaranteed path.
    """
    def __init__(self, browser: "QTextBrowser", handler) -> None:
        super().__init__(browser)
        self._browser = browser
        self._handler = handler

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.Type.MouseButtonRelease:
            anchor = self._browser.anchorAt(event.pos())
            if anchor and anchor.startswith("beer://"):
                self._handler(QUrl(anchor))
                return True
        return False


def _install_beer_link_filter(browser: "QTextBrowser", handler) -> None:
    """Install _BeerLinkFilter on *browser*'s viewport and disable Qt link-following."""
    browser.setOpenLinks(False)
    f = _BeerLinkFilter(browser, handler)
    browser.viewport().installEventFilter(f)


# ---------------------------------------------------------------------------

from PySide6.QtCore import Slot as _Slot

class _StructBridge(QObject):
    """QObject exposed to the 3Dmol JS page via QWebChannel.
    Receives residue-click notifications from the structure viewer.
    """
    def __init__(self, main_window: "ProteinAnalyzerGUI"):
        super().__init__(main_window)
        self._mw = main_window

    @_Slot(int)
    def residueClicked(self, resi: int) -> None:
        self._mw._on_struct_residue_picked(resi)

# ---------------------------------------------------------------------------

class ProteinAnalyzerGUI(QMainWindow):
    def __init__(self, embedder: "SequenceEmbedder | None" = None):
        super().__init__()
        self._embedder = embedder
        self.setWindowTitle("BEER - Biophysical Evaluation Engine for Residues")
        self.resize(1200, 900)
        self._is_dark = False
        self.setStyleSheet(LIGHT_THEME_CSS)


        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

        # Permanent ESM2 status indicator (right side of status bar)
        self._esm2_indicator = QLabel()
        self._esm2_indicator.setContentsMargins(0, 0, 8, 0)
        self.statusBar.addPermanentWidget(self._esm2_indicator)
        self._update_esm2_indicator("ready")

        # State
        self.analysis_data       = None
        self.batch_data          = []
        _cfg = _config.load()
        self.default_window_size  = _cfg.get("window_size", 9)
        self.default_pH           = _cfg.get("ph", 7.0)
        self.use_reducing         = _cfg.get("use_reducing", False)
        self.custom_pka           = _cfg.get("custom_pka", None)
        self.colormap             = _cfg.get("colormap", "coolwarm")
        self.heatmap_cmap         = _cfg.get("heatmap_cmap", "viridis")
        self.transparent_bg       = _cfg.get("transparent_bg", False)
        self.label_font_size      = _cfg.get("label_font_size", 11)
        self.tick_font_size       = _cfg.get("tick_font_size", 9)
        self.marker_size          = _cfg.get("marker_size", 10)
        self.show_bead_labels     = _cfg.get("show_bead_labels", True)
        self.graph_color          = NAMED_COLORS.get(_cfg.get("graph_color", "Royal Blue"), "#4361ee")
        self.show_heading         = _cfg.get("show_heading", True)
        self.show_grid            = _cfg.get("show_grid", True)
        self.default_graph_format = _cfg.get("graph_format", "PNG")
        self.app_font_size        = _cfg.get("app_font_size", 12)
        self.enable_tooltips        = _cfg.get("enable_tooltips", True)
        self.colorblind_safe        = _cfg.get("colorblind_safe", False)
        self._esm2_missing_warned   = False   # show ESM2 missing notice at most once
        self._last_was_bilstm       = False   # True only after BiLSTM Analysis completes
        # Lazy AI section state
        self._ai_computed_sections: set[str] = set()   # "AI:<name>" keys that have real scores
        self._active_ai_worker: "AISectionWorker | None" = None  # at most one at a time
        self._history             = []   # session-only: never restored from disk
        self.hydro_scale          = _cfg.get("hydro_scale", "Kyte-Doolittle")
        self.sequence_name       = ""
        self._tooltips: dict     = {}
        self._analysis_worker    = None
        self._progress_dlg       = None
        self._pending_pdb        = None   # stored when loadPDB is called before page ready
        self._struct_pdb_str     = None   # raw PDB string of currently loaded structure
        self._struct_marker_resi = None   # residue last clicked in 3D viewer; re-applied on graph redraw
        self._struct_sasa_data   = {}     # {PDB resi: RSA 0..1} computed on PDB load
        self._struct_sasa_raw    = {}     # {PDB resi: ASA Å²} computed on PDB load
        self._sasa_show_asa      = False  # toggle state: False=RSA, True=ASA

        # --- New state for AlphaFold / Pfam / BLAST ---
        self.current_accession   = ""   # last successfully fetched UniProt accession
        self._source_id          = ""   # original fetch ID (UniProt acc or PDB ID) — survives rename
        self.alphafold_data      = None  # dict: pdb_str, plddt, dist_matrix, accession
        self._struct_is_alphafold = False # True only when structure came from AlphaFold DB
        self.batch_struct        = {}   # maps batch rec_id -> per-chain struct dict
        self.pfam_domains        = []   # list of domain dicts from Pfam
        self._alphafold_worker   = None
        self._pfam_worker        = None
        self._blast_worker       = None

        # --- New state for extended features ---
        self._blast_timer        = None
        self._blast_start_time   = None
        self._undo_seq           = None
        self._undo_name          = None
        self._elm_worker         = None
        self._disprot_worker     = None
        self._phasepdb_worker    = None
        self._mobidb_worker      = None
        self._variants_worker    = None
        self._intact_worker      = None
        self.intact_data         = {}   # IntAct interaction results
        self._msa_mi_apc         = None # APC-corrected MI matrix from last MSA run
        self.elm_data            = []   # list of ELM instances
        self.disprot_data        = {}   # DisProt disorder regions
        self.phasepdb_data       = {}   # PhaSepDB lookup result
        self.mobidb_data         = {}   # MobiDB disorder consensus
        self.variants_data       = []   # UniProt natural variants
        self._uniprot_features   = {}   # UniProtFeaturesWorker results (feature→regions)
        self._uniprot_feat_worker = None
        self._uniprot_card       = {}   # parsed UniProt entry metadata for Summary tab
        self._seq_search_worker  = None  # UniProtSequenceSearchWorker
        self._msa_sequences      = []   # list of aligned sequences
        self._msa_names          = []   # corresponding names
        self._plugins            = []   # loaded plugin modules
        # Lazy graph generation
        self._graph_generators: dict = {}   # title → callable that returns Figure
        self._generated_graphs: set  = set()  # titles already rendered this session
        self._load_plugins()

        self.setAcceptDrops(True)
        self.check_dependencies()
        self.main_tabs = NavTabWidget()
        self.setCentralWidget(self.main_tabs)
        self.init_analysis_tab()
        self.init_report_tab()
        self.init_summary_tab()
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
        self._apply_browser_palette()

    # --- Browser palette (ensures HTML viewport matches theme) ---

    def _apply_browser_palette(self) -> None:
        """Set QPalette Base/Text on every QTextBrowser so the HTML viewport
        matches the active theme regardless of OS dark-mode state.
        Also fixes the right_tabs (Report / Alanine Scan) pane background."""
        from PySide6.QtGui import QPalette, QColor
        is_dark = hasattr(self, "theme_toggle") and self.theme_toggle.isChecked()
        bg   = QColor("#16213e" if is_dark else "#ffffff")
        fg   = QColor("#e2e8f0" if is_dark else "#1a1a2e")
        browsers: list = list(self.findChildren(QTextBrowser))
        for br in browsers:
            pal = br.palette()
            pal.setColor(QPalette.ColorRole.Base,   bg)
            pal.setColor(QPalette.ColorRole.Text,   fg)
            pal.setColor(QPalette.ColorRole.Window, bg)
            br.setPalette(pal)
        # Fix the Report / Alanine Scan QTabWidget pane background explicitly
        if hasattr(self, "_right_tabs"):
            pal = self._right_tabs.palette()
            pal.setColor(QPalette.ColorRole.Window, bg)
            pal.setColor(QPalette.ColorRole.Base,   bg)
            self._right_tabs.setPalette(pal)
            self._right_tabs.setAutoFillBackground(True)
            for child in self._right_tabs.findChildren(QWidget):
                child_pal = child.palette()
                child_pal.setColor(QPalette.ColorRole.Window, bg)
                child_pal.setColor(QPalette.ColorRole.Base,   bg)
                child.setPalette(child_pal)
                child.setAutoFillBackground(True)

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

    _GRAPH_HINTS: dict = {
        "Amino Acid Composition (Bar)": (
            "Bar chart of raw amino acid counts.\n\n"
            "The 20 standard amino acids are plotted in alphabetical order. "
            "Counts are derived directly from the input sequence; no window averaging is applied.\n\n"
            "Interpretation: High Gly/Ala content suggests a flexible or disordered backbone. "
            "Enrichment in Arg/Lys indicates a positively charged protein; Asp/Glu enrichment indicates negative charge."
        ),
        "Amino Acid Composition (Pie)": (
            "Pie chart of amino acid frequencies (%).\n\n"
            "Residues with zero count are excluded. Colour coding is consistent with the bar chart.\n\n"
            "Tip: IDPs typically show depletion of order-promoting residues (C, W, F, Y, I, L, V) "
            "and enrichment of disorder-promoting residues (E, K, R, S, P, Q)."
        ),
        "Hydrophobicity Profile": None,  # dynamic — see _hydrophobicity_hint()
        "Local Charge Profile": (
            "Sliding-window mean net charge per residue (NCPR).\n\n"
            "NCPR = (f\u207a \u2212 f\u207b) averaged over a window, "
            "where f\u207a = fraction positive (K, R) and f\u207b = fraction negative (D, E).\n\n"
            "Positive values (blue) = net positive; negative values (red) = net negative. "
            "Useful for identifying charged patches, NLS signals, and polyampholyte regions."
        ),
        "Disorder Profile": (
            "Per-residue disorder probability from the BEER AI Predictions head (0 = ordered, 1 = disordered).\n\n"
            "Architecture: ESM2 650M embeddings → 2-layer BiLSTM classifier (hidden=256) → sigmoid. "
            "Trained on UniProt Swiss-Prot 'Disordered region' annotations (AUROC 0.9999 on held-out test set). "
            "Threshold at F1-max ≈ 0.5 (dashed line). MC-Dropout (20 passes) provides ±1σ uncertainty band.\n\n"
            "Regions consistently above the threshold are predicted intrinsically disordered (IDRs)."
        ),
        "Linear Sequence Map": (
            "Unified ruler view of all predicted sequence features.\n\n"
            "Tracks: transmembrane helices, signal peptide cleavage site, PTM positions "
            "(phospho, ubiq, SUMO, glyco, methyl), LARKS motifs, and hydrophobicity.\n\n"
            "Overlapping TM and PTM regions may indicate regulated membrane protein function."
        ),
        "Isoelectric Focus": (
            "Net charge vs pH titration curve.\n\n"
            "Charge computed via Henderson\u2013Hasselbalch:\n"
            "  Q(pH) = \u03a3 q\u1d62 / (1 + 10^(\u00b1(pH \u2212 pK\u2090\u1d62)))\n"
            "summed over all ionisable groups (N-term, C-term, D, E, H, C, Y, K, R).\n\n"
            "The pI is where Q = 0. Dashed line marks physiological pH 7.4. "
            "Default pK\u2090 values follow Lehninger (2005); custom values set in Settings."
        ),
        "Charge Decoration": (
            "Das\u2013Pappu phase diagram of sequence charge patterning.\n\n"
            "x-axis: FCR = f\u207a + f\u207b (fraction of charged residues)\n"
            "y-axis: NCPR = |f\u207a \u2212 f\u207b| (net charge per residue)\n\n"
            "Coloured regions: weak polyampholyte/polyelectrolyte (low FCR), "
            "strong polyampholyte (high FCR, low NCPR), strong polyelectrolyte (high NCPR).\n"
            "Reference: Das & Pappu, PNAS 110:13392, 2013."
        ),
        "Cation\u2013\u03c0 Map": (
            "Pairwise cation\u2013\u03c0 proximity map.\n\n"
            "Cation\u2013\u03c0 interactions occur between positively charged side chains (K, R) and "
            "aromatic rings (F, W, Y). Proximity score = 1/|i\u2212j| for pairs within 8 residues.\n\n"
            "Colorbar: Proximity 1/|i\u2212j| (higher = closer in sequence). "
            "Colourmap is the one selected in Settings.\n\n"
            "\u0394G \u2248 \u22121 to \u22123 kcal/mol. Enriched in phase-separating IDPs and RNA-binding proteins.\n"
            "Reference: Gallivan & Dougherty, PNAS 96:9459, 1999."
        ),
        "Sticker Map": (
            "Stickers-and-spacers model visualisation.\n\n"
            "Stickers (aromatic: F, W, Y, H) in magenta; charged residues (K/R blue; D/E red) "
            "on a sequence ruler.\n\n"
            "Sticker\u2013sticker interactions drive condensate formation; "
            "spacers modulate valency and phase boundaries.\n"
            "Reference: Mittag & Pappu, eLife 11:e75818, 2022."
        ),
        "Helical Wheel": (
            "Helical-wheel projection of the sequence.\n\n"
            "Residues projected with 100\u00b0/residue rotation (\u03b1-helix geometry). "
            "A hydrophobic face (yellow sector) indicates an amphipathic helix capable of "
            "membrane insertion or protein\u2013protein interaction.\n\n"
            "Reference: Schiffer & Edmundson, Biophys. J. 7:121, 1967."
        ),
        "TM Topology": (
            "Transmembrane topology prediction.\n\n"
            "KD sliding-window profile overlaid with threshold (KD \u2265 1.8). "
            "If DeepTMHMM has been run, topology is annotated using the positive-inside rule "
            "(von Heijne, EMBO J. 5:3021, 1986).\n\n"
            "DeepTMHMM reference: Hallgren et al., Nat. Methods 2022."
        ),
        "Uversky Phase Plot": (
            "Charge\u2013hydrophobicity phase diagram for disorder prediction.\n\n"
            "Boundary: H* = 2.785 \u00b7 |R| + 0.446 (Uversky et al. 2000).\n"
            "Points above the line are predicted disordered; below = ordered.\n\n"
            "\u26a0 Caution: derived from only 28 IDPs + 91 folded proteins. "
            "Use as visual context only; the Disorder Profile tab is more reliable.\n"
            "Reference: Uversky et al., Proteins 41:415, 2000."
        ),
        "Single-Residue Perturbation Map": (
            "In silico saturation mutagenesis heatmap.\n\n"
            "Score(i\u2192j) = |\u0394GRAVY| + |\u0394NCPR|\n\n"
            "where \u0394GRAVY and \u0394NCPR are changes in global hydrophobicity index and net charge "
            "per residue upon the substitution. Hot = strongly destabilising; white dots = wild type.\n\n"
            "Note: this is a biophysical perturbation metric, not a pathogenicity predictor. "
            "Heatmap colour scale selectable in Settings."
        ),
        "SASA Profile": (
            "Per-residue solvent-accessible surface area (SASA) from the loaded 3D structure.\n\n"
            "Computed using the Shrake\u2013Rupley algorithm (BioPython). Available only when a\n"
            "structure is loaded (AlphaFold or PDB fetch, or local file).\n\n"
            "RSA (relative solvent accessibility, 0\u20131, dimensionless):\n"
            "  < 0.20:  Buried residue\n"
            "  0.20\u20130.50: Partially exposed\n"
            "  > 0.50:  Exposed\n\n"
            "Toggle to 'Show raw ASA (Å\u00b2)' to see absolute solvent-accessible area.\n"
            "Smoothing follows the window size set in Settings."
        ),
        "pLDDT Profile": (
            "AlphaFold per-residue confidence score (pLDDT, 0\u2013100).\n\n"
            "  > 90:  Very high confidence (well-structured)\n"
            "  70\u201390: Confident backbone\n"
            "  50\u201370: Low confidence (flexible/disordered)\n"
            "  < 50:  Very low confidence (likely IDR)\n\n"
            "pLDDT strongly correlates with experimental disorder scores.\n"
            "Reference: Jumper et al., Nature 596:583, 2021."
        ),
        "Distance Map": (
            "C\u03b1\u2013C\u03b1 pairwise distance matrix (\u00c5) from the loaded structure.\n\n"
            "Pixel (i, j) = Euclidean distance between C\u03b1 atoms of residues i and j. "
            "Pink contour lines mark the 8 \u00c5 contact threshold.\n\n"
            "Patterns:\n"
            "  \u2022 Off-diagonal stripes \u2192 \u03b1-helices (i, i+4 contacts)\n"
            "  \u2022 Triangular patches  \u2192 \u03b2-strands\n"
            "  \u2022 Scattered clusters  \u2192 long-range tertiary contacts\n\n"
            "Heatmap colour scale selectable in Settings."
        ),
        "Domain Architecture": (
            "Pfam domain architecture on a sequence ruler.\n\n"
            "Coloured boxes = Pfam-A domains fetched from EBI REST API. "
            "Disorder (grey gradient) and TM helices (orange) overlaid when available.\n\n"
            "Reference: Mistry et al. (Pfam), Nucleic Acids Res. 49:D412, 2021."
        ),
        "SS Bead Model": (
            "Linear bead model coloured by secondary structure from PDB HELIX/SHEET records.\n\n"
            "Each bead = one residue. Red = \u03b1-Helix, Gold = \u03b2-Sheet, Grey = Coil/Loop.\n\n"
            "Provides a sequence-level view of the fold topology — complement to the Ramachandran "
            "plot. Available after any structure is loaded."
        ),
        "Ramachandran Plot": (
            "\u03c6/\u03c8 backbone dihedral angle scatter plot.\n\n"
            "\u03c6 (phi): C\u2013N\u2013C\u03b1\u2013C torsion angle (around N\u2013C\u03b1 bond).\n"
            "\u03c8 (psi): N\u2013C\u03b1\u2013C\u2013N torsion angle (around C\u03b1\u2013C bond).\n\n"
            "Dark shading = favoured (\u226598% of high-res structures); light = allowed; "
            "white = disallowed. Outliers may indicate modelling errors.\n"
            "Reference: Ramachandran et al., J. Mol. Biol. 7:95, 1963."
        ),
        "Residue Contact Network": (
            "Graph of residue\u2013residue contacts (C\u03b1\u2013C\u03b1 < 8 \u00c5).\n\n"
            "Node size \u221d betweenness centrality. Node colour = eigenvector centrality "
            "(colourmap selectable in Settings).\n\n"
            "Hub residues (high degree/centrality) are often structurally important; "
            "mutations at hubs tend to be destabilising."
        ),
        "\u03b2-Aggregation Profile": (
            "Per-residue \u03b2-aggregation propensity (ZYGGREGATOR algorithm).\n\n"
            "Score estimates the tendency to join amyloid-like \u03b2-sheet stacks. "
            "Values > 1.0 (threshold) are aggregation-prone.\n\n"
            "Reference: Tartaglia et al., J. Mol. Biol. 380:425, 2008."
        ),
        "Solubility Profile": (
            "CamSol intrinsic solubility score per residue.\n\n"
            "Sliding-window sum of per-residue solubility parameters combining hydrophobicity, "
            "charge, and backbone flexibility. Negative values indicate lower intrinsic solubility "
            "(Sormanni et al. 2015); interpretation requires experimental validation.\n\n"
            "Reference: Sormanni et al., J. Mol. Biol. 427(2):478-490, 2015."
        ),
        "Hydrophobic Moment": (
            "Eisenberg hydrophobic moment \u03bcH \u2014 a measure of amphipathicity.\n\n"
            "\u03bcH = \u221a[(\u03a3 h\u1d62 sin(\u03b4i))\u00b2 + (\u03a3 h\u1d62 cos(\u03b4i))\u00b2] / n\n"
            "\u03b4 = 100\u00b0/residue (\u03b1-helix) or 160\u00b0/residue (\u03b2-strand); "
            "h\u1d62 = Eisenberg consensus hydrophobicity.\n\n"
            "\u03bcH \u2265 0.35 identifies candidate amphipathic helices (\u03b1-helix, \u03b4=100\u00b0/residue). "
            "Membrane activity requires experimental validation.\n"
            "Reference: Eisenberg et al., PNAS 81:140, 1984."
        ),
        "Annotation Track": (
            "Five-track integrated overview.\n\n"
            "Tracks (top to bottom):\n"
            "  1. Disorder score (metapredict, 0\u20131)\n"
            "  2. Hydrophobicity (sliding-window KD)\n"
            "  3. Aggregation propensity (ZYGGREGATOR)\n"
            "  4. Feature annotations: TM helices, signal peptide, LARKS\n"
            "  5. Residue ruler"
        ),
        "Cleavage Map": (
            "Predicted proteolytic cleavage sites for 9 enzymes.\n\n"
            "Enzymes: Trypsin (K/R|\u00acP), Chymotrypsin (F/W/Y|\u00acP), Glu-C, Lys-C, Lys-N, "
            "Asp-N, CNBr (Met), Pepsin, Thermolysin.\n\n"
            "Sites shown as coloured tick marks. Tryptic peptides listed with predicted m/z (2+ charge). "
            "Useful for LC-MS/MS experiment design and sequence coverage estimation."
        ),
        "PLAAC Profile": (
            "Prion-like amino acid composition (PLAAC) score per residue.\n\n"
            "Positive values = prion-like sequence character (Q/N-rich low-complexity). "
            "HMM trained on yeast prion-like domains (PrLDs).\n\n"
            "PrLDs are enriched in FUS, TDP-43, hnRNPA1, and other RBPs linked to "
            "amyloid and phase separation.\n"
            "Reference: Lancaster et al., Cell 149:936, 2014."
        ),
        "RNA-Binding Profile": (
            "Per-residue RNA-binding propensity score.\n\n"
            "Scored using a sliding-window sum of RBP-associated amino acid weights "
            "(Arg, Gly, Tyr enriched in RNA-binding). Canonical RBP motifs detected:\n"
            "  \u2022 RGG/RGX boxes\n"
            "  \u2022 RRM (RNA recognition motif)\n"
            "  \u2022 KH domain\n"
            "  \u2022 SR repeats\n\n"
            "Motif presence (RGG, RRM, KH, SR) indicates candidate RNA-binding regions; "
            "experimental validation is required. No validated composite score threshold is applied."
        ),
        "SCD Profile": (
            "Sequence Charge Decoration (SCD) per-residue contribution.\n\n"
            "SCD = (1/N) \u00b7 \u03a3\u1d62<\u2c7c \u03c3\u1d62\u03c3\u2c7c \u00b7 |i\u2212j|^0.5\n\n"
            "where \u03c3\u1d62 = +1 (K/R), \u22121 (D/E), 0 otherwise.\n\n"
            "High positive SCD = alternating charges (compact polyampholyte). "
            "High negative SCD = like-charge clusters (extended chain).\n"
            "Reference: Sawle & Ghosh, J. Chem. Phys. 143:085101, 2015."
        ),
        "Truncation Series": (
            "Biophysical properties across progressive N- or C-terminal truncations.\n\n"
            "At each truncation step the following are re-computed:\n"
            "  \u2022 Disorder fraction (metapredict mean)\n"
            "  \u2022 GRAVY index (mean KD hydrophobicity)\n"
            "  \u2022 Isoelectric point (pI)\n\n"
            "Useful for rational design of truncated constructs that preserve biophysical character."
        ),
        "MSA Conservation": (
            "Per-column sequence conservation across the loaded MSA.\n\n"
            "Conservation = 1 \u2212 H_norm,  where\n"
            "  H = \u2212\u03a3\u1d62 p\u1d62 \u00b7 log\u2082(p\u1d62)  (Shannon entropy)\n"
            "  H_norm = H / log\u2082(20)\n\n"
            "Score 1.0 = fully conserved. Score \u2248 0 = maximally variable. "
            "Highly conserved positions are functionally or structurally important."
        ),
        "MSA Covariance": (
            "Pairwise residue coevolution from MSA mutual information (MI) with APC correction.\n\n"
            "MI(i,j) = \u03a3\u2090 \u03a3\u1d67 f(a,b) \u00b7 log(f(a,b) / (f(a)\u00b7f(b)))\n\n"
            "APC correction removes phylogenetic background noise:\n"
            "  MI_APC(i,j) = MI(i,j) \u2212 MI(i,\u00b7)\u00b7MI(\u00b7,j) / MI(\u00b7,\u00b7)\n\n"
            "High MI-APC = co-evolving pairs, often spatially proximal or functionally coupled.\n"
            "Reference: Dunn et al., Bioinformatics 24:333, 2008.  "
            "Heatmap colour scale selectable in Settings."
        ),
        "Complex Mass": (
            "Molecular weight composition of a multi-subunit complex.\n\n"
            "Each bar = one subunit \u00d7 copy number. Total = \u03a3 (MW_subunit \u00d7 n_copies).\n\n"
            "MW computed from amino acid composition using average monoisotopic residue masses. "
            "Useful for SEC-MALS, AUC, or native MS experimental planning."
        ),
        "Variant Effect Map": (
            "ESM2 log-likelihood ratio (LLR) map for all single-residue substitutions.\n\n"
            "LLR(i, a) = log P(a | context) \u2212 log P(WT | context)\n\n"
            "Positive LLR (blue) = tolerated/favoured substitution. "
            "Negative LLR (red) = likely deleterious.\n\n"
            "Lower panel: mean LLR per position \u2014 "
            "positions with low mean LLR are evolutionarily constrained.\n\n"
            "Reference: Rives et al. (ESM2), PNAS 118:e2016239118, 2021."
        ),
        "AlphaMissense": (
            "AlphaMissense pathogenicity scores for all single-amino-acid substitutions.\n\n"
            "Score 0 = benign, 1 = pathogenic.\n"
            "  \u2022 > 0.564: likely pathogenic\n"
            "  \u2022 0.340\u20130.564: ambiguous\n"
            "  \u2022 < 0.340: likely benign\n\n"
            "AlphaMissense uses AlphaFold structure + evolutionary context. "
            "Scores fetched from EBI AlphaFold API.\n\n"
            "Lower panel: mean pathogenicity per position \u2014 "
            "positions with high mean scores are intolerant to substitution.\n\n"
            "Reference: Cheng et al., Science 381:eadg7492, 2023."
        ),
    }


    @staticmethod
    def _make_training_placeholder_fig(tab_name: str, feat_name: str):
        """Return a matplotlib Figure shown for BiLSTM heads still being trained."""
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure
        fig = Figure(figsize=(8, 3.5), dpi=100)
        fig.set_facecolor("#fffbf0")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#fffbf0")
        ax.axis("off")
        ax.text(0.5, 0.62, "[Training]  Model Training in Progress",
                ha="center", va="center", fontsize=15, fontweight="bold",
                color="#b45309", transform=ax.transAxes)
        ax.text(0.5, 0.42,
                f"The  {feat_name}  AI prediction head is currently being trained on\n"
                f"UniProt Swiss-Prot annotations. This graph will appear\n"
                f"automatically once the model file is ready.",
                ha="center", va="center", fontsize=10, color="#78350f",
                linespacing=1.6, transform=ax.transAxes)
        ax.text(0.5, 0.15,
                "Architecture: ESM2 650M → 2-layer BiLSTM classifier (hidden=256) → sigmoid",
                ha="center", va="center", fontsize=8, color="#a16207",
                fontstyle="italic", transform=ax.transAxes)
        for sp in ax.spines.values():
            sp.set_visible(False)
        fig.tight_layout(pad=1.5)
        return fig

    def _show_named_graph(self, graph_name: str) -> None:
        """Select graph_name in the Graphs tab tree and render it."""
        if not hasattr(self, "graph_list"):
            return
        items = self.graph_list.findItems(graph_name, Qt.MatchFlag.MatchExactly)
        if items:
            self.graph_list.setCurrentItem(items[0])
            self._on_graph_selected(items[0])

    def _jump_to_graph(self, graph_name: str) -> None:
        """Switch to the Graphs tab and select graph_name."""
        for i in range(self.main_tabs.nav_list.count()):
            if "Graphs" in self.main_tabs.nav_list.item(i).text():
                self.main_tabs.setCurrentIndex(i)
                break
        self._show_named_graph(graph_name)

    def _replace_graph(self, title: str, fig):
        """Swap graph canvas in the named tab, preserving uncertainty checkbox state."""
        import matplotlib.pyplot as _plt
        tab, vb = self.graph_tabs[title]
        # Close the old figure to free memory before clearing the layout
        for i in range(vb.count()):
            item = vb.itemAt(i)
            if item and isinstance(item.widget(), FigureCanvas):
                _plt.close(item.widget().figure)
                break
        self._clear_layout(vb)
        # High-DPI: match canvas DPI to physical screen resolution
        dpr = self.devicePixelRatioF() if hasattr(self, "devicePixelRatioF") else 1.0
        fig.set_dpi(min(150, max(96, int(96 * dpr))))
        # Re-apply tight layout after DPI change so labels/titles are never clipped.
        # Skip figures that use constrained_layout (they handle spacing themselves).
        import warnings as _w
        if not getattr(fig, "get_constrained_layout", lambda: False)() \
                and not getattr(fig, "_beer_manual_layout", False):
            with _w.catch_warnings():
                _w.filterwarnings("ignore", message=".*tight_layout.*", category=UserWarning)
                try:
                    fig.set_tight_layout({"pad": 2.0, "h_pad": 1.5, "w_pad": 1.5})
                except Exception:
                    pass
        canvas = FigureCanvas(fig)
        canvas.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        canvas.customContextMenuRequested.connect(
            lambda pos, c=canvas: self._graph_context_menu(c, pos))
        toolbar = NavigationToolbar2QT(canvas, self)
        from PySide6.QtCore import QSize as _QSize
        toolbar.setIconSize(_QSize(20, 20))
        is_dark = hasattr(self, "theme_toggle") and self.theme_toggle.isChecked()
        if is_dark:
            toolbar.setStyleSheet(
                "QToolBar { background: #1e2640; border: none; border-radius: 4px;"
                "           padding: 2px; spacing: 1px; }"
                "QToolButton { background: transparent; border: none; border-radius: 3px;"
                "              padding: 3px; color: #e8eaef; }"
                "QToolButton:hover   { background: rgba(255,255,255,0.15); }"
                "QToolButton:pressed { background: rgba(255,255,255,0.25); }"
                "QToolButton:checked { background: rgba(67,97,238,0.55); }"
            )
        else:
            toolbar.setStyleSheet(
                "QToolBar { background: #eef0f8; border: 1px solid #d0d4e0;"
                "           border-radius: 4px; padding: 2px; spacing: 1px; }"
                "QToolButton { background: #ffffff; border: 1px solid #d0d4e0;"
                "              border-radius: 3px; padding: 3px; color: #2d3748; }"
                "QToolButton:hover   { background: #e0e4f4; border-color: #4361ee; }"
                "QToolButton:pressed { background: #c8d0ec; }"
                "QToolButton:checked { background: #4361ee; border-color: #3451d1; color: #ffffff; }"
            )
            # matplotlib reads QPalette to decide icon colour at __init__ time;
            # on macOS system-dark-mode the palette reports a dark window even
            # when BEER's CSS theme is light, producing white icons on a light
            # background.  Re-tint every icon to dark after toolbar creation.
            self._tint_toolbar_icons_dark(toolbar)
        vb.addWidget(toolbar)
        vb.addWidget(canvas)
        # Vertical crosshair on single-axes profile graphs (residue-position x-axis)
        _PROFILE_GRAPHS = BILSTM_PROFILE_TABS | {
            "Hydrophobicity Profile", "Local Charge Profile",
            "SCD Profile", "SHD Profile", "RNA-Binding Profile",
            "β-Aggregation Profile", "Solubility Profile",
            "pLDDT Profile", "SASA Profile", "Hydrophobic Moment",
        }
        if title in _PROFILE_GRAPHS and len(canvas.figure.axes) == 1:
            try:
                from matplotlib.widgets import Cursor as _MplCursor
                _MplCursor(canvas.figure.axes[0], useblit=True,
                           color="#4361ee", linewidth=0.7, linestyle="--", alpha=0.6)
            except Exception:
                pass
            # Bidirectional graph↔structure link: hover on graph → highlight in 3D
            if hasattr(self, "structure_viewer"):
                self._wire_graph_struct_hover(canvas)
            # Re-apply structure position marker if one is active
            _marker = getattr(self, "_struct_marker_resi", None)
            if _marker is not None:
                self._apply_struct_marker_to_canvas(_marker, canvas)
        hint = (self._hydrophobicity_hint() if title == "Hydrophobicity Profile"
                else self._GRAPH_HINTS.get(title, "") or "")
        if hint:
            from PySide6.QtWidgets import (QToolButton as _QGTB, QDialog as _QDlg,
                                           QVBoxLayout as _QVB, QTextBrowser as _QTB,
                                           QDialogButtonBox as _QBB)
            from PySide6.QtCore import Qt as _Qt2
            info_btn = _QGTB()
            info_btn.setText("\u24d8")
            info_btn.setMaximumWidth(28)
            info_btn.setObjectName("info_btn")
            info_btn.setToolTip("Click for description, equations, and references.")

            def _show_info(_, h=hint, t=title):
                from beer.gui.themes import DARK_THEME_CSS as _DCSS, LIGHT_THEME_CSS as _LCSS
                _dark = hasattr(self, "theme_toggle") and self.theme_toggle.isChecked()
                dlg = _QDlg(self)
                dlg.setWindowTitle(t)
                dlg.setMinimumWidth(540)
                dlg.setMinimumHeight(360)
                dlg.setStyleSheet(_DCSS if _dark else _LCSS)
                vbl = _QVB(dlg)
                browser = _QTB()
                browser.setObjectName("info_dialog")
                browser.setOpenExternalLinks(False)
                browser.setReadOnly(True)
                browser.setPlainText(h)
                vbl.addWidget(browser)
                bb = _QBB(_QBB.StandardButton.Close)
                bb.rejected.connect(dlg.reject)
                bb.accepted.connect(dlg.accept)
                vbl.addWidget(bb)
                dlg.exec()

            info_btn.clicked.connect(_show_info)
            vb.addWidget(info_btn, alignment=Qt.AlignmentFlag.AlignRight)

        # ── RSA / ASA toggle for SASA Profile ───────────────────────────
        if title == "SASA Profile":
            _sasa_chk = QCheckBox("Show raw ASA (Å²)  [default: RSA, dimensionless 0–1]")
            _sasa_chk.setChecked(getattr(self, "_sasa_show_asa", False))
            _sasa_chk.setToolTip(
                "Unchecked: Relative Solvent Accessibility (RSA, 0–1, dimensionless).\n"
                "Checked: Absolute Solvent-Accessible Surface Area (ASA, Å²)."
            )
            def _toggle_sasa_mode(checked):
                self._sasa_show_asa = checked
                self._rebuild_sasa_graph()
            _sasa_chk.toggled.connect(_toggle_sasa_mode)
            vb.addWidget(_sasa_chk, alignment=Qt.AlignmentFlag.AlignLeft)

        # ── "Show Uncertainty" checkbox for BiLSTM profile tabs ─────────
        if title in BILSTM_PROFILE_TABS and self.analysis_data:
            _unc_chk = QCheckBox("Show Uncertainty (MC-Dropout)")
            # Restore checked state across redraws (preserved in _bilstm_unc_state).
            _unc_chk.setChecked(getattr(self, "_bilstm_unc_state", {}).get(title, False))
            _unc_chk.setToolTip(
                "Compute ±1σ uncertainty band via MC-Dropout "
                "(20 stochastic forward passes). Adds ~2–5 s per profile."
            )

            def _toggle_uncertainty(checked, _title=title):
                if not hasattr(self, "_bilstm_unc_state"):
                    self._bilstm_unc_state = {}
                self._bilstm_unc_state[_title] = checked
                self._rebuild_bilstm_with_uncertainty(_title, checked)

            _unc_chk.toggled.connect(_toggle_uncertainty)
            vb.addWidget(_unc_chk, alignment=Qt.AlignmentFlag.AlignLeft)

        # ── Inline colormap for heatmap graphs ──────────────────────────────
        _HEATMAP_GRAPHS = {
            "Distance Map", "Residue Contact Network",
            "Cation\u2013\u03c0 Map", "MSA Covariance",
            "Single-Residue Perturbation Map",
            "Variant Effect Map", "AlphaMissense",
            "Helical Wheel",
        }
        if title in _HEATMAP_GRAPHS:
            _cmap_row = QWidget()
            _cmap_rl  = QHBoxLayout(_cmap_row)
            _cmap_rl.setContentsMargins(0, 2, 0, 2)
            _cmap_rl.addWidget(QLabel("Colormap:"))
            _cmap_combo = QComboBox()
            _cmap_combo.addItems(NAMED_COLORMAPS)
            _cmap_combo.setCurrentText(self.heatmap_cmap)
            _cmap_combo.setMaximumWidth(180)
            _cmap_combo.setToolTip("Change colormap for this graph.")

            def _on_cmap_changed(_cmap_name, _t=title):
                self.heatmap_cmap = _cmap_name
                self._generated_graphs.discard(_t)
                self.update_graph_tabs()
                self._render_graph(_t)

            _cmap_combo.currentTextChanged.connect(_on_cmap_changed)
            _cmap_rl.addWidget(_cmap_combo)
            _cmap_rl.addStretch()
            vb.addWidget(_cmap_row)

        _btn_bar = QWidget()
        _btn_row = QHBoxLayout(_btn_bar)
        _btn_row.setContentsMargins(0, 2, 0, 2)
        _btn_row.addWidget(QLabel("Save as:"))
        _fmt_combo = QComboBox()
        _fmt_combo.addItems(["PNG", "SVG", "PDF"])
        _fmt_combo.setCurrentText(self.default_graph_format.upper())
        _fmt_combo.setMinimumWidth(70)
        _fmt_combo.setMaximumWidth(80)
        _fmt_combo.setToolTip("Format for saving this graph.")
        _btn_row.addWidget(_fmt_combo)
        _btn_row.addStretch()
        btn = QPushButton("Save Graph")
        btn.clicked.connect(lambda _, t=title, c=_fmt_combo: self.save_graph(t, c.currentText()))
        _export_btn = QPushButton("Export Data")
        _export_btn.setToolTip("Export the underlying data as CSV or JSON")
        _export_btn.clicked.connect(lambda _, t=title: self.export_graph_data(t))
        _copy_btn = QPushButton("Copy to Clipboard")
        _copy_btn.setToolTip("Copy figure to clipboard as PNG")
        _copy_btn.clicked.connect(lambda _, t=title: self._copy_graph_to_clipboard(t))
        _btn_row.addWidget(btn)
        _btn_row.addWidget(_export_btn)
        _btn_row.addWidget(_copy_btn)
        vb.addWidget(_btn_bar)

    @staticmethod
    def _tint_toolbar_icons_dark(toolbar):
        """Re-colour all matplotlib toolbar icons to a dark shade.

        NavigationToolbar2QT reads QPalette at __init__ time to decide whether
        to invert icons.  On macOS with system dark mode the palette reports a
        dark window even when BEER's CSS theme is light, so icons end up white
        on a white background.  This method forces every icon to #2d3748 while
        preserving the alpha channel.
        """
        from PySide6.QtGui import QPainter, QColor, QPixmap, QIcon
        from PySide6.QtCore import Qt, QSize
        from PySide6.QtWidgets import QToolButton as _TB
        target = QColor("#2d3748")
        for btn in toolbar.findChildren(_TB):
            icon = btn.icon()
            if icon.isNull():
                continue
            src = icon.pixmap(QSize(24, 24))
            if src.isNull():
                continue
            result = QPixmap(src.size())
            result.setDevicePixelRatio(src.devicePixelRatio())
            result.fill(Qt.GlobalColor.transparent)
            p = QPainter(result)
            p.drawPixmap(0, 0, src)
            p.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_SourceIn)
            p.fillRect(result.rect(), target)
            p.end()
            btn.setIcon(QIcon(result))

    def _update_esm2_indicator(self, state: str = "ready",
                               disorder_method: str = "") -> None:
        """Update the permanent disorder-method status label in the status bar.

        state: 'ready'      — ESM2 available, model not yet run this session
               'active'     — ESM2 was used in the last analysis
               'metapredict'— ESM2 unavailable; metapredict used
               'classical'  — ESM2 and metapredict both unavailable
               'missing'    — fair-esm / torch not installed (no analysis yet)
        """
        from beer.embeddings import ESM2_AVAILABLE
        model = getattr(self._embedder, "model_name", None)
        parts = model.split("_") if model else []
        try:
            size_tag = next(p for p in parts if p.endswith("M") or p.endswith("B"))
        except StopIteration:
            size_tag = ""

        if not ESM2_AVAILABLE or self._embedder is None:
            if state == "metapredict":
                text       = "Disorder \u00b7 metapredict"
                esm2_state = "metapredict"
            elif state == "classical":
                text       = "Disorder \u00b7 propensity scale"
                esm2_state = "classical"
            else:
                text       = "ESM2 \u00b7 not installed"
                esm2_state = "missing"
        elif state == "active":
            text       = f"ESM2 {size_tag} \u00b7 active \u2714"
            esm2_state = "active"
        else:
            text       = f"ESM2 {size_tag} \u00b7 ready"
            esm2_state = "ready"

        self._esm2_indicator.setText(text)
        self._esm2_indicator.setObjectName("esm2_lbl")
        self._esm2_indicator.setProperty("esm2_state", esm2_state)
        self._esm2_indicator.style().unpolish(self._esm2_indicator)
        self._esm2_indicator.style().polish(self._esm2_indicator)

    def _mark_chip_fetched(self, btn: "QPushButton") -> None:
        """Turn a chip button green (property-based, theme-aware) to signal a successful fetch."""
        btn.setProperty("chip_state", "fetched")
        btn.style().unpolish(btn)
        btn.style().polish(btn)

    def _mark_chip_loading(self, btn: "QPushButton") -> None:
        """Set chip button to amber to indicate an in-progress fetch/computation."""
        btn.setProperty("chip_state", "loading")
        btn.style().unpolish(btn)
        btn.style().polish(btn)

    def _mark_chip_normal(self, btn: "QPushButton") -> None:
        """Reset chip button to default (unfetched) state."""
        btn.setProperty("chip_state", "normal")
        btn.style().unpolish(btn)
        btn.style().polish(btn)

    def _title_from_canvas(self, canvas) -> str | None:
        """Reverse-look-up the graph title for a given FigureCanvas."""
        for title, (_, vb) in self.graph_tabs.items():
            for i in range(vb.count()):
                item = vb.itemAt(i)
                if item and item.widget() is canvas:
                    return title
        return None

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
        has = bool(self.batch_data)
        self.chain_combo.setEnabled(has)
        self._chain_row_widget.setVisible(has)
        if has:
            self.chain_combo.setCurrentIndex(0)

    def _load_batch(self, entries: list):
        """Analyze and load a list of (id, seq) pairs into the batch table."""
        self.batch_data.clear()
        self.batch_table.setRowCount(0)
        # Reset structure state; callers that bring structure (import_pdb,
        # fetch_accession PDB branch, _on_alphafold_finished) re-populate these
        # after calling _load_batch.
        self.batch_struct         = {}
        self.alphafold_data       = None
        self._struct_is_alphafold = False
        self.export_structure_btn.setEnabled(False)
        for rec_id, seq in entries:
            if not is_valid_protein(seq):
                continue
            data = AnalysisTools.analyze_sequence(seq, 7.0, self.default_window_size, self.use_reducing, self.custom_pka, hydro_scale=self.hydro_scale, embedder=self._embedder)
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

        # ── Row 1: Import ▾ | Fetch bar | Find UniProt ID | Biological Assembly ──
        row1 = QHBoxLayout()
        row1.setSpacing(6)

        self.import_btn = QPushButton("Import \u25be")
        self.import_btn.setToolTip("Import a sequence or structure file")
        self.import_btn.setMinimumHeight(32)
        import_menu = QMenu(self.import_btn)
        act_fasta = import_menu.addAction("FASTA (.fasta / .fa)")
        act_pdb   = import_menu.addAction("PDB (.pdb)")
        act_mmcif = import_menu.addAction("mmCIF / CIF (.cif)")
        act_fasta.triggered.connect(self.import_fasta)
        act_pdb.triggered.connect(self.import_pdb)
        act_mmcif.triggered.connect(self.import_mmcif)
        self.import_btn.setMenu(import_menu)
        # Hidden refs kept for any callers that use these button names
        self.import_fasta_btn = QPushButton(); self.import_fasta_btn.hide()
        self.import_fasta_btn.clicked.connect(self.import_fasta)
        self.import_pdb_btn = QPushButton(); self.import_pdb_btn.hide()
        self.import_pdb_btn.clicked.connect(self.import_pdb)
        self.import_mmcif_btn = QPushButton(); self.import_mmcif_btn.hide()
        self.import_mmcif_btn.clicked.connect(self.import_mmcif)
        row1.addWidget(self.import_btn)

        row1.addSpacing(6)
        row1.addWidget(QLabel("Fetch:"))
        self.accession_input = QLineEdit()
        self.accession_input.setPlaceholderText("e.g. P04637 (p53)  ·  1UBQ (ubiquitin)")
        row1.addWidget(self.accession_input, 1)
        fetch_btn = QPushButton("Fetch")
        fetch_btn.setMinimumHeight(30)
        fetch_btn.clicked.connect(self.fetch_accession)
        row1.addWidget(fetch_btn)

        row1.addSpacing(6)
        self.bio_assembly_chk = QCheckBox("Biological Assembly")
        self.bio_assembly_chk.setToolTip(
            "When fetching a PDB ID: download the full biological assembly\n"
            "instead of the asymmetric unit. Uses RCSB assembly1.cif.")
        row1.addWidget(self.bio_assembly_chk)

        row1.addSpacing(4)
        self.find_uniprot_btn = QPushButton("Find UniProt ID")
        self.find_uniprot_btn.setMinimumHeight(30)
        self.find_uniprot_btn.setToolTip(
            "Search UniProt for the current sequence (exact hash match, then BLAST).\n"
            "Once found, the accession enables AlphaFold/ELM/DisProt/annotation buttons.")
        self.find_uniprot_btn.clicked.connect(self.find_uniprot_from_sequence)
        self.find_uniprot_btn.setEnabled(False)
        row1.addWidget(self.find_uniprot_btn)
        row1.addStretch()

        outer.addLayout(row1)

        # ── Row 2: Mutate | Analyze | BiLSTM Analysis | History | Session ▾ ──
        row2 = QHBoxLayout()
        row2.setSpacing(6)

        self.mutate_btn = QPushButton("Mutate\u2026")
        self.mutate_btn.setToolTip("Introduce a point mutation at any position (run analysis first)")
        self.mutate_btn.setMinimumHeight(30)
        self.mutate_btn.setEnabled(False)
        self.mutate_btn.clicked.connect(self.open_mutation_dialog)
        row2.addWidget(self.mutate_btn)

        self.analyze_btn = QPushButton("Analyze  [Ctrl+\u21b5]")
        self.analyze_btn.setToolTip(
            "Run classical biophysical analysis (composition, charge, hydrophobicity, etc.).\n"
            "Fast — does not use ESM2. Use AI Analysis for deep-learning predictions.")
        self.analyze_btn.setMinimumHeight(30)
        self.analyze_btn.setObjectName("primary_btn")
        self.analyze_btn.clicked.connect(self.on_analyze)
        row2.addWidget(self.analyze_btn)

        self.bilstm_analyze_btn = QPushButton("AI Analysis")
        self.bilstm_analyze_btn.setToolTip(
            "Run ESM2 650M embedding + all AI prediction heads (ESM2 → BiLSTM classifiers).\n"
            "Run classical Analyze first, then click this to add AI annotations.\n"
            "Note: embedding a long protein on CPU can take several minutes.")
        self.bilstm_analyze_btn.setMinimumHeight(30)
        self.bilstm_analyze_btn.setEnabled(False)
        self.bilstm_analyze_btn.clicked.connect(self.on_bilstm_analyze)
        row2.addWidget(self.bilstm_analyze_btn)

        row2.addSpacing(8)
        row2.addWidget(QLabel("History:"))
        self.history_combo = QComboBox()
        self.history_combo.setMinimumWidth(200)
        self.history_combo.addItem("\u2014 recent sequences \u2014")
        self.history_combo.currentIndexChanged.connect(self._on_history_selected)
        row2.addWidget(self.history_combo)

        row2.addStretch()

        session_btn = QPushButton("Session \u25be")
        session_btn.setToolTip("Save / load session or open Figure Composer")
        session_btn.setMinimumHeight(28)
        session_menu = QMenu(session_btn)
        self.session_save_btn = QPushButton(); self.session_save_btn.hide()
        self.session_save_btn.clicked.connect(self.session_save)
        self.session_load_btn = QPushButton(); self.session_load_btn.hide()
        self.session_load_btn.clicked.connect(self.session_load)
        self.figure_composer_btn = QPushButton(); self.figure_composer_btn.hide()
        self.figure_composer_btn.clicked.connect(self.open_figure_composer)
        act_save    = session_menu.addAction("Save Session")
        act_load    = session_menu.addAction("Load Session")
        session_menu.addSeparator()
        act_compose = session_menu.addAction("Figure Composer")
        act_save.triggered.connect(self.session_save)
        act_load.triggered.connect(self.session_load)
        act_compose.triggered.connect(self.open_figure_composer)
        session_btn.setMenu(session_menu)
        row2.addWidget(session_btn)

        outer.addLayout(row2)

        # ── Sequence input ───────────────────────────────────────────────────
        self._seq_label = QLabel("Protein Sequence:")
        self._seq_label.setObjectName("accent_lbl")
        outer.addWidget(self._seq_label)

        self.seq_text = QTextEdit()
        self.seq_text.setPlaceholderText("Paste a protein sequence here, or use Import\u2026")
        self.seq_text.setFont(QFont("Courier New", 10))
        self.seq_text.setFixedHeight(100)
        self.seq_text.setAcceptDrops(True)
        outer.addWidget(self.seq_text)

        # ── Chain selector (hidden until multi-chain data is loaded) ────────────
        self._chain_row_widget = QWidget()
        chain_row = QHBoxLayout(self._chain_row_widget)
        chain_row.setContentsMargins(0, 0, 0, 0)
        chain_lbl = QLabel("Chain:")
        chain_lbl.setStyleSheet("font-weight:600;")
        self.chain_combo = QComboBox()
        self.chain_combo.setFixedWidth(160)
        self.chain_combo.setEnabled(False)
        self.chain_combo.currentTextChanged.connect(self.on_chain_selected)
        chain_row.addWidget(chain_lbl)
        chain_row.addWidget(self.chain_combo)
        chain_row.addStretch()
        self._chain_row_widget.hide()
        outer.addWidget(self._chain_row_widget)

        # ── External Data chips (always visible; inactive until fetch/analyze) ─
        self._ext_data_panel = QWidget()
        self._ext_data_panel.setObjectName("ext_data_panel")
        ext_vbox = QVBoxLayout(self._ext_data_panel)
        ext_vbox.setContentsMargins(0, 2, 0, 2)
        ext_vbox.setSpacing(2)

        chips_row = QHBoxLayout()
        chips_row.setSpacing(4)

        def _sep():
            f = QFrame()
            f.setFrameShape(QFrame.Shape.VLine)
            f.setFrameShadow(QFrame.Shadow.Plain)
            f.setObjectName("v_sep")
            f.setMaximumHeight(20)
            return f

        def _chip(label, tip, slot):
            b = QPushButton(label)
            b.setObjectName("chip_btn")
            b.setProperty("chip_state", "normal")
            b.setEnabled(False)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            chips_row.addWidget(b)
            return b

        grp_lbl = QLabel("Structure")
        grp_lbl.setObjectName("group_lbl")
        chips_row.addWidget(grp_lbl)
        self.fetch_af_btn = _chip("AlphaFold",
            "Fetch AlphaFold predicted structure (requires UniProt accession)",
            self.fetch_alphafold)
        self.fetch_pfam_btn = _chip("Pfam",
            "Fetch Pfam domain annotations from InterPro",
            self.fetch_pfam)
        self.fetch_deeptmhmm_btn = _chip("DeepTMHMM",
            "Run DeepTMHMM transmembrane topology prediction (requires internet + pybiolib)",
            self._run_deeptmlhmm)
        self.fetch_signalp6_btn = _chip("SignalP 6",
            "Run SignalP 6.0 signal peptide prediction via BioLib (requires internet + pybiolib)",
            self._run_signalp6)
        # UniProt Tracks is accessible from the Graphs tab top bar; no chip here.
        self.fetch_uniprot_tracks_btn = QPushButton(); self.fetch_uniprot_tracks_btn.hide()
        self.fetch_uniprot_tracks_btn.clicked.connect(self.fetch_uniprot_features)

        chips_row.addSpacing(4); chips_row.addWidget(_sep()); chips_row.addSpacing(4)

        grp_lbl2 = QLabel("Disorder / IDP")
        grp_lbl2.setObjectName("group_lbl")
        chips_row.addWidget(grp_lbl2)
        self.fetch_elm_btn = _chip("ELM",
            "Fetch experimentally validated linear motifs from ELM (UniProt only)",
            self.fetch_elm)
        self.fetch_disprot_btn = _chip("DisProt",
            "Fetch disorder annotations from DisProt (UniProt only)",
            self.fetch_disprot)
        self.fetch_mobidb_btn = _chip("MobiDB",
            "Fetch consensus disorder annotations from MobiDB (UniProt only)",
            self.fetch_mobidb)
        self.fetch_phasepdb_btn = _chip("PhaSepDB",
            "Check phase-separation database PhaSepDB (UniProt only)",
            self.fetch_phasepdb)

        chips_row.addSpacing(4); chips_row.addWidget(_sep()); chips_row.addSpacing(4)

        grp_lbl3 = QLabel("Variants & Interactions")
        grp_lbl3.setObjectName("group_lbl")
        chips_row.addWidget(grp_lbl3)
        self.fetch_variants_btn = _chip("Variants",
            "Fetch natural variants and mutagenesis data from UniProt",
            self.fetch_variants)
        self.fetch_alphafold_missense_btn = _chip("AlphaMissense",
            "Fetch AlphaMissense variant pathogenicity scores from EBI (UniProt only)",
            lambda: self._run_alphafold_missense(self.current_accession))
        self.fetch_intact_btn = _chip("IntAct",
            "Fetch curated binary interactions from IntAct / EBI (UniProt only)",
            self.fetch_intact)

        chips_row.addStretch()
        ext_vbox.addLayout(chips_row)

        # ── PDB cross-reference chips — shown after UniProt fetch ────────────
        self._pdb_xref_inner = QWidget()
        self._pdb_xref_layout = QVBoxLayout(self._pdb_xref_inner)
        self._pdb_xref_layout.setContentsMargins(4, 2, 4, 2)
        self._pdb_xref_layout.setSpacing(3)
        self._pdb_xref_inner.hide()
        ext_vbox.addWidget(self._pdb_xref_inner)

        self._ext_data_panel.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        outer.addWidget(self._ext_data_panel)

        # Convenience list for bulk enable/disable
        self._db_fetch_btns = [
            self.fetch_af_btn, self.fetch_pfam_btn, self.fetch_elm_btn,
            self.fetch_disprot_btn, self.fetch_mobidb_btn, self.fetch_phasepdb_btn,
            self.fetch_variants_btn, self.fetch_intact_btn,
            self.fetch_alphafold_missense_btn, self.fetch_deeptmhmm_btn,
        ]

        # ── Sequence viewer (Search / Highlight / Clear / Copy Sequence) ─────
        sv_hdr = QHBoxLayout()
        self._seq_view_label = QLabel("Sequence Viewer:")
        self._seq_view_label.setObjectName("accent_lbl")
        sv_hdr.addWidget(self._seq_view_label)
        sv_hdr.addSpacing(8)
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
        self.motif_match_lbl = QLabel("")
        self.motif_match_lbl.setObjectName("accent_lbl")
        sv_hdr.addWidget(self.motif_match_lbl)
        sv_hdr.addStretch()
        copy_seq_btn = QPushButton("Copy Sequence")
        copy_seq_btn.setToolTip("Copy the full sequence or a selected range to clipboard")
        copy_seq_btn.setMinimumHeight(26)
        copy_seq_btn.clicked.connect(self._copy_sequence_menu)
        sv_hdr.addWidget(copy_seq_btn)
        outer.addLayout(sv_hdr)

        self.seq_viewer = QTextBrowser()
        self.seq_viewer.setFont(QFont("Courier New", 10))
        outer.addWidget(self.seq_viewer, 1)

        # ── Bottom bar: Clear All ────────────────────────────────────────────
        bottom_row = QHBoxLayout()
        bottom_row.addStretch()
        clear_protein_btn = QPushButton("Clear All")
        clear_protein_btn.setToolTip("Clear the loaded protein, analysis, graphs and structure")
        clear_protein_btn.setMinimumHeight(28)
        clear_protein_btn.setObjectName("delete_btn")
        clear_protein_btn.clicked.connect(self._clear_all)
        bottom_row.addWidget(clear_protein_btn)
        outer.addLayout(bottom_row)

        # Stub out _seq_info_label so existing code that references it doesn't crash
        self._seq_info_label = QLabel("")
        self._seq_info_label.hide()

    # ── Report Tab ───────────────────────────────────────────────────────────

    def init_report_tab(self):
        """Report tab: section tree + content stack, Alanine Scan, Export."""
        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(6, 6, 6, 6)
        vbox.setSpacing(4)
        self.main_tabs.addTab(container, "Report")

        # Stub protein info bar (referenced elsewhere; hidden permanently here)
        self._protein_info_bar = QTextBrowser()
        self._protein_info_bar.setObjectName("info_bar")
        self._protein_info_bar.hide()

        # ── Report panel (section tree left | content stack right) ────────
        report_panel = QWidget()
        report_h     = QHBoxLayout(report_panel)
        report_h.setContentsMargins(0, 0, 0, 0)
        report_h.setSpacing(0)

        self.report_section_list = QTreeWidget()
        self.report_section_list.setObjectName("report_nav")
        self.report_section_list.setFixedWidth(170)
        self.report_section_list.setHeaderHidden(True)
        self.report_section_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.report_section_list.setIndentation(12)
        report_h.addWidget(self.report_section_list)

        rsep = QFrame()
        rsep.setFrameShape(QFrame.Shape.VLine)
        rsep.setFrameShadow(QFrame.Shadow.Plain)
        rsep.setObjectName("nav_sep")
        report_h.addWidget(rsep)

        self.report_stack = QStackedWidget()
        report_h.addWidget(self.report_stack, 1)

        self.report_section_tabs = {}
        self._report_sec_to_idx: dict = {}
        _stack_idx = 0
        bold_font = QFont(); bold_font.setBold(True)
        _grouped_secs = {s for _, secs in _REPORT_SECTION_GROUPS for s in secs}

        def _build_section_widget(sec: str) -> QTextBrowser:
            tab = QWidget()
            vb  = QVBoxLayout(tab)
            vb.setContentsMargins(4, 4, 4, 4)
            btn_row = QHBoxLayout()
            btn_row.setSpacing(4)
            hint = _SECTION_HINTS.get(sec, "")
            if hint:
                from PySide6.QtWidgets import QToolButton as _QTB
                help_btn = _QTB()
                help_btn.setText("?")
                help_btn.setMaximumWidth(24)
                help_btn.setMaximumHeight(24)
                help_btn.setStyleSheet("QToolButton { font-weight:bold; border-radius:10px; }")
                help_btn.setToolTip(hint)
                help_btn.clicked.connect(
                    lambda _, h=hint, s=sec: QMessageBox.information(self, s, h))
                btn_row.addWidget(help_btn)
            if sec == "Composition":
                for lbl, mode in [("A\u2013Z", "alpha"), ("By Freq", "composition"),
                                   ("Hydro \u2191", "hydro_inc"), ("Hydro \u2193", "hydro_dec")]:
                    b = QPushButton(lbl)
                    b.setMaximumWidth(90)
                    b.setMinimumHeight(26)
                    b.clicked.connect(lambda _, m=mode: self.sort_composition(m))
                    btn_row.addWidget(b)
            btn_row.addStretch()
            export_sec_btn = QPushButton("Export Section")
            export_sec_btn.setMaximumWidth(110)
            export_sec_btn.setMinimumHeight(26)
            export_sec_btn.setToolTip(f"Export the {sec} section as CSV or text")
            export_sec_btn.clicked.connect(lambda _, s=sec: self._export_section(s))
            btn_row.addWidget(export_sec_btn)
            copy_btn = QPushButton("Copy Table")
            copy_btn.setMaximumWidth(100)
            copy_btn.setMinimumHeight(26)
            copy_btn.clicked.connect(lambda _, s=sec: self._copy_section(s))
            btn_row.addWidget(copy_btn)
            vb.addLayout(btn_row)
            browser = QTextBrowser()
            _install_beer_link_filter(browser, self._on_report_link_clicked)
            vb.addWidget(browser)
            return tab, browser

        for group_name, group_secs in _REPORT_SECTION_GROUPS:
            grp_item = QTreeWidgetItem([group_name])
            grp_item.setFont(0, bold_font)
            grp_item.setFlags(grp_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self.report_section_list.addTopLevelItem(grp_item)
            for sec in group_secs:
                if sec not in REPORT_SECTIONS:
                    continue
                leaf = QTreeWidgetItem([sec])
                leaf.setData(0, Qt.ItemDataRole.UserRole, sec)
                grp_item.addChild(leaf)
                tab, browser = _build_section_widget(sec)
                self.report_stack.addWidget(tab)
                self.report_section_tabs[sec] = browser
                self._report_sec_to_idx[sec] = _stack_idx
                _stack_idx += 1
            grp_item.setExpanded(True)

        for sec in REPORT_SECTIONS:
            if sec not in _grouped_secs:
                leaf = QTreeWidgetItem([sec])
                leaf.setData(0, Qt.ItemDataRole.UserRole, sec)
                self.report_section_list.addTopLevelItem(leaf)
                tab, browser = _build_section_widget(sec)
                self.report_stack.addWidget(tab)
                self.report_section_tabs[sec] = browser
                self._report_sec_to_idx[sec] = _stack_idx
                _stack_idx += 1

        # ── AI Predictions dynamic group (populated after AI Analysis) ──────
        self._ai_pred_grp_item = QTreeWidgetItem(["AI Predictions"])
        self._ai_pred_grp_item.setFont(0, bold_font)
        self._ai_pred_grp_item.setFlags(
            self._ai_pred_grp_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        self.report_section_list.addTopLevelItem(self._ai_pred_grp_item)
        self._ai_pred_grp_item.setHidden(True)
        self._ai_pred_section_keys: list[str] = []
        self.report_section_list.itemClicked.connect(self._on_report_section_clicked)
        self.report_section_list.setCurrentItem(
            self.report_section_list.topLevelItem(0).child(0)
            if self.report_section_list.topLevelItem(0) else None)

        # ── Sub-tab widget: Report | Alanine Scan ─────────────────────────
        self._right_tabs = QTabWidget()
        self._right_tabs.addTab(report_panel, "Report")
        self._right_tabs.addTab(self._build_alanine_scan_panel(), "Alanine Scan")
        vbox.addWidget(self._right_tabs, 1)

        # (Export Complete Report removed in v2.0 — use per-section Export buttons)

    # ── Summary Tab ──────────────────────────────────────────────────────────

    def init_summary_tab(self):
        container = QWidget()
        vb = QVBoxLayout(container)
        vb.setContentsMargins(16, 12, 16, 12)
        vb.setSpacing(8)
        self.main_tabs.addTab(container, "Summary")

        self._summary_tab_browser = QTextBrowser()
        self._summary_tab_browser.setOpenExternalLinks(False)
        self._summary_tab_browser.setObjectName("summary_tab_browser")
        self._summary_tab_browser.setHtml(
            "<div style='font-family:sans-serif;color:#888;padding:40px;text-align:center'>"
            "<p style='font-size:15px'>Run analysis to see the protein summary.</p>"
            "</div>"
        )
        vb.addWidget(self._summary_tab_browser, 1)

    # ── Alanine Scan Sub-Tab ─────────────────────────────────────────────────

    def _build_alanine_scan_panel(self) -> QWidget:
        """Build the persistent Alanine Scan sub-tab widget."""
        w = QWidget()
        vb = QVBoxLayout(w)
        vb.setContentsMargins(10, 10, 10, 10)
        vb.setSpacing(8)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Scan range — Start:"))
        self._ala_start = QSpinBox()
        self._ala_start.setRange(1, 9999)
        self._ala_start.setValue(1)
        self._ala_start.setMaximumWidth(80)
        ctrl.addWidget(self._ala_start)
        ctrl.addWidget(QLabel("End:"))
        self._ala_end = QSpinBox()
        self._ala_end.setRange(1, 9999)
        self._ala_end.setValue(50)
        self._ala_end.setMaximumWidth(80)
        ctrl.addWidget(self._ala_end)
        self._ala_run_btn = QPushButton("Run Alanine Scan")
        self._ala_run_btn.setMinimumHeight(30)
        self._ala_run_btn.clicked.connect(self._run_alanine_scan)
        ctrl.addWidget(self._ala_run_btn)
        ctrl.addStretch()
        self._ala_export_btn = QPushButton("Export CSV")
        self._ala_export_btn.setMinimumHeight(30)
        self._ala_export_btn.setEnabled(False)
        self._ala_export_btn.clicked.connect(self._export_ala_csv)
        ctrl.addWidget(self._ala_export_btn)
        vb.addLayout(ctrl)

        note = QLabel(
            "Systematically mutates each residue in the range to Alanine and reports "
            "changes in GRAVY, net charge, disorder fraction, and molecular weight."
        )
        note.setWordWrap(True)
        note.setObjectName("status_lbl")
        vb.addWidget(note)

        splitter = QSplitter(Qt.Orientation.Vertical)
        self._ala_browser = QTextBrowser()
        self._ala_browser.setObjectName("ala_browser")
        splitter.addWidget(self._ala_browser)

        self._ala_canvas_container = QWidget()
        _cc_vb = QVBoxLayout(self._ala_canvas_container)
        _cc_vb.setContentsMargins(0, 0, 0, 0)
        self._ala_fig = Figure(figsize=(8, 3))
        self._ala_canvas = FigureCanvas(self._ala_fig)
        _cc_vb.addWidget(self._ala_canvas)
        splitter.addWidget(self._ala_canvas_container)
        splitter.setSizes([300, 200])
        vb.addWidget(splitter, 1)

        self._ala_results: list[dict] = []
        return w

    def _run_alanine_scan(self) -> None:
        seq = self.seq_text.toPlainText().strip().upper()
        seq = "".join(c for c in seq if c.isalpha())
        if not seq:
            QMessageBox.warning(self, "Alanine Scan", "Please enter a protein sequence first.")
            return
        start = max(1, self._ala_start.value())
        end   = min(len(seq), self._ala_end.value())
        if start > end:
            QMessageBox.warning(self, "Alanine Scan", "Start must be ≤ End.")
            return

        from beer.analysis.core import (
            sliding_window_hydrophobicity, HYDROPHOBICITY_SCALES,
            sliding_window_ncpr,
        )

        def _fast_props(s: str) -> dict:
            from Bio.SeqUtils.ProtParam import ProteinAnalysis as _PA
            try:
                pa = _PA(s)
                gravy = pa.gravy()
                mw    = pa.molecular_weight()
                nc    = pa.charge_at_pH(7.0)
            except Exception:
                gravy, mw, nc = 0.0, 0.0, 0.0
            n = len(s)
            kd_scale = HYDROPHOBICITY_SCALES["Kyte-Doolittle"]["values"]
            w = min(7, n)
            hydro = sliding_window_hydrophobicity(s, w, kd_scale)
            dis_f = sum(1 for v in hydro if v < -0.5) / max(n, 1)
            return {"gravy": gravy, "mw": mw, "nc": nc, "dis_f": dis_f}

        wt_props = _fast_props(seq)
        results = []
        for i in range(start - 1, end):
            aa = seq[i]
            if aa == "A":
                results.append({
                    "pos": i + 1, "wt": aa,
                    "d_gravy": 0.0, "d_mw": 0.0, "d_nc": 0.0, "d_dis": 0.0,
                })
                continue
            mut_seq = seq[:i] + "A" + seq[i+1:]
            mp = _fast_props(mut_seq)
            results.append({
                "pos": i + 1, "wt": aa,
                "d_gravy": round(mp["gravy"] - wt_props["gravy"], 4),
                "d_mw":    round(mp["mw"]    - wt_props["mw"],    2),
                "d_nc":    round(mp["nc"]    - wt_props["nc"],    3),
                "d_dis":   round(mp["dis_f"] - wt_props["dis_f"], 4),
            })
        self._ala_results = results

        # Table HTML
        rows_html = "".join(
            f"<tr>"
            f"<td>{r['pos']}</td><td>{r['wt']}</td>"
            f"<td style='color:{'#ef4444' if r['d_gravy']>0 else '#22c55e'}'>{r['d_gravy']:+.4f}</td>"
            f"<td style='color:{'#ef4444' if r['d_mw']>0 else '#22c55e'}'>{r['d_mw']:+.2f}</td>"
            f"<td style='color:{'#ef4444' if r['d_nc']>0 else '#22c55e'}'>{r['d_nc']:+.3f}</td>"
            f"<td style='color:{'#ef4444' if r['d_dis']>0 else '#22c55e'}'>{r['d_dis']:+.4f}</td>"
            f"</tr>"
            for r in results
        )
        html = (
            "<style>body{{font-family:sans-serif;font-size:12px}}"
            "table{{border-collapse:collapse;width:100%}}"
            "th,td{{border:1px solid #e2e8f0;padding:4px 8px;text-align:center}}"
            "th{{background:#f8fafc;font-weight:600}}</style>"
            "<h3>Alanine Scan Results</h3>"
            "<p>Wild-type: GRAVY={gravy:.4f} | MW={mw:.1f} Da | Charge={nc:.3f} | DisF={dis:.4f}</p>"
            "<table><tr><th>Pos</th><th>WT</th><th>ΔGRAVY</th><th>ΔMW (Da)</th>"
            "<th>ΔCharge</th><th>ΔDis.Frac</th></tr>"
            "{rows}</table>"
        ).format(
            gravy=wt_props["gravy"], mw=wt_props["mw"],
            nc=wt_props["nc"], dis=wt_props["dis_f"],
            rows=rows_html,
        )
        self._ala_browser.setHtml(html)

        # Bar chart (ΔGRAVY)
        self._ala_fig.clear()
        ax = self._ala_fig.add_subplot(111)
        positions = [r["pos"] for r in results]
        d_gravy   = [r["d_gravy"] for r in results]
        colors = ["#ef4444" if v > 0 else "#22c55e" for v in d_gravy]
        ax.bar(positions, d_gravy, color=colors, edgecolor="none", width=0.8)
        ax.axhline(0, color="#64748b", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Sequence position", fontsize=9)
        ax.set_ylabel("ΔGRAVY", fontsize=9)
        ax.set_title(f"Alanine scan ΔGRAVY  ({start}–{end})", fontsize=10)
        ax.tick_params(labelsize=8)
        self._ala_fig.tight_layout(pad=0.8)
        self._ala_canvas.draw()
        self._ala_export_btn.setEnabled(True)

    def _export_ala_csv(self) -> None:
        if not self._ala_results:
            return
        fn, _ = QFileDialog.getSaveFileName(
            self, "Export Alanine Scan", "alanine_scan.csv", "CSV (*.csv)",
            options=QFileDialog.Option.DontUseNativeDialog)
        if not fn:
            return
        if not fn.lower().endswith(".csv"):
            fn += ".csv"
        import csv as _csv
        with open(fn, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=["pos", "wt", "d_gravy", "d_mw", "d_nc", "d_dis"])
            w.writeheader()
            w.writerows(self._ala_results)
        self.statusBar.showMessage(f"Alanine scan exported: {fn}", 4000)

    def _build_summary_tab_html(self, data: dict) -> str:
        """Generate grouped bullet-point HTML for the Summary tab."""
        seq = data.get("seq", "")
        L   = len(seq)
        mw  = data.get("mol_weight", 0)
        pI  = data.get("iso_point", 0.0)
        gravy = data.get("gravy", 0.0)
        fcr   = data.get("fcr", 0.0)
        ncpr  = data.get("ncpr", 0.0)
        kappa = data.get("kappa", 0.0)
        omega = data.get("omega", 0.0)
        disorder_scores = data.get("disorder_scores") or []
        sp_prof  = data.get("sp_bilstm_profile") or []
        tm_prof  = data.get("tm_bilstm_profile") or []
        cc_prof  = data.get("cc_bilstm_profile") or []
        aggr     = data.get("aggr_profile") or []
        catg     = data.get("catgranule")
        rbp      = data.get("rbp") or {}
        larks    = data.get("larks") or []
        phospho  = data.get("phospho_sites") or {}
        gpi      = data.get("gpi_result") or {}
        motifs   = data.get("motifs") or {}
        pfam     = getattr(self, "pfam_domains", []) or []

        def _sec(title, items):
            if not items:
                return ""
            lis = "".join(f"<li style='margin:3px 0'>{i}</li>" for i in items)
            return (
                f"<h3 style='margin:14px 0 4px;color:#1a1a2e;font-size:13px;"
                f"border-bottom:1px solid #e2e8f0;padding-bottom:3px'>{title}</h3>"
                f"<ul style='margin:0;padding-left:18px;font-size:12px'>{lis}</ul>"
            )

        # ── Identity ──────────────────────────────────────────────────────
        identity = [
            f"<b>Length:</b> {L} residues",
            f"<b>Molecular weight:</b> {mw/1000:.2f} kDa",
            f"<b>Isoelectric point (pI):</b> {pI:.2f}",
            f"<b>GRAVY:</b> {gravy:+.3f} "
            f"({'hydrophobic' if gravy > 0 else 'hydrophilic'})",
        ]

        # ── Disorder & structure ──────────────────────────────────────────
        struct = []
        if disorder_scores:
            d_frac = sum(1 for v in disorder_scores if v > 0.5) / L
            if d_frac >= 0.5:
                struct.append(f"<b>Highly disordered</b> — {d_frac*100:.0f}% residues predicted disordered (AI Predictions, AUROC 0.9999)")
            elif d_frac >= 0.25:
                struct.append(f"<b>Partially disordered</b> — {d_frac*100:.0f}% disordered regions (AI Predictions)")
            else:
                struct.append(f"<b>Predominantly ordered</b> — {d_frac*100:.0f}% disordered (AI Predictions)")
        if sp_prof:
            sp_max = max(sp_prof[:35]) if len(sp_prof) >= 10 else 0
            if sp_max > 0.7:
                struct.append(f"<b>Signal peptide</b> detected at N-terminus (AI Predictions, score {sp_max:.2f}, AUROC 0.9999)")
        if gpi.get("has_gpi"):
            struct.append("<b>GPI anchor</b> signal at C-terminus")
        if tm_prof:
            n_tm = sum(1 for v in tm_prof if v > 0.5)
            if n_tm >= 15:
                struct.append(f"<b>Transmembrane protein</b> — ~{n_tm} residues in TM helices (AI Predictions, AUROC 0.992)")
        if cc_prof:
            n_cc = sum(1 for v in cc_prof if v > 0.5)
            if n_cc >= 10:
                struct.append(f"<b>Coiled-coil region</b> — {n_cc} residues (AI Predictions)")
        if pfam:
            struct.append(f"<b>Pfam domains:</b> {', '.join(d.get('name','?') for d in pfam[:5])}"
                          + (" …" if len(pfam) > 5 else ""))

        # ── Charge & IDP ──────────────────────────────────────────────────
        charge = [
            f"<b>FCR:</b> {fcr:.3f} · <b>NCPR:</b> {ncpr:+.3f} · "
            f"<b>κ:</b> {kappa:.3f} · <b>Ω:</b> {omega:.3f}",
        ]
        if fcr >= 0.35:
            charge.append("Strong polyelectrolyte (Das-Pappu FCR ≥ 0.35)")
        if larks:
            charge.append(f"<b>{len(larks)} LARKS</b> — potential amyloid-like interaction cores")

        # ── Phase separation & condensates ────────────────────────────────
        phase = []
        if catg is not None:
            sign = "prone" if catg > 0 else "resistant"
            phase.append(f"<b>catGRANULE score:</b> {catg:+.2f} ({sign} to condensate formation)")
        if rbp.get("composite_score", 0) > 0:
            phase.append(f"<b>RNA-binding (catRAPID):</b> ω̄ = {rbp['composite_score']:.2f}")
            if rbp.get("motifs"):
                phase.append(f"RNA-binding motifs: {', '.join(rbp['motifs'][:4])}")

        # ── Aggregation ───────────────────────────────────────────────────
        aggr_items = []
        if aggr:
            n_hot = sum(1 for v in aggr if v >= 1.0)
            if n_hot >= 4:
                aggr_items.append(f"<b>{n_hot} residues</b> in β-aggregation hotspots (ZYGGREGATOR Z ≥ 1.0)")
            else:
                aggr_items.append(f"Low aggregation propensity ({n_hot} hotspot residues)")

        # ── Post-translational modifications ──────────────────────────────
        ptm = []
        if phospho:
            n_p = sum(len(v) for v in phospho.values())
            if n_p:
                detail = "; ".join(f"{k}: {len(v)}" for k, v in phospho.items() if v)
                ptm.append(f"<b>{n_p} phosphorylation site(s)</b> predicted ({detail})")
        if motifs:
            n_m = len(motifs)
            if n_m:
                names = list(dict.fromkeys(
                    m.get("name", m.get("motif", "")) for m in motifs
                ))
                ptm.append(f"<b>{n_m} linear motif(s)</b> matched "
                           f"({', '.join(names[:4])}"
                           + (" …" if len(names) > 4 else "") + ")")

        # ── UniProt / PDB card ─────────────────────────────────────────────
        card = getattr(self, "_uniprot_card", {})
        card_items = []
        if card:
            src = card.get("source", "")
            acc = card.get("accession", "")
            name = card.get("name", "")
            if src == "UniProt":
                gene = card.get("gene", "")
                org  = card.get("organism", "")
                hdr  = f"<b>{name}</b>" if name else acc
                if gene:
                    hdr += f" &nbsp;|&nbsp; {gene}"
                if org:
                    hdr += f" &nbsp;(<i>{org}</i>)"
                hdr += f" &nbsp;<span style='color:#64748b;font-size:10px'>[{acc}]</span>"
                card_items.append(hdr)
                for t in card.get("function", [])[:2]:
                    card_items.append(t)
                subcel = card.get("subcellular", [])
                if subcel:
                    card_items.append("<b>Location:</b> " + "; ".join(subcel[:6]))
                diseases = card.get("diseases", [])
                if diseases:
                    card_items.append(
                        "<b>Disease associations:</b> " + "; ".join(diseases[:5])
                        + (" …" if len(diseases) > 5 else ""))
                ptm_notes = card.get("ptm", [])
                if ptm_notes:
                    card_items.append("<b>PTM notes:</b> " + ptm_notes[0][:300])
                kws = card.get("keywords", [])
                if kws:
                    card_items.append(
                        "<b>Keywords:</b> " + ", ".join(kws[:12])
                        + (" …" if len(kws) > 12 else ""))
            elif src == "PDB":
                card_items.append(
                    f"<b>PDB {acc}</b>" + (f" — {name}" if name else ""))

        # ── Assemble ──────────────────────────────────────────────────────
        html = (
            "<div style='font-family:sans-serif;max-width:800px'>"
            f"<h2 style='color:#1a1a2e;margin:0 0 4px'>{self._display_name()}</h2>"
            f"<p style='color:#64748b;font-size:11px;margin:0 0 10px'>"
            f"BEER v2.0 · AI Predictions analysis</p>"
        )
        if card_items:
            src_label = card.get("source", "Entry")
            html += _sec(f"{src_label} Entry", card_items)
        html += _sec("Identity", identity)
        html += _sec("Structure & Folding", struct)
        html += _sec("Charge & IDP Properties", charge)
        html += _sec("Aggregation", aggr_items)
        html += _sec("Phase Separation & RNA Binding", phase)
        html += _sec("Post-translational Modifications", ptm)
        html += "</div>"
        return html

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

        left_outer = QWidget()
        left_vb = QVBoxLayout(left_outer)
        left_vb.setContentsMargins(0, 0, 0, 0)
        left_vb.setSpacing(2)
        self._graph_filter = QLineEdit()
        self._graph_filter.setPlaceholderText("Filter graphs\u2026")
        self._graph_filter.setMaximumHeight(26)
        self._graph_filter.textChanged.connect(self._filter_graph_tree)
        left_vb.addWidget(self._graph_filter)
        left_vb.addWidget(self.graph_tree)
        outer.addWidget(left_outer)

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

        # Top bar: ROI input + Save All button
        top_bar = QHBoxLayout()
        top_bar.setSpacing(6)
        roi_lbl = QLabel("Region of Interest:")
        roi_lbl.setObjectName("roi_lbl")
        top_bar.addWidget(roi_lbl)
        self._roi_input = QLineEdit()
        self._roi_input.setPlaceholderText("e.g. 50-120")
        self._roi_input.setMaximumWidth(110)
        self._roi_input.setMaximumHeight(26)
        self._roi_input.setToolTip(
            "Highlight a residue range across all position-based graphs.\n"
            "Format: start-end (e.g. 50-120). Clear to remove highlight."
        )
        self._roi_input.returnPressed.connect(self._apply_roi_highlight)
        top_bar.addWidget(self._roi_input)
        roi_btn = QPushButton("Apply")
        roi_btn.setMinimumWidth(60)
        roi_btn.setMaximumHeight(26)
        roi_btn.clicked.connect(self._apply_roi_highlight)
        top_bar.addWidget(roi_btn)
        roi_clear = QPushButton("Clear")
        roi_clear.setMinimumWidth(54)
        roi_clear.setMaximumHeight(26)
        roi_clear.clicked.connect(self._clear_roi_highlight)
        top_bar.addWidget(roi_clear)
        top_bar.addStretch()
        self._graphs_clear_marker_btn = QPushButton("Clear Marker")
        self._graphs_clear_marker_btn.setMaximumHeight(26)
        self._graphs_clear_marker_btn.setEnabled(False)
        self._graphs_clear_marker_btn.setToolTip(
            "Remove the red dashed position marker from all profile graphs.")
        self._graphs_clear_marker_btn.clicked.connect(self._on_clear_struct_marker)
        top_bar.addWidget(self._graphs_clear_marker_btn)
        top_bar.addSpacing(4)
        self._graphs_uniprot_btn = QPushButton("UniProt Tracks")
        self._graphs_uniprot_btn.setMinimumWidth(120)
        self._graphs_uniprot_btn.setMaximumHeight(26)
        self._graphs_uniprot_btn.setToolTip(
            "Fetch UniProt feature annotations and overlay them on AI-head graphs.")
        self._graphs_uniprot_btn.setEnabled(False)
        self._graphs_uniprot_btn.clicked.connect(self.fetch_uniprot_features)
        top_bar.addWidget(self._graphs_uniprot_btn)
        top_bar.addSpacing(8)
        # (Save All Graphs removed in v2.0 — use per-graph Save Graph button)
        right_v.addLayout(top_bar)
        self._roi_start: int | None = None
        self._roi_end:   int | None = None

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
                ph.setObjectName("placeholder_lbl")
                vb.addWidget(ph)
                _ph_btn_bar = QWidget()
                _ph_btn_row = QHBoxLayout(_ph_btn_bar)
                _ph_btn_row.setContentsMargins(0, 2, 0, 2)
                _ph_btn_row.addStretch()
                save_btn = QPushButton("Save Graph")
                save_btn.setEnabled(False)
                export_btn = QPushButton("Export Data")
                export_btn.setEnabled(False)
                copy_btn = QPushButton("Copy to Clipboard")
                copy_btn.setEnabled(False)
                _ph_btn_row.addWidget(save_btn)
                _ph_btn_row.addWidget(export_btn)
                _ph_btn_row.addWidget(copy_btn)
                vb.addWidget(_ph_btn_bar)

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

    # ── colour-scheme options per colour mode ────────────────────────────────
    _STRUCT_SCHEMES = {
        "pLDDT / B-factor":    ["Red-White-Blue", "Blue-White-Red", "Rainbow", "Sinebow", "Greyscale"],
        "Residue Type":         ["Amino Acid (UniProt)", "Shapely"],
        "Chain":                ["Chain Colors"],
        "Charge":               ["Standard", "Vivid", "Pastel", "Monochrome", "Neon"],
        "Hydrophobicity":       ["Cyan-White-Orange", "Blue-White-Red", "Green-White-Red", "Thermal", "Purple-White-Green"],
        "Mass":                 ["Blue-to-Red", "Rainbow", "Sinebow", "Greyscale"],
        "Secondary Structure":  ["JMol", "PyMOL", "Pastel", "Lesk", "Cinema", "Vivid"],
        "Spectrum (N→C)":       ["Rainbow (N→C)", "Blue→Red (N→C)", "Sinebow (N→C)", "Greyscale (N→C)", "Reverse (C→N)"],
        "Solvent Accessibility": [
            "Buried→Exposed (Blue→Red)",
            "Exposed→Buried (Red→Blue)",
            "Viridis (Buried→Exposed)",
            "Plasma (Buried→Exposed)",
            "Magma (Buried→Exposed)",
            "Cyan→Orange",
        ],
        "AI Features":          ["Disorder"],   # populated dynamically in _update_scheme_combo
        "Aggregation (ZYGGREGATOR)": ["Propensity Scale", "Hotspots Only", "Fire", "Inferno", "Viridis"],
    }
    _STRUCT_MODE_KEY = {
        "pLDDT / B-factor":         "plddt",
        "Residue Type":              "residue",
        "Chain":                     "chain",
        "Charge":                    "charge",
        "Hydrophobicity":            "hydrophobicity",
        "Mass":                      "mass",
        "Secondary Structure":       "secondary_structure",
        "Spectrum (N→C)":            "spectrum",
        "Solvent Accessibility":     "sasa",
        "AI Features":               "feature",
        "Aggregation (ZYGGREGATOR)": "zyggregator",
    }
    _STRUCT_PANEL_CSS_LIGHT = """
        QScrollArea { border: none; background: transparent; }
        QWidget#structCtrl { background: transparent; }
        QGroupBox {
            font-weight: 700; font-size: 9pt; color: #3b4fc8;
            border: 1px solid #e2e6f5; border-radius: 6px;
            margin-top: 8px; padding: 10px 6px 6px 6px; background: white;
        }
        QGroupBox::title {
            subcontrol-origin: margin; subcontrol-position: top left;
            left: 8px; padding: 0 4px; color: #3b4fc8; background: white;
        }
        QPushButton {
            border: 1px solid #c8d0ec; border-radius: 5px;
            padding: 4px 8px; background: white; color: #2d3748;
            font-size: 9pt; min-height: 26px;
        }
        QPushButton:hover { background: #eef1fc; border-color: #4361ee; color: #4361ee; }
        QPushButton:pressed { background: #dce2fb; }
        QPushButton:checked { background: #4361ee; color: white; border-color: #3451c5; font-weight: 600; }
        QComboBox {
            border: 1px solid #c8d0ec; border-radius: 5px;
            padding: 3px 6px; background: white; color: #2d3748;
            font-size: 9pt; min-height: 24px;
        }
        QComboBox:hover { border-color: #4361ee; }
        QLabel { font-size: 9pt; color: #5a6787; background: transparent; }
        QCheckBox { font-size: 9pt; color: #2d3748; spacing: 6px; background: transparent; }
        QCheckBox::indicator {
            width: 14px; height: 14px;
            border: 1px solid #c8d0ec; border-radius: 3px; background: white;
        }
        QCheckBox::indicator:checked { background: #4361ee; border-color: #3451c5; }
        QTabWidget::pane {
            border: 1px solid #d1d9f0; border-radius: 0 5px 5px 5px;
            background: #f4f6fd;
        }
    """
    # Applied directly to tabBar() to work around macOS native style overriding color
    _STRUCT_TABBAR_CSS_LIGHT = """
        QTabBar::tab {
            padding: 5px 8px; min-width: 52px;
            font-size: 8.5pt; font-weight: 600;
            background: #e8eaf4; color: #2d3748;
            border: 1px solid #d1d9f0; border-bottom: none;
            border-radius: 5px 5px 0 0; margin-right: 2px;
        }
        QTabBar::tab:selected { background: #f4f6fd; color: #3b4fc8;
            border-bottom: 1px solid #f4f6fd; }
        QTabBar::tab:hover:!selected { background: #dde0f0; color: #3b4fc8; }
    """
    _STRUCT_PANEL_CSS_DARK = """
        QScrollArea { border: none; background: transparent; }
        QWidget#structCtrl { background: transparent; }
        QGroupBox {
            font-weight: 700; font-size: 9pt; color: #4cc9f0;
            border: 1px solid #2d3561; border-radius: 6px;
            margin-top: 8px; padding: 10px 6px 6px 6px; background: #16213e;
        }
        QGroupBox::title {
            subcontrol-origin: margin; subcontrol-position: top left;
            left: 8px; padding: 0 4px; color: #4cc9f0; background: #16213e;
        }
        QPushButton {
            border: 1px solid #2d3561; border-radius: 5px;
            padding: 4px 8px; background: #16213e; color: #e2e8f0;
            font-size: 9pt; min-height: 26px;
        }
        QPushButton:hover { background: #1a3a5c; border-color: #4cc9f0; color: #4cc9f0; }
        QPushButton:pressed { background: #0f3460; }
        QPushButton:checked { background: #4cc9f0; color: #1a1a2e; border-color: #3ab7dd; font-weight: 600; }
        QComboBox {
            border: 1px solid #2d3561; border-radius: 5px;
            padding: 3px 6px; background: #16213e; color: #e2e8f0;
            font-size: 9pt; min-height: 24px;
        }
        QComboBox:hover { border-color: #4cc9f0; }
        QLabel { font-size: 9pt; color: #94a3b8; background: transparent; }
        QCheckBox { font-size: 9pt; color: #e2e8f0; spacing: 6px; background: transparent; }
        QCheckBox::indicator {
            width: 14px; height: 14px;
            border: 1px solid #2d3561; border-radius: 3px; background: #16213e;
        }
        QCheckBox::indicator:checked { background: #4cc9f0; border-color: #3ab7dd; }
        QTabWidget::pane {
            border: 1px solid #1a3a5c; border-radius: 0 5px 5px 5px;
            background: #0f3460;
        }
    """
    _STRUCT_TABBAR_CSS_DARK = """
        QTabBar::tab {
            padding: 5px 8px; min-width: 52px;
            font-size: 8.5pt; font-weight: 600;
            background: #16213e; color: #c8d8e8;
            border: 1px solid #1a3a5c; border-bottom: none;
            border-radius: 5px 5px 0 0; margin-right: 2px;
        }
        QTabBar::tab:selected { background: #0f3460; color: #4cc9f0;
            border-bottom: 1px solid #0f3460; }
        QTabBar::tab:hover:!selected { background: #1a3a5c; color: #4cc9f0; }
    """

    def init_structure_tab(self):
        """Tab for interactive 3D structure viewer (PDB upload, RCSB PDB fetch, or AlphaFold fetch)."""
        container = QWidget()
        layout    = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        self.main_tabs.addTab(container, "Structure")

        # ── top info row ──────────────────────────────────────────────────────
        info_row = QHBoxLayout()
        self.af_status_lbl = QLabel("No structure loaded.  Import a PDB file, fetch a PDB ID, or fetch AlphaFold.")
        self.af_status_lbl.setObjectName("status_lbl")
        self.af_status_lbl.setProperty("status_state", "idle")
        info_row.addWidget(self.af_status_lbl, 1)
        self.export_structure_btn = QPushButton("Export Structure / Sequence")
        self.export_structure_btn.setToolTip(
            "Export as PDB, mmCIF, GRO, XYZ (requires loaded structure) or FASTA (requires analysis)")
        self.export_structure_btn.clicked.connect(self.export_structure_dialog)
        info_row.addWidget(self.export_structure_btn)
        layout.addLayout(info_row)

        if _WEBENGINE_AVAILABLE:
            # ── content: left control panel + right viewer ────────────────────
            content_row = QHBoxLayout()
            content_row.setSpacing(8)

            # ── left panel: 4-tab control widget ─────────────────────────────
            self.struct_ctrl_scroll = QTabWidget()
            self.struct_ctrl_scroll.setFixedWidth(242)
            # Fusion style makes QTabBar respect stylesheets identically on
            # macOS (native style overrides colours) and Linux (Breeze/GTK vary).
            _fusion = QStyleFactory.create("Fusion")
            if _fusion:
                self.struct_ctrl_scroll.setStyle(_fusion)
                self.struct_ctrl_scroll.tabBar().setStyle(_fusion)
            self.struct_ctrl_scroll.setStyleSheet(self._STRUCT_PANEL_CSS_LIGHT)
            self.struct_ctrl_scroll.tabBar().setStyleSheet(self._STRUCT_TABBAR_CSS_LIGHT)
            ctrl_tabs = self.struct_ctrl_scroll

            def _tab_page(label: str) -> QVBoxLayout:
                """Create a scrollable tab page; return its inner layout."""
                page = QWidget()
                page.setObjectName("structCtrl")
                vbox = QVBoxLayout(page)
                vbox.setContentsMargins(6, 8, 6, 8)
                vbox.setSpacing(6)
                scroll = QScrollArea()
                scroll.setWidgetResizable(True)
                scroll.setFrameShape(QScrollArea.Shape.NoFrame)
                scroll.setWidget(page)
                ctrl_tabs.addTab(scroll, label)
                vbox._page = page   # keep alive
                return vbox

            view_l     = _tab_page("View")
            interact_l = _tab_page("Interact")

            # ══ VIEW TAB ══════════════════════════════════════════════════════

            # ── Representation ────────────────────────────────────────────────
            rep_grp = QGroupBox("Representation")
            rep_gl = QVBoxLayout(rep_grp)
            rep_gl.setSpacing(5)
            self.struct_rep_combo = QComboBox()
            self.struct_rep_combo.addItems(["Cartoon", "Stick", "Sphere", "Line", "Cross", "Trace", "Surface"])
            self.struct_rep_combo.setToolTip(
                "Cartoon: ribbon\nStick: bonds\nSphere: VDW\nLine: wireframe\n"
                "Cross: crosshair atoms\nTrace: Cα backbone\nSurface: molecular surface")
            self.struct_rep_combo.currentTextChanged.connect(self._on_struct_rep_changed)
            rep_gl.addWidget(self.struct_rep_combo)
            view_l.addWidget(rep_grp)

            # ── Color ─────────────────────────────────────────────────────────
            color_grp = QGroupBox("Color")
            color_gl = QFormLayout(color_grp)
            color_gl.setSpacing(5)
            color_gl.setContentsMargins(6, 4, 6, 6)
            self.struct_color_mode_combo = QComboBox()
            self.struct_color_mode_combo.addItems(list(self._STRUCT_SCHEMES.keys()))
            self.struct_color_mode_combo.currentTextChanged.connect(self._on_struct_color_mode_changed)
            color_gl.addRow("Mode:", self.struct_color_mode_combo)
            self.struct_scheme_combo = QComboBox()
            self.struct_scheme_combo.currentTextChanged.connect(self._on_struct_scheme_changed)
            color_gl.addRow("Scheme:", self.struct_scheme_combo)
            self.struct_ai_gradient_lbl = QLabel("Gradient:")
            self.struct_ai_gradient_combo = QComboBox()
            self.struct_ai_gradient_combo.addItems(
                ["Hot (White→Red)", "Fire (Black→Yellow)", "Plasma",
                 "Viridis", "Cold (White→Blue)", "Classic (White→Color)"])
            self.struct_ai_gradient_combo.setToolTip(
                "Colormap applied to AI feature prediction scores.\n"
                "Only active when 'AI Features' color mode is selected.")
            self.struct_ai_gradient_combo.currentTextChanged.connect(
                self._on_struct_ai_gradient_changed)
            color_gl.addRow(self.struct_ai_gradient_lbl, self.struct_ai_gradient_combo)
            self.struct_ai_gradient_lbl.setVisible(False)
            self.struct_ai_gradient_combo.setVisible(False)
            view_l.addWidget(color_grp)

            # ── Residue Labels ────────────────────────────────────────────────
            lbl_grp = QGroupBox("Residue Labels")
            lbl_gl = QFormLayout(lbl_grp)
            lbl_gl.setSpacing(5)
            lbl_gl.setContentsMargins(6, 4, 6, 6)
            self.struct_reslbl_cb = QCheckBox("Label top residues")
            self.struct_reslbl_cb.setToolTip(
                "Show residue labels on the top-N highest-scoring residues\n"
                "for the currently selected AI Feature color mode.\n"
                "Switch to AI Features color mode first.")
            lbl_gl.addRow(self.struct_reslbl_cb)
            self.struct_reslbl_spin = QSpinBox()
            self.struct_reslbl_spin.setRange(1, 50)
            self.struct_reslbl_spin.setValue(10)
            self.struct_reslbl_spin.setEnabled(False)
            self.struct_reslbl_spin.setToolTip("Number of top-scoring residues to label")
            lbl_gl.addRow("Top N:", self.struct_reslbl_spin)
            self.struct_reslbl_cb.toggled.connect(self._on_reslbl_toggled)
            self.struct_reslbl_spin.valueChanged.connect(self._on_reslbl_n_changed)
            view_l.addWidget(lbl_grp)

            # ── Legend ────────────────────────────────────────────────────────
            legend_grp = QGroupBox("Legend")
            legend_gl = QVBoxLayout(legend_grp)
            legend_gl.setContentsMargins(8, 4, 8, 6)
            self.struct_colorbar_cb = QCheckBox("Show color bar")
            self.struct_colorbar_cb.setChecked(True)
            self.struct_colorbar_cb.toggled.connect(self._on_struct_colorbar_toggled)
            legend_gl.addWidget(self.struct_colorbar_cb)
            view_l.addWidget(legend_grp)

            # ── Overlays ─────────────────────────────────────────────────────
            ovr_grp = QGroupBox("Overlays")
            ovr_gl = QFormLayout(ovr_grp)
            ovr_gl.setContentsMargins(8, 4, 8, 6)
            ovr_gl.setSpacing(4)

            # H-bonds checkbox
            self.struct_hbond_cb = QCheckBox("H-bonds")
            self.struct_hbond_cb.setToolTip(
                "Backbone N–H···O bonds (N–O < 3.5 Å, non-adjacent residues).")
            self.struct_hbond_cb.toggled.connect(
                lambda on: self._js(f"toggleHBonds({'true' if on else 'false'});"))
            ovr_gl.addRow(self.struct_hbond_cb)

            # H-bond style row
            _hb_row = QWidget(); _hb_rl = QHBoxLayout(_hb_row)
            _hb_rl.setContentsMargins(0, 0, 0, 0); _hb_rl.setSpacing(4)
            self._hbond_color = "#44ccff"
            self._hbond_color_btn = QPushButton()
            self._hbond_color_btn.setFixedSize(22, 22)
            self._hbond_color_btn.setStyleSheet(
                f"background:{self._hbond_color};border:1px solid #ccc;border-radius:3px;")
            self._hbond_color_btn.setToolTip("H-bond colour")
            def _pick_hbond_color():
                from PySide6.QtWidgets import QColorDialog as _CD
                c = _CD.getColor(
                    parent=self, title="H-bond colour",
                    options=_CD.ColorDialogOption.ShowAlphaChannel)
                if c.isValid():
                    self._hbond_color = c.name()
                    self._hbond_color_btn.setStyleSheet(
                        f"background:{self._hbond_color};"
                        f"border:1px solid #ccc;border-radius:3px;")
                    self._js(f"setHBondStyle('{self._hbond_color}',{self._hbond_radius_sb.value():.2f});")
            self._hbond_color_btn.clicked.connect(_pick_hbond_color)
            _hb_rl.addWidget(self._hbond_color_btn)
            _hb_rl.addWidget(QLabel("r:"))
            from PySide6.QtWidgets import QDoubleSpinBox as _DSB
            self._hbond_radius_sb = _DSB()
            self._hbond_radius_sb.setRange(0.02, 0.25); self._hbond_radius_sb.setSingleStep(0.01)
            self._hbond_radius_sb.setValue(0.07); self._hbond_radius_sb.setFixedWidth(58)
            self._hbond_radius_sb.setToolTip("Cylinder radius (Å)")
            self._hbond_radius_sb.valueChanged.connect(
                lambda v: self._js(f"setHBondStyle('{self._hbond_color}',{v:.2f});"))
            _hb_rl.addWidget(self._hbond_radius_sb)
            _hb_rl.addStretch()
            ovr_gl.addRow(_hb_row)

            # Contacts checkbox + style
            self.struct_contacts_cb = QCheckBox("Contacts (8 Å)")
            self.struct_contacts_cb.setToolTip(
                "Cα–Cα pairs within 8 Å — shows the spatial contact network.\n"
                "Dense regions = packed core. Sparse = flexible loops.")
            self.struct_contacts_cb.toggled.connect(
                lambda on: self._js(f"toggleContacts({'true' if on else 'false'});"))
            ovr_gl.addRow(self.struct_contacts_cb)

            _ct_row = QWidget(); _ct_rl = QHBoxLayout(_ct_row)
            _ct_rl.setContentsMargins(0, 0, 0, 0); _ct_rl.setSpacing(4)
            self._contact_color = "#888888"
            self._contact_color_btn = QPushButton()
            self._contact_color_btn.setFixedSize(22, 22)
            self._contact_color_btn.setStyleSheet(
                f"background:{self._contact_color};border:1px solid #ccc;border-radius:3px;")
            self._contact_color_btn.setToolTip("Contact line colour")
            def _pick_contact_color():
                from PySide6.QtWidgets import QColorDialog as _CD2
                c = _CD2.getColor(parent=self, title="Contact colour")
                if c.isValid():
                    self._contact_color = c.name()
                    self._contact_color_btn.setStyleSheet(
                        f"background:{self._contact_color};"
                        f"border:1px solid #ccc;border-radius:3px;")
                    self._js(f"setContactStyle('{self._contact_color}',{self._contact_opacity_sb.value():.2f});")
            self._contact_color_btn.clicked.connect(_pick_contact_color)
            _ct_rl.addWidget(self._contact_color_btn)
            _ct_rl.addWidget(QLabel("α:"))
            self._contact_opacity_sb = _DSB()
            self._contact_opacity_sb.setRange(0.05, 1.0); self._contact_opacity_sb.setSingleStep(0.05)
            self._contact_opacity_sb.setValue(0.30); self._contact_opacity_sb.setFixedWidth(58)
            self._contact_opacity_sb.setToolTip("Line opacity (0–1)")
            self._contact_opacity_sb.valueChanged.connect(
                lambda v: self._js(f"setContactStyle('{self._contact_color}',{v:.2f});"))
            _ct_rl.addWidget(self._contact_opacity_sb)
            _ct_rl.addStretch()
            ovr_gl.addRow(_ct_row)

            view_l.addWidget(ovr_grp)

            # ── Background ───────────────────────────────────────────────────
            bg_grp = QGroupBox("Background")
            bg_gl = QGridLayout(bg_grp)
            bg_gl.setSpacing(4)
            bg_gl.setContentsMargins(6, 4, 6, 6)
            for col, (lbl, hex_) in enumerate([("Black", "#1a1a2e"), ("White", "#ffffff"), ("Grey", "#555566")]):
                b = QPushButton(lbl)
                b.clicked.connect(lambda _, c=hex_: self._js(f"setBackground('{c}');"))
                bg_gl.addWidget(b, 0, col)
            custom_bg = QPushButton("Custom color…")
            custom_bg.clicked.connect(self._pick_background_color)
            bg_gl.addWidget(custom_bg, 1, 0, 1, 3)
            view_l.addWidget(bg_grp)

            # ── Motion ────────────────────────────────────────────────────────
            motion_grp = QGroupBox("Motion")
            motion_gl = QVBoxLayout(motion_grp)
            motion_gl.setSpacing(5)
            motion_gl.setContentsMargins(6, 4, 6, 6)
            axis_row = QHBoxLayout()
            axis_lbl = QLabel("Spin axis:")
            axis_lbl.setFixedWidth(62)
            axis_row.addWidget(axis_lbl)
            self.struct_spin_axis_combo = QComboBox()
            self.struct_spin_axis_combo.addItems(["Y  (vertical)", "X  (tilt)", "Z  (roll)"])
            self.struct_spin_axis_combo.setToolTip("Axis of rotation for auto-spin")
            self.struct_spin_axis_combo.currentIndexChanged.connect(self._on_struct_spin_axis_changed)
            axis_row.addWidget(self.struct_spin_axis_combo, 1)
            motion_gl.addLayout(axis_row)
            self.struct_spin_btn = QPushButton("Spin: Off")
            self.struct_spin_btn.setCheckable(True)
            self.struct_spin_btn.setToolTip("Toggle continuous auto-rotation")
            self.struct_spin_btn.toggled.connect(self._on_struct_spin_toggled)
            motion_gl.addWidget(self.struct_spin_btn)
            view_l.addWidget(motion_grp)

            # ── Export / Reset ────────────────────────────────────────────────
            snap_grp = QGroupBox("Export")
            snap_gl = QVBoxLayout(snap_grp)
            snap_gl.setContentsMargins(6, 4, 6, 6)
            snap_gl.setSpacing(5)
            reset_btn = QPushButton("Reset View")
            reset_btn.setToolTip("Reset representation, colour, background and camera to defaults")
            reset_btn.clicked.connect(self._reset_struct_view)
            snap_gl.addWidget(reset_btn)
            snapshot_btn = QPushButton("Snapshot PNG")
            snapshot_btn.setToolTip("Render the current view to a PNG file")
            snapshot_btn.clicked.connect(self._take_structure_snapshot)
            snap_gl.addWidget(snapshot_btn)
            view_l.addWidget(snap_grp)

            # ── Chains ────────────────────────────────────────────────────────
            chains_grp = QGroupBox("Chains")
            chains_gl = QVBoxLayout(chains_grp)
            chains_gl.setSpacing(4)
            chains_gl.setContentsMargins(6, 4, 6, 6)
            chains_info = QLabel("Toggle individual chain visibility.")
            chains_info.setStyleSheet("color:#7880a8; font-size:8pt; padding-bottom:2px;")
            chains_info.setWordWrap(True)
            chains_gl.addWidget(chains_info)
            chain_btn_row = QHBoxLayout()
            chain_all_btn = QPushButton("Show All")
            chain_all_btn.setFixedHeight(26)
            chain_all_btn.clicked.connect(self._show_all_chains)
            chain_none_btn = QPushButton("Hide All")
            chain_none_btn.setFixedHeight(26)
            chain_none_btn.clicked.connect(self._hide_all_chains)
            chain_btn_row.addWidget(chain_all_btn)
            chain_btn_row.addWidget(chain_none_btn)
            chains_gl.addLayout(chain_btn_row)
            self._chain_cbs_widget = QWidget()
            self._chain_cbs_layout = QVBoxLayout(self._chain_cbs_widget)
            self._chain_cbs_layout.setContentsMargins(0, 0, 0, 0)
            self._chain_cbs_layout.setSpacing(2)
            chains_gl.addWidget(self._chain_cbs_widget)
            self._chain_checkboxes: dict = {}
            self._chains_grp = self._chain_cbs_widget
            view_l.addWidget(chains_grp)
            view_l.addStretch()

            # ══ INTERACT TAB ══════════════════════════════════════════════════

            # ── Selection ─────────────────────────────────────────────────────
            sel_grp = QGroupBox("Selection")
            sel_gl = QVBoxLayout(sel_grp)
            sel_gl.setSpacing(5)
            sel_gl.setContentsMargins(6, 4, 6, 6)
            sel_hint = QLabel("e.g.  45  ·  10-50  ·  LEU  ·  A:10-50")
            sel_hint.setStyleSheet("color:#7880a8; font-size:8pt;")
            sel_gl.addWidget(sel_hint)
            sel_row = QHBoxLayout()
            self.struct_sel_edit = QLineEdit()
            self.struct_sel_edit.setPlaceholderText("number, range, or residue name")
            self.struct_sel_edit.setToolTip(
                "Select residues by:\n"
                "  number:    45\n"
                "  range:     10-50\n"
                "  name:      LEU\n"
                "  chain:     A:10-50\n"
                "  multiple:  45, LEU, A:100-120\n\n"
                "Press Enter or click Go."
            )
            self.struct_sel_edit.returnPressed.connect(self._on_struct_selection_apply)
            sel_row.addWidget(self.struct_sel_edit, 1)
            sel_go_btn = QPushButton("Go")
            sel_go_btn.setFixedWidth(34)
            sel_go_btn.setToolTip("Apply selection")
            sel_go_btn.clicked.connect(self._on_struct_selection_apply)
            sel_row.addWidget(sel_go_btn)
            sel_gl.addLayout(sel_row)
            sel_clear_btn = QPushButton("Clear Selection")
            sel_clear_btn.setToolTip("Deselect all highlighted residues")
            sel_clear_btn.clicked.connect(self._on_struct_selection_clear)
            sel_gl.addWidget(sel_clear_btn)
            self._sel_count_lbl = QLabel("")
            self._sel_count_lbl.setStyleSheet("color:#a8b4f0; font-size:8pt; padding-top:1px;")
            sel_gl.addWidget(self._sel_count_lbl)
            interact_l.addWidget(sel_grp)

            # ── Measure ───────────────────────────────────────────────────────
            meas_grp = QGroupBox("Measure")
            meas_gl = QFormLayout(meas_grp)
            meas_gl.setSpacing(5)
            meas_gl.setContentsMargins(6, 4, 6, 6)
            self.struct_meas_mode_combo = QComboBox()
            self.struct_meas_mode_combo.addItems([
                "Distance  (2 atoms)",
                "Angle  (3 atoms)",
                "Dihedral  (4 atoms)",
            ])
            self.struct_meas_mode_combo.setToolTip(
                "Distance: exact atom–atom in Å\n"
                "Angle: three-atom bond angle in °\n"
                "Dihedral: four-atom torsion angle in °"
            )
            self.struct_meas_mode_combo.currentIndexChanged.connect(
                self._on_struct_measure_mode_changed)
            meas_gl.addRow("Type:", self.struct_meas_mode_combo)
            self._meas_hint_lbl = QLabel("Click 2 atoms on the structure")
            self._meas_hint_lbl.setStyleSheet("color:#7880a8; font-size:8pt;")
            meas_gl.addRow(self._meas_hint_lbl)
            self.struct_dist_btn = QPushButton("Pick Atoms: Off")
            self.struct_dist_btn.setCheckable(True)
            self.struct_dist_btn.setToolTip(
                "Enable atom-picking mode.\n"
                "Click atoms on the structure to measure.\n"
                "Normal click (popup) is disabled while active."
            )
            self.struct_dist_btn.toggled.connect(self._on_struct_measure_toggled)
            meas_gl.addRow(self.struct_dist_btn)
            meas_clear_btn = QPushButton("Clear Measurements")
            meas_clear_btn.setToolTip("Remove all measurement labels and lines")
            meas_clear_btn.clicked.connect(lambda: self._js("clearDistances();"))
            meas_gl.addRow(meas_clear_btn)
            interact_l.addWidget(meas_grp)

            # ── Position marker status (clear button lives in the Graphs tab top bar)
            self._marker_pos_lbl = QLabel("No marker set")
            self._marker_pos_lbl.setStyleSheet("color:#7880a8; font-size:8pt; padding:2px 0;")
            interact_l.addWidget(self._marker_pos_lbl)
            interact_l.addStretch()



            content_row.addWidget(ctrl_tabs)

            # ── 3-D viewer ────────────────────────────────────────────────────
            self.structure_viewer = QWebEngineView()
            self.structure_viewer.setMinimumHeight(500)
            # When the base page finishes loading, deliver any queued PDB
            self.structure_viewer.loadFinished.connect(self._on_structure_page_loaded)
            content_row.addWidget(self.structure_viewer, 1)

            # QWebChannel: expose Python bridge so JS can call residueClicked()
            try:
                from PySide6.QtWebChannel import QWebChannel as _QWC
                self._struct_bridge = _StructBridge(self)
                _wc = _QWC(self.structure_viewer.page())
                _wc.registerObject("bridge", self._struct_bridge)
                self.structure_viewer.page().setWebChannel(_wc)
                self._struct_webchannel = _wc   # keep alive
            except Exception:
                self._struct_bridge = None

            # Inject bundled 3Dmol.js at DocumentCreation so $3Dmol is
            # guaranteed to exist before any page script runs (avoids CDN dependency)
            try:
                import os as _os
                from PySide6.QtWebEngineCore import QWebEngineScript as _WES
                _3dmol_path = _os.path.join(_os.path.dirname(__file__), "3Dmol-min.js")
                with open(_3dmol_path, "r", encoding="utf-8") as _f:
                    _3dmol_src = _f.read()
                _s = _WES()
                _s.setName("3dmol-bundled")
                _s.setSourceCode(_3dmol_src)
                _s.setInjectionPoint(_WES.InjectionPoint.DocumentCreation)
                _s.setWorldId(_WES.ScriptWorldId.MainWorld)
                self.structure_viewer.page().scripts().insert(_s)
            except Exception:
                pass   # falls back to CDN script tag in HTML

            layout.addLayout(content_row, 1)

            # Load the base page once.  All subsequent structure swaps go via
            # loadPDB() JS call, never reloading the page.
            self.structure_viewer.setHtml(
                self._3DMOL_HTML.format(pdb_json='null'))

            # Populate scheme combo for the default mode
            self._update_scheme_combo(self.struct_color_mode_combo.currentText())

        else:
            msg = QLabel(
                "PySide6-WebEngine is not installed.\n"
                "Install it with:  pip install PySide6-WebEngine\n\n"
                "You can still export the structure and open it in PyMOL or UCSF ChimeraX."
            )
            msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
            msg.setObjectName("placeholder_lbl")
            layout.addWidget(msg, 1)
            self.structure_viewer = None

    # ─── 3-D viewer HTML ─────────────────────────────────────────────────────
    _3DMOL_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  html,body {{ margin:0; padding:0; overflow:hidden; background:#ffffff; width:100%; height:100%; }}
  #vp {{ width:100%; height:100vh; position:relative; }}

  /* ── color bar overlay ──────────────────────────────────────────────────── */
  #colorbar {{
    display:none; position:absolute; bottom:16px; right:14px;
    background:rgba(15,18,38,0.82); border-radius:8px;
    padding:10px 10px 8px 10px; font-family:system-ui,sans-serif;
    font-size:10px; color:#e8eaf0; pointer-events:none; z-index:99;
    box-shadow:0 4px 16px rgba(0,0,0,0.55);
    border:1px solid rgba(255,255,255,0.10); user-select:none;
  }}
  #colorbar.cb-wide {{ min-width:110px; }}
  #cb-title {{
    text-align:center; font-size:9.5px; font-weight:700; letter-spacing:0.04em;
    color:#a8b4f0; margin-bottom:6px; white-space:nowrap;
  }}
  #cb-bar-wrap {{ display:flex; align-items:stretch; gap:6px; }}
  #cb-gradient {{
    width:16px; min-height:100px; border-radius:4px;
    border:1px solid rgba(255,255,255,0.12); flex-shrink:0;
  }}
  #cb-tick-col {{ display:flex; flex-direction:column; justify-content:space-between; min-width:30px; }}
  .cb-tick {{ font-size:9px; color:#c8d0ec; white-space:nowrap; }}
  .cb-tick-top {{ text-align:left; }}
  .cb-tick-mid {{ text-align:left; color:#8890b8; }}
  .cb-tick-bot {{ text-align:left; }}
  #cb-unit {{ text-align:center; font-size:8.5px; color:#7880a8; margin-top:5px; }}
  /* categorical */
  #cb-entries {{ display:none; }}
  .cb-entry {{ display:flex; align-items:center; gap:7px; margin-bottom:4px; white-space:nowrap; }}
  .cb-swatch {{ width:12px; height:12px; border-radius:3px; flex-shrink:0; border:1px solid rgba(255,255,255,0.2); }}

  /* ── residue popup ─────────────────────────────────────────────────────── */
  #residue-popup {{
    display:none; position:absolute; top:14px; left:14px;
    background:rgba(10,12,30,0.92); border-radius:8px;
    padding:10px 13px 8px 13px; font-family:system-ui,sans-serif;
    font-size:11px; color:#e8eaf0; z-index:200;
    box-shadow:0 4px 18px rgba(0,0,0,0.65);
    border:1px solid rgba(255,215,0,0.45); min-width:190px;
    pointer-events:auto;
  }}
  #popup-title {{
    font-weight:700; font-size:12px; color:#FFD700; margin-bottom:7px;
    border-bottom:1px solid rgba(255,215,0,0.3); padding-bottom:4px;
  }}
  .popup-row {{ display:flex; justify-content:space-between; gap:12px; margin:3px 0; }}
  .popup-label {{ color:#94a3b8; }}
  .popup-val {{ color:#e2e8f0; font-weight:600; }}
  .popup-bar-wrap {{ width:60px; height:8px; background:#1e293b; border-radius:4px; margin-top:1px; overflow:hidden; }}
  .popup-bar {{ height:100%; border-radius:4px; }}
  #popup-close {{
    position:absolute; top:6px; right:8px; cursor:pointer;
    color:#94a3b8; font-size:13px; line-height:1;
  }}
  #popup-close:hover {{ color:#ffd700; }}


</style>
</head><body>
<div id="vp">
  <div id="colorbar">
    <div id="cb-title"></div>
    <div id="cb-bar-wrap">
      <div id="cb-gradient"></div>
      <div id="cb-tick-col">
        <span class="cb-tick cb-tick-top" id="cb-tick-max"></span>
        <span class="cb-tick cb-tick-mid" id="cb-tick-mid"></span>
        <span class="cb-tick cb-tick-bot" id="cb-tick-min"></span>
      </div>
    </div>
    <div id="cb-unit"></div>
    <div id="cb-entries"></div>
  </div>
  <div id="residue-popup">
    <span id="popup-close" onclick="closePopup()">✕</span>
    <div id="popup-title">—</div>
    <div id="popup-rows"></div>
  </div>
</div>
<script src="qrc:///qtwebchannel/qwebchannel.js"></script>
<!-- 3Dmol.js injected at DocumentCreation via QWebEngineScript (bundled in package) -->
<script>
var viewer   = null;
var pdbData  = {pdb_json};       // null on initial empty load
var repMode  = 'cartoon';
var colorMode   = 'plddt';
var colorScheme = 'Red-White-Blue';
var repOpacity  = 0.90;
var colorBarVisible = true;
var surfaceObj  = null;   // numeric surface ID (set in addSurface .then callback)
var surfaceGen  = 0;      // generation counter — incremented on every applyStyle call
var allChains   = [];     // all chain IDs present in the current model
var seqLen      = 0;      // protein length — set by setColorMode for Spectrum mode

// ── Python bridge (graph↔structure bidirectional link) ────────────────────
var _bridge = null;
if(typeof QWebChannel !== 'undefined' && typeof qt !== 'undefined') {{
    new QWebChannel(qt.webChannelTransport, function(ch) {{
        _bridge = ch.objects.bridge;
    }});
}}

// ── Kyte-Doolittle ──────────────────────────────────────────────────────────
var KD = {{'ILE':4.5,'VAL':4.2,'LEU':3.8,'PHE':2.8,'CYS':2.5,'MET':1.9,'ALA':1.8,
           'GLY':-0.4,'THR':-0.7,'SER':-0.8,'TRP':-0.9,'TYR':-1.3,'PRO':-1.6,
           'HIS':-3.2,'GLU':-3.5,'GLN':-3.5,'ASP':-3.5,'ASN':-3.5,'LYS':-3.9,'ARG':-4.5}};

// ── Residue masses (Da) ────────────────────────────────────────────────────
var MASS = {{'ALA':89,'ARG':174,'ASN':132,'ASP':133,'CYS':121,'GLN':146,'GLU':147,
             'GLY':75,'HIS':155,'ILE':131,'LEU':131,'LYS':146,'MET':149,'PHE':165,
             'PRO':115,'SER':105,'THR':119,'TRP':204,'TYR':181,'VAL':117}};

function lerp(a,b,t){{ return a+(b-a)*t; }}
function rgb(r,g,b){{ return 'rgb('+Math.round(r)+','+Math.round(g)+','+Math.round(b)+')'; }}

// ── hydrophobicity colorfuncs ──────────────────────────────────────────────
function _hydro_CWO(atom){{
    var kd=KD[atom.resn]!==undefined?KD[atom.resn]:0;
    var t=(kd+4.5)/9.0;
    if(t<0.5){{ var s=t*2; return rgb(lerp(44,255,s),lerp(123,255,s),255); }}
    var s=(t-0.5)*2; return rgb(255,lerp(255,140,s),lerp(255,0,s));
}}
function _hydro_BWR(atom){{
    var kd=KD[atom.resn]!==undefined?KD[atom.resn]:0;
    var t=(kd+4.5)/9.0;
    if(t<0.5){{ var s=t*2; return rgb(lerp(0,255,s),lerp(0,255,s),255); }}
    var s=(t-0.5)*2; return rgb(255,lerp(255,0,s),lerp(255,0,s));
}}
function _hydro_GWR(atom){{
    var kd=KD[atom.resn]!==undefined?KD[atom.resn]:0;
    var t=(kd+4.5)/9.0;
    if(t<0.5){{ var s=t*2; return rgb(lerp(0,255,s),255,lerp(0,255,s)); }}
    var s=(t-0.5)*2; return rgb(255,lerp(255,0,s),0);
}}

// ── mass colorfuncs ────────────────────────────────────────────────────────
function _mass_BR(atom){{
    var m=MASS[atom.resn]!==undefined?MASS[atom.resn]:110;
    var t=Math.max(0,Math.min(1,(m-75)/129.0));
    return rgb(lerp(68,220,t),lerp(119,60,t),lerp(204,68,t));
}}
function _mass_RB(atom){{
    var m=MASS[atom.resn]!==undefined?MASS[atom.resn]:110;
    var t=Math.max(0,Math.min(1,(m-75)/129.0));
    var seg=t*4; var i=Math.min(Math.floor(seg),3); var f=seg-i;
    var cols=[[0,0,255],[0,255,255],[0,255,0],[255,255,0],[255,0,0]];
    return rgb(lerp(cols[i][0],cols[i+1][0],f),lerp(cols[i][1],cols[i+1][1],f),lerp(cols[i][2],cols[i+1][2],f));
}}

// ── charge colorfuncs ─────────────────────────────────────────────────────
function _charge(atom){{   // Standard
    if(['ARG','LYS','HIS'].indexOf(atom.resn)>=0) return '#5588ff';
    if(['ASP','GLU'].indexOf(atom.resn)>=0)        return '#ff5555';
    return '#aaaaaa';
}}
function _charge_vivid(atom){{
    if(['ARG','LYS','HIS'].indexOf(atom.resn)>=0) return '#0033ff';
    if(['ASP','GLU'].indexOf(atom.resn)>=0)        return '#ff0000';
    return '#cccccc';
}}
function _charge_pastel(atom){{
    if(['ARG','LYS','HIS'].indexOf(atom.resn)>=0) return '#99b3ff';
    if(['ASP','GLU'].indexOf(atom.resn)>=0)        return '#ffaaaa';
    return '#eeeeee';
}}
function _charge_mono(atom){{
    if(['ARG','LYS','HIS'].indexOf(atom.resn)>=0) return '#ffffff';
    if(['ASP','GLU'].indexOf(atom.resn)>=0)        return '#222222';
    return '#888888';
}}

// ── hydrophobicity extra colorfuncs ───────────────────────────────────────
function _hydro_thermal(atom){{
    var kd=KD[atom.resn]!==undefined?KD[atom.resn]:0;
    var t=(kd+4.5)/9.0;
    if(t<0.5){{ var s=t*2; return rgb(lerp(0,255,s),lerp(60,200,s),lerp(200,60,s)); }}
    var s=(t-0.5)*2; return rgb(255,lerp(200,80,s),lerp(60,0,s));
}}
function _hydro_PWG(atom){{
    var kd=KD[atom.resn]!==undefined?KD[atom.resn]:0;
    var t=(kd+4.5)/9.0;
    if(t<0.5){{ var s=t*2; return rgb(lerp(128,255,s),lerp(0,255,s),lerp(128,255,s)); }}
    var s=(t-0.5)*2; return rgb(lerp(255,0,s),lerp(255,128,s),lerp(255,0,s));
}}

// ── mass extra colorfuncs ─────────────────────────────────────────────────
function _mass_sinebow(atom){{
    var m=MASS[atom.resn]!==undefined?MASS[atom.resn]:110;
    var t=Math.max(0,Math.min(1,(m-75)/129.0));
    return rgb(
        Math.round(255*Math.pow(Math.sin(Math.PI*(t+0)),2)),
        Math.round(255*Math.pow(Math.sin(Math.PI*(t+1/3)),2)),
        Math.round(255*Math.pow(Math.sin(Math.PI*(t+2/3)),2))
    );
}}
function _mass_grey(atom){{
    var m=MASS[atom.resn]!==undefined?MASS[atom.resn]:110;
    var t=Math.max(0,Math.min(1,(m-75)/129.0));
    var v=Math.round(lerp(30,220,t)); return rgb(v,v,v);
}}

// ── secondary structure colorfuncs ────────────────────────────────────────
function _ss_jmol(atom){{
    var s=atom.ss;
    if(s==='h'||s==='H') return '#FF0080';
    if(s==='s'||s==='S') return '#FFFF00';
    return '#FFFFFF';
}}
function _ss_pymol(atom){{
    var s=atom.ss;
    if(s==='h'||s==='H') return '#FF6666';
    if(s==='s'||s==='S') return '#6699FF';
    return '#CCCCCC';
}}
function _ss_pastel(atom){{
    var s=atom.ss;
    if(s==='h'||s==='H') return '#ffb3c6';
    if(s==='s'||s==='S') return '#b3d9ff';
    return '#e8e8e8';
}}
function _ss_lesk(atom){{
    var s=atom.ss;
    if(s==='h'||s==='H') return '#ff0000';
    if(s==='s'||s==='S') return '#ffff00';
    return '#ffffff';
}}
function _ss_cinema(atom){{
    // CLC / Cinema scheme: helix=deep sky blue, strand=dark orange, coil=light grey
    var s=atom.ss;
    if(s==='h'||s==='H') return '#00BFFF';
    if(s==='s'||s==='S') return '#FF8C00';
    return '#dddddd';
}}
function _ss_vivid(atom){{
    // High-saturation: helix=magenta, strand=cyan, coil=dark grey
    var s=atom.ss;
    if(s==='h'||s==='H') return '#ee00ee';
    if(s==='s'||s==='S') return '#00dddd';
    return '#555555';
}}

// ── pLDDT greyscale ───────────────────────────────────────────────────────
function _plddt_grey(atom){{
    var t=Math.max(0,Math.min(1,(atom.b||0)/100.0));
    var v=Math.round(lerp(30,220,t)); return rgb(v,v,v);
}}

// ── spectrum (N→C) colorfuncs & scheme ────────────────────────────────────
function _spectrum_grey(atom){{
    var t=seqLen>1?Math.max(0,Math.min(1,(atom.resi-1)/(seqLen-1))):0;
    var v=Math.round(lerp(30,220,t)); return rgb(v,v,v);
}}
function _spectrumScheme(){{
    var mx=seqLen>0?seqLen:9999;
    if(colorScheme==='Rainbow (N\u2192C)')        return {{gradient:'roygb',prop:'resi',min:mx,max:1}};
    if(colorScheme==='Blue\u2192Red (N\u2192C)') return {{gradient:'rwb',prop:'resi',min:1,max:mx}};
    if(colorScheme==='Sinebow (N\u2192C)')        return {{gradient:'sinebow',prop:'resi',min:1,max:mx}};
    if(colorScheme==='Reverse (C\u2192N)')        return {{gradient:'roygb',prop:'resi',min:1,max:mx}};
    if(colorScheme==='Greyscale (N\u2192C)')      return null;
    return {{gradient:'roygb',prop:'resi',min:mx,max:1}};
}}

function _getColorFunc(){{
    if(colorMode==='hydrophobicity'){{
        if(colorScheme==='Blue-White-Red')      return _hydro_BWR;
        if(colorScheme==='Green-White-Red')     return _hydro_GWR;
        if(colorScheme==='Thermal')             return _hydro_thermal;
        if(colorScheme==='Purple-White-Green')  return _hydro_PWG;
        return _hydro_CWO;
    }}
    if(colorMode==='mass'){{
        if(colorScheme==='Rainbow')   return _mass_RB;
        if(colorScheme==='Sinebow')   return _mass_sinebow;
        if(colorScheme==='Greyscale') return _mass_grey;
        return _mass_BR;
    }}
    if(colorMode==='charge'){{
        if(colorScheme==='Vivid')      return _charge_vivid;
        if(colorScheme==='Pastel')     return _charge_pastel;
        if(colorScheme==='Monochrome') return _charge_mono;
        if(colorScheme==='Neon')       return _charge_neon;
        return _charge;
    }}
    if(colorMode==='secondary_structure'){{
        if(colorScheme==='Pastel')  return _ss_pastel;
        if(colorScheme==='Lesk')    return _ss_lesk;
        if(colorScheme==='Cinema')  return _ss_cinema;
        if(colorScheme==='Vivid')   return _ss_vivid;
        return colorScheme==='PyMOL' ? _ss_pymol : _ss_jmol;
    }}
    if(colorMode==='plddt' && colorScheme==='Greyscale') return _plddt_grey;
    if(colorMode==='spectrum' && colorScheme==='Greyscale (N\u2192C)') return _spectrum_grey;
    if(colorMode==='feature') return _feature_colorfunc;
    if(colorMode==='resi_colormap') return _resiColorMap_func;
    return null;
}}

// ── pLDDT colorscheme object (string-based gradients — constructor objects
//    cause range().length errors in some 3Dmol CDN builds) ─────────────────
function _plddtScheme(){{
    // "rwb"   : min value → red, max value → blue  (AlphaFold convention)
    // reversed min/max achieves blue-at-low / red-at-high
    if(colorScheme==='Blue-White-Red') return {{prop:'b',gradient:'rwb',min:100,max:0}};
    if(colorScheme==='Rainbow')        return {{prop:'b',gradient:'roygb',min:0,max:100}};
    if(colorScheme==='Sinebow')        return {{prop:'b',gradient:'sinebow',min:0,max:100}};
    return {{prop:'b',gradient:'rwb',min:0,max:100}};  // Red-White-Blue (default)
}}

// ── build per-representation style options ─────────────────────────────────
function _styleOpts(rep,opacity){{
    var op=opacity!==undefined?opacity:repOpacity;
    var o={{}};
    if(colorMode==='plddt'){{
        if(colorScheme==='Greyscale') o[rep]={{colorfunc:_plddt_grey,opacity:op}};
        else o[rep]={{colorscheme:_plddtScheme(),opacity:op}};
    }} else if(colorMode==='residue'){{
        o[rep]={{colorscheme:colorScheme==='Shapely'?'shapely':'amino',opacity:op}};
    }} else if(colorMode==='chain'){{
        o[rep]={{colorscheme:'chain',opacity:op}};
    }} else if(colorMode==='spectrum'){{
        var spec=_spectrumScheme();
        if(spec===null) o[rep]={{colorfunc:_spectrum_grey,opacity:op}};
        else if(typeof spec==='string') o[rep]={{colorscheme:spec,opacity:op}};
        else o[rep]={{colorscheme:spec,opacity:op}};
    }} else {{
        o[rep]={{colorfunc:_getColorFunc(),opacity:op}};
    }}
    return o;
}}

// ── chain visibility ───────────────────────────────────────────────────────
var hiddenChains = {{}};   // {{chainId: true}} for hidden chains

function setChainVisible(chainId, visible){{
    if(visible) delete hiddenChains[chainId];
    else hiddenChains[chainId]=true;
    applyStyle();
}}
function showAllChains(){{ hiddenChains={{}}; applyStyle(); }}
function hideAllChains(chainList){{
    hiddenChains={{}};
    if(chainList){{ chainList.forEach(function(c){{ hiddenChains[c]=true; }}); }}
    applyStyle();
}}

// ── apply representation ───────────────────────────────────────────────────
function applyStyle(){{
    if(!viewer) return;

    // Remove any existing surface.  surfaceObj holds the numeric ID that was
    // stored asynchronously in the addSurface .then() callback.
    if(surfaceObj!==null){{ viewer.removeSurface(surfaceObj); surfaceObj=null; }}

    // Bump the generation so any in-flight addSurface callback knows it's stale.
    surfaceGen++;
    var myGen=surfaceGen;

    var hiddenList=Object.keys(hiddenChains);

    if(repMode==='surface'){{
        // Show a ghost cartoon immediately while the surface is computing.
        viewer.setStyle({{}},_styleOpts('cartoon',Math.min(repOpacity*0.22,0.20)));
        // Hide hidden chains in ghost too
        hiddenList.forEach(function(c){{ viewer.setStyle({{chain:c}},{{}}); }});
        var sOpts={{opacity:repOpacity}};
        if(colorMode==='plddt'){{
            if(colorScheme==='Greyscale') sOpts.colorfunc=_plddt_grey;
            else sOpts.colorscheme=_plddtScheme();
        }} else if(colorMode==='residue') sOpts.colorscheme=colorScheme==='Shapely'?'shapely':'amino';
        else if(colorMode==='chain')   sOpts.colorscheme='chain';
        else if(colorMode==='spectrum'){{
            var spec=_spectrumScheme();
            if(spec===null) sOpts.colorfunc=_spectrum_grey;
            else if(typeof spec==='string') sOpts.colorscheme=spec;
            else sOpts.colorscheme=spec;
        }} else                           sOpts.colorfunc=_getColorFunc();
        // Build chain selector: only generate surface for visible chains.
        var visibleChains = hiddenList.length > 0
            ? allChains.filter(function(c){{ return !hiddenChains[c]; }})
            : [];
        var surfaceSel = (visibleChains.length > 0) ? {{chain: visibleChains}} : {{}};
        // addSurface is async — store the ID only when the Promise resolves.
        viewer.addSurface($3Dmol.SurfaceType.MS,sOpts,surfaceSel).then(function(id){{
            if(surfaceGen===myGen){{
                surfaceObj=id;   // still the current generation — keep it
            }} else {{
                viewer.removeSurface(id);  // user switched away — discard immediately
                viewer.render();
            }}
        }});
    }} else {{
        var opts=_styleOpts(repMode);
        if(repMode==='sphere') opts[repMode].radius=0.5;
        viewer.setStyle({{}},opts);
        // Override hidden chains with empty style (invisible)
        hiddenList.forEach(function(c){{ viewer.setStyle({{chain:c}},{{}}); }});
    }}
    viewer.render();
    updateColorBar();
}}

// ── color bar ─────────────────────────────────────────────────────────────
var _CB = {{
  plddt:{{
    'Red-White-Blue':{{css:'linear-gradient(to top,#cc0000,#ffffff,#0044dd)',min:'0',mid:'50',max:'100',unit:'pLDDT score'}},
    'Blue-White-Red':{{css:'linear-gradient(to top,#0044dd,#ffffff,#cc0000)',min:'0',mid:'50',max:'100',unit:'pLDDT score'}},
    'Rainbow':       {{css:'linear-gradient(to top,#ff0000,#ff8800,#ffff00,#00cc00,#0000ff)',min:'0',mid:'50',max:'100',unit:'pLDDT score'}},
    'Sinebow':       {{css:'linear-gradient(to top,#ff4040,#ffcc00,#40ff80,#0080ff,#cc00ff)',min:'0',mid:'50',max:'100',unit:'pLDDT score'}},
  }},
  hydrophobicity:{{
    'Cyan-White-Orange':{{css:'linear-gradient(to top,#00b4d8,#caf0f8,#ffffff,#ffd6a5,#ff8800)',min:'-4.5',mid:'0.0',max:'+4.5',unit:'Kyte-Doolittle'}},
    'Blue-White-Red':   {{css:'linear-gradient(to top,#3b82f6,#ffffff,#ef4444)',min:'-4.5',mid:'0.0',max:'+4.5',unit:'Kyte-Doolittle'}},
    'Green-White-Red':  {{css:'linear-gradient(to top,#22c55e,#ffffff,#ef4444)',min:'-4.5',mid:'0.0',max:'+4.5',unit:'Kyte-Doolittle'}},
  }},
  mass:{{
    'Blue-to-Red':{{css:'linear-gradient(to top,#4477cc,#88bbee,#eeddaa,#cc5522)',min:'75 Da',mid:'~140',max:'204 Da',unit:'Residue mass (Da)'}},
    'Rainbow':    {{css:'linear-gradient(to top,#0000ff,#00ffff,#00ff00,#ffff00,#ff0000)',min:'75 Da',mid:'~140',max:'204 Da',unit:'Residue mass (Da)'}},
  }},
  charge:   {{type:'cat',entries:[
    {{c:'#5588ff',l:'Positive (K/R/H)'}},{{c:'#ff5555',l:'Negative (D/E)'}},{{c:'#aaaaaa',l:'Neutral'}}
  ]}},
  residue:  {{type:'cat',entries:[
    {{c:'#64F73F',l:'Nonpolar (A/V/I/L/M/F/W/P)'}},{{c:'#12D3FF',l:'Polar (S/T/C/Y/N/Q)'}},
    {{c:'#FF2655',l:'Negative (D/E)'}},{{c:'#4550FA',l:'Positive (K/R/H)'}},{{c:'#E2E2E2',l:'Glycine (G)'}}
  ]}},
  chain:    {{type:'cat',entries:[
    {{c:'#ff8800',l:'Chain A'}},{{c:'#00aaff',l:'Chain B'}},
    {{c:'#ff44cc',l:'Chain C'}},{{c:'#44ff88',l:'Chain D'}},{{c:'#bbbbbb',l:'+ others'}}
  ]}},
  secondary_structure: {{type:'cat',entries:[
    {{c:'#FF0080',l:'Helix (JMol: pink / PyMOL: salmon)'}},
    {{c:'#FFFF00',l:'Sheet (JMol: yellow / PyMOL: blue)'}},
    {{c:'#FFFFFF',l:'Coil / loop'}}
  ]}},
  spectrum: {{
    'Rainbow (N\u2192C)':   {{css:'linear-gradient(to top,#0000ff,#00ffff,#00ff00,#ffff00,#ff0000)',min:'N-term',mid:'middle',max:'C-term',unit:'Sequence position'}},
    'Blue\u2192Red (N\u2192C)': {{css:'linear-gradient(to top,#0000ff,#ffffff,#ff0000)',min:'N-term',mid:'middle',max:'C-term',unit:'Sequence position'}},
    'Sinebow (N\u2192C)':   {{css:'linear-gradient(to top,#4040ff,#40ffff,#40ff40,#ffff40,#ff4040)',min:'N-term',mid:'middle',max:'C-term',unit:'Sequence position'}},
    'Greyscale (N\u2192C)': {{css:'linear-gradient(to top,#1e1e1e,#dddddd)',min:'N-term',mid:'middle',max:'C-term',unit:'Sequence position'}},
    'Reverse (C\u2192N)':   {{css:'linear-gradient(to top,#ff0000,#ffff00,#00ff00,#00ffff,#0000ff)',min:'N-term',mid:'middle',max:'C-term',unit:'Sequence position'}},
  }},
  feature: {{
    '_dyn': true,
  }},
  resi_colormap: {{
    '_dyn': true,
  }},
}};

function updateColorBar(){{
    var bar=document.getElementById('colorbar');
    if(!colorBarVisible){{ bar.style.display='none'; return; }}
    var meta=_CB[colorMode];
    if(!meta){{ bar.style.display='none'; return; }}
    bar.style.display='block';
    var title=document.getElementById('cb-title');
    var grad =document.getElementById('cb-gradient');
    var tmax =document.getElementById('cb-tick-max');
    var tmid =document.getElementById('cb-tick-mid');
    var tmin =document.getElementById('cb-tick-min');
    var unit =document.getElementById('cb-unit');
    var ents =document.getElementById('cb-entries');
    var wrap =document.getElementById('cb-bar-wrap');

    if(colorMode==='feature'){{
        wrap.style.display='flex'; unit.style.display='block';
        ents.style.display='none'; bar.classList.remove('cb-wide');
        title.textContent=featureName;
        grad.style.background='linear-gradient(to top,#ffffff,'+featureColor+')';
        tmax.textContent='1.0'; tmid.textContent='0.5'; tmin.textContent='0.0';
        unit.textContent='Prediction score'; bar.style.display='block'; return;
    }}
    if(colorMode==='resi_colormap'){{
        wrap.style.display='flex'; unit.style.display='block';
        ents.style.display='none'; bar.classList.remove('cb-wide');
        title.textContent=_resiColorMapName||'Solvent Accessibility';
        // Show scheme name as the gradient label; actual colors vary per residue
        grad.style.background='linear-gradient(to top,#313695,#ffffbf,#a50026)';
        tmax.textContent='1.0 (exposed)'; tmid.textContent='0.5'; tmin.textContent='0.0 (buried)';
        unit.textContent='RSA'; bar.style.display='block'; return;
    }}
    if(meta.type==='cat'){{
        wrap.style.display='none'; unit.style.display='none';
        ents.style.display='block'; bar.classList.add('cb-wide');
        var label={{'charge':'Charge','residue':'Residue type','chain':'Chain'}};
        title.textContent=label[colorMode]||colorMode;
        ents.innerHTML='';
        meta.entries.forEach(function(e){{
            var row=document.createElement('div'); row.className='cb-entry';
            var sw=document.createElement('div');  sw.className='cb-swatch'; sw.style.background=e.c;
            var lbl=document.createElement('span'); lbl.textContent=e.l;
            row.appendChild(sw); row.appendChild(lbl); ents.appendChild(row);
        }});
    }} else {{
        var cfg=meta[colorScheme]||Object.values(meta)[0];
        wrap.style.display='flex'; unit.style.display='block';
        ents.style.display='none'; bar.classList.remove('cb-wide');
        title.textContent=colorMode==='plddt'?'pLDDT':colorMode==='hydrophobicity'?'Hydrophobicity':colorMode==='spectrum'?'Sequence Position':'Mass';
        grad.style.background=cfg.css;
        tmax.textContent=cfg.max; tmid.textContent=cfg.mid; tmin.textContent=cfg.min;
        unit.textContent=cfg.unit;
    }}
}}

// ── public API ─────────────────────────────────────────────────────────────
function setRepresentation(r)  {{ repMode=r; applyStyle(); }}
function setColorMode(m,s,extra){{ colorMode=m; if(s) colorScheme=s; if(extra) seqLen=extra; applyStyle(); updateColorBar(); }}
function setScheme(s,extra)    {{ colorScheme=s; if(extra) seqLen=extra; applyStyle(); updateColorBar(); }}
function setBackground(c)      {{ document.documentElement.style.background=c; document.body.style.background=c; if(viewer){{ viewer.setBackgroundColor(c); viewer.render(); }} }}
function setColorBarVisible(v) {{ colorBarVisible=v; updateColorBar(); }}
function setSpin(on,axis)      {{ if(!viewer) return; if(on) viewer.spin(axis||'y',1); else viewer.spin(false); viewer.render(); }}
function resetView()           {{
    repMode='cartoon'; colorMode='plddt'; colorScheme='Red-White-Blue';
    hiddenChains={{}};
    setBackground('#ffffff');
    if(viewer){{ viewer.spin(false); viewer.zoomTo(); }}
    applyStyle();
}}

// ── Neon charge colorfunc ──────────────────────────────────────────────────
function _charge_neon(atom){{
    if(['ARG','LYS','HIS'].indexOf(atom.resn)>=0) return '#0088ff';
    if(['ASP','GLU'].indexOf(atom.resn)>=0)        return '#ff1155';
    return '#aaaaaa';
}}

// ── H-bond / contact overlays ──────────────────────────────────────────────
var hbondVisible    = false;
var contactsVisible = false;
var hbondShapes     = [];
var contactShapes   = [];
var hbondColor      = '#44ccff';
var hbondRadius     = 0.07;
var contactColor    = '#888888';
var contactOpacity  = 0.30;

function setHBondStyle(color, radius){{
    hbondColor  = color  || '#44ccff';
    hbondRadius = radius || 0.07;
    if(hbondVisible){{ toggleHBonds(false); toggleHBonds(true); }}
}}
function setContactStyle(color, opacity){{
    contactColor   = color   || '#888888';
    contactOpacity = opacity || 0.30;
    if(contactsVisible){{ toggleContacts(false); toggleContacts(true); }}
}}

function _detectBackboneHBonds(){{
    if(!viewer) return [];
    var donors    = viewer.selectedAtoms({{atom:'N'}});
    var acceptors = viewer.selectedAtoms({{atom:'O'}});
    var cutSq = 3.5*3.5;
    var out = [];
    donors.forEach(function(d){{
        acceptors.forEach(function(a){{
            if(d.chain===a.chain && Math.abs(d.resi-a.resi)<=2) return;
            var dx=d.x-a.x, dy=d.y-a.y, dz=d.z-a.z;
            if(dx*dx+dy*dy+dz*dz < cutSq)
                out.push({{x1:d.x,y1:d.y,z1:d.z,x2:a.x,y2:a.y,z2:a.z}});
        }});
    }});
    return out;
}}

function toggleHBonds(on){{
    hbondVisible = on;
    hbondShapes.forEach(function(s){{ try{{viewer.removeShape(s);}}catch(e){{}} }});
    hbondShapes = [];
    if(on && viewer){{
        _detectBackboneHBonds().forEach(function(h){{
            hbondShapes.push(viewer.addCylinder({{
                start:{{x:h.x1,y:h.y1,z:h.z1}},
                end:{{x:h.x2,y:h.y2,z:h.z2}},
                radius:hbondRadius, color:hbondColor, opacity:0.72,
                dashed:true, fromCap:0, toCap:0
            }}));
        }});
        viewer.render();
    }} else if(viewer){{ viewer.render(); }}
}}

function toggleContacts(on){{
    contactsVisible = on;
    contactShapes.forEach(function(s){{ try{{viewer.removeShape(s);}}catch(e){{}} }});
    contactShapes = [];
    if(on && viewer){{
        var cas = viewer.selectedAtoms({{atom:'CA'}});
        var n = cas.length;
        var cutSq = 8*8;
        for(var i=0;i<n;i++){{
            for(var j=i+2;j<n;j++){{
                var a=cas[i],b=cas[j];
                var dx=a.x-b.x,dy=a.y-b.y,dz=a.z-b.z;
                if(dx*dx+dy*dy+dz*dz < cutSq){{
                    contactShapes.push(viewer.addLine({{
                        start:{{x:a.x,y:a.y,z:a.z}},
                        end:{{x:b.x,y:b.y,z:b.z}},
                        color:contactColor, linewidth:0.5, opacity:contactOpacity
                    }}));
                }}
            }}
        }}
        viewer.render();
    }} else if(viewer){{ viewer.render(); }}
}}

// ── Feature score coloring ────────────────────────────────────────────────
var featureScores = {{}};   // {{resi(1-based): score 0..1}}
var featureColor  = '#f3722c';
var featureName   = 'disorder';
var featureGradient = 'hot';   // 'hot' | 'plasma' | 'viridis' | 'fire' | 'cold' | 'classic'

function _hexToRgb(h){{
    var r=parseInt(h.slice(1,3),16),g=parseInt(h.slice(3,5),16),b=parseInt(h.slice(5,7),16);
    return [r,g,b];
}}

// Gradient helpers: t in [0,1] → rgb(...)
function _gradient_hot(t){{
    // white → yellow → orange → red
    if(t<0.5){{var s=t*2; return 'rgb(255,'+Math.round(255-s*55)+','+Math.round(255*(1-s))+')';}}
    var s=(t-0.5)*2;
    return 'rgb(255,'+Math.round(200*(1-s))+',0)';
}}
function _gradient_fire(t){{
    // black → red → orange → yellow (afmhot-like)
    if(t<0.33){{var s=t/0.33; return 'rgb('+Math.round(s*220)+',0,0)';}}
    if(t<0.66){{var s=(t-0.33)/0.33; return 'rgb(220,'+Math.round(s*140)+',0)';}}
    var s=(t-0.66)/0.34;
    return 'rgb('+Math.round(220+s*35)+','+Math.round(140+s*115)+','+Math.round(s*200)+')';
}}
function _gradient_plasma(t){{
    // simplified plasma: dark purple → pink-purple → orange → yellow
    var stops=[
        [13,8,135],[84,2,163],[139,10,165],[185,50,137],
        [219,92,104],[244,136,73],[253,187,44],[240,249,33]
    ];
    var n=stops.length-1;
    var i=Math.min(Math.floor(t*n),n-1);
    var f=t*n-i;
    var a=stops[i],b=stops[i+1];
    return 'rgb('+Math.round(a[0]+(b[0]-a[0])*f)+','+
                  Math.round(a[1]+(b[1]-a[1])*f)+','+
                  Math.round(a[2]+(b[2]-a[2])*f)+')';
}}
function _gradient_viridis(t){{
    var stops=[
        [68,1,84],[72,40,120],[62,83,160],[49,124,183],
        [38,173,166],[53,183,121],[110,206,88],[180,222,44],[253,231,37]
    ];
    var n=stops.length-1;
    var i=Math.min(Math.floor(t*n),n-1);
    var f=t*n-i;
    var a=stops[i],b=stops[i+1];
    return 'rgb('+Math.round(a[0]+(b[0]-a[0])*f)+','+
                  Math.round(a[1]+(b[1]-a[1])*f)+','+
                  Math.round(a[2]+(b[2]-a[2])*f)+')';
}}
function _gradient_cold(t){{
    // white → light blue → deep navy
    var r=Math.round(255-t*215), g=Math.round(255-t*175), b=255;
    return 'rgb('+r+','+g+','+b+')';
}}

function _feature_colorfunc(atom){{
    var score=featureScores[atom.resi];
    if(score===undefined) return '#cccccc';
    var t=Math.max(0,Math.min(1,score));
    if(featureGradient==='hot')     return _gradient_hot(t);
    if(featureGradient==='fire')    return _gradient_fire(t);
    if(featureGradient==='plasma')  return _gradient_plasma(t);
    if(featureGradient==='viridis') return _gradient_viridis(t);
    if(featureGradient==='cold')    return _gradient_cold(t);
    // classic: white → featureColor
    var c=_hexToRgb(featureColor);
    return 'rgb('+Math.round(255+(c[0]-255)*t)+','+
                  Math.round(255+(c[1]-255)*t)+','+
                  Math.round(255+(c[2]-255)*t)+')';
}}
function setFeatureData(name, scores, hexColor){{
    featureName=name; featureScores=scores; featureColor=hexColor||'#f3722c';
    colorMode='feature';
    applyStyle();
    updateColorBar();
}}
function setFeatureGradient(g){{
    featureGradient=g||'hot';
    if(colorMode==='feature'){{ applyStyle(); updateColorBar(); }}
}}

// ── Pre-computed per-residue hex color map (e.g. SASA with matplotlib cmap) ─
var _resiColorMap = {{}};   // {{resi: '#rrggbb'}}
var _resiColorMapName = '';

function setResidueColorMap(colorMap, name){{
    _resiColorMap = colorMap || {{}};
    _resiColorMapName = name || '';
    colorMode = 'resi_colormap';
    applyStyle();
    updateColorBar();
}}

function _resiColorMap_func(atom){{
    var c = _resiColorMap[atom.resi];
    return c || '#aaaaaa';
}}

// ── loadPDB: swap in a new structure without reloading the page ────────────
function _refreshAllChains(){{
    allChains = [];
    var _m = viewer.getModel(0);
    if(!_m) return;
    var _a = _m.selectedAtoms({{}});
    var _cs = {{}};
    _a.forEach(function(a){{ if(a.chain) _cs[a.chain]=true; }});
    allChains = Object.keys(_cs);
}}

function loadPDB(data){{
    pdbData = data || null;   // always store first — init() picks this up if viewer not ready yet
    hiddenChains = {{}};       // reset chain visibility on every new structure
    allChains = [];
    if(!viewer) return;        // CDN still loading; init() will load pdbData when ready
    viewer.clear();
    if(pdbData){{
        viewer.addModel(pdbData, "pdb");
        _refreshAllChains();
        viewer.zoomTo();
        applyStyle();
        _installClickHandler();
        // re-apply any active overlays on new structure
        if(hbondVisible)    toggleHBonds(true);
        if(contactsVisible) toggleContacts(true);
    }} else {{
        viewer.render();
        updateColorBar();
    }}
}}

// ── Per-residue multi-score data (populated from Python) ──────────────────
var allResScores = {{}};  // {{resi: {{disorder:f, signal:f, tm:f, ...}}}}

function setAllResidueScores(data){{
    allResScores = data || {{}};
}}

// ── Residue labels (top-N AI-scoring residues) ────────────────────────────
var _resiLabelObjs = [];

function clearResidueLabels(){{
    _resiLabelObjs.forEach(function(lb){{ try{{ viewer.removeLabel(lb); }}catch(e){{}} }});
    _resiLabelObjs = [];
    if(viewer) viewer.render();
}}

function showResidueLabels(n){{
    clearResidueLabels();
    if(!n || n<=0 || !featureScores || !viewer) return;
    var entries=[];
    Object.keys(featureScores).forEach(function(r){{
        entries.push({{resi:parseInt(r), score:featureScores[r]}});
    }});
    if(!entries.length) return;
    entries.sort(function(a,b){{ return b.score-a.score; }});
    var topN=entries.slice(0,n);
    var col=featureColor||'#fffacd';
    topN.forEach(function(e){{
        var atoms=viewer.selectedAtoms({{resi:e.resi,atom:'CA'}});
        if(!atoms||!atoms.length) return;
        var a=atoms[0];
        var lb=viewer.addLabel(
            (a.resn||'?')+e.resi+'\\n'+e.score.toFixed(2),
            {{position:{{x:a.x,y:a.y,z:a.z}},
              backgroundColor:'rgba(10,12,30,0.85)',
              fontColor:col, fontSize:11, fontFamily:'system-ui',
              borderThickness:0.5, borderColor:'rgba(255,255,255,0.2)',
              padding:3, inFront:true}}
        );
        _resiLabelObjs.push(lb);
    }});
    viewer.render();
}}

// ── Residue highlight (gold sphere overlay, PyMOL-style) ──────────────────
var _hlResi  = null;
var _hlChain = null;

function highlightResidue(resi, chain){{
    _hlResi  = resi;
    _hlChain = chain;
    _applyHighlight();
}}

function clearHighlight(){{
    _hlResi = null; _hlChain = null;
    applyStyle();
}}

function _applyHighlight(){{
    if(!viewer || !_hlResi) return;
    var sel = {{resi: _hlResi}};
    if(_hlChain) sel.chain = _hlChain;
    viewer.setStyle(sel, {{sphere:{{color:'#FFD700', radius:0.85, opacity:0.92}}}});
    viewer.render();
}}

// ── Residue popup ─────────────────────────────────────────────────────────
// [key, label, color, decimals, bar_scale]
// bar_scale: value is divided by this to get the 0-1 bar fraction
// For AI scores (0-1): bar_scale=1. For pLDDT (stored 0-1): bar_scale=1.
// For B-factor (raw Å², typically 5-100): bar_scale=100.
var _SCORE_LABELS = [
    ['disorder',  'Disorder',          '#818cf8', 3,   1],
    ['signal',    'Signal Peptide',    '#f472b6', 3,   1],
    ['tm',        'TM Helix',          '#34d399', 3,   1],
    ['intramem',  'Intramembrane',     '#6ee7b7', 3,   1],
    ['cc',        'Coiled-Coil',       '#fb923c', 3,   1],
    ['dna',       'DNA-Binding',       '#60a5fa', 3,   1],
    ['act',       'Active Site',       '#f87171', 3,   1],
    ['bnd',       'Binding Site',      '#a78bfa', 3,   1],
    ['phos',      'Phosphorylation',   '#fbbf24', 3,   1],
    ['lcd',       'Low Complexity',    '#94a3b8', 3,   1],
    ['znf',       'Zinc Finger',       '#4ade80', 3,   1],
    ['glyc',      'Glycosylation',     '#f9a8d4', 3,   1],
    ['ubiq',      'Ubiquitination',    '#fb7185', 3,   1],
    ['meth',      'Methylation',       '#a3e635', 3,   1],
    ['acet',      'Acetylation',       '#38bdf8', 3,   1],
    ['lipid',     'Lipidation',        '#e879f9', 3,   1],
    ['plddt',     'pLDDT (0\u20131)',  '#fde68a', 3,   1],
    ['bfactor',   'B-factor (\u00c5\u00b2)', '#d4a85a', 1, 100],
];

function showResiduePopup(resi, resn, chain){{
    var popup = document.getElementById('residue-popup');
    var title = document.getElementById('popup-title');
    var rows  = document.getElementById('popup-rows');
    title.textContent = resn + ' ' + resi + '  (Chain ' + chain + ')';
    var scores = allResScores[resi] || {{}};
    var html = '';
    var hasAny = false;
    _SCORE_LABELS.forEach(function(t){{
        var key=t[0], label=t[1], color=t[2], dec=t[3]||3, bscale=t[4]||1;
        var v = scores[key];
        if(v === undefined || v === null) return;
        hasAny = true;
        var pct = Math.min(100, Math.round(v / bscale * 100));
        html += '<div class="popup-row">'
              + '<span class="popup-label">' + label + '</span>'
              + '<span class="popup-val">' + v.toFixed(dec) + '</span>'
              + '</div>'
              + '<div class="popup-bar-wrap"><div class="popup-bar" style="width:'+pct+'%;background:'+color+'"></div></div>';
    }});
    if(!hasAny){{
        html = '<div style="color:#64748b;font-size:10px">No prediction scores available.<br>Run analysis first.</div>';
    }}
    rows.innerHTML = html;
    popup.style.display = 'block';
}}

function closePopup(){{
    document.getElementById('residue-popup').style.display='none';
    clearHighlight();
}}

// ── Text-based residue selection (cyan sphere overlay) ────────────────────
var _selResidues = {{}};  // {{resi: chain|null}} — populated by applySelection()

function clearSelection(){{
    _selResidues = {{}};
    applyStyle();
}}

function applySelection(spec){{
    if(!viewer || !pdbData) return 0;
    _selResidues = _parseSelection(spec);
    applyStyle();
    return Object.keys(_selResidues).length;
}}

function _parseSelection(spec){{
    spec = (spec || '').trim();
    if(!spec) return {{}};
    var result = {{}};
    var parts = spec.split(/[,;]/);
    parts.forEach(function(part){{
        part = part.trim();
        if(!part) return;
        // chain prefix: "A:45" or "A:10-50"
        var chainMatch = part.match(/^([A-Za-z]):(.+)$/);
        var chain = chainMatch ? chainMatch[1].toUpperCase() : null;
        var core  = chainMatch ? chainMatch[2].trim() : part;
        // range: 10-50
        var rangeMatch = core.match(/^([0-9]+)-([0-9]+)$/);
        if(rangeMatch){{
            var lo=parseInt(rangeMatch[1]), hi=parseInt(rangeMatch[2]);
            for(var r=lo; r<=hi; r++) result[r] = chain;
            return;
        }}
        // single number
        var numMatch = core.match(/^([0-9]+)$/);
        if(numMatch){{ result[parseInt(numMatch[1])] = chain; return; }}
        // residue name (3-letter or 1-letter): LEU, A, etc — scan atoms
        var nameMatch = core.match(/^([A-Za-z]{{1,3}})$/);
        if(nameMatch && viewer && viewer.getModel()){{
            var resn = nameMatch[1].toUpperCase();
            viewer.getModel().atoms.forEach(function(atom){{
                if(atom.resn === resn || (atom.resn && atom.resn.trim() === resn)){{
                    result[atom.resi] = chain || atom.chain || null;
                }}
            }});
        }}
    }});
    return result;
}}

function _applySelectionOverlay(){{
    if(!viewer) return;
    var keys = Object.keys(_selResidues);
    if(!keys.length) return;
    keys.forEach(function(resi){{
        var ch = _selResidues[resi];
        var sel = {{resi: parseInt(resi)}};
        if(ch) sel.chain = ch;
        viewer.addStyle(sel, {{sphere:{{color:'#00e5ff', radius:0.82, opacity:0.78}}}});
    }});
    viewer.render();
}}

// ── Patch applyStyle to re-apply highlight + selection after style change ──
var _origApplyStyle = applyStyle;
applyStyle = function(){{
    _origApplyStyle();
    if(_hlResi) _applyHighlight();
    _applySelectionOverlay();
}};

// ── Geometry measurement (distance / angle / dihedral) ────────────────────
var _distMode   = false;          // measure mode on/off
var _measMode   = 'distance';     // 'distance' | 'angle' | 'dihedral'
var _measAtoms  = [];             // collected atom picks
var _distShapes = [];
var _distLabels = [];
var _MEAS_NEEDS = {{distance:2, angle:3, dihedral:4}};

function enterDistanceMode(){{ _distMode=true; _measAtoms=[]; }}
function exitDistanceMode(){{
    _distMode=false; _measAtoms=[];
    applyStyle();
}}
function setMeasureMode(mode){{
    _measMode = mode;
    _measAtoms = [];
    if(_distMode) applyStyle();   // clear partial picks
}}
function clearDistances(){{
    _distShapes.forEach(function(s){{ try{{viewer.removeShape(s);}}catch(e){{}} }});
    _distLabels.forEach(function(l){{ try{{viewer.removeLabel(l);}}catch(e){{}} }});
    _distShapes=[]; _distLabels=[];
    if(viewer) viewer.render();
}}

// ── vector math helpers ───────────────────────────────────────────────────
function _vsub(a,b){{ return {{x:a.x-b.x,y:a.y-b.y,z:a.z-b.z}}; }}
function _dot(a,b){{ return a.x*b.x+a.y*b.y+a.z*b.z; }}
function _cross(a,b){{ return {{x:a.y*b.z-a.z*b.y, y:a.z*b.x-a.x*b.z, z:a.x*b.y-a.y*b.x}}; }}
function _norm(v){{ return Math.sqrt(_dot(v,v)); }}
function _mid2(a,b){{ return {{x:(a.x+b.x)/2,y:(a.y+b.y)/2,z:(a.z+b.z)/2}}; }}
function _mid4(ps){{ return {{x:(ps[0].x+ps[1].x+ps[2].x+ps[3].x)/4,
                              y:(ps[0].y+ps[1].y+ps[2].y+ps[3].y)/4,
                              z:(ps[0].z+ps[1].z+ps[2].z+ps[3].z)/4}}; }}

function _calcAngle(p1,p2,p3){{
    var v1=_vsub(p1,p2), v2=_vsub(p3,p2);
    return Math.acos(Math.max(-1,Math.min(1,_dot(v1,v2)/(_norm(v1)*_norm(v2)))))*180/Math.PI;
}}

function _calcDihedral(p1,p2,p3,p4){{
    var b1=_vsub(p2,p1), b2=_vsub(p3,p2), b3=_vsub(p4,p3);
    var n1=_cross(b1,b2), n2=_cross(b2,b3);
    var n1n=_norm(n1), n2n=_norm(n2), b2n=_norm(b2);
    if(n1n<1e-10 || n2n<1e-10 || b2n<1e-10) return NaN;
    var n1u={{x:n1.x/n1n,y:n1.y/n1n,z:n1.z/n1n}};
    var n2u={{x:n2.x/n2n,y:n2.y/n2n,z:n2.z/n2n}};
    var b2u={{x:b2.x/b2n,y:b2.y/b2n,z:b2.z/b2n}};
    var m1=_cross(n1u,b2u);
    return Math.atan2(_dot(m1,n2u),_dot(n1u,n2u))*180/Math.PI;
}}

function _addMeasLine(p1,p2,col){{
    var sh=viewer.addCylinder({{
        start:{{x:p1.x,y:p1.y,z:p1.z}},end:{{x:p2.x,y:p2.y,z:p2.z}},
        radius:0.06,dashed:true,color:col,fromCap:1,toCap:1
    }});
    _distShapes.push(sh);
}}

function _addMeasLabel(text,pos){{
    var lb=viewer.addLabel(text,{{
        position:pos,backgroundColor:'rgba(10,12,30,0.88)',
        fontColor:'#fffacd',fontSize:13,fontFamily:'system-ui',
        borderThickness:0.5,borderColor:'rgba(255,255,255,0.28)',
        padding:3,inFront:true
    }});
    _distLabels.push(lb);
}}

function _finalizeMeasurement(pts){{
    var ps=pts.map(function(p){{return {{x:p.x,y:p.y,z:p.z}};}}); // positions only
    if(_measMode==='distance'){{
        var dx=ps[1].x-ps[0].x,dy=ps[1].y-ps[0].y,dz=ps[1].z-ps[0].z;
        var d=Math.sqrt(dx*dx+dy*dy+dz*dz);
        _addMeasLine(ps[0],ps[1],'#ffff44');
        _addMeasLabel(d.toFixed(2)+' \u00c5', _mid2(ps[0],ps[1]));
    }} else if(_measMode==='angle'){{
        _addMeasLine(ps[0],ps[1],'#ffaa00');
        _addMeasLine(ps[1],ps[2],'#ffaa00');
        var ang=_calcAngle(ps[0],ps[1],ps[2]);
        _addMeasLabel(ang.toFixed(1)+'\u00b0', ps[1]);
    }} else {{
        _addMeasLine(ps[0],ps[1],'#88aaff');
        _addMeasLine(ps[1],ps[2],'#88aaff');
        _addMeasLine(ps[2],ps[3],'#88aaff');
        var dih=_calcDihedral(ps[0],ps[1],ps[2],ps[3]);
        _addMeasLabel(isNaN(dih)?'collinear':dih.toFixed(1)+'\u00b0', _mid4(ps));
    }}
    viewer.render();
}}

function _handleDistClick(atom){{
    // Use exact clicked atom coordinates (not Cα)
    var pick={{resi:atom.resi,chain:atom.chain,resn:atom.resn||'',
               atm:atom.atom||atom.elem||'',x:atom.x,y:atom.y,z:atom.z}};
    _measAtoms.push(pick);
    // small numbered pick label — no intrusive sphere overlay
    var n=_measAtoms.length;
    var lb=viewer.addLabel(''+n,{{
        position:{{x:atom.x,y:atom.y,z:atom.z}},
        backgroundColor:'rgba(255,140,0,0.82)',fontColor:'#fff',
        fontSize:11,padding:2,inFront:true
    }});
    _distLabels.push(lb);
    viewer.render();
    var need=_MEAS_NEEDS[_measMode]||2;
    if(_measAtoms.length>=need){{
        _finalizeMeasurement(_measAtoms.slice(0,need));
        _measAtoms=[];
        // pick-number labels stay with measurement; cleared only by clearDistances()
    }}
}}

// ── Install click handler once viewer is ready ────────────────────────────
function _installClickHandler(){{
    if(!viewer) return;
    viewer.setClickable({{}}, true, function(atom, _v, _event, _container){{
        if(_distMode){{
            _handleDistClick(atom);
        }} else {{
            highlightResidue(atom.resi, atom.chain);
            showResiduePopup(atom.resi, atom.resn || atom.elem, atom.chain);
            if(_bridge) _bridge.residueClicked(atom.resi);
        }}
    }});
}}

function init(){{
    viewer=$3Dmol.createViewer("vp",{{backgroundColor:"#ffffff",antialias:true}});
    if(pdbData){{          // null on the initial empty-page load
        viewer.addModel(pdbData,"pdb");
        _refreshAllChains();
        viewer.zoomTo();   // set camera BEFORE rendering
        applyStyle();      // renders with correct camera
    }}
    viewer.render();
    updateColorBar();
    _installClickHandler();
}}
window.addEventListener("load",init);
</script>
</body></html>"""

    def _js(self, code: str) -> None:
        """Run JavaScript in the structure viewer (no-op if unavailable)."""
        if self.structure_viewer is not None:
            self.structure_viewer.page().runJavaScript(code)

    # Feature name → analysis_data key mapping (keys must match core.py return dict)
    _FEATURE_SCORE_KEYS = {
        "Disorder":        "disorder_scores",
        "Signal Peptide":  "sp_bilstm_profile",
        "Transmembrane":   "tm_bilstm_profile",
        "Intramembrane":   "intramem_bilstm_profile",
        "Coiled-Coil":     "cc_bilstm_profile",
        "DNA-Binding":     "dna_bilstm_profile",
        "Active Site":     "act_bilstm_profile",
        "Binding Site":    "bnd_bilstm_profile",
        "Phosphorylation": "phos_bilstm_profile",
        "Low Complexity":  "lcd_bilstm_profile",
        "Zinc Finger":     "znf_bilstm_profile",
        "Glycosylation":   "glyc_bilstm_profile",
        "Ubiquitination":  "ubiq_bilstm_profile",
        "Methylation":     "meth_bilstm_profile",
        "Acetylation":     "acet_bilstm_profile",
        "Lipidation":      "lipid_bilstm_profile",
        "Disulfide Bond":  "disulf_bilstm_profile",
        "Functional Motif":"motif_bilstm_profile",
        "Propeptide":      "prop_bilstm_profile",
        "Repeat Region":        "rep_bilstm_profile",
        "RNA-Binding":          "rnabind_bilstm_profile",
        "Nucleotide-Binding":   "nucbind_bilstm_profile",
        "Transit Peptide":      "transit_bilstm_profile",
    }

    def _available_feature_schemes(self) -> list[str]:
        """Return all AI feature names (all 24 heads). Uncomputed ones show a popup on select."""
        return list(self._FEATURE_SCORE_KEYS.keys())

    def _update_scheme_combo(self, mode: str) -> None:
        if mode == "AI Features":
            schemes = self._available_feature_schemes()
        else:
            schemes = self._STRUCT_SCHEMES.get(mode, ["Default"])
        self.struct_scheme_combo.blockSignals(True)
        self.struct_scheme_combo.clear()
        self.struct_scheme_combo.addItems(schemes)
        self.struct_scheme_combo.blockSignals(False)

    # Maps human-readable gradient combo text → JS featureGradient key
    _AI_GRADIENT_MAP = {
        "Hot (White→Red)":       "hot",
        "Fire (Black→Yellow)":   "fire",
        "Plasma":                "plasma",
        "Viridis":               "viridis",
        "Cold (White→Blue)":     "cold",
        "Classic (White→Color)": "classic",
    }

    def _push_feature_scores(self, feature_label: str,
                             gradient: str = "Hot (White→Red)") -> None:
        """Send per-residue scores for feature_label to the 3Dmol JS layer."""
        from beer.graphs._style import FEATURE_COLORS
        import json as _json
        ad = self.analysis_data or {}
        key = self._FEATURE_SCORE_KEYS.get(feature_label, "disorder_scores")
        scores = ad.get(key) or []
        if not scores:
            QMessageBox.information(
                self, "AI Predictions Not Yet Computed",
                f"AI Predictions have not been run yet.\n\n"
                "Click the \u2018AI Analysis\u2019 button (next to Analyze) to compute "
                "all 23 per-residue prediction heads, then select this color scheme again.")
            return
        feat_key = feature_label.lower().replace(" ", "_")
        color = FEATURE_COLORS.get(feat_key, "#f3722c")
        scores_dict = {i + 1: float(v) for i, v in enumerate(scores)}
        grad_key = self._AI_GRADIENT_MAP.get(gradient, "hot")
        self._js(
            f"setFeatureData({_json.dumps(feature_label)},"
            f"{_json.dumps(scores_dict)},"
            f"{_json.dumps(color)});"
        )
        self._js(f"setFeatureGradient({_json.dumps(grad_key)});")
        self._refresh_reslabels_if_active()

    def _push_zyggregator_scores(self, scheme: str) -> None:
        """Color structure by ZYGGREGATOR β-aggregation propensity."""
        import json as _json
        import matplotlib.cm as _cm
        import matplotlib.colors as _mc
        ad = self.analysis_data or {}
        scores = ad.get("aggr_profile") or []
        if not scores:
            QMessageBox.information(
                self, "ZYGGREGATOR Not Yet Computed",
                "β-Aggregation profile is not available yet.\n\n"
                "Open the 'β-Aggregation & Solubility' section in the Analysis tab first.")
            return
        mx = max(scores) if scores else 1.0
        norm_scores = [min(float(v) / max(mx, 0.001), 1.0) for v in scores]

        _AGGR_CMAP = {
            "Fire":    "afmhot",
            "Inferno": "inferno",
            "Viridis": "viridis",
        }
        if scheme in _AGGR_CMAP:
            cmap = _cm.get_cmap(_AGGR_CMAP[scheme])
            resi_colors = {i + 1: _mc.to_hex(cmap(v)) for i, v in enumerate(norm_scores)}
            self._js(
                f"setResidueColorMap({_json.dumps(resi_colors)},"
                f"'ZYGGREGATOR \u03b2-Aggregation');"
            )
            return
        if scheme == "Hotspots Only":
            scores_dict = {i + 1: (1.0 if float(v) >= 1.0 else 0.0)
                           for i, v in enumerate(scores)}
        else:
            scores_dict = {i + 1: v for i, v in enumerate(norm_scores)}
        self._js(
            f"setFeatureData('ZYGGREGATOR \u03b2-Aggregation',"
            f"{_json.dumps(scores_dict)},'#e07a5f');"
        )
        self._js("setFeatureGradient('hot');")

    def _push_sasa_scores(self, scheme: str) -> None:
        """Color structure by per-residue relative solvent-accessible surface area.

        Each scheme maps RSA (0→1) through a distinct matplotlib colormap so that
        buried vs. exposed residues are immediately visually discernible.
        The JS setFeatureData mechanism uses a single accent color for the gradient
        endpoint; instead we pre-compute per-residue hex colors and push a custom
        colorfunc via setCustomColorMap.
        """
        import json as _json
        import matplotlib.cm as _cm
        import matplotlib.colors as _mc
        sasa = getattr(self, "_struct_sasa_data", {})
        if not sasa:
            QMessageBox.information(
                self, "SASA Not Available",
                "Solvent accessibility could not be computed.\n\n"
                "A 3D structure must be loaded first.")
            return

        _SCHEME_CMAP = {
            "Buried→Exposed (Blue→Red)":    ("RdBu_r",   False),
            "Exposed→Buried (Red→Blue)":    ("RdBu",     False),
            "Viridis (Buried→Exposed)":     ("viridis",  False),
            "Plasma (Buried→Exposed)":      ("plasma",   False),
            "Magma (Buried→Exposed)":       ("magma",    False),
            "Cyan→Orange":                  ("PuOr_r",   False),
        }
        cmap_name, _ = _SCHEME_CMAP.get(scheme, ("RdBu_r", False))
        cmap = _cm.get_cmap(cmap_name)

        # Pre-compute a hex color per residue and push as per-residue color dict
        resi_colors: dict[int, str] = {}
        for resi, rsa in sasa.items():
            rgba = cmap(float(rsa))
            resi_colors[resi] = _mc.to_hex(rgba)

        # Push via dedicated JS function that accepts {resi: hexColor} directly
        self._js(
            f"setResidueColorMap({_json.dumps(resi_colors)}, "
            f"'Solvent Accessibility (RSA)');"
        )

    def _populate_sasa_report_section(self) -> None:
        """Generate and populate the SASA Profile report section from current SASA data."""
        if "SASA Profile" not in getattr(self, "report_section_tabs", {}):
            return
        rsa = getattr(self, "_struct_sasa_data", {})
        asa = getattr(self, "_struct_sasa_raw", {})
        if not rsa:
            return
        rsa_vals = [rsa[k] for k in sorted(rsa)]
        asa_vals = [asa[k] for k in sorted(asa)]
        n = len(rsa_vals)
        mean_rsa = sum(rsa_vals) / n
        n_buried  = sum(1 for v in rsa_vals if v < 0.20)
        n_partial = sum(1 for v in rsa_vals if 0.20 <= v < 0.50)
        n_exposed = sum(1 for v in rsa_vals if v >= 0.50)
        mean_asa = sum(asa_vals) / len(asa_vals) if asa_vals else 0.0
        from beer.reports.css import get_report_css
        css = get_report_css(getattr(self, "_is_dark", False))
        html = (
            f"<html><head><style>{css}</style></head><body>"
            f"<h2>Solvent Accessibility (SASA)</h2>"
            f"<p>Per-residue solvent-accessible surface area computed from the loaded 3D structure "
            f"using the Shrake–Rupley algorithm. RSA (relative solvent accessibility) is normalised "
            f"by Miller et al. maximum ASA values.</p>"
            f"<h3>Summary ({n} residues)</h3>"
            f"<table><tr><th>Category</th><th>Threshold</th><th>Count</th><th>%</th></tr>"
            f"<tr><td>Buried</td><td>RSA &lt; 0.20</td><td>{n_buried}</td>"
            f"<td>{100*n_buried/n:.1f}%</td></tr>"
            f"<tr><td>Partially exposed</td><td>0.20 ≤ RSA &lt; 0.50</td><td>{n_partial}</td>"
            f"<td>{100*n_partial/n:.1f}%</td></tr>"
            f"<tr><td>Exposed</td><td>RSA ≥ 0.50</td><td>{n_exposed}</td>"
            f"<td>{100*n_exposed/n:.1f}%</td></tr>"
            f"</table>"
            f"<p>Mean RSA: <b>{mean_rsa:.3f}</b> &nbsp;|&nbsp; "
            f"Mean ASA: <b>{mean_asa:.1f} Å²</b></p>"
            f"<p><em>The toggle in the graph tab switches between RSA (dimensionless, 0–1) "
            f"and raw ASA (Å²).</em></p>"
            f"</body></html>"
        )
        # Append sparkline of RSA values
        import urllib.parse as _up
        spar_uri = self._make_sparkline_png(rsa_vals, "#4cc9f0", threshold=None)
        href = "beer://graph/" + _up.quote("SASA Profile")
        html += (
            f"<div style='margin:10px 0 6px;'>"
            f"<img src='{spar_uri}' style='width:100%;height:72px;"
            f"border-radius:6px;display:block;'/>"
            f"<div style='text-align:right;margin-top:3px;'>"
            f"<a href='{href}' style='font-family:sans-serif;font-size:10px;"
            f"color:#4361ee;text-decoration:none;'>\u2192 Full Graph</a>"
            f"</div></div>"
        )
        self.report_section_tabs["SASA Profile"].setHtml(html)
        if self.analysis_data is not None:
            self.analysis_data.setdefault("report_sections", {})["SASA Profile"] = html

    def _rebuild_sasa_graph(self) -> None:
        """Regenerate the SASA Profile graph in-place after RSA/ASA toggle."""
        if "SASA Profile" not in self.graph_tabs:
            return
        rsa = getattr(self, "_struct_sasa_data", {})
        asa = getattr(self, "_struct_sasa_raw", {})
        if not rsa:
            return
        lf = getattr(self, "_label_font", 14)
        tf = getattr(self, "_tick_font", 12)
        fig = create_sasa_figure(
            rsa, asa,
            window=self.default_window_size,
            show_asa=getattr(self, "_sasa_show_asa", False),
            label_font=lf, tick_font=tf,
        )
        self._replace_graph("SASA Profile", fig)

    def _on_struct_rep_changed(self, rep_label: str) -> None:
        self._js(f"setRepresentation('{rep_label.lower()}');")

    def _on_struct_color_mode_changed(self, mode: str) -> None:
        self._update_scheme_combo(mode)
        is_ai = (mode == "AI Features")
        if hasattr(self, "struct_ai_gradient_lbl"):
            self.struct_ai_gradient_lbl.setVisible(is_ai)
            self.struct_ai_gradient_combo.setVisible(is_ai)
        key = self._STRUCT_MODE_KEY.get(mode, "plddt")
        scheme = self.struct_scheme_combo.currentText()
        if mode == "AI Features":
            grad = getattr(self, "struct_ai_gradient_combo", None)
            self._push_feature_scores(
                scheme, gradient=grad.currentText() if grad else "Hot (White→Red)")
        elif mode == "Aggregation (ZYGGREGATOR)":
            self._push_zyggregator_scores(scheme)
        elif mode == "Solvent Accessibility":
            self._push_sasa_scores(scheme)
        elif mode == "Spectrum (N→C)":
            seq_len = len((self.analysis_data or {}).get("seq", "")) or 9999
            self._js(f"setColorMode('{key}','{scheme}',{seq_len});")
        else:
            self._js(f"setColorMode('{key}','{scheme}');")

    def _on_struct_scheme_changed(self, scheme: str) -> None:
        if not scheme:
            return
        mode = self.struct_color_mode_combo.currentText()
        if mode == "AI Features":
            grad = getattr(self, "struct_ai_gradient_combo", None)
            self._push_feature_scores(
                scheme, gradient=grad.currentText() if grad else "Hot (White→Red)")
        elif mode == "Aggregation (ZYGGREGATOR)":
            self._push_zyggregator_scores(scheme)
        elif mode == "Solvent Accessibility":
            self._push_sasa_scores(scheme)
        elif mode == "Spectrum (N→C)":
            seq_len = len((self.analysis_data or {}).get("seq", "")) or 9999
            self._js(f"setScheme('{scheme}',{seq_len});")
        else:
            self._js(f"setScheme('{scheme}');")

    def _on_struct_ai_gradient_changed(self, text: str) -> None:
        grad_key = self._AI_GRADIENT_MAP.get(text, "hot")
        self._js(f"setFeatureGradient({repr(grad_key)});")

    @staticmethod
    def _parse_ss_composition(pdb_str: str) -> dict[str, int]:
        """Count residues assigned to helix / sheet / coil from HELIX+SHEET records."""
        helix_res: set[tuple[str, int]] = set()
        sheet_res: set[tuple[str, int]] = set()
        for line in pdb_str.splitlines():
            if line.startswith("HELIX "):
                try:
                    chain = line[19:20].strip()
                    start = int(line[21:25])
                    end   = int(line[33:37])
                    for r in range(start, end + 1):
                        helix_res.add((chain, r))
                except (ValueError, IndexError):
                    pass
            elif line.startswith("SHEET "):
                try:
                    chain = line[21:22].strip()
                    start = int(line[22:26])
                    end   = int(line[33:37])
                    for r in range(start, end + 1):
                        sheet_res.add((chain, r))
                except (ValueError, IndexError):
                    pass
        total = helix_count = sheet_count = 0
        seen: set[tuple[str, int]] = set()
        for line in pdb_str.splitlines():
            if not line.startswith("ATOM  "):
                continue
            try:
                chain = line[21:22].strip()
                resi  = int(line[22:26])
                key   = (chain, resi)
                if key in seen:
                    continue
                seen.add(key)
                total += 1
                if key in helix_res:
                    helix_count += 1
                elif key in sheet_res:
                    sheet_count += 1
            except (ValueError, IndexError):
                pass
        return {"helix": helix_count, "sheet": sheet_count,
                "coil": max(0, total - helix_count - sheet_count), "total": total}

    def _on_struct_colorbar_toggled(self, checked: bool) -> None:
        self._js(f"setColorBarVisible({'true' if checked else 'false'});")

    def _on_struct_spin_toggled(self, checked: bool) -> None:
        self.struct_spin_btn.setText("Spin: On" if checked else "Spin: Off")
        axis = self.struct_spin_axis_combo.currentText()[0].lower()  # 'y', 'x', or 'z'
        self._js(f"setSpin({'true' if checked else 'false'},'{axis}');")

    def _on_struct_spin_axis_changed(self) -> None:
        if self.struct_spin_btn.isChecked():
            axis = self.struct_spin_axis_combo.currentText()[0].lower()
            self._js(f"setSpin(true,'{axis}');")

    def _on_struct_selection_apply(self) -> None:
        spec = self.struct_sel_edit.text().strip()
        if not spec:
            return
        import json as _json
        self.structure_viewer.page().runJavaScript(
            f"applySelection({_json.dumps(spec)})",
            lambda count: self._sel_count_lbl.setText(
                f"{count} residue(s) selected" if isinstance(count, int) and count > 0
                else ("No match" if count == 0 else "")
            ) if self.structure_viewer else None
        )

    def _on_struct_selection_clear(self) -> None:
        self.struct_sel_edit.clear()
        self._sel_count_lbl.setText("")
        self._js("clearSelection();")

    def _on_struct_measure_toggled(self, checked: bool) -> None:
        self.struct_dist_btn.setText("Pick Atoms: On" if checked else "Pick Atoms: Off")
        self._js("enterDistanceMode();" if checked else "exitDistanceMode();")

    def _on_struct_measure_mode_changed(self, index: int) -> None:
        _hints = ["Pick 2 residues", "Pick 3 residues", "Pick 4 residues"]
        _modes = ["distance", "angle", "dihedral"]
        self._meas_hint_lbl.setText(_hints[index])
        self._js(f"setMeasureMode('{_modes[index]}');")

    # ── Graph ↔ Structure bidirectional link ──────────────────────────────────

    _STRUCT_LINK_GRAPHS = BILSTM_PROFILE_TABS | {
        "Hydrophobicity Profile", "Local Charge Profile",
        "SCD Profile", "SHD Profile", "RNA-Binding Profile",
        "β-Aggregation Profile", "Solubility Profile",
        "pLDDT Profile", "SASA Profile", "Hydrophobic Moment",
    }

    def _on_struct_residue_picked(self, resi: int) -> None:
        """Called by JS bridge when an atom is clicked in the structure viewer."""
        self._mark_graph_residue(resi)
        if hasattr(self, "_marker_pos_lbl"):
            self._marker_pos_lbl.setText(f"Residue {resi} marked")
        if hasattr(self, "_graphs_clear_marker_btn"):
            self._graphs_clear_marker_btn.setEnabled(True)

    def _on_clear_struct_marker(self) -> None:
        self._clear_struct_graph_marker()
        self._js("clearHighlight();")   # also remove gold sphere from 3D viewer
        if hasattr(self, "_marker_pos_lbl"):
            self._marker_pos_lbl.setText("No marker set")
        if hasattr(self, "_graphs_clear_marker_btn"):
            self._graphs_clear_marker_btn.setEnabled(False)

    def _mark_graph_residue(self, resi: int) -> None:
        """Draw/update a dashed vertical position line on all profile graph canvases.
        Also remembers resi so newly-created canvases get the marker immediately.
        """
        self._struct_marker_resi = resi
        self._apply_struct_marker_to_canvas(resi)

    def _apply_struct_marker_to_canvas(self, resi: int, canvas=None) -> None:
        """Apply or update the position marker.  If canvas is given, update only that one;
        otherwise iterate all profile graph canvases.
        """
        def _mark(c):
            if c is None or not c.figure.axes:
                return
            ax = c.figure.axes[0]
            old = getattr(ax, "_beer_struct_vline", None)
            if old is not None:
                try:
                    old.remove()
                except Exception:
                    pass
            ax._beer_struct_vline = ax.axvline(
                resi, color="#ff6b6b", linewidth=1.3,
                linestyle="--", alpha=0.82, zorder=5,
            )
            c.draw_idle()

        if canvas is not None:
            _mark(canvas)
            return
        for title, (_, vb) in self.graph_tabs.items():
            if title not in self._STRUCT_LINK_GRAPHS:
                continue
            _mark(self._find_canvas(vb))

    def _clear_struct_graph_marker(self) -> None:
        """Remove the position marker from all profile graphs and reset state."""
        self._struct_marker_resi = None
        for title, (_, vb) in self.graph_tabs.items():
            if title not in self._STRUCT_LINK_GRAPHS:
                continue
            c = self._find_canvas(vb)
            if c is None or not c.figure.axes:
                continue
            ax = c.figure.axes[0]
            old = getattr(ax, "_beer_struct_vline", None)
            if old is not None:
                try:
                    old.remove()
                except Exception:
                    pass
                ax._beer_struct_vline = None
            c.draw_idle()

    def _wire_graph_struct_hover(self, canvas) -> None:
        """Wire graph interactions to the 3D structure viewer.

        Hover  → shows residue number in the status bar (preview only).
        Click  → persistent gold highlight in 3D (survives tab switch).
                 Also stamps the position marker on all profile graphs.
        """
        def _on_motion(event):
            if event.inaxes and event.xdata is not None:
                r = int(round(event.xdata))
                if r >= 1:
                    self.statusBar.showMessage(f"Residue {r}", 0)

        def _on_leave(_event):
            self.statusBar.clearMessage()

        def _on_click(event):
            if event.inaxes and event.xdata is not None and event.button == 1:
                r = int(round(event.xdata))
                if r >= 1:
                    # Persistent highlight in 3D — stays until next click or Clear
                    if self.structure_viewer is not None:
                        self._js(f"highlightResidue({r}, null);")
                    # Stamp position marker on all profile graphs
                    self._mark_graph_residue(r)
                    if hasattr(self, "_marker_pos_lbl"):
                        self._marker_pos_lbl.setText(f"Residue {r} marked")
                    if hasattr(self, "_graphs_clear_marker_btn"):
                        self._graphs_clear_marker_btn.setEnabled(True)

        canvas.mpl_connect("motion_notify_event", _on_motion)
        canvas.mpl_connect("axes_leave_event", _on_leave)
        canvas.mpl_connect("button_press_event", _on_click)

    def _on_reslbl_toggled(self, checked: bool) -> None:
        self.struct_reslbl_spin.setEnabled(checked)
        if checked:
            self._js(f"showResidueLabels({self.struct_reslbl_spin.value()});")
        else:
            self._js("clearResidueLabels();")

    def _on_reslbl_n_changed(self, n: int) -> None:
        if self.struct_reslbl_cb.isChecked():
            self._js(f"showResidueLabels({n});")

    def _refresh_reslabels_if_active(self) -> None:
        """Call after pushing new feature scores so labels stay in sync."""
        if hasattr(self, "struct_reslbl_cb") and self.struct_reslbl_cb.isChecked():
            self._js(f"showResidueLabels({self.struct_reslbl_spin.value()});")

    def _reset_struct_view(self) -> None:
        """Reset the 3D viewer and all controls to their defaults."""
        for combo, text in [
            (self.struct_rep_combo,        "Cartoon"),
            (self.struct_color_mode_combo, "pLDDT / B-factor"),
        ]:
            combo.blockSignals(True)
            combo.setCurrentText(text)
            combo.blockSignals(False)
        self._update_scheme_combo("pLDDT / B-factor")
        if self.struct_spin_btn.isChecked():
            self.struct_spin_btn.setChecked(False)   # triggers spin-off via toggled signal
        if hasattr(self, "struct_sel_edit"):
            self.struct_sel_edit.clear()
            self._sel_count_lbl.setText("")
        if hasattr(self, "struct_dist_btn") and self.struct_dist_btn.isChecked():
            self.struct_dist_btn.setChecked(False)
        self._js("clearSelection(); clearDistances(); clearResidueLabels(); resetView();")
        if hasattr(self, "struct_reslbl_cb"):
            self.struct_reslbl_cb.setChecked(False)

    def _pick_background_color(self) -> None:
        color = QColorDialog.getColor(parent=self)
        if color.isValid():
            self._js(f"setBackground('{color.name()}');")

    def _take_structure_snapshot(self) -> None:
        if self.structure_viewer is None:
            return
        fn, _ = QFileDialog.getSaveFileName(
            self, "Save Snapshot", "structure.png", "PNG Images (*.png)",
            options=QFileDialog.Option.DontUseNativeDialog)
        if not fn:
            return
        if not fn.lower().endswith(".png"):
            fn += ".png"

        def _receive(uri):
            if not uri:
                QMessageBox.warning(self, "Snapshot", "No structure loaded — nothing to render.")
                return
            try:
                data = base64.b64decode(uri.split(",", 1)[1])
                with open(fn, "wb") as fh:
                    fh.write(data)
                self.statusBar.showMessage(f"Snapshot saved: {fn}", 4000)
            except Exception as exc:
                QMessageBox.critical(self, "Snapshot Failed", str(exc))

        self.structure_viewer.page().runJavaScript(
            "viewer ? viewer.pngURI() : null", 0, _receive)

    def _on_structure_page_loaded(self, ok: bool) -> None:
        """Called once when the base 3Dmol page finishes loading.
        If a PDB was queued before the page was ready, deliver it now."""
        if ok and self._pending_pdb is not None:
            pdb_json = self._pending_pdb
            self._pending_pdb = None
            self.structure_viewer.page().runJavaScript(f"loadPDB({pdb_json});")

    @staticmethod
    def _compute_sasa(pdb_str: str) -> tuple[dict[int, float], dict[int, float]]:
        """Return (rsa_dict, asa_dict) using Shrake-Rupley.

        Both dicts are keyed by actual PDB residue sequence number (matches
        atom.resi in 3Dmol — correct for AlphaFold, gapped PDB chains, and
        multi-chain complexes).  RSA is normalised by Miller et al. max-ASA
        values; ASA is the raw value in Å².  Returns ({}, {}) on any error.
        """
        _MAX_ASA = {
            "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0,
            "CYS": 167.0, "GLN": 225.0, "GLU": 223.0, "GLY":  97.0,
            "HIS": 224.0, "ILE": 197.0, "LEU": 201.0, "LYS": 236.0,
            "MET": 224.0, "PHE": 240.0, "PRO": 159.0, "SER": 155.0,
            "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0,
        }
        try:
            from io import StringIO as _SIO
            from Bio.PDB import PDBParser as _PP
            from Bio.PDB.SASA import ShrakeRupley as _SR
            parser = _PP(QUIET=True)
            structure = parser.get_structure("s", _SIO(pdb_str))
            sr = _SR()
            sr.compute(structure, level="R")
            rsa_dict: dict[int, float] = {}
            asa_dict: dict[int, float] = {}
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if residue.id[0] != " ":   # skip HETATM / water
                            continue
                        resi_num = residue.id[1]
                        resname  = residue.get_resname().strip()
                        raw_asa  = residue.sasa
                        max_asa  = _MAX_ASA.get(resname, 200.0)
                        rsa_dict[resi_num] = round(min(1.0, raw_asa / max_asa), 4)
                        asa_dict[resi_num] = round(raw_asa, 2)
                break   # first model only
            return rsa_dict, asa_dict
        except Exception:
            return {}, {}

    def _load_structure_viewer(self, pdb_str: str) -> None:
        """Swap in a new structure without reloading the 3Dmol page."""
        if not _WEBENGINE_AVAILABLE or self.structure_viewer is None:
            return
        self._struct_pdb_str = pdb_str          # keep for Ramachandran / SS extraction
        pdb_json = json.dumps(pdb_str)
        # Keep as pending so loadFinished can retry if the page is still loading.
        self._pending_pdb = pdb_json
        # 1-arg form is the only safe form in PySide6 (no 2-arg callback variant).
        self._js(f"loadPDB({pdb_json});")
        self._populate_chain_controls(pdb_str)
        # Annotate disorder regions and signal peptide in 3D viewer after a short delay
        from PySide6.QtCore import QTimer as _QT
        _QT.singleShot(800, self._annotate_structure_viewer)
        # Compute SASA deferred — Shrake-Rupley is CPU-bound and must not block
        # the event loop before loadPDB has been dispatched to WebEngine.
        _pdb_copy = pdb_str
        def _deferred_sasa():
            self._struct_sasa_data, self._struct_sasa_raw = self._compute_sasa(_pdb_copy)
            self._populate_sasa_report_section()
            if self.analysis_data:
                self.update_graph_tabs()
        _QT.singleShot(200, _deferred_sasa)

    @staticmethod
    def _parse_pdb_chains(pdb_str: str) -> list[str]:
        """Return sorted list of unique chain IDs found in ATOM/HETATM records."""
        seen: dict = {}
        for line in pdb_str.splitlines():
            if line.startswith(("ATOM  ", "HETATM")):
                chain = line[21:22].strip()
                if chain and chain not in seen:
                    seen[chain] = None
        return sorted(seen.keys())

    def _populate_chain_controls(self, pdb_str: str) -> None:
        """Build per-chain visibility checkboxes from a PDB string."""
        if not hasattr(self, "_chains_grp"):
            return
        chains = self._parse_pdb_chains(pdb_str)
        # Clear existing checkboxes
        self._chain_checkboxes.clear()
        while self._chain_cbs_layout.count():
            item = self._chain_cbs_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        if len(chains) <= 1:
            self._chains_grp.setVisible(False)
            return
        for chain_id in chains:
            cb = QCheckBox(f"Chain {chain_id}")
            cb.setChecked(True)
            cb.toggled.connect(lambda checked, c=chain_id:
                self._js(f"setChainVisible('{c}', {'true' if checked else 'false'});"))
            self._chain_cbs_layout.addWidget(cb)
            self._chain_checkboxes[chain_id] = cb
        self._chains_grp.setVisible(True)

    def _show_all_chains(self) -> None:
        for cb in self._chain_checkboxes.values():
            cb.blockSignals(True)
            cb.setChecked(True)
            cb.blockSignals(False)
        self._js("showAllChains();")

    def _hide_all_chains(self) -> None:
        chain_list = list(self._chain_checkboxes.keys())
        for cb in self._chain_checkboxes.values():
            cb.blockSignals(True)
            cb.setChecked(False)
            cb.blockSignals(False)
        import json as _json
        self._js(f"hideAllChains({_json.dumps(chain_list)});")

    def _annotate_structure_viewer(self) -> None:
        """Overlay disorder and signal-peptide annotations in the 3D viewer via JS."""
        if not _WEBENGINE_AVAILABLE or self.structure_viewer is None:
            return
        if not self.analysis_data:
            return
        disorder = self.analysis_data.get("disorder_scores", [])
        sp = self.analysis_data.get("sp_result", {})
        # Build JS: addLabel calls for disordered stretches
        js_parts = ["if(typeof viewer!=='undefined' && viewer){",
                    "viewer.removeAllLabels();"]
        if disorder:
            threshold = 0.5
            in_region = False
            start = 0
            L = len(disorder)
            for i, s in enumerate(disorder):
                if s >= threshold and not in_region:
                    in_region = True; start = i + 1
                elif s < threshold and in_region:
                    in_region = False
                    mid = (start + i) // 2
                    js_parts.append(
                        f"viewer.addLabel('IDR {start}–{i}',"
                        f"{{position:{{resi:{mid}}},backgroundColor:'0x4361ee',"
                        f"fontColor:'white',fontSize:8,backgroundOpacity:0.7}});")
            if in_region:
                mid = (start + L) // 2
                js_parts.append(
                    f"viewer.addLabel('IDR {start}–{L}',"
                    f"{{position:{{resi:{mid}}},backgroundColor:'0x4361ee',"
                    f"fontColor:'white',fontSize:8,backgroundOpacity:0.7}});")
        # Signal peptide label
        sp_end = sp.get("sp_end", 0) if isinstance(sp, dict) else 0
        if sp_end and sp_end > 0:
            js_parts.append(
                f"viewer.addLabel('Signal peptide (1–{sp_end})',"
                f"{{position:{{resi:{max(1, sp_end//2)}}},backgroundColor:'0xf72585',"
                f"fontColor:'white',fontSize:8,backgroundOpacity:0.7}});")
        js_parts.append("viewer.render();}")
        try:
            self._js("".join(js_parts))
        except Exception:
            pass
        self._push_all_residue_scores()

    def _push_all_residue_scores(self) -> None:
        """Push per-residue multi-feature scores to the 3Dmol popup layer."""
        if not _WEBENGINE_AVAILABLE or self.structure_viewer is None:
            return
        ad = self.analysis_data or {}
        _KEY_MAP = [
            ("disorder",  "disorder_scores"),
            ("signal",    "sp_bilstm_profile"),
            ("tm",        "tm_bilstm_profile"),
            ("intramem",  "intramem_bilstm_profile"),
            ("cc",        "cc_bilstm_profile"),
            ("dna",       "dna_bilstm_profile"),
            ("act",       "act_bilstm_profile"),
            ("bnd",       "bnd_bilstm_profile"),
            ("phos",      "phos_bilstm_profile"),
            ("lcd",       "lcd_bilstm_profile"),
            ("znf",       "znf_bilstm_profile"),
            ("glyc",      "glyc_bilstm_profile"),
            ("ubiq",      "ubiq_bilstm_profile"),
            ("meth",      "meth_bilstm_profile"),
            ("acet",      "acet_bilstm_profile"),
            ("lipid",     "lipid_bilstm_profile"),
            ("disulf",    "disulf_bilstm_profile"),
            ("motif",     "motif_bilstm_profile"),
            ("prop",      "prop_bilstm_profile"),
            ("rep",       "rep_bilstm_profile"),
            ("rnabind",   "rnabind_bilstm_profile"),
            ("nucbind",   "nucbind_bilstm_profile"),
            ("transit",   "transit_bilstm_profile"),
        ]
        all_scores: dict[int, dict] = {}
        for js_key, data_key in _KEY_MAP:
            arr = ad.get(data_key) or []
            for i, v in enumerate(arr):
                resi = i + 1
                if resi not in all_scores:
                    all_scores[resi] = {}
                all_scores[resi][js_key] = round(float(v), 4)
        # pLDDT (AlphaFold, 0-100 → stored 0-1) or B-factor (crystallographic, Å², stored raw)
        af = self.alphafold_data or {}
        bfac_arr = af.get("plddt") or []
        is_af = getattr(self, "_struct_is_alphafold", False)
        bfac_key = "plddt" if is_af else "bfactor"
        for i, v in enumerate(bfac_arr):
            resi = i + 1
            if resi not in all_scores:
                all_scores[resi] = {}
            val = round(float(v) / 100.0, 4) if is_af else round(float(v), 2)
            all_scores[resi][bfac_key] = val
        if all_scores:
            self._js(f"setAllResidueScores({json.dumps(all_scores)});")

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
        self.blast_stop_btn = QPushButton("Stop BLAST")
        self.blast_stop_btn.setMinimumHeight(30)
        self.blast_stop_btn.setObjectName("danger_btn")
        self.blast_stop_btn.clicked.connect(self._stop_blast)
        self.blast_stop_btn.setVisible(False)
        ctrl_row.addWidget(self.blast_stop_btn)
        ctrl_row.addStretch()
        layout.addLayout(ctrl_row)

        self.blast_status_lbl = QLabel("Ready.  Run analysis first, then click 'BLAST Current Sequence'.")
        self.blast_status_lbl.setObjectName("status_lbl")
        self.blast_status_lbl.setProperty("status_state", "idle")
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
            te.setPlaceholderText(f"Paste {lbl} here\u2026")
            te.setFont(QFont("Courier New", 10))
            te.setMaximumHeight(120)
            v.addWidget(te)
            setattr(self, attr, te)
            use_btn = QPushButton(f"Set {lbl} as Main Sequence")
            use_btn.setMinimumHeight(26)
            use_btn.clicked.connect(lambda _, a=attr: self._use_compare_seq(a))
            v.addWidget(use_btn)
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
        self.batch_load_btn = QPushButton("Load Selected")
        self.batch_load_btn.setMinimumHeight(30)
        self.batch_load_btn.setToolTip("Load the selected sequence into the Analysis tab and run analysis")
        self.batch_load_btn.clicked.connect(self._load_selected_batch_row)
        btn_row.addWidget(self.batch_export_csv_btn)
        btn_row.addWidget(self.batch_export_json_btn)
        btn_row.addWidget(self.batch_load_btn)
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
        self.batch_table.setToolTip("Select a row and click 'Load Selected', or double-click to view details")
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
            lbl.setObjectName("section_header")
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

        self.hydro_scale_combo = QComboBox()
        self.hydro_scale_combo.addItems(list(HYDROPHOBICITY_SCALES.keys()))
        self.hydro_scale_combo.setCurrentText(self.hydro_scale)
        self._set_tooltip(self.hydro_scale_combo,
            "Hydrophobicity scale for sliding-window profiles and GRAVY calculation.\n"
            "Kyte-Doolittle is the standard general-purpose scale.\n"
            "Wimley-White/Hessa/GES are best for membrane proteins.\n"
            "Urry is most relevant for IDP/phase-separation research.")
        form.addRow("Hydrophobicity Scale:", self.hydro_scale_combo)

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
  <li><b>Find UniProt ID</b> — paste any sequence and BEER will attempt to identify the matching UniProt Swiss-Prot entry by parsing FASTA headers or matching by exact sequence length. If found, the accession is set automatically and all external databases are populated.</li>
</ul>
<h2>UniProt Tracks</h2>
<p>After fetching a UniProt accession, click <b>UniProt Tracks</b> (Structure toolbar) to download all UniProt feature annotations for this protein.
Once loaded, BEER automatically adds dual-track graph panels that overlay the BEER AI Predictions (top) with the curated UniProt reference (bottom) for features including disorder, signal peptide, transmembrane helices, and more.</p>
<h2>Feature Coloring on 3D Structure</h2>
<p>In the Structure tab, select <b>Feature Score</b> from the color mode dropdown to color the protein by any per-residue AI prediction score.
Choose the feature (Disorder, Signal Peptide, etc.) from the adjacent selector; residues are colored white→feature color according to their predicted probability.</p>
<h2>Navigation</h2>
<p>Use the <b>left sidebar</b> to switch between sections. Keyboard shortcuts:</p>
<table>
  <tr><th>Shortcut</th><th>Action</th></tr>
  <tr><td>Ctrl+Enter</td><td>Run analysis</td></tr>
  <tr><td>Ctrl+G</td><td>Jump to Graphs</td></tr>
  <tr><td>Ctrl+2</td><td>Switch to Structure tab</td></tr>
  <tr><td>Ctrl+3</td><td>Switch to BLAST tab</td></tr>
  <tr><td>Ctrl+7</td><td>Switch to MSA tab</td></tr>
  <tr><td>Ctrl+Z</td><td>Undo last mutation</td></tr>
  <tr><td>Ctrl+S</td><td>Save session</td></tr>
  <tr><td>Ctrl+O</td><td>Load session</td></tr>
  <tr><td>Ctrl+F</td><td>Focus motif search</td></tr>
  <tr><td>Ctrl+/</td><td>Show full keyboard shortcut reference</td></tr>
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
Each graph has a <b>Save Graph</b> button to save that graph individually.</p>
<h2>Composition</h2>
<ul>
  <li><b>AA Composition (Bar)</b> — amino acid counts and frequencies, sortable by name, frequency, or hydrophobicity.</li>
</ul>
<h2>Profiles</h2>
<ul>
  <li><b>Hydrophobicity Profile</b> — Kyte-Doolittle sliding-window average (window set in Settings).</li>
  <li><b>Local Charge Profile</b> — sliding-window NCPR.</li>
  <li><b>Disorder Profile</b> — AI Predictions per-residue disorder probability (ESM2 → BiLSTM); fill = disordered (&gt;0.5).</li>
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
  <li><b>Single-Residue Perturbation Map</b> — 20×n heatmap of |ΔGRAVY| + |ΔNCPR| for every possible single-residue substitution; white dot = wild type. Highlights positions where amino acid identity most strongly influences global hydrophobicity and charge. Available for sequences ≤500 aa.</li>
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
<h2>Coiled-Coil Propensity Profile</h2>
<p>Heptad-weighted propensity profile computed with a 28-residue (4-heptad) sliding window. Positions a and d
(hydrophobic core) are down-weighted relative to a position-weighted Lupas-derived scale. The score is
normalised relative to the sequence maximum (not an absolute calibrated scale), so a high score means strong
coiled-coil character <em>relative to the rest of the sequence</em>. Regions above 0.50 are highlighted.
This is a propensity indicator — for definitive coiled-coil prediction use COILS or DeepCoil.</p>
<h2>Single-Residue Perturbation Map</h2>
<p>Every residue is substituted in silico to all 20 amino acids. The heatmap colour encodes
|ΔGRAVY| + |ΔNCPR| — the combined change in global hydrophobicity and net charge per residue.
Hot positions are those where the wild-type identity most strongly determines the global physicochemical
character of the sequence. Wild-type residues are shown as white dots. Available for sequences ≤ 500 aa.</p>
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
            ("AI Predictions", """
<h1>AI Predictions</h1>
<p>BEER v2.0 includes 24 per-residue AI prediction heads. Each head uses ESM2 650M embeddings
fed into a 2-layer BiLSTM classifier trained on curated structural databases and UniProt Swiss-Prot annotations.
Every head produces a per-residue probability in [0, 1].
After running <b>AI Analysis</b>, the Report tab shows an <b>AI Predictions</b> section with one
entry per head that ran, including predicted regions with residue ranges and the threshold used.
Heads without a trained model file are silently skipped.</p>
<h2>Architecture</h2>
<p>ESM2 650M (1280-dim, frozen) → 2-layer Bidirectional LSTM (hidden = 256) → Linear(512 → 1) → Sigmoid.
Transmembrane head uses a BiLSTM-CRF decoder (Viterbi decoding, enforces valid outside→helix→inside topology).
Aggregation head uses a BiLSTM-Window architecture (9-residue average-pool before sigmoid).
All heads are trained with focal loss and MMseqs2-clustered train/val/test splits.
Classification threshold is set at the F1-maximising point on the validation set (≈ 0.5 for most heads).</p>
<h2>All 24 Heads</h2>
<table>
  <tr><th>Head</th><th>Primary Training Source</th><th>Graph Tab</th></tr>
  <tr><td>Disorder</td><td>DisProt experimental → UniProt ft_region:disordered</td><td>Disorder Profile</td></tr>
  <tr><td>Signal Peptide</td><td>UniProt ft_signal (Swiss-Prot)</td><td>Signal Peptide Profile</td></tr>
  <tr><td>Transmembrane</td><td>UniProt ft_transmem → BiLSTM-CRF</td><td>Transmembrane Profile</td></tr>
  <tr><td>Intramembrane</td><td>UniProt ft_intramem</td><td>Intramembrane Profile</td></tr>
  <tr><td>Coiled-Coil</td><td>UniProt ft_coiled</td><td>Coiled-Coil Profile</td></tr>
  <tr><td>DNA-Binding</td><td>BioLiP (PDB-derived protein-DNA contacts)</td><td>DNA-Binding Profile</td></tr>
  <tr><td>RNA Binding</td><td>BioLiP (PDB-derived protein-RNA contacts)</td><td>RNA Binding Profile</td></tr>
  <tr><td>Active Site</td><td>M-CSA mechanistically validated catalytic residues</td><td>Active Site Profile</td></tr>
  <tr><td>Binding Site</td><td>BioLiP small-molecule binding residues</td><td>Binding Site Profile</td></tr>
  <tr><td>Phosphorylation</td><td>dbPTM (PSP + PhosphoELM + HPRD aggregate)</td><td>Phosphorylation Profile</td></tr>
  <tr><td>Low-Complexity</td><td>UniProt ft_compbias</td><td>Low-Complexity Profile</td></tr>
  <tr><td>Zinc Finger</td><td>BioLiP Zn-coordinating residues</td><td>Zinc Finger Profile</td></tr>
  <tr><td>Glycosylation</td><td>GlyConnect site-resolved glycoproteomics</td><td>Glycosylation Profile</td></tr>
  <tr><td>Ubiquitination</td><td>dbPTM</td><td>Ubiquitination Profile</td></tr>
  <tr><td>Methylation</td><td>dbPTM</td><td>Methylation Profile</td></tr>
  <tr><td>Acetylation</td><td>dbPTM</td><td>Acetylation Profile</td></tr>
  <tr><td>Lipidation</td><td>UniProt ft_lipid</td><td>Lipidation Profile</td></tr>
  <tr><td>Disulfide Bond</td><td>UniProt ft_disulfid</td><td>Disulfide Bond Profile</td></tr>
  <tr><td>Functional Motif</td><td>UniProt ft_motif</td><td>Functional Motif Profile</td></tr>
  <tr><td>Propeptide</td><td>UniProt ft_propep</td><td>Propeptide Profile</td></tr>
  <tr><td>Repeat Region</td><td>UniProt ft_repeat</td><td>Repeat Region Profile</td></tr>
  <tr><td>Nucleotide-Binding</td><td>BioLiP (ATP/ADP/NAD/FAD/CoA/…)</td><td>Nucleotide-Binding Profile</td></tr>
  <tr><td>Transit Peptide</td><td>UniProt ft_transit</td><td>Transit Peptide Profile</td></tr>
  <tr><td>Aggregation Propensity</td><td>WALTZ-DB 2.0 + AmyLoad + AmyPro + PDB fibrils</td><td>Aggregation Propensity Profile</td></tr>
</table>
<h2>UniProt Annotation Overlay</h2>
<p>Each graph tab shows the AI prediction probability curve as the primary element.
When <b>UniProt Tracks</b> are fetched (Structure toolbar), the curated Swiss-Prot annotation
for that feature is overlaid on the same axes as a semi-transparent background span and a rug strip.
A stats box shows sensitivity and precision of the AI prediction vs the UniProt reference.</p>
<h2>MC-Dropout Uncertainty</h2>
<p>For profile graphs, enable <b>Show Uncertainty (MC-Dropout)</b> to run 20 stochastic forward passes
(Gal &amp; Ghahramani 2016), producing per-residue mean ± 1σ confidence intervals shown as a shaded band.</p>
<h2>Classical Methods</h2>
<p>Classical Analyze (no ESM2) is always available as a fast fallback:
Disorder — Uversky sliding-window propensity; Signal Peptide — von Heijne D-score;
TM helices — Kyte-Doolittle w=19; Coiled-coil — heptad-weighted propensity.
Classical results are always shown in the Report tab regardless of AI Analysis status.</p>
<h2>ESM2 Status Indicator</h2>
<p>The <b>ESM2</b> status badge (top-right of the main window) shows: Ready, Busy, or Offline.
When offline, BEER falls back to classical sequence-based methods automatically.</p>
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
  <li><b>Default Graph Format</b> — PNG, SVG, or PDF for Save Graph.</li>
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

        self._help_browsers: list[tuple["QTextBrowser", str]] = []

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
            self._help_browsers.append((browser, html_body))

        help_nav.currentRowChanged.connect(help_stack.setCurrentIndex)
        help_nav.setCurrentRow(0)

        # Citation + Methods toolbar
        cite_bar = QHBoxLayout()

        cite_btn = QPushButton("Copy Citation (BibTeX)")
        cite_btn.setMinimumHeight(32)
        cite_btn.setToolTip("Copy BibTeX citation for BEER to clipboard")
        cite_btn.clicked.connect(self._copy_beer_citation)
        cite_bar.addWidget(cite_btn)

        about_btn = QPushButton("About BEER")
        about_btn.setMinimumHeight(32)
        about_btn.setToolTip("Version, author, and citation information")
        about_btn.clicked.connect(self._show_about)
        cite_bar.addWidget(about_btn)

        cite_bar.addStretch()
        outer_v.addLayout(cite_bar)

    def _show_about(self):
        import beer as _beer
        dlg = QDialog(self)
        dlg.setWindowTitle("About BEER")
        dlg.setMinimumWidth(420)
        layout = QVBoxLayout(dlg)
        layout.setSpacing(12)
        layout.setContentsMargins(28, 24, 28, 24)

        _logo_path = str(importlib.resources.files("beer").joinpath("beer.png"))
        if os.path.exists(_logo_path):
            logo_lbl = QLabel()
            pix = QPixmap(_logo_path).scaledToHeight(140, Qt.TransformationMode.SmoothTransformation)
            logo_lbl.setPixmap(pix)
            logo_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(logo_lbl)

        title_lbl = QLabel("<b style='font-size:16pt;'>BEER</b>")
        title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_lbl)

        sub_lbl = QLabel("Biophysical Evaluation Engine for Residues")
        sub_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(sub_lbl)

        ver_lbl = QLabel(f"Version {_beer.__version__}")
        ver_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(ver_lbl)

        layout.addWidget(_make_hsep())

        info_lbl = QLabel(
            "<p style='text-align:center;'>"
            "Saumyak Mukherjee<br>"
            "Theoretical Biophysics<br>"
            "Max Planck Institute of Biophysics, Frankfurt am Main, Germany"
            "</p>"
        )
        info_lbl.setWordWrap(True)
        info_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info_lbl)

        layout.addWidget(_make_hsep())

        cite_lbl = QLabel(
            "<p style='text-align:center;font-size:8pt;'>"
            "If you use BEER in your research, please cite:<br>"
            "Mukherjee, S. <i>arXiv</i>:2504.20561<br>"
            "<a href='https://doi.org/10.48550/arXiv.2504.20561'>https://doi.org/10.48550/arXiv.2504.20561</a>"
            "</p>"
        )
        cite_lbl.setOpenExternalLinks(True)
        cite_lbl.setWordWrap(True)
        cite_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(cite_lbl)

        license_lbl = QLabel("<p style='text-align:center;font-size:8pt;color:gray;'>GNU General Public License v2</p>")
        license_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(license_lbl)

        close_btn = QPushButton("Close")
        close_btn.setMinimumHeight(32)
        close_btn.clicked.connect(dlg.accept)
        layout.addWidget(close_btn)

        dlg.exec()

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
            f"(Biophysical Evaluation Engine for Residues), a Python desktop "
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
            # Load the full PDB into the 3D viewer (all chains visible + chain controls).
            first_id = entries[0][0]
            if first_id in self.batch_struct:
                self.alphafold_data = self.batch_struct[first_id]
            self._struct_is_alphafold = False
            self._load_structure_viewer(pdb_str)
            self.export_structure_btn.setEnabled(True)
            n_res = sum(len(seq) for _, seq in entries)
            self.af_status_lbl.setText(
                f"Loaded {os.path.basename(file_name)}  —  "
                f"{len(chain_structs)} chain(s), {n_res} residues total"
            )
            self.af_status_lbl.setProperty("status_state", "success")
            self.af_status_lbl.style().unpolish(self.af_status_lbl)
            self.af_status_lbl.style().polish(self.af_status_lbl)
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

    def import_mmcif(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open mmCIF File", "",
            "mmCIF Files (*.cif *.mmcif);;All Files (*)")
        if not file_name:
            return
        try:
            chains = import_mmcif_sequence(file_name)
        except Exception as e:
            QMessageBox.critical(self, "mmCIF Error", f"Failed to parse file: {e}")
            return
        if not chains:
            QMessageBox.warning(self, "No Chains", "No valid chains found in mmCIF file.")
            return
        try:
            with open(file_name, "r") as fh:
                cif_str = fh.read()
        except OSError:
            cif_str = None
        cif_base = os.path.splitext(os.path.basename(file_name))[0]
        entries  = [(f"{cif_base}_{cid}", seq) for cid, seq in chains.items()]
        self._load_batch(entries)
        if cif_str:
            chain_structs = extract_chain_structures_mmcif(cif_str)
            for cid_letter, struct in chain_structs.items():
                rec_id = f"{cif_base}_{cid_letter}"
                self.batch_struct[rec_id] = struct
            first_id = entries[0][0]
            if first_id in self.batch_struct:
                self.alphafold_data = self.batch_struct[first_id]
            self._struct_is_alphafold = False
            self._load_structure_viewer(cif_str)
            self.export_structure_btn.setEnabled(True)
            n_res = sum(len(seq) for _, seq in entries)
            self.af_status_lbl.setText(
                f"Loaded {os.path.basename(file_name)}  —  "
                f"{len(chain_structs)} chain(s), {n_res} residues total"
            )
            self.af_status_lbl.setProperty("status_state", "success")
            self.af_status_lbl.style().unpolish(self.af_status_lbl)
            self.af_status_lbl.style().polish(self.af_status_lbl)
        self.sequence_name = entries[0][0] if entries else cif_base
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
            f"<style>{get_report_css(self._is_dark)}</style>"
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

        # ── One-time ESM2 download warning ───────────────────────────────────
        from beer.embeddings import ESM2_AVAILABLE
        if ESM2_AVAILABLE and self._embedder is not None:
            import pathlib
            _mn = getattr(self._embedder, "model_name", "esm2_t33_650M_UR50D")
            _cache_pt = (pathlib.Path.home() / ".cache/torch/hub/checkpoints"
                         / f"{_mn}.pt")
            if not _cache_pt.exists() and not getattr(self, "_esm2_download_warned", False):
                self._esm2_download_warned = True
                _sizes = {
                    "esm2_t6_8M_UR50D": "~30 MB", "esm2_t12_35M_UR50D": "~140 MB",
                    "esm2_t30_150M_UR50D": "~580 MB", "esm2_t33_650M_UR50D": "~2.6 GB",
                }
                _sz = _sizes.get(_mn, "~2.6 GB")
                reply = QMessageBox.information(
                    self, "First-time ESM2 Setup",
                    f"<b>One-time model download required</b><br><br>"
                    f"The ESM2 650M language model ({_sz}) will be downloaded "
                    f"from Meta's model hub on the first analysis.<br><br>"
                    f"<b>Estimated time:</b> 2–15 minutes depending on your connection.<br>"
                    f"<b>Location:</b> <code>~/.cache/torch/hub/checkpoints/</code><br><br>"
                    f"BEER will appear frozen during the download — this is normal. "
                    f"The model is cached permanently; subsequent runs are instant.",
                    QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
                )
                if reply == QMessageBox.StandardButton.Cancel:
                    return

        self._last_was_bilstm = False
        self.analyze_btn.setEnabled(False)
        self.bilstm_analyze_btn.setEnabled(False)
        self.statusBar.showMessage("Analyzing…")

        self._progress_dlg = QProgressDialog(
            "Running classical analysis…", "Cancel", 0, 0, self)
        self._progress_dlg.setWindowTitle("BEER Analysis")
        self._progress_dlg.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress_dlg.setMinimumDuration(500)
        self._progress_dlg.canceled.connect(self._cancel_analysis)
        self._progress_dlg.show()

        # Classical analysis: embedder=None skips ESM2 and all BiLSTM heads
        self._analysis_worker = AnalysisWorker(
            seq, pH, self.default_window_size, self.use_reducing, self.custom_pka,
            hydro_scale=self.hydro_scale,
            embedder=None,
        )
        self._analysis_worker.finished.connect(self._on_worker_finished)
        self._analysis_worker.error.connect(self._on_worker_error)
        self._analysis_worker.start()

    def on_bilstm_analyze(self):
        """Run ESM2 embedding + all AI prediction heads on the current sequence."""
        if not self.analysis_data:
            QMessageBox.warning(self, "AI Analysis",
                                "Run classical Analyze first.")
            return
        seq = self.analysis_data["seq"]

        try:
            pH = float(self.ph_input.text())
        except ValueError:
            pH = self.default_pH

        # One-time download warning (same as before, but only shown for BiLSTM)
        from beer.embeddings import ESM2_AVAILABLE
        if ESM2_AVAILABLE and self._embedder is not None:
            import pathlib
            _mn = getattr(self._embedder, "model_name", "esm2_t33_650M_UR50D")
            _cache_pt = (pathlib.Path.home() / ".cache/torch/hub/checkpoints"
                         / f"{_mn}.pt")
            if not _cache_pt.exists() and not getattr(self, "_esm2_download_warned", False):
                self._esm2_download_warned = True
                _sizes = {
                    "esm2_t6_8M_UR50D": "~30 MB", "esm2_t12_35M_UR50D": "~140 MB",
                    "esm2_t30_150M_UR50D": "~580 MB", "esm2_t33_650M_UR50D": "~2.6 GB",
                }
                _sz = _sizes.get(_mn, "~2.6 GB")
                reply = QMessageBox.information(
                    self, "First-time ESM2 Setup",
                    f"<b>One-time model download required</b><br><br>"
                    f"The ESM2 650M language model ({_sz}) will be downloaded "
                    f"from Meta's model hub on the first run.<br><br>"
                    f"<b>Estimated time:</b> 2–15 minutes depending on your connection.<br>"
                    f"<b>Location:</b> <code>~/.cache/torch/hub/checkpoints/</code><br><br>"
                    f"BEER will appear frozen during download — this is normal. "
                    f"The model is cached permanently; subsequent runs are instant.",
                    QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
                )
                if reply == QMessageBox.StandardButton.Cancel:
                    return

        self._last_was_bilstm = True
        self.bilstm_analyze_btn.setEnabled(False)
        self.analyze_btn.setEnabled(False)
        self.statusBar.showMessage("Running AI Analysis (ESM2 650M)…")

        _mn = getattr(self._embedder, "model_name", "") if self._embedder else ""
        _parts = _mn.split("_")
        try:
            _tag = f"ESM2 {next(p for p in _parts if p.endswith('M') or p.endswith('B'))}"
        except StopIteration:
            _tag = "ESM2 650M"
        self._progress_dlg = QProgressDialog(
            f"Running AI Analysis ({_tag})…", "Cancel", 0, 0, self)
        self._progress_dlg.setWindowTitle("BEER AI Analysis")
        self._progress_dlg.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress_dlg.setMinimumDuration(300)
        self._progress_dlg.canceled.connect(self._cancel_analysis)
        self._progress_dlg.show()

        self._analysis_worker = AnalysisWorker(
            seq, pH, self.default_window_size, self.use_reducing, self.custom_pka,
            hydro_scale=self.hydro_scale,
            embedder=self._embedder,
        )
        self._analysis_worker.finished.connect(self._on_worker_finished)
        self._analysis_worker.error.connect(self._on_worker_error)
        self._analysis_worker.start()

    def _refresh_report_sections(self):
        """Re-render all report section browsers with the current theme CSS."""
        if not self.analysis_data:
            return
        old_css = REPORT_CSS_DARK if not self._is_dark else REPORT_CSS
        new_css = get_report_css(self._is_dark)
        sections = self.analysis_data.get("report_sections", {})
        for sec, browser in self.report_section_tabs.items():
            html = sections.get(sec, "")
            if not html:
                continue
            html = html.replace(old_css, new_css)
            browser.setHtml(html)

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

        is_dark = hasattr(self, "theme_toggle") and self.theme_toggle.isChecked()
        bg_color   = "#1a1a2e" if is_dark else "#ffffff"
        text_color = "#e2e8f0" if is_dark else "#1a1a2e"
        pos_color  = "#94a3b8" if is_dark else "#718096"
        hdr_color  = "#4cc9f0" if is_dark else "#4361ee"

        lines      = text.split("\n")
        html_lines = []
        for ln in lines:
            if ln.startswith(">"):
                html_lines.append(
                    f'<span style="color:{hdr_color};font-weight:700;">{ln}</span>'
                )
            elif ln and ln.lstrip()[0:1].isdigit():
                parts = ln.split("  ", 1)
                if len(parts) == 2:
                    pos_str, seq_str = parts
                    coloured = _colour_residues(seq_str)
                    html_lines.append(
                        f'<span style="color:{pos_color};">{pos_str}</span>'
                        f'&nbsp;&nbsp;{coloured}'
                    )
                else:
                    html_lines.append(_colour_residues(ln))
            else:
                html_lines.append(f'<span style="color:{text_color};">{ln}</span>')
        html = (
            f'<style>body{{font-family:"Courier New",monospace;font-size:10pt;'
            f'background:{bg_color};padding:8px;line-height:2.0;}}</style>'
            + "<br>".join(html_lines)
        )
        self.seq_viewer.setHtml(html)

    def update_graph_tabs(self):
        """Register lazy graph generators; only render the currently-visible graph now."""
        if not self.analysis_data:
            return
        self._build_graph_generators()
        self._generated_graphs.clear()
        # Immediately render whichever graph is currently selected
        self._render_visible_graph()

    def _build_graph_generators(self):
        """Populate self._graph_generators with {title: callable} for all graphs."""
        if not self.analysis_data:
            return
        ad  = self.analysis_data       # captured by reference — always current
        seq = ad["seq"]
        lf  = self.label_font_size
        tf  = self.tick_font_size
        sh  = self.show_heading
        sg  = self.show_grid
        sbl = self.show_bead_labels
        cm  = self.colormap
        hcm = self.heatmap_cmap
        pk  = self.custom_pka
        sn  = self.sequence_name
        hs  = self.hydro_scale

        def _wrap(fn):
            """Apply heading/grid overrides to ALL axes and suptitle."""
            fig = fn()
            if not sh:
                # Remove both figure-level suptitle and any per-axes title
                fig.suptitle("")
                for ax in fig.axes:
                    ax.set_title("")
            for ax in fig.axes:
                if sg:
                    ax.grid(True, linestyle="--", linewidth=0.3,
                            alpha=0.5, color="#c8cdd8")
                    ax.set_axisbelow(True)
                else:
                    ax.grid(False)
            # Provenance watermark
            _prov = f"BEER v2.0  |  {sn}" if sn else "BEER v2.0"
            fig.text(0.99, 0.01, _prov, ha="right", va="bottom",
                     fontsize=7, color="#adb5bd",
                     transform=fig.transFigure)
            # Apply ROI highlight if set
            self._apply_roi_to_figure(fig)
            return fig

        gens = {}
        gens["Amino Acid Composition (Bar)"] = lambda: _wrap(lambda: create_amino_acid_composition_figure(
            ad["aa_counts"], ad["aa_freq"], label_font=lf, tick_font=tf))
        gens["Hydrophobicity Profile"] = lambda: _wrap(lambda: create_hydrophobicity_figure(
            ad["hydro_profile"], ad["window_size"], hs, label_font=lf, tick_font=tf))
        gens["Sticker Map"] = lambda: _wrap(lambda: create_sticker_map_figure(
            seq, sbl, label_font=lf, tick_font=tf))
        gens["Local Charge Profile"] = lambda: _wrap(lambda: create_local_charge_figure(
            ad["ncpr_profile"], ad["window_size"], label_font=lf, tick_font=tf))
        gens["Cation\u2013\u03c0 Map"] = lambda: _wrap(lambda: create_cation_pi_map_figure(
            seq, label_font=lf, tick_font=tf, cmap=hcm))
        gens["Isoelectric Focus"] = lambda: _wrap(lambda: create_isoelectric_focus_figure(
            seq, label_font=lf, tick_font=tf, pka=pk))
        gens["Helical Wheel"] = lambda: _wrap(lambda: create_helical_wheel_figure(seq, label_font=lf, hydro_scale=hs, cmap=self.heatmap_cmap))
        gens["Charge Decoration"] = lambda: _wrap(lambda: create_charge_decoration_figure(
            ad["fcr"], ad["ncpr"], label_font=lf, tick_font=tf))
        gens["Linear Sequence Map"] = lambda: _wrap(lambda: create_linear_sequence_map_figure(
            seq, ad["hydro_profile"], ad["ncpr_profile"], ad["disorder_scores"],
            label_font=lf, tick_font=tf))
        # BiLSTM heads — registered whenever the data key is present in analysis_data.
        # This covers both full AI Analysis and lazy per-head computation.
        _uniprot_feats = getattr(self, "_uniprot_features", {})
        _ALL_BILSTM_HEADS: list[tuple[str, str, str]] = [
            ("disorder_scores",         "disorder",          "Disorder Profile"),
            ("sp_bilstm_profile",       "signal_peptide",    "Signal Peptide Profile"),
            ("tm_bilstm_profile",       "transmembrane",     "Transmembrane Profile"),
            ("intramem_bilstm_profile", "intramembrane",     "Intramembrane Profile"),
            ("cc_bilstm_profile",       "coiled_coil",       "Coiled-Coil Profile"),
            ("dna_bilstm_profile",      "dna_binding",       "DNA-Binding Profile"),
            ("act_bilstm_profile",      "active_site",       "Active Site Profile"),
            ("bnd_bilstm_profile",      "binding_site",      "Binding Site Profile"),
            ("phos_bilstm_profile",     "phosphorylation",   "Phosphorylation Profile"),
            ("lcd_bilstm_profile",      "lcd",               "Low-Complexity Profile"),
            ("znf_bilstm_profile",      "zinc_finger",       "Zinc Finger Profile"),
            ("glyc_bilstm_profile",     "glycosylation",     "Glycosylation Profile"),
            ("ubiq_bilstm_profile",     "ubiquitination",    "Ubiquitination Profile"),
            ("meth_bilstm_profile",     "methylation",       "Methylation Profile"),
            ("acet_bilstm_profile",     "acetylation",       "Acetylation Profile"),
            ("lipid_bilstm_profile",    "lipidation",        "Lipidation Profile"),
            ("disulf_bilstm_profile",   "disulfide",         "Disulfide Bond Profile"),
            ("motif_bilstm_profile",    "motif",             "Functional Motif Profile"),
            ("prop_bilstm_profile",     "propeptide",        "Propeptide Profile"),
            ("rep_bilstm_profile",      "repeat",            "Repeat Region Profile"),
            ("rnabind_bilstm_profile",  "rna_binding",       "RNA Binding Profile"),
            ("nucbind_bilstm_profile",  "nucleotide_binding","Nucleotide-Binding Profile"),
            ("transit_bilstm_profile",  "transit_peptide",   "Transit Peptide Profile"),
            ("agg_bilstm_profile",      "aggregation",       "Aggregation Propensity Profile"),
        ]
        for _ad_key, _feat, _tab_name in _ALL_BILSTM_HEADS:
            if ad.get(_ad_key):
                _extra_kw = {}
                if _ad_key == "disorder_scores":
                    _extra_kw["uncertainty"] = ad.get("disorder_uncertainty")
                gens[_tab_name] = (
                    lambda f=_feat, s=ad[_ad_key], kw=_extra_kw:
                    _wrap(lambda: create_bilstm_profile_figure(
                        f, s,
                        uniprot_regions=_uniprot_feats.get(f) or None,
                        label_font=lf, tick_font=tf, **kw))
                )
        gens["TM Topology"] = lambda: _wrap(lambda: create_tm_topology_figure(
            seq, ad.get("tm_helices", []), label_font=lf, tick_font=tf))
        gens["Uversky Phase Plot"] = lambda: _wrap(lambda: create_uversky_phase_plot(
            seq, label_font=lf, tick_font=tf))
        gens["Single-Residue Perturbation Map"] = lambda: _wrap(lambda: create_saturation_mutagenesis_figure(
            seq, label_font=lf, tick_font=tf, cmap=hcm))
        gens["Domain Architecture"] = lambda: _wrap(lambda: create_domain_architecture_figure(
            len(seq), self.pfam_domains, seq=seq,
            disorder_scores=ad.get("disorder_scores"),
            tm_helices=None,  # classical TMHMM shown separately in TM Topology
            label_font=lf, tick_font=tf))
        _annot_aggr = ad.get("aggr_profile", calc_aggregation_profile(seq))
        gens["Annotation Track"] = lambda: _wrap(lambda: create_annotation_track_figure(
            seq, ad.get("disorder_scores", []), ad.get("hydro_profile", []),
            _annot_aggr,
            ad.get("tm_helices", []),
            ad.get("larks", []), ad.get("sp_result", {}),
            label_font=lf, tick_font=tf))
        gens["Cleavage Map"] = lambda: _wrap(lambda: create_cleavage_map_figure(
            seq, ad.get("prot_sites", {}), label_font=lf, tick_font=tf))
        if _HAS_AGGREGATION:
            _aggr_prof = ad.get("aggr_profile", calc_aggregation_profile(seq))
            gens["\u03b2-Aggregation Profile"] = lambda: _wrap(lambda: create_aggregation_profile_figure(
                seq, _aggr_prof, predict_aggregation_hotspots(seq),
                label_font=lf, tick_font=tf))
            gens["Solubility Profile"] = lambda: _wrap(lambda: create_solubility_profile_figure(
                seq, calc_camsolmt_score(seq), label_font=lf, tick_font=tf))
        _am_data = getattr(self, "_alphafold_missense_data", None)
        if _am_data:
            gens["AlphaMissense"] = lambda: _wrap(lambda: create_alphafold_missense_figure(
                _am_data, seq=seq, label_font=lf, tick_font=tf, cmap=self.heatmap_cmap))
        else:
            def _am_placeholder_fig():
                from matplotlib.figure import Figure as _Fig
                _f = _Fig(figsize=(9, 4), dpi=120)
                _ax = _f.add_subplot(111)
                _ax.text(0.5, 0.55, "AlphaMissense data not loaded.",
                         ha="center", va="center", transform=_ax.transAxes,
                         fontsize=13, color="#374151")
                _ax.text(0.5, 0.42, "Fetch a UniProt entry first, then click\n"
                         "the \u201cAlphaMissense\u201d button in the toolbar.",
                         ha="center", va="center", transform=_ax.transAxes,
                         fontsize=10, color="#718096")
                _ax.set_axis_off()
                return _f
            gens["AlphaMissense"] = lambda: _wrap(_am_placeholder_fig)
        if _HAS_AMPHIPATHIC:
            gens["Hydrophobic Moment"] = lambda: _wrap(lambda: create_hydrophobic_moment_figure(
                seq, ad.get("moment_alpha", []), ad.get("moment_beta", []),
                ad.get("amph_regions", []), label_font=lf, tick_font=tf))
        if _HAS_RBP:
            gens["RNA-Binding Profile"] = lambda: _wrap(lambda: create_rbp_profile_figure(
                seq, ad.get("rbp_profile", []),
                ad.get("rbp", {}).get("motifs_found", []),
                label_font=lf, tick_font=tf))
        if _HAS_SCD:
            gens["SCD Profile"] = lambda: _wrap(lambda: create_scd_profile_figure(
                seq, ad.get("scd_profile", []), window=20, label_font=lf, tick_font=tf))
        if ad.get("shd_profile"):
            _shd = ad["shd_profile"]
            _hs_shd = ad.get("hydro_scale", hs)
            gens["SHD Profile"] = lambda: _wrap(lambda: create_shd_profile_figure(
                seq, _shd, window=20, scale_name=_hs_shd, label_font=lf, tick_font=tf))
        if ad.get("plaac"):
            gens["PLAAC Profile"] = lambda: _wrap(lambda: create_plaac_profile_figure(
                ad["plaac"], label_font=lf, tick_font=tf))

        # Structure-derived (require loaded PDB — independent of AlphaFold)
        _pdb_for_ss = getattr(self, "_struct_pdb_str", None)
        if _pdb_for_ss:
            gens["SS Bead Model"] = lambda _p=_pdb_for_ss: _wrap(
                lambda: create_bead_model_ss_figure(
                    _p, show_labels=True, label_font=lf, tick_font=tf))

        if getattr(self, "_struct_sasa_data", {}):
            _rsa = dict(self._struct_sasa_data)
            _asa = dict(self._struct_sasa_raw)
            _win = self.default_window_size
            _show_asa = getattr(self, "_sasa_show_asa", False)
            gens["SASA Profile"] = lambda: _wrap(lambda: create_sasa_figure(
                _rsa, _asa, window=_win, show_asa=_show_asa,
                label_font=lf, tick_font=tf))

        # Structure-dependent
        afd = self.alphafold_data
        if afd:
            plddt = afd.get("plddt")
            if plddt and len(plddt) == len(seq):
                _is_af = getattr(self, "_struct_is_alphafold", False)
                gens["pLDDT Profile"] = lambda _iaf=_is_af: _wrap(
                    lambda: create_plddt_figure(
                        afd["plddt"], label_font=lf, tick_font=tf,
                        use_bfactor=not _iaf))
            dm = afd.get("dist_matrix")
            if dm is not None and dm.ndim == 2 and dm.shape[0] == len(seq) > 0:
                gens["Distance Map"] = lambda: _wrap(lambda: create_distance_map_figure(
                    afd["dist_matrix"], label_font=lf, tick_font=tf, cmap=hcm))
                gens["Residue Contact Network"] = lambda: _wrap(lambda: create_contact_network_figure(
                    seq, afd["dist_matrix"], label_font=lf, tick_font=tf, cmap=hcm))
            if _HAS_PHI_PSI:
                gens["Ramachandran Plot"] = lambda: _wrap(lambda: create_ramachandran_figure(
                    _extract_phi_psi(afd["pdb_str"]), label_font=lf, tick_font=tf))

        # MSA
        if self._msa_sequences:
            gens["MSA Conservation"] = lambda: _wrap(lambda: create_msa_conservation_figure(
                self._msa_sequences, self._msa_names, label_font=lf, tick_font=tf))
        if self._msa_mi_apc is not None:
            gens["MSA Covariance"] = lambda: _wrap(lambda: create_msa_covariance_figure(
                self._msa_mi_apc, label_font=lf, tick_font=tf, cmap=hcm))

        # Variant Effect Map (ESM2)
        if self._embedder is not None and "Variant Effect Map" in self.graph_tabs:
            gens["Variant Effect Map"] = lambda: self._gen_variant_effect_fig(
                seq, lf, tf, cmap=self.heatmap_cmap)

        self._graph_generators = gens

    def _render_visible_graph(self) -> None:
        """Render the graph currently selected in the tree (if not already done)."""
        item = self.graph_tree.currentItem()
        if item is None:
            return
        title = item.data(0, Qt.ItemDataRole.UserRole)
        if title:
            self._render_graph(title)

    def _render_graph(self, title: str) -> None:
        """Generate and display a single graph on demand."""
        if title not in self._graph_generators:
            return
        if title in self._generated_graphs:
            return  # already rendered; canvas is still in the layout
        try:
            import logging as _log
            fig = self._graph_generators[title]()
            self._replace_graph(title, fig)
            self._generated_graphs.add(title)
        except Exception as _exc:
            import logging as _log2
            _log2.getLogger("beer.graphs").warning(
                "Failed to render graph '%s': %s", title, _exc, exc_info=True)

    def _gen_variant_effect_fig(self, seq: str, lf: int, tf: int, cmap: str = "RdBu_r"):
        """Generate ESM2 variant effect map (called lazily)."""
        from beer.analysis.variant_scoring import compute_single_mutant_llr
        from beer.graphs.variant_map import create_variant_effect_figure
        llr = compute_single_mutant_llr(seq, self._embedder)
        if llr is None:
            from matplotlib.figure import Figure as _Fig
            fig = _Fig(figsize=(6, 3))
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "ESM2 not available for variant scoring",
                    ha="center", va="center", transform=ax.transAxes, color="#718096")
            ax.axis("off")
            return fig
        return create_variant_effect_figure(seq, llr, label_font=lf, tick_font=tf, cmap=cmap)

    def show_batch_details(self, row, _):
        sid = self.batch_table.item(row, 0).text()
        for cid, seq, data in self.batch_data:
            if cid == sid:
                self.seq_text.setPlainText(seq)
                self.analysis_data  = data
                self.sequence_name  = cid
                secs = data.get("report_sections", {})
                for sec, browser in self.report_section_tabs.items():
                    if sec in secs:
                        browser.setHtml(secs[sec])
                self._restore_chain_structure(cid)
                self._update_seq_viewer()
                self.update_graph_tabs()
                return

    def _load_selected_batch_row(self):
        """Load the currently selected multichain table row into the Analysis tab."""
        rows = self.batch_table.selectedItems()
        if not rows:
            QMessageBox.information(self, "Load Selected",
                                    "Select a row in the table first.")
            return
        row = self.batch_table.currentRow()
        self.show_batch_details(row, None)
        self.main_tabs.setCurrentIndex(0)

    # --- Graph tree handler ---

    def _on_graph_tree_clicked(self, item: QTreeWidgetItem, _col: int):
        title = item.data(0, Qt.ItemDataRole.UserRole)
        if not title or title not in self._graph_title_to_stack_idx:
            return
        self.graph_stack.setCurrentIndex(self._graph_title_to_stack_idx[title])
        # If this is a BiLSTM graph and the data isn't ready yet, trigger it.
        data_key = _GRAPH_TITLE_TO_AI_DATA_KEY.get(title)
        sec_key  = _GRAPH_TITLE_TO_AI_SEC.get(title)
        if (data_key and sec_key
                and self.analysis_data
                and not self.analysis_data.get(data_key)
                and sec_key not in self._ai_computed_sections):
            self._trigger_ai_section(sec_key)
            # Also navigate the report sidebar to the matching section.
            self._select_report_section(sec_key)
            return  # graph will render when _on_ai_section_finished fires
        self._render_graph(title)  # lazy: no-op if already rendered

    def _graph_nav_next(self) -> None:
        """Select the next leaf in the graph tree."""
        self._graph_nav_step(+1)

    def _graph_nav_prev(self) -> None:
        """Select the previous leaf in the graph tree."""
        self._graph_nav_step(-1)

    def _graph_nav_step(self, direction: int) -> None:
        """Move to the next (+1) or previous (-1) leaf item in the graph tree."""
        leaves = []
        for i in range(self.graph_tree.topLevelItemCount()):
            cat = self.graph_tree.topLevelItem(i)
            for j in range(cat.childCount()):
                leaves.append(cat.child(j))
        if not leaves:
            return
        cur = self.graph_tree.currentItem()
        try:
            idx = leaves.index(cur)
        except ValueError:
            idx = -1
        new_idx = (idx + direction) % len(leaves)
        new_item = leaves[new_idx]
        self.graph_tree.setCurrentItem(new_item)
        self._on_graph_tree_clicked(new_item, 0)

    # --- Export ---
    # (Export Complete Report removed in v2.0 — use per-section Export buttons in Report tab)

    def export_structure_dialog(self):
        """Show format chooser then export structure or sequence."""
        has_struct = bool(self.alphafold_data)
        has_seq    = bool(self.analysis_data)
        if not has_struct and not has_seq:
            QMessageBox.warning(self, "Export Structure / Sequence",
                                "No structure or sequence loaded.")
            return
        dlg = FormatChooserDialog(
            "Export Structure / Sequence",
            [
                ("PDB (.pdb)   — requires loaded structure",   "pdb",   has_struct),
                ("mmCIF (.cif) — requires loaded structure",   "mmcif", has_struct),
                ("GRO (.gro)   — requires loaded structure",   "gro",   has_struct),
                ("XYZ (.xyz)   — requires loaded structure",   "xyz",   has_struct),
                ("FASTA (.fasta) — requires analysis",         "fasta", has_seq),
            ],
            self,
        )
        if dlg.exec() != QDialog.Accepted:
            return
        self._export_structure_as(dlg.selected_key())

    def _export_structure_as(self, fmt: str):
        """Write the current structure/sequence in the requested format."""
        name = self.sequence_name or "structure"
        ext_map = {"pdb": "pdb", "mmcif": "cif", "gro": "gro",
                   "xyz": "xyz", "fasta": "fasta"}
        flt_map = {
            "pdb":   "PDB Files (*.pdb)",
            "mmcif": "mmCIF Files (*.cif)",
            "gro":   "GROMACS GRO Files (*.gro)",
            "xyz":   "XYZ Files (*.xyz)",
            "fasta": "FASTA Files (*.fasta *.fa)",
        }
        ext = ext_map[fmt]
        fn, _ = QFileDialog.getSaveFileName(
            self, "Export Structure / Sequence",
            f"{name}.{ext}", flt_map[fmt])
        if not fn:
            return

        try:
            if fmt == "fasta":
                seq  = self.analysis_data.get("seq", "")
                sname = self.sequence_name or "protein"
                with open(fn, "w") as f:
                    f.write(f">{sname}\n")
                    for i in range(0, len(seq), 60):
                        f.write(seq[i:i + 60] + "\n")

            else:
                pdb_str = self.alphafold_data["pdb_str"]
                if fmt == "pdb":
                    content = pdb_str
                elif fmt == "mmcif":
                    content = pdb_to_mmcif(pdb_str)
                elif fmt == "gro":
                    content = pdb_to_gro(pdb_str)
                elif fmt == "xyz":
                    content = pdb_to_xyz(pdb_str)
                with open(fn, "w") as f:
                    f.write(content)

            self.statusBar.showMessage(
                f"Exported {fmt.upper()} to {os.path.basename(fn)}", 4000)
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def save_graph(self, title: str, fmt: str = ""):
        tab, vb = self.graph_tabs[title]
        canvas  = self._find_canvas(vb)
        if not canvas:
            QMessageBox.warning(self, "No Graph", "Graph not available.")
            return
        ext  = (fmt or self.default_graph_format).lower()
        fn, _ = QFileDialog.getSaveFileName(
            self, "Save Graph", "", f"{ext.upper()} Files (*.{ext})"
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

    def _copy_graph_to_clipboard(self, title: str):
        _, vb = self.graph_tabs[title]
        canvas = self._find_canvas(vb)
        if not canvas:
            return
        buf = BytesIO()
        canvas.figure.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        QApplication.clipboard().setImage(QImage.fromData(buf.getvalue()))
        self.statusBar.showMessage("Figure copied to clipboard.", 2000)

    def export_graph_data(self, title: str):
        if not self.analysis_data:
            QMessageBox.warning(self, "No Data", "Run analysis first.")
            return
        afd = self.alphafold_data or {}
        extra = {
            "pfam_domains":       self.pfam_domains,
            "plddt":              afd.get("plddt", []),
            "alphafold_missense": getattr(self, "_alphafold_missense_data", None) or {},
            "msa_sequences":      self._msa_sequences,
        }
        result = get_graph_data(title, self.analysis_data, extra)
        if result is None:
            QMessageBox.information(self, "Not Available",
                                    f"No exportable data for '{title}'.")
            return
        stem, content, ext = result
        ext_filter = "CSV Files (*.csv)" if ext == "csv" else "JSON Files (*.json)"
        fn, _ = QFileDialog.getSaveFileName(
            self, "Export Graph Data", f"{stem}.{ext}", ext_filter)
        if fn:
            if not fn.lower().endswith(f".{ext}"):
                fn += f".{ext}"
            with open(fn, "w", encoding="utf-8") as fh:
                fh.write(content)
            self.statusBar.showMessage(f"Data exported to {os.path.basename(fn)}", 2000)

    # ── BiLSTM uncertainty ───────────────────────────────────────────────────

    _TAB_TO_KEY: dict = {
        "Disorder Profile":        ("disorder_scores",         "disorder"),
        "Signal Peptide Profile":  ("sp_bilstm_profile",       "signal_peptide"),
        "Transmembrane Profile":   ("tm_bilstm_profile",       "transmembrane"),
        "Intramembrane Profile":   ("intramem_bilstm_profile", "intramembrane"),
        "Coiled-Coil Profile":     ("cc_bilstm_profile",       "coiled_coil"),
        "DNA-Binding Profile":     ("dna_bilstm_profile",      "dna_binding"),
        "Active Site Profile":     ("act_bilstm_profile",      "active_site"),
        "Binding Site Profile":    ("bnd_bilstm_profile",      "binding_site"),
        "Phosphorylation Profile": ("phos_bilstm_profile",     "phosphorylation"),
        "Low-Complexity Profile":  ("lcd_bilstm_profile",      "lcd"),
        "Zinc Finger Profile":     ("znf_bilstm_profile",      "zinc_finger"),
        "Glycosylation Profile":   ("glyc_bilstm_profile",     "glycosylation"),
        "Ubiquitination Profile":  ("ubiq_bilstm_profile",     "ubiquitination"),
        "Methylation Profile":     ("meth_bilstm_profile",     "methylation"),
        "Acetylation Profile":     ("acet_bilstm_profile",     "acetylation"),
        "Lipidation Profile":      ("lipid_bilstm_profile",    "lipidation"),
        "Disulfide Bond Profile":  ("disulf_bilstm_profile",   "disulfide"),
        "Functional Motif Profile":("motif_bilstm_profile",    "motif"),
        "Propeptide Profile":      ("prop_bilstm_profile",     "propeptide"),
        "Repeat Region Profile":   ("rep_bilstm_profile",      "repeat"),
        "RNA Binding Profile":               ("rnabind_bilstm_profile",  "rna_binding"),
        "Nucleotide-Binding Profile":        ("nucbind_bilstm_profile",  "nucleotide_binding"),
        "Transit Peptide Profile":           ("transit_bilstm_profile",  "transit_peptide"),
        "Aggregation Propensity Profile":    ("agg_bilstm_profile",      "aggregation"),
    }

    def _rebuild_bilstm_with_uncertainty(self, title: str, show_unc: bool):
        """Recompute the named BiLSTM profile with/without MC-Dropout uncertainty.

        When show_unc is True the heavy computation runs in MCDropoutWorker
        (a QThread) to avoid blocking the main thread and causing a segfault.
        When show_unc is False the graph is redrawn immediately without a band.
        """
        if not self.analysis_data:
            return
        if title not in self._TAB_TO_KEY:
            return
        ad_key, feat = self._TAB_TO_KEY[title]
        scores = self.analysis_data.get(ad_key)
        if not scores:
            return

        if not show_unc:
            # Redraw without uncertainty band immediately on the main thread.
            self._render_bilstm_figure(title, feat, scores, uncertainty=None)
            return

        # Heavy path: run MC-Dropout in a worker thread.
        from beer.network.workers import MCDropoutWorker
        seq = self.analysis_data.get("seq", "")
        self.statusBar.showMessage(
            f"Computing MC-Dropout uncertainty for {title}…", 0)
        self._progress_dlg = QProgressDialog(
            f"MC-Dropout uncertainty for {title}…", None, 0, 0, self)
        self._progress_dlg.setWindowTitle("BEER MC-Dropout")
        self._progress_dlg.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress_dlg.setMinimumDuration(300)
        self._progress_dlg.show()

        worker = MCDropoutWorker(title, feat, seq, self._embedder)
        worker.result_ready.connect(self._on_mc_dropout_done)
        worker.error.connect(self._on_mc_dropout_error)
        worker.finished.connect(worker.deleteLater)
        self._mc_dropout_worker = worker
        worker.start()

    def _on_mc_dropout_done(self, title: str, uncertainty: list) -> None:
        if hasattr(self, "_progress_dlg") and self._progress_dlg:
            self._progress_dlg.close()
            self._progress_dlg = None
        if not self.analysis_data or title not in self._TAB_TO_KEY:
            return
        ad_key, feat = self._TAB_TO_KEY[title]
        scores = self.analysis_data.get(ad_key) or []
        self.statusBar.showMessage(f"Uncertainty ready for {title}.", 3000)
        self._render_bilstm_figure(title, feat, scores, uncertainty=uncertainty)

    def _on_mc_dropout_error(self, title: str, msg: str) -> None:
        if hasattr(self, "_progress_dlg") and self._progress_dlg:
            self._progress_dlg.close()
            self._progress_dlg = None
        self.statusBar.showMessage(
            f"MC-Dropout failed for {title}: {msg[:80]}", 6000)

    def _render_bilstm_figure(self, title: str, feat: str, scores: list,
                               uncertainty=None) -> None:
        from beer.graphs.profiles import create_bilstm_profile_figure
        lf = self.label_font_size
        tf = self.tick_font_size
        _uniprot_feats = getattr(self, "_uniprot_features", {})
        fig = create_bilstm_profile_figure(
            feat, scores, uncertainty=uncertainty,
            uniprot_regions=_uniprot_feats.get(feat) or None,
            label_font=lf, tick_font=tf,
        )
        # Apply the same heading/grid/provenance settings as _wrap() in _build_graph_generators.
        if not self.show_heading:
            fig.suptitle("")
            for ax in fig.axes:
                ax.set_title("")
        for ax in fig.axes:
            if self.show_grid:
                ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.5, color="#c8cdd8")
                ax.set_axisbelow(True)
            else:
                ax.grid(False)
        _prov = f"BEER v2.0  |  {self.sequence_name}" if self.sequence_name else "BEER v2.0"
        fig.text(0.99, 0.01, _prov, ha="right", va="bottom",
                 fontsize=7, color="#9ca3af", alpha=0.7,
                 transform=fig.transFigure)
        self._apply_roi_to_figure(fig)
        self._generated_graphs.discard(title)
        self._replace_graph(title, fig)
        self._generated_graphs.add(title)

    # ── Region-of-Interest highlight ─────────────────────────────────────────

    def _apply_roi_highlight(self):
        """Parse ROI input, store bounds, and redraw the current graph."""
        text = self._roi_input.text().strip()
        if not text:
            self._clear_roi_highlight()
            return
        import re
        m = re.match(r"(\d+)\s*[-–]\s*(\d+)", text)
        if not m:
            self.statusBar.showMessage("ROI format: start-end  (e.g. 50-120)", 3000)
            return
        self._roi_start = int(m.group(1))
        self._roi_end   = int(m.group(2))
        if self._roi_start >= self._roi_end:
            self.statusBar.showMessage("ROI start must be less than end.", 3000)
            self._roi_start = self._roi_end = None
            return
        self._refresh_current_graph_roi()
        self.statusBar.showMessage(
            f"ROI highlight: residues {self._roi_start}–{self._roi_end}", 3000)

    def _clear_roi_highlight(self):
        self._roi_start = self._roi_end = None
        self._roi_input.clear()
        self._refresh_current_graph_roi()
        self.statusBar.showMessage("ROI highlight cleared.", 2000)

    def _refresh_current_graph_roi(self):
        """Re-render the currently visible graph so ROI band is added/removed."""
        cur = self.graph_stack.currentWidget()
        if cur is None:
            return
        for canvas in cur.findChildren(FigureCanvas):
            self._apply_roi_to_figure(canvas.figure)
            canvas.draw_idle()
            break

    def apply_roi_to_figure(self, fig):
        """Public alias used by _wrap() — apply stored ROI to any new figure."""
        self._apply_roi_to_figure(fig)

    def _apply_roi_to_figure(self, fig):
        """Draw (or remove) the ROI axvspan on every positional axis in fig."""
        import numpy as np
        ROI_TAG = "_beer_roi_span"
        for ax in fig.get_axes():
            # Remove any previous ROI patches (ArtistList doesn't support slice assignment)
            for p in [p for p in ax.patches if getattr(p, ROI_TAG, False)]:
                p.remove()
            for ln in [ln for ln in ax.lines if getattr(ln, ROI_TAG, False)]:
                ln.remove()
            if self._roi_start is None:
                continue
            xlim = ax.get_xlim()
            xspan = abs(xlim[1] - xlim[0])
            # Only draw on axes whose x-range looks like residue positions (>= 10)
            if xspan < 10:
                continue
            span = ax.axvspan(
                self._roi_start - 0.5, self._roi_end + 0.5,
                alpha=0.18, color="#f59e0b", zorder=1, lw=0)
            setattr(span, ROI_TAG, True)
            # Add a thin edge line at both boundaries
            for xv in (self._roi_start - 0.5, self._roi_end + 0.5):
                ln = ax.axvline(xv, color="#d97706", lw=0.8,
                                linestyle="--", alpha=0.6, zorder=2)
                setattr(ln, ROI_TAG, True)

    # save_all_graphs removed in v2.0 — use per-graph Save Graph button

    def export_batch_csv(self):
        if not self.batch_data:
            QMessageBox.warning(self, "Export CSV",
                                "No multichain data loaded. Import a multi-FASTA or multi-chain PDB first.")
            return
        fn, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv)")
        if not fn:
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
        if not self.batch_data:
            QMessageBox.warning(self, "Export JSON",
                                "No multichain data loaded. Import a multi-FASTA or multi-chain PDB first.")
            return
        fn, _ = QFileDialog.getSaveFileName(self, "Save JSON", "", "JSON Files (*.json)")
        if not fn:
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

    def _hydrophobicity_hint(self) -> str:
        scale = getattr(self, "hydro_scale", "Kyte-Doolittle")
        meta  = HYDROPHOBICITY_SCALES.get(scale, {})
        ref   = meta.get("ref", "")
        ylabel = meta.get("ylabel", scale)
        _SCALE_NOTES = {
            "Kyte-Doolittle":
                "Range −4.5 (Arg) to +4.5 (Ile). Values > 1.8 sustained over ≥ 20 residues "
                "suggest a transmembrane helix (Kyte & Doolittle 1982).",
            "Wimley-White":
                "Whole-residue free energies of transfer from water into lipid bilayer "
                "(Wimley & White 1996). Negative = membrane-preferring.",
            "Hessa":
                "Apparent ΔG for translocon-mediated membrane insertion (Hessa et al. 2005, "
                "Nat. Methods). Negative values favour insertion.",
            "Moon-Fleming":
                "Water-to-bilayer transfer ΔG using OmpLA host-guest system "
                "(Moon & Fleming 2011). Negative = membrane-preferring.",
            "GES":
                "Goldman-Engelman-Steitz scale: free energy of transfer from lipid to water "
                "(Engelman et al. 1986). Positive = lipid-preferring.",
            "Hopp-Woods":
                "Hydrophilicity scale; positive = hydrophilic. Used for predicting antigenic "
                "surface regions (Hopp & Woods 1981).",
            "Eisenberg":
                "Consensus hydrophobicity scale normalised to unit variance "
                "(Eisenberg et al. 1984).",
            "Fauche-Pliska":
                "Octanol/water partition logP — measures membrane-partitioning tendency "
                "(Fauchère & Pliska 1983).",
            "Urry":
                "Inverse temperature transition scale for IDP phase separation; "
                "values calibrated on elastin-like peptides (Urry et al. 1992).",
        }
        note = _SCALE_NOTES.get(scale, "")
        return (
            f"Sliding-window average of residue hydrophobicity.\n\n"
            f"Formula: H(i) = (1/w) · Σ h(j)  for j = i−⌊w/2⌋ to i+⌊w/2⌋\n"
            f"where h(j) is the per-residue score and w is the window size (default 9).\n\n"
            f"Currently using: {scale}  ({ref})\n"
            f"Y-axis: {ylabel}\n\n"
            + (note + "\n\n" if note else "")
            + "Other scales selectable in Settings: Wimley–White, Hessa, GES, "
            "Hopp–Woods, Fauchère–Pliska, Urry, Moon–Fleming, Eisenberg."
        )

    # --- Settings ---

    def toggle_theme(self):
        is_dark = self.theme_toggle.isChecked()
        self._is_dark = is_dark
        if is_dark:
            self.setStyleSheet(DARK_THEME_CSS)
            plt.style.use("dark_background")
            accent = "#4cc9f0"
            struct_css = self._STRUCT_PANEL_CSS_DARK
            struct_title_color = "#4cc9f0"
        else:
            self.setStyleSheet(LIGHT_THEME_CSS)
            plt.style.use("default")
            accent = "#4361ee"
            struct_css = self._STRUCT_PANEL_CSS_LIGHT
            struct_title_color = "#3b4fc8"

        # Update structure control panel stylesheet
        if hasattr(self, "struct_ctrl_scroll"):
            self.struct_ctrl_scroll.setStyleSheet(struct_css)
            tabbar_css = (self._STRUCT_TABBAR_CSS_DARK
                          if is_dark else self._STRUCT_TABBAR_CSS_LIGHT)
            self.struct_ctrl_scroll.tabBar().setStyleSheet(tabbar_css)
        if hasattr(self, "_struct_title_lbl"):
            self._struct_title_lbl.setStyleSheet(
                f"font-weight:700; font-size:10pt; color:{struct_title_color};"
                " padding:4px 0; background:transparent;"
            )

        # Re-polish chip buttons so property-based chip/fetched styles update
        if hasattr(self, "_db_fetch_btns"):
            for _btn in self._db_fetch_btns:
                _btn.style().unpolish(_btn)
                _btn.style().polish(_btn)

        # Re-style matplotlib navigation toolbars to match the new theme
        _toolbar_light_css = (
            "QToolBar { background: #eef0f8; border: 1px solid #d0d4e0;"
            "           border-radius: 4px; padding: 2px; spacing: 1px; }"
            "QToolButton { background: #ffffff; border: 1px solid #d0d4e0;"
            "              border-radius: 3px; padding: 3px; color: #2d3748; }"
            "QToolButton:hover   { background: #e0e4f4; border-color: #4361ee; }"
            "QToolButton:pressed { background: #c8d0ec; }"
            "QToolButton:checked { background: #4361ee; border-color: #3451d1; color: #ffffff; }"
        )
        _toolbar_dark_css = (
            "QToolBar { background: #1e2640; border: none; border-radius: 4px;"
            "           padding: 2px; spacing: 1px; }"
            "QToolButton { background: transparent; border: none; border-radius: 3px;"
            "              padding: 3px; color: #e8eaef; }"
            "QToolButton:hover   { background: rgba(255,255,255,0.15); }"
            "QToolButton:pressed { background: rgba(255,255,255,0.25); }"
            "QToolButton:checked { background: rgba(67,97,238,0.55); }"
        )
        _tb_css = _toolbar_dark_css if is_dark else _toolbar_light_css
        if hasattr(self, "graph_stack"):
            for _tb in self.graph_stack.findChildren(NavigationToolbar2QT):
                _tb.setStyleSheet(_tb_css)
                if not is_dark:
                    self._tint_toolbar_icons_dark(_tb)

        # Re-render sequence viewer with updated colors
        if self.analysis_data:
            self._update_seq_viewer()

        # Re-render help tab browsers with theme-correct CSS
        report_css = get_report_css(is_dark)
        for browser, html_body in getattr(self, "_help_browsers", []):
            browser.setHtml(
                f"<style>{report_css} body{{padding:12px;}}</style>" + html_body
            )

        # Re-render report section browsers if analysis is loaded
        if self.analysis_data:
            self._refresh_report_sections()

        label = "Dark" if is_dark else "Light"
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
        # heatmap_cmap is set per-graph via the inline colormap dropdown
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
        # default_graph_format is set per-graph via the inline save format dropdown
        self.use_reducing         = self.reducing_checkbox.isChecked()
        self.hydro_scale          = self.hydro_scale_combo.currentText()

        # Sequence name override — propagate immediately everywhere
        name_override = self.seq_name_input.text().strip()
        if name_override and name_override != self.sequence_name:
            self.sequence_name = name_override
            self._propagate_name_change()

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
        self._apply_browser_palette()

        if self.analysis_data:
            for sec, browser in self.report_section_tabs.items():
                if sec in self.analysis_data["report_sections"]:
                    browser.setHtml(self.analysis_data["report_sections"][sec])
            self._update_seq_viewer()
            self.update_graph_tabs()
            self._sparkline_links_wired = False  # re-wire after theme reset
            self._append_sparklines(self.analysis_data)
            self._append_mini_graphs()

        # Persist settings to disk
        _config.save({
            "theme_dark":       self.theme_toggle.isChecked(),
            "window_size":      self.default_window_size,
            "ph":               self.default_pH,
            "use_reducing":     self.use_reducing,
            "custom_pka":       self.custom_pka,
            "colormap":         self.colormap,
            "heatmap_cmap":     self.heatmap_cmap,
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
            "enable_tooltips":      self.enable_tooltips,
            "colorblind_safe":      getattr(self, "colorblind_safe", False),
            "esm2_model":           "esm2_t33_650M_UR50D",
            "hydro_scale":          self.hydro_scale,
        })
        self.statusBar.showMessage("Settings applied and saved.", 5000)

    def reset_defaults(self):
        self.window_size_input.setText("9")
        self.hydro_scale_combo.setCurrentText("Kyte-Doolittle")
        self.ph_input.setText("7.0")
        self.pka_input.setText("")
        self.reducing_checkbox.setChecked(False)
        self.label_checkbox.setChecked(True)
        self.heatmap_cmap = "viridis"
        self.label_font_input.setText("11")
        self.tick_font_input.setText("9")
        self.marker_size_input.setText("10")
        self.graph_color_combo.setCurrentText("Royal Blue")
        self.default_graph_format = "PNG"
        self.heading_checkbox.setChecked(True)
        self.grid_checkbox.setChecked(True)
        self.transparent_bg_checkbox.setChecked(True)
        self.theme_toggle.setChecked(False)
        self.tooltips_checkbox.setChecked(True)
        self.apply_settings()

    # --- DeepTMHMM / AlphaMissense ---

    def _run_deeptmlhmm(self):
        if not self.analysis_data:
            return
        seq = self.analysis_data.get("seq", "")
        if not seq:
            return

        # Inform user about the BioLib authentication requirement before submitting.
        msg = (
            "<b>BioLib authentication required</b><br><br>"
            "DeepTMHMM runs on BioLib's cloud servers. You must be logged in "
            "before submitting a job, otherwise the result will be empty.<br><br>"
            "To authenticate, run once in a terminal:<br>"
            "<code>&nbsp;&nbsp;python -m biolib login</code><br><br>"
            "The local TMHMM&nbsp;2.0 result (already shown) will be kept if "
            "DeepTMHMM fails."
        )
        dlg = QMessageBox(self)
        dlg.setWindowTitle("DeepTMHMM — BioLib login")
        dlg.setIcon(QMessageBox.Icon.Information)
        dlg.setText(msg)
        dlg.setStandardButtons(
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
        dlg.button(QMessageBox.StandardButton.Ok).setText("I am logged in — Continue")
        if dlg.exec() != QMessageBox.StandardButton.Ok:
            return

        from beer.network.workers import DeepTMHMMWorker
        self.fetch_deeptmhmm_btn.setEnabled(False)
        self._mark_chip_loading(self.fetch_deeptmhmm_btn)
        self._deeptmlhmm_worker = DeepTMHMMWorker(seq, self)
        self._deeptmlhmm_worker.finished.connect(self._on_deeptmlhmm_done)
        self._deeptmlhmm_worker.error.connect(self._on_deeptmlhmm_error)
        self._deeptmlhmm_worker.start()

    def _on_deeptmlhmm_done(self, helices):
        self.fetch_deeptmhmm_btn.setEnabled(True)
        self._mark_chip_fetched(self.fetch_deeptmhmm_btn)
        n = len(helices)
        self.statusBar.showMessage(
            f"DeepTMHMM: {n} TM {'helix' if n == 1 else 'helices'} detected.", 5000)
        if self.analysis_data:
            self.analysis_data["tm_helices"] = helices
            self._generated_graphs.discard("TM Topology")
            self._generated_graphs.discard("Domain Architecture")
            self._generated_graphs.discard("Annotation Track")
            self._render_visible_graph()

    def _on_deeptmlhmm_error(self, msg: str):
        self.fetch_deeptmhmm_btn.setEnabled(True)
        self._mark_chip_normal(self.fetch_deeptmhmm_btn)
        QMessageBox.warning(
            self, "DeepTMHMM Error",
            f"{msg}\n\nThe local TMHMM\u00a02.0 result is preserved."
        )

    # --- SignalP 6.0 ---

    def _run_signalp6(self):
        if not self.analysis_data:
            return
        seq = self.analysis_data.get("seq", "")
        if not seq:
            return

        msg = (
            "<b>BioLib authentication required</b><br><br>"
            "SignalP&nbsp;6.0 runs on BioLib's cloud servers. You must be logged in "
            "before submitting a job.<br><br>"
            "To authenticate, run once in a terminal:<br>"
            "<code>&nbsp;&nbsp;python -m biolib login</code><br><br>"
            "The local von Heijne signal peptide result (already shown) will be "
            "kept if SignalP&nbsp;6.0 fails."
        )
        dlg = QMessageBox(self)
        dlg.setWindowTitle("SignalP 6.0 — BioLib login")
        dlg.setIcon(QMessageBox.Icon.Information)
        dlg.setText(msg)
        dlg.setStandardButtons(
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
        dlg.button(QMessageBox.StandardButton.Ok).setText("I am logged in — Continue")
        if dlg.exec() != QMessageBox.StandardButton.Ok:
            return

        from beer.network.workers import SignalP6Worker
        self.fetch_signalp6_btn.setEnabled(False)
        self._mark_chip_loading(self.fetch_signalp6_btn)
        self._signalp6_worker = SignalP6Worker(seq, parent=self)
        self._signalp6_worker.finished.connect(self._on_signalp6_done)
        self._signalp6_worker.error.connect(self._on_signalp6_error)
        self._signalp6_worker.start()

    def _on_signalp6_done(self, result: dict):
        self.fetch_signalp6_btn.setEnabled(True)
        self._mark_chip_fetched(self.fetch_signalp6_btn)
        cs = result.get("cleavage_site", -1)
        prob = result.get("probability", 0.0)
        sig_type = result.get("signal_type", "OTHER")
        if sig_type == "OTHER":
            self.statusBar.showMessage("SignalP 6.0: No signal peptide detected.", 5000)
        else:
            self.statusBar.showMessage(
                f"SignalP 6.0: {sig_type}, CS after pos {cs}, P={prob:.3f}", 6000)
        if self.analysis_data:
            self.analysis_data["signalp6"] = result
            self._generated_graphs.discard("Signal Peptide & GPI")
            self._render_visible_graph()

    def _on_signalp6_error(self, msg: str):
        self.fetch_signalp6_btn.setEnabled(True)
        self._mark_chip_normal(self.fetch_signalp6_btn)
        QMessageBox.warning(
            self, "SignalP 6.0 Error",
            f"{msg}\n\nThe local von Heijne result is preserved."
        )

    def _run_alphafold_missense(self, uniprot_id: str):
        if not uniprot_id:
            QMessageBox.warning(self, "AlphaMissense",
                                "No UniProt accession loaded.\n"
                                "Fetch a protein via the Fetch bar first.")
            return
        from beer.network.workers import AlphaMissenseWorker
        self.fetch_alphafold_missense_btn.setEnabled(False)
        self._mark_chip_loading(self.fetch_alphafold_missense_btn)
        self._am_worker = AlphaMissenseWorker(uniprot_id, self)
        self._am_worker.finished.connect(self._on_alphafold_missense_done)
        self._am_worker.error.connect(self._on_alphafold_missense_error)
        self._am_worker.start()

    def _on_alphafold_missense_done(self, data: dict):
        self._alphafold_missense_data = data
        self.fetch_alphafold_missense_btn.setEnabled(True)
        self._mark_chip_fetched(self.fetch_alphafold_missense_btn)
        # Rebuild generators so the AlphaMissense graph uses real data, not placeholder
        self.update_graph_tabs()

    def _on_alphafold_missense_error(self, msg: str):
        self.fetch_alphafold_missense_btn.setEnabled(True)
        self._mark_chip_normal(self.fetch_alphafold_missense_btn)
        QMessageBox.warning(self, "AlphaMissense Error", msg)

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
            self.export_structure_btn.setEnabled(True)
        else:
            self.export_structure_btn.setEnabled(False)

    def on_chain_selected(self, text: str):
        for cid, seq, data in self.batch_data:
            if cid == text:
                self.seq_text.setPlainText(seq)
                self.analysis_data = data
                self.sequence_name = cid
                secs = data.get("report_sections", {})
                for sec, browser in self.report_section_tabs.items():
                    if sec in secs:
                        browser.setHtml(secs[sec])
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
        QShortcut(QKeySequence("Ctrl+G"),      self,
                  lambda: self.main_tabs.setCurrentIndex(1))
        QShortcut(QKeySequence("Ctrl+2"), self,
                  lambda: self.main_tabs.setCurrentIndex(2))   # Structure
        QShortcut(QKeySequence("Ctrl+3"), self,
                  lambda: self.main_tabs.setCurrentIndex(3))   # BLAST
        QShortcut(QKeySequence("Ctrl+7"), self,
                  lambda: self.main_tabs.setCurrentIndex(7))   # MSA
        QShortcut(QKeySequence("Ctrl+Z"), self, self._undo_mutation)
        QShortcut(QKeySequence("Ctrl+S"),      self, self.session_save)
        QShortcut(QKeySequence("Ctrl+O"),      self, self.session_load)
        QShortcut(QKeySequence("Ctrl+F"),      self,
                  lambda: self.motif_input.setFocus())
        QShortcut(QKeySequence("Ctrl+/"),      self, self.show_shortcuts)
        QShortcut(QKeySequence("Ctrl+Right"), self, self._graph_nav_next)
        QShortcut(QKeySequence("Ctrl+Left"),  self, self._graph_nav_prev)

    def show_shortcuts(self):
        """Show keyboard shortcut reference overlay."""
        shortcuts = [
            ("Ctrl+Return", "Analyze sequence"),
            ("Ctrl+G",      "Switch to Graphs tab"),
            ("Ctrl+2",      "Switch to Structure tab"),
            ("Ctrl+3",      "Switch to BLAST tab"),
            ("Ctrl+7",      "Switch to MSA tab"),
            ("Ctrl+Z",      "Undo last mutation"),
            ("Ctrl+Right",  "Next graph"),
            ("Ctrl+Left",   "Previous graph"),
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
        # Update disorder-method indicator based on what was actually used
        from beer.embeddings import ESM2_AVAILABLE
        _dmethod = data.get("disorder_method", "")
        if self._embedder is not None and self._embedder.is_available():
            self._update_esm2_indicator("active")
        elif "metapredict" in _dmethod:
            self._update_esm2_indicator("metapredict")
        else:
            self._update_esm2_indicator("classical")
        if not ESM2_AVAILABLE and not self._esm2_missing_warned:
            self._esm2_missing_warned = True
            self.statusBar.showMessage(
                "ESM2 not installed \u2014 disorder uses metapredict/classical fallback. "
                "Install with: pip install fair-esm torch", 8000
            )
        seq  = data["seq"]
        self._run_plugins(seq, data)
        self.analysis_data = data
        if not getattr(self, "_restoring_snapshot", False):
            self._add_to_history(self.sequence_name, seq, data)
        # ── Populate Summary tab ──────────────────────────────────────────
        self._summary_tab_browser.setHtml(self._build_summary_tab_html(data))
        for sec, browser in self.report_section_tabs.items():
            if sec in data["report_sections"]:
                browser.setHtml(data["report_sections"][sec])
        self._update_seq_viewer()
        self.update_graph_tabs()
        self._append_sparklines(data)
        self._append_mini_graphs()
        if self._last_was_bilstm:
            # Full AI Analysis completed — populate all sections with real data.
            self._populate_ai_report_sections(data)
            self._ai_computed_sections = {
                f"AI:{name}"
                for name, dk, _, _ in _AI_HEAD_SPECS
                if data.get(dk)
            }
        else:
            # Classical analysis — populate sidebar with lazy-load placeholders.
            self._ai_computed_sections.clear()
            self._setup_ai_section_placeholders()
        # Refresh AI Features scheme combo so newly available heads appear
        if hasattr(self, "struct_color_mode_combo"):
            cur_mode = self.struct_color_mode_combo.currentText()
            if cur_mode == "AI Features":
                self._update_scheme_combo("AI Features")
        self.analyze_btn.setEnabled(True)
        self.bilstm_analyze_btn.setEnabled(True)
        # Enable all analysis-dependent buttons
        for btn in (self.mutate_btn, self.trunc_run_btn,
                    self.fetch_deeptmhmm_btn, self.fetch_signalp6_btn,
                    self.find_uniprot_btn):
            btn.setEnabled(True)
        self.fetch_uniprot_tracks_btn.setEnabled(bool(self.current_accession))
        self._graphs_uniprot_btn.setEnabled(bool(self.current_accession))
        self.trunc_run_btn.setToolTip("Run truncation series analysis")
        self.setWindowTitle(f"BEER — {self._display_name()}")
        self.statusBar.showMessage(
            f"Analysis complete  |  {len(seq)} aa  |  {self.sequence_name}", 4000
        )
        self._prepend_section_summaries()

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

    def _display_name(self) -> str:
        """Return 'Name (ID)' where ID is the UniProt accession or PDB ID, if known."""
        name = self.sequence_name or "Protein"
        src = self._source_id or self.current_accession
        return f"{name} ({src})" if src else name

    def _propagate_name_change(self) -> None:
        """Re-render every UI element that shows the protein name."""
        if not self.analysis_data:
            return
        self.setWindowTitle(f"BEER — {self._display_name()}")
        self._summary_tab_browser.setHtml(
            self._build_summary_tab_html(self.analysis_data))
        if self._history:
            self._history[0]["name"] = self.sequence_name or "Sequence"
            self._rebuild_history_combo()
        self.statusBar.showMessage(
            f"Name updated: {self._display_name()}", 3000)

    def _make_snapshot(self, data: dict) -> dict:
        """Capture the current full session state as a restorable snapshot."""
        return {
            "name":             self.sequence_name or "Sequence",
            "seq":              data.get("seq", ""),
            "analysis_data":    data,
            "accession":        self.current_accession,
            "source_id":        self._source_id,
            "last_was_bilstm":  self._last_was_bilstm,
            "alphafold_data":   self.alphafold_data,
            "pfam_domains":     list(self.pfam_domains),
            "uniprot_features": dict(self._uniprot_features),
        }

    def _add_to_history(self, name: str, seq: str, data: dict):
        snap = self._make_snapshot(data)
        snap["name"] = name or "Sequence"
        # deduplicate by sequence
        self._history = [s for s in self._history if s["seq"] != seq]
        self._history.insert(0, snap)
        self._history = self._history[:10]
        self._rebuild_history_combo()

    def _update_current_snapshot(self):
        """Update history[0] with latest structure/annotation state."""
        if not self._history or not self.analysis_data:
            return
        snap = self._history[0]
        snap["alphafold_data"]   = self.alphafold_data
        snap["pfam_domains"]     = list(self.pfam_domains)
        snap["uniprot_features"] = dict(self._uniprot_features)
        snap["accession"]        = self.current_accession
        snap["source_id"]        = self._source_id

    def _rebuild_history_combo(self):
        self.history_combo.blockSignals(True)
        self.history_combo.clear()
        self.history_combo.addItem("— recent sequences —")
        for snap in self._history:
            self.history_combo.addItem(snap["name"])
        self.history_combo.setCurrentIndex(1)
        self.history_combo.blockSignals(False)

    def _on_history_selected(self, idx: int):
        if idx <= 0:
            return
        self._restore_snapshot(self._history[idx - 1])

    def _restore_snapshot(self, snap: dict):
        """Restore a full session snapshot without running new analysis."""
        self._do_reset()

        # Restore core state
        self.sequence_name      = snap["name"]
        self.current_accession  = snap.get("accession", "")
        self._source_id         = snap.get("source_id", snap.get("accession", ""))
        self._last_was_bilstm   = snap.get("last_was_bilstm", False)
        self.alphafold_data     = snap.get("alphafold_data")
        self.pfam_domains       = list(snap.get("pfam_domains", []))
        self._uniprot_features  = dict(snap.get("uniprot_features", {}))

        data = snap.get("analysis_data")
        if data:
            self.seq_text.setPlainText(data.get("seq", ""))
            # Re-render reports and graphs from stored data
            self._restoring_snapshot = True
            try:
                self._on_worker_finished(data)
            finally:
                self._restoring_snapshot = False
        else:
            self.seq_text.setPlainText(snap.get("seq", ""))

        # Restore 3D structure
        if self.alphafold_data:
            pdb_str = self.alphafold_data.get("pdb_str", "")
            if pdb_str:
                self._load_structure_viewer(pdb_str)
            self.export_structure_btn.setEnabled(True)
            n_res = len(self.alphafold_data.get("plddt", []))
            mean_plddt = (sum(self.alphafold_data["plddt"]) / n_res
                         if n_res else 0)
            src = self.alphafold_data.get("accession", self.sequence_name)
            self.af_status_lbl.setText(
                f"Structure: {src}  ({n_res} residues, mean pLDDT = {mean_plddt:.1f})")
            self.af_status_lbl.setProperty("status_state", "success")
            self.af_status_lbl.style().unpolish(self.af_status_lbl)
            self.af_status_lbl.style().polish(self.af_status_lbl)

        # Restore chip button states
        has_acc   = bool(self.current_accession)
        has_af    = bool(self.alphafold_data)
        has_pfam  = bool(self.pfam_domains)
        has_feats = bool(self._uniprot_features)
        for btn in self._db_fetch_btns:
            btn.setEnabled(has_acc)
            btn.setProperty("chip_state", "normal")
            btn.style().unpolish(btn); btn.style().polish(btn)
        if has_af:
            self._mark_chip_fetched(self.fetch_af_btn)
        if has_pfam:
            self.fetch_pfam_btn.setEnabled(True)
            self._mark_chip_fetched(self.fetch_pfam_btn)
        if has_feats:
            self.fetch_uniprot_tracks_btn.setEnabled(True)
            self._mark_chip_fetched(self.fetch_uniprot_tracks_btn)
        self.statusBar.showMessage(
            f"Restored: {snap['name']}  ({len(data.get('seq',''))} aa)"
            if data else f"Restored: {snap['name']}", 3000)

    def closeEvent(self, event):
        """On close, wipe history from config so next session starts clean."""
        _config.set_value("recent_sequences", [])
        super().closeEvent(event)

    # --- Accession fetch ---

    def fetch_accession(self):
        acc = self.accession_input.text().strip()
        if not acc:
            QMessageBox.warning(self, "Fetch", "Enter a UniProt ID or PDB ID.")
            return
        # Reset previous protein state before loading a new one
        if self.analysis_data or self.seq_text.toPlainText().strip():
            self._do_reset()
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
            self.statusBar.showMessage(f"Downloading structure for {acc.upper()}…")
            try:
                struct_str, is_cif = self._fetch_pdb_structure(acc)
                chain_structs = (extract_chain_structures_mmcif(struct_str)
                                 if is_cif else extract_chain_structures(struct_str))
                pdb_str = struct_str  # used for 3D viewer below
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
                self._struct_is_alphafold = False
                # Load the full PDB into the 3D viewer (all chains + chain controls).
                self._load_structure_viewer(pdb_str)
                self.export_structure_btn.setEnabled(True)
                self.af_status_lbl.setText(
                    f"Loaded PDB {acc.upper()}  —  "
                    f"{len(chain_structs)} chain(s), {len(tagged)} sequence(s)"
                )
                self.af_status_lbl.setProperty("status_state", "success")
                self.af_status_lbl.style().unpolish(self.af_status_lbl)
                self.af_status_lbl.style().polish(self.af_status_lbl)
            except Exception:
                pass  # Structure fetch is best-effort; sequences are already loaded
        else:
            rid, seq = entries[0]
        self.seq_text.setPlainText(seq)
        self.sequence_name = rid
        # Store accession; AlphaFold/Pfam/ELM/DisProt/PhaSepDB need a UniProt ID
        self.current_accession = acc if not is_pdb else ""
        self._source_id        = acc   # always preserve the original fetch ID
        # Enable structure chips always; disorder/interaction chips require UniProt
        self.fetch_af_btn.setEnabled(True)
        self.fetch_pfam_btn.setEnabled(True)
        self.fetch_elm_btn.setEnabled(not is_pdb)
        self.fetch_disprot_btn.setEnabled(not is_pdb)
        self.fetch_phasepdb_btn.setEnabled(not is_pdb)
        self.fetch_mobidb_btn.setEnabled(not is_pdb)
        self.fetch_variants_btn.setEnabled(not is_pdb)
        self.fetch_intact_btn.setEnabled(not is_pdb)
        self.fetch_alphafold_missense_btn.setEnabled(not is_pdb)
        self.fetch_uniprot_tracks_btn.setEnabled(not is_pdb)
        self.accession_input.clear()
        src = "PDB" if is_pdb else "UniProt"
        msg = f"Fetched {rid} from {src}  ({len(seq)} aa)"
        if is_pdb and len(entries) > 1:
            msg += f"  \u2014 {len(entries)} chains loaded"
        self.statusBar.showMessage(msg, 4000)

        # Sequence is now in the text box — user presses Analyze when ready.

        # Fetch and display protein summary (best-effort, non-blocking)
        self._fetch_and_show_protein_summary(acc, is_pdb)

    def _fetch_and_show_protein_summary(self, acc: str, is_pdb: bool) -> None:
        """Fetch metadata from UniProt or RCSB, cache in _uniprot_card, refresh Summary tab."""
        try:
            if is_pdb:
                url = f"https://data.rcsb.org/rest/v1/core/entry/{acc.upper()}"
                req = urllib.request.Request(url, headers={
                    "Accept": "application/json", "User-Agent": "BEER/2.0"})
                with urllib.request.urlopen(req, timeout=8) as resp:
                    data = json.loads(resp.read().decode())
                self._uniprot_card = {
                    "source": "PDB",
                    "accession": acc.upper(),
                    "name": data.get("struct", {}).get("title", ""),
                }
            else:
                url = f"https://rest.uniprot.org/uniprotkb/{acc}.json"
                req = urllib.request.Request(url, headers={
                    "Accept": "application/json", "User-Agent": "BEER/2.0"})
                with urllib.request.urlopen(req, timeout=8) as resp:
                    data = json.loads(resp.read().decode())
                # ── Protein name ──────────────────────────────────────────
                pd_obj = data.get("proteinDescription", {})
                rec = pd_obj.get("recommendedName") or (pd_obj.get("submittedNames") or [{}])[0]
                prot_name = (rec.get("fullName") or {}).get("value", "")
                # ── Gene / organism ───────────────────────────────────────
                genes = data.get("genes", [])
                gene = (genes[0].get("geneName") or {}).get("value", "") if genes else ""
                organism = data.get("organism", {}).get("scientificName", "")
                # ── Parse comments by type ────────────────────────────────
                func_texts, subcel_texts, disease_texts, ptm_texts, caution_texts = [], [], [], [], []
                for c in data.get("comments", []):
                    ct = c.get("commentType", "")
                    if ct == "FUNCTION":
                        for t in c.get("texts", []):
                            v = t.get("value", "").strip()
                            if v:
                                func_texts.append(v)
                    elif ct == "SUBCELLULAR LOCATION":
                        for loc in c.get("subcellularLocations", []):
                            lv = (loc.get("location") or {}).get("value", "")
                            if lv:
                                subcel_texts.append(lv)
                    elif ct == "DISEASE":
                        d = c.get("disease", {})
                        dname = d.get("diseaseId", "") or d.get("description", "")
                        if dname:
                            disease_texts.append(dname)
                    elif ct == "PTM":
                        for t in c.get("texts", []):
                            v = t.get("value", "").strip()
                            if v:
                                ptm_texts.append(v)
                    elif ct == "CAUTION":
                        for t in c.get("texts", []):
                            v = t.get("value", "").strip()
                            if v:
                                caution_texts.append(v)
                # ── Keywords ──────────────────────────────────────────────
                keywords = [kw.get("name", "") for kw in data.get("keywords", []) if kw.get("name")]
                self._uniprot_card = {
                    "source": "UniProt",
                    "accession": acc,
                    "name": prot_name,
                    "gene": gene,
                    "organism": organism,
                    "function": func_texts,
                    "subcellular": list(dict.fromkeys(subcel_texts)),
                    "diseases": disease_texts,
                    "ptm": ptm_texts,
                    "keywords": keywords,
                }
            # Refresh summary tab with new card data if analysis is done
            if self.analysis_data:
                self._summary_tab_browser.setHtml(
                    self._build_summary_tab_html(self.analysis_data))
            # ── PDB cross-reference chips (UniProt only) ──────────────────
            if not is_pdb:
                self._populate_pdb_xref_chips(acc)
        except Exception:
            pass  # summary is informational only

    def _populate_pdb_xref_chips(self, uniprot_id: str) -> None:
        """Fetch PDB xrefs for *uniprot_id* and show as clickable chips in a grid."""
        while self._pdb_xref_layout.count():
            item = self._pdb_xref_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                while item.layout().count():
                    sub = item.layout().takeAt(0)
                    if sub.widget():
                        sub.widget().deleteLater()

        xrefs = fetch_uniprot_pdb_xrefs(uniprot_id)
        if not xrefs:
            self._pdb_xref_inner.hide()
            return

        refs_capped = xrefs[:40]
        header_row = QHBoxLayout()
        header_row.setSpacing(4)
        header_row.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(f"PDB structures ({len(refs_capped)}):")
        lbl.setStyleSheet("font-size:9pt; color:#4a5568; font-weight:600;")
        header_row.addWidget(lbl)
        header_row.addStretch()
        self._pdb_xref_layout.addLayout(header_row)

        # Grid layout: all columns equal-stretch so every chip shares the same width.
        _COLS = 8
        from PySide6.QtWidgets import QGridLayout as _QGL
        grid = _QGL()
        grid.setSpacing(4)
        grid.setContentsMargins(0, 0, 0, 0)
        for c in range(_COLS):
            grid.setColumnStretch(c, 1)

        for i, ref in enumerate(refs_capped):
            r, c = divmod(i, _COLS)
            pdb_id = ref["id"]
            method = ref.get("method", "")
            res    = ref.get("resolution", "")
            tip    = f"{pdb_id}  |  {method}"
            if res and res != "-":
                tip += f"  {res}"
            chains = ref.get("chains", "")
            if chains:
                tip += f"\nChains: {chains}"
            btn = QPushButton(pdb_id)
            btn.setToolTip(tip)
            btn.setFixedHeight(24)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn.setObjectName("chip_btn")
            btn.clicked.connect(lambda _=False, pid=pdb_id: self._load_pdb_from_chip(pid))
            grid.addWidget(btn, r, c)

        self._pdb_xref_layout.addLayout(grid)

        self._pdb_xref_inner.show()

    def _load_pdb_from_chip(self, pdb_id: str) -> None:
        """Load a PDB ID from a cross-reference chip (respects Bio. Assembly checkbox)."""
        self.accession_input.setText(pdb_id)
        self.fetch_accession()

    def _fetch_pdb_fasta(self, pdb_id: str) -> str:
        """Fetch FASTA sequence(s) from RCSB PDB for a given 4-char PDB ID."""
        url = f"https://www.rcsb.org/fasta/entry/{pdb_id.upper()}"
        req = urllib.request.Request(url, headers={"User-Agent": "BEER/2.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode()

    def _fetch_pdb_structure(self, pdb_id: str) -> tuple[str, bool]:
        """Download coordinate file from RCSB.

        Returns (structure_string, is_cif).  When the Bio. Assembly checkbox
        is checked, fetches the assembly1 mmCIF; otherwise the asymmetric-unit PDB.
        Falls back to asymmetric-unit PDB if the assembly CIF fetch fails.
        """
        pdb_id = pdb_id.upper()
        if getattr(self, "bio_assembly_chk", None) and self.bio_assembly_chk.isChecked():
            try:
                cif_str = fetch_rcsb_assembly_cif(pdb_id, assembly=1)
                return cif_str, True
            except Exception:
                self.statusBar.showMessage(
                    f"Assembly CIF not found for {pdb_id}; falling back to asymmetric unit.", 4000)
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        req = urllib.request.Request(url, headers={"User-Agent": "BEER/2.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode(), False

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
        self._mark_chip_loading(self.fetch_af_btn)
        self._alphafold_worker = AlphaFoldWorker(acc)
        self._alphafold_worker.progress.connect(
            lambda msg: self.statusBar.showMessage(msg))
        self._alphafold_worker.finished.connect(self._on_alphafold_finished)
        self._alphafold_worker.error.connect(self._on_alphafold_error)
        self._alphafold_worker.start()

    def _on_alphafold_finished(self, data: dict):
        self.alphafold_data = data
        self._struct_is_alphafold = True
        if self.sequence_name:
            self.batch_struct[self.sequence_name] = data
        self._update_current_snapshot()
        self.fetch_af_btn.setEnabled(True)
        self._mark_chip_fetched(self.fetch_af_btn)
        self.export_structure_btn.setEnabled(True)
        n_res = len(data.get("plddt", []))
        mean_plddt = (sum(data["plddt"]) / n_res) if n_res else 0
        self.af_status_lbl.setText(
            f"Loaded AlphaFold structure for {data['accession']}  "
            f"({n_res} residues, mean pLDDT = {mean_plddt:.1f})"
        )
        self.af_status_lbl.setProperty("status_state", "success")
        self.af_status_lbl.style().unpolish(self.af_status_lbl)
        self.af_status_lbl.style().polish(self.af_status_lbl)
        self._load_structure_viewer(data["pdb_str"])
        if self.analysis_data:
            self.update_graph_tabs()
        self.statusBar.showMessage(
            f"AlphaFold structure loaded  ({data['accession']})", 4000)

    def _on_alphafold_error(self, msg: str):
        self.fetch_af_btn.setEnabled(True)
        self._mark_chip_normal(self.fetch_af_btn)
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
        self._mark_chip_loading(self.fetch_pfam_btn)
        self.statusBar.showMessage(f"Fetching Pfam domains for {acc}…")
        self._pfam_worker = PfamWorker(acc)
        self._pfam_worker.finished.connect(self._on_pfam_finished)
        self._pfam_worker.error.connect(self._on_pfam_error)
        self._pfam_worker.start()

    def _on_pfam_finished(self, domains: list):
        self.pfam_domains = domains
        self._update_current_snapshot()
        self.fetch_pfam_btn.setEnabled(True)
        if not domains:
            self._mark_chip_normal(self.fetch_pfam_btn)
            self.statusBar.showMessage("No Pfam domains found for this protein.", 4000)
            return
        self._mark_chip_fetched(self.fetch_pfam_btn)
        if self.analysis_data:
            self.update_graph_tabs()
        self.statusBar.showMessage(
            f"Loaded {len(domains)} Pfam domain(s).", 4000)

    def _on_pfam_error(self, msg: str):
        self.fetch_pfam_btn.setEnabled(True)
        self._mark_chip_normal(self.fetch_pfam_btn)
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
        self.blast_stop_btn.setVisible(True)
        self.blast_table.setRowCount(0)
        self._blast_worker = BlastWorker(seq, database=db, hitlist_size=n)
        self._blast_worker.progress.connect(
            lambda msg: self.blast_status_lbl.setText(msg))
        self._blast_worker.finished.connect(self._on_blast_finished)
        self._blast_worker.error.connect(self._on_blast_error)
        import time as _time
        self._blast_start_time = _time.time()
        from PySide6.QtCore import QTimer
        self._blast_timer = QTimer(self)
        self._blast_timer.timeout.connect(self._update_blast_elapsed)
        self._blast_timer.start(1000)
        self._blast_worker.start()

    def _stop_blast(self):
        """Cancel the running BLAST search."""
        if self._blast_worker and self._blast_worker.isRunning():
            self._blast_worker.cancel()
            if not self._blast_worker.wait(2000):
                self._blast_worker.terminate()
                self._blast_worker.wait()
        if self._blast_timer:
            self._blast_timer.stop()
            self._blast_timer = None
        self._blast_start_time = None
        self.blast_stop_btn.setVisible(False)
        self.blast_run_btn.setEnabled(True)
        self.blast_status_lbl.setObjectName("status_lbl")
        self.blast_status_lbl.setProperty("status_state", "idle")
        self.blast_status_lbl.setText("BLAST stopped.")

    def _on_blast_finished(self, hits: list):
        if self._blast_timer:
            self._blast_timer.stop()
            self._blast_timer = None
        self._blast_start_time = None
        self.blast_stop_btn.setVisible(False)
        self.blast_run_btn.setEnabled(True)
        self.blast_status_lbl.setProperty("status_state", "success")
        self.blast_status_lbl.style().unpolish(self.blast_status_lbl)
        self.blast_status_lbl.style().polish(self.blast_status_lbl)
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
            load_btn.setToolTip("Load this sequence into the Analysis tab and run analysis")
            load_btn.clicked.connect(
                lambda _, h=hit: self._load_blast_hit(h))
            self.blast_table.setCellWidget(row, 6, load_btn)
        self.blast_table.resizeColumnsToContents()
        self.statusBar.showMessage(f"BLAST complete — {len(hits)} hits", 4000)

    def _on_blast_error(self, msg: str):
        if self._blast_timer:
            self._blast_timer.stop()
            self._blast_timer = None
        self._blast_start_time = None
        self.blast_stop_btn.setVisible(False)
        self.blast_run_btn.setEnabled(True)
        self.blast_status_lbl.setProperty("status_state", "error")
        self.blast_status_lbl.style().unpolish(self.blast_status_lbl)
        self.blast_status_lbl.style().polish(self.blast_status_lbl)
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
        self._undo_seq  = seq
        self._undo_name = self.sequence_name
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
        count = len(matches)
        self.motif_match_lbl.setText(f"{count} match{'es' if count != 1 else ''}")
        self.statusBar.showMessage(
            f"{count} match(es) found for '{pattern}'", 3000
        )

    def clear_motif_highlight(self):
        self.motif_input.clear()
        self.motif_match_lbl.setText("")
        self._update_seq_viewer()

    # ── Copy Sequence ──────────────────────────────────────────────────────
    def _copy_sequence_menu(self):
        """Show popup menu: copy whole sequence or a user-defined range."""
        seq = self.seq_text.toPlainText().strip()
        # Strip FASTA header lines if present
        lines = [l for l in seq.splitlines() if not l.startswith(">")]
        seq = "".join(lines).replace(" ", "").upper()
        if not seq:
            QMessageBox.information(self, "Copy Sequence", "No sequence loaded.")
            return

        menu = QMenu(self)
        act_full  = menu.addAction(f"Copy whole sequence  ({len(seq)} aa)")
        act_range = menu.addAction("Copy range…")
        action = menu.exec(self.sender().mapToGlobal(
            self.sender().rect().bottomLeft()))
        if action == act_full:
            QApplication.clipboard().setText(seq)
            self.statusBar.showMessage("Whole sequence copied to clipboard.", 2500)
        elif action == act_range:
            start, ok1 = QInputDialog.getInt(
                self, "Copy Range", f"Start residue (1–{len(seq)}):",
                1, 1, len(seq))
            if not ok1:
                return
            end, ok2 = QInputDialog.getInt(
                self, "Copy Range", f"End residue ({start}–{len(seq)}):",
                min(start + 9, len(seq)), start, len(seq))
            if not ok2:
                return
            subseq = seq[start - 1:end]
            QApplication.clipboard().setText(subseq)
            self.statusBar.showMessage(
                f"Residues {start}–{end} ({len(subseq)} aa) copied to clipboard.", 2500)

    # ── Clear All ──────────────────────────────────────────────────────────
    def _do_reset(self):
        """Perform a full UI reset without asking for confirmation."""
        self._protein_info_bar.hide()
        self._seq_info_label.setText("")
        self._seq_info_label.hide()
        self._undo_seq  = None
        self._undo_name = None

        self.seq_text.clear()
        self.seq_viewer.clear()
        self.sequence_name = ""
        self.analysis_data = None
        self.batch_data.clear()
        self.current_accession = ""
        self._source_id = ""
        self.alphafold_data = None
        self.pfam_domains = []
        self.motif_input.clear()
        self.motif_match_lbl.setText("")

        self.chain_combo.blockSignals(True)
        self.chain_combo.clear()
        self.chain_combo.setEnabled(False)
        self.chain_combo.blockSignals(False)
        self._chain_row_widget.hide()

        while self._pdb_xref_layout.count():
            _item = self._pdb_xref_layout.takeAt(0)
            if _item.widget():
                _item.widget().deleteLater()
            elif _item.layout():
                while _item.layout().count():
                    _sub = _item.layout().takeAt(0)
                    if _sub.widget():
                        _sub.widget().deleteLater()
        self._pdb_xref_inner.hide()

        for browser in self.report_section_tabs.values():
            browser.clear()

        for key in list(self._ai_pred_section_keys):
            browser = self.report_section_tabs.pop(key, None)
            self._report_sec_to_idx.pop(key, None)
            if browser is not None:
                parent_widget = browser.parent()
                if parent_widget is not None:
                    self.report_stack.removeWidget(parent_widget)
                    parent_widget.deleteLater()
        self._ai_pred_section_keys.clear()
        while self._ai_pred_grp_item.childCount() > 0:
            self._ai_pred_grp_item.removeChild(self._ai_pred_grp_item.child(0))
        self._ai_pred_grp_item.setHidden(True)
        self._last_was_bilstm = False
        self._ai_computed_sections.clear()
        if self._active_ai_worker and self._active_ai_worker.isRunning():
            self._active_ai_worker.terminate()
            self._active_ai_worker = None

        for _tab, vb in self.graph_tabs.values():
            self._clear_layout(vb)

        if self.structure_viewer is not None:
            self._js("loadPDB(null);")

        self._msa_sequences = []
        self._msa_names     = []
        self._msa_mi_apc    = None

        self.elm_data          = []
        self.disprot_data      = {}
        self.phasepdb_data     = {}
        self.mobidb_data       = {}
        self.variants_data     = []
        self.intact_data       = {}
        self._uniprot_features = {}
        self._uniprot_card     = {}

        for _w in (self._uniprot_feat_worker, self._seq_search_worker):
            if _w and _w.isRunning():
                _w.terminate()
        self._uniprot_feat_worker = None
        self._seq_search_worker   = None

        for btn in (self.mutate_btn,
                    self.export_structure_btn, self.find_uniprot_btn,
                    self.trunc_run_btn, self.fetch_signalp6_btn,
                    self.bilstm_analyze_btn, self._graphs_uniprot_btn):
            btn.setEnabled(False)
        chip_buttons = self._db_fetch_btns + [
            self.fetch_uniprot_tracks_btn, self.fetch_deeptmhmm_btn,
            self.fetch_signalp6_btn,
        ]
        seen = set()
        for btn in chip_buttons:
            if id(btn) in seen:
                continue
            seen.add(id(btn))
            btn.setEnabled(False)
            btn.setProperty("chip_state", "normal")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

        self.setWindowTitle("BEER")
        self._update_esm2_indicator("ready")

    def _clear_all(self):
        """Reset the entire session: sequence, analysis, graphs, structure."""
        reply = QMessageBox.question(
            self, "Clear All",
            "This will clear the loaded protein, all analysis results, graphs and structure.\n"
            "Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._do_reset()
        self.accession_input.clear()
        self.statusBar.showMessage("Session cleared.", 2500)

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
            agg_a = da.get("aggr_profile", [])
            agg_b = db.get("aggr_profile", [])

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

    def _export_section(self, sec: str):
        browser = self.report_section_tabs.get(sec)
        if not browser:
            return
        text = browser.toPlainText()
        if not text.strip():
            QMessageBox.information(self, "Export Section", "Run analysis first.")
            return
        fn, _ = QFileDialog.getSaveFileName(
            self, f"Export {sec}", f"{sec.replace(' ', '_')}.txt",
            "Text Files (*.txt);;CSV Files (*.csv)"
        )
        if not fn:
            return
        with open(fn, "w", encoding="utf-8") as fh:
            fh.write(text)
        self.statusBar.showMessage(f"'{sec}' exported to {fn}", 3000)

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
        self.app_font_size    = state.get("app_font_size", 12)
        self.label_font_size  = state.get("label_font_size", 11)
        self.tick_font_size   = state.get("tick_font_size", 9)
        # Update settings UI widgets
        self.ph_input.setText(str(self.default_pH))
        self.window_size_input.setText(str(self.default_window_size))
        self.transparent_bg_checkbox.setChecked(self.transparent_bg)
        self.label_font_input.setText(str(self.label_font_size))
        self.tick_font_input.setText(str(self.tick_font_size))
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
        self.trunc_run_btn = QPushButton("Run Truncation Series")
        self.trunc_run_btn.setMinimumHeight(30)
        self.trunc_run_btn.setEnabled(False)
        self.trunc_run_btn.setToolTip("Run analysis on the Analysis tab first")
        self.trunc_run_btn.clicked.connect(self.run_truncation_series)
        ctrl.addWidget(self.trunc_run_btn)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self.trunc_status_lbl = QLabel("Run analysis first, then click 'Run Truncation Series'.")
        self.trunc_status_lbl.setObjectName("status_lbl")
        self.trunc_status_lbl.setProperty("status_state", "idle")
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
        self.msa_run_btn = QPushButton("Run MSA Analysis")
        self.msa_run_btn.setMinimumHeight(30)
        self.msa_run_btn.clicked.connect(self.run_msa)
        run_msa_btn = self.msa_run_btn
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
        self.msa_viewer.append(
            "<hr><p style='color:#4361ee;font-weight:bold;'>Conservation graph \u2192 Graphs tab: "
            "Evolutionary &amp; Comparative \u2192 MSA Conservation</p>")
        # Conservation graph
        if _HAS_NEW_GRAPHS:
            fig = create_msa_conservation_figure(
                aligned, names,
                label_font=self.label_font_size, tick_font=self.tick_font_size)
            self._replace_graph("MSA Conservation", fig)
        # Covariance graph (MI with APC; capped at 500 columns)
        n_cols = len(aligned[0]) if aligned else 0
        if n_cols > 500:
            self.msa_viewer.append(
                "<p style='color:#e06c00'><b>Covariance:</b> alignment has "
                f"{n_cols} columns \u2014 exceeds 500-column limit; skipped.</p>")
            self._msa_mi_apc = None
        elif len(aligned) < 4:
            self._msa_mi_apc = None
            self.msa_viewer.append(
                "<p style='color:#e06c00;'>Covariance: need \u22654 sequences for MI/APC computation.</p>")
        else:
            from beer.analysis.msa_covariance import calc_msa_mutual_information
            _, mi_apc = calc_msa_mutual_information(aligned)
            self._msa_mi_apc = mi_apc
            cov_fig = create_msa_covariance_figure(
                mi_apc,
                label_font=self.label_font_size, tick_font=self.tick_font_size)
            self._replace_graph("MSA Covariance", cov_fig)
            self.msa_viewer.append(
                "<p style='color:#4361ee;font-weight:bold;'>Covariance heatmap \u2192 Graphs tab: "
                "Evolutionary &amp; Comparative \u2192 MSA Covariance</p>")
        self.statusBar.showMessage(
            f"MSA: {len(aligned)} sequences, {n_cols} alignment columns", 3000)

    def _clear_msa(self):
        self._msa_sequences = []
        self._msa_names     = []
        self._msa_mi_apc    = None
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
        self._mark_chip_loading(self.fetch_elm_btn)
        self.statusBar.showMessage(f"Fetching ELM instances for {acc}…")
        seq = self.analysis_data["seq"] if self.analysis_data else ""
        self._elm_worker = ELMWorker(acc, seq)
        self._elm_worker.finished.connect(self._on_elm_finished)
        self._elm_worker.error.connect(self._on_elm_error)
        self._elm_worker.start()

    def _on_elm_finished(self, instances: list):
        self.elm_data = instances
        self.fetch_elm_btn.setEnabled(True)
        if instances:
            self._mark_chip_fetched(self.fetch_elm_btn)
        else:
            self._mark_chip_normal(self.fetch_elm_btn)
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
        self._mark_chip_normal(self.fetch_elm_btn)
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
        self._mark_chip_loading(self.fetch_disprot_btn)
        self.statusBar.showMessage(f"Fetching DisProt annotations for {acc}…")
        self._disprot_worker = DisPRotWorker(acc)
        self._disprot_worker.finished.connect(self._on_disprot_finished)
        self._disprot_worker.error.connect(self._on_disprot_error)
        self._disprot_worker.start()

    def _on_disprot_finished(self, data: dict):
        self.disprot_data = data
        self.fetch_disprot_btn.setEnabled(True)
        regions = data.get("regions", [])
        if regions:
            self._mark_chip_fetched(self.fetch_disprot_btn)
        else:
            self._mark_chip_normal(self.fetch_disprot_btn)
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
        self._mark_chip_normal(self.fetch_disprot_btn)
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
        self._mark_chip_loading(self.fetch_phasepdb_btn)
        self.statusBar.showMessage(f"Checking PhaSepDB for {acc}…")
        self._phasepdb_worker = PhaSepDBWorker(acc)
        self._phasepdb_worker.finished.connect(self._on_phasepdb_finished)
        self._phasepdb_worker.error.connect(self._on_phasepdb_error)
        self._phasepdb_worker.start()

    def _on_phasepdb_finished(self, data: dict):
        self.phasepdb_data = data
        self.fetch_phasepdb_btn.setEnabled(True)
        if data.get("found"):
            self._mark_chip_fetched(self.fetch_phasepdb_btn)
        else:
            self._mark_chip_normal(self.fetch_phasepdb_btn)
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
        self._mark_chip_normal(self.fetch_phasepdb_btn)
        QMessageBox.warning(self, "PhaSepDB Error", msg)

    # ── MobiDB ────────────────────────────────────────────────────────────────

    def fetch_mobidb(self):
        acc = self.current_accession
        if not acc:
            QMessageBox.warning(self, "MobiDB", "Fetch a UniProt accession first.")
            return
        if self._mobidb_worker and self._mobidb_worker.isRunning():
            return
        self.fetch_mobidb_btn.setEnabled(False)
        self._mark_chip_loading(self.fetch_mobidb_btn)
        self.statusBar.showMessage(f"Fetching MobiDB annotations for {acc}…")
        self._mobidb_worker = MobiDBWorker(acc)
        self._mobidb_worker.finished.connect(self._on_mobidb_finished)
        self._mobidb_worker.error.connect(self._on_mobidb_error)
        self._mobidb_worker.start()

    def _on_mobidb_finished(self, data: dict):
        self.mobidb_data = data
        self.fetch_mobidb_btn.setEnabled(True)
        if data.get("found"):
            self._mark_chip_fetched(self.fetch_mobidb_btn)
        else:
            self._mark_chip_normal(self.fetch_mobidb_btn)
        if not data.get("found"):
            QMessageBox.information(self, "MobiDB",
                "This protein was not found in MobiDB, or has no consensus disorder annotations.")
            self.statusBar.showMessage("MobiDB: not found.", 3000)
            return
        frac = data.get("fraction_disorder", 0.0)
        n_pred = data.get("n_predictors", 0)
        regions = data.get("disorder_regions", [])
        lines = [
            f"<h2>MobiDB: {data.get('accession', '')}</h2>",
            f"<p><b>Consensus disorder fraction:</b> {frac:.1%} "
            f"({n_pred} predictor(s))</p>",
        ]
        if regions:
            lines.append("<h3>Disordered regions</h3><table border='1' cellspacing='0' cellpadding='4'>")
            lines.append("<tr><th>Start</th><th>End</th><th>Length</th></tr>")
            for r in regions:
                s, e = r.get("start", "?"), r.get("end", "?")
                ln = (e - s + 1) if isinstance(s, int) and isinstance(e, int) else "?"
                lines.append(f"<tr><td>{s}</td><td>{e}</td><td>{ln}</td></tr>")
            lines.append("</table>")
        html = "".join(lines)
        dlg = QDialog(self); dlg.setWindowTitle("MobiDB Disorder Consensus")
        dlg.resize(480, 320)
        bw = QTextBrowser(dlg); bw.setHtml(html)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, dlg)
        bb.rejected.connect(dlg.accept)
        lay = QVBoxLayout(dlg); lay.addWidget(bw); lay.addWidget(bb)
        dlg.exec()
        self.statusBar.showMessage(
            f"MobiDB: {frac:.1%} disordered, {len(regions)} region(s).", 4000)

    def _on_mobidb_error(self, msg: str):
        self.fetch_mobidb_btn.setEnabled(True)
        self._mark_chip_normal(self.fetch_mobidb_btn)
        self.statusBar.showMessage("MobiDB fetch failed.", 2000)
        QMessageBox.warning(self, "MobiDB Error", msg)

    # ── UniProt Feature Tracks (dual-track visualization) ─────────────────────

    def fetch_uniprot_features(self):
        acc = self.current_accession
        if not acc:
            QMessageBox.warning(self, "UniProt Tracks",
                "Fetch a UniProt accession first (or use 'Find UniProt ID').")
            return
        if self._uniprot_feat_worker and self._uniprot_feat_worker.isRunning():
            return
        self.fetch_uniprot_tracks_btn.setEnabled(False)
        self._mark_chip_loading(self.fetch_uniprot_tracks_btn)
        self.statusBar.showMessage(f"Fetching UniProt feature annotations for {acc}…")
        self._uniprot_feat_worker = UniProtFeaturesWorker(acc, parent=self)
        self._uniprot_feat_worker.finished.connect(self._on_uniprot_features_finished)
        self._uniprot_feat_worker.error.connect(self._on_uniprot_features_error)
        self._uniprot_feat_worker.start()

    def _on_uniprot_features_finished(self, data: dict):
        self._uniprot_features = data
        self._update_current_snapshot()
        self.fetch_uniprot_tracks_btn.setEnabled(True)
        n = sum(len(v) for v in data.values())
        # Map UniProt feature keys → graph tab names for the status message
        _feat_to_tab = {
            "signal_peptide": "Signal Peptide Profile", "transmembrane": "Transmembrane Profile",
            "intramembrane": "Intramembrane Profile",   "coiled_coil": "Coiled-Coil Profile",
            "dna_binding": "DNA-Binding Profile",       "rna_binding": "RNA Binding Profile",
            "active_site": "Active Site Profile",       "binding_site": "Binding Site Profile",
            "zinc_finger": "Zinc Finger Profile",       "disulfide": "Disulfide Bond Profile",
            "glycosylation": "Glycosylation Profile",   "phosphorylation": "Phosphorylation Profile",
            "propeptide": "Propeptide Profile",         "repeat": "Repeat Region Profile",
            "motif": "Functional Motif Profile",        "transit_peptide": "Transit Peptide Profile",
            "lipidation": "Lipidation Profile",
        }
        if n:
            self._mark_chip_fetched(self.fetch_uniprot_tracks_btn)
            tabs_with_overlay = [_feat_to_tab[k] for k in data if k in _feat_to_tab]
            if tabs_with_overlay:
                msg = ("UniProt overlay available on: "
                       + ", ".join(tabs_with_overlay[:5])
                       + ("…" if len(tabs_with_overlay) > 5 else "")
                       + " — navigate to those profile graphs to see annotations.")
            else:
                msg = f"UniProt features loaded ({n} annotation(s)) — navigate to profile graphs."
            self.statusBar.showMessage(msg, 8000)
        else:
            self._mark_chip_normal(self.fetch_uniprot_tracks_btn)
            self.statusBar.showMessage("UniProt features: no annotations found.", 3000)
        # Always rebuild graph generators with the new UniProt data so overlays appear
        if self.analysis_data:
            self.update_graph_tabs()

    def _on_uniprot_features_error(self, msg: str):
        self.fetch_uniprot_tracks_btn.setEnabled(True)
        self._mark_chip_normal(self.fetch_uniprot_tracks_btn)
        self.statusBar.showMessage("UniProt features fetch failed.", 2000)

    # ── Sequence → UniProt ID lookup ──────────────────────────────────────────

    def find_uniprot_from_sequence(self):
        """Search UniProt for the current sequence and populate current_accession."""
        if not self.analysis_data:
            return
        seq = self.analysis_data.get("seq", "")
        if not seq:
            return
        if self._seq_search_worker and self._seq_search_worker.isRunning():
            return
        self.find_uniprot_btn.setEnabled(False)
        self.statusBar.showMessage("Searching UniProt for sequence…")
        self._seq_search_worker = UniProtSequenceSearchWorker(
            seq, name_hint=self.sequence_name or "", parent=self)
        self._seq_search_worker.finished.connect(self._on_seq_search_finished)
        self._seq_search_worker.error.connect(self._on_seq_search_error)
        self._seq_search_worker.progress.connect(
            lambda msg: self.statusBar.showMessage(msg))
        self._seq_search_worker.start()

    def _on_seq_search_finished(self, acc: str):
        self.find_uniprot_btn.setEnabled(True)
        if not acc:
            self.statusBar.showMessage("No UniProt match found.", 4000)
            QMessageBox.information(
                self, "UniProt Search — Not Found",
                "No UniProt Swiss-Prot entry could be matched to this sequence.\n\n"
                "This can happen if the sequence:\n"
                "  • is not in UniProt Swiss-Prot (e.g. TrEMBL-only or novel)\n"
                "  • contains non-canonical residues or modifications\n"
                "  • is a fragment of a longer canonical entry\n\n"
                "You can enter the accession manually in the Fetch field."
            )
            return
        # Delegate to fetch_accession — populates name, info bar, all chip buttons
        self.accession_input.setText(acc)
        self.fetch_accession()

    def _on_seq_search_error(self, msg: str):
        self.find_uniprot_btn.setEnabled(True)
        self.statusBar.showMessage("UniProt sequence search failed.", 3000)
        QMessageBox.warning(self, "UniProt Search Error",
                            f"Sequence search encountered an error:\n\n{msg}")

    # ── UniProt Variants ───────────────────────────────────────────────────────

    def fetch_variants(self):
        acc = self.current_accession
        if not acc:
            QMessageBox.warning(self, "Variants", "Fetch a UniProt accession first.")
            return
        if self._variants_worker and self._variants_worker.isRunning():
            return
        self.fetch_variants_btn.setEnabled(False)
        self._mark_chip_loading(self.fetch_variants_btn)
        self.statusBar.showMessage(f"Fetching UniProt variants for {acc}…")
        self._variants_worker = UniProtVariantsWorker(acc)
        self._variants_worker.finished.connect(self._on_variants_finished)
        self._variants_worker.error.connect(self._on_variants_error)
        self._variants_worker.start()

    def _on_variants_finished(self, variants: list):
        self.variants_data = variants
        self.fetch_variants_btn.setEnabled(True)
        if variants:
            self._mark_chip_fetched(self.fetch_variants_btn)
        else:
            self._mark_chip_normal(self.fetch_variants_btn)
        if not variants:
            QMessageBox.information(self, "UniProt Variants",
                "No natural variants or mutagenesis data found for this protein.")
            self.statusBar.showMessage("Variants: none found.", 3000)
            return
        lines = [
            "<h2>UniProt Variants</h2>",
            "<table border='1' cellspacing='0' cellpadding='4'>",
            "<tr><th>Pos</th><th>From</th><th>To</th><th>Type</th><th>Description</th></tr>",
        ]
        for v in variants:
            desc = v.get("description", "")[:80]
            vtype = v.get("type", "")
            lines.append(
                f"<tr><td>{v.get('position','?')}</td>"
                f"<td>{v.get('original','?')}</td>"
                f"<td>{v.get('variant','?')}</td>"
                f"<td>{vtype}</td>"
                f"<td>{desc}</td></tr>"
            )
        lines.append("</table>")
        html = "".join(lines)
        dlg = QDialog(self); dlg.setWindowTitle("UniProt Variants")
        dlg.resize(700, 480)
        bw = QTextBrowser(dlg); bw.setHtml(html)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, dlg)
        bb.rejected.connect(dlg.accept)
        lay = QVBoxLayout(dlg); lay.addWidget(bw); lay.addWidget(bb)
        dlg.exec()
        self.statusBar.showMessage(f"Variants: {len(variants)} annotation(s) loaded.", 4000)

    def _on_variants_error(self, msg: str):
        self.fetch_variants_btn.setEnabled(True)
        self._mark_chip_normal(self.fetch_variants_btn)
        self.statusBar.showMessage("Variants fetch failed.", 2000)
        QMessageBox.warning(self, "Variants Error", msg)

    # ── IntAct ────────────────────────────────────────────────────────────────

    def fetch_intact(self):
        acc = self.current_accession
        if not acc:
            QMessageBox.warning(self, "IntAct", "Fetch a UniProt accession first.")
            return
        if self._intact_worker and self._intact_worker.isRunning():
            return
        self.fetch_intact_btn.setEnabled(False)
        self._mark_chip_loading(self.fetch_intact_btn)
        self.statusBar.showMessage(f"Fetching IntAct interactions for {acc}…")
        self._intact_worker = IntActWorker(acc)
        self._intact_worker.finished.connect(self._on_intact_finished)
        self._intact_worker.error.connect(self._on_intact_error)
        self._intact_worker.start()

    def _on_intact_finished(self, data: dict):
        self.intact_data = data
        self.fetch_intact_btn.setEnabled(True)
        interactions = data.get("interactions", [])
        if interactions:
            self._mark_chip_fetched(self.fetch_intact_btn)
        else:
            self._mark_chip_normal(self.fetch_intact_btn)
        if not interactions:
            QMessageBox.information(self, "IntAct",
                "No curated interactions found for this protein in IntAct.\n"
                "(Only experimentally validated binary interactions are included.)")
            self.statusBar.showMessage("IntAct: no interactions found.", 3000)
            return

        lines = [
            f"<h2>IntAct: {data.get('accession', '')}</h2>",
            f"<p><b>{len(interactions)}</b> binary interaction(s) retrieved "
            f"(up to 100 shown).</p>",
            "<table border='1' cellspacing='0' cellpadding='4'>",
            "<tr><th>Partner ID</th><th>Partner Name</th>"
            "<th>Detection Method</th><th>Interaction Type</th>"
            "<th>MI-score</th><th>PMID</th></tr>",
        ]
        for ix in interactions:
            score = ix.get("score")
            score_txt = f"{score:.2f}" if score is not None else "—"
            pmid = ix.get("pmid", "-")
            pmid_link = (f"<a href='https://pubmed.ncbi.nlm.nih.gov/{pmid}'>{pmid}</a>"
                         if pmid not in ("-", "") else "—")
            lines.append(
                f"<tr>"
                f"<td>{ix.get('partner_id', '—')}</td>"
                f"<td>{ix.get('partner_name', '—')}</td>"
                f"<td>{ix.get('detection_method', '—')}</td>"
                f"<td>{ix.get('interaction_type', '—')}</td>"
                f"<td>{score_txt}</td>"
                f"<td>{pmid_link}</td>"
                f"</tr>"
            )
        lines.append("</table>")
        lines.append(
            "<p class='note'>Source: IntAct (EBI) via PSICQUIC. "
            "MI-score = IntAct confidence score (0–1). "
            "Only experimentally validated interactions are included.</p>"
        )
        html = "".join(lines)
        dlg = QDialog(self)
        dlg.setWindowTitle(f"IntAct Interactions — {data.get('accession', '')}")
        dlg.resize(860, 500)
        bw = QTextBrowser(dlg)
        bw.setOpenExternalLinks(True)
        bw.setHtml(html)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, dlg)
        bb.rejected.connect(dlg.accept)
        lay = QVBoxLayout(dlg)
        lay.addWidget(bw)
        lay.addWidget(bb)
        dlg.exec()
        self.statusBar.showMessage(
            f"IntAct: {len(interactions)} interaction(s) loaded.", 4000)

    def _on_intact_error(self, msg: str):
        self.fetch_intact_btn.setEnabled(True)
        self._mark_chip_normal(self.fetch_intact_btn)
        self.statusBar.showMessage("IntAct fetch failed.", 2000)
        QMessageBox.warning(self, "IntAct Error", msg)

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
                    # Add to UI section list (as top-level item in QTreeWidget)
                    leaf = QTreeWidgetItem([sec_name])
                    leaf.setData(0, Qt.ItemDataRole.UserRole, sec_name)
                    self.report_section_list.addTopLevelItem(leaf)
                    tab = QWidget()
                    vb  = QVBoxLayout(tab)
                    vb.setContentsMargins(4, 4, 4, 4)
                    browser = QTextBrowser()
                    vb.addWidget(browser)
                    idx = self.report_stack.addWidget(tab)
                    self.report_section_tabs[sec_name] = browser
                    self._report_sec_to_idx[sec_name] = idx
            except Exception as e:
                print(f"[BEER] Plugin runtime error ({plugin.PLUGIN_NAME}): {e}",
                      file=sys.stderr)

    # ── New UX methods ────────────────────────────────────────────────────────


    def _undo_mutation(self):
        if self._undo_seq is None:
            self.statusBar.showMessage("Nothing to undo.", 2000)
            return
        self.seq_text.setPlainText(self._undo_seq)
        self.sequence_name = self._undo_name
        self._undo_seq  = None
        self._undo_name = None
        self.statusBar.showMessage("Mutation undone \u2014 re-running analysis\u2026", 2000)
        self.on_analyze()

    def _update_blast_elapsed(self):
        import time as _time
        if self._blast_start_time is None:
            return
        elapsed = int(_time.time() - self._blast_start_time)
        self.blast_status_lbl.setObjectName("status_lbl")
        self.blast_status_lbl.setProperty("status_state", "idle")
        self.blast_status_lbl.setText(f"Searching NCBI\u2026  {elapsed} s elapsed")

    def _on_report_section_clicked(self, item, _col=0):
        sec = item.data(0, Qt.ItemDataRole.UserRole)
        if not sec:
            return
        if sec in self._report_sec_to_idx:
            self.report_stack.setCurrentIndex(self._report_sec_to_idx[sec])
        # Trigger lazy computation for AI sections that haven't been computed yet.
        if sec.startswith("AI:") and sec not in self._ai_computed_sections:
            self._trigger_ai_section(sec)

    def _select_graph_tree_item(self, graph_title: str) -> None:
        """Highlight the graph tree leaf for *graph_title* without switching tabs."""
        for i in range(self.graph_tree.topLevelItemCount()):
            cat = self.graph_tree.topLevelItem(i)
            for j in range(cat.childCount()):
                leaf = cat.child(j)
                if leaf.data(0, Qt.ItemDataRole.UserRole) == graph_title:
                    self.graph_tree.setCurrentItem(leaf)
                    return

    def _select_report_section(self, sec_key: str) -> None:
        """Highlight *sec_key* in the report tree and show its stack page."""
        for i in range(self._ai_pred_grp_item.childCount()):
            leaf = self._ai_pred_grp_item.child(i)
            if leaf.data(0, Qt.ItemDataRole.UserRole) == sec_key:
                self.report_section_list.setCurrentItem(leaf)
                if sec_key in self._report_sec_to_idx:
                    self.report_stack.setCurrentIndex(self._report_sec_to_idx[sec_key])
                break

    def _filter_graph_tree(self, text: str):
        text = text.strip().lower()
        for i in range(self.graph_tree.topLevelItemCount()):
            cat = self.graph_tree.topLevelItem(i)
            any_visible = False
            for j in range(cat.childCount()):
                leaf = cat.child(j)
                title = leaf.data(0, Qt.ItemDataRole.UserRole) or leaf.text(0)
                visible = not text or text in title.lower()
                leaf.setHidden(not visible)
                if visible:
                    any_visible = True
            cat.setHidden(not any_visible and bool(text))
            if any_visible and text:
                cat.setExpanded(True)

    def _prepend_section_summaries(self):
        """Prepend a one-line interpretation line to key report sections."""
        if not self.analysis_data:
            return
        d = self.analysis_data
        summaries = {}
        # Properties
        mw = d.get("mol_weight", 0); pi = d.get("iso_point", 0); gravy = d.get("gravy", 0)
        charge_label = "acidic" if pi < 7 else ("basic" if pi > 7.5 else "near-neutral")
        summaries["Properties"] = (
            f"MW {mw:.0f} Da \u00b7 pI {pi:.2f} ({charge_label}) \u00b7 "
            f"GRAVY {gravy:+.3f} ({'hydrophobic' if gravy > 0 else 'hydrophilic'})")
        # Disorder
        dis = d.get("disorder_scores", d.get("disorder_profile", []))
        if dis:
            frac = sum(1 for v in dis if v > 0.5) / len(dis) * 100
            label = "largely disordered" if frac > 60 else ("partially disordered" if frac > 30 else "mostly ordered")
            summaries["Disorder"] = f"{frac:.0f}% of residues predicted disordered \u2014 {label}."
        # Charge
        fcr = d.get("fcr", 0); ncpr = d.get("ncpr", 0)
        summaries["Charge"] = (
            f"FCR {fcr:.3f} \u00b7 NCPR {ncpr:+.3f} "
            f"({'net positive' if ncpr > 0.05 else 'net negative' if ncpr < -0.05 else 'near-neutral charge'})")
        # β-Aggregation
        seq = d.get("seq", "")
        try:
            from beer.analysis.aggregation import calc_aggregation_profile as _cap
            agg = _cap(seq) if seq else []
        except Exception:
            agg = []
        if agg:
            hot = sum(1 for v in agg if v > 1.0)
            summaries["\u03b2-Aggregation & Solubility"] = (
                f"{hot} residue(s) above aggregation threshold (ZYGGREGATOR > 1.0).")
        # Signal peptide
        sp = d.get("sp_result", {})
        if sp:
            has_sp = sp.get("has_signal_peptide", False)
            prob   = sp.get("signal_peptide_prob", 0)
            _sp_src = "BiLSTM" if d.get("sp_bilstm_profile") is not None else "D-score"
            summaries["Signal Peptide & GPI"] = (
                ("Signal peptide detected" if has_sp else "No signal peptide predicted")
                + f" ({_sp_src} score {prob:.2f}).")
        for sec, summary in summaries.items():
            browser = self.report_section_tabs.get(sec)
            if browser:
                current_html = browser.toHtml()
                banner = (f"<p style='background:#f0f4ff;border-left:3px solid #4361ee;"
                          f"padding:4px 8px;margin:0 0 6px 0;font-size:9pt;color:#2d3748;'>"
                          f"<b>Summary:</b> {summary}</p>")
                if "Summary:" not in current_html:
                    browser.setHtml(banner + current_html)

    # ── Inline sparklines ────────────────────────────────────────────────────

    # Maps each report-section name to: data key in analysis_data, full graph
    # title (for navigation), sparkline colour, optional threshold value.
    _SPARKLINE_MAP: dict[str, tuple[str, str, str, float | None]] = {
        # section                     data_key                    graph_title                       colour     threshold
        "Disorder":               ("disorder_scores",         "Disorder Profile", "#4361ee", 0.5),
        "Hydrophobicity":         ("hydro_profile",           "Hydrophobicity Profile",         "#f77f00", 0.0),
        "Charge":                 ("ncpr_profile",            "Local Charge Profile",            "#e63946", 0.0),
        "Charge Decoration (SCD)":("scd_profile",             "SCD Profile",                    "#9b5de5", None),
        "Hydrophobicity Decoration (SHD)": ("shd_profile",   "SHD Profile",                    "#f77f00", None),
        "RNA Binding":            ("rbp_profile",             "RNA-Binding Profile",             "#2dc653", None),
        "β-Aggregation & Solubility": ("aggr_profile",        "β-Aggregation Profile",          "#e07a5f", 1.0),
        "Signal Peptide & GPI":   ("sp_bilstm_profile",       "Signal Peptide Profile",          "#f72585", 0.5),
        "TM Helices":             ("tm_bilstm_profile",       "Transmembrane Profile",           "#34d399", 0.5),
        "Intramembrane":          ("intramem_bilstm_profile", "Intramembrane Profile",           "#6ee7b7", 0.5),
        "Coiled-Coil":            ("cc_bilstm_profile",       "Coiled-Coil Profile",             "#fb923c", 0.5),
        "DNA-Binding":            ("dna_bilstm_profile",      "DNA-Binding Profile",             "#60a5fa", 0.5),
        "Active Site":            ("act_bilstm_profile",      "Active Site Profile",             "#f87171", 0.5),
        "Binding Site":           ("bnd_bilstm_profile",      "Binding Site Profile",            "#a78bfa", 0.5),
        "Phosphorylation":        ("phos_bilstm_profile",     "Phosphorylation Profile",         "#fbbf24", 0.5),
        "Low-Complexity":         ("lcd_bilstm_profile",      "Low-Complexity Profile",          "#94a3b8", 0.5),
        "Zinc Finger":            ("znf_bilstm_profile",      "Zinc Finger Profile",             "#4ade80", 0.5),
        "Glycosylation":          ("glyc_bilstm_profile",     "Glycosylation Profile",           "#f9a8d4", 0.5),
        "Ubiquitination":         ("ubiq_bilstm_profile",     "Ubiquitination Profile",          "#fb7185", 0.5),
        "Methylation":            ("meth_bilstm_profile",     "Methylation Profile",             "#a3e635", 0.5),
        "Acetylation":            ("acet_bilstm_profile",     "Acetylation Profile",             "#38bdf8", 0.5),
        "Lipidation":             ("lipid_bilstm_profile",    "Lipidation Profile",              "#e879f9", 0.5),
        "Disulfide Bonds":        ("disulf_bilstm_profile",   "Disulfide Bond Profile",          "#fde68a", 0.5),
        "Functional Motifs":      ("motif_bilstm_profile",    "Functional Motif Profile",        "#c4b5fd", 0.5),
        "Propeptide":             ("prop_bilstm_profile",     "Propeptide Profile",              "#fdba74", 0.5),
        "Repeat Regions":         ("rep_bilstm_profile",      "Repeat Region Profile",           "#67e8f9", 0.5),
        "Amphipathic Helices":    ("moment_alpha",            "Hydrophobic Moment",              "#4cc9f0", None),
    }

    @staticmethod
    def _make_sparkline_png(
        values: list[float],
        color: str,
        threshold: float | None = None,
        width_px: int = 520,
        height_px: int = 72,
    ) -> str:
        """Return a base64 data-URI PNG of a stripped-down sparkline.

        No axes, no labels, no ticks — just the filled profile shape and an
        optional threshold line.  Designed to sit directly below the report
        HTML text at a glance-friendly height.
        """
        import io, base64
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
        import numpy as _np

        dpi = 96
        fig, ax = _plt.subplots(
            figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#fafafa")

        x = _np.arange(len(values))
        y = _np.array(values, dtype=float)

        # Determine y-bounds with a small pad
        ymin, ymax = float(_np.nanmin(y)), float(_np.nanmax(y))
        pad = max((ymax - ymin) * 0.08, 0.02)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_xlim(0, max(len(values) - 1, 1))

        # Fill below (or between zero and line for signed data)
        if threshold is not None and threshold == 0.0:
            ax.fill_between(x, 0, y, where=(y >= 0),
                            color=color, alpha=0.28, linewidth=0)
            ax.fill_between(x, y, 0, where=(y < 0),
                            color="#e63946", alpha=0.22, linewidth=0)
        else:
            ax.fill_between(x, ymin - pad, y,
                            color=color, alpha=0.22, linewidth=0)

        ax.plot(x, y, color=color, linewidth=0.9, solid_capstyle="round")

        if threshold is not None:
            ax.axhline(threshold, color="#64748b",
                       linewidth=0.7, linestyle="--", alpha=0.7)

        # Strip all decorations
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.01,
                    facecolor=fig.get_facecolor())
        _plt.close(fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        return f"data:image/png;base64,{b64}"

    def _append_sparklines(self, data: dict) -> None:
        """Append a sparkline + 'View Full Graph' link to each relevant section."""
        for sec, (data_key, graph_title, color, threshold) in self._SPARKLINE_MAP.items():
            browser = self.report_section_tabs.get(sec)
            if browser is None:
                continue
            values = data.get(data_key) or []
            if not values:
                # Untrained BiLSTM head — append a muted badge only
                badge = (
                    "<div style='margin:8px 0 4px;padding:5px 10px;"
                    "background:#fafafa;border:1px solid #e2e8f0;border-radius:6px;"
                    "display:inline-block;font-family:sans-serif;font-size:10px;"
                    "color:#94a3b8'>AI head not yet trained — run training to "
                    "enable this profile.</div>"
                )
                current = browser.toHtml()
                if "AI head not yet trained" not in current:
                    browser.setHtml(current + badge)
                continue

            uri = self._make_sparkline_png(values, color, threshold)
            # URL-encode the graph title for the anchor href
            import urllib.parse as _up
            href = "beer://graph/" + _up.quote(graph_title)
            html_block = (
                f"<div style='margin:10px 0 6px;'>"
                f"<img src='{uri}' style='width:100%;height:72px;"
                f"border-radius:6px;display:block;'/>"
                f"<div style='text-align:right;margin-top:3px;'>"
                f"<a href='{href}' style='font-family:sans-serif;font-size:10px;"
                f"color:#4361ee;text-decoration:none;'>→ Full Graph</a>"
                f"</div></div>"
            )
            current = browser.toHtml()
            if "beer://graph/" not in current:
                browser.setHtml(current + html_block)

    # Sections with a corresponding graph but no per-residue 1-D profile.
    # Each entry gets a "→ View Graph" navigation link appended (no sparkline image).
    _GRAPH_LINK_MAP: dict[str, list[str]] = {
        "Composition":         ["Amino Acid Composition (Bar)"],
        "Properties":          ["Isoelectric Focus", "Charge Decoration"],
        "Aromatic & \u03c0":   ["Cation\u2013\u03c0 Map"],
        "Sticker & Spacer":    ["Sticker Map"],
        "Repeat Motifs":       ["Annotation Track"],
        "LARKS":               ["Annotation Track"],
        "Linear Motifs":       ["Linear Sequence Map"],
        "Tandem Repeats":      ["Cleavage Map"],
        "Proteolytic Map":     ["Cleavage Map"],
    }

    def _append_mini_graphs(self) -> None:
        """Append '→ View Graph' links to sections that have graphs but no sparkline."""
        import urllib.parse as _up
        for sec, graph_titles in self._GRAPH_LINK_MAP.items():
            browser = self.report_section_tabs.get(sec)
            if browser is None:
                continue
            current = browser.toHtml()
            if "beer://graph/" in current:
                continue  # already wired
            links = []
            for title in graph_titles:
                if title not in self._graph_generators:
                    continue
                href = "beer://graph/" + _up.quote(title)
                links.append(
                    f"<a href='{href}' style='font-family:sans-serif;font-size:10px;"
                    f"color:#4361ee;text-decoration:none;'>\u2192 {title}</a>"
                )
            if links:
                block = (
                    "<div style='margin:8px 0 4px;text-align:right;'>"
                    + " &nbsp;|&nbsp; ".join(links)
                    + "</div>"
                )
                browser.setHtml(current + block)


    # ── AI Predictions report sections ───────────────────────────────────────

    @staticmethod
    def _get_predicted_regions(scores: list[float], threshold: float = 0.5) -> list[tuple[int, int]]:
        """Return list of (start, end) 1-based residue ranges above threshold."""
        regions: list[tuple[int, int]] = []
        in_region = False
        start = 0
        for i, v in enumerate(scores):
            if v > threshold and not in_region:
                in_region = True
                start = i
            elif v <= threshold and in_region:
                in_region = False
                regions.append((start + 1, i))
        if in_region:
            regions.append((start + 1, len(scores)))
        return regions

    def _build_ai_head_html(
        self,
        display_name: str,
        scores: list[float],
        graph_title: str,
        auroc: str,
        threshold: float = 0.5,
        sparkline_uri: str = "",
    ) -> str:
        """Return HTML for one AI Predictions head section."""
        import urllib.parse as _up
        mean_score = sum(scores) / len(scores) if scores else 0.0
        regions = self._get_predicted_regions(scores, threshold)
        graph_href = "beer://graph/" + _up.quote(graph_title)
        # Build regions table
        if regions:
            region_rows = "".join(
                f"<tr><td>{k}</td><td>{s}–{e}</td><td>{e - s + 1}</td></tr>"
                for k, (s, e) in enumerate(regions, 1)
            )
            regions_html = (
                f"<h3 style='margin:10px 0 4px;font-size:12px'>Predicted Regions</h3>"
                f"<table><tr><th>#</th><th>Residues</th><th>Length (aa)</th></tr>"
                f"{region_rows}</table>"
                f"<p class='note'>{len(regions)} region(s) above threshold {threshold:.2f}.</p>"
            )
        else:
            regions_html = f"<p class='note'>No regions predicted above threshold {threshold:.2f}.</p>"
        auroc_str = f"AUROC {auroc}" if auroc != "—" else "AUROC: training in progress"
        if sparkline_uri:
            sparkline_html = (
                f"<div style='margin:10px 0 6px;'>"
                f"<img src='{sparkline_uri}' style='width:100%;height:72px;"
                f"border-radius:6px;display:block;'/>"
                f"<div style='text-align:right;margin-top:3px;'>"
                f"<a href='{graph_href}' style='font-family:sans-serif;font-size:10px;"
                f"color:#4361ee;text-decoration:none;'>→ Full Graph</a>"
                f"</div></div>"
            )
        else:
            sparkline_html = (
                f"<p style='margin:6px 0;'>"
                f"<a href='{graph_href}' style='color:#4361ee;font-size:10px;'>→ View full graph</a></p>"
            )
        return (
            f"<h2>{display_name} <span style='font-size:10px;font-weight:normal;"
            f"color:#64748b'>AI Predictions · {auroc_str}</span></h2>"
            f"<table>"
            f"<tr><th>Property</th><th>Value</th></tr>"
            f"<tr><td>Mean score</td><td>{mean_score:.3f}</td></tr>"
            f"<tr><td>Classification threshold</td><td>{threshold:.2f}</td></tr>"
            f"<tr><td>Sequence length</td><td>{len(scores)} aa</td></tr>"
            f"</table>"
            f"{regions_html}"
            f"{sparkline_html}"
            f"<p class='note'>Method: ESM2 650M embeddings → 2-layer BiLSTM classifier → sigmoid. "
            f"Trained on UniProt Swiss-Prot annotations.</p>"
        )

    # ── Lazy AI section loading ───────────────────────────────────────────────

    def _setup_ai_section_placeholders(self) -> None:
        """Populate the AI Predictions sidebar with 'click to compute' placeholders.

        Called after a classical (no-embedder) analysis so that all 23 AI heads
        appear in the sidebar immediately.  Clicking any entry triggers
        `_trigger_ai_section`, which launches an `AISectionWorker` for that head.
        """
        from beer.reports.css import make_style_tag
        _style = make_style_tag(dark=getattr(self, "_is_dark", False))

        # Tear down any previous AI section widgets.
        for key in list(self._ai_pred_section_keys):
            browser = self.report_section_tabs.pop(key, None)
            idx     = self._report_sec_to_idx.pop(key, None)
            if browser is not None:
                pw = browser.parent()
                if pw is not None:
                    self.report_stack.removeWidget(pw)
                    pw.deleteLater()
        self._ai_pred_section_keys.clear()
        while self._ai_pred_grp_item.childCount() > 0:
            self._ai_pred_grp_item.removeChild(self._ai_pred_grp_item.child(0))

        _embedder_ready = (
            self._embedder is not None and self._embedder.is_available()
        )
        _placeholder_html = _style + (
            "<h2>AI Prediction — not yet computed</h2>"
            "<div style='background:#f0f4ff;border-left:4px solid #4361ee;"
            "padding:16px;border-radius:4px;margin:12px 0'>"
            "<b>Click this section in the sidebar to compute the prediction.</b>"
            "<p style='margin:6px 0 0'>BEER will run the ESM2 650M embedding and "
            "the corresponding BiLSTM head for this feature only. "
            "The embedding is cached after the first computation, so subsequent "
            "sections are fast.</p>"
            + (""
               if _embedder_ready
               else "<p style='color:#b45309'><b>⚠ ESM2 not loaded.</b> "
                    "The AI Analysis button will attempt to load it on first use.</p>")
            + "</div>"
        )

        for display_name, data_key, graph_title, auroc in _AI_HEAD_SPECS:
            sec_key = f"AI:{display_name}"

            tab = QWidget()
            vb  = QVBoxLayout(tab)
            vb.setContentsMargins(4, 4, 4, 4)
            browser = QTextBrowser()
            _install_beer_link_filter(browser, self._on_report_link_clicked)
            browser.setHtml(_placeholder_html)
            vb.addWidget(browser)

            idx = self.report_stack.addWidget(tab)
            self.report_section_tabs[sec_key] = browser
            self._report_sec_to_idx[sec_key]  = idx
            self._ai_pred_section_keys.append(sec_key)

            leaf = QTreeWidgetItem([display_name])
            leaf.setData(0, Qt.ItemDataRole.UserRole, sec_key)
            self._ai_pred_grp_item.addChild(leaf)

        self._ai_pred_grp_item.setHidden(False)
        self._ai_pred_grp_item.setExpanded(True)

    def _trigger_ai_section(self, sec_key: str) -> None:
        """Launch an `AISectionWorker` for *sec_key* (e.g. 'AI:Disorder').

        Guards:
        - Does nothing if the section is already computed.
        - Does nothing if no analysis has been run yet.
        - If another AI section is already computing, shows a status message and
          returns; the user can click again once the current job finishes.
        """
        if sec_key in self._ai_computed_sections:
            return
        if not self.analysis_data:
            return

        if self._active_ai_worker is not None and self._active_ai_worker.isRunning():
            self.statusBar.showMessage(
                "AI computation already in progress — please wait.", 3000)
            return

        # Find the matching spec entry.
        display_name = sec_key[3:]  # strip "AI:"
        spec = next(
            ((dn, dk, gt, au) for dn, dk, gt, au in _AI_HEAD_SPECS
             if dn == display_name),
            None,
        )
        if spec is None:
            return
        _, data_key, graph_title, auroc = spec

        # Show a "computing" placeholder in the section browser.
        browser = self.report_section_tabs.get(sec_key)
        if browser:
            from beer.reports.css import make_style_tag
            _style = make_style_tag(dark=getattr(self, "_is_dark", False))
            browser.setHtml(_style + (
                f"<h2>{display_name} — computing…</h2>"
                "<div style='background:#fff8e1;border-left:4px solid #f59e0b;"
                "padding:16px;border-radius:4px;margin:12px 0'>"
                f"<b>⏳ Running ESM2 BiLSTM for <em>{display_name}</em>…</b>"
                "<p style='margin:6px 0 0'>This may take a moment on first use "
                "(the embedding is cached for subsequent sections).</p>"
                "</div>"
            ))

        seq = self.analysis_data.get("seq", "")
        self.statusBar.showMessage(
            f"Computing AI prediction: {display_name}…")

        self._progress_dlg = QProgressDialog(
            f"Computing {display_name}…", "Cancel", 0, 0, self)
        self._progress_dlg.setWindowTitle("BEER AI Prediction")
        self._progress_dlg.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress_dlg.setMinimumDuration(400)
        self._progress_dlg.canceled.connect(self._cancel_analysis)
        self._progress_dlg.show()

        worker = AISectionWorker(sec_key, data_key, seq, self._embedder)
        worker.result_ready.connect(self._on_ai_section_finished)
        worker.error.connect(self._on_ai_section_error)
        worker.finished.connect(self._on_ai_worker_thread_done)
        self._active_ai_worker = worker
        worker.start()

    def _on_ai_worker_thread_done(self) -> None:
        """Release the worker only after the OS thread has fully stopped."""
        self._active_ai_worker = None

    def _on_ai_section_finished(self, sec_key: str, scores: list) -> None:
        """Handle successful completion of a single AI section worker."""
        if hasattr(self, "_progress_dlg") and self._progress_dlg:
            self._progress_dlg.close()
            self._progress_dlg = None
        if not self.analysis_data:
            return

        display_name = sec_key[3:]
        spec = next(
            ((dn, dk, gt, au) for dn, dk, gt, au in _AI_HEAD_SPECS
             if dn == display_name),
            None,
        )
        if spec is None:
            return
        _, data_key, graph_title, auroc = spec

        # Persist scores in analysis_data so graphs and downstream sections can use them.
        self.analysis_data[data_key] = scores

        # Mark as computed before building HTML (prevents re-trigger on re-click).
        self._ai_computed_sections.add(sec_key)

        # Build sparkline and section HTML.
        color, threshold = {
            v[0]: (v[2], v[3]) for v in self._SPARKLINE_MAP.values()
        }.get(data_key, ("#4361ee", 0.5))
        sparkline_uri = self._make_sparkline_png(scores, color, threshold)

        from beer.reports.css import make_style_tag
        _style = make_style_tag(dark=getattr(self, "_is_dark", False))
        html = _style + self._build_ai_head_html(
            display_name, scores, graph_title, auroc, sparkline_uri=sparkline_uri)

        browser = self.report_section_tabs.get(sec_key)
        if browser:
            browser.setHtml(html)

        # Register the graph generator for this head now that data is available,
        # then schedule graph rendering and tree sync on the next event loop tick
        # (avoids segfault from creating Qt widgets inside a signal handler).
        self._build_graph_generators()
        self._generated_graphs.discard(graph_title)
        self._generated_graphs.discard("Overview")

        def _deferred_graph_update():
            self._select_graph_tree_item(graph_title)
            expected_graph_idx = self._graph_title_to_stack_idx.get(graph_title)
            if (expected_graph_idx is not None
                    and self.graph_stack.currentIndex() == expected_graph_idx):
                self._render_graph(graph_title)

        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, _deferred_graph_update)

        # Refresh AI Features structure coloring: update combo list and push scores
        # if this newly computed feature happens to be the active one.
        if hasattr(self, "struct_color_mode_combo"):
            if self.struct_color_mode_combo.currentText() == "AI Features":
                self._update_scheme_combo("AI Features")
                feature_label = _AI_DISPLAY_TO_FEATURE_LABEL.get(display_name, "")
                if feature_label and self.struct_scheme_combo.currentText() == feature_label:
                    self._push_feature_scores(feature_label)

        self.statusBar.showMessage(
            f"AI prediction complete: {display_name}", 3000)

    def _on_ai_section_error(self, sec_key: str, msg: str) -> None:
        """Handle a failed AI section computation."""
        if hasattr(self, "_progress_dlg") and self._progress_dlg:
            self._progress_dlg.close()
            self._progress_dlg = None
        display_name = sec_key[3:]
        browser = self.report_section_tabs.get(sec_key)
        if browser:
            from beer.reports.css import make_style_tag
            _style = make_style_tag(dark=getattr(self, "_is_dark", False))
            browser.setHtml(_style + (
                f"<h2>{display_name} — error</h2>"
                "<div style='background:#fff0f0;border-left:4px solid #ef4444;"
                "padding:16px;border-radius:4px;margin:12px 0'>"
                f"<b>Computation failed.</b><pre style='font-size:9pt;"
                f"white-space:pre-wrap'>{msg}</pre>"
                "<p>Check that ESM2 and the model file are installed correctly, "
                "then click this section again to retry.</p>"
                "</div>"
            ))
        self.statusBar.showMessage(
            f"AI prediction failed: {display_name} — {msg[:80]}", 6000)

    # ── (end lazy AI section loading) ─────────────────────────────────────────

    def _populate_ai_report_sections(self, data: dict) -> None:
        """Build/rebuild the AI Predictions tree node with one child per head that ran."""
        from beer.reports.css import make_style_tag
        _style = make_style_tag(dark=getattr(self, "_is_dark", False))

        # Remove previous AI section widgets from the stack
        for key in self._ai_pred_section_keys:
            browser = self.report_section_tabs.pop(key, None)
            idx = self._report_sec_to_idx.pop(key, None)
            if browser is not None:
                parent_widget = browser.parent()
                if parent_widget is not None:
                    self.report_stack.removeWidget(parent_widget)
                    parent_widget.deleteLater()

        self._ai_pred_section_keys.clear()

        # Remove old children from tree
        while self._ai_pred_grp_item.childCount() > 0:
            self._ai_pred_grp_item.removeChild(self._ai_pred_grp_item.child(0))

        bold_font = QFont()
        bold_font.setBold(True)

        # Reverse-map data_key → (color, threshold) from the sparkline map.
        _dk_to_color: dict[str, tuple[str, float | None]] = {
            v[0]: (v[2], v[3]) for v in self._SPARKLINE_MAP.values()
        }

        for display_name, data_key, graph_title, auroc in _AI_HEAD_SPECS:
            scores = data.get(data_key) or []
            if not scores:
                continue
            sec_key = f"AI:{display_name}"

            # Generate inline sparkline for this head.
            color, threshold = _dk_to_color.get(data_key, ("#4361ee", 0.5))
            sparkline_uri = self._make_sparkline_png(scores, color, threshold)

            # Build section widget (tab + browser)
            tab = QWidget()
            vb = QVBoxLayout(tab)
            vb.setContentsMargins(4, 4, 4, 4)
            btn_row = QHBoxLayout()
            btn_row.setSpacing(4)
            btn_row.addStretch()
            copy_btn = QPushButton("Copy Table")
            copy_btn.setMaximumWidth(100)
            copy_btn.setMinimumHeight(26)
            copy_btn.clicked.connect(lambda _, s=sec_key: self._copy_section(s))
            btn_row.addWidget(copy_btn)
            vb.addLayout(btn_row)
            browser = QTextBrowser()
            _install_beer_link_filter(browser, self._on_report_link_clicked)
            html = _style + self._build_ai_head_html(
                display_name, scores, graph_title, auroc, sparkline_uri=sparkline_uri)
            browser.setHtml(html)
            vb.addWidget(browser)

            idx = self.report_stack.addWidget(tab)
            self.report_section_tabs[sec_key] = browser
            self._report_sec_to_idx[sec_key] = idx
            self._ai_pred_section_keys.append(sec_key)

            leaf = QTreeWidgetItem([display_name])
            leaf.setData(0, Qt.ItemDataRole.UserRole, sec_key)
            self._ai_pred_grp_item.addChild(leaf)

        if self._ai_pred_section_keys:
            self._ai_pred_grp_item.setHidden(False)
            self._ai_pred_grp_item.setExpanded(True)
        else:
            self._ai_pred_grp_item.setHidden(True)

    def _on_report_link_clicked(self, url) -> None:
        """Handle 'beer://graph/<title>' links from inline sparklines."""
        import urllib.parse as _up
        raw = url.toString() if hasattr(url, "toString") else str(url)
        if not raw.startswith("beer://graph/"):
            return
        graph_title = _up.unquote(raw[len("beer://graph/"):])
        self._navigate_to_graph(graph_title)

    def _navigate_to_graph(self, graph_title: str) -> None:
        """Switch to the Graphs tab and select the named graph."""
        # NavTabWidget stores tab names in nav_list items as "  {icon}  {name}"
        graphs_idx = -1
        for i in range(self.main_tabs.nav_list.count()):
            if "Graphs" in self.main_tabs.nav_list.item(i).text():
                graphs_idx = i
                break
        if graphs_idx >= 0:
            self.main_tabs.setCurrentIndex(graphs_idx)

        # Find the leaf in the graph tree, expand its parent, and render.
        for i in range(self.graph_tree.topLevelItemCount()):
            cat = self.graph_tree.topLevelItem(i)
            for j in range(cat.childCount()):
                leaf = cat.child(j)
                if leaf.data(0, Qt.ItemDataRole.UserRole) == graph_title:
                    cat.setExpanded(True)
                    self.graph_tree.setCurrentItem(leaf)
                    self.graph_tree.scrollToItem(leaf)
                    self._on_graph_tree_clicked(leaf, 0)
                    return

    def _use_compare_seq(self, attr: str):
        te = getattr(self, attr, None)
        if te is None:
            return
        raw = te.toPlainText().strip()
        entries = self._parse_pasted_text(raw)
        seq = entries[0][1] if entries else raw.replace("\n", "").upper()
        if not seq or not is_valid_protein(seq):
            QMessageBox.warning(self, "Compare", "No valid sequence in that panel.")
            return
        self.seq_text.setPlainText(seq)
        name = entries[0][0] if entries else ""
        if name:
            self.sequence_name = name
        self.main_tabs.setCurrentIndex(0)
        self.on_analyze()

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


