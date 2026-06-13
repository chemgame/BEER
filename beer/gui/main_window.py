"""BEER main application window (PySide6)."""
from __future__ import annotations

import importlib
import importlib.util
import importlib.resources
import html as _html_mod
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
from contextlib import contextmanager

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
    QCheckBox, QRadioButton, QStatusBar, QComboBox, QFormLayout,
    QSplitter, QScrollArea, QFrame, QDialog, QDialogButtonBox,
    QSpinBox, QProgressDialog, QAbstractItemView,
    QListWidget, QListWidgetItem, QTreeWidget, QTreeWidgetItem, QStackedWidget,
    QInputDialog, QApplication, QDoubleSpinBox, QGroupBox, QMenu, QSlider,
    QColorDialog, QSizePolicy, QStyleFactory, QButtonGroup, QHeaderView,
)
from PySide6.QtGui import QFont, QKeySequence, QAction, QShortcut, QImage, QIcon, QPixmap, QColor
from PySide6.QtCore import Qt, QThread, Signal, QObject, QEvent, QUrl
from PySide6.QtPrintSupport import QPrinter
try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    _WEBENGINE_AVAILABLE = True
except ImportError:
    _WEBENGINE_AVAILABLE = False

from beer.gui.themes import (
    LIGHT_THEME_CSS, DARK_THEME_CSS,
    STRUCT_PANEL_CSS_LIGHT, STRUCT_TABBAR_CSS_LIGHT,
    STRUCT_PANEL_CSS_DARK, STRUCT_TABBAR_CSS_DARK,
)
from beer import config as _config
from beer.gui.nav_widget import NavTabWidget
from beer.gui.dialogs import MutationDialog, _FigureComposerDialog, FormatChooserDialog
from beer.io.structure_formats import pdb_to_mmcif, pdb_to_gro, pdb_to_xyz
from beer.io.graph_data_export import get_graph_data
from beer.constants import (
    NAMED_COLORS, NAMED_COLORMAPS, GRAPH_TITLES, GRAPH_CATEGORIES,
    REPORT_SECTIONS, VALID_AMINO_ACIDS, _AA_COLOURS, BILSTM_PROFILE_TABS,
    KYTE_DOOLITTLE, DEFAULT_PKA, PKA_SETS, DEFAULT_PKA_SET, DISORDER_PROPENSITY,
    AI_TRAIN_MAX_LEN, AM_PATHOGENIC_THRESHOLD, AM_BENIGN_THRESHOLD,
    CHOU_FASMAN_HELIX, CHOU_FASMAN_SHEET, LINEAR_MOTIFS,
    STICKER_AROMATIC, STICKER_ELECTROSTATIC,
    HYDROPHOBICITY_SCALES,
)
from beer.utils.sequence import clean_sequence, is_valid_protein, format_sequence_block
from beer.utils.biophysics import calc_net_charge
from beer.utils.pdb import (
    import_pdb_sequence, extract_chain_structures, extract_phi_psi,
    import_mmcif_sequence, extract_chain_structures_mmcif,
    parse_ca_atoms, parse_helix_sheet_records,
)
from beer.io.provenance import figure_metadata
from beer.analysis.core import AnalysisTools
from beer.analysis.aggregation import (
    calc_aggregation_profile, predict_aggregation_hotspots, calc_camsolmt_score,
)

# Alias used in graph generation block (original beer.py convention)
_extract_phi_psi = extract_phi_psi
from beer.network._http import fetch_uniprot_pdb_xrefs, fetch_rcsb_assembly_cif
from beer.network.workers import (
    AnalysisWorker, AlphaFoldWorker, ESMFold2Worker, PfamWorker, BlastWorker,
    ELMWorker, DisPRotWorker, PhaSepDBWorker,
    MobiDBWorker, UniProtVariantsWorker, IntActWorker,
    UniProtSequenceSearchWorker, UniProtFeaturesWorker,
    AISectionWorker, OverlayWorker, FetchAccessionWorker, BatchAnalysisWorker,
    ProteinSummaryWorker, CompositeStructureWorker,
    CompositeStructureESMFold2Worker, FetchPDBStructureWorker,
)
from beer.graphs import (
    create_domain_architecture_figure,
    create_amino_acid_composition_figure,
    create_hydrophobicity_figure, create_aggregation_profile_figure,
    create_solubility_profile_figure, create_scd_profile_figure,
    create_isoelectric_focus_figure,
    create_local_charge_figure, create_charge_decoration_figure,
    create_helical_wheel_figure,
    create_sticker_map_figure, create_hydrophobic_moment_figure,
    create_cation_pi_map_figure,
    create_local_complexity_figure, create_ramachandran_figure,
    create_plddt_figure, create_sasa_figure,
    create_distance_map_figure,
    create_structure_comparison_figure,
    create_msa_conservation_figure, create_complex_mw_figure,
    create_truncation_series_figure,
    create_saturation_mutagenesis_figure, create_uversky_phase_plot,
    create_cleavage_map_figure,
    create_plaac_profile_figure,
    create_msa_covariance_figure,
    create_bilstm_profile_figure,
    create_bilstm_dual_track_figure,
    create_shd_profile_figure,
)
from beer.graphs.variant_map import create_alphafold_missense_figure
from beer.graphs.overlay import create_overlay_figure
from beer.reports.css import REPORT_CSS, REPORT_CSS_DARK, get_report_css
from beer.reports.sections import (
    format_aggregation_report, format_signal_report,
    format_amphipathic_report, format_scd_report,
    format_repeats_report,
)
from beer.io.export import ExportTools
from beer.io.session import save_session, load_session
from beer.embeddings.base import SequenceEmbedder

# Ordered list of (display_name, analysis_data_key) for custom graph tabs.
# Classical RNA-binding removed: BiLSTM RNA-Binding head supersedes it.
_OVERLAY_PROFILE_MAP: list[tuple[str, str]] = [
    # Classical sequence profiles
    ("Hydrophobicity",       "hydro_profile"),
    ("Local Charge",         "ncpr_profile"),
    ("β-Aggregation",        "aggr_profile"),
    ("SCD Profile",          "scd_profile"),
    ("SHD Profile",          "shd_profile"),
    ("CatGranule",           "catgranule_profile"),
    # BiLSTM per-residue heads
    ("Disorder",             "disorder_scores"),
    ("Signal Peptide",       "sp_bilstm_profile"),
    ("Transmembrane",        "tm_bilstm_profile"),
    ("Intramembrane",        "intramem_bilstm_profile"),
    ("Coiled-Coil",          "cc_bilstm_profile"),
    ("DNA-Binding",          "dna_bilstm_profile"),
    ("RNA-Binding",          "rnabind_bilstm_profile"),
    ("Active Site",          "act_bilstm_profile"),
    ("Binding Site",         "bnd_bilstm_profile"),
    ("Phosphorylation",      "phos_bilstm_profile"),
    ("Low Complexity",       "lcd_bilstm_profile"),
    ("Zinc Finger",          "znf_bilstm_profile"),
    ("Glycosylation",        "glyc_bilstm_profile"),
    ("Ubiquitination",       "ubiq_bilstm_profile"),
    ("Methylation",          "meth_bilstm_profile"),
    ("Acetylation",          "acet_bilstm_profile"),
    ("Lipidation",           "lipid_bilstm_profile"),
    ("Disulfide Bonds",      "disulf_bilstm_profile"),
    ("Functional Motifs",    "motif_bilstm_profile"),
    ("Propeptide",           "prop_bilstm_profile"),
    ("Repeat Regions",       "rep_bilstm_profile"),
    ("Nucleotide-Binding",   "nucbind_bilstm_profile"),
    ("Transit Peptide",      "transit_bilstm_profile"),
    ("Helix (SS)",           "ss3_h_profile"),
    ("Strand (SS)",          "ss3_e_profile"),
    ("Coil (SS)",            "ss3_c_profile"),
]


def _calc_batch_stats(seq: str, data: dict) -> tuple:
    """Return (hydro%, hydrophil%, pos%, neg%, neu%) for a sequence."""
    length = len(seq)
    if length == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    aa_counts = data.get("aa_counts", {})
    hydro = sum(1 for aa in seq if KYTE_DOOLITTLE.get(aa, 0.0) > 0) / length * 100
    pos   = sum(aa_counts.get(k, 0) for k in ("K", "R", "H")) / length * 100
    neg   = sum(aa_counts.get(k, 0) for k in ("D", "E")) / length * 100
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
    "Disorder":                  "AI Predictions disorder score (ESMC 600M → BiLSTM classifier, AUROC 0.9999); classical propensity fallback. Predicted IDR regions listed with residue ranges.",
    "Repeat Motifs":             "RGG/RG, FG, SR/RS, QN/NQ repeat motifs relevant to RNA-binding and phase separation.",
    "Sticker & Spacer":          "Aromatic/charged stickers and uncharged spacers \u2014 key determinants of condensate properties.",
    "TM Helices":                "Kyte\u2013Doolittle sliding-window TM helix prediction; inside-positive topology rule.",
    "LARKS":                     "Low-complexity Aromatic-Rich Kinked Segments \u2014 structural motifs associated with amyloid-like fibres (Hughes et al. 2018).",
    "Linear Motifs":             "Regex scan for 15+ short linear motifs: NLS, NES, PxxP, 14-3-3, KFERQ, KDEL, and more.",
    "\u03b2-Aggregation & Solubility": "ZYGGREGATOR \u03b2-aggregation hotspots and CamSol intrinsic solubility per residue.",
    "PTM Sites":                 "ESMC-predicted phosphorylation, ubiquitination, SUMOylation, glycosylation, and methylation sites.",
    "Signal Peptide & GPI":      "ESMC signal-peptide probe (AUC 1.00); n/h/c-region annotation and GPI signal detection.",
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
    ("Structure & Topology", ["Amphipathic Helices", "SASA Profile"]),
    ("Functional Sites", ["Linear Motifs", "Tandem Repeats", "Proteolytic Map",
                           "\u03b2-Aggregation & Solubility"]),
]

# ---------------------------------------------------------------------------

# Milliseconds to wait for a background worker to finish before proceeding
# (graceful stop on re-trigger or app close). One value so the timeout is
# consistent across all workers rather than an ad-hoc 300/500/1000/1500.
_WORKER_WAIT_MS = 1500

# AI Predictions head specs: (display_name, data_key, graph_title, auroc)
_AI_HEAD_SPECS: list[tuple[str, str, str, str]] = [
    ("Disorder",            "disorder_scores",          "Disorder Profile",            "0.9923"),
    ("Signal Peptide",      "sp_bilstm_profile",        "Signal Peptide Profile",      "0.9999"),
    ("Transmembrane",       "tm_bilstm_profile",        "Transmembrane Profile",       "0.9826"),
    ("Intramembrane",       "intramem_bilstm_profile",  "Intramembrane Profile",       "0.9853"),
    ("Coiled-Coil",         "cc_bilstm_profile",        "Coiled-Coil Profile",         "0.9819"),
    ("DNA-Binding",         "dna_bilstm_profile",       "DNA-Binding Profile",         "0.9251"),
    ("RNA Binding",         "rnabind_bilstm_profile",   "RNA Binding Profile",         "0.9198"),
    ("Active Site",         "act_bilstm_profile",       "Active Site Profile",         "0.9926"),
    ("Binding Site",        "bnd_bilstm_profile",       "Binding Site Profile",        "0.8331"),
    ("Phosphorylation",     "phos_bilstm_profile",      "Phosphorylation Profile",     "0.9766"),
    ("Low-Complexity",      "lcd_bilstm_profile",       "Low-Complexity Profile",      "0.9672"),
    ("Zinc Finger",         "znf_bilstm_profile",       "Zinc Finger Profile",         "0.9335"),
    ("Glycosylation",       "glyc_bilstm_profile",      "Glycosylation Profile",       "0.9958"),
    ("Ubiquitination",      "ubiq_bilstm_profile",      "Ubiquitination Profile",      "0.9843"),
    ("Methylation",         "meth_bilstm_profile",      "Methylation Profile",         "0.9702"),
    ("Acetylation",         "acet_bilstm_profile",      "Acetylation Profile",         "0.9870"),
    ("Lipidation",          "lipid_bilstm_profile",     "Lipidation Profile",          "0.9983"),
    ("Disulfide Bond",      "disulf_bilstm_profile",    "Disulfide Bond Profile",      "0.9990"),
    ("Functional Motif",    "motif_bilstm_profile",     "Functional Motif Profile",    "0.9722"),
    ("Propeptide",          "prop_bilstm_profile",      "Propeptide Profile",          "0.9879"),
    ("Repeat Region",       "rep_bilstm_profile",       "Repeat Region Profile",       "0.9835"),
    ("Nucleotide-Binding",  "nucbind_bilstm_profile",   "Nucleotide-Binding Profile",  "0.9774"),
    ("Transit Peptide",     "transit_bilstm_profile",   "Transit Peptide Profile",     "0.9950"),
    ("SS3: α-Helix",        "ss3_h_profile",  "Secondary Structure: Helix Profile",   "0.843"),
    ("SS3: β-Strand",       "ss3_e_profile",  "Secondary Structure: Strand Profile",  "0.843"),
    ("SS3: Coil/Loop",      "ss3_c_profile",  "Secondary Structure: Coil Profile",    "0.843"),
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
    "SS3: α-Helix":       "SS3 Helix",
    "SS3: β-Strand":      "SS3 Strand",
    "SS3: Coil/Loop":     "SS3 Coil",
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
        self.setMinimumSize(1024, 680)
        self._is_dark = False
        self.setStyleSheet(LIGHT_THEME_CSS)


        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

        # Permanent ESMC status indicator (right side of status bar)
        self._esm2_indicator = QLabel()
        self._esm2_indicator.setContentsMargins(0, 0, 8, 0)
        self.statusBar.addPermanentWidget(self._esm2_indicator)
        self._update_esm2_indicator("ready")

        # State
        self.analysis_data       = None
        self.batch_data          = []
        _cfg = _config.load()
        self.default_window_size  = _cfg.get("window_size", 9)
        self.default_pH           = _cfg.get("ph", 7.4)
        self.use_reducing         = _cfg.get("use_reducing", False)
        self.custom_pka           = _cfg.get("custom_pka", None)
        self.pka_set              = _cfg.get("pka_set", DEFAULT_PKA_SET)
        self.colormap             = _cfg.get("colormap", "coolwarm")
        self.heatmap_cmap         = _cfg.get("heatmap_cmap", "viridis")
        self.transparent_bg       = _cfg.get("transparent_bg", True)
        self.label_font_size      = _cfg.get("label_font_size", 14)
        self.tick_font_size       = _cfg.get("tick_font_size", 12)
        self.marker_size          = _cfg.get("marker_size", 10)
        self.show_bead_labels     = _cfg.get("show_bead_labels", True)
        self.graph_color          = NAMED_COLORS.get(_cfg.get("graph_color", "Royal Blue"), "#4361ee")
        self.show_heading         = _cfg.get("show_heading", True)
        self.show_grid            = _cfg.get("show_grid", True)
        self.default_graph_format = _cfg.get("graph_format", "PNG")
        self.app_font_size        = _cfg.get("app_font_size", 12)
        self.enable_tooltips        = _cfg.get("enable_tooltips", True)
        self.colorblind_safe        = _cfg.get("colorblind_safe", False)
        self._esmc_missing_warned   = False   # show ESMC missing notice at most once
        self._last_was_bilstm       = False   # True only after BiLSTM Analysis completes
        # Lazy AI section state
        self._ai_computed_sections: set[str] = set()   # "AI:<name>" keys that have real scores
        self._active_ai_worker: "AISectionWorker | None" = None  # at most one at a time
        self._ai_longseq_warned_seq: str | None = None  # seq already warned about >1024 aa AI cap
        self._history             = []   # session-only: never restored from disk
        self.hydro_scale          = _cfg.get("hydro_scale", "Kyte-Doolittle")
        self.sequence_name       = ""
        self._tooltips: dict     = {}
        self._analysis_worker    = None
        self._discarded_workers: set = set()   # keep refs alive until threads finish
        self._composite_worker      = None
        self._composite_esm_worker  = None        # Fix PDB tab: ESMFold2 gap-fill worker
        self._exp_pdb_str           = None        # experimental PDB; set on non-AF structure load
        self._comp_exp_pdb_str      = None        # Fix PDB tab: own experimental PDB (independent)
        self._comp_pdb_fetch_worker = None        # Fix PDB tab: PDB fetch worker
        self._progress_dlg       = None
        self._pending_pdb        = None   # stored when loadPDB is called before page ready
        self._struct_page_ready  = False  # True once the 3Dmol base page has loaded
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
        self.esmfold2_data       = None  # dict: pdb_str, plddt, dist_matrix, aligned_pdb, rmsd_per_res
        self._struct_source      = "none"  # "none"|"alphafold"|"esmfold2"|"pdb"|"overlay"
        self._graph_struct_src   = "alphafold"  # "alphafold"|"esmfold2"|"both" — graphs source selector
        self._af_data            = None  # kept for Structure Comparison
        self.batch_struct        = {}   # maps batch rec_id -> per-chain struct dict
        self.pfam_domains        = []   # list of domain dicts from Pfam
        self._alphafold_worker   = None
        self._esmfold2_worker    = None
        self._overlay_worker     = None
        self._mc_dropout_worker  = None
        self._fetch_acc_worker   = None
        self._batch_analysis_worker = None
        self._protein_summary_worker = None
        self._pfam_worker        = None
        self._blast_worker       = None

        # --- New state for extended features ---
        self._blast_timer        = None
        self._blast_start_time   = None
        self._undo_seq           = None
        self._undo_name          = None
        self._undo_stack: list   = []    # snapshots for Ctrl+Z (mutation / Clear All / load)
        self._elm_worker         = None
        self._disprot_worker     = None
        self._phasepdb_worker    = None
        self._mobidb_worker      = None
        self._variants_worker       = None
        self._intact_worker         = None
        self._variant_effect_worker = None
        self._trunc_worker          = None
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
        self.init_report_tab()   # also hosts the Summary/Overview as its first section
        self.init_graphs_tab()
        self.init_structure_tab()
        self.init_blast_tab()
        self.init_batch_tab()
        self.init_comparison_tab()
        self.init_truncation_tab()
        self.init_msa_tab()
        self.init_complex_tab()
        self.init_composite_tab()
        self.init_settings_tab()
        self.init_help_tab()
        # Group the sidebar into logical clusters (visual order only — stack page
        # indices are unchanged, so keyboard shortcuts stay valid).
        self.main_tabs.set_display_order(
            ["Analysis", "Report", "Graphs",
             "Structure", "Fix PDB",
             "Compare", "Multichain Analysis", "Truncation", "MSA",
             "Protein Complex", "BLAST",
             "Settings", "Help"],
            group_breaks={"Structure", "Compare", "Settings"},
        )
        self._setup_shortcuts()
        self._disable_result_tabs()

        # Restore persisted theme
        if _cfg.get("theme_dark", False):
            self.theme_toggle.setChecked(True)
            self.setStyleSheet(DARK_THEME_CSS)
            plt.style.use("dark_background")
            self.main_tabs.set_icon_color("#cdd5ea")
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
                if isinstance(w, FigureCanvas):
                    plt.close(w.figure)
                w.setParent(None)

    @staticmethod
    def _clear_layout_deep(layout) -> None:
        """Remove and deleteLater every widget in *layout*, descending one level
        into nested sub-layouts. Shared by the several places that rebuild a
        dynamic widget row (PDB cross-refs, chain checkboxes, …)."""
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                sub = item.layout()
                while sub.count():
                    s = sub.takeAt(0)
                    if s.widget():
                        s.widget().deleteLater()

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
            "Architecture: ESMC 600M embeddings → 2-layer BiLSTM classifier (hidden=256) → sigmoid. "
            "Trained on UniProt Swiss-Prot 'Disordered region' annotations (AUROC 0.9999 on held-out test set). "
            "Threshold at F1-max ≈ 0.5 (dashed line). MC-Dropout (20 passes) provides ±1σ uncertainty band.\n\n"
            "Regions consistently above the threshold are predicted intrinsically disordered (IDRs)."
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
        "Ramachandran Plot": (
            "\u03c6/\u03c8 backbone dihedral angle scatter plot.\n\n"
            "\u03c6 (phi): C\u2013N\u2013C\u03b1\u2013C torsion angle (around N\u2013C\u03b1 bond).\n"
            "\u03c8 (psi): N\u2013C\u03b1\u2013C\u2013N torsion angle (around C\u03b1\u2013C bond).\n\n"
            "Dark shading = favoured (\u226598% of high-res structures); light = allowed; "
            "white = disallowed. Outliers may indicate modelling errors.\n"
            "Reference: Ramachandran et al., J. Mol. Biol. 7:95, 1963."
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
            "ESMC log-likelihood ratio (LLR) map for all single-residue substitutions.\n\n"
            "LLR(i, a) = log P(a | context) \u2212 log P(WT | context)\n\n"
            "Positive LLR (blue) = tolerated/favoured substitution. "
            "Negative LLR (red) = likely deleterious.\n\n"
            "Lower panel: mean LLR per position \u2014 "
            "positions with low mean LLR are evolutionarily constrained.\n\n"
            "NOTE: LLR is an evolutionary / language-model fitness score "
            "(sequence likelihood under ESMC). It is NOT a thermodynamic "
            "stability change \u0394\u0394G and must not be read as one \u2014 "
            "the units are log-probability (nats), not kcal/mol.\n\n"
            "Reference: Lin et al. (ESMC), PNAS 118:e2016239118, 2021."
        ),
        "AlphaMissense": (
            "AlphaMissense pathogenicity scores for all single-amino-acid substitutions.\n\n"
            "Score 0 = benign, 1 = pathogenic.\n"
            f"  \u2022 > {AM_PATHOGENIC_THRESHOLD}: likely pathogenic\n"
            f"  \u2022 {AM_BENIGN_THRESHOLD}\u2013{AM_PATHOGENIC_THRESHOLD}: ambiguous\n"
            f"  \u2022 < {AM_BENIGN_THRESHOLD}: likely benign\n\n"
            "AlphaMissense uses AlphaFold structure + evolutionary context. "
            "Scores fetched from EBI AlphaFold API.\n\n"
            "Lower panel: mean pathogenicity per position \u2014 "
            "positions with high mean scores are intolerant to substitution.\n\n"
            "Reference: Cheng et al., Science 381:eadg7492, 2023."
        ),
    }


    @staticmethod
    def _make_unavail_fig(title: str, body: str = "", is_dark: bool = False):
        """Return a themed matplotlib Figure for any 'not available' graph state."""
        from matplotlib.figure import Figure
        bg     = "#16213e" if is_dark else "#f8f9fd"
        accent = "#7b9cff" if is_dark else "#4361ee"
        muted  = "#94a3b8" if is_dark else "#718096"
        fig = Figure(figsize=(8, 3.5), dpi=100)
        fig.set_facecolor(bg)
        ax = fig.add_subplot(111)
        ax.set_facecolor(bg)
        ax.axis("off")
        y_title = 0.58 if body else 0.5
        ax.text(0.5, y_title, title,
                ha="center", va="center", fontsize=14, fontweight="bold",
                color=accent, transform=ax.transAxes)
        if body:
            ax.text(0.5, 0.36, body,
                    ha="center", va="center", fontsize=10, color=muted,
                    linespacing=1.6, transform=ax.transAxes)
        for sp in ax.spines.values():
            sp.set_visible(False)
        fig.tight_layout(pad=1.5)
        return fig

    def _make_training_placeholder_fig(self, tab_name: str, feat_name: str):
        """Return a 'model not yet trained' placeholder Figure (delegates to _make_unavail_fig)."""
        return self._make_unavail_fig(
            "Model Not Yet Trained",
            f"The  {feat_name}  prediction head has not been trained yet.\n"
            f"This graph will appear automatically once the model file is ready.\n"
            f"Architecture: ESMC 600M → 2-layer BiLSTM classifier → sigmoid",
            is_dark=getattr(self, "_is_dark", False),
        )

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
                self.main_tabs.nav_list.setCurrentRow(i)   # i is a row, not a stack idx
                break
        self._show_named_graph(graph_name)

    def _apply_fig_theme(self, fig) -> None:
        """Recolour a figure's chrome to match the current light/dark theme (screen)."""
        try:
            from beer.graphs._style import apply_figure_theme
            apply_figure_theme(fig, getattr(self, "_is_dark", False))
        except Exception:
            pass

    @contextmanager
    def _figure_export_light(self, fig):
        """Restyle a figure to the light palette for export, then restore it to the
        current on-screen theme — saved/copied figures are always publication-light."""
        from beer.graphs._style import apply_figure_theme
        apply_figure_theme(fig, False)
        try:
            yield
        finally:
            apply_figure_theme(fig, getattr(self, "_is_dark", False))
            try:
                if fig.canvas is not None:
                    fig.canvas.draw_idle()
            except Exception:
                pass

    def _replace_graph(self, title: str, fig):
        """Swap graph canvas in the named tab, preserving uncertainty checkbox state."""
        import matplotlib.pyplot as _plt
        tab, vb = self.graph_tabs[title]
        # Mark as populated so it is reachable in the graph tree even when the
        # main Analysis tab has not been run (e.g. self-contained MSA graphs).
        self._generated_graphs.add(title)
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
                    fig.set_tight_layout({"pad": 1.0, "h_pad": 0.8, "w_pad": 0.8})
                except Exception:
                    pass
        self._apply_fig_theme(fig)
        canvas = FigureCanvas(fig)
        from PySide6.QtWidgets import QSizePolicy as _SP
        canvas.setSizePolicy(_SP.Policy.Expanding, _SP.Policy.Expanding)
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
        vb.addWidget(canvas, 1)
        # Vertical crosshair on single-axes profile graphs (residue-position x-axis)
        _PROFILE_GRAPHS = BILSTM_PROFILE_TABS | {
            "Hydrophobicity Profile", "Local Charge Profile",
            "SCD Profile", "SHD Profile",
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
                dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
                dlg.setStyleSheet(_DCSS if _dark else _LCSS)
                vbl = _QVB(dlg)
                browser = _QTB()
                browser.setObjectName("info_dialog")
                browser.setOpenExternalLinks(False)
                browser.setReadOnly(True)
                browser.setPlainText(h)
                vbl.addWidget(browser)
                bb = _QBB(_QBB.StandardButton.Close)
                bb.rejected.connect(dlg.close)
                bb.accepted.connect(dlg.close)
                vbl.addWidget(bb)
                dlg.show()  # non-modal: user can interact with the app while reading

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
            "Distance Map",
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
        _btn_row.addStretch()
        btn = QPushButton("Save")
        btn.setToolTip("Save this figure — choose PNG / SVG / PDF in the save dialog")
        btn.clicked.connect(lambda _, t=title: self.save_graph(t))
        _export_btn = QPushButton("Data")
        _export_btn.setToolTip("Export the underlying data as CSV or JSON")
        _export_btn.clicked.connect(lambda _, t=title: self.export_graph_data(t))
        _copy_btn = QPushButton("Copy")
        _copy_btn.setToolTip("Copy figure to clipboard as PNG")
        _copy_btn.clicked.connect(lambda _, t=title: self._copy_graph_to_clipboard(t))
        _zip_btn = QPushButton("All (ZIP)")
        _zip_btn.setToolTip("Save all generated graphs as PNG files in a ZIP archive")
        _zip_btn.clicked.connect(self._export_all_graphs_zip)
        _zip_btn.setEnabled(bool(getattr(self, "analysis_data", None)))
        _bundle_btn = QPushButton("Bundle…")
        _bundle_btn.setToolTip("One-click publication bundle: vector figures + per-graph "
                               "data + combined per-residue CSV + report.html + provenance (ZIP)")
        _bundle_btn.clicked.connect(self._export_publication_bundle)
        _bundle_btn.setEnabled(bool(getattr(self, "analysis_data", None)))
        if not hasattr(self, "_zip_btns"):
            self._zip_btns = []
        self._zip_btns.append(_zip_btn)
        self._zip_btns.append(_bundle_btn)
        _btn_row.addWidget(btn)
        _btn_row.addWidget(_export_btn)
        _btn_row.addWidget(_copy_btn)
        _btn_row.addWidget(_zip_btn)
        _btn_row.addWidget(_bundle_btn)
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

    # ── Custom graph tabs (Feature Overlay + Feature Correlation) ───────────

    def _build_profile_checklist(self) -> "tuple[QWidget, list]":
        """Shared helper: scrollable QCheckBox list for all overlay profiles.

        Returns (scroll_widget, list_of_QCheckBox).  All checkboxes start
        disabled — call _populate_checkboxes() after analysis to enable them.
        """
        from PySide6.QtWidgets import QScrollArea as _SA

        inner = QWidget()
        inner_vb = QVBoxLayout(inner)
        inner_vb.setContentsMargins(2, 2, 2, 2)
        inner_vb.setSpacing(1)
        checkboxes: list[QCheckBox] = []
        for display_name, _ in _OVERLAY_PROFILE_MAP:
            cb = QCheckBox(display_name)
            cb.setEnabled(False)
            inner_vb.addWidget(cb)
            checkboxes.append(cb)
        inner_vb.addStretch()

        scroll = _SA()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(inner)
        return scroll, checkboxes

    # Keys that are lazily computed by BiLSTM heads (not in analysis_data at start)
    _BILSTM_LAZY_KEYS: frozenset = frozenset({
        "disorder_scores",
        "sp_bilstm_profile", "tm_bilstm_profile", "intramem_bilstm_profile",
        "cc_bilstm_profile", "dna_bilstm_profile", "act_bilstm_profile",
        "bnd_bilstm_profile", "phos_bilstm_profile", "lcd_bilstm_profile",
        "znf_bilstm_profile", "glyc_bilstm_profile", "ubiq_bilstm_profile",
        "meth_bilstm_profile", "acet_bilstm_profile", "lipid_bilstm_profile",
        "disulf_bilstm_profile", "motif_bilstm_profile", "prop_bilstm_profile",
        "rep_bilstm_profile", "rnabind_bilstm_profile", "nucbind_bilstm_profile",
        "transit_bilstm_profile",
        "ss3_h_profile", "ss3_e_profile", "ss3_c_profile",
    })

    def _populate_checkboxes(self, checkboxes: list) -> None:
        """Enable checkboxes based on data availability.

        Classical profiles: enabled only if data already in analysis_data.
        BiLSTM profiles: enabled if the embedder is available (data computed on demand).
        """
        if self.analysis_data is None:
            return
        ad = self.analysis_data
        embedder_ready = (self._embedder is not None
                          and self._embedder.is_available())
        for cb, (_, key) in zip(checkboxes, _OVERLAY_PROFILE_MAP):
            if key in self._BILSTM_LAZY_KEYS:
                cb.setEnabled(embedder_ready)
            else:
                data = ad.get(key)
                try:
                    available = (data is not None
                                 and hasattr(data, "__len__")
                                 and len(data) > 0)
                except Exception:
                    available = False
                cb.setEnabled(available)


    def _render_custom_canvas(
        self, tab_title: str, fig, clear_slot
    ) -> None:
        """Place a figure in a custom tab's right pane with Save + Clear buttons."""
        import matplotlib.pyplot as _plt
        from PySide6.QtWidgets import QSizePolicy as _SP
        from PySide6.QtCore import QSize as _QSize

        _, right_vb = self.graph_tabs[tab_title]

        for i in range(right_vb.count()):
            w = right_vb.itemAt(i)
            if w and isinstance(w.widget(), FigureCanvas):
                _plt.close(w.widget().figure)
                break
        self._clear_layout(right_vb)

        dpr = self.devicePixelRatioF() if hasattr(self, "devicePixelRatioF") else 1.0
        fig.set_dpi(min(150, max(96, int(96 * dpr))))

        self._apply_fig_theme(fig)
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(_SP.Policy.Expanding, _SP.Policy.Expanding)
        canvas.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        canvas.customContextMenuRequested.connect(
            lambda pos, c=canvas: self._graph_context_menu(c, pos))
        toolbar = NavigationToolbar2QT(canvas, self)
        toolbar.setIconSize(_QSize(20, 20))
        self._tint_toolbar_icons_dark(toolbar)
        right_vb.addWidget(toolbar)
        right_vb.addWidget(canvas, 1)

        _btn_bar = QWidget()
        _btn_row = QHBoxLayout(_btn_bar)
        _btn_row.setContentsMargins(0, 2, 0, 2)
        _btn_row.addStretch()
        _save_btn = QPushButton("Save")
        _save_btn.setToolTip("Save this figure — choose PNG / SVG / PDF in the save dialog")
        _save_btn.clicked.connect(lambda _, t=tab_title: self.save_graph(t))
        _clear_btn = QPushButton("Clear")
        _clear_btn.setToolTip("Deselect all features and return to initial state.")
        _clear_btn.clicked.connect(clear_slot)
        _btn_row.addWidget(_save_btn)
        _btn_row.addWidget(_clear_btn)
        right_vb.addWidget(_btn_bar)

    def _restore_custom_placeholder(self, tab_title: str, text: str) -> None:
        """Clear the right pane and show the placeholder label."""
        import matplotlib.pyplot as _plt
        _, right_vb = self.graph_tabs.get(tab_title, (None, None))
        if right_vb is None:
            return
        for i in range(right_vb.count()):
            w = right_vb.itemAt(i)
            if w and isinstance(w.widget(), FigureCanvas):
                _plt.close(w.widget().figure)
                break
        self._clear_layout(right_vb)
        ph = QLabel(text)
        ph.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ph.setObjectName("placeholder_lbl")
        right_vb.addWidget(ph)

    # ── Feature Overlay ──────────────────────────────────────────────────────

    def _build_overlay_tab(self):
        from PySide6.QtWidgets import QSplitter as _QSpl

        panel = QWidget()
        outer_vb = QVBoxLayout(panel)
        outer_vb.setContentsMargins(4, 4, 4, 4)
        splitter = _QSpl(Qt.Orientation.Horizontal)

        left = QWidget()
        left.setFixedWidth(200)
        left_vb = QVBoxLayout(left)
        left_vb.setContentsMargins(4, 4, 4, 4)
        left_vb.addWidget(QLabel("<b>Select features:</b>"))

        scroll, checkboxes = self._build_profile_checklist()
        left_vb.addWidget(scroll, 1)
        self._overlay_checkboxes = checkboxes

        normalize_chk = QCheckBox("Normalize 0–1")
        normalize_chk.setChecked(True)
        normalize_chk.setToolTip(
            "Rescale each profile independently to [0, 1] for visual comparison.\n"
            "Uncheck to plot raw values (profiles may have incompatible y-scales)."
        )
        left_vb.addWidget(normalize_chk)
        self._overlay_normalize = normalize_chk

        plot_btn = QPushButton("Plot Overlay")
        plot_btn.setEnabled(False)
        plot_btn.setToolTip("Generate overlay figure for checked features.")
        plot_btn.clicked.connect(self._plot_overlay)
        left_vb.addWidget(plot_btn)
        self._overlay_plot_btn = plot_btn

        right = QWidget()
        right_vb = QVBoxLayout(right)
        right_vb.setContentsMargins(0, 0, 0, 0)
        ph = QLabel("Run analysis, select features, then click Plot Overlay.")
        ph.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ph.setObjectName("placeholder_lbl")
        right_vb.addWidget(ph)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        outer_vb.addWidget(splitter)
        return panel, right_vb

    def _plot_overlay(self) -> None:
        if not hasattr(self, "_overlay_checkboxes") or self.analysis_data is None:
            return
        selected = self._build_selected_list(self._overlay_checkboxes)
        if not selected:
            from PySide6.QtWidgets import QMessageBox as _QMB
            _QMB.information(self, "Feature Overlay",
                             "No features selected. Check at least one feature.")
            return
        normalize = self._overlay_normalize.isChecked()
        self._launch_overlay_worker(
            selected,
            on_finished=lambda profiles: self._on_overlay_finished(profiles, normalize),
            title="Feature Overlay",
        )

    def _on_overlay_finished(self, profiles: dict, normalize: bool) -> None:
        if not profiles:
            return
        fig = create_overlay_figure(
            profiles, normalize=normalize,
            label_font=self.label_font_size, tick_font=self.tick_font_size,
        )
        self._render_custom_canvas("Feature Overlay", fig, self._clear_overlay)

    def _clear_overlay(self) -> None:
        if hasattr(self, "_overlay_checkboxes"):
            for cb in self._overlay_checkboxes:
                cb.setChecked(False)
        self._restore_custom_placeholder(
            "Feature Overlay",
            "Run analysis, select features, then click Plot Overlay."
        )

    # ── Feature Correlation ──────────────────────────────────────────────────

    def _build_correlation_tab(self):
        from PySide6.QtWidgets import QSplitter as _QSpl

        panel = QWidget()
        outer_vb = QVBoxLayout(panel)
        outer_vb.setContentsMargins(4, 4, 4, 4)
        splitter = _QSpl(Qt.Orientation.Horizontal)

        left = QWidget()
        left.setFixedWidth(200)
        left_vb = QVBoxLayout(left)
        left_vb.setContentsMargins(4, 4, 4, 4)
        left_vb.addWidget(QLabel("<b>Select features:</b>"))

        scroll, checkboxes = self._build_profile_checklist()
        left_vb.addWidget(scroll, 1)
        self._corr_checkboxes = checkboxes

        left_vb.addWidget(QLabel("Colormap:"))
        cmap_combo = QComboBox()
        cmap_combo.addItems(NAMED_COLORMAPS)
        cmap_combo.setCurrentText("coolwarm")
        cmap_combo.setToolTip("Colormap for the correlation heatmap.")
        left_vb.addWidget(cmap_combo)
        self._corr_cmap_combo = cmap_combo

        plot_btn = QPushButton("Plot Correlation")
        plot_btn.setEnabled(False)
        plot_btn.setToolTip("Generate pairwise Pearson correlation heatmap.")
        plot_btn.clicked.connect(self._plot_correlation)
        left_vb.addWidget(plot_btn)
        self._corr_plot_btn = plot_btn

        right = QWidget()
        right_vb = QVBoxLayout(right)
        right_vb.setContentsMargins(0, 0, 0, 0)
        ph = QLabel("Run analysis, select features, then click Plot Correlation.")
        ph.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ph.setObjectName("placeholder_lbl")
        right_vb.addWidget(ph)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        outer_vb.addWidget(splitter)
        return panel, right_vb

    def _plot_correlation(self) -> None:
        if not hasattr(self, "_corr_checkboxes") or self.analysis_data is None:
            return
        selected = self._build_selected_list(self._corr_checkboxes)
        if len(selected) < 2:
            from PySide6.QtWidgets import QMessageBox as _QMB
            _QMB.information(self, "Feature Correlation",
                             "Select at least 2 features to compute correlation.")
            return
        cmap = self._corr_cmap_combo.currentText()
        self._launch_overlay_worker(
            selected,
            on_finished=lambda profiles: self._on_correlation_finished(profiles, cmap),
            title="Feature Correlation",
        )

    def _on_correlation_finished(self, profiles: dict, cmap: str) -> None:
        if len(profiles) < 2:
            return
        from beer.graphs.overlay import create_correlation_figure
        fig = create_correlation_figure(
            profiles, cmap=cmap,
            label_font=self.label_font_size, tick_font=self.tick_font_size,
        )
        self._render_custom_canvas("Feature Correlation", fig, self._clear_correlation)

    def _clear_correlation(self) -> None:
        if hasattr(self, "_corr_checkboxes"):
            for cb in self._corr_checkboxes:
                cb.setChecked(False)
        self._restore_custom_placeholder(
            "Feature Correlation",
            "Run analysis, select features, then click Plot Correlation."
        )

    def _build_selected_list(
        self, checkboxes: list
    ) -> "list[tuple[str, str, list | None]]":
        """Return (display_name, key, data_or_None) for every checked, enabled checkbox."""
        if self.analysis_data is None:
            return []
        ad = self.analysis_data
        result = []
        for cb, (display_name, key) in zip(checkboxes, _OVERLAY_PROFILE_MAP):
            if not cb.isChecked() or not cb.isEnabled():
                continue
            data = ad.get(key)
            try:
                if data is not None and len(data) == 0:
                    data = None
            except Exception:
                data = None
            result.append((display_name, key, data))
        return result

    def _launch_overlay_worker(
        self,
        selected: list,
        on_finished,
        title: str,
    ) -> None:
        """Launch OverlayWorker for *selected* profiles, showing a progress dialog.

        If no BiLSTM heads need computation, calls *on_finished* immediately
        without spinning up a thread (avoids unnecessary overhead).
        """
        if self._overlay_worker is not None and self._overlay_worker.isRunning():
            return

        needs_compute = any(
            data is None and key in self._BILSTM_LAZY_KEYS
            for _, key, data in selected
        )
        if not needs_compute:
            # All data already in analysis_data — collect synchronously and plot
            profiles = {name: list(data) for name, _, data in selected if data}
            on_finished(profiles)
            return

        # Show progress dialog consistent with the AI heads in Reports/Graphs
        self._overlay_progress_dlg = QProgressDialog(
            "Computing AI profiles…", None, 0, 0, self)
        self._overlay_progress_dlg.setWindowTitle(f"BEER — {title}")
        self._overlay_progress_dlg.setWindowModality(Qt.WindowModality.WindowModal)
        self._overlay_progress_dlg.setMinimumDuration(300)
        self._overlay_progress_dlg.show()

        seq = self.analysis_data.get("seq", "")
        worker = OverlayWorker(
            selected, seq, self._embedder,
            self._BILSTM_LAZY_KEYS,
        )
        worker.progress.connect(
            lambda msg: self._overlay_progress_dlg.setLabelText(msg))
        worker.finished.connect(
            lambda profiles, computed: self._on_overlay_worker_done(profiles, computed, on_finished))
        worker.error.connect(
            lambda msg: self.statusBar.showMessage(f"Overlay warning: {msg}", 4000))
        self._overlay_worker = worker
        worker.start()

    def _on_overlay_worker_done(self, profiles: dict, computed: dict, on_finished) -> None:
        dlg = getattr(self, "_overlay_progress_dlg", None)
        if dlg is not None:
            dlg.reset()
            dlg.close()
            self._overlay_progress_dlg = None
        self._overlay_worker = None
        # Merge newly computed BiLSTM keys into analysis_data on the main thread.
        if computed and self.analysis_data:
            self.analysis_data.update(computed)
        # Re-register graph generators for any BiLSTM heads that OverlayWorker
        # just added to analysis_data.  Without this, those graph tabs show
        # "not available" even though the data is now cached.
        if self.analysis_data:
            self._build_graph_generators()
        on_finished(profiles)

    def _populate_overlay_checklist(self) -> None:
        """Called from update_graph_tabs — enables checkboxes for both custom tabs."""
        if hasattr(self, "_overlay_checkboxes"):
            self._populate_checkboxes(self._overlay_checkboxes)
            if hasattr(self, "_overlay_plot_btn"):
                self._overlay_plot_btn.setEnabled(True)
        if hasattr(self, "_corr_checkboxes"):
            self._populate_checkboxes(self._corr_checkboxes)
            if hasattr(self, "_corr_plot_btn"):
                self._corr_plot_btn.setEnabled(True)

    # ── end Custom graph tabs ────────────────────────────────────────────────

    def _update_esm2_indicator(self, state: str = "ready",
                               disorder_method: str = "") -> None:
        """Update the permanent disorder-method status label in the status bar.

        state: 'ready'      — ESMC available, model not yet run this session
               'active'     — ESMC was used in the last analysis
               'metapredict'— ESMC unavailable; metapredict used
               'classical'  — ESMC and metapredict both unavailable
               'missing'    — esm / torch not installed (no analysis yet)
        """
        from beer.embeddings import ESMC_AVAILABLE
        model = getattr(self._embedder, "model_name", None)
        parts = model.split("_") if model else []
        try:
            size_tag = next(p for p in parts if p.endswith("M") or p.endswith("B"))
        except StopIteration:
            size_tag = ""

        if not ESMC_AVAILABLE or self._embedder is None:
            if state == "metapredict":
                text       = "Disorder \u00b7 metapredict"
                esm2_state = "metapredict"
            elif state == "classical":
                text       = "Disorder \u00b7 propensity scale"
                esm2_state = "classical"
            else:
                text       = "ESMC \u00b7 not installed"
                esm2_state = "missing"
        elif state == "active":
            text       = f"ESMC {size_tag} \u00b7 active \u2714"
            esm2_state = "active"
        else:
            text       = f"ESMC {size_tag} \u00b7 ready"
            esm2_state = "ready"

        self._esm2_indicator.setText(text)
        self._esm2_indicator.setObjectName("esm2_lbl")
        self._esm2_indicator.setProperty("esm2_state", esm2_state)
        self._esm2_indicator.style().unpolish(self._esm2_indicator)
        self._esm2_indicator.style().polish(self._esm2_indicator)

    def set_embedder(self, embedder) -> None:
        """Set embedder after window creation (used for deferred background loading)."""
        self._embedder = embedder
        self._update_esm2_indicator("ready")

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

    def _mark_chip_error(self, btn: "QPushButton") -> None:
        """Set chip button to error state (red border) to signal a failed fetch."""
        btn.setProperty("chip_state", "error")
        btn.style().unpolish(btn)
        btn.style().polish(btn)

    def _set_status_lbl(self, lbl: "QLabel", text: str, state: str) -> None:
        """Update a status label with an appropriate icon prefix and CSS state."""
        prefix = {"success": "✓ ", "error": "✕ ", "idle": ""}.get(state, "")
        lbl.setText(prefix + text)
        lbl.setProperty("status_state", state)
        lbl.style().unpolish(lbl)
        lbl.style().polish(lbl)

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
                with self._figure_export_light(canvas.figure):
                    canvas.figure.savefig(fn, format=ext, dpi=300, bbox_inches="tight",
                                          metadata=figure_metadata(ext))
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
        """Start async analysis of (id, seq) pairs into the batch table."""
        if self._batch_analysis_worker is not None and self._batch_analysis_worker.isRunning():
            self._batch_analysis_worker.requestInterruption()
            self._batch_analysis_worker.wait(_WORKER_WAIT_MS)
        self.batch_data.clear()
        self.batch_table.setRowCount(0)
        # Reset structure state; callers that bring structure (import_pdb,
        # fetch_accession PDB branch, _on_alphafold_finished) re-populate these
        # after calling _load_batch.
        self.batch_struct         = {}
        self.alphafold_data       = None
        self.esmfold2_data        = None
        self._af_data             = None
        self._struct_source       = "none"
        self._struct_is_alphafold = False
        self.export_structure_btn.setEnabled(False)
        # Classical-only (embedder=None) so every chain is analysed in milliseconds
        # and the chain selector / Multichain table populate immediately. Passing the
        # ESMC embedder here forced a full language-model forward per chain (and a
        # ~2.4 GB first-run download), which left the multichain UI empty for minutes.
        # Per-chain AI heads are still available on demand when a chain is loaded.
        worker = BatchAnalysisWorker(
            entries, 7.0, self.default_window_size,
            self.use_reducing, self.custom_pka,
            hydro_scale=self.hydro_scale,
            embedder=None,
        )
        worker.chain_result.connect(self._on_batch_chain_result)
        worker.finished.connect(self._on_batch_finished)
        worker.progress.connect(lambda m: self.statusBar.showMessage(m))
        worker.finished.connect(lambda _: setattr(self, "_batch_analysis_worker", None))
        self._batch_analysis_worker = worker
        worker.start()

    def _on_batch_chain_result(self, rec_id: str, seq: str, data: dict) -> None:
        self.batch_data.append((rec_id, seq, data))
        self._populate_batch_row(rec_id, seq, data)
        self._populate_chain_combo()

    def _on_batch_finished(self, skipped: list) -> None:
        if skipped:
            names = ", ".join(skipped[:5]) + ("…" if len(skipped) > 5 else "")
            QMessageBox.warning(
                self, "Sequences Skipped",
                f"{len(skipped)} sequence(s) were skipped due to invalid or "
                f"unsupported characters, or being shorter than 5 residues:\n{names}"
            )

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
        row1.addWidget(self.import_btn)

        row1.addSpacing(6)
        row1.addWidget(QLabel("Fetch:"))
        self.accession_input = QLineEdit()
        self.accession_input.setPlaceholderText("e.g. P35637 (FUS)  ·  1UBQ  ·  P08100 (rhodopsin)")
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

        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.setToolTip(
            "Run classical biophysical analysis (composition, charge, hydrophobicity, etc.).\n"
            "AI prediction heads are computed on demand when you click them in Graphs or Reports.\n"
            "Keyboard shortcut: Ctrl+Enter")
        self.analyze_btn.setMinimumHeight(30)
        self.analyze_btn.setObjectName("primary_btn")
        self.analyze_btn.clicked.connect(self._on_analyze_btn_clicked)
        row2.addWidget(self.analyze_btn)

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

        # ── Welcome / empty-state callout (hidden once analysis has run) ─────
        self._welcome_banner = QFrame()
        self._welcome_banner.setObjectName("welcome_banner")
        _wb_layout = QVBoxLayout(self._welcome_banner)
        _wb_layout.setContentsMargins(14, 10, 14, 10)
        _wb_layout.setSpacing(5)

        _wb_headline = QLabel("<b>Get started</b>")
        _wb_headline.setObjectName("welcome_lbl")
        _wb_layout.addWidget(_wb_headline)

        _wb_hint = QLabel(
            "Enter a UniProt or PDB accession above and click Fetch, "
            "or paste a sequence below and click Analyze."
        )
        _wb_hint.setObjectName("status_lbl")
        _wb_hint.setWordWrap(True)
        _wb_layout.addWidget(_wb_hint)

        _wb_chips = QHBoxLayout()
        _wb_chips.setSpacing(6)
        _wb_chips.setContentsMargins(0, 3, 0, 0)
        for _acc, _label in [
            ("P35637", "P35637 · FUS"),
            ("P08100", "P08100 · Rhodopsin"),
            ("1UBQ",   "1UBQ · Ubiquitin"),
            ("4HHB",   "4HHB · Haemoglobin"),
        ]:
            _wb_btn = QPushButton(_label)
            _wb_btn.setObjectName("chip_btn")
            _wb_btn.clicked.connect(
                lambda _checked=False, _a=_acc:
                self._on_welcome_link(f"beer://fetch/{_a}")
            )
            _wb_chips.addWidget(_wb_btn)
        _wb_chips.addStretch()
        _wb_layout.addLayout(_wb_chips)

        outer.addWidget(self._welcome_banner)

        # ── Sequence input ───────────────────────────────────────────────────
        self._seq_label = QLabel("Protein Sequence:")
        self._seq_label.setObjectName("accent_lbl")
        outer.addWidget(self._seq_label)

        self.seq_text = QTextEdit()
        self.seq_text.setPlaceholderText("Paste a protein sequence here, or use Import\u2026")
        _mono_font = QFont()
        _mono_font.setFamilies(["JetBrains Mono", "Cascadia Code", "Menlo", "Consolas", "Courier New"])
        _mono_font.setPointSize(10)
        _mono_font.setStyleHint(QFont.StyleHint.Monospace)
        self.seq_text.setFont(_mono_font)
        self.seq_text.setFixedHeight(72)
        self.seq_text.setAcceptDrops(True)
        outer.addWidget(self.seq_text)

        # ── Chain selector (hidden until multi-chain data is loaded) ────────────
        self._chain_row_widget = QWidget()
        chain_row = QHBoxLayout(self._chain_row_widget)
        chain_row.setContentsMargins(0, 0, 0, 0)
        chain_lbl = QLabel("Chain:")
        chain_lbl.setObjectName("chain_lbl")
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

        def _chip(label, tip, slot, row):
            b = QPushButton(label)
            b.setObjectName("chip_btn")
            b.setProperty("chip_state", "normal")
            b.setEnabled(False)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            row.addWidget(b)
            return b

        chips_row1 = QHBoxLayout()
        chips_row1.setSpacing(4)
        grp_lbl = QLabel("Structure")
        grp_lbl.setObjectName("group_lbl")
        chips_row1.addWidget(grp_lbl)
        self.fetch_af_btn = _chip("AlphaFold",
            "Fetch AlphaFold predicted structure (requires UniProt accession)",
            self.fetch_alphafold, chips_row1)
        self.predict_struct_btn = _chip("Predict Structure",
            "Predict structure with ESMFold2 via EvolutionaryScale Forge API (requires BioHub API key in Settings)",
            self._run_esmfold2, chips_row1)
        self.fetch_pfam_btn = _chip("Pfam",
            "Fetch Pfam domain annotations from InterPro",
            self.fetch_pfam, chips_row1)
        # UniProt Tracks is accessible from the Graphs tab top bar; no chip here.
        self.fetch_uniprot_tracks_btn = QPushButton(); self.fetch_uniprot_tracks_btn.hide()
        self.fetch_uniprot_tracks_btn.clicked.connect(self.fetch_uniprot_features)
        chips_row1.addStretch()

        chips_row2 = QHBoxLayout()
        chips_row2.setSpacing(4)
        grp_lbl2 = QLabel("Disorder / IDP")
        grp_lbl2.setObjectName("group_lbl")
        chips_row2.addWidget(grp_lbl2)
        self.fetch_elm_btn = _chip("ELM",
            "Fetch experimentally validated linear motifs from ELM (UniProt only)",
            self.fetch_elm, chips_row2)
        self.fetch_disprot_btn = _chip("DisProt",
            "Fetch disorder annotations from DisProt (UniProt only)",
            self.fetch_disprot, chips_row2)
        self.fetch_mobidb_btn = _chip("MobiDB",
            "Fetch consensus disorder annotations from MobiDB (UniProt only)",
            self.fetch_mobidb, chips_row2)
        self.fetch_phasepdb_btn = _chip("PhaSepDB",
            "Check phase-separation database PhaSepDB (UniProt only)",
            self.fetch_phasepdb, chips_row2)
        chips_row2.addStretch()

        chips_row3 = QHBoxLayout()
        chips_row3.setSpacing(4)
        grp_lbl3 = QLabel("Variants & Interactions")
        grp_lbl3.setObjectName("group_lbl")
        chips_row3.addWidget(grp_lbl3)
        self.fetch_variants_btn = _chip("Variants",
            "Fetch natural variants and mutagenesis data from UniProt",
            self.fetch_variants, chips_row3)
        self.fetch_alphafold_missense_btn = _chip("AlphaMissense",
            "Fetch AlphaMissense variant pathogenicity scores from EBI (UniProt only)",
            lambda: self._run_alphafold_missense(self.current_accession), chips_row3)
        self.fetch_intact_btn = _chip("IntAct",
            "Fetch curated binary interactions from IntAct / EBI (UniProt only)",
            self.fetch_intact, chips_row3)
        chips_row3.addStretch()

        ext_vbox.addLayout(chips_row1)
        ext_vbox.addLayout(chips_row2)
        ext_vbox.addLayout(chips_row3)

        # ── PDB cross-reference chips — shown after UniProt fetch ────────────
        self._pdb_xref_inner = QWidget()
        self._pdb_xref_layout = QVBoxLayout(self._pdb_xref_inner)
        self._pdb_xref_layout.setContentsMargins(4, 2, 4, 2)
        self._pdb_xref_layout.setSpacing(3)
        self._pdb_xref_inner.hide()
        ext_vbox.addWidget(self._pdb_xref_inner)

        self._ext_data_panel.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        outer.addWidget(self._ext_data_panel)

        # Convenience list for bulk enable/disable
        self._db_fetch_btns = [
            self.fetch_af_btn, self.predict_struct_btn, self.fetch_pfam_btn,
            self.fetch_elm_btn, self.fetch_disprot_btn, self.fetch_mobidb_btn,
            self.fetch_phasepdb_btn, self.fetch_variants_btn, self.fetch_intact_btn,
            self.fetch_alphafold_missense_btn,
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
        self.seq_viewer.setFont(_mono_font)
        self.seq_viewer.setMinimumHeight(48)
        self.seq_viewer.setMaximumHeight(160)
        outer.addWidget(self.seq_viewer)
        outer.addStretch(1)

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

    def _on_welcome_link(self, url: str) -> None:
        """Handle clicks on example accession links in the welcome banner."""
        if url.startswith("beer://fetch/"):
            acc = url[len("beer://fetch/"):]
            self.accession_input.setText(acc)
            self.fetch_accession()

    # ── Nav tab gating ───────────────────────────────────────────────────────

    # Tab names that require a completed analysis to be useful. Referenced by name
    # (not index) so adding/removing/reordering tabs never breaks the gating.
    _RESULT_TAB_NAMES = ("Report", "Graphs")

    def _goto_tab(self, name: str) -> None:
        """Switch to a tab by name (robust to stack-index changes)."""
        idx = self.main_tabs.stack_for_name(name)
        if idx >= 0:
            self.main_tabs.setCurrentIndex(idx)

    def _set_nav_tab_enabled(self, idx: int, enabled: bool) -> None:
        row = self.main_tabs.row_for_stack(idx)
        item = self.main_tabs.nav_list.item(row) if row >= 0 else None
        if item is None:
            return
        if enabled:
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        else:
            item.setFlags(Qt.ItemFlag(0))
            if self.main_tabs.nav_list.currentRow() == row:
                self.main_tabs.setCurrentIndex(0)

    def _on_msa_link(self, url) -> None:
        """Handle 'opengraph:<title>' links emitted from the MSA preview pane."""
        s = url.toString()
        if s.startswith("opengraph:"):
            from urllib.parse import unquote
            self._open_graph(unquote(s.split(":", 1)[1]))

    def _open_graph(self, title: str) -> None:
        """Enable the Graphs tab, switch to it, and display a specific graph."""
        if title not in self._graph_title_to_stack_idx:
            return
        self._set_nav_tab_enabled(self.main_tabs.stack_for_name("Graphs"), True)
        self._goto_tab("Graphs")
        # Select the matching leaf in the graph tree (expands its category).
        from PySide6.QtWidgets import QTreeWidgetItemIterator
        it = QTreeWidgetItemIterator(self.graph_tree)
        while it.value():
            node = it.value()
            if node.data(0, Qt.ItemDataRole.UserRole) == title:
                parent = node.parent()
                if parent is not None:
                    parent.setExpanded(True)
                self.graph_tree.setCurrentItem(node)
                break
            it += 1
        self.graph_stack.setCurrentIndex(self._graph_title_to_stack_idx[title])
        self._render_graph(title)

    def _disable_result_tabs(self) -> None:
        for name in self._RESULT_TAB_NAMES:
            self._set_nav_tab_enabled(self.main_tabs.stack_for_name(name), False)

    def _enable_result_tabs(self) -> None:
        for name in self._RESULT_TAB_NAMES:
            self._set_nav_tab_enabled(self.main_tabs.stack_for_name(name), True)

    def _flash_nav_tab(self, idx: int, flashes: int = 3) -> None:
        """Briefly flash a nav tab item to draw attention after it is enabled.

        *idx* is a stack page index; map it to its current sidebar row so this
        stays correct regardless of the visual ordering (set_display_order)."""
        from PySide6.QtCore import QTimer
        from PySide6.QtGui import QColor, QBrush
        row = self.main_tabs.row_for_stack(idx)
        item = self.main_tabs.nav_list.item(row) if row >= 0 else None
        if item is None:
            return
        accent = "#7b9cff" if getattr(self, "_is_dark", False) else "#4361ee"
        _on = [True]
        _count = [0]

        def _tick():
            if _on[0]:
                item.setForeground(QBrush(QColor(accent)))
            else:
                item.setForeground(QBrush())
            _on[0] = not _on[0]
            _count[0] += 1
            if _count[0] >= flashes * 2:
                item.setForeground(QBrush())

        _t = QTimer(self)
        _t.setInterval(300)
        _t.timeout.connect(_tick)
        _t.setSingleShot(False)
        _count_max = flashes * 2

        def _stop_when_done():
            if _count[0] >= _count_max:
                _t.stop()
                _t.deleteLater()
        _t.timeout.connect(_stop_when_done)
        _t.start()

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
        self.report_section_list.setFixedWidth(220)
        self.report_section_list.setHeaderHidden(True)
        self.report_section_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.report_section_list.setIndentation(12)
        self.report_section_list.header().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents)
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
                help_btn.setObjectName("help_btn")
                help_btn.setText("?")
                help_btn.setMaximumWidth(24)
                help_btn.setMaximumHeight(24)
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
            export_sec_btn.setMaximumWidth(130)
            export_sec_btn.setMinimumHeight(26)
            export_sec_btn.setToolTip(f"Export the {sec} section as CSV or text")
            export_sec_btn.clicked.connect(lambda _, s=sec: self._export_section(s))
            btn_row.addWidget(export_sec_btn)
            copy_btn = QPushButton("Copy Table")
            copy_btn.setMaximumWidth(112)
            copy_btn.setMinimumHeight(26)
            copy_btn.clicked.connect(lambda _, s=sec: self._copy_section(s))
            btn_row.addWidget(copy_btn)
            vb.addLayout(btn_row)
            browser = QTextBrowser()
            _install_beer_link_filter(browser, self._on_report_link_clicked)
            vb.addWidget(browser)
            return tab, browser

        # Summary / Overview as the first report section (replaces the old Summary tab).
        _ov_leaf = QTreeWidgetItem(["Summary"])
        _ov_leaf.setData(0, Qt.ItemDataRole.UserRole, "Summary")
        _ov_leaf.setToolTip(0, "One-page overview of key results")
        self.report_section_list.addTopLevelItem(_ov_leaf)
        _ov_tab = QWidget()
        _ov_vb = QVBoxLayout(_ov_tab)
        _ov_vb.setContentsMargins(16, 12, 16, 12)
        self._summary_tab_browser = QTextBrowser()
        self._summary_tab_browser.setOpenExternalLinks(False)
        self._summary_tab_browser.setObjectName("summary_tab_browser")
        self._summary_tab_browser.setHtml(
            "<div style='font-family:sans-serif;color:#888;padding:40px;text-align:center'>"
            "<p style='font-size:15px'>Run analysis to see the protein summary.</p></div>")
        _ov_vb.addWidget(self._summary_tab_browser, 1)
        self.report_stack.addWidget(_ov_tab)
        self.report_section_tabs["Summary"] = self._summary_tab_browser
        self._report_sec_to_idx["Summary"] = _stack_idx
        _stack_idx += 1

        for group_name, group_secs in _REPORT_SECTION_GROUPS:
            grp_item = QTreeWidgetItem([f"▶  {group_name}"])
            grp_item.setFont(0, bold_font)
            grp_item.setFlags(grp_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self.report_section_list.addTopLevelItem(grp_item)
            for sec in group_secs:
                if sec not in REPORT_SECTIONS:
                    continue
                leaf = QTreeWidgetItem([sec])
                leaf.setData(0, Qt.ItemDataRole.UserRole, sec)
                leaf.setToolTip(0, sec)
                grp_item.addChild(leaf)
                tab, browser = _build_section_widget(sec)
                self.report_stack.addWidget(tab)
                self.report_section_tabs[sec] = browser
                self._report_sec_to_idx[sec] = _stack_idx
                _stack_idx += 1
            grp_item.setExpanded(False)

        for sec in REPORT_SECTIONS:
            if sec not in _grouped_secs:
                leaf = QTreeWidgetItem([sec])
                leaf.setData(0, Qt.ItemDataRole.UserRole, sec)
                leaf.setToolTip(0, sec)
                self.report_section_list.addTopLevelItem(leaf)
                tab, browser = _build_section_widget(sec)
                self.report_stack.addWidget(tab)
                self.report_section_tabs[sec] = browser
                self._report_sec_to_idx[sec] = _stack_idx
                _stack_idx += 1

        # ── AI Predictions dynamic group (populated lazily per-head) ──────────
        self._ai_pred_grp_item = QTreeWidgetItem(["▶  AI Feature Predictions"])
        self._ai_pred_grp_item.setFont(0, bold_font)
        self._ai_pred_grp_item.setFlags(
            self._ai_pred_grp_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        self.report_section_list.addTopLevelItem(self._ai_pred_grp_item)
        self._ai_pred_grp_item.setHidden(True)
        self._ai_pred_section_keys: list[str] = []
        self.report_section_list.itemClicked.connect(self._on_report_section_clicked)
        self.report_section_list.itemExpanded.connect(self._on_report_tree_expanded)
        self.report_section_list.itemCollapsed.connect(self._on_report_tree_collapsed)
        self.report_section_list.setCurrentItem(
            self.report_section_list.topLevelItem(0).child(0)
            if self.report_section_list.topLevelItem(0) else None)

        # ── Sub-tab widget: Report | Alanine Scan ─────────────────────────
        self._right_tabs = QTabWidget()
        self._right_tabs.addTab(report_panel, "Report")
        self._right_tabs.addTab(self._build_alanine_scan_panel(), "Alanine Scan")
        vbox.addWidget(self._right_tabs, 1)

        # (Export Complete Report removed in v2.0 — use per-section Export buttons)

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
            "<style>body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;font-size:12px}}"
            "table{{border-collapse:collapse;width:100%}}"
            "th,td{{border:1px solid #e2e8f0;padding:4px 8px;text-align:center}}"
            "th{{background:#4361ee;color:#ffffff;font-weight:600}}</style>"
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
        larks    = data.get("larks") or []
        gpi      = data.get("gpi_result") or {}
        motifs   = data.get("motifs") or {}
        pfam     = getattr(self, "pfam_domains", []) or []

        _dark    = getattr(self, "_is_dark", False)
        _accent  = "#7b9cff" if _dark else "#4361ee"
        _border  = "#2d3561" if _dark else "#e2e8f0"
        _muted   = "#94a3b8" if _dark else "#64748b"
        _fg      = "#e2e8f0" if _dark else "#1a1a2e"

        def _sec(title, items):
            if not items:
                return ""
            lis = "".join(f"<li style='margin:3px 0'>{i}</li>" for i in items)
            return (
                f"<h3 style='margin:14px 0 4px;color:{_accent};font-size:13px;"
                f"border-bottom:1px solid {_border};padding-bottom:3px'>{title}</h3>"
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
            f"<h2 style='color:{_accent};margin:0 0 4px'>{_html_mod.escape(self._display_name())}</h2>"
            f"<p style='color:{_muted};font-size:11px;margin:0 0 10px'>"
            f"BEER v3.0 · AI Predictions analysis</p>"
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
        self.graph_tree.setFixedWidth(220)
        self.graph_tree.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.graph_tree.setIndentation(14)
        self.graph_tree.header().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents)

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
        self._graphs_clear_tracks_btn = QPushButton("✕ Clear Tracks")
        self._graphs_clear_tracks_btn.setMaximumHeight(26)
        self._graphs_clear_tracks_btn.setToolTip("Remove UniProt annotation overlays from all graphs.")
        self._graphs_clear_tracks_btn.setEnabled(False)
        self._graphs_clear_tracks_btn.clicked.connect(self._clear_uniprot_features)
        top_bar.addWidget(self._graphs_clear_tracks_btn)
        right_v.addLayout(top_bar)
        self._roi_start: int | None = None
        self._roi_end:   int | None = None

        self.graph_stack = QStackedWidget()
        right_v.addWidget(self.graph_stack, 1)

        # Structure source selector — persistent bar below graph, visible only in overlay mode
        self._struct_src_bar = QWidget()
        _src_row = QHBoxLayout(self._struct_src_bar)
        _src_row.setContentsMargins(4, 2, 4, 2)
        self._graph_src_lbl = QLabel("Structure source:")
        self._graph_src_lbl.setObjectName("roi_lbl")
        _src_row.addWidget(self._graph_src_lbl)
        self._graph_src_combo = QComboBox()
        self._graph_src_combo.addItems(["AlphaFold", "ESMFold2", "Both"])
        self._graph_src_combo.setMaximumHeight(26)
        self._graph_src_combo.setMaximumWidth(130)
        self._graph_src_combo.setToolTip(
            "Which structure to use for structure-derived graphs\n"
            "(pLDDT, SASA, Ramachandran, Distance Map).\n"
            "Only active when both AlphaFold and ESMFold2 are loaded."
        )
        self._graph_src_combo.currentTextChanged.connect(self._on_graph_struct_src_changed)
        _src_row.addWidget(self._graph_src_combo)
        _src_row.addStretch()
        self._struct_src_bar.setVisible(False)
        right_v.addWidget(self._struct_src_bar)

        # ── Populate tree and stack ──────────────────────────────────────────
        self.graph_tabs = {}
        self._graph_title_to_stack_idx: dict = {}
        bold_font = QFont()
        bold_font.setBold(True)
        bold_font.setPointSize(10)

        for category, titles in GRAPH_CATEGORIES:
            cat_item = QTreeWidgetItem([f"▶  {category}"])
            cat_item.setFont(0, bold_font)
            cat_item.setFlags(cat_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self.graph_tree.addTopLevelItem(cat_item)

            for title in titles:
                leaf = QTreeWidgetItem([f"  {title}"])
                leaf.setData(0, Qt.ItemDataRole.UserRole, title)
                leaf.setToolTip(0, title)
                cat_item.addChild(leaf)

                if title == "Feature Overlay":
                    panel, vb = self._build_overlay_tab()
                elif title == "Feature Correlation":
                    panel, vb = self._build_correlation_tab()
                else:
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

            cat_item.setExpanded(False)

        self.graph_tree.itemClicked.connect(self._on_graph_tree_clicked)
        self.graph_tree.itemExpanded.connect(self._on_graph_tree_expanded)
        self.graph_tree.itemCollapsed.connect(self._on_graph_tree_collapsed)
        # Select first graph
        first_cat = self.graph_tree.topLevelItem(0)
        if first_cat and first_cat.childCount():
            first_leaf = first_cat.child(0)
            self.graph_tree.setCurrentItem(first_leaf)
            self.graph_stack.setCurrentIndex(0)

    # ── colour-scheme options per colour mode ────────────────────────────────
    # One shared colormap set for every mode that offers a colormap choice and
    # renders through the JS _schemeColor() path. Every name here MUST have a
    # matching case in struct_viewer.html _schemeColor().
    _STRUCT_COLORMAPS = [
        "Viridis", "Plasma", "Rainbow", "Sinebow",
        "Red-White-Blue", "Blue-White-Red", "Greyscale", "Thermal", "Fire",
    ]
    _STRUCT_SCHEMES = {
        "pLDDT / B-factor":    ["Red-White-Blue", "Blue-White-Red", "Rainbow", "Sinebow", "Greyscale"],
        "Residue Type":         ["Amino Acid (UniProt)", "Shapely"],
        "Chain":                ["Chain Colors"],
        "Charge":               _STRUCT_COLORMAPS,
        "Hydrophobicity":       _STRUCT_COLORMAPS,
        "Mass":                 _STRUCT_COLORMAPS,
        "Secondary Structure":  ["JMol", "PyMOL", "Pastel", "Lesk", "Cinema", "Vivid"],
        "Residue Number":       _STRUCT_COLORMAPS,
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
        "Residue Number":            "spectrum",
        "Solvent Accessibility":     "sasa",
        "AI Features":               "feature",
        "Aggregation (ZYGGREGATOR)": "zyggregator",
    }
    _AI_GRADIENT_MAP = {
        "Plasma (Purple→Yellow)":   "plasma",
        "Viridis (Purple→Lime)":    "viridis",
        "Blue→Red (Diverging)":     "bwr",
        "Teal→Orange":              "RdYlGn_r",
        "Green→Purple":             "PRGn_r",
        "Fire (Black→Yellow)":      "afmhot",
        "Hot (Black→Red→White)":    "hot",
        "Cold (White→Blue)":        "Blues",
        "Coolwarm":                 "coolwarm",
    }
    _AI_COLOR_ITEMS = [
        ("Orange",  "#f3722c"),
        ("Cyan",    "#00b4d8"),
        ("Magenta", "#f72585"),
        ("Green",   "#2dc653"),
        ("Blue",    "#3a86ff"),
        ("Red",     "#e63946"),
        ("Custom",  None),
    ]
    def init_structure_tab(self):
        """Tab for interactive 3D structure viewer (PDB upload, RCSB PDB fetch, or AlphaFold fetch)."""
        from PySide6.QtWidgets import (
            QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
            QCheckBox, QRadioButton, QScrollArea, QLineEdit, QSlider, QFrame,
            QButtonGroup, QSizePolicy, QStyleFactory,
        )
        from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve
        from PySide6.QtGui import QColor
        from beer.gui.themes import STRUCT_PANEL_CSS_LIGHT, STRUCT_PANEL_CSS_DARK

        container = QWidget()
        layout    = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        self.main_tabs.addTab(container, "Structure")

        # ── top info row ──────────────────────────────────────────────────────────
        info_row = QHBoxLayout()
        self.af_status_lbl = QLabel(
            "No structure loaded.  Import a PDB file, fetch a PDB ID, or fetch AlphaFold.")
        self.af_status_lbl.setObjectName("status_lbl")
        self.af_status_lbl.setProperty("status_state", "idle")
        info_row.addWidget(self.af_status_lbl, 1)
        self.export_structure_btn = QPushButton("Export…")
        self.export_structure_btn.setObjectName("secondary_btn")
        self.export_structure_btn.setToolTip(
            "Export as PDB, mmCIF, GRO, XYZ (requires loaded structure) or FASTA (requires analysis)")
        self.export_structure_btn.clicked.connect(self.export_structure_dialog)
        info_row.addWidget(self.export_structure_btn)
        layout.addLayout(info_row)

        # ── guard: WebEngine may be absent ───────────────────────────────────────
        try:
            from PySide6.QtWebEngineWidgets import QWebEngineView as _QWEView
            _webengine_ok = True
        except ImportError:
            _webengine_ok = False

        if not _webengine_ok:
            msg = QLabel(
                "The 3D viewer needs Qt WebEngine, which is not installed.\n"
                "Install it with:  pip install PySide6-Addons   "
                "(or: pip install PySide6-Addons)\n\n"
                "You can still export the structure and open it in PyMOL or UCSF ChimeraX.")
            msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
            msg.setObjectName("placeholder_lbl")
            layout.addWidget(msg, 1)
            self.structure_viewer = None
            return

        # ══════════════════════════════════════════════════════════════════════════
        # Helper: collapsible card section
        # ══════════════════════════════════════════════════════════════════════════
        def _make_card_section(title: str, content_widget: QWidget,
                               expanded: bool = False) -> tuple[QWidget, QWidget]:
            """Return (wrapper, content_widget).

            wrapper  — the outer card that goes into the panel layout
            content  — the passed-in widget whose visibility is toggled
            """
            # Card shell
            card = QWidget()
            card.setProperty("class", "card")    # for CSS QWidget.card
            card_vbox = QVBoxLayout(card)
            card_vbox.setContentsMargins(0, 0, 0, 0)
            card_vbox.setSpacing(0)

            # Toggle button (full-width section header)
            arrow = "▼" if expanded else "▶"
            hdr = QPushButton(f" {arrow}  {title.upper()}")
            hdr.setProperty("class", "section_toggle")
            hdr.setCursor(Qt.CursorShape.PointingHandCursor)
            hdr.setCheckable(False)

            # Content area
            content_widget.setVisible(expanded)
            inner = QWidget()
            inner.setVisible(expanded)
            inner_vbox = QVBoxLayout(inner)
            inner_vbox.setContentsMargins(10, 4, 10, 8)
            inner_vbox.setSpacing(5)
            inner_vbox.addWidget(content_widget)

            def _toggle():
                vis = not inner.isVisible()
                inner.setVisible(vis)
                content_widget.setVisible(vis)
                hdr.setText(f" {'▼' if vis else '▶'}  {title.upper()}")

            hdr.clicked.connect(_toggle)
            card_vbox.addWidget(hdr)
            card_vbox.addWidget(inner)
            return card, content_widget

        # ══════════════════════════════════════════════════════════════════════════
        # Left scroll panel
        # ══════════════════════════════════════════════════════════════════════════
        _ctrl_page = QWidget()
        _ctrl_page.setObjectName("structCtrl")
        _ctrl_vbox = QVBoxLayout(_ctrl_page)
        _ctrl_vbox.setContentsMargins(6, 8, 6, 8)
        _ctrl_vbox.setSpacing(6)

        self.struct_ctrl_scroll = QScrollArea()
        self.struct_ctrl_scroll.setFixedWidth(260)
        self.struct_ctrl_scroll.setWidgetResizable(True)
        self.struct_ctrl_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.struct_ctrl_scroll.setWidget(_ctrl_page)
        _fusion = QStyleFactory.create("Fusion")
        if _fusion:
            self.struct_ctrl_scroll.setStyle(_fusion)
        # Apply theme CSS
        _css = STRUCT_PANEL_CSS_DARK if getattr(self, "_is_dark", False) else STRUCT_PANEL_CSS_LIGHT
        self.struct_ctrl_scroll.setStyleSheet(_css)

        vl = _ctrl_vbox   # alias for brevity

        # ── 1. RESET VIEW (always visible, top of panel) ─────────────────────────
        _rst_btn = QPushButton("↺  Reset View")
        _rst_btn.setObjectName("struct_reset_btn")
        _rst_btn.setToolTip("Reset representation, colour, background and camera to defaults")
        _rst_btn.clicked.connect(self._reset_struct_view)
        vl.addWidget(_rst_btn)

        # ── 2. REPRESENTATION ────────────────────────────────────────────────────
        rep_body = QWidget()
        rep_vbox = QVBoxLayout(rep_body)
        rep_vbox.setContentsMargins(0, 0, 0, 0)
        rep_vbox.setSpacing(5)

        self.struct_rep_combo = QComboBox()
        self.struct_rep_combo.addItems(["Cartoon", "Stick", "Sphere", "Surface"])
        self.struct_rep_combo.setToolTip("Cartoon: ribbon\nStick: bonds\nSphere: VDW\nSurface: molecular surface")
        rep_vbox.addWidget(self.struct_rep_combo)

        self.struct_rep_combo.currentTextChanged.connect(
            lambda text: self._js(f"setRepresentation('{text.lower()}');"))

        rep_card, _ = _make_card_section("Representation", rep_body, expanded=True)
        vl.addWidget(rep_card)

        # ── 3. COLOR ──────────────────────────────────────────────────────────────
        color_body = QWidget()
        color_vbox = QVBoxLayout(color_body)
        color_vbox.setContentsMargins(0, 0, 0, 0)
        color_vbox.setSpacing(6)

        # Mode row
        _mode_hl = QHBoxLayout()
        _mode_hl.setSpacing(5)
        _mode_lbl = QLabel("Mode:")
        _mode_lbl.setFixedWidth(46)
        _mode_hl.addWidget(_mode_lbl)
        self.struct_color_mode_combo = QComboBox()
        self.struct_color_mode_combo.addItems(list(self._STRUCT_SCHEMES.keys()))
        self._set_tooltip(self.struct_color_mode_combo,
            "What the 3D structure is coloured by: pLDDT/B-factor confidence, "
            "residue type, chain, charge, hydrophobicity, mass, secondary structure, "
            "residue number, solvent accessibility, a computed AI feature, or "
            "aggregation propensity.")
        _mode_hl.addWidget(self.struct_color_mode_combo, 1)
        color_vbox.addLayout(_mode_hl)

        # Scheme row
        _scheme_hl = QHBoxLayout()
        _scheme_hl.setSpacing(5)
        self.struct_scheme_lbl = QLabel("Scheme:")
        self.struct_scheme_lbl.setFixedWidth(46)
        _scheme_hl.addWidget(self.struct_scheme_lbl)
        self.struct_scheme_combo = QComboBox()
        self._set_tooltip(self.struct_scheme_combo,
            "Colour palette / scheme applied within the selected colour mode "
            "(e.g. Red-White-Blue, Rainbow, Viridis, JMol). Options change with "
            "the colour mode.")
        _scheme_hl.addWidget(self.struct_scheme_combo, 1)
        color_vbox.addLayout(_scheme_hl)

        # ── AI Features sub-panel ────────────────────────────────────────────────
        self._struct_ai_color      = "#f3722c"
        self._struct_ai_color_mode = "gradient"   # 'gradient' | 'binary'

        self._ai_ctrl_container = QWidget()
        self._ai_ctrl_container.setMaximumHeight(0)
        self._ai_ctrl_container.setVisible(False)
        _ai_vbox = QVBoxLayout(self._ai_ctrl_container)
        _ai_vbox.setContentsMargins(0, 2, 0, 0)
        _ai_vbox.setSpacing(4)

        # Gradient / Binary toggle row
        _ai_style_hl = QHBoxLayout()
        _ai_style_hl.setSpacing(0)
        _ai_style_lbl = QLabel("Style:")
        _ai_style_lbl.setFixedWidth(46)
        _ai_style_hl.addWidget(_ai_style_lbl)
        self._ai_grad_btn = QPushButton("Gradient")
        self._ai_grad_btn.setCheckable(True)
        self._ai_grad_btn.setChecked(True)
        self._ai_grad_btn.setFixedHeight(22)
        self._ai_grad_btn.setProperty("class", "mode_btn")
        self._ai_bin_btn = QPushButton("Binary")
        self._ai_bin_btn.setCheckable(True)
        self._ai_bin_btn.setChecked(False)
        self._ai_bin_btn.setFixedHeight(22)
        self._ai_bin_btn.setProperty("class", "mode_btn")
        self._ai_style_grp = QButtonGroup(self)
        self._ai_style_grp.setExclusive(True)
        self._ai_style_grp.addButton(self._ai_grad_btn, 0)
        self._ai_style_grp.addButton(self._ai_bin_btn,  1)
        self._ai_style_grp.buttonClicked.connect(self._on_ai_style_toggled)
        _ai_style_hl.addWidget(self._ai_grad_btn)
        _ai_style_hl.addWidget(self._ai_bin_btn)
        _ai_style_hl.addStretch()
        _ai_vbox.addLayout(_ai_style_hl)

        # Colormap dropdown (gradient mode)
        _grad_hl = QHBoxLayout()
        _grad_hl.setSpacing(5)
        self.struct_ai_gradient_lbl = QLabel("Colormap:")
        self.struct_ai_gradient_lbl.setFixedWidth(56)
        _grad_hl.addWidget(self.struct_ai_gradient_lbl)
        self.struct_ai_gradient_combo = QComboBox()
        self.struct_ai_gradient_combo.addItems(list(self._AI_GRADIENT_MAP.keys()))
        self.struct_ai_gradient_combo.setCurrentText("Plasma (Purple→Yellow)")
        self.struct_ai_gradient_combo.currentTextChanged.connect(
            self._on_ai_gradient_combo_changed)
        _grad_hl.addWidget(self.struct_ai_gradient_combo, 1)
        _ai_vbox.addLayout(_grad_hl)

        # Color dropdown (binary mode)
        _bin_hl = QHBoxLayout()
        _bin_hl.setSpacing(5)
        self.struct_ai_color_lbl = QLabel("Color:")
        self.struct_ai_color_lbl.setFixedWidth(56)
        _bin_hl.addWidget(self.struct_ai_color_lbl)
        self.struct_ai_color_combo = QComboBox()
        for _cn, _ in self._AI_COLOR_ITEMS:
            self.struct_ai_color_combo.addItem(_cn)
        self.struct_ai_color_combo.currentTextChanged.connect(
            self._on_ai_color_combo_changed)
        _bin_hl.addWidget(self.struct_ai_color_combo, 1)
        _ai_vbox.addLayout(_bin_hl)

        # Initially hide binary controls (gradient is default)
        self.struct_ai_color_lbl.setVisible(False)
        self.struct_ai_color_combo.setVisible(False)

        color_vbox.addWidget(self._ai_ctrl_container)

        # Wire mode/scheme combo signals
        self.struct_color_mode_combo.currentTextChanged.connect(
            self._on_struct_color_mode_changed)
        self.struct_scheme_combo.currentTextChanged.connect(
            self._on_struct_scheme_changed)
        # Populate scheme combo for the default mode and set initial visibility
        self._update_scheme_combo(self.struct_color_mode_combo.currentText())

        # Color bar toggle (belongs with Color controls)
        self.struct_colorbar_cb = QCheckBox("Show color bar")
        self.struct_colorbar_cb.setChecked(True)
        self.struct_colorbar_cb.toggled.connect(self._on_struct_colorbar_toggled)
        color_vbox.addWidget(self.struct_colorbar_cb)

        color_card, _ = _make_card_section("Color", color_body, expanded=True)
        vl.addWidget(color_card)

        # ── 4. OVERLAY (shown only when overlay loaded) ───────────────────────────
        # NOTE: _overlay_row is assigned later (after _make_card_section) to the card widget
        # so that main_window._update_overlay_controls(.setVisible) works on the real layout item.
        _ov_body = QWidget()
        _ov_vbox = QVBoxLayout(_ov_body)
        _ov_vbox.setContentsMargins(0, 0, 0, 0)
        _ov_vbox.setSpacing(5)

        # AF / ESM checkboxes
        _ov_chk_hl = QHBoxLayout()
        _ov_chk_hl.setSpacing(8)
        self._overlay_af_chk = QCheckBox("AlphaFold")
        self._overlay_af_chk.setChecked(True)
        self._overlay_af_chk.setToolTip("Toggle AlphaFold structure visibility in overlay")
        self._overlay_af_chk.toggled.connect(self._on_overlay_af_toggled)
        self._overlay_esm_chk = QCheckBox("ESMFold2")
        self._overlay_esm_chk.setChecked(True)
        self._overlay_esm_chk.setToolTip("Toggle ESMFold2 structure visibility in overlay")
        self._overlay_esm_chk.toggled.connect(self._on_overlay_esm_toggled)
        _ov_chk_hl.addWidget(self._overlay_af_chk)
        _ov_chk_hl.addWidget(self._overlay_esm_chk)
        _ov_chk_hl.addStretch()
        _ov_vbox.addLayout(_ov_chk_hl)

        # Align-on radio buttons
        _ov_align_hl = QHBoxLayout()
        _ov_align_hl.setSpacing(4)
        _ov_align_lbl = QLabel("Align on:")
        _ov_align_hl.addWidget(_ov_align_lbl)
        self._align_all_radio = QRadioButton("All Cα")
        self._align_all_radio.setChecked(True)
        self._align_all_radio.setToolTip("Align all Cα atoms (default)")
        self._align_struct_radio = QRadioButton("Structured")
        self._align_struct_radio.setToolTip(
            "Align using helices and sheets only (parsed from HELIX/SHEET PDB records;\n"
            "falls back to pLDDT > 70 if records are absent)")
        self._align_all_radio.toggled.connect(self._on_align_mode_changed)
        _ov_align_hl.addWidget(self._align_all_radio)
        _ov_align_hl.addWidget(self._align_struct_radio)
        _ov_align_hl.addStretch()
        _ov_vbox.addLayout(_ov_align_hl)

        _ov_card, _ = _make_card_section("Overlay", _ov_body, expanded=True)
        # Keep both old name (_overlay_row — used by main_window._update_overlay_controls)
        # and new name (_overlay_card) pointing to the same widget so both code paths work.
        self._overlay_card = _ov_card
        self._overlay_row  = _ov_card   # backward-compat alias for main_window._update_overlay_controls
        _ov_card.setVisible(False)    # hidden until overlay loads
        vl.addWidget(_ov_card)

        # ── 5. INTERACTIONS ───────────────────────────────────────────────────────
        inter_body = QWidget()
        inter_vbox = QVBoxLayout(inter_body)
        inter_vbox.setContentsMargins(0, 0, 0, 0)
        inter_vbox.setSpacing(5)

        self.struct_hbond_cb = QCheckBox("Show H-bonds")
        self.struct_hbond_cb.setToolTip(
            "Backbone N–H···O bonds (N–O < 3.5 Å, non-adjacent residues).")
        self.struct_hbond_cb.toggled.connect(
            lambda on: self._js(f"toggleHBonds({'true' if on else 'false'});"))
        inter_vbox.addWidget(self.struct_hbond_cb)

        inter_card, _ = _make_card_section("Interactions", inter_body, expanded=False)
        vl.addWidget(inter_card)

        # ── 6. MEASUREMENTS ──────────────────────────────────────────────────────
        meas_body = QWidget()
        meas_vbox = QVBoxLayout(meas_body)
        meas_vbox.setContentsMargins(0, 0, 0, 0)
        meas_vbox.setSpacing(6)

        _meas_hint = QLabel("Click atoms on the structure after activating a mode.")
        _meas_hint.setObjectName("struct_hint")
        _meas_hint.setWordWrap(True)
        meas_vbox.addWidget(_meas_hint)

        # Exclusive mode pill buttons: Distance | Angle | Dihedral
        _meas_pill_hl = QHBoxLayout()
        _meas_pill_hl.setSpacing(0)
        _meas_pill_hl.setContentsMargins(0, 0, 0, 0)
        self.struct_dist_btn    = QPushButton("Distance")
        self.struct_angle_btn   = QPushButton("Angle")
        self.struct_dihedral_btn = QPushButton("Dihedral")
        for _mb in (self.struct_dist_btn, self.struct_angle_btn, self.struct_dihedral_btn):
            _mb.setCheckable(True)
            _mb.setProperty("class", "mode_btn")
            _mb.setFixedHeight(26)
        _meas_mode_grp = QButtonGroup(self)
        _meas_mode_grp.setExclusive(False)   # we handle exclusivity manually for toggle-off
        _meas_mode_grp.addButton(self.struct_dist_btn,     0)
        _meas_mode_grp.addButton(self.struct_angle_btn,    1)
        _meas_mode_grp.addButton(self.struct_dihedral_btn, 2)
        self._meas_mode_grp = _meas_mode_grp
        _meas_pill_hl.addWidget(self.struct_dist_btn)
        _meas_pill_hl.addWidget(self.struct_angle_btn)
        _meas_pill_hl.addWidget(self.struct_dihedral_btn)
        _meas_pill_hl.addStretch()
        meas_vbox.addLayout(_meas_pill_hl)

        self._meas_active_btn = None   # track currently active mode button

        _mode_labels = {
            self.struct_dist_btn:     "distance",
            self.struct_angle_btn:    "angle",
            self.struct_dihedral_btn: "dihedral",
        }

        def _on_meas_btn_clicked(btn: QPushButton) -> None:
            """Toggle exclusive measurement mode with re-click-to-off."""
            if btn is self._meas_active_btn:
                # same button: toggle off
                btn.setChecked(False)
                self._meas_active_btn = None
                self._js("exitDistanceMode();")
            else:
                # switch to new mode
                if self._meas_active_btn is not None:
                    self._meas_active_btn.setChecked(False)
                btn.setChecked(True)
                self._meas_active_btn = btn
                mode = _mode_labels[btn]
                self._js("enterDistanceMode();")
                self._js(f"setMeasureMode('{mode}');")

        for _mb in (self.struct_dist_btn, self.struct_angle_btn, self.struct_dihedral_btn):
            _mb.clicked.connect(lambda checked, b=_mb: _on_meas_btn_clicked(b))

        meas_clear_btn = QPushButton("Clear Measurements")
        meas_clear_btn.setToolTip("Remove all measurement labels and lines")
        meas_clear_btn.clicked.connect(lambda: self._js("clearDistances();"))
        meas_vbox.addWidget(meas_clear_btn)

        meas_card, _ = _make_card_section("Measurements", meas_body, expanded=False)
        vl.addWidget(meas_card)

        # ── 7. SELECTION ─────────────────────────────────────────────────────────
        sel_body = QWidget()
        sel_vbox = QVBoxLayout(sel_body)
        sel_vbox.setContentsMargins(0, 0, 0, 0)
        sel_vbox.setSpacing(5)

        _sel_hint = QLabel("e.g.  45  ·  10-50  ·  LEU  ·  A:10-50")
        _sel_hint.setObjectName("struct_hint")
        sel_vbox.addWidget(_sel_hint)

        _sel_row_hl = QHBoxLayout()
        _sel_row_hl.setSpacing(4)
        self.struct_sel_edit = QLineEdit()
        self.struct_sel_edit.setPlaceholderText("number, range, or residue name")
        self.struct_sel_edit.setToolTip(
            "Select residues by:\n"
            "  number:    45\n"
            "  range:     10-50\n"
            "  name:      LEU\n"
            "  chain:     A:10-50\n"
            "  multiple:  45, LEU, A:100-120\n\n"
            "Press Enter or click Go.")
        self.struct_sel_edit.returnPressed.connect(self._on_struct_selection_apply)
        _sel_row_hl.addWidget(self.struct_sel_edit, 1)
        _sel_go_btn = QPushButton("Go")
        _sel_go_btn.setFixedWidth(38)
        _sel_go_btn.setFixedHeight(28)
        _sel_go_btn.setToolTip("Apply selection")
        _sel_go_btn.clicked.connect(self._on_struct_selection_apply)
        _sel_row_hl.addWidget(_sel_go_btn)
        sel_vbox.addLayout(_sel_row_hl)

        _sel_btn_hl = QHBoxLayout()
        _sel_btn_hl.setSpacing(4)
        _sel_focus_btn = QPushButton("Focus")
        _sel_focus_btn.setToolTip("Zoom the camera to the current selection")
        _sel_focus_btn.clicked.connect(lambda: self._js("focusSelection();"))
        _sel_btn_hl.addWidget(_sel_focus_btn)
        _sel_clear_btn = QPushButton("Clear Selection")
        _sel_clear_btn.setToolTip("Deselect all highlighted residues")
        _sel_clear_btn.clicked.connect(self._on_struct_selection_clear)
        _sel_btn_hl.addWidget(_sel_clear_btn)
        sel_vbox.addLayout(_sel_btn_hl)

        self._sel_count_lbl = QLabel("")
        self._sel_count_lbl.setObjectName("struct_count")
        sel_vbox.addWidget(self._sel_count_lbl)

        sel_card, _ = _make_card_section("Selection", sel_body, expanded=False)
        vl.addWidget(sel_card)

        # ── 7b. DISPLAY (native Mol* extras) ──────────────────────────────────────
        disp_body = QWidget()
        disp_vbox = QVBoxLayout(disp_body)
        disp_vbox.setContentsMargins(0, 0, 0, 0)
        disp_vbox.setSpacing(5)

        # Sequence panel + biological assembly toggles
        self.struct_seq_cb = QCheckBox("Show sequence panel")
        self.struct_seq_cb.setToolTip("Native clickable Mol* sequence bar (top of viewer)")
        self.struct_seq_cb.toggled.connect(
            lambda on: self._js(f"setSequenceVisible({'true' if on else 'false'});"))
        disp_vbox.addWidget(self.struct_seq_cb)

        self.struct_assembly_cb = QCheckBox("Biological assembly")
        self.struct_assembly_cb.setToolTip("Show assembly 1 (oligomeric state) vs the asymmetric unit")
        self.struct_assembly_cb.toggled.connect(
            lambda on: self._js(f"setAssembly({'true' if on else 'false'});"))
        disp_vbox.addWidget(self.struct_assembly_cb)

        # Spin / Rock animation pills
        _anim_hl = QHBoxLayout()
        _anim_hl.setSpacing(0)
        _anim_lbl = QLabel("Animate:")
        _anim_lbl.setFixedWidth(54)
        _anim_hl.addWidget(_anim_lbl)
        self.struct_spin_btn = QPushButton("Spin")
        self.struct_rock_btn = QPushButton("Rock")
        for _ab in (self.struct_spin_btn, self.struct_rock_btn):
            _ab.setCheckable(True)
            _ab.setProperty("class", "mode_btn")
            _ab.setFixedHeight(24)
        self._anim_grp = QButtonGroup(self)
        self._anim_grp.setExclusive(False)   # manual toggle-off + mutual exclusion
        self._anim_grp.addButton(self.struct_spin_btn, 0)
        self._anim_grp.addButton(self.struct_rock_btn, 1)
        self.struct_spin_btn.clicked.connect(lambda: self._on_struct_anim_clicked("spin"))
        self.struct_rock_btn.clicked.connect(lambda: self._on_struct_anim_clicked("rock"))
        _anim_hl.addWidget(self.struct_spin_btn)
        _anim_hl.addWidget(self.struct_rock_btn)
        _anim_hl.addStretch()
        disp_vbox.addLayout(_anim_hl)

        # Hetero component visibility (ligand / water / ion)
        _het_lbl = QLabel("Show heteroatoms:")
        disp_vbox.addWidget(_het_lbl)
        _het_hl = QHBoxLayout()
        _het_hl.setSpacing(8)
        self.struct_lig_cb = QCheckBox("Ligands")
        self.struct_wat_cb = QCheckBox("Water")
        self.struct_ion_cb = QCheckBox("Ions")
        self.struct_lig_cb.setChecked(True)   # ligands shown by default
        for _cb, _kind in ((self.struct_lig_cb, "ligand"),
                           (self.struct_wat_cb, "water"),
                           (self.struct_ion_cb, "ion")):
            _cb.toggled.connect(
                lambda on, k=_kind: self._js(f"setComponentVisible('{k}',{'true' if on else 'false'});"))
            _het_hl.addWidget(_cb)
        _het_hl.addStretch()
        disp_vbox.addLayout(_het_hl)

        disp_card, _ = _make_card_section("Display", disp_body, expanded=False)
        vl.addWidget(disp_card)

        # ── 8. CHAINS (shown only for multi-chain structures) ─────────────────────
        chains_body = QWidget()
        chains_vbox = QVBoxLayout(chains_body)
        chains_vbox.setContentsMargins(0, 0, 0, 0)
        chains_vbox.setSpacing(5)

        _chain_info = QLabel("Toggle individual chain visibility.")
        _chain_info.setObjectName("struct_hint")
        _chain_info.setWordWrap(True)
        chains_vbox.addWidget(_chain_info)

        _chain_btn_hl = QHBoxLayout()
        _chain_btn_hl.setSpacing(4)
        _chain_all_btn = QPushButton("Show All")
        _chain_all_btn.setFixedHeight(26)
        _chain_all_btn.clicked.connect(self._show_all_chains)
        _chain_none_btn = QPushButton("Hide All")
        _chain_none_btn.setFixedHeight(26)
        _chain_none_btn.clicked.connect(self._hide_all_chains)
        _chain_btn_hl.addWidget(_chain_all_btn)
        _chain_btn_hl.addWidget(_chain_none_btn)
        chains_vbox.addLayout(_chain_btn_hl)

        self._chain_cbs_widget = QWidget()
        self._chain_cbs_layout = QVBoxLayout(self._chain_cbs_widget)
        self._chain_cbs_layout.setContentsMargins(0, 0, 0, 0)
        self._chain_cbs_layout.setSpacing(2)
        chains_vbox.addWidget(self._chain_cbs_widget)
        self._chain_checkboxes: dict = {}

        _chains_card, _ = _make_card_section("Chains", chains_body, expanded=True)
        # Keep references so _populate_chain_controls can show/hide and auto-expand.
        # Also set backward-compat aliases used by main_window._populate_chain_controls:
        #   self._chains_accordion  → the collapsible wrapper (used for .setVisible and layout ops)
        #   self._chains_grp_box    → the inner body widget (used for .isVisible check)
        self._chains_card      = _chains_card
        self._chains_accordion = _chains_card    # backward-compat alias
        self._chains_grp_box   = chains_body     # backward-compat alias (inner content widget)
        self._chains_card.setVisible(False)   # hidden until a multi-chain protein loads
        vl.addWidget(_chains_card)

        # ── 9. VIEW (Background + Color Bar + Snapshot) ───────────────────────────
        view_body = QWidget()
        view_vbox = QVBoxLayout(view_body)
        view_vbox.setContentsMargins(0, 0, 0, 0)
        view_vbox.setSpacing(6)

        # Background color row: preset buttons
        _bg_lbl = QLabel("Background:")
        view_vbox.addWidget(_bg_lbl)
        _bg_hl = QHBoxLayout()
        _bg_hl.setSpacing(4)
        for _lbl, _hex in [("Black", "#1a1a2e"), ("White", "#ffffff"), ("Grey", "#555566")]:
            _bb = QPushButton(_lbl)
            _bb.setFixedHeight(26)
            _bb.clicked.connect(lambda _, c=_hex: self._js(f"setBackground('{c}');"))
            _bg_hl.addWidget(_bb)
        view_vbox.addLayout(_bg_hl)
        _custom_bg_btn = QPushButton("Custom color…")
        _custom_bg_btn.setFixedHeight(26)
        _custom_bg_btn.clicked.connect(self._pick_background_color)
        view_vbox.addWidget(_custom_bg_btn)

        view_card, _ = _make_card_section("View", view_body, expanded=False)
        vl.addWidget(view_card)

        # Stretch at the bottom
        vl.addStretch(1)

        # ══════════════════════════════════════════════════════════════════════════
        # Assemble content row: scroll panel + 3D viewer
        # ══════════════════════════════════════════════════════════════════════════
        content_row = QHBoxLayout()
        content_row.setSpacing(8)
        content_row.addWidget(self.struct_ctrl_scroll)

        # ── 3D viewer ─────────────────────────────────────────────────────────────
        from PySide6.QtWebEngineWidgets import QWebEngineView
        self.structure_viewer = QWebEngineView()
        self.structure_viewer.setMinimumHeight(500)
        self.structure_viewer.loadFinished.connect(self._on_structure_page_loaded)
        content_row.addWidget(self.structure_viewer, 1)

        # QWebChannel: expose Python bridge so JS can call residueClicked()
        try:
            from PySide6.QtWebChannel import QWebChannel as _QWC
            # _StructBridge must be defined in the class (already present in main_window.py)
            self._struct_bridge = _StructBridge(self)   # noqa: F821 (defined in main_window.py)
            _wc = _QWC(self.structure_viewer.page())
            _wc.registerObject("bridge", self._struct_bridge)
            self.structure_viewer.page().setWebChannel(_wc)
            self._struct_webchannel = _wc   # keep alive
        except Exception:
            self._struct_bridge = None

        # Inject bundled molstar.js at DocumentCreation
        _molstar_css_main = ""
        try:
            import os as _os
            from PySide6.QtWebEngineCore import QWebEngineScript as _WES
            _js_path  = _os.path.join(_os.path.dirname(__file__), "molstar.js")
            _css_path = _os.path.join(_os.path.dirname(__file__), "molstar.css")
            with open(_js_path,  "r", encoding="utf-8") as _f: _molstar_src = _f.read()
            with open(_css_path, "r", encoding="utf-8") as _f: _molstar_css_main = _f.read()
            _s = _WES()
            _s.setName("molstar-struct")
            _s.setSourceCode(_molstar_src)
            _s.setInjectionPoint(_WES.InjectionPoint.DocumentCreation)
            _s.setWorldId(_WES.ScriptWorldId.MainWorld)
            self.structure_viewer.page().scripts().insert(_s)
        except Exception:
            import logging as _log
            _log.getLogger("beer.gui").warning(
                "3D structure viewer (Mol*) failed to initialise — the Structure "
                "tab will be unavailable. Ensure 'PySide6-Addons' (QtWebEngine) is "
                "installed: pip install PySide6-Addons", exc_info=True)
            self._molstar_init_failed = True

        layout.addLayout(content_row, 1)

        # Load base page once — all subsequent structure swaps go via loadPDB() JS call.
        _bg = "#1a1a2e" if getattr(self, "_is_dark", False) else "#ffffff"
        _html_path = _os.path.join(_os.path.dirname(__file__), "struct_viewer.html")
        try:
            with open(_html_path, "r", encoding="utf-8") as _fh:
                _html_template = _fh.read()
        except OSError:
            _html_template = "<html><body style='background:black'></body></html>"
        self.structure_viewer.setHtml(
            _html_template.replace("__BG__", _bg).replace("__CSS__", _molstar_css_main))

    def _js(self, code: str) -> None:
        """Run JavaScript in the structure viewer (no-op if unavailable)."""
        if self.structure_viewer is None:
            return
        try:
            page = self.structure_viewer.page()
            if page is not None:
                page.runJavaScript(code)
        except RuntimeError:
            # Underlying C++ QWebEnginePage already deleted (e.g. during shutdown)
            pass

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
        "SS3 Helix":            "ss3_h_profile",
        "SS3 Strand":           "ss3_e_profile",
        "SS3 Coil":             "ss3_c_profile",
    }

    def _available_feature_schemes(self) -> list[str]:
        """Return AI feature names for heads that have already been computed."""
        ad = self.analysis_data or {}
        return [
            label for label, key in self._FEATURE_SCORE_KEYS.items()
            if ad.get(key)
        ]

    # Modes rendered by a Mol* built-in color theme (canonical palette).
    # Their scheme dropdown has no effect, so it is hidden to avoid misleading
    # the user. All other modes route through schemes that do apply.
    _STRUCT_BUILTIN_THEME_MODES = frozenset({
        "pLDDT / B-factor",
        "Residue Type",
        "Chain",
        "Secondary Structure",
    })

    def _update_scheme_combo(self, mode: str) -> None:
        if mode == "AI Features":
            schemes = self._available_feature_schemes()
        else:
            schemes = self._STRUCT_SCHEMES.get(mode, ["Default"])
        self.struct_scheme_combo.blockSignals(True)
        self.struct_scheme_combo.clear()
        self.struct_scheme_combo.addItems(schemes)
        self.struct_scheme_combo.blockSignals(False)
        # Hide the scheme row for built-in-theme modes (scheme is inert there)
        show_scheme = mode not in self._STRUCT_BUILTIN_THEME_MODES
        self.struct_scheme_lbl.setVisible(show_scheme)
        self.struct_scheme_combo.setVisible(show_scheme)


    # Per-feature F1-maximising thresholds (from training validation sets)
    _FEATURE_THRESHOLDS: dict[str, float] = {
        "Disorder":        0.56235,
        "Signal Peptide":  0.70173,
        "Transmembrane":   0.81339,
        "Intramembrane":   0.67273,
        "Coiled-Coil":     0.55178,
        "DNA-Binding":     0.87760,
        "Active Site":     0.86688,
        "Binding Site":    0.98014,
        "Phosphorylation": 0.79967,
        "Low Complexity":  0.65838,
        "Glycosylation":   0.80024,
        "Ubiquitination":  0.83320,
    }

    def _push_feature_scores(self, feature_label: str) -> None:
        """Send per-residue scores to JS for structure coloring (gradient or binary)."""
        import json as _json
        ad = self.analysis_data or {}
        key = self._FEATURE_SCORE_KEYS.get(feature_label, "disorder_scores")
        scores = ad.get(key) or []
        if not scores:
            QMessageBox.information(
                self, "AI Prediction Not Yet Computed",
                f"’{feature_label}’ has not been computed yet.\n\n"
                "Click the feature in the Graphs or Reports tab to trigger computation,\n"
                "then return here to use it for structure coloring.")
            return
        threshold  = self._FEATURE_THRESHOLDS.get(feature_label, 0.5)
        color_mode = getattr(self, "_struct_ai_color_mode", "gradient")
        if color_mode == "gradient":
            import matplotlib.cm as _cm
            import matplotlib.colors as _mc
            grad_text = self.struct_ai_gradient_combo.currentText()
            cmap_name = self._AI_GRADIENT_MAP.get(grad_text, "plasma")
            cmap      = matplotlib.colormaps[cmap_name]
            resi_colors = {i + 1: _mc.to_hex(cmap(float(v))) for i, v in enumerate(scores)}
            # build CSS gradient for color bar (sample 9 points)
            css_stops = ",".join(
                _mc.to_hex(cmap(k / 8)) for k in range(9)
            )
            meta = {
                "css":  f"linear-gradient(to top,{css_stops})",
                "min":  "0.0",
                "mid":  "0.5",
                "max":  "1.0",
                "unit": f"{feature_label} score",
            }
            self._js(
                f"setResidueColorMap({_json.dumps(resi_colors)},"
                f"{_json.dumps(feature_label)},"
                f"{_json.dumps(meta)});"
            )
        else:
            color       = getattr(self, "_struct_ai_color", "#f3722c")
            scores_dict = {i + 1: float(v) for i, v in enumerate(scores)}
            self._js(
                f"setFeatureData({_json.dumps(feature_label)},"
                f"{_json.dumps(scores_dict)},"
                f"{_json.dumps(color)},"
                f"{threshold});"
            )

    def _set_ai_feature_color(self, hex_color: str) -> None:
        """Update the highlight color for AI feature binary coloring."""
        self._struct_ai_color = hex_color
        if getattr(self, "struct_color_mode_combo", None):
            if self.struct_color_mode_combo.currentText() == "AI Features":
                scheme = self.struct_scheme_combo.currentText()
                if scheme:
                    self._push_feature_scores(scheme)

    def _pick_ai_feature_color(self) -> None:
        """Open a colour dialog and apply the chosen colour."""
        from PySide6.QtWidgets import QColorDialog
        from PySide6.QtGui import QColor
        initial = QColor(getattr(self, "_struct_ai_color", "#f3722c"))
        color = QColorDialog.getColor(initial, self, "Pick AI Feature Highlight Color")
        if color.isValid():
            self._set_ai_feature_color(color.name())

    def _animate_ai_controls(self, show: bool) -> None:
        """Smoothly expand or collapse the AI sub-control container."""
        from PySide6.QtCore import QPropertyAnimation, QEasingCurve
        ctr = self._ai_ctrl_container
        if show:
            ctr.setVisible(True)
            target_h = ctr.sizeHint().height() or 80
        else:
            target_h = 0
        anim = QPropertyAnimation(ctr, b"maximumHeight", self)
        anim.setDuration(180)
        anim.setStartValue(ctr.maximumHeight())
        anim.setEndValue(target_h)
        anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
        if not show:
            anim.finished.connect(lambda: ctr.setVisible(False))
        anim.start()
        self._ai_ctrl_anim = anim   # keep alive

    def _on_ai_style_toggled(self, btn) -> None:
        """Switch between Gradient and Binary coloring modes."""
        mode = "gradient" if btn is self._ai_grad_btn else "binary"
        self._struct_ai_color_mode = mode
        in_grad = (mode == "gradient")
        self.struct_ai_gradient_lbl.setVisible(in_grad)
        self.struct_ai_gradient_combo.setVisible(in_grad)
        self.struct_ai_color_lbl.setVisible(not in_grad)
        self.struct_ai_color_combo.setVisible(not in_grad)
        if self.struct_color_mode_combo.currentText() == "AI Features":
            scheme = self.struct_scheme_combo.currentText()
            if scheme:
                self._push_feature_scores(scheme)

    def _on_ai_gradient_combo_changed(self, text: str) -> None:
        if self.struct_color_mode_combo.currentText() == "AI Features":
            scheme = self.struct_scheme_combo.currentText()
            if scheme:
                self._push_feature_scores(scheme)

    def _on_ai_color_combo_changed(self, text: str) -> None:
        if text == "Custom":
            self._pick_ai_feature_color()
            return
        for name, hexval in self._AI_COLOR_ITEMS:
            if name == text and hexval:
                self._struct_ai_color = hexval
                if self.struct_color_mode_combo.currentText() == "AI Features":
                    scheme = self.struct_scheme_combo.currentText()
                    if scheme:
                        self._push_feature_scores(scheme)
                break

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
        _AGGR_CB_META = {
            "Fire":    {"css": "linear-gradient(to top,#000000,#aa3300,#ff6600,#ffaa00,#ffff00)",
                        "min": "0.0", "mid": "0.5", "max": "max", "unit": "Normalised \u03b2-aggregation"},
            "Inferno": {"css": "linear-gradient(to top,#000004,#3b0f70,#8c2981,#de4968,#fcfdbf)",
                        "min": "0.0", "mid": "0.5", "max": "max", "unit": "Normalised \u03b2-aggregation"},
            "Viridis": {"css": "linear-gradient(to top,#440154,#31688e,#35b779,#fde725)",
                        "min": "0.0", "mid": "0.5", "max": "max", "unit": "Normalised \u03b2-aggregation"},
        }
        if scheme in _AGGR_CMAP:
            cmap = matplotlib.colormaps[_AGGR_CMAP[scheme]]
            resi_colors = {i + 1: _mc.to_hex(cmap(v)) for i, v in enumerate(norm_scores)}
            meta = _AGGR_CB_META[scheme]
            self._js(
                f"setResidueColorMap({_json.dumps(resi_colors)},"
                f"'ZYGGREGATOR \u03b2-Aggregation',{_json.dumps(meta)});"
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
            "Buried→Exposed (Blue→Red)":    "RdBu_r",
            "Exposed→Buried (Red→Blue)":    "RdBu",
            "Viridis (Buried→Exposed)":     "viridis",
            "Plasma (Buried→Exposed)":      "plasma",
            "Magma (Buried→Exposed)":       "magma",
            "Cyan→Orange":                  "PuOr_r",
        }
        _SCHEME_CB_META = {
            "Buried→Exposed (Blue→Red)":
                {"css": "linear-gradient(to top,#3b6fc9,#f7f7f7,#cc1111)",
                 "min": "0.0 (buried)", "mid": "0.5", "max": "1.0 (exposed)", "unit": "RSA"},
            "Exposed→Buried (Red→Blue)":
                {"css": "linear-gradient(to top,#cc1111,#f7f7f7,#3b6fc9)",
                 "min": "0.0 (buried)", "mid": "0.5", "max": "1.0 (exposed)", "unit": "RSA"},
            "Viridis (Buried→Exposed)":
                {"css": "linear-gradient(to top,#440154,#31688e,#35b779,#fde725)",
                 "min": "0.0 (buried)", "mid": "0.5", "max": "1.0 (exposed)", "unit": "RSA"},
            "Plasma (Buried→Exposed)":
                {"css": "linear-gradient(to top,#0d0887,#7e03a8,#cc4778,#f89540,#f0f921)",
                 "min": "0.0 (buried)", "mid": "0.5", "max": "1.0 (exposed)", "unit": "RSA"},
            "Magma (Buried→Exposed)":
                {"css": "linear-gradient(to top,#000004,#3b0f70,#8c2981,#de4968,#fcfdbf)",
                 "min": "0.0 (buried)", "mid": "0.5", "max": "1.0 (exposed)", "unit": "RSA"},
            "Cyan→Orange":
                {"css": "linear-gradient(to top,#7b3294,#c2a5cf,#f7f7f7,#fdae61,#e66101)",
                 "min": "0.0 (buried)", "mid": "0.5", "max": "1.0 (exposed)", "unit": "RSA"},
        }
        cmap_name = _SCHEME_CMAP.get(scheme, "RdBu_r")
        cmap = matplotlib.colormaps[cmap_name]
        meta = _SCHEME_CB_META.get(
            scheme,
            {"css": "linear-gradient(to top,#3b6fc9,#f7f7f7,#cc1111)",
             "min": "0.0 (buried)", "mid": "0.5", "max": "1.0 (exposed)", "unit": "RSA"},
        )

        # Pre-compute a hex color per residue and push as per-residue color dict
        resi_colors: dict[int, str] = {}
        for resi, rsa in sasa.items():
            rgba = cmap(float(rsa))
            resi_colors[resi] = _mc.to_hex(rgba)

        # Push via dedicated JS function that accepts {resi: hexColor} and colorbar metadata
        self._js(
            f"setResidueColorMap({_json.dumps(resi_colors)}, "
            f"'Solvent Accessibility (RSA)', {_json.dumps(meta)});"
        )

    def _populate_sasa_report_section(self) -> None:
        """Generate and populate the SASA Profile report section from current SASA data."""
        if "SASA Profile" not in getattr(self, "report_section_tabs", {}):
            return
        rsa = getattr(self, "_struct_sasa_data", {})
        asa = getattr(self, "_struct_sasa_raw", {})
        if not rsa:
            from beer.reports.css import get_report_css
            css = get_report_css(getattr(self, "_is_dark", False))
            html = (
                f"<html><head><style>{css}</style></head><body>"
                f"<h2>Solvent Accessibility (SASA)</h2>"
                f"<p>SASA data requires a 3D structure. No structure is currently loaded.</p>"
                f"<p>To compute SASA:</p>"
                f"<ol>"
                f"<li>Fetch an AlphaFold model via <b>Fetch AlphaFold</b></li>"
                f"<li>Import a local PDB file via <b>Import PDB</b></li>"
                f"<li>Fetch a structure by PDB ID via <b>Fetch PDB ID</b></li>"
                f"</ol>"
                f"<p>Once a structure is loaded, SASA is computed automatically using the "
                f"Shrake–Rupley algorithm.</p>"
                f"</body></html>"
            )
            self.report_section_tabs["SASA Profile"].setHtml(html)
            if self.analysis_data is not None:
                self.analysis_data.setdefault("report_sections", {})["SASA Profile"] = html
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
        spar_uri = self._make_sparkline_png(rsa_vals, "#7b9cff", threshold=None)
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

    def _struct_seq_len(self) -> int:
        """Sequence length for the Residue Number colour gradient — best available source."""
        seq = (
            (self.analysis_data or {}).get("seq")
            or (self.alphafold_data or {}).get("seq")
            or (self.esmfold2_data  or {}).get("seq")
            or self.seq_text.toPlainText().strip()
        )
        return len(seq) if seq else 0   # 0 → JS _autoSeqLen takes over

    def _on_struct_rep_changed(self, rep_label: str) -> None:
        self._js(f"setRepresentation('{rep_label.lower()}');")

    def _on_struct_color_mode_changed(self, mode: str) -> None:
        self._update_scheme_combo(mode)
        is_ai = (mode == "AI Features")
        if hasattr(self, "_ai_ctrl_container"):
            self._animate_ai_controls(is_ai)
        if is_ai and hasattr(self, "struct_ai_color_lbl"):
            in_grad = self._struct_ai_color_mode == "gradient"
            self.struct_ai_gradient_lbl.setVisible(in_grad)
            self.struct_ai_gradient_combo.setVisible(in_grad)
            self.struct_ai_color_lbl.setVisible(not in_grad)
            self.struct_ai_color_combo.setVisible(not in_grad)
        key = self._STRUCT_MODE_KEY.get(mode, "plddt")
        scheme = self.struct_scheme_combo.currentText()
        if mode == "AI Features":
            if not scheme:
                return
            self._push_feature_scores(scheme)
        elif mode == "Aggregation (ZYGGREGATOR)":
            self._push_zyggregator_scores(scheme)
        elif mode == "Solvent Accessibility":
            self._push_sasa_scores(scheme)
        elif mode == "Residue Number":
            seq_len = self._struct_seq_len()
            self._js(f"setColorMode('{key}','{scheme}',{seq_len});")
        else:
            self._js(f"setColorMode('{key}','{scheme}');")

    def _on_struct_scheme_changed(self, scheme: str) -> None:
        if not scheme:
            return
        mode = self.struct_color_mode_combo.currentText()
        if mode == "AI Features":
            self._push_feature_scores(scheme)
        elif mode == "Aggregation (ZYGGREGATOR)":
            self._push_zyggregator_scores(scheme)
        elif mode == "Solvent Accessibility":
            self._push_sasa_scores(scheme)
        elif mode == "Residue Number":
            seq_len = self._struct_seq_len()
            self._js(f"setScheme('{scheme}',{seq_len});")
        else:
            self._js(f"setScheme('{scheme}');")

    def _on_struct_colorbar_toggled(self, checked: bool) -> None:
        self._js(f"setColorBarVisible({'true' if checked else 'false'});")

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

    def _on_struct_anim_clicked(self, which: str) -> None:
        """Spin/Rock are mutually exclusive; clicking an active mode toggles it off.
        Button check-state is already flipped by Qt before this runs."""
        if which == "spin":
            if self.struct_spin_btn.isChecked():
                self.struct_rock_btn.setChecked(False)
                self._js("setSpin(true,'spin');")
            else:
                self._js("setSpin(false,'spin');")
        else:  # rock
            if self.struct_rock_btn.isChecked():
                self.struct_spin_btn.setChecked(False)
                self._js("setSpin(true,'rock');")
            else:
                self._js("setSpin(false,'rock');")

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
        if hasattr(self, "struct_sel_edit"):
            self.struct_sel_edit.clear()
            self._sel_count_lbl.setText("")
        if hasattr(self, "struct_dist_btn") and self.struct_dist_btn.isChecked():
            self.struct_dist_btn.setChecked(False)
        # Reset Display controls (Mol* extras) without firing their JS handlers
        for _w, _checked in (
            (getattr(self, "struct_spin_btn", None), False),
            (getattr(self, "struct_rock_btn", None), False),
            (getattr(self, "struct_seq_cb", None), False),
            (getattr(self, "struct_assembly_cb", None), False),
            (getattr(self, "struct_lig_cb", None), True),
            (getattr(self, "struct_wat_cb", None), False),
            (getattr(self, "struct_ion_cb", None), False),
        ):
            if _w is not None:
                _w.blockSignals(True)
                _w.setChecked(_checked)
                _w.blockSignals(False)
        self._js("clearSelection(); clearDistances(); clearResidueLabels(); "
                 "setSpin(false); setSequenceVisible(false); resetView();")

    def _pick_background_color(self) -> None:
        color = QColorDialog.getColor(parent=self)
        if color.isValid():
            self._js(f"setBackground('{color.name()}');")

    def _on_structure_page_loaded(self, ok: bool) -> None:
        """Called once when the base 3Dmol page finishes loading.
        If a PDB was queued before the page was ready, deliver it now."""
        if ok:
            self._struct_page_ready = True
            # Apply current theme background immediately
            bg = "#1a1a2e" if getattr(self, "_is_dark", False) else "#ffffff"
            self.structure_viewer.page().runJavaScript(f"setBackground('{bg}');")
        if ok and self._pending_pdb is not None:
            pdb_json = self._pending_pdb
            self._pending_pdb = None
            self.structure_viewer.page().runJavaScript(f"loadPDB({pdb_json});")
        elif not ok:
            self._pending_pdb = None

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

    def _load_structure_viewer(self, pdb_str: str, compute_sasa: bool = True) -> None:
        """Swap in a new structure without reloading the 3Dmol page.

        compute_sasa: when False, skip the implicit full-structure SASA computation.
        Multi-chain loads pass False because per-chain SASA is computed by
        _restore_chain_structure for the selected chain; computing SASA on the full
        complex here would clobber that with chain-collided residue numbers.
        """
        if not _WEBENGINE_AVAILABLE or self.structure_viewer is None:
            return
        self._struct_pdb_str = pdb_str          # keep for Ramachandran / SS extraction
        # Cache experimental PDB (used by SASA, distance map, etc.)
        # Do not overwrite when source is AlphaFold or ESMFold2.
        _src = getattr(self, "_struct_source", "none")
        if _src not in ("alphafold", "esmfold2", "overlay"):
            self._exp_pdb_str = pdb_str
        pdb_json = json.dumps(pdb_str)
        # Keep as pending so loadFinished can retry if the page is still loading.
        self._pending_pdb = pdb_json
        # 1-arg form is the only safe form in PySide6 (no 2-arg callback variant).
        self._js(f"loadPDB({pdb_json});")
        self._populate_chain_controls(pdb_str)
        self._update_overlay_controls()
        # Annotate disorder regions and signal peptide in 3D viewer after a short delay
        from PySide6.QtCore import QTimer as _QT
        _QT.singleShot(800, self._annotate_structure_viewer)
        # Compute SASA deferred — Shrake-Rupley is CPU-bound and must not block
        # the event loop before loadPDB has been dispatched to WebEngine.
        if compute_sasa:
            _pdb_copy = pdb_str
            def _deferred_sasa():
                self._struct_sasa_data, self._struct_sasa_raw = self._compute_sasa(_pdb_copy)
                self._populate_sasa_report_section()
                if self.analysis_data:
                    self.update_graph_tabs()
            _QT.singleShot(200, _deferred_sasa)

    @staticmethod
    def _parse_pdb_chains(pdb_str: str) -> list[str]:
        """Return sorted list of unique chain IDs from ATOM records (handles PDB and mmCIF).
        HETATM-only chains (ligands/water with a separate chain letter) are excluded —
        they would appear as mystery chain entries that confuse users."""
        seen: dict = {}
        # PDB format: chain ID at column 22 (0-indexed 21); ATOM records only.
        for line in pdb_str.splitlines():
            if line.startswith("ATOM  "):
                chain = line[21:22].strip()
                if chain and chain not in seen:
                    seen[chain] = None
        if seen:
            return sorted(seen.keys())
        # mmCIF format: find auth_asym_id column in _atom_site loop
        import re as _re
        headers: list[str] = []
        asym_idx = -1
        in_loop = False
        for line in pdb_str.splitlines():
            s = line.strip()
            if s.startswith("_atom_site."):
                headers.append(s)
                in_loop = True
            elif in_loop and s and not s.startswith("_") and not s.startswith("#"):
                if asym_idx < 0:
                    for i, h in enumerate(headers):
                        if "auth_asym_id" in h:
                            asym_idx = i
                            break
                if asym_idx >= 0:
                    parts = s.split()
                    if len(parts) > asym_idx:
                        chain = parts[asym_idx].strip("\"'.")
                        if chain and chain not in seen:
                            seen[chain] = None
            elif in_loop and s.startswith("#"):
                in_loop = False
                headers = []
                asym_idx = -1
        return sorted(seen.keys())

    def _populate_chain_controls(self, pdb_str: str) -> None:
        """Build per-chain visibility checkboxes from a PDB string."""
        if not hasattr(self, "_chain_cbs_layout"):
            return
        chains = self._parse_pdb_chains(pdb_str)
        # Clear existing checkboxes
        self._chain_checkboxes.clear()
        self._clear_layout_deep(self._chain_cbs_layout)
        # Show or hide the entire accordion section based on chain count
        if hasattr(self, "_chains_accordion"):
            self._chains_accordion.setVisible(len(chains) > 1)
        if len(chains) <= 1:
            return
        for chain_id in chains:
            cb = QCheckBox(f"Chain {chain_id}")
            cb.setChecked(True)
            cb.toggled.connect(lambda checked, c=chain_id:
                self._js(f"setChainVisible('{c}', {'true' if checked else 'false'});"))
            self._chain_cbs_layout.addWidget(cb)
            self._chain_checkboxes[chain_id] = cb
        # Auto-expand the accordion so chain checkboxes are immediately visible
        if hasattr(self, "_chains_grp_box") and not self._chains_grp_box.isVisible():
            self._chains_grp_box.setVisible(True)
            if hasattr(self, "_chains_accordion"):
                _al = self._chains_accordion.layout()
                if _al and _al.count() > 0:
                    _hdr = _al.itemAt(0).widget()
                    if isinstance(_hdr, QPushButton) and "▶" in _hdr.text():
                        _hdr.setText(_hdr.text().replace("▶", "▼"))

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
        """Push per-residue AI scores into the Mol* viewer."""
        if not _WEBENGINE_AVAILABLE or self.structure_viewer is None:
            return
        if not self.analysis_data:
            return
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
            ("ss3h",      "ss3_h_profile"),
            ("ss3e",      "ss3_e_profile"),
            ("ss3c",      "ss3_c_profile"),
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
        pdb_str = None
        if self.alphafold_data:
            pdb_str = self.alphafold_data.get("pdb_str")
        if not pdb_str and getattr(self, "_exp_pdb_str", None):
            pdb_str = self._exp_pdb_str
        if not pdb_str:
            self.statusBar.showMessage("No structure loaded to save.", 3000)
            return
        fn, _ = QFileDialog.getSaveFileName(self, "Save PDB", "", "PDB Files (*.pdb)")
        if fn:
            if not fn.lower().endswith(".pdb"):
                fn += ".pdb"
            try:
                with open(fn, "w") as f:
                    f.write(pdb_str)
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
        self.blast_db_combo.setMaximumWidth(160)
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
        self.blast_table.setColumnCount(8)
        self.blast_table.setHorizontalHeaderLabels(
            ["Accession", "Description", "Length", "Score", "E-value", "% Identity", "Load", "Copy"]
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

        _hint = QLabel("Paste two sequences below and click <b>Compare</b> to compare biophysical properties side by side.")
        _hint.setObjectName("placeholder_lbl")
        _hint.setWordWrap(True)
        layout.addWidget(_hint)

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

        _hint = QLabel("Import a multi-chain PDB file (via <b>Import PDB</b> on the Analysis tab) to populate this table with per-chain biophysical data.")
        _hint.setObjectName("placeholder_lbl")
        _hint.setWordWrap(True)
        layout.addWidget(_hint)

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
        self.ph_input = QDoubleSpinBox()
        self.ph_input.setRange(0.0, 14.0)
        self.ph_input.setSingleStep(0.1)
        self.ph_input.setDecimals(1)
        self.ph_input.setValue(self.default_pH)
        self._set_tooltip(self.ph_input, "Sets the pH value used for net-charge calculations (0–14).")
        form.addRow("Default pH:", self.ph_input)

        self.window_size_input = QSpinBox()
        self.window_size_input.setRange(3, 51)
        self.window_size_input.setSingleStep(2)
        self.window_size_input.setValue(self.default_window_size)
        self._set_tooltip(self.window_size_input, "Length of sliding window for hydrophobicity profiles (odd numbers recommended).")
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

        self.pka_set_combo = QComboBox()
        self.pka_set_combo.addItems(list(PKA_SETS.keys()))
        self.pka_set_combo.setCurrentText(self.pka_set)
        self._set_tooltip(self.pka_set_combo,
            "Published pKa set used for pI and net-charge calculations.\n"
            "pI can shift 0.5–1.0 unit between sets. Choose "
            "'Bjellqvist (ProtParam)' to match ExPASy ProtParam.\n"
            "A custom pKa list below (if filled) overrides this.")
        form.addRow("pKa Set:", self.pka_set_combo)

        self.pka_input = QLineEdit("")
        self.pka_input.setPlaceholderText("override: 9.69,2.34,3.90,4.07,8.18,10.46,6.04,10.54,12.48")
        self._set_tooltip(self.pka_input, "Optional custom override of the pKa Set above. Leave blank to use the selected set. Provide exactly 9 comma-separated floats: N-term, C-term, D, E, C, Y, H, K, R.")
        self._pka_error_lbl = QLabel("")
        self._pka_error_lbl.setObjectName("status_lbl")
        self._pka_error_lbl.setProperty("status_state", "error")
        self._pka_error_lbl.hide()
        def _validate_pka_field():
            raw = self.pka_input.text().strip()
            if not raw:
                self._pka_error_lbl.hide()
                return
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            if len(parts) != 9:
                self._pka_error_lbl.setText(f"Need exactly 9 values — got {len(parts)}.")
                self._pka_error_lbl.show()
                self.pka_input.setStyleSheet("border: 1px solid #e74c3c;")
            else:
                try:
                    list(map(float, parts))
                    self._pka_error_lbl.hide()
                    self.pka_input.setStyleSheet("")
                except ValueError:
                    self._pka_error_lbl.setText("Non-numeric value in pKa list.")
                    self._pka_error_lbl.show()
                    self.pka_input.setStyleSheet("border: 1px solid #e74c3c;")
        self.pka_input.editingFinished.connect(_validate_pka_field)
        form.addRow("Override pKa (N,C,D,E,C,Y,H,K,R):", self.pka_input)
        form.addRow("", self._pka_error_lbl)

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
        self.label_font_input = QSpinBox()
        self.label_font_input.setRange(6, 32)
        self.label_font_input.setValue(self.label_font_size)
        self._set_tooltip(self.label_font_input, "Font size for axis labels and titles (6–32 pt).")
        form3.addRow("Label Font Size:", self.label_font_input)

        self.tick_font_input = QSpinBox()
        self.tick_font_input.setRange(6, 28)
        self.tick_font_input.setValue(self.tick_font_size)
        self._set_tooltip(self.tick_font_input, "Font size for tick labels (6–28 pt).")
        form3.addRow("Tick Font Size:", self.tick_font_input)

        self.marker_size_input = QSpinBox()
        self.marker_size_input.setRange(1, 30)
        self.marker_size_input.setValue(self.marker_size)
        self._set_tooltip(self.marker_size_input, "Size of data markers in line and scatter graphs (1–30).")
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

        # ── API Keys ──────────────────────────────────────────────────────────
        _api_lbl = QLabel("API Keys")
        _api_lbl.setObjectName("settings_section_lbl")
        layout.addWidget(_api_lbl)
        form5 = QFormLayout()
        form5.setHorizontalSpacing(20)
        form5.setVerticalSpacing(8)
        form5.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.biohub_api_key_input = QLineEdit()
        self.biohub_api_key_input.setPlaceholderText("Paste your EvolutionaryScale BioHub API key here")
        self.biohub_api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.biohub_api_key_input.setText(_config.load().get("biohub_api_key", ""))
        self.biohub_api_key_input.setToolTip(
            "Required for Predict Structure (ESMFold2 via Forge API).\n"
            "Get your key at: biohub.ai")
        form5.addRow("BioHub API Key:", self.biohub_api_key_input)
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

        # Left panel: search bar + section list
        _left_w = QWidget()
        _left_w.setFixedWidth(172)
        _left_v = QVBoxLayout(_left_w)
        _left_v.setContentsMargins(4, 4, 4, 2)
        _left_v.setSpacing(3)

        from PySide6.QtWidgets import QLineEdit as _QLE
        self._help_search = _QLE()
        self._help_search.setPlaceholderText("Search…")
        self._help_search.setClearButtonEnabled(True)
        self._help_search.setFixedHeight(26)
        _left_v.addWidget(self._help_search)

        help_nav = QListWidget()
        help_nav.setObjectName("report_nav")
        help_nav.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        _left_v.addWidget(help_nav, 1)
        help_h.addWidget(_left_w)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFrameShadow(QFrame.Shadow.Plain); sep.setObjectName("nav_sep")
        help_h.addWidget(sep)

        help_stack = QStackedWidget()
        help_h.addWidget(help_stack, 1)

        _HELP_SECTIONS = [
            ("Getting Started", """
<h1>Getting Started</h1>
<p>BEER v3.0 uses <b>ESMC 600M</b> (EvolutionaryScale) as its sequence embedding backbone.
All 24 per-residue AI prediction heads run on ESMC embeddings — no classical fallback methods are used
for any feature that has a trained AI head.</p>
<h2>Input methods</h2>
<ul>
  <li><b>Paste sequence</b> — type or paste a bare amino-acid string (ACDEFG…) or FASTA block into the sequence box and click <b>Analyze [Ctrl+Enter]</b>.</li>
  <li><b>Import FASTA</b> — load a .fa / .fasta file (single or multi-sequence).</li>
  <li><b>Import PDB</b> — extract sequence(s) from a local PDB file.</li>
  <li><b>Fetch</b> — enter a <b>UniProt ID</b> (e.g. <tt>P04637</tt>) or a 4-character <b>PDB ID</b> (e.g. <tt>1ABC</tt>) and click <b>Fetch</b>.
    UniProt IDs automatically set the accession for <b>Fetch AlphaFold</b>, <b>Fetch Pfam</b>, and other databases, and trigger analysis immediately.
    PDB IDs download all chains and the coordinate file from RCSB; structural graphs (Ramachandran, distance map, etc.) are shown immediately.</li>
  <li><b>Find UniProt ID</b> — paste any sequence and BEER will search UniProt Swiss-Prot by sequence hash (exact match) and fall back to a BLAST search if needed. If a match is found, the accession is set automatically and all external databases are populated.</li>
</ul>
<h2>Welcome chips</h2>
<p>When no sequence is loaded, the welcome banner shows four example proteins (FUS, Rhodopsin, Ubiquitin, Haemoglobin). Click any chip to fetch and analyse that protein immediately — useful for exploring BEER for the first time.</p>
<h2>UniProt Tracks</h2>
<p>After fetching a UniProt accession, click <b>UniProt Tracks</b> (Structure toolbar) to download all feature annotations for the protein from UniProt.
Once loaded, BEER adds per-residue annotation overlays to the AI prediction graphs in the Graphs tab, showing curated Swiss-Prot regions alongside the AI probability curve.</p>
<h2>Feature Coloring on 3D Structure</h2>
<p>In the Structure tab, select <b>Feature Score</b> from the color mode dropdown to color the protein by any per-residue AI prediction score.
Choose the feature (Disorder, Signal Peptide, etc.) from the adjacent selector; residues are colored white→feature color according to their predicted probability.</p>
<h2>Custom Graphs</h2>
<p>The <b>Graphs → Custom</b> category provides two multi-feature plot types:</p>
<ul>
  <li><b>Feature Overlay</b> — select any combination of the 32 available per-residue profiles and overlay them on a shared x-axis (residue position). A "Normalize [0–1]" option rescales each profile independently. Each curve is automatically assigned a distinct color.</li>
  <li><b>Feature Correlation</b> — select two or more profiles and compute a pairwise Pearson r correlation matrix, displayed as a heatmap. Choose from a range of colormaps. Useful for identifying which features co-vary along the sequence.</li>
</ul>
<p>BiLSTM-based features in the checklist are enabled as soon as ESMC is available. Selecting an AI feature that has not yet been computed for the current sequence automatically triggers a background computation with a progress indicator.</p>
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
<p>Per-residue secondary structure (Helix / Strand / Coil) is predicted by the SS3 AI head
(BiLSTM-CRF Q3 decoder, Q3 accuracy 0.843). Per-residue disorder probability is predicted by
the Disorder AI head (BiLSTM, AUROC 0.9923). Both are shown in <b>Graphs → AI Predictions</b>.</p>
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
<p>Transmembrane topology prediction in BEER v3.0 uses an AI <b>BiLSTM-CRF</b> head trained on
UniProt Swiss-Prot <tt>ft_transmem</tt> annotations (AUROC 0.9870). The CRF decoder enforces
valid outside→helix→inside topological constraints via Viterbi decoding, preventing physically
impossible helix arrangements. No external server is required.</p>
<h2>Transmembrane Profile graph</h2>
<p>Per-residue TM probability (0–1) from the BiLSTM-CRF model, shown in the <b>Graphs → AI Predictions</b>
category. Regions exceeding the F1-maximising threshold are highlighted and listed in the
<b>AI Predictions → Transmembrane</b> report section with their residue ranges.</p>
<h2>TM Topology graph</h2>
<p>A simplified snake-plot derived from the AI-predicted helix positions. The yellow band represents
the membrane. Blue rectangles are TM helices labelled with their residue range. Loops are drawn
above (extracellular) or below (cytoplasmic) the band according to the predicted topology,
using the inside-positive rule to assign orientation.</p>
"""),
            ("AlphaFold & 3D Structure", """
<h1>3D Structure &amp; AlphaFold Integration</h1>
<p>Structure data can come from four sources:</p>
<ul>
  <li><b>Import PDB</b> — load a local .pdb file directly.</li>
  <li><b>Fetch PDB ID</b> — enter a 4-character RCSB PDB code in the accession field; sequences and
      the coordinate file are both downloaded automatically.</li>
  <li><b>Fetch AlphaFold</b> — requires a UniProt accession.  Fetch it with the <b>Fetch</b> button first,
      then click <b>Fetch AlphaFold</b> to download the EBI AlphaFold2 predicted structure.</li>
  <li><b>Predict Structure (ESMFold2)</b> — folds the current sequence with ESMFold2 via the
      EvolutionaryScale Forge API.  No accession needed; requires a free <b>BioHub API key</b>
      (set it in <b>Settings → BioHub API Key</b>).  Useful to compare an AI-predicted structure
      with an AlphaFold model or an uploaded experimental PDB.</li>
</ul>
<p><b>AlphaFold + ESMFold2 overlay:</b> when both an AlphaFold and an ESMFold2 structure are
loaded, the viewer overlays them (AlphaFold blue, ESMFold2 orange) and a Cα RMSD profile becomes
available. Colour controls are locked to keep the two models distinct; un-check either model in the
<b>Overlay</b> panel to regain full colour control over the remaining structure.</p>
<p>For every structure source, Cα coordinates are extracted per chain and used to compute:</p>
<ul>
  <li>Per-residue <b>pLDDT / B-factor</b> scores.</li>
  <li>Cα pairwise <b>distance matrix</b>.</li>
</ul>
<h2>Fix PDB (gap-filling)</h2>
<p>The <b>Fix PDB</b> tab completes an experimental structure that has missing
(unresolved) residues by transplanting predicted coordinates into the gaps.
Fetch the experimental structure by PDB ID, then fill the gaps using either:</p>
<p>On fetch the structure is reduced to protein atoms — water, ions, and ligands
are stripped (modified residues such as selenomethionine are retained) so
heteroatoms cannot disturb the experimental-vs-predicted sequence alignment.</p>
<ul>
  <li><b>Fix PDB</b> — fetches the matching <b>AlphaFold</b> model (enter the UniProt
      accession) and uses it as the gap-fill source.</li>
  <li><b>Fix with ESMFold2</b> — folds the chain with <b>ESMFold2</b> via the BioHub
      Forge API, then fills the gaps with that prediction.  The folded sequence is
      reconstructed from the experimental chain's resolved residue range with
      placeholder residues at the unresolved positions, so the prediction contains a
      residue for every gap (gap backbones are transplanted; their identities are
      placeholders — use the AlphaFold path when exact gap residue identities matter).
      Requires a free <b>BioHub API key</b> (<b>Settings → BioHub API Key</b>) but no
      UniProt accession.  Use this when no AlphaFold model exists.</li>
</ul>
<p>Both paths align the experimental and predicted chains, Kabsch-superpose the
predicted model, transplant predicted atoms into the missing-residue gaps, and
mark provenance in the output (experimental residues at occupancy 1.00, gap-fill
at 0.00; REMARK records name the gap-fill source).  The coverage bar and stats
report how many residues are experimental vs. predicted.</p>

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
<a href="https://molstar.org">Mol*</a> (bundled locally — fully offline, no CDN), embedded via
Qt WebEngine. Requires the <b>PySide6-Addons</b> package (ships QtWebEngineWidgets):</p>
<pre>pip install PySide6-Addons</pre>
<p>If not installed, the structure can still be exported (PDB / mmCIF / GRO / XYZ) and opened in
PyMOL, UCSF ChimeraX, or a web viewer.</p>
<p>Colour modes include pLDDT / B-factor, Residue Type, Chain, Charge, Hydrophobicity, Mass,
Secondary Structure, Residue Number, Solvent Accessibility, AI Features, and Aggregation;
representations include Cartoon, Stick, Sphere, Line, Trace, and Surface. The <b>Interact</b> panel
supports residue selection and distance / angle / dihedral measurements.</p>
<p><b>Multi-chain structures:</b> for a fetched or imported multi-chain PDB/mmCIF, per-chain
visibility checkboxes (plus <b>Show all</b> / <b>Hide all</b>) appear in the <b>Chains</b> panel.
Switching the chain selector on the Analysis tab updates the per-chain profiles (pLDDT, distance map,
Ramachandran, SASA) while the viewer keeps showing the full assembly.</p>
"""),
            ("Pfam Domains", """
<h1>Pfam Domain Annotations</h1>
<p>Requires an internet connection and a valid UniProt accession. Click <b>Fetch Pfam</b>
after loading a UniProt accession. Not available for PDB IDs.</p>
<h2>Data source</h2>
<p>Queries the <b>EMBL-EBI InterPro REST API</b> for all Pfam-family entries associated
with the given UniProt protein. Results include domain name, accession, and start/end residue positions.</p>
<h2>Domain Architecture graph</h2>
<p>The Domain Architecture graph is shown in the Graphs tab after Pfam data has been fetched.
Each Pfam domain is drawn as a labelled coloured rectangle on a linear sequence axis.
Overlapping or adjacent domains are stacked to avoid label collisions.</p>
<p class="note">Only Pfam-family entries are shown. The graph is not available until <b>Fetch Pfam</b> is clicked.</p>
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
  <li><b>Disorder Profile</b> — AI per-residue disorder probability (ESMC → BiLSTM, AUROC 0.9923); fill = disordered above threshold.</li>
  <li><b>Linear Sequence Map</b> — colour-coded overview of hydrophobicity and charge along the sequence.</li>
</ul>
<h2>Charge &amp; π-Interactions</h2>
<ul>
  <li><b>Isoelectric Focus</b> — Henderson-Hasselbalch charge curve 0–14; pI and physiological pH 7.4 charge marked.</li>
  <li><b>Charge Decoration</b> — Das-Pappu FCR vs |NCPR| phase diagram.</li>
  <li><b>Cation–π Map</b> — proximity heat map (1/distance) for K/R ↔ F/W/Y pairs within ±8 residues.</li>
</ul>
<h2>Structure &amp; Folding</h2>
<ul>
  <li><b>Sticker Map</b> — aromatic (amber), basic (blue), acidic (pink), spacer (grey).</li>
  <li><b>Helical Wheel</b> — Cartesian projection of first 18 residues at 100°/step; KD coloured with luminance-contrast labels.</li>
  <li><b>TM Topology</b> — snake-plot derived from AI BiLSTM-CRF transmembrane prediction (see Transmembrane Helices).</li>
</ul>
<h2>Structural Graphs</h2>
<ul>
  <li><b>pLDDT Profile</b> — per-residue B-factor confidence (0–100). Available after any structure is loaded.</li>
  <li><b>Cα Distance Map</b> — pairwise distance heatmap with 8 Å contact contour. Available after any structure is loaded.</li>
  <li><b>Ramachandran Plot</b> — φ/ψ dihedral angles coloured by secondary structure. Available after any structure is loaded.</li>
  <li><b>Residue Contact Network</b> — graph of residues within 8 Å contact distance. Available after any structure is loaded.</li>
  <li><b>Domain Architecture</b> — Pfam domain rectangles on a linear sequence axis. Available after Fetch Pfam.</li>
</ul>
<h2>Phase Separation / IDP</h2>
<ul>
  <li><b>Uversky Phase Plot</b> — mean |net charge| vs mean normalised hydrophobicity; Uversky boundary line separates IDP from ordered proteins.</li>
  <li><b>Single-Residue Perturbation Map</b> — 20×n heatmap of |ΔGRAVY| + |ΔNCPR| for every possible single-residue substitution; white dot = wild type. Available for sequences ≤500 aa.</li>
</ul>
<h2>AI Predictions</h2>
<p>All 24 AI prediction heads have their own graph tab under <b>Graphs → AI Predictions</b>.
Each shows the per-residue probability curve with the F1-maximising threshold line and optionally a
UniProt annotation overlay and MC-Dropout uncertainty band. See <b>AI Predictions</b> help for the full list.</p>
<h2>Custom</h2>
<ul>
  <li><b>Feature Overlay</b> — overlay any combination of the 32 per-residue profiles (classical + all 24 AI heads + SS3 components) on a shared x-axis. Optional 0–1 normalisation. Each curve gets a unique color.</li>
  <li><b>Feature Correlation</b> — pairwise Pearson r heatmap for any selected subset of profiles. Selectable colormap.</li>
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
<h2>Coiled-Coil Profile</h2>
<p>Per-residue coiled-coil probability from the AI BiLSTM head (AUROC 0.9819) trained on
UniProt Swiss-Prot <tt>ft_coiled</tt> annotations. Available in <b>Graphs → AI Predictions → Coiled-Coil Profile</b>.</p>
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
<p>BEER v3.0 includes 24 per-residue AI prediction heads. Each head uses <b>ESMC 600M</b> embeddings
(1152-dim, frozen) fed into a 2-layer BiLSTM classifier trained on curated structural databases
and UniProt Swiss-Prot annotations. Every head produces a per-residue probability in [0, 1].
Each AI head is computed on demand — click any <b>AI Predictions</b> entry in the sidebar to trigger it.
The Report tab shows predicted regions, residue ranges, and the threshold used.</p>
<h2>Architecture</h2>
<p>ESMC 600M (1152-dim, frozen) → 2-layer Bidirectional LSTM (hidden = 256) → Linear(512 → 1) → Sigmoid.<br>
<b>Transmembrane</b> uses a BiLSTM-CRF decoder (Viterbi, enforces valid outside→helix→inside topology).<br>
<b>Secondary Structure (SS3)</b> uses a BiLSTM-CRF Q3 decoder (Helix / Strand / Coil).<br>
All heads trained with focal loss and MMseqs2-clustered train/val/test splits.
Classification threshold set at the F1-maximising point on the validation set.</p>
<h2>Sequence length</h2>
<p>All 24 heads were trained on sequences <b>truncated to 1024 residues</b> — no head saw a
residue beyond position 1024 during training. There is <b>no hard cap at inference</b>: longer
proteins are embedded and scored end-to-end and you always get a full-length profile.
However, predictions for residues <b>past position 1024 are extrapolation</b> and should be
treated with caution. When you run an AI head on a sequence longer than 1024 residues, BEER
shows a one-time notice. Classical (non-AI) analyses have no length limit.</p>
<h2>All 24 Heads</h2>
<table>
  <tr><th>Head</th><th>AUROC</th><th>Primary Training Source</th></tr>
  <tr><td>Signal Peptide</td><td>0.9999</td><td>UniProt ft_signal (Swiss-Prot)</td></tr>
  <tr><td>Disulfide Bond</td><td>0.9990</td><td>UniProt ft_disulfid</td></tr>
  <tr><td>Zinc Finger</td><td>0.9972</td><td>BioLiP Zn-coordinating residues</td></tr>
  <tr><td>Methylation</td><td>0.9965</td><td>dbPTM</td></tr>
  <tr><td>Transit Peptide</td><td>0.9950</td><td>UniProt ft_transit</td></tr>
  <tr><td>Glycosylation</td><td>0.9938</td><td>GlyConnect site-resolved glycoproteomics</td></tr>
  <tr><td>DNA-Binding</td><td>0.9936</td><td>BioLiP (PDB-derived protein-DNA contacts)</td></tr>
  <tr><td>Disorder</td><td>0.9923</td><td>DisProt + UniProt ft_region:disordered</td></tr>
  <tr><td>Lipidation</td><td>0.9928</td><td>UniProt ft_lipid</td></tr>
  <tr><td>Active Site</td><td>0.9918</td><td>M-CSA mechanistically validated catalytic residues</td></tr>
  <tr><td>Acetylation</td><td>0.9894</td><td>dbPTM</td></tr>
  <tr><td>Propeptide</td><td>0.9879</td><td>UniProt ft_propep</td></tr>
  <tr><td>Transmembrane</td><td>0.9870</td><td>UniProt ft_transmem (BiLSTM-CRF)</td></tr>
  <tr><td>Ubiquitination</td><td>0.9868</td><td>dbPTM</td></tr>
  <tr><td>Intramembrane</td><td>0.9853</td><td>UniProt ft_intramem</td></tr>
  <tr><td>Repeat Region</td><td>0.9835</td><td>UniProt ft_repeat</td></tr>
  <tr><td>Coiled-Coil</td><td>0.9819</td><td>UniProt ft_coiled</td></tr>
  <tr><td>Phosphorylation</td><td>0.9779</td><td>dbPTM (PSP + PhosphoELM + HPRD aggregate)</td></tr>
  <tr><td>Nucleotide-Binding</td><td>0.9774</td><td>BioLiP (ATP/ADP/NAD/FAD/CoA/…)</td></tr>
  <tr><td>Functional Motif</td><td>0.9722</td><td>UniProt ft_motif</td></tr>
  <tr><td>Low-Complexity</td><td>0.9688</td><td>UniProt ft_compbias</td></tr>
  <tr><td>Binding Site</td><td>0.9631</td><td>BioLiP small-molecule binding residues</td></tr>
  <tr><td>RNA Binding</td><td>0.9198</td><td>BioLiP (PDB-derived protein-RNA contacts)</td></tr>
  <tr><td>Secondary Structure (SS3)</td><td>0.843 Q3</td><td>UniProt ft_helix / ft_strand (BiLSTM-CRF Q3)</td></tr>
</table>
<h2>UniProt Annotation Overlay</h2>
<p>Each graph tab shows the AI prediction probability curve as the primary element.
When <b>UniProt Tracks</b> are fetched (Structure toolbar), the curated Swiss-Prot annotation
for that feature is overlaid on the same axes as a semi-transparent background span and a rug strip.
A stats box shows sensitivity and precision of the AI prediction vs the UniProt reference.</p>
<h2>MC-Dropout Uncertainty</h2>
<p>For profile graphs, enable <b>Show Uncertainty (MC-Dropout)</b> to run 20 stochastic forward passes
(Gal &amp; Ghahramani 2016), producing per-residue mean ± 1σ confidence intervals shown as a shaded band.</p>
<h2>ESMC Status Indicator</h2>
<p>The <b>ESMC</b> status badge (top-right of the main window) shows: Ready, Busy, or Offline.
AI predictions require ESMC to be available. When ESMC is offline, AI prediction tabs are disabled.</p>
"""),
            ("Multichain & Compare", """
<h1>Multichain Analysis</h1>
<p>When a multi-FASTA file or a PDB/mmCIF with multiple chains is imported (or a multi-chain PDB
ID is fetched), every chain is analysed and listed in the <b>Multichain Analysis</b> table.
Double-click any row to load that chain into the Analysis section, or use the <b>Chain</b>
selector that appears above the sequence box on the Analysis tab to switch chains.</p>
<p>Selecting a chain updates that chain's sequence and per-chain profiles (pLDDT, distance map,
Ramachandran, SASA); the 3D viewer keeps showing the full assembly so the per-chain visibility
controls stay available.</p>
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

        # Wire search bar: filter sections by name OR find text in page
        _section_names = [name for name, _ in _HELP_SECTIONS]

        def _on_help_search(text: str) -> None:
            q = text.strip().lower()
            if not q:
                # Show all sections, clear any highlights
                for i in range(help_nav.count()):
                    help_nav.item(i).setHidden(False)
                cur = help_stack.currentWidget()
                if cur:
                    browser_w = cur.findChild(QTextBrowser)
                    if browser_w:
                        browser_w.find("")  # clear highlight
                return
            # Filter nav by section name
            matched_idx = -1
            for i, name in enumerate(_section_names):
                hidden = q not in name.lower()
                help_nav.item(i).setHidden(hidden)
                if not hidden and matched_idx < 0:
                    matched_idx = i
            # Select the first visible match and search within it
            if matched_idx >= 0:
                help_nav.setCurrentRow(matched_idx)
            # Also search within the currently displayed page
            cur = help_stack.currentWidget()
            if cur:
                browser_w = cur.findChild(QTextBrowser)
                if browser_w:
                    browser_w.find(text)

        self._help_search.textChanged.connect(_on_help_search)

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
             of Protein Sequences with ESMC-Augmented Predictors},
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
        esm2_str = (f" ESMC embeddings (model: {esm2}) were used to augment "
                    f"disorder, aggregation, signal peptide, and PTM predictions.")  \
                   if esm2 else ""
        # Provenance: stamp versions and the parameters actually used so the run
        # is reproducible from the paragraph alone.
        try:
            import beer as _bp
            _beer_ver = getattr(_bp, "__version__", "3")
        except Exception:
            _beer_ver = "3"
        _esmc_ver = f", ESMC model {esm2}" if esm2 else ""
        _manual_pka = len([p for p in self.pka_input.text().split(",")
                           if p.strip()]) == 9 if hasattr(self, "pka_input") else False
        _pka_str = (f", pKa set = {getattr(self, 'pka_set', DEFAULT_PKA_SET)}"
                    + (" (custom override)" if _manual_pka else ""))

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
            f"Analysis parameters: hydrophobicity scale = {self.hydro_scale}, "
            f"sliding window size = {self.default_window_size}, pH = {self.default_pH:.1f}"
            f"{_pka_str}. Software: BEER v{_beer_ver}"
            f"{_esmc_ver}."
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
        self._save_and_reset()
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
        self._save_and_reset()
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
            self._struct_source = "pdb"
            self._reset_struct_view()
            self._load_structure_viewer(pdb_str, compute_sasa=False)
            self.export_structure_btn.setEnabled(True)
            self.predict_struct_btn.setEnabled(True)
            n_res = sum(len(seq) for _, seq in entries)
            self._set_status_lbl(
                self.af_status_lbl,
                f"Loaded {os.path.basename(file_name)}  —  "
                f"{len(chain_structs)} chain(s), {n_res} residues total",
                "success",
            )
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
        self._save_and_reset()
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
            self._load_structure_viewer(cif_str, compute_sasa=False)
            self.export_structure_btn.setEnabled(True)
            n_res = sum(len(seq) for _, seq in entries)
            self._set_status_lbl(
                self.af_status_lbl,
                f"Loaded {os.path.basename(file_name)}  —  "
                f"{len(chain_structs)} chain(s), {n_res} residues total",
                "success",
            )
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
        counts = self.analysis_data.get("aa_counts", {})
        freq   = self.analysis_data.get("aa_freq", {})
        if not counts or not freq:
            return
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

    def _on_analyze_btn_clicked(self):
        """Analyse button: reset only when a previously analysed sequence is being replaced."""
        raw = self.seq_text.toPlainText().strip()
        if not raw:
            QMessageBox.warning(self, "Input", "Enter or paste a sequence.")
            return
        if self.analysis_data:
            entries = self._parse_pasted_text(raw)
            new_seq = entries[0][1] if entries else ""
            if new_seq != self.analysis_data.get("seq", ""):
                self._save_and_reset()
                self.seq_text.setPlainText(raw)
        self.on_analyze()

    def on_analyze(self):
        if self._analysis_worker is not None and self._analysis_worker.isRunning():
            return
        raw = self.seq_text.toPlainText()
        if not raw.strip():
            QMessageBox.warning(self, "Input", "Enter or paste a sequence.")
            return

        pH = self.ph_input.value()

        entries = self._parse_pasted_text(raw)
        if not entries:
            QMessageBox.warning(
                self, "Invalid Input",
                "No valid protein sequences found.\n"
                "Ensure sequences contain only standard amino acid letters (ACDEFGHIKLMNPQRSTVWY)."
            )
            return
        entries = [(rid, seq) for rid, seq in entries if len(seq) >= 5]
        if not entries:
            QMessageBox.warning(
                self, "Sequence Too Short",
                "Sequences must be at least 5 amino acids long to analyse."
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

        # Warn if non-canonical residues present (analysis proceeds, positions blank in profiles)
        _CANON_20 = set("ACDEFGHIKLMNPQRSTVWY")
        _noncanon = sorted(set(aa for aa in seq if aa not in _CANON_20))
        if _noncanon:
            QMessageBox.warning(
                self, "Non-canonical residues",
                f"Sequence contains non-canonical residues: {', '.join(_noncanon)}\n\n"
                "Analysis will proceed. Scale-based profiles will show 0 at these "
                "positions (effectively blank). Composition counts will omit them.",
            )

        # Clear chain combo when manually typing
        if not self.batch_data:
            self.chain_combo.clear()
            self.chain_combo.setEnabled(False)

        # ── One-time ESMC download warning ───────────────────────────────────
        from beer.embeddings import ESMC_AVAILABLE
        if ESMC_AVAILABLE and self._embedder is not None:
            import pathlib
            _mn = getattr(self._embedder, "model_name", "esmc_600m")
            _hf_hub = pathlib.Path.home() / ".cache/huggingface/hub"
            _downloaded = any(_hf_hub.glob("models--EvolutionaryScale--esmc-*"))
            if not _downloaded and not getattr(self, "_esm2_download_warned", False):
                self._esm2_download_warned = True
                _sizes = {
                    "esm2_t6_8M_UR50D": "~30 MB", "esm2_t12_35M_UR50D": "~140 MB",
                    "esmc_300m": "~1.2 GB", "esmc_600m": "~2.4 GB",
                }
                _sz = _sizes.get(_mn, "~2.6 GB")
                reply = QMessageBox.information(
                    self, "First-time ESMC Setup",
                    f"<b>One-time model download required</b><br><br>"
                    f"The ESMC 600M language model ({_sz}) will be downloaded "
                    f"from Meta's model hub on the first analysis.<br><br>"
                    f"<b>Estimated time:</b> 2–15 minutes depending on your connection.<br>"
                    f"<b>Location:</b> <code>~/.cache/huggingface/hub/</code><br><br>"
                    f"BEER will appear frozen during the download — this is normal. "
                    f"The model is cached permanently; subsequent runs are instant.",
                    QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
                )
                if reply == QMessageBox.StandardButton.Cancel:
                    return

        self._last_was_bilstm = False
        self.analyze_btn.setEnabled(False)
        self.statusBar.showMessage("Analyzing…")

        self._progress_dlg = QProgressDialog(
            "Running analysis…", "Cancel", 0, 0, self)
        self._progress_dlg.setWindowTitle("BEER Analysis")
        self._progress_dlg.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress_dlg.setMinimumDuration(500)
        self._progress_dlg.canceled.connect(self._cancel_analysis)
        self._progress_dlg.show()

        # Retire any previous worker safely before replacing the reference.
        if self._analysis_worker is not None:
            self._discard_worker(self._analysis_worker)
            self._analysis_worker = None

        # Classical analysis: embedder=None skips ESMC and all BiLSTM heads
        self._analysis_worker = AnalysisWorker(
            seq, pH, self.default_window_size, self.use_reducing, self.custom_pka,
            hydro_scale=self.hydro_scale,
            embedder=None,
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
        self._populate_sasa_report_section()

    def _update_seq_viewer(self, highlight_pattern: str = ""):
        """Refresh the sequence viewer panel with colour-coded residues (UniProt style)."""
        if not self.analysis_data:
            return
        seq  = self.analysis_data["seq"]
        name = self.sequence_name or ""
        text = format_sequence_block(seq, name=name)

        # Pre-compile highlight pattern — guard against ReDoS from user input.
        # Reject nested quantifiers and excessively long patterns.
        hl_re = None
        if highlight_pattern and len(highlight_pattern) <= 80:
            _pat = highlight_pattern.upper()
            _safe = not bool(re.search(r'\([^)]*[+*][^)]*\)[+*?]|\(\?[^)]*\)[+*]', _pat))
            if _safe:
                try:
                    hl_re = re.compile(_pat)
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
        hdr_color  = "#7b9cff" if is_dark else "#4361ee"

        lines      = text.split("\n")
        html_lines = []
        for ln in lines:
            if ln.startswith(">"):
                html_lines.append(
                    f'<span style="color:{hdr_color};font-weight:700;">{_html_mod.escape(ln)}</span>'
                )
            elif ln and ln.lstrip()[0:1].isdigit():
                lstripped = ln.lstrip()
                parts = lstripped.split("  ", 1)
                if len(parts) == 2:
                    indent = " " * (len(ln) - len(lstripped))
                    pos_str = indent + parts[0]
                    seq_str = parts[1]
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
            f'<style>body{{font-family:"Courier New",Courier,monospace;font-size:10pt;'
            f'background:{bg_color};padding:10px 12px;line-height:1.7;'
            f'white-space:pre;}}</style>'
            + "<br>".join(html_lines)
        )
        self.seq_viewer.setHtml(html)

    def update_graph_tabs(self):
        """Register lazy graph generators; only render the currently-visible graph now."""
        if not self.analysis_data:
            return
        self._zip_btns = []   # reset so stale (destroyed) buttons don't accumulate
        self._build_graph_generators()
        self._generated_graphs.clear()
        self._populate_overlay_checklist()
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
            _prov = f"BEER v3.0  |  {sn}" if sn else "BEER v3.0"
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
        # BiLSTM heads — registered whenever the data key is present in analysis_data.
        # This covers lazy per-head computation triggered by graph/report visibility.
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
            ("ss3_h_profile",  "secondary_structure_helix",   "Secondary Structure: Helix Profile"),
            ("ss3_e_profile",  "secondary_structure_strand",  "Secondary Structure: Strand Profile"),
            ("ss3_c_profile",  "secondary_structure_coil",    "Secondary Structure: Coil Profile"),
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
            elif _tab_name not in gens:
                gens[_tab_name] = (
                    lambda t=_tab_name, f=_feat:
                    self._make_training_placeholder_fig(t, f)
                )
        gens["Uversky Phase Plot"] = lambda: _wrap(lambda: create_uversky_phase_plot(
            seq, label_font=lf, tick_font=tf))
        gens["Single-Residue Perturbation Map"] = lambda: _wrap(lambda: create_saturation_mutagenesis_figure(
            seq, label_font=lf, tick_font=tf, cmap=hcm))
        gens["Cleavage Map"] = lambda: _wrap(lambda: create_cleavage_map_figure(
            seq, ad.get("prot_sites", {}), label_font=lf, tick_font=tf))
        _pfam = list(self.pfam_domains) if self.pfam_domains else []
        if _pfam:
            gens["Pfam Domains"] = lambda _p=_pfam: _wrap(
                lambda: create_domain_architecture_figure(
                    len(seq), _p, label_font=lf, tick_font=tf))
        if _HAS_AGGREGATION:
            _aggr_prof = ad.get("aggr_profile", calc_aggregation_profile(seq))
            _hotspots = predict_aggregation_hotspots(seq)  # sequence-based; no UniProt needed
            gens["\u03b2-Aggregation Profile"] = lambda _h=_hotspots: _wrap(
                lambda: create_aggregation_profile_figure(
                    seq, _aggr_prof, _h, label_font=lf, tick_font=tf))
            gens["Solubility Profile"] = lambda: _wrap(lambda: create_solubility_profile_figure(
                seq, calc_camsolmt_score(seq), label_font=lf, tick_font=tf))
        _am_data = getattr(self, "_alphafold_missense_data", None)
        if _am_data:
            gens["AlphaMissense"] = lambda: _wrap(lambda: create_alphafold_missense_figure(
                _am_data, seq=seq, label_font=lf, tick_font=tf, cmap=self.heatmap_cmap))
        else:
            gens["AlphaMissense"] = lambda: self._make_unavail_fig(
                "AlphaMissense Data Not Loaded",
                "Fetch a UniProt entry first, then click the \u201cAlphaMissense\u201d chip.",
                is_dark=getattr(self, "_is_dark", False),
            )
        if _HAS_AMPHIPATHIC:
            _amph = ad.get("amph_regions", []) if _uniprot_feats else []
            gens["Hydrophobic Moment"] = lambda _a=_amph: _wrap(
                lambda: create_hydrophobic_moment_figure(
                    seq, ad.get("moment_alpha", []), ad.get("moment_beta", []),
                    _a, label_font=lf, tick_font=tf))
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
        _STRUCT_MSG = (
            "Load a 3D structure to see this graph.\n\n"
            "Use Fetch AlphaFold, Import PDB, or Fetch PDB ID."
        )
        _is_dk = getattr(self, "_is_dark", False)

        # Determine primary/secondary data sources from _graph_struct_src selector
        _src     = getattr(self, "_struct_source",    "none")
        _gsrc    = getattr(self, "_graph_struct_src", "alphafold")
        _in_ovl  = (_src == "overlay")
        esmd     = self.esmfold2_data

        # Primary afd: the single structure shown when not in "both" mode
        if _in_ovl and _gsrc == "esmfold2":
            afd = esmd
        elif _src == "esmfold2":
            afd = esmd
        else:
            afd = self.alphafold_data

        # Secondary afd2: used only when _gsrc == "both" and overlay active
        afd2 = esmd if (_in_ovl and _gsrc == "both" and esmd) else None

        # pLDDT source label
        _plddt_src = "esmfold2" if (_gsrc == "esmfold2" or _src == "esmfold2") else "alphafold"

        # SASA — primary source
        _rsa_p = dict(getattr(self, "_struct_sasa_data", {}) or {})
        _asa_p = dict(getattr(self, "_struct_sasa_raw",  {}) or {})
        if _in_ovl and _gsrc == "esmfold2" and esmd and "sasa_data" in esmd:
            _rsa_p = dict(esmd["sasa_data"])
            _asa_p = dict(esmd.get("sasa_raw", {}))
        # SASA — secondary (ESM) for "both"
        _rsa2 = dict(esmd.get("sasa_data", {})) if afd2 and "sasa_data" in esmd else None
        _asa2 = dict(esmd.get("sasa_raw",  {})) if afd2 and "sasa_data" in esmd else None

        _win      = self.default_window_size
        _show_asa = getattr(self, "_sasa_show_asa", False)

        if _rsa_p:
            gens["SASA Profile"] = lambda: _wrap(lambda: create_sasa_figure(
                _rsa_p, _asa_p, window=_win, show_asa=_show_asa,
                label_font=lf, tick_font=tf,
                other_rsa=_rsa2, other_asa=_asa2))
        else:
            gens["SASA Profile"] = lambda: self._make_unavail_fig(
                "SASA Profile", _STRUCT_MSG, is_dark=_is_dk)

        if afd:
            # pLDDT Profile
            plddt = afd.get("plddt")
            _is_bfac = _plddt_src not in ("alphafold", "esmfold2")
            if plddt and len(plddt) == len(seq):
                _other_pl = list(afd2.get("plddt") or []) if afd2 else None
                gens["pLDDT Profile"] = lambda _pl=list(plddt), _ops=_other_pl, _ps=_plddt_src, _ibf=_is_bfac: _wrap(
                    lambda: create_plddt_figure(
                        _pl, label_font=lf, tick_font=tf,
                        use_bfactor=_ibf, source=_ps,
                        other_plddt=_ops))
            else:
                gens["pLDDT Profile"] = lambda: self._make_unavail_fig(
                    "pLDDT / B-factor Profile", _STRUCT_MSG, is_dark=_is_dk)

            # Distance Map
            dm  = afd.get("dist_matrix")
            dm2 = afd2.get("dist_matrix") if afd2 else None
            if dm is not None and dm.ndim == 2 and dm.shape[0] == len(seq) > 0:
                gens["Distance Map"] = lambda _dm=dm, _dm2=dm2: _wrap(
                    lambda: create_distance_map_figure(
                        _dm, label_font=lf, tick_font=tf, cmap=hcm,
                        other_dist_matrix=_dm2))
            else:
                gens["Distance Map"] = lambda: self._make_unavail_fig(
                    "Distance Map", _STRUCT_MSG, is_dark=_is_dk)

            # Ramachandran
            if _HAS_PHI_PSI:
                _pp1 = _extract_phi_psi(afd.get("pdb_str", ""))
                _pp2 = _extract_phi_psi(afd2.get("aligned_pdb") or afd2.get("pdb_str", "")) if afd2 else None
                gens["Ramachandran Plot"] = lambda _p=_pp1, _p2=_pp2: _wrap(
                    lambda: create_ramachandran_figure(
                        _p, label_font=lf, tick_font=tf, other_phi_psi=_p2))
            else:
                gens["Ramachandran Plot"] = lambda: self._make_unavail_fig(
                    "Ramachandran Plot", _STRUCT_MSG, is_dark=_is_dk)

            # Structure Comparison — Cα RMSD only, available in overlay mode
            if esmd and _src == "overlay":
                _rmsd = esmd.get("rmsd_per_res") or []
                gens["Structure Comparison"] = lambda _r=_rmsd: _wrap(
                    lambda: create_structure_comparison_figure(
                        [], [], _r or None, label_font=lf, tick_font=tf))
        else:
            for _t in ("pLDDT Profile", "Distance Map",
                       "Ramachandran Plot", "Structure Comparison"):
                gens[_t] = lambda t=_t: self._make_unavail_fig(
                    t, _STRUCT_MSG, is_dark=_is_dk)

        if "Structure Comparison" not in gens:
            gens["Structure Comparison"] = lambda: self._make_unavail_fig(
                "Structure Comparison",
                "Fetch an AlphaFold structure and predict with ESMFold2 to compare "
                "confidence profiles with Cα RMSD overlay.",
                is_dark=_is_dk)

        # MSA
        if self._msa_sequences:
            gens["MSA Conservation"] = lambda: _wrap(lambda: create_msa_conservation_figure(
                self._msa_sequences, self._msa_names, label_font=lf, tick_font=tf))
        if self._msa_mi_apc is not None:
            gens["MSA Covariance"] = lambda: _wrap(lambda: create_msa_covariance_figure(
                self._msa_mi_apc, label_font=lf, tick_font=tf, cmap=hcm))

        # Variant Effect Map (ESMC)
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
        """Return a placeholder immediately and start VariantEffectWorker in background."""
        from beer.network.workers import VariantEffectWorker
        if self._variant_effect_worker and self._variant_effect_worker.isRunning():
            self._variant_effect_worker.quit()
            self._variant_effect_worker.wait(_WORKER_WAIT_MS)
        worker = VariantEffectWorker(seq, self._embedder, parent=self)
        worker.finished.connect(
            lambda llr, s=seq, l=lf, t=tf, c=cmap: self._on_variant_effect_done(llr, s, l, t, c))
        worker.error.connect(self._on_variant_effect_error)
        self._variant_effect_worker = worker
        self._variant_effect_worker_cmap = cmap
        worker.start()
        return self._make_unavail_fig(
            "Computing Variant Effect Map…",
            "ESMC inference is running in the background.\nThe map will appear when ready.",
            is_dark=getattr(self, "_is_dark", False),
        )

    def _on_variant_effect_done(self, llr, seq: str, lf: int, tf: int, cmap: str):
        from beer.graphs.variant_map import create_variant_effect_figure
        try:
            fig = create_variant_effect_figure(seq, llr, label_font=lf, tick_font=tf, cmap=cmap)
            self._replace_graph("Variant Effect Map", fig)
            self._generated_graphs.add("Variant Effect Map")
        except Exception as exc:
            import logging as _log
            _log.getLogger("beer.graphs").warning(
                "Variant effect figure failed: %s", exc, exc_info=True)

    def _on_variant_effect_error(self, msg: str):
        import logging as _log
        _log.getLogger("beer.graphs").warning("Variant effect worker error: %s", msg)
        fig = self._make_unavail_fig(
            "Variant Effect Map Unavailable",
            msg,
            is_dark=getattr(self, "_is_dark", False),
        )
        self._replace_graph("Variant Effect Map", fig)
        self._generated_graphs.add("Variant Effect Map")

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
                self._reset_ai_state()   # drop previous chain's AI results
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

    # --- Graph tree handlers ---

    def _on_graph_tree_expanded(self, item: QTreeWidgetItem) -> None:
        txt = item.text(0)
        if txt.startswith("▶"):
            item.setText(0, "▼" + txt[1:])

    def _on_graph_tree_collapsed(self, item: QTreeWidgetItem) -> None:
        txt = item.text(0)
        if txt.startswith("▼"):
            item.setText(0, "▶" + txt[1:])

    def _on_report_tree_expanded(self, item: QTreeWidgetItem) -> None:
        txt = item.text(0)
        if txt.startswith("▶"):
            item.setText(0, "▼" + txt[1:])

    def _on_report_tree_collapsed(self, item: QTreeWidgetItem) -> None:
        txt = item.text(0)
        if txt.startswith("▼"):
            item.setText(0, "▶" + txt[1:])

    def _on_graph_tree_clicked(self, item: QTreeWidgetItem, _col: int):
        title = item.data(0, Qt.ItemDataRole.UserRole)
        if not title or title not in self._graph_title_to_stack_idx:
            # Category row — toggle expand/collapse
            item.setExpanded(not item.isExpanded())
            return
        # Self-contained graphs (MSA, etc.) populate without the Analysis tab —
        # only block leaves that have no content yet.
        if not self.analysis_data and title not in self._generated_graphs:
            QMessageBox.information(
                self, "Analysis Required",
                "<b>No analysis data available.</b><br><br>"
                "Enter a protein sequence in the Analysis tab and click "
                "<b>Analyze</b> to compute biophysical properties.<br><br>"
                "Graphs will become available once analysis is complete.")
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
        if not self.analysis_data:
            return
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
                ("Feature track (.gff3) — requires analysis",          "gff3", has_seq),
                ("PyMOL colouring (.pml) — requires analysis",         "pml",  has_seq),
                ("ChimeraX colouring (.cxc) — requires analysis",      "cxc",  has_seq),
                ("Structure coloured by track (.pdb) — needs structure", "color_pdb",
                 has_struct and has_seq),
            ],
            self,
        )
        if dlg.exec() != QDialog.Accepted:
            return
        key = dlg.selected_key()
        if key in ("gff3", "pml", "cxc", "color_pdb"):
            self._export_track(key)
        else:
            self._export_structure_as(key)

    def _export_track(self, key: str) -> None:
        """Export a GFF3 feature track or a PyMOL/ChimeraX per-residue colouring."""
        from beer.io import track_export as _te
        data = self.analysis_data or {}
        name = self.sequence_name or "protein"
        if key == "gff3":
            text = _te.features_to_gff3(data, name)
            fn, _ = QFileDialog.getSaveFileName(
                self, "Export Feature Track", f"{name}.gff3", "GFF3 (*.gff3 *.gff)")
            if not fn:
                return
            with open(fn, "w") as f:
                f.write(text)
            self.statusBar.showMessage(
                f"Feature track exported → {os.path.basename(fn)}", 4000)
            return
        # pml / cxc need a per-residue track to colour by
        tracks = _te.available_colour_tracks(data)
        if not tracks:
            QMessageBox.information(
                self, "No Tracks Computed",
                "No per-residue AI tracks are computed yet.\n\n"
                "Open an AI prediction (e.g. Disorder) in the Graphs or Reports tab "
                "first, then export again.")
            return
        chooser = FormatChooserDialog(
            "Colour by which track?",
            [(lbl, k, True) for k, lbl in tracks], self)
        if chooser.exec() != QDialog.Accepted:
            return
        tkey = chooser.selected_key()
        tlabel = dict((k, l) for k, l in tracks).get(tkey, tkey)
        # Use the structure's actual primary chain (not hard-coded "A").
        pdb_str = (self.alphafold_data or {}).get("pdb_str", "") or ""
        _chains = self._parse_pdb_chains(pdb_str) if pdb_str else []
        chain = _chains[0] if _chains else "A"
        if key == "color_pdb":
            text = _te.coloring_to_bfactor_pdb(pdb_str, data.get(tkey))
            ext, flt, what = "pdb", "PDB Files (*.pdb)", "B-factor-coloured PDB"
        elif key == "pml":
            text = _te.coloring_to_pymol(data, tkey, tlabel, chain)
            ext, flt, what = "pml", "PyMOL Scripts (*.pml)", "Colouring script"
        else:
            text = _te.coloring_to_chimerax(data, tkey, tlabel, chain)
            ext, flt, what = "cxc", "ChimeraX Scripts (*.cxc)", "Colouring script"
        if not text:
            QMessageBox.information(self, "No Data", "That track has no scores.")
            return
        fn, _ = QFileDialog.getSaveFileName(
            self, f"Export {what}", f"{name}_{tkey}.{ext}", flt)
        if not fn:
            return
        with open(fn, "w") as f:
            f.write(text)
        self.statusBar.showMessage(
            f"{what} exported → {os.path.basename(fn)}", 4000)

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
                pdb_str = (self.alphafold_data or {}).get("pdb_str") or ""
                if not pdb_str:
                    QMessageBox.warning(self, "Export Error", "No structure data available.")
                    return
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
        # Offer all formats in the save dialog's file-type selector (default first).
        _filters = {"png": "PNG image (*.png)", "svg": "SVG vector (*.svg)", "pdf": "PDF (*.pdf)"}
        _default = (fmt or self.default_graph_format).lower()
        if _default not in _filters:
            _default = "png"
        _order = [_default] + [k for k in ("png", "svg", "pdf") if k != _default]
        filter_str = ";;".join(_filters[k] for k in _order)
        safe = title.replace("/", "-").replace("\\", "-")
        fn, sel = QFileDialog.getSaveFileName(
            self, "Save Graph", f"{safe}.{_default}", filter_str)
        if not fn:
            return
        # Resolve format: typed extension wins, else the chosen file-type filter.
        ext = next((k for k in _filters if fn.lower().endswith(f".{k}")), None)
        if ext is None:
            ext = next((k for k in _filters if _filters[k] == sel), _default)
            fn += f".{ext}"
        use_transparent = self.transparent_bg and ext in ("png", "svg")
        with self._figure_export_light(canvas.figure):
            canvas.figure.savefig(
                fn, format=ext, dpi=300, bbox_inches="tight",
                facecolor="none" if use_transparent else "white",
                transparent=use_transparent, metadata=figure_metadata(ext)
            )
        QMessageBox.information(self, "Saved", f"{title} → {fn}")

    def _copy_graph_to_clipboard(self, title: str):
        _, vb = self.graph_tabs[title]
        canvas = self._find_canvas(vb)
        if not canvas:
            return
        buf = BytesIO()
        with self._figure_export_light(canvas.figure):
            canvas.figure.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        QApplication.clipboard().setImage(QImage.fromData(buf.getvalue()))
        self.statusBar.showMessage("Figure copied to clipboard.", 2000)

    def _export_all_graphs_zip(self):
        """Package all generated graph figures as PNGs in a ZIP archive."""
        import zipfile
        if not self._generated_graphs:
            QMessageBox.warning(self, "No Graphs", "No graphs have been generated yet.")
            return
        fn, _ = QFileDialog.getSaveFileName(
            self, "Export All Graphs", "beer_graphs.zip", "ZIP Archives (*.zip)")
        if not fn:
            return
        if not fn.lower().endswith(".zip"):
            fn += ".zip"
        try:
            use_transparent = self.transparent_bg
            with zipfile.ZipFile(fn, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for title in sorted(self._generated_graphs):
                    if title not in self.graph_tabs:
                        continue
                    _, vb = self.graph_tabs[title]
                    canvas = self._find_canvas(vb)
                    if not canvas:
                        continue
                    buf = BytesIO()
                    with self._figure_export_light(canvas.figure):
                        canvas.figure.savefig(
                            buf, format="png", dpi=300, bbox_inches="tight",
                            facecolor="none" if use_transparent else "white",
                            transparent=use_transparent, metadata=figure_metadata("png"),
                        )
                    safe_name = title.replace("/", "-").replace("\\", "-")
                    zf.writestr(f"{safe_name}.png", buf.getvalue())
            n = len(self._generated_graphs)
            self.statusBar.showMessage(f"Exported {n} graph(s) to {os.path.basename(fn)}", 5000)
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def _bundle_provenance_text(self) -> str:
        """Plain-text provenance + run parameters for the publication bundle."""
        from beer.io.provenance import text_header
        ad = self.analysis_data or {}
        esm = getattr(self._embedder, "model_name", None)
        lines = [text_header("", title="BEER publication bundle").rstrip("\n"), ""]
        lines += [
            f"Sequence name   : {self.sequence_name or 'protein'}",
            f"Length          : {len(ad.get('seq', ''))} aa",
            f"pH              : {self.default_pH}",
            f"Sliding window  : {self.default_window_size}",
            f"Hydrophobicity  : {self.hydro_scale}",
            f"pKa set         : {getattr(self, 'pka_set', DEFAULT_PKA_SET)}",
            f"ESMC model      : {esm or 'not loaded (classical only)'}",
        ]
        return "\n".join(lines) + "\n"

    def _bundle_report_html(self) -> str:
        secs = (self.analysis_data or {}).get("report_sections", {})
        body = "\n<hr/>\n".join(h for h in secs.values() if h)
        return ("<!DOCTYPE html><html><head><meta charset='utf-8'>"
                "<title>BEER report</title></head><body>" + body + "</body></html>")

    def _bundle_all_tracks_csv(self) -> str:
        """Wide CSV: every full-length per-residue track aligned by position."""
        ad = self.analysis_data or {}
        seq = ad.get("seq", "")
        L = len(seq)
        if not L:
            return ""
        cols = [("position", list(range(1, L + 1))), ("residue", list(seq))]
        candidates = [("disorder_scores", "Disorder")]
        candidates += [(dk, dn) for dn, dk, _gt, _au in _AI_HEAD_SPECS]
        _seen = set()
        for key, label in candidates:
            if key in _seen:
                continue
            v = ad.get(key)
            if isinstance(v, list) and len(v) == L:
                _seen.add(key)
                cols.append((label, [round(float(x), 6) for x in v]))
        header = ",".join(c[0] for c in cols)
        rows = [",".join(str(c[1][i]) for c in cols) for i in range(L)]
        return header + "\n" + "\n".join(rows) + "\n"

    def _export_publication_bundle(self):
        """One-click bundle: vector figures + per-graph data + a wide tracks CSV +
        the combined report + a provenance file, packaged as a single ZIP."""
        import zipfile
        if not self.analysis_data:
            QMessageBox.warning(self, "No Data", "Run analysis first.")
            return
        if not self._generated_graphs:
            QMessageBox.warning(self, "No Graphs",
                                "Open at least one graph first so it can be bundled.")
            return
        chooser = FormatChooserDialog(
            "Figure format for the bundle",
            [("SVG (vector, editable)", "svg", True), ("PDF (vector)", "pdf", True)],
            self)
        if chooser.exec() != QDialog.Accepted:
            return
        vext = chooser.selected_key()
        name = (self.sequence_name or "protein").replace(" ", "_")
        fn, _ = QFileDialog.getSaveFileName(
            self, "Export Publication Bundle",
            f"beer_bundle_{name}.zip", "ZIP Archives (*.zip)")
        if not fn:
            return
        if not fn.lower().endswith(".zip"):
            fn += ".zip"
        try:
            afd = self.alphafold_data or {}
            extra = {
                "pfam_domains":       self.pfam_domains,
                "plddt":              afd.get("plddt", []),
                "alphafold_missense": getattr(self, "_alphafold_missense_data", None) or {},
                "msa_sequences":      self._msa_sequences,
            }
            n_fig = n_data = 0
            with zipfile.ZipFile(fn, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for title in sorted(self._generated_graphs):
                    if title not in self.graph_tabs:
                        continue
                    canvas = self._find_canvas(self.graph_tabs[title][1])
                    if canvas is None:
                        continue
                    buf = BytesIO()
                    with self._figure_export_light(canvas.figure):
                        canvas.figure.savefig(buf, format=vext, bbox_inches="tight",
                                              metadata=figure_metadata(vext))
                    safe = title.replace("/", "-").replace("\\", "-")
                    zf.writestr(f"figures/{safe}.{vext}", buf.getvalue())
                    n_fig += 1
                    res = get_graph_data(title, self.analysis_data, extra)
                    if res is not None:
                        stem, content, dext = res
                        zf.writestr(f"data/{stem}.{dext}", content)
                        n_data += 1
                tracks = self._bundle_all_tracks_csv()
                if tracks:
                    zf.writestr("data/all_residue_tracks.csv", tracks)
                zf.writestr("report.html", self._bundle_report_html())
                zf.writestr("PROVENANCE.txt", self._bundle_provenance_text())
            self.statusBar.showMessage(
                f"Publication bundle → {os.path.basename(fn)}", 5000)
            QMessageBox.information(
                self, "Bundle Exported",
                f"Wrote {n_fig} vector figure(s), {n_data} data file(s), a combined "
                f"per-residue CSV, report.html and PROVENANCE.txt to:\n{fn}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

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
        "Secondary Structure: Helix Profile":  ("ss3_h_profile",  "secondary_structure_helix"),
        "Secondary Structure: Strand Profile": ("ss3_e_profile",  "secondary_structure_strand"),
        "Secondary Structure: Coil Profile":   ("ss3_c_profile",  "secondary_structure_coil"),
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
        from PySide6.QtWidgets import QApplication as _QApp
        seq = self.analysis_data.get("seq", "")

        # Initialise the ESMC backbone on the MAIN thread before handing off to
        # the worker. Doing the very first torch/model init (download + load)
        # inside a transient QThread can hard-segfault on macOS; the AI-section
        # path already loads it on the main thread the same way. Once loaded
        # here, the worker only runs forward(), which is serialized and safe.
        if self._embedder is not None:
            _QApp.setOverrideCursor(Qt.CursorShape.WaitCursor)
            self.statusBar.showMessage("Preparing ESMC model…", 0)
            try:
                ready = self._embedder.is_available()
            finally:
                _QApp.restoreOverrideCursor()
            if not ready:
                self.statusBar.showMessage(
                    "ESMC model unavailable — cannot compute MC-Dropout.", 4000)
                return

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
        _prov = f"BEER v3.0  |  {self.sequence_name}" if self.sequence_name else "BEER v3.0"
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
        from beer.graphs._style import set_figure_dark
        set_figure_dark(is_dark)   # new figures get themed chrome; export forces light
        if is_dark:
            self.setStyleSheet(DARK_THEME_CSS)
            plt.style.use("dark_background")
            accent = "#7b9cff"
            struct_css = STRUCT_PANEL_CSS_DARK
            struct_title_color = "#7b9cff"
        else:
            self.setStyleSheet(LIGHT_THEME_CSS)
            plt.style.use("default")
            accent = "#4361ee"
            struct_css = STRUCT_PANEL_CSS_LIGHT
            struct_title_color = "#3b4fc8"
        self.main_tabs.set_icon_color("#cdd5ea" if is_dark else "#46506e")

        # Update structure control panel stylesheet
        if hasattr(self, "struct_ctrl_scroll"):
            self.struct_ctrl_scroll.setStyleSheet(struct_css)
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
            self._summary_tab_browser.setHtml(
                self._build_summary_tab_html(self.analysis_data))

        # Update 3D structure viewer background to match theme
        if hasattr(self, "structure_viewer") and self._struct_page_ready:
            bg = "#1a1a2e" if is_dark else "#ffffff"
            self._js(f"setBackground('{bg}');")

        # Update composite structure viewer background to match theme
        if hasattr(self, "_composite_viewer") and self._composite_viewer is not None:
            bg = "#1a1a2e" if is_dark else "#ffffff"
            self._comp_js(f"if(typeof setBackground==='function'){{setBackground('{bg}');}}")

        # Re-render all graphs with the new matplotlib style
        if self.analysis_data:
            self._generated_graphs.clear()
            self._render_visible_graph()

        _config.set_value("theme_dark", is_dark)

        label = "Dark" if is_dark else "Light"
        self.statusBar.showMessage(f"{label} theme activated", 2000)

    def apply_settings(self):
        self.default_window_size = self.window_size_input.value()
        self.default_pH          = self.ph_input.value()
        self.show_bead_labels    = self.label_checkbox.isChecked()
        self.transparent_bg      = self.transparent_bg_checkbox.isChecked()
        # heatmap_cmap is set per-graph via the inline colormap dropdown
        self.label_font_size     = self.label_font_input.value()
        self.tick_font_size      = self.tick_font_input.value()
        self.marker_size         = self.marker_size_input.value()
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

        self.pka_set = self.pka_set_combo.currentText()
        raw_pka = [p.strip() for p in self.pka_input.text().split(",") if p.strip()]
        # Effective pKa table passed to analysis (None = BEER default):
        # a manual 9-value override wins; otherwise the selected named set.
        if self.pka_set == DEFAULT_PKA_SET:
            self.custom_pka = None
        else:
            self.custom_pka = dict(PKA_SETS[self.pka_set])
        if len(raw_pka) == 9:
            try:
                vals = list(map(float, raw_pka))
                self.custom_pka = {
                    'NTERM': vals[0], 'CTERM': vals[1], 'D': vals[2], 'E': vals[3],
                    'C': vals[4], 'Y': vals[5], 'H': vals[6], 'K': vals[7], 'R': vals[8],
                }
            except ValueError:
                QMessageBox.warning(self, "pKa list",
                                    "Custom pKa list could not be parsed – using the selected set.")

        self.enable_tooltips = self.tooltips_checkbox.isChecked()
        self._apply_tooltips()
        if self.theme_toggle.isChecked():
            self.setStyleSheet(DARK_THEME_CSS)
        else:
            self.setStyleSheet(LIGHT_THEME_CSS)
        self._apply_browser_palette()

        if self.analysis_data:
            # Recompute the classical analysis so pH / window / hydrophobicity
            # scale / pKa-set changes take effect immediately on Apply (no manual
            # re-Analyze). Already-computed AI head scores are sequence-only, so
            # they are preserved rather than recomputed.
            _seq = self.analysis_data.get("seq", "")
            if _seq:
                try:
                    _ai_keep = {dk: self.analysis_data[dk]
                                for _dn, dk, _gt, _au in _AI_HEAD_SPECS
                                if self.analysis_data.get(dk)}
                    _new = AnalysisTools.analyze_sequence(
                        _seq, self.default_pH, self.default_window_size,
                        self.use_reducing, self.custom_pka,
                        hydro_scale=self.hydro_scale, embedder=None)
                    _new.update(_ai_keep)
                    self.analysis_data = _new
                except Exception:
                    pass
            _sections = self.analysis_data.get("report_sections", {})
            for sec, browser in self.report_section_tabs.items():
                if sec in _sections:
                    browser.setHtml(_sections[sec])
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
            "pka_set":          self.pka_set,
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
            "esmc_model":           "esmc_600m",
            "hydro_scale":          self.hydro_scale,
            "biohub_api_key":       self.biohub_api_key_input.text().strip(),
        })
        self.statusBar.showMessage("Settings applied and saved.", 5000)

    def reset_defaults(self):
        self.window_size_input.setValue(9)
        self.hydro_scale_combo.setCurrentText("Kyte-Doolittle")
        self.pka_set_combo.setCurrentText(DEFAULT_PKA_SET)
        self.ph_input.setValue(7.4)
        self.pka_input.setText("")
        self._pka_error_lbl.hide()
        self.pka_input.setStyleSheet("")
        self.reducing_checkbox.setChecked(False)
        self.label_checkbox.setChecked(True)
        self.heatmap_cmap = "viridis"
        self.label_font_input.setValue(14)
        self.tick_font_input.setValue(12)
        self.marker_size_input.setValue(10)
        self.graph_color_combo.setCurrentText("Royal Blue")
        self.default_graph_format = "PNG"
        self.heading_checkbox.setChecked(True)
        self.grid_checkbox.setChecked(True)
        self.transparent_bg_checkbox.setChecked(True)
        self.theme_toggle.setChecked(False)
        self.tooltips_checkbox.setChecked(True)
        self.apply_settings()

    # --- AlphaMissense ---

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
        self._mark_chip_error(self.fetch_alphafold_missense_btn)
        QMessageBox.warning(self, "AlphaMissense Error", msg)

    # --- Chain selection ---

    def _restore_chain_structure(self, cid: str):
        """Point the per-chain structure graphs at the selected chain.

        When structure data was loaded per-chain (PDB upload / RCSB PDB fetch),
        batch_struct maps each rec_id to its own struct dict so every chain gets
        its own Ramachandran plot, distance map, pLDDT profile and SASA profile.

        The 3D viewer keeps showing the FULL multi-chain structure (loaded once at
        import/fetch) so the per-chain visibility checkboxes, "Show/Hide all chains"
        controls and chain colouring continue to work. Selecting a chain here must
        NOT reload a single-chain model into the viewer — doing so collapsed the
        structure to one chain and removed the chain controls.

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
        if struct and struct.get("pdb_str"):
            self.export_structure_btn.setEnabled(True)
            # Per-chain SASA so the SASA profile/report match the displayed chain,
            # cached on first use. The 3D viewer still shows the full structure.
            if "sasa_data" not in struct:
                struct["sasa_data"], struct["sasa_raw"] = self._compute_sasa(struct["pdb_str"])
            self._struct_sasa_data = dict(struct["sasa_data"])
            self._struct_sasa_raw  = dict(struct["sasa_raw"])
            self._populate_sasa_report_section()
        else:
            self.export_structure_btn.setEnabled(False)

    def _reset_ai_state(self) -> None:
        """Drop per-protein AI state so a newly selected sequence (e.g. another
        chain) recomputes its own AI heads instead of showing the previous one's.
        Without this, the 'already computed' guard and the cached AI graphs/
        sections leak across chains and the wrong scores are displayed."""
        self._ai_computed_sections.clear()
        self._bilstm_unc_state = {}
        for _t in _GRAPH_TITLE_TO_AI_SEC:
            self._generated_graphs.discard(_t)
        self._setup_ai_section_placeholders()

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
                self._reset_ai_state()   # drop previous chain's AI results
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
        QShortcut(QKeySequence("Ctrl+Enter"),  self, self.on_analyze)
        QShortcut(QKeySequence("Ctrl+G"), self, lambda: self._goto_tab("Graphs"))
        QShortcut(QKeySequence("Ctrl+2"), self, lambda: self._goto_tab("Structure"))
        QShortcut(QKeySequence("Ctrl+3"), self, lambda: self._goto_tab("BLAST"))
        QShortcut(QKeySequence("Ctrl+7"), self, lambda: self._goto_tab("MSA"))
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
        from beer.embeddings import ESMC_AVAILABLE
        _dmethod = data.get("disorder_method", "")
        if self._embedder is not None and self._embedder.is_available():
            self._update_esm2_indicator("active")
        elif "metapredict" in _dmethod:
            self._update_esm2_indicator("metapredict")
        else:
            self._update_esm2_indicator("classical")
        if not ESMC_AVAILABLE and not self._esmc_missing_warned:
            self._esmc_missing_warned = True
            self.statusBar.showMessage(
                "ESMC not installed \u2014 disorder uses metapredict/classical fallback. "
                "Install with: pip install esm", 8000
            )
        seq  = data["seq"]
        self._run_plugins(seq, data)
        self.analysis_data = data
        if not getattr(self, "_restoring_snapshot", False):
            self._add_to_history(self.sequence_name, seq, data)
        # ── Populate Summary tab ──────────────────────────────────────────
        self._summary_tab_browser.setHtml(self._build_summary_tab_html(data))
        _rsecs = data.get("report_sections", {})
        for sec, browser in self.report_section_tabs.items():
            if sec in _rsecs:
                browser.setHtml(_rsecs[sec])
        self._populate_sasa_report_section()
        self._update_seq_viewer()
        self.update_graph_tabs()
        self._append_sparklines(data)
        self._append_mini_graphs()
        # Classical analysis — populate sidebar with lazy-load placeholders.
        self._ai_computed_sections.clear()
        self._bilstm_unc_state = {}   # reset MC-Dropout checkbox on new protein
        self._setup_ai_section_placeholders()
        # Refresh AI Features scheme combo so newly available heads appear
        if hasattr(self, "struct_color_mode_combo"):
            cur_mode = self.struct_color_mode_combo.currentText()
            if cur_mode == "AI Features":
                self._update_scheme_combo("AI Features")
        self.analyze_btn.setEnabled(True)
        # Enable all analysis-dependent buttons
        for btn in (self.mutate_btn, self.trunc_run_btn,
                    self.find_uniprot_btn):
            btn.setEnabled(True)
        self.predict_struct_btn.setEnabled(True)   # ESMFold needs only a sequence
        self.fetch_uniprot_tracks_btn.setEnabled(bool(self.current_accession))
        self._graphs_uniprot_btn.setEnabled(bool(self.current_accession))
        for _zb in getattr(self, "_zip_btns", []):
            try:
                _zb.setEnabled(True)
            except RuntimeError:
                pass  # C++ object already deleted by a layout clear
        self.trunc_run_btn.setToolTip("Run truncation series analysis")
        self.setWindowTitle(f"BEER — {self._display_name()}")
        self.statusBar.showMessage(
            f"Analysis complete  |  {len(seq)} aa  |  {self.sequence_name}", 4000
        )
        self._prepend_section_summaries()
        # Unlock gated result tabs and hide the welcome banner
        self._enable_result_tabs()
        if hasattr(self, "_welcome_banner"):
            self._welcome_banner.hide()
        # Briefly highlight Report tab to draw attention
        self._flash_nav_tab(self.main_tabs.stack_for_name("Report"))

    def _on_worker_error(self, msg: str):
        if hasattr(self, "_progress_dlg") and self._progress_dlg:
            self._progress_dlg.close()
            self._progress_dlg = None
        self.analysis_data = None
        self._ai_computed_sections.clear()
        self.analyze_btn.setEnabled(True)
        self.statusBar.showMessage("Analysis failed", 3000)
        QMessageBox.critical(self, "Analysis Error", msg)

    def _discard_worker(self, worker) -> None:
        """Safely retire a running worker without destroying its QThread prematurely.

        terminate() is unreliable for Python threads on macOS (pthread_cancel only
        fires at OS-level cancellation points; pure Python/C-extension code has none).
        Dropping the Python reference while the thread is alive causes
        'QThread: Destroyed while thread is still running' → abort.

        Instead: disconnect UI signals, park the worker in _discarded_workers so the
        reference stays alive, and reconnect finished/error to remove it from the set
        when the thread exits naturally. No blocking, no terminate.
        """
        try: worker.finished.disconnect()
        except RuntimeError: pass
        try: worker.error.disconnect()
        except RuntimeError: pass
        if worker.isRunning():
            self._discarded_workers.add(worker)
            worker.finished.connect(lambda _data, w=worker: self._discarded_workers.discard(w))
            worker.error.connect(lambda _msg, w=worker: self._discarded_workers.discard(w))

    def _cancel_analysis(self):
        if self._analysis_worker is not None:
            self._discard_worker(self._analysis_worker)
            self._analysis_worker = None
        if self._active_ai_worker is not None:
            self._discard_worker(self._active_ai_worker)
            self._active_ai_worker = None
        # Close the dialog explicitly — QProgressDialog does not auto-hide on cancel.
        dlg, self._progress_dlg = self._progress_dlg, None
        if dlg is not None:
            dlg.close()
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
            "name":                    self.sequence_name or "Sequence",
            "seq":                     data.get("seq", ""),
            "analysis_data":           data,
            "accession":               self.current_accession,
            "source_id":               self._source_id,
            "last_was_bilstm":         self._last_was_bilstm,
            "alphafold_data":          self.alphafold_data,
            "esmfold2_data":           self.esmfold2_data,
            "alphafold_missense_data": getattr(self, "_alphafold_missense_data", None),
            "pfam_domains":            list(self.pfam_domains),
            "uniprot_features":        dict(self._uniprot_features),
            "elm_data":                list(self.elm_data),
            "disprot_data":            dict(self.disprot_data),
            "phasepdb_data":           dict(self.phasepdb_data),
            "mobidb_data":             dict(self.mobidb_data),
            "variants_data":           list(self.variants_data),
            "intact_data":             dict(self.intact_data),
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
        snap["alphafold_data"]          = self.alphafold_data
        snap["esmfold2_data"]           = self.esmfold2_data
        snap["alphafold_missense_data"] = getattr(self, "_alphafold_missense_data", None)
        snap["pfam_domains"]            = list(self.pfam_domains)
        snap["uniprot_features"]        = dict(self._uniprot_features)
        snap["accession"]               = self.current_accession
        snap["source_id"]               = self._source_id
        snap["elm_data"]                = list(self.elm_data)
        snap["disprot_data"]            = dict(self.disprot_data)
        snap["phasepdb_data"]           = dict(self.phasepdb_data)
        snap["mobidb_data"]             = dict(self.mobidb_data)
        snap["variants_data"]           = list(self.variants_data)
        snap["intact_data"]             = dict(self.intact_data)

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
        self.sequence_name      = snap.get("name", "Sequence")
        self.current_accession  = snap.get("accession", "")
        self._source_id         = snap.get("source_id", snap.get("accession", ""))
        self._last_was_bilstm   = snap.get("last_was_bilstm", False)
        self.alphafold_data              = snap.get("alphafold_data")
        self.esmfold2_data               = snap.get("esmfold2_data")
        self._alphafold_missense_data    = snap.get("alphafold_missense_data")
        self.pfam_domains                = list(snap.get("pfam_domains", []))
        self._uniprot_features  = dict(snap.get("uniprot_features", {}))
        self.elm_data           = list(snap.get("elm_data", []))
        self.disprot_data       = dict(snap.get("disprot_data", {}))
        self.phasepdb_data      = dict(snap.get("phasepdb_data", {}))
        self.mobidb_data        = dict(snap.get("mobidb_data", {}))
        self.variants_data      = list(snap.get("variants_data", []))
        self.intact_data        = dict(snap.get("intact_data", {}))

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
            _plddt_arr = self.alphafold_data.get("plddt", [])
            n_res = len(_plddt_arr)
            mean_plddt = sum(_plddt_arr) / n_res if n_res else 0
            src = self.alphafold_data.get("accession", self.sequence_name)
            self._set_status_lbl(
                self.af_status_lbl,
                f"Structure: {src}  ({n_res} residues, mean pLDDT = {mean_plddt:.1f})",
                "success",
            )

        # Restore chip button states
        has_acc   = bool(self.current_accession)
        has_af    = bool(self.alphafold_data)
        has_pfam  = bool(self.pfam_domains)
        has_feats = bool(self._uniprot_features)
        has_seq = bool(self.seq_text.toPlainText().strip())
        for btn in self._db_fetch_btns:
            btn.setEnabled(has_acc)
            btn.setProperty("chip_state", "normal")
            btn.style().unpolish(btn); btn.style().polish(btn)
        if has_seq:
            self.predict_struct_btn.setEnabled(True)
        if has_af:
            self._mark_chip_fetched(self.fetch_af_btn)
        if has_pfam:
            # mirror AlphaFold: stay gated on has_acc, just mark as fetched
            self._mark_chip_fetched(self.fetch_pfam_btn)
        if has_feats:
            self.fetch_uniprot_tracks_btn.setEnabled(True)
            self._mark_chip_fetched(self.fetch_uniprot_tracks_btn)
        if self.elm_data:
            self.fetch_elm_btn.setEnabled(True)
            self._mark_chip_fetched(self.fetch_elm_btn)
        if self.disprot_data.get("regions"):
            self.fetch_disprot_btn.setEnabled(True)
            self._mark_chip_fetched(self.fetch_disprot_btn)
        if self.phasepdb_data:
            self.fetch_phasepdb_btn.setEnabled(True)
            self._mark_chip_fetched(self.fetch_phasepdb_btn)
        if self.mobidb_data.get("found"):
            self.fetch_mobidb_btn.setEnabled(True)
            self._mark_chip_fetched(self.fetch_mobidb_btn)
        if self.variants_data:
            self.fetch_variants_btn.setEnabled(True)
            self._mark_chip_fetched(self.fetch_variants_btn)
        if self.intact_data:
            self.fetch_intact_btn.setEnabled(True)
            self._mark_chip_fetched(self.fetch_intact_btn)
        self.statusBar.showMessage(
            f"Restored: {snap['name']}  ({len(data.get('seq',''))} aa)"
            if data else f"Restored: {snap['name']}", 3000)

    @staticmethod
    def _stop_worker(w, wait_ms: int = _WORKER_WAIT_MS) -> None:
        """Cooperatively stop a QThread worker; use terminate() only as a last
        resort. terminate() is unreliable (esp. on macOS) and can corrupt
        torch/CUDA state or leave locks held, so we first ask the worker to stop
        (cancel flag + interruption request) and wait; terminate() is reached
        only if the thread ignores the request and stays stuck."""
        try:
            if w is None or not w.isRunning():
                return
            if hasattr(w, "cancel"):
                w.cancel()                 # sets a flag honoured by loop workers
            w.requestInterruption()        # cooperative signal
            w.quit()                       # no-op for run()-override threads
            if not w.wait(wait_ms):
                w.terminate()              # last resort
                w.wait(300)
        except RuntimeError:
            pass                           # underlying C++ object already deleted

    def closeEvent(self, event):
        """Stop every running worker thread and wipe history on close.

        Workers are auto-discovered (any QThread attribute) rather than listed by
        hand, so a new worker can never be forgotten in the shutdown path."""
        from PySide6.QtCore import QThread as _QThread
        for _val in list(self.__dict__.values()):
            if isinstance(_val, _QThread):
                self._stop_worker(_val)
        _config.set_value("recent_sequences", [])
        super().closeEvent(event)

    # --- Accession fetch ---

    def fetch_accession(self):
        acc = self.accession_input.text().strip()
        if not acc:
            QMessageBox.warning(self, "Fetch", "Enter a UniProt ID or PDB ID.")
            return
        if self._fetch_acc_worker is not None and self._fetch_acc_worker.isRunning():
            return
        # Reset previous protein state before loading a new one
        if self.analysis_data or self.seq_text.toPlainText().strip():
            self._save_and_reset()
        # Detect PDB ID: legacy 4-char (digit-first alnum) or RCSB extended pdb_XXXXXXXX (12-char)
        import re as _re_pdb
        is_pdb = bool(
            _re_pdb.fullmatch(r"[0-9][A-Za-z0-9]{3}", acc)
            or _re_pdb.fullmatch(r"(?i)pdb_[0-9a-z]{8}", acc)
        )
        bio_asm = (getattr(self, "bio_assembly_chk", None) is not None
                   and self.bio_assembly_chk.isChecked())
        self.statusBar.showMessage(f"Fetching {acc}…")
        self._fetch_acc_worker = FetchAccessionWorker(acc, is_pdb, bio_asm)
        self._fetch_acc_worker.fetched_sequence.connect(self._on_fetch_accession_sequence)
        self._fetch_acc_worker.fetched_structure.connect(self._on_fetch_accession_structure)
        self._fetch_acc_worker.error.connect(self._on_fetch_accession_error)
        self._fetch_acc_worker.progress.connect(lambda msg: self.statusBar.showMessage(msg))
        self._fetch_acc_worker.finished.connect(lambda: setattr(self, "_fetch_acc_worker", None))
        self._fetch_acc_worker.start()
        return

    def _on_fetch_accession_error(self, msg: str) -> None:
        self._pending_pdb_tagged = None
        self._pending_pdb_acc    = ""
        self.statusBar.showMessage("Fetch failed", 3000)
        QMessageBox.warning(self, "Fetch Failed", msg)

    def _on_fetch_accession_sequence(self, raw: str, acc: str) -> None:
        is_pdb = (len(acc) in (4, 8) and acc[0].isdigit() and acc.isalnum())
        entries = self._parse_pasted_text(raw)
        if not entries:
            QMessageBox.warning(self, "Fetch", "No valid protein sequence returned.")
            return
        if is_pdb:
            tagged = [(rid.split("|")[0], seq) for rid, seq in entries]
            self._load_batch(tagged)
            rid, seq = tagged[0]
            self._pending_pdb_tagged = tagged
            self._pending_pdb_acc    = acc
        else:
            rid, seq = entries[0]
        self.seq_text.setPlainText(seq)
        self.sequence_name = rid
        self.current_accession = acc if not is_pdb else ""
        self._source_id        = acc
        self.fetch_af_btn.setEnabled(not is_pdb)
        self.predict_struct_btn.setEnabled(True)
        self.fetch_pfam_btn.setEnabled(not is_pdb)   # Pfam needs a UniProt accession, like AlphaFold
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

    def _on_fetch_accession_structure(self, struct_str: str, is_cif: bool) -> None:
        tagged = getattr(self, "_pending_pdb_tagged", None)
        acc    = getattr(self, "_pending_pdb_acc",    "")
        if not tagged:
            return
        try:
            chain_structs = (extract_chain_structures_mmcif(struct_str)
                             if is_cif else extract_chain_structures(struct_str))
            chain_letters = list(chain_structs.keys())
            for i, (rec_id, _) in enumerate(tagged):
                if i < len(chain_letters):
                    self.batch_struct[rec_id] = chain_structs[chain_letters[i]]
            if tagged[0][0] in self.batch_struct:
                self.alphafold_data = self.batch_struct[tagged[0][0]]
            self._struct_is_alphafold = False
            self._struct_source = "pdb"
            self._reset_struct_view()
            self._load_structure_viewer(struct_str, compute_sasa=False)
            self.export_structure_btn.setEnabled(True)
            self._set_status_lbl(
                self.af_status_lbl,
                f"Loaded PDB {acc.upper()}  —  "
                f"{len(chain_structs)} chain(s), {len(tagged)} sequence(s)",
                "success",
            )
        except Exception:
            pass

    def _fetch_and_show_protein_summary(self, acc: str, is_pdb: bool) -> None:
        """Start a background worker to fetch protein metadata; update UI via slot."""
        # Cancel any still-running summary fetch for a previous accession.
        if self._protein_summary_worker is not None and self._protein_summary_worker.isRunning():
            self._protein_summary_worker.quit()
            self._protein_summary_worker.wait(_WORKER_WAIT_MS)
        self.statusBar.showMessage(f"Loading protein summary for {acc}…")
        worker = ProteinSummaryWorker(acc, is_pdb)
        worker.result.connect(lambda card: self._on_protein_summary_result(card, acc, is_pdb))
        worker.error.connect(lambda _err: setattr(self, "_protein_summary_worker", None))
        worker.finished.connect(lambda: setattr(self, "_protein_summary_worker", None))
        self._protein_summary_worker = worker
        worker.start()

    def _on_protein_summary_result(self, card: dict, acc: str, is_pdb: bool) -> None:
        """Slot: receive parsed card dict from ProteinSummaryWorker and refresh the UI."""
        self._uniprot_card = card
        # Refresh summary tab with new card data if analysis is done.
        if self.analysis_data:
            self._summary_tab_browser.setHtml(
                self._build_summary_tab_html(self.analysis_data))
        # PDB cross-reference chips (UniProt only).
        if not is_pdb:
            self._populate_pdb_xref_chips(acc)

    def _populate_pdb_xref_chips(self, uniprot_id: str) -> None:
        """Fetch PDB xrefs for *uniprot_id* and show as clickable chips in a grid."""
        self._clear_layout_deep(self._pdb_xref_layout)

        xrefs = fetch_uniprot_pdb_xrefs(uniprot_id)
        if not xrefs:
            self._pdb_xref_inner.hide()
            return

        refs_capped = xrefs[:40]
        header_row = QHBoxLayout()
        header_row.setSpacing(4)
        header_row.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(f"PDB structures ({len(refs_capped)}):")
        lbl.setObjectName("pdb_xref_lbl")
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
        req = urllib.request.Request(url, headers={"User-Agent": "BEER/3.0"})
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
        req = urllib.request.Request(url, headers={"User-Agent": "BEER/3.0"})
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
        self._af_data = data
        self._struct_is_alphafold = True
        self._struct_source = "alphafold"
        self._update_composite_btn_state()
        if self.sequence_name:
            self.batch_struct[self.sequence_name] = data
        self._update_current_snapshot()
        self.fetch_af_btn.setEnabled(True)
        self._mark_chip_fetched(self.fetch_af_btn)
        self.export_structure_btn.setEnabled(True)
        _af_plddt = data.get("plddt", [])
        n_res = len(_af_plddt)
        mean_plddt = (sum(_af_plddt) / n_res) if n_res else 0
        _af_acc = data.get("accession", self.sequence_name or "Unknown")

        if self.esmfold2_data:
            self._struct_source = "overlay"
            self._struct_is_alphafold = False
            self._set_status_lbl(
                self.af_status_lbl,
                f"Overlay: AlphaFold + ESMFold2 for {_af_acc}  (mean pLDDT = {mean_plddt:.1f})",
                "success",
            )
            self._load_overlay_viewer()
        else:
            self._set_status_lbl(
                self.af_status_lbl,
                f"Loaded AlphaFold structure for {_af_acc}  "
                f"({n_res} residues, mean pLDDT = {mean_plddt:.1f})",
                "success",
            )
            self._reset_struct_view()
            self._load_structure_viewer(data["pdb_str"])

        if self.analysis_data:
            self.update_graph_tabs()
        self.statusBar.showMessage(
            f"AlphaFold structure loaded  ({_af_acc})", 4000)

    def _on_alphafold_error(self, msg: str):
        self.fetch_af_btn.setEnabled(True)
        self._mark_chip_error(self.fetch_af_btn)
        self.statusBar.showMessage("AlphaFold fetch failed", 3000)
        QMessageBox.warning(self, "AlphaFold Error", msg)

    # --- ESMFold2 (BioHub Forge API) ---

    @staticmethod
    def _extract_ca_coords(pdb_str: str) -> dict:
        """Return {(resSeq, iCode): xyz} for Cα atoms (first chain only)."""
        import numpy as _np
        recs = parse_ca_atoms(pdb_str)
        if not recs:
            return {}
        first = recs[0]["chain"]
        return {(r["resseq"], r["icode"]): _np.array([r["x"], r["y"], r["z"]])
                for r in recs if r["chain"] == first}

    @staticmethod
    def _extract_bfactors_from_pdb(pdb_str: str) -> list:
        """Per-residue B-factors from Cα atoms (first chain, residue order)."""
        recs = parse_ca_atoms(pdb_str)
        if not recs:
            return []
        first = recs[0]["chain"]
        chain_recs = sorted((r for r in recs if r["chain"] == first),
                            key=lambda r: (r["resseq"], r["icode"]))
        return [r["bfac"] for r in chain_recs]

    @staticmethod
    def _kabsch_align(mobile: "np.ndarray", ref: "np.ndarray") -> "np.ndarray":
        """Kabsch superposition. Returns mobile coords aligned onto ref."""
        import numpy as _np
        ref_c    = ref    - ref.mean(axis=0)
        mob_c    = mobile - mobile.mean(axis=0)
        H        = mob_c.T @ ref_c
        U, _, Vt = _np.linalg.svd(H)
        d        = _np.sign(_np.linalg.det(Vt.T @ U.T))
        R        = Vt.T @ _np.diag([1, 1, d]) @ U.T
        return mob_c @ R.T + ref.mean(axis=0)

    @staticmethod
    def _compute_hbond_ss(pdb_str: str) -> set:
        """Compute helix/sheet residues from 3D coordinates using H-bond geometry.

        Mirrors the algorithm in 3Dmol.js function L():
          - Collect all backbone N and O atoms.
          - H-bond proxy: closest N-O pair per atom within 3.2 Å, |Δresi| ≥ 4,
            same chain.
          - Helix: H-bond with |Δresi| == 4 (α-helix i→i+4 pattern).
          - Sheet: non-helix H-bonded atom whose partner is also non-helix
            (mutual pairing → β-sheet).
          - Smoothing: isolated single-residue assignments removed.
        Returns set of integer residue numbers assigned helix or sheet.
        """
        import math as _math

        # Parse backbone N and O atoms: {serial: (chain, resi, atom_name, x, y, z)}
        no_atoms: "list[tuple]" = []  # (chain, resi, atom_name, x, y, z)
        for line in pdb_str.splitlines():
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name not in ("N", "O"):
                continue
            try:
                chain = line[21]
                resi  = int(line[22:26])
                x     = float(line[30:38])
                y     = float(line[38:46])
                z     = float(line[46:54])
            except (ValueError, IndexError):
                continue
            no_atoms.append((chain, resi, atom_name, x, y, z))

        if not no_atoms:
            return set()

        # Sort by z for early-exit in distance search (mirrors 3Dmol sort)
        no_atoms.sort(key=lambda a: a[5])
        n = len(no_atoms)
        CUTOFF    = 3.2
        CUTOFF_SQ = CUTOFF * CUTOFF

        # Find best H-bond partner per atom (closest N-O within cutoff, |Δresi|≥4)
        hbond_other: "dict[int, int]" = {}   # atom_idx → partner_idx
        hbond_dsq:   "dict[int, float]" = {} # atom_idx → distance²

        for i in range(n):
            ci, ri, ni_name, xi, yi, zi = no_atoms[i]
            best_j, best_dsq = -1, float("inf")
            for j in range(i + 1, n):
                cj, rj, nj_name, xj, yj, zj = no_atoms[j]
                dz = zj - zi
                if dz > CUTOFF:
                    break
                if ni_name == nj_name:   # N-N or O-O: not an H-bond
                    continue
                if abs(yj - yi) > CUTOFF or abs(xj - xi) > CUTOFF:
                    continue
                dsq = (xj - xi) ** 2 + (yj - yi) ** 2 + dz * dz
                if dsq > CUTOFF_SQ:
                    continue
                if ci == cj and abs(rj - ri) < 4:
                    continue
                if dsq < best_dsq:
                    best_dsq, best_j = dsq, j
                # Also update j's best if this is better
                if dsq < hbond_dsq.get(j, float("inf")):
                    hbond_dsq[j]  = dsq
                    hbond_other[j] = i
            if best_j >= 0 and best_dsq < hbond_dsq.get(i, float("inf")):
                hbond_dsq[i]  = best_dsq
                hbond_other[i] = best_j

        # Build per-(chain, resi) SS table from H-bond assignments
        # key: (chain, resi) → 'h' | 's' | 'maybesheet' | None
        ss_table: "dict[tuple, str]" = {}

        for idx, partner_idx in hbond_other.items():
            ci, ri, _, _, _, _ = no_atoms[idx]
            cj, rj, _, _, _, _ = no_atoms[partner_idx]
            if ci == cj and abs(rj - ri) == 4:
                ss_table[(ci, ri)] = "h"

        # Helix smoothing: isolated non-helix in a helix run → promote
        chains: "dict[str, list[int]]" = {}
        for (ch, ri), _ in ss_table.items():
            chains.setdefault(ch, []).append(ri)
        for ch, resis in chains.items():
            resis_sorted = sorted(set(resis))
            for ri in resis_sorted:
                prev_h = ss_table.get((ch, ri - 1)) == "h"
                next_h = ss_table.get((ch, ri + 1)) == "h"
                if prev_h and next_h and ss_table.get((ch, ri)) != "h":
                    ss_table[(ch, ri)] = "h"

        # Sheet: non-helix H-bonded atoms → maybesheet, then mutual → 's'
        for idx, partner_idx in hbond_other.items():
            ci, ri, _, _, _, _ = no_atoms[idx]
            if ss_table.get((ci, ri)) != "h":
                ss_table[(ci, ri)] = "maybesheet"

        for idx, partner_idx in hbond_other.items():
            ci, ri, _, _, _, _ = no_atoms[idx]
            cj, rj, _, _, _, _ = no_atoms[partner_idx]
            if ss_table.get((ci, ri)) == "maybesheet":
                partner_state = ss_table.get((cj, rj))
                if partner_state in ("maybesheet", "s"):
                    ss_table[(ci, ri)] = "s"
                    ss_table[(cj, rj)] = "s"

        # Sheet smoothing: isolated non-sheet in a sheet run → promote
        for ch in set(k[0] for k in ss_table):
            all_resi = sorted(ri for (c, ri) in ss_table if c == ch)
            for ri in all_resi:
                if (ss_table.get((ch, ri - 1)) == "s"
                        and ss_table.get((ch, ri + 1)) == "s"
                        and ss_table.get((ch, ri)) != "s"):
                    ss_table[(ch, ri)] = "s"

        # Remove isolated single-residue helix or sheet assignments
        result: set = set()
        for (ch, ri), state in ss_table.items():
            if state not in ("h", "s"):
                continue
            has_neighbor = (ss_table.get((ch, ri - 1)) == state
                            or ss_table.get((ch, ri + 1)) == state)
            if has_neighbor:
                result.add(ri)
        return result

    @staticmethod
    def _parse_ss_residues(pdb_str: str) -> set:
        """Return residue numbers (int) in helix or sheet secondary structure.

        Uses HELIX/SHEET PDB records when present (depositor-assigned, authoritative).
        Falls back to coordinate-based H-bond detection (_compute_hbond_ss) when
        records are absent — mirrors the algorithm used by 3Dmol.js so the mask is
        consistent with the SS coloring shown in the viewer.
        """
        _helix, _sheet = parse_helix_sheet_records(pdb_str)
        resi_set = {r for (_ch, r) in (_helix | _sheet)}
        if resi_set:
            return resi_set
        return ProteinAnalyzerGUI._compute_hbond_ss(pdb_str)

    # Three-letter to one-letter amino acid code map (standard 20 + common variants)
    _AA3TO1: "dict[str, str]" = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
        # Common modified / selenomethionine residues
        "MSE": "M", "SEC": "C", "PYL": "K", "HSD": "H", "HSE": "H",
        "HSP": "H", "HIE": "H", "HID": "H", "HIP": "H", "CYX": "C",
    }

    @staticmethod
    def _extract_seq_and_ca(pdb_str: str) -> "tuple[str, list]":
        """Extract sequence and ordered Cα records from PDB ATOM lines (first chain only).

        Returns:
            seq   — one-letter sequence string of residues with coordinates
            records — list of (resnum, icode, xyz) in sequence order, same length as seq
        Insertion codes (col 27) are preserved so 45/45A/45B are treated as distinct.
        Unknown residues are mapped to 'X' and included so alignment indices stay correct.
        """
        import numpy as _np
        aa3to1 = ProteinAnalyzerGUI._AA3TO1
        seen: "dict[tuple, int]" = {}   # (resnum, icode) → index in records
        records: list = []
        seq_chars: list = []
        first_chain = None
        for line in pdb_str.splitlines():
            if not line.startswith("ATOM"):
                continue
            name  = line[12:16].strip()
            chain = line[21:22].strip()
            if first_chain is None:
                first_chain = chain
            if chain != first_chain:
                continue
            if name != "CA":
                continue
            try:
                resnum = int(line[22:26])
                icode  = line[26:27].strip()   # insertion code, often blank
                aa3    = line[17:20].strip()
                xyz    = _np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            except (ValueError, IndexError):
                continue
            key = (resnum, icode)
            if key in seen:
                continue   # duplicate ATOM entry for same residue — skip
            seen[key] = len(records)
            records.append((resnum, icode, xyz))
            seq_chars.append(aa3to1.get(aa3, "X"))
        return "".join(seq_chars), records

    @staticmethod
    def _needleman_wunsch(seq_a: str, seq_b: str) -> "list[tuple[int,int]]":
        """Global pairwise alignment (Needleman-Wunsch) using a simple identity matrix.

        Scoring: match +2, mismatch −1, gap open −4, gap extend −1 (affine via two matrices).
        Returns list of (i, j) index pairs where both sequences have a residue (no gaps).
        Runs in O(N·M) time and memory — fine for typical protein lengths (< 2000 aa).
        """
        import numpy as _np
        n, m = len(seq_a), len(seq_b)
        GAP_OPEN = -4
        GAP_EXT  = -1
        MATCH    =  2
        MISMATCH = -1

        # Affine gap: three matrices M (match), X (gap in seq_b), Y (gap in seq_a)
        NEG_INF = -10 ** 9
        M = _np.full((n + 1, m + 1), NEG_INF, dtype=_np.float32)
        X = _np.full((n + 1, m + 1), NEG_INF, dtype=_np.float32)
        Y = _np.full((n + 1, m + 1), NEG_INF, dtype=_np.float32)
        M[0, 0] = 0.0
        for i in range(1, n + 1):
            X[i, 0] = GAP_OPEN + (i - 1) * GAP_EXT
        for j in range(1, m + 1):
            Y[0, j] = GAP_OPEN + (j - 1) * GAP_EXT

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                s = MATCH if seq_a[i - 1] == seq_b[j - 1] else MISMATCH
                M[i, j] = s + max(M[i-1, j-1], X[i-1, j-1], Y[i-1, j-1])
                X[i, j] = max(M[i-1, j] + GAP_OPEN, X[i-1, j] + GAP_EXT,
                               Y[i-1, j] + GAP_OPEN)
                Y[i, j] = max(M[i, j-1] + GAP_OPEN, X[i, j-1] + GAP_OPEN,
                               Y[i, j-1] + GAP_EXT)

        # Traceback
        pairs: list = []
        i, j = n, m
        # Determine which matrix we end in
        best_end = max(M[n, m], X[n, m], Y[n, m])
        state = "M" if M[n, m] == best_end else ("X" if X[n, m] == best_end else "Y")
        while i > 0 or j > 0:
            if state == "M":
                pairs.append((i - 1, j - 1))   # both consumed → aligned pair
                s = MATCH if seq_a[i-1] == seq_b[j-1] else MISMATCH
                if   M[i-1,j-1] + s == M[i,j]: state = "M"; i -= 1; j -= 1
                elif X[i-1,j-1] + s == M[i,j]: state = "X"; i -= 1; j -= 1
                else:                           state = "Y"; i -= 1; j -= 1
            elif state == "X":
                # gap in seq_b (seq_a advances)
                if   M[i-1,j] + GAP_OPEN == X[i,j]: state = "M"
                elif X[i-1,j] + GAP_EXT  == X[i,j]: state = "X"
                else:                                state = "Y"
                i -= 1
            else:  # Y: gap in seq_a (seq_b advances)
                if   M[i,j-1] + GAP_OPEN == Y[i,j]: state = "M"
                elif X[i,j-1] + GAP_OPEN == Y[i,j]: state = "X"
                else:                                state = "Y"
                j -= 1
        pairs.reverse()
        return pairs   # [(i_in_seq_a, j_in_seq_b), ...]

    @staticmethod
    def _align_pdb_to_ref(mobile_pdb: str, ref_pdb: "str | dict",
                          mask_resi: "set | None" = None) -> tuple:
        """Sequence-aware Kabsch alignment of mobile_pdb onto ref_pdb.

        Uses Needleman-Wunsch sequence alignment to build residue correspondence,
        handling missing residues, insertion codes, and non-sequential PDB numbering.
        ref_pdb may be a PDB string or the legacy {resnum: xyz} dict (backward compat).
        mask_resi: residue numbers in the *reference* to restrict the SVD fit (structured-only).
        Returns (aligned_pdb_str, rmsd_per_res).
        """
        import numpy as _np

        # ── Build reference Cα list ───────────────────────────────────────────
        if isinstance(ref_pdb, dict):
            # Legacy path: caller already extracted {resnum: xyz}
            ref_seq, ref_records = "", []
            for rnum, xyz in sorted(ref_pdb.items()):
                ref_seq += "X"
                ref_records.append((rnum, "", xyz))
        else:
            ref_seq, ref_records = ProteinAnalyzerGUI._extract_seq_and_ca(ref_pdb)

        mob_seq, mob_records = ProteinAnalyzerGUI._extract_seq_and_ca(mobile_pdb)

        if not mob_records or not ref_records:
            return mobile_pdb, []

        # ── Sequence alignment → residue correspondence ───────────────────────
        # Fast path: identical sequences → direct 1:1 mapping (common for AF↔ESMFold2).
        # Only run O(N·M) NW when sequences differ (e.g. experimental PDB with missing residues).
        if mob_seq == ref_seq:
            pairs = [(i, i) for i in range(len(mob_seq))]
        else:
            pairs = ProteinAnalyzerGUI._needleman_wunsch(mob_seq, ref_seq)
        # pairs: (mob_idx, ref_idx) — both non-gapped positions

        if len(pairs) < 4:
            return mobile_pdb, []

        # ── Build Cα coordinate arrays for the aligned pairs ──────────────────
        mob_xyz = _np.array([mob_records[i][2] for i, _ in pairs])
        ref_xyz = _np.array([ref_records[j][2] for _, j in pairs])

        # Mask for structured-only fit: identify which pairs have ref residue in mask
        if mask_resi:
            fit_mask = _np.array([ref_records[j][0] in mask_resi for _, j in pairs])
            if fit_mask.sum() >= 4:
                mob_fit = mob_xyz[fit_mask]
                ref_fit = ref_xyz[fit_mask]
            else:
                mob_fit, ref_fit = mob_xyz, ref_xyz
        else:
            mob_fit, ref_fit = mob_xyz, ref_xyz

        # ── Kabsch SVD fit ────────────────────────────────────────────────────
        mob_center = mob_fit.mean(axis=0)
        ref_center = ref_fit.mean(axis=0)
        H          = (mob_fit - mob_center).T @ (ref_fit - ref_center)
        U, _, Vt   = _np.linalg.svd(H)
        d          = _np.sign(_np.linalg.det(Vt.T @ U.T))
        R          = Vt.T @ _np.diag([1, 1, d]) @ U.T

        # ── Apply transform to ALL atoms in the mobile PDB ───────────────────
        new_lines = []
        for line in mobile_pdb.splitlines():
            if line.startswith("ATOM") or line.startswith("HETATM"):
                try:
                    orig = _np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                    new  = (orig - mob_center) @ R.T + ref_center
                    line = line[:30] + f"{new[0]:8.3f}{new[1]:8.3f}{new[2]:8.3f}" + line[54:]
                except (ValueError, IndexError):
                    pass
            new_lines.append(line)

        # ── Per-residue RMSD over aligned pairs (post-alignment) ─────────────
        mob_aligned = (mob_xyz - mob_center) @ R.T + ref_center
        rmsd_vals   = _np.linalg.norm(mob_aligned - ref_xyz, axis=1).tolist()
        # Map back to full reference sequence positions; unaligned positions stay NaN.
        # NaN is preserved so the RMSD plot x-axis maps correctly to residue numbers —
        # create_structure_comparison_figure skips NaN spans rather than collapsing them.
        pair_ref_idx = {j: v for (_, j), v in zip(pairs, rmsd_vals)}
        rmsd_full    = [pair_ref_idx.get(k, float("nan")) for k in range(len(ref_records))]
        return "\n".join(new_lines), rmsd_full

    def _on_align_mode_changed(self) -> None:
        """Re-run alignment and reload overlay when the align-mode radio changes."""
        if self.alphafold_data and self.esmfold2_data:
            self._load_overlay_viewer()

    def _load_overlay_viewer(self) -> None:
        """Load AlphaFold + ESMFold2 as a two-model overlay in the 3Dmol viewer."""
        if not _WEBENGINE_AVAILABLE or self.structure_viewer is None:
            return
        if not (self.alphafold_data and self.esmfold2_data):
            return
        af_pdb  = self.alphafold_data["pdb_str"]
        esm_pdb = self.esmfold2_data["pdb_str"]
        structured_only = (
            hasattr(self, "_align_struct_radio") and self._align_struct_radio.isChecked()
        )
        mask = None
        if structured_only:
            af_ss  = self._parse_ss_residues(af_pdb)
            esm_ss = self._parse_ss_residues(esm_pdb)
            mask   = af_ss & esm_ss if (af_ss and esm_ss) else (af_ss or esm_ss or None)
        aligned_esm, rmsd_full = self._align_pdb_to_ref(esm_pdb, af_pdb, mask_resi=mask)
        self.esmfold2_data["aligned_pdb"]  = aligned_esm
        self.esmfold2_data["rmsd_per_res"] = rmsd_full
        self._struct_pdb_str = af_pdb
        af_json  = json.dumps(af_pdb)
        esm_json = json.dumps(aligned_esm)
        self._js(f"loadOverlay({af_json}, {esm_json});")
        self._populate_chain_controls(af_pdb)
        self._update_overlay_controls()
        from PySide6.QtCore import QTimer as _QT
        _QT.singleShot(800, self._annotate_structure_viewer)
        _af_copy = af_pdb
        def _deferred_sasa():
            self._struct_sasa_data, self._struct_sasa_raw = self._compute_sasa(_af_copy)
            self._populate_sasa_report_section()
            if self.analysis_data:
                self.update_graph_tabs()
        _QT.singleShot(200, _deferred_sasa)

    def _on_graph_struct_src_changed(self, text: str) -> None:
        mapping = {"AlphaFold": "alphafold", "ESMFold2": "esmfold2", "Both": "both"}
        self._graph_struct_src = mapping.get(text, "alphafold")
        # Ensure ESMFold2 SASA computed if needed
        if self._graph_struct_src in ("esmfold2", "both"):
            self._ensure_esm_sasa()
        self.update_graph_tabs()

    def _ensure_esm_sasa(self) -> None:
        """Compute and cache SASA for the ESMFold2 structure if not yet done."""
        if not self.esmfold2_data:
            return
        if "sasa_data" in self.esmfold2_data:
            return
        pdb = self.esmfold2_data.get("aligned_pdb") or self.esmfold2_data.get("pdb_str", "")
        if not pdb:
            return
        try:
            sasa_data, sasa_raw = self._compute_sasa(pdb)
            self.esmfold2_data["sasa_data"] = sasa_data
            self.esmfold2_data["sasa_raw"]  = sasa_raw
        except Exception:
            pass

    def _update_overlay_controls(self) -> None:
        """Show/hide overlay toggle row and graph source selector based on overlay state."""
        if not hasattr(self, "_overlay_row"):
            return
        in_overlay = bool(self.alphafold_data and self.esmfold2_data)
        self._overlay_row.setVisible(in_overlay)
        if hasattr(self, "_struct_src_bar"):
            self._struct_src_bar.setVisible(in_overlay)
        if not in_overlay:
            self._graph_struct_src = (
                "esmfold2" if self._struct_source == "esmfold2" else "alphafold"
            )
            if hasattr(self, "_graph_src_combo"):
                self._graph_src_combo.blockSignals(True)
                self._graph_src_combo.setCurrentText("AlphaFold")
                self._graph_src_combo.blockSignals(False)
        # When entering overlay, reset to both-visible → lock colour.
        if in_overlay:
            if hasattr(self, "_overlay_af_chk"):  self._overlay_af_chk.setChecked(True)
            if hasattr(self, "_overlay_esm_chk"): self._overlay_esm_chk.setChecked(True)
            self._set_overlay_color_lock(locked=True)
        else:
            self._set_overlay_color_lock(locked=False)

    def _set_overlay_color_lock(self, locked: bool) -> None:
        """Enable/disable colour combos; locked=True when both models are simultaneously visible."""
        tip = ("Colour is fixed to blue/orange while both structures are visible.\n"
               "Hide one structure to apply a custom colour scheme.") if locked else ""
        for combo in (
            getattr(self, "struct_color_mode_combo", None),
            getattr(self, "struct_scheme_combo", None),
        ):
            if combo is not None:
                combo.setEnabled(not locked)
                combo.setToolTip(tip)

    def _on_overlay_af_toggled(self, visible: bool) -> None:
        self._js(f"showModelAF({'true' if visible else 'false'});")
        both_on = visible and getattr(self, "_overlay_esm_chk", None) and self._overlay_esm_chk.isChecked()
        self._set_overlay_color_lock(locked=bool(both_on))

    def _on_overlay_esm_toggled(self, visible: bool) -> None:
        self._js(f"showModelESM({'true' if visible else 'false'});")
        both_on = visible and getattr(self, "_overlay_af_chk", None) and self._overlay_af_chk.isChecked()
        self._set_overlay_color_lock(locked=bool(both_on))

    def _run_esmfold2(self):
        seq = (self.analysis_data or {}).get("seq") or self.seq_text.toPlainText().strip()
        if not seq:
            QMessageBox.warning(self, "No Sequence",
                "Load a protein sequence before predicting structure.")
            return
        if self._esmfold2_worker is not None:
            if self._esmfold2_worker.isRunning():
                return
            # Previous worker finished but Qt hasn't fully cleaned up its thread.
            # Wait briefly so Qt can complete internal teardown before we drop the reference.
            self._esmfold2_worker.wait(_WORKER_WAIT_MS)
            self._esmfold2_worker = None
        api_token = _config.load().get("biohub_api_key", "").strip()
        if not api_token:
            _mb = QMessageBox(self)
            _mb.setWindowTitle("BioHub API Key Required")
            _mb.setIcon(QMessageBox.Icon.Warning)
            _mb.setTextFormat(Qt.TextFormat.RichText)
            _mb.setText(
                "No BioHub API key found.<br><br>"
                "Register at <a href='https://biohub.ai'>biohub.ai</a> to get a free key,<br>"
                "then add it in <b>Settings → BioHub API Key</b>."
            )
            _mb.exec()
            return
        self.predict_struct_btn.setEnabled(False)
        self._mark_chip_loading(self.predict_struct_btn)
        self.statusBar.showMessage("Predicting structure with ESMFold2…")
        self._esmfold2_worker = ESMFold2Worker(seq, api_token)
        self._esmfold2_worker.finished.connect(self._on_esmfold2_finished)
        self._esmfold2_worker.error.connect(self._on_esmfold2_error)
        self._esmfold2_worker.start()

    def _on_esmfold2_finished(self, pdb_str: str):
        self.predict_struct_btn.setEnabled(True)
        self._mark_chip_fetched(self.predict_struct_btn)
        self.export_structure_btn.setEnabled(True)
        esm_plddt_raw = self._extract_bfactors_from_pdb(pdb_str)
        if esm_plddt_raw and max(esm_plddt_raw) <= 1.0:
            esm_plddt = [v * 100.0 for v in esm_plddt_raw]
        else:
            esm_plddt = esm_plddt_raw
        ca = self._extract_ca_coords(pdb_str)
        import numpy as _np
        if ca:
            coords_arr  = _np.array([ca[r] for r in sorted(ca)])
            diff        = coords_arr[:, None, :] - coords_arr[None, :, :]
            dist_matrix = _np.sqrt((diff ** 2).sum(axis=-1))
        else:
            dist_matrix = None
        self.esmfold2_data = {
            "pdb_str":    pdb_str,
            "plddt":      esm_plddt,
            "dist_matrix": dist_matrix,
        }
        name = self.sequence_name or "ESMFold2"
        if self.alphafold_data:
            self._struct_source      = "overlay"
            self._struct_is_alphafold = False
            self._set_status_lbl(
                self.af_status_lbl,
                f"Overlay: AlphaFold + ESMFold2 for {name}",
                "success",
            )
            self._load_overlay_viewer()
        else:
            self._struct_source      = "esmfold2"
            self._struct_is_alphafold = False
            self._set_status_lbl(
                self.af_status_lbl,
                f"ESMFold2 structure predicted for {name}",
                "success",
            )
            self._reset_struct_view()
            self._load_structure_viewer(pdb_str)
        self._esmfold2_worker = None
        if self.analysis_data:
            self.update_graph_tabs()
        self.statusBar.showMessage("ESMFold2 structure loaded", 4000)

    def _on_esmfold2_error(self, msg: str):
        self.predict_struct_btn.setEnabled(True)
        self._mark_chip_error(self.predict_struct_btn)
        self.statusBar.showMessage("ESMFold2 prediction failed", 3000)
        QMessageBox.warning(self, "ESMFold2 Error", msg)

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
        self._mark_chip_error(self.fetch_pfam_btn)
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
        self._stop_worker(self._blast_worker)
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
        self._set_status_lbl(self.blast_status_lbl, f"{len(hits)} hit(s) returned.", "success")
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
            copy_btn = QPushButton("Copy")
            copy_btn.setToolTip("Copy accession to clipboard")
            copy_btn.clicked.connect(
                lambda _, acc=hit["accession"]: (
                    QApplication.clipboard().setText(acc),
                    self.statusBar.showMessage(f"Copied: {acc}", 2000),
                ))
            self.blast_table.setCellWidget(row, 7, copy_btn)
        self.blast_table.resizeColumnsToContents()
        self.statusBar.showMessage(f"BLAST complete — {len(hits)} hits", 4000)

    def _on_blast_error(self, msg: str):
        if self._blast_timer:
            self._blast_timer.stop()
            self._blast_timer = None
        self._blast_start_time = None
        self.blast_stop_btn.setVisible(False)
        self.blast_run_btn.setEnabled(True)
        self._set_status_lbl(self.blast_status_lbl, f"Error: {msg}", "error")
        QMessageBox.warning(self, "BLAST Error", msg)

    def _load_blast_hit(self, hit: dict):
        seq = hit.get("subject", "")
        if not seq or not is_valid_protein(seq):
            QMessageBox.warning(self, "Load Hit", "Subject sequence is not a valid protein.")
            return
        self._save_and_reset()
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
        self._push_undo()
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
    def _save_and_reset(self):
        """Save current session to history, then perform a full UI reset."""
        if self.analysis_data:
            self._update_current_snapshot()
            self._push_undo()   # loading a new protein is undoable (Ctrl+Z)
        self._do_reset()

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
        self.batch_struct = {}
        self.current_accession = ""
        self._source_id = ""
        self.alphafold_data  = None
        self.esmfold2_data   = None
        self._af_data        = None
        self._struct_source  = "none"
        self._exp_pdb_str    = None
        self._struct_pdb_str = None
        self._struct_sasa_data = {}
        self._struct_sasa_raw  = {}
        self.pfam_domains = []
        self.motif_input.clear()
        self.motif_match_lbl.setText("")

        self.chain_combo.blockSignals(True)
        self.chain_combo.clear()
        self.chain_combo.setEnabled(False)
        self.chain_combo.blockSignals(False)
        self._chain_row_widget.hide()

        self._clear_layout_deep(self._pdb_xref_layout)
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
        if self._active_ai_worker is not None:
            self._stop_worker(self._active_ai_worker)
            self._active_ai_worker = None

        self._clear_struct_graph_marker()
        self._graph_generators.clear()
        self._generated_graphs.clear()
        for _tab, vb in self.graph_tabs.values():
            self._clear_layout(vb)

        if self.structure_viewer is not None:
            self._reset_struct_view()
            if hasattr(self, "struct_hbond_cb"):
                self.struct_hbond_cb.setChecked(False)
            if hasattr(self, "struct_contacts_cb"):
                self.struct_contacts_cb.setChecked(False)
            if hasattr(self, "struct_colorbar_cb"):
                self.struct_colorbar_cb.setChecked(True)
            if hasattr(self, "_ai_grad_btn"):
                self._struct_ai_color_mode = "gradient"
                self._struct_ai_color = "#f3722c"
                self._ai_grad_btn.setChecked(True)
                self._ai_bin_btn.setChecked(False)
            if hasattr(self, "struct_ai_gradient_combo"):
                self.struct_ai_gradient_combo.blockSignals(True)
                self.struct_ai_gradient_combo.setCurrentText("Plasma (Purple→Yellow)")
                self.struct_ai_gradient_combo.blockSignals(False)
            if hasattr(self, "struct_ai_color_combo"):
                self.struct_ai_color_combo.blockSignals(True)
                self.struct_ai_color_combo.setCurrentIndex(0)
                self.struct_ai_color_combo.blockSignals(False)
            if hasattr(self, "_hbond_color_btn"):
                self._hbond_color = "#44ccff"
                self._hbond_color_btn.setStyleSheet(
                    "background:#44ccff;border:1px solid #ccc;border-radius:3px;")
                self._hbond_radius_sb.setValue(0.07)
            if hasattr(self, "_contact_color_btn"):
                self._contact_color = "#888888"
                self._contact_color_btn.setStyleSheet(
                    "background:#888888;border:1px solid #ccc;border-radius:3px;")
                self._contact_opacity_sb.setValue(0.30)
            self._js("clearHighlight(); loadPDB(null);")

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
            self._stop_worker(_w)
        self._uniprot_feat_worker = None
        self._seq_search_worker   = None

        for btn in (self.mutate_btn,
                    self.export_structure_btn, self.find_uniprot_btn,
                    self.trunc_run_btn,
                    self._graphs_uniprot_btn, self._graphs_clear_tracks_btn):
            btn.setEnabled(False)
        chip_buttons = self._db_fetch_btns + [
            self.fetch_uniprot_tracks_btn,
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
        self._disable_result_tabs()
        if hasattr(self, "_welcome_banner"):
            self._welcome_banner.show()

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
        self._push_undo()   # Clear All is undoable (Ctrl+Z)
        self._do_reset()
        self.accession_input.clear()
        self.statusBar.showMessage("Session cleared — press Ctrl+Z to undo.", 4000)

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

            plt.close('all')
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
            "pka_set":     self.pka_set,
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
        self.pka_set          = state.get("pka_set", DEFAULT_PKA_SET)
        if hasattr(self, "pka_set_combo"):
            self.pka_set_combo.setCurrentText(self.pka_set)
        self.transparent_bg   = state.get("transparent_bg", True)
        self.app_font_size    = state.get("app_font_size", 12)
        self.label_font_size  = state.get("label_font_size", 14)
        self.tick_font_size   = state.get("tick_font_size", 12)
        # Update settings UI widgets
        self.ph_input.setValue(self.default_pH)
        self.window_size_input.setValue(self.default_window_size)
        self.transparent_bg_checkbox.setChecked(self.transparent_bg)
        self.label_font_input.setValue(self.label_font_size)
        self.tick_font_input.setValue(self.tick_font_size)
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

        _hint = QLabel("Run analysis on the Analysis tab first, then use the controls below to generate a truncation series.")
        _hint.setObjectName("placeholder_lbl")
        _hint.setWordWrap(True)
        layout.addWidget(_hint)

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

        _hint = QLabel("Paste ≥2 sequences in multi-FASTA format below and click <b>Run MSA Analysis</b> to compute conservation and covariance.")
        _hint.setObjectName("placeholder_lbl")
        _hint.setWordWrap(True)
        layout.addWidget(_hint)

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
        self.msa_pssm_btn = QPushButton("Export PSSM (CSV)")
        self.msa_pssm_btn.setMinimumHeight(30)
        self.msa_pssm_btn.setToolTip(
            "Export a position-specific scoring matrix (log2 odds vs. Swiss-Prot "
            "background) for the current alignment.")
        self.msa_pssm_btn.setEnabled(False)
        self.msa_pssm_btn.clicked.connect(self._export_pssm)
        ctrl.addWidget(self.msa_pssm_btn)
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
        self.msa_viewer.setOpenLinks(False)   # route opengraph: links ourselves
        self.msa_viewer.anchorClicked.connect(self._on_msa_link)
        right_v.addWidget(self.msa_viewer)
        splitter.addWidget(right_w)
        layout.addWidget(splitter, 1)

    def init_complex_tab(self):
        """Tab for protein complex stoichiometry calculations."""
        container = QWidget()
        layout    = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        self.main_tabs.addTab(container, "Protein Complex")

        _hint = QLabel("Paste chain sequences in multi-FASTA format (chain ID in header), enter the stoichiometry, and click <b>Calculate Complex</b>.")
        _hint.setObjectName("placeholder_lbl")
        _hint.setWordWrap(True)
        layout.addWidget(_hint)

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

    # ── Composite Structure tab ───────────────────────────────────────────────

    # Mol* composite viewer template — __BG__ and __CSS__ replaced at runtime.
    _COMPOSITE_HTML = """\
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    html,body { margin:0; padding:0; overflow:hidden;
      background:__BG__; width:100%; height:100%; }
    #vp { width:100%; height:100vh; position:relative; }
    .msp-plugin { background:transparent !important; }
  </style>
  <style id="msp-css">__CSS__</style>
</head>
<body><div id="vp"></div>
<script>
var _viewer = null;
window.addEventListener('DOMContentLoaded', function() {
  molstar.Viewer.create('vp', {
    layoutIsExpanded: false,
    layoutShowControls: false,
    layoutShowRemoteState: false,
    layoutShowSequence: false,
    layoutShowLog: false,
    layoutShowLeftPanel: false,
    collapseLeftPanel: true,
    viewportShowExpand: false,
    viewportShowControls: false,
    viewportShowSettings: false,
    viewportShowSelectionMode: false,
    viewportShowAnimation: false,
    viewportShowTrajectoryControls: false,
  }).then(function(v) {
    _viewer = v;
    setBackground('__BG__');
  });
});

function encodeSourceBfac(pdb_str, pred_resis, lowconf_resis) {
  var predSet = new Set(pred_resis);
  var lcSet   = new Set(lowconf_resis);
  return pdb_str.split('\\n').map(function(line) {
    if ((line.startsWith('ATOM  ') || line.startsWith('HETATM')) && line.length >= 60) {
      var resi = parseInt(line.substring(22, 26), 10);
      if (!isNaN(resi)) {
        var bfac = lcSet.has(resi) ? '  40.00' : predSet.has(resi) ? '  75.00' : ' 100.00';
        return line.substring(0, 60) + bfac + (line.length > 67 ? line.substring(67) : '');
      }
    }
    return line;
  }).join('\\n');
}

async function loadComposite(pdb_str, pred_resis, lowconf_resis) {
  if (!_viewer) return;
  await _viewer.plugin.clear();
  var encoded = encodeSourceBfac(pdb_str, pred_resis, lowconf_resis);
  await _viewer.loadStructureFromData(encoded, 'pdb', false);
  var structs = _viewer.plugin.managers.structure.hierarchy.current.structures;
  if (structs.length) {
    try {
      await _viewer.plugin.managers.structure.component.updateRepresentationsTheme(
        structs[0].components, {color: 'plddt-confidence'}
      );
    } catch(e) { console.warn('plddt-confidence unavailable, using default:', e); }
  }
  if (_viewer.plugin.canvas3d) _viewer.plugin.canvas3d.requestCameraReset();
}

function spinComposite(on) {
  if (_viewer && _viewer.plugin.canvas3d)
    _viewer.plugin.canvas3d.setProps({trackball: {spin: !!on, spinSpeed: 1}});
}

function resetCompositeView() {
  if (_viewer && _viewer.plugin.canvas3d)
    _viewer.plugin.canvas3d.requestCameraReset();
}

function setBackground(c) {
  document.documentElement.style.background = c;
  document.body.style.background = c;
  if (_viewer && _viewer.plugin.canvas3d) {
    var h = c.replace('#', '');
    var r = parseInt(h.substring(0,2), 16);
    var g = parseInt(h.substring(2,4), 16);
    var b = parseInt(h.substring(4,6), 16);
    _viewer.plugin.canvas3d.setProps({renderer: {backgroundColor: (r<<16)|(g<<8)|b}});
  }
}
</script>
</body></html>"""

    def init_composite_tab(self) -> None:
        """Build the Fix PDB tab — fully independent of the main analysis/structure tabs."""
        container = QWidget()
        layout    = QVBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        self.main_tabs.addTab(container, "Fix PDB")

        # ── Status / info row ─────────────────────────────────────────────────
        self._comp_status_lbl = QLabel(
            "Step 1: enter a PDB ID and fetch the experimental structure. "
            "Step 2: enter the UniProt accession for the AlphaFold model. "
            "Step 3: click Fix PDB.")
        self._comp_status_lbl.setObjectName("status_lbl")
        self._comp_status_lbl.setWordWrap(True)
        layout.addWidget(self._comp_status_lbl)

        # ── PDB ID fetch row ──────────────────────────────────────────────────
        pdb_row = QHBoxLayout()
        _pdb_lbl = QLabel("PDB ID:")
        _pdb_lbl.setFixedWidth(130)
        pdb_row.addWidget(_pdb_lbl)
        self._comp_pdb_input = QLineEdit()
        self._comp_pdb_input.setPlaceholderText("e.g. 1EMA")
        self._comp_pdb_input.setMaximumWidth(80)
        self._comp_pdb_input.setToolTip("4-character PDB ID of the experimental structure to complete.")
        self._comp_pdb_input.returnPressed.connect(self._comp_fetch_pdb)
        pdb_row.addWidget(self._comp_pdb_input)
        self._comp_pdb_fetch_btn = QPushButton("Fetch PDB")
        self._comp_pdb_fetch_btn.setObjectName("chip_btn")
        self._comp_pdb_fetch_btn.clicked.connect(self._comp_fetch_pdb)
        pdb_row.addWidget(self._comp_pdb_fetch_btn)
        self._comp_pdb_status_lbl = QLabel("")
        self._comp_pdb_status_lbl.setObjectName("status_lbl")
        pdb_row.addWidget(self._comp_pdb_status_lbl)
        pdb_row.addStretch(1)
        layout.addLayout(pdb_row)

        # ── Structural issues panel ───────────────────────────────────────────
        _issues_lbl = QLabel("Structural issues:")
        _issues_lbl.setObjectName("section_lbl")
        layout.addWidget(_issues_lbl)
        self._comp_issues_browser = QTextBrowser()
        self._comp_issues_browser.setMaximumHeight(130)
        self._comp_issues_browser.setHtml(
            "<p style='color:#888;font-size:11px;margin:4px'>"
            "Fetch a PDB to see structural issues.</p>")
        layout.addWidget(self._comp_issues_browser)

        # ── UniProt accession input row ────────────────────────────────────────
        acc_row = QHBoxLayout()
        _acc_lbl = QLabel("UniProt accession:")
        _acc_lbl.setFixedWidth(130)
        acc_row.addWidget(_acc_lbl)
        self._comp_acc_input = QLineEdit()
        self._comp_acc_input.setPlaceholderText("e.g. P42212")
        self._comp_acc_input.setMaximumWidth(140)
        self._comp_acc_input.setToolTip(
            "UniProt accession for this protein — used to fetch the AlphaFold model.")
        self._comp_acc_input.textChanged.connect(self._update_composite_btn_state)
        acc_row.addWidget(self._comp_acc_input)
        acc_row.addStretch(1)
        layout.addLayout(acc_row)

        # ── Controls row ──────────────────────────────────────────────────────
        ctrl_row = QHBoxLayout()
        self._comp_build_btn = QPushButton("Fix PDB")
        self._comp_build_btn.setObjectName("chip_btn")
        self._comp_build_btn.setMinimumHeight(30)
        self._comp_build_btn.setEnabled(False)
        self._comp_build_btn.clicked.connect(self._run_composite_build)
        ctrl_row.addWidget(self._comp_build_btn)

        # Alternative gap-fill path: ESMFold2 (no AlphaFold model required).
        self._comp_esm_btn = QPushButton("Fix with ESMFold2")
        self._comp_esm_btn.setObjectName("chip_btn")
        self._comp_esm_btn.setMinimumHeight(30)
        self._comp_esm_btn.setEnabled(False)
        self._comp_esm_btn.setToolTip(
            "Predict the full structure from the SEQRES sequence with ESMFold2 "
            "(BioHub Forge, requires an API key) and fill gaps with it — use when "
            "no AlphaFold model is available. No UniProt accession needed.")
        self._comp_esm_btn.clicked.connect(self._run_composite_build_esmfold2)
        ctrl_row.addWidget(self._comp_esm_btn)

        self._comp_export_btn = QPushButton("Export PDB…")
        self._comp_export_btn.setObjectName("secondary_btn")
        self._comp_export_btn.setEnabled(False)
        self._comp_export_btn.clicked.connect(self._export_composite_pdb)
        ctrl_row.addWidget(self._comp_export_btn)

        self._comp_clear_btn = QPushButton("Clear")
        self._comp_clear_btn.setObjectName("secondary_btn")
        self._comp_clear_btn.setToolTip("Reset all inputs and results in this tab")
        self._comp_clear_btn.clicked.connect(self._comp_clear)
        ctrl_row.addWidget(self._comp_clear_btn)

        ctrl_row.addStretch(1)

        # Legend
        _legend = QLabel(
            "<span style='color:#0053d6'>■</span> Experimental &nbsp;"
            "<span style='color:#65cbf3'>■</span> AF fill (pLDDT≥70) &nbsp;"
            "<span style='color:#ff7d45'>■</span> AF low-conf (&lt;70)")
        _legend.setTextFormat(Qt.TextFormat.RichText)
        ctrl_row.addWidget(_legend)
        layout.addLayout(ctrl_row)

        # ── Coverage bar (HTML) ───────────────────────────────────────────────
        self._comp_coverage_bar = QTextBrowser()
        self._comp_coverage_bar.setMaximumHeight(48)
        self._comp_coverage_bar.setHtml(
            "<p style='color:#888;font-size:11px;margin:4px'>"
            "Coverage bar will appear here after build.</p>")
        layout.addWidget(self._comp_coverage_bar)

        # ── Stats label ───────────────────────────────────────────────────────
        self._comp_stats_lbl = QLabel("")
        self._comp_stats_lbl.setObjectName("status_lbl")
        self._comp_stats_lbl.setWordWrap(True)
        layout.addWidget(self._comp_stats_lbl)

        # ── Mol* viewer ───────────────────────────────────────────────────────
        self._composite_viewer = None
        self._composite_pdb_str = None

        if _WEBENGINE_AVAILABLE:
            from PySide6.QtWebEngineWidgets import QWebEngineView as _WEV
            self._composite_viewer = _WEV()
            self._composite_viewer.setMinimumHeight(400)

            # Inject bundled molstar.js at DocumentCreation
            try:
                import os as _os
                from PySide6.QtWebEngineCore import QWebEngineScript as _WES
                _js_path  = _os.path.join(_os.path.dirname(__file__), "molstar.js")
                _css_path = _os.path.join(_os.path.dirname(__file__), "molstar.css")
                with open(_js_path,  "r", encoding="utf-8") as _f:
                    _molstar_src = _f.read()
                with open(_css_path, "r", encoding="utf-8") as _f:
                    _molstar_css = _f.read()
                _s = _WES()
                _s.setName("molstar-composite")
                _s.setSourceCode(_molstar_src)
                _s.setInjectionPoint(_WES.InjectionPoint.DocumentCreation)
                _s.setWorldId(_WES.ScriptWorldId.MainWorld)
                self._composite_viewer.page().scripts().insert(_s)
            except Exception:
                _molstar_css = ""

            _bg = "#1a1a2e" if getattr(self, "_is_dark", False) else "#ffffff"
            _html = (self._COMPOSITE_HTML
                     .replace("__BG__", _bg)
                     .replace("__CSS__", _molstar_css))
            self._composite_viewer.setHtml(
                _html,
                __import__("PySide6.QtCore", fromlist=["QUrl"]).QUrl("qrc:///"))
            layout.addWidget(self._composite_viewer, 1)
        else:
            _ph = QLabel("Install Qt WebEngine to enable the 3D viewer:  pip install PySide6-Addons")
            _ph.setAlignment(Qt.AlignmentFlag.AlignCenter)
            _ph.setObjectName("placeholder_lbl")
            layout.addWidget(_ph, 1)

    def _comp_js(self, code: str) -> None:
        if self._composite_viewer is not None:
            self._composite_viewer.page().runJavaScript(code)

    def _comp_fetch_pdb(self) -> None:
        """Fetch the experimental PDB for the Fix PDB tab (independent of main structure viewer)."""
        pdb_id = self._comp_pdb_input.text().strip().upper()
        if not pdb_id or len(pdb_id) != 4:
            QMessageBox.warning(self, "Invalid PDB ID", "Enter a 4-character PDB ID (e.g. 1EMA).")
            return
        if self._comp_pdb_fetch_worker and self._comp_pdb_fetch_worker.isRunning():
            return
        self._comp_pdb_fetch_btn.setEnabled(False)
        self._comp_pdb_status_lbl.setText(f"Fetching {pdb_id}…")
        self._comp_pdb_fetch_worker = FetchPDBStructureWorker(pdb_id)
        self._comp_pdb_fetch_worker.fetched.connect(self._comp_on_pdb_fetched)
        self._comp_pdb_fetch_worker.error.connect(self._comp_on_pdb_error)
        self._comp_pdb_fetch_worker.start()

    def _comp_on_pdb_fetched(self, pdb_str: str) -> None:
        """Handle successful PDB fetch for Fix PDB tab."""
        self._comp_pdb_fetch_btn.setEnabled(True)
        # Strip water, ions, and ligands up front so only protein atoms remain —
        # heteroatoms (e.g. a Ca²⁺ ion named "CA") would otherwise corrupt the
        # experimental↔predicted sequence alignment during gap-filling.
        from beer.analysis.composite_structure import strip_to_protein
        self._comp_exp_pdb_str = strip_to_protein(pdb_str)
        pdb_id = self._comp_pdb_input.text().strip().upper()

        issues = self._analyze_pdb_errors(self._comp_exp_pdb_str)
        self._comp_show_issues(issues, pdb_id)

        if any(i["level"] == "error" for i in issues):
            self._comp_pdb_status_lbl.setText(
                f"{pdb_id} loaded — {sum(i['level']=='error' for i in issues)} error(s) detected")
        elif issues:
            self._comp_pdb_status_lbl.setText(
                f"{pdb_id} loaded — {len(issues)} notice(s)")
        else:
            self._comp_pdb_status_lbl.setText(f"{pdb_id} loaded ✓")

        self._update_composite_btn_state()

    def _comp_show_issues(self, issues: list, pdb_id: str) -> None:
        """Render issues list into _comp_issues_browser."""
        if not issues:
            self._comp_issues_browser.setHtml(
                f"<p style='color:#4caf50;font-size:11px;margin:4px'>"
                f"✓ No structural issues detected in {pdb_id}.</p>")
            return
        _color = {"error": "#ff6b35", "warn": "#ffc107", "info": "#888888"}
        _icon  = {"error": "✖", "warn": "⚠", "info": "ℹ"}
        rows = "".join(
            f"<tr><td style='color:{_color[i['level']]};padding:1px 6px 1px 2px;'>"
            f"{_icon[i['level']]}</td>"
            f"<td style='color:{_color[i['level']]};font-size:11px;padding:1px 0;'>"
            f"{i['msg']}</td></tr>"
            for i in issues
        )
        self._comp_issues_browser.setHtml(
            f"<table style='font-size:11px;margin:4px;border-spacing:0;'>{rows}</table>")

    @staticmethod
    def _analyze_pdb_errors(pdb_str: str) -> list:
        """Scan PDB text for common structural issues.

        Returns list of {level: 'error'|'warn'|'info', msg: str}.
        Levels: error = will impede modelling; warn = may cause issues; info = informational.
        """
        from beer.analysis.composite_structure import _AA3TO1 as _aa3to1
        import math as _math

        issues: list = []
        chains: dict  = {}     # chain -> [(resseq, resname, icode)]
        chain_atoms: dict = {} # chain -> {resseq -> set(atom_names)}
        chain_ca: dict = {}    # chain -> [(resseq, x, y, z)]
        altloc_res: set = set()
        model_count = 0

        for line in pdb_str.splitlines():
            if line.startswith("MODEL"):
                model_count += 1
                continue
            is_atom  = line.startswith("ATOM  ")
            is_hetat = line.startswith("HETATM")
            if not (is_atom or is_hetat) or len(line) < 27:
                continue
            try:
                resseq = int(line[22:26])
            except ValueError:
                continue
            chain   = line[21] if len(line) > 21 else " "
            resname = line[17:20].strip()
            atname  = line[12:16].strip()
            altloc  = line[16] if len(line) > 16 else " "
            icode   = line[26] if len(line) > 26 else " "

            if altloc not in (" ", "A"):
                altloc_res.add((chain, resseq))

            is_aa = _aa3to1.get(resname, "X") != "X"
            if not (is_atom and is_aa) and not (is_hetat and is_aa):
                continue

            if chain not in chains:
                chains[chain] = []
                chain_atoms[chain] = {}
                chain_ca[chain] = []

            seen_keys = {(r[0], r[2]) for r in chains[chain]}
            if (resseq, icode) not in seen_keys:
                chains[chain].append((resseq, resname, icode))

            chain_atoms[chain].setdefault(resseq, set()).add(atname)

            if atname == "CA" and is_atom:
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    if not any(r[0] == resseq for r in chain_ca[chain]):
                        chain_ca[chain].append((resseq, x, y, z))
                except ValueError:
                    pass

        if model_count > 1:
            issues.append({"level": "warn",
                           "msg": f"Multiple NMR/ensemble models ({model_count}); only MODEL 1 used."})

        if altloc_res:
            issues.append({"level": "info",
                           "msg": f"Alternate conformations in {len(altloc_res)} residue(s); only altloc A retained."})

        for ch in sorted(chains.keys()):
            residues = sorted(chains[ch], key=lambda r: (r[0], r[2]))

            # Missing residue gaps (numbering holes)
            gaps = []
            for i in range(len(residues) - 1):
                curr_seq = residues[i][0]
                next_seq = residues[i + 1][0]
                if next_seq - curr_seq > 1:
                    gaps.append(f"{curr_seq + 1}–{next_seq - 1}")
            if gaps:
                extra = f" (+{len(gaps)-5} more)" if len(gaps) > 5 else ""
                issues.append({"level": "error",
                               "msg": f"Chain {ch}: {len(gaps)} gap(s) — missing residues "
                                      f"{', '.join(gaps[:5])}{extra}"})

            # Chain breaks (Cα–Cα > 4.5 Å between sequential residues)
            ca_list = chain_ca[ch]
            breaks = []
            for i in range(len(ca_list) - 1):
                r1, x1, y1, z1 = ca_list[i]
                r2, x2, y2, z2 = ca_list[i + 1]
                d = _math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
                if d > 4.5:
                    breaks.append(f"res {r1}–{r2} ({d:.1f} Å)")
            if breaks:
                extra = f" (+{len(breaks)-3} more)" if len(breaks) > 3 else ""
                issues.append({"level": "error",
                               "msg": f"Chain {ch}: backbone break(s) — {', '.join(breaks[:3])}{extra}"})

            # Incomplete backbone (missing N, CA, or C)
            bad_bb = []
            for resseq, resname, icode in residues:
                atoms = chain_atoms[ch].get(resseq, set())
                missing = [a for a in ("N", "CA", "C") if a not in atoms]
                if missing:
                    bad_bb.append(f"{resname}{resseq}({','.join(missing)})")
            if bad_bb:
                extra = f" (+{len(bad_bb)-5} more)" if len(bad_bb) > 5 else ""
                issues.append({"level": "warn",
                               "msg": f"Chain {ch}: incomplete backbone — "
                                      f"{', '.join(bad_bb[:5])}{extra}"})

            # Non-canonical ATOM residues
            noncanon = sorted({r[1] for r in residues
                               if _aa3to1.get(r[1], "X") == "X"
                               and r[1] not in ("HOH", "WAT", "DOD")})
            if noncanon:
                issues.append({"level": "info",
                               "msg": f"Chain {ch}: non-canonical residues — "
                                      f"{', '.join(noncanon)}"})

        return issues

    def _comp_on_pdb_error(self, msg: str) -> None:
        self._comp_pdb_fetch_btn.setEnabled(True)
        self._comp_pdb_status_lbl.setText("Fetch failed")
        QMessageBox.warning(self, "PDB Fetch Failed", msg)

    def _update_composite_btn_state(self) -> None:
        """Enable/disable Build button based on current state."""
        if not hasattr(self, "_comp_build_btn"):
            return
        has_exp = self._comp_exp_pdb_str is not None
        acc = self._comp_acc_input.text().strip()
        has_acc = bool(acc)
        self._comp_build_btn.setEnabled(has_exp and has_acc)
        # ESMFold2 path needs only the experimental PDB (no UniProt accession).
        if hasattr(self, "_comp_esm_btn"):
            self._comp_esm_btn.setEnabled(has_exp)
        if not has_exp:
            self._comp_status_lbl.setText(
                "Step 1: enter a PDB ID above and click Fetch PDB.")
        elif not has_acc:
            self._comp_status_lbl.setText(
                "Step 2: enter the UniProt accession (e.g. P42212 for GFP) — "
                "used to fetch the matching AlphaFold model.")
        else:
            self._comp_status_lbl.setText(
                f"Ready — experimental structure loaded, accession {acc}. "
                "Click Fix PDB to fetch AlphaFold and fill missing residues.")

    def _comp_set_build_btns_enabled(self, enabled: bool) -> None:
        """Enable/disable both gap-fill buttons together (gated by current state)."""
        self._update_composite_btn_state()
        if not enabled:
            self._comp_build_btn.setEnabled(False)
            self._comp_esm_btn.setEnabled(False)

    def _run_composite_build(self) -> None:
        acc = self._comp_acc_input.text().strip()
        if not self._comp_exp_pdb_str or not acc:
            return
        if self._composite_worker and self._composite_worker.isRunning():
            return
        if self._composite_esm_worker and self._composite_esm_worker.isRunning():
            return

        self._comp_set_build_btns_enabled(False)
        self._comp_status_lbl.setText("Fetching AlphaFold model…")

        self._composite_worker = CompositeStructureWorker(
            self._comp_exp_pdb_str, acc)
        self._composite_worker.progress.connect(self._comp_status_lbl.setText)
        self._composite_worker.finished.connect(
            lambda r: self._on_composite_finished(r, "AlphaFold"))
        self._composite_worker.error.connect(self._on_composite_error)
        self._composite_worker.start()

    def _comp_full_sequence(self) -> str | None:
        """Sequence to fold for the ESMFold2 gap-fill, for the primary chain.

        Built from the experimental structure's resolved residue-number range, with
        placeholder residues at every unresolved position (``build_fold_sequence``).
        This guarantees the folded model has a residue at each gap so the gaps can
        actually be filled — unlike SEQRES/FASTA, which collapse modified residues
        (e.g. the GFP chromophore) to a single ``X`` and would leave multi-residue
        gaps unfillable. SEQRES then the RCSB FASTA are fallbacks (with ``X`` mapped
        to a placeholder rather than deleted, which would shift the registration).
        Returns None when no sequence can be obtained.
        """
        from beer.analysis.composite_structure import (
            build_fold_sequence, parse_seqres, primary_seqres_chain)
        pdb_str = self._comp_exp_pdb_str or ""
        seq = build_fold_sequence(pdb_str)
        if seq and len(seq) >= 4:
            return seq
        # Fallback 1: SEQRES (X → placeholder, not deleted).
        chain = primary_seqres_chain(pdb_str)
        sq = parse_seqres(pdb_str, chain).replace("X", "A")
        if sq and len(sq) >= 4:
            return sq
        # Fallback 2: RCSB entry FASTA (longest record).
        pdb_id = self._comp_pdb_input.text().strip().upper()
        if len(pdb_id) == 4:
            try:
                fasta = self._fetch_pdb_fasta(pdb_id)
                records, cur = [], []
                for line in fasta.splitlines():
                    if line.startswith(">"):
                        if cur:
                            records.append("".join(cur)); cur = []
                    else:
                        cur.append(line.strip())
                if cur:
                    records.append("".join(cur))
                if records:
                    return max(records, key=len).replace("X", "A")
            except Exception:
                pass
        return None

    def _run_composite_build_esmfold2(self) -> None:
        """Fill gaps using an ESMFold2 prediction instead of AlphaFold."""
        if not self._comp_exp_pdb_str:
            return
        if self._composite_worker and self._composite_worker.isRunning():
            return
        if self._composite_esm_worker and self._composite_esm_worker.isRunning():
            return

        full_seq = self._comp_full_sequence()
        if not full_seq:
            QMessageBox.warning(
                self, "No Full Sequence",
                "Could not obtain a complete (SEQRES/FASTA) sequence for this "
                "entry. ESMFold2 needs the full construct sequence — including "
                "missing residues — to fill gaps.")
            return

        api_token = _config.load().get("biohub_api_key", "").strip()
        if not api_token:
            _mb = QMessageBox(self)
            _mb.setWindowTitle("BioHub API Key Required")
            _mb.setIcon(QMessageBox.Icon.Warning)
            _mb.setTextFormat(Qt.TextFormat.RichText)
            _mb.setText(
                "No BioHub API key found.<br><br>"
                "Register at <a href='https://biohub.ai'>biohub.ai</a> to get a free key,<br>"
                "then add it in <b>Settings → BioHub API Key</b>."
            )
            _mb.exec()
            return

        self._comp_set_build_btns_enabled(False)
        self._comp_status_lbl.setText("Predicting full structure with ESMFold2…")

        self._composite_esm_worker = CompositeStructureESMFold2Worker(
            self._comp_exp_pdb_str, full_seq, api_token)
        self._composite_esm_worker.progress.connect(self._comp_status_lbl.setText)
        self._composite_esm_worker.finished.connect(
            lambda r: self._on_composite_finished(r, "ESMFold2"))
        self._composite_esm_worker.error.connect(self._on_composite_error)
        self._composite_esm_worker.start()

    def _on_composite_finished(self, result, source: str = "AlphaFold") -> None:
        from beer.analysis.composite_structure import CompositeResult
        self._comp_set_build_btns_enabled(True)
        self._composite_pdb_str = result.pdb_str
        # Short tag for source-specific UI text (AF / ESMFold2).
        _fill_tag = "ESMFold2" if source == "ESMFold2" else "AF"

        # ── Stats label ───────────────────────────────────────────────────────
        n_total = result.n_experimental + result.n_predicted + result.n_low_confidence
        pct_exp  = 100 * result.n_experimental / max(1, n_total)
        pct_pred = 100 * result.n_predicted / max(1, n_total)
        pct_lc   = 100 * result.n_low_confidence / max(1, n_total)
        warn_html = ""
        if result.junction_warnings:
            warn_html = ("<br><span style='color:#ff6b35'>"
                         + "<br>".join(result.junction_warnings) + "</span>")
        # ESMFold2 gap-fill uses placeholder residue identities (see
        # build_fold_sequence) — warn that only backbone geometry is meaningful.
        if source == "ESMFold2" and (result.n_predicted + result.n_low_confidence) > 0:
            warn_html += ("<br><span style='color:#ff6b35'>⚠ Gap residue identities "
                          "are placeholders (alanine); only backbone geometry is "
                          "modelled. Use the AlphaFold path for exact gap identities."
                          "</span>")
        lc_html = ""
        if result.n_low_confidence:
            lc_html = (f", <span style='color:#aaaaaa'>"
                       f"{result.n_low_confidence} low-confidence fill "
                       f"(pLDDT&lt;70, {pct_lc:.0f}%)</span>")
        stats = (
            f"<b>{n_total} residues total</b> — "
            f"{result.n_experimental} experimental ({pct_exp:.0f}%), "
            f"<span style='color:#FF8C00'>{result.n_predicted} {_fill_tag} fill "
            f"(pLDDT≥70, {pct_pred:.0f}%)</span>{lc_html} "
            f"in {result.n_gaps} gap region{'s' if result.n_gaps != 1 else ''}. "
            f"Fit RMSD: {result.rmsd_fit:.2f} Å.{warn_html}"
        )
        self._comp_stats_lbl.setText(stats)
        self._comp_stats_lbl.setTextFormat(Qt.TextFormat.RichText)
        self._comp_status_lbl.setText(
            f"Complete structure built successfully ({source} gap-fill).")

        # ── Coverage bar ──────────────────────────────────────────────────────
        # Qt's rich-text engine does not support display:inline-block on spans;
        # use a table with bgcolor-coloured cells (well-supported subset).
        _src_color = {
            "experimental":   "#4a90d9",
            "predicted":      "#FF8C00",
            "low_confidence": "#aaaaaa",
        }
        cells = []
        for r in result.residues:
            col = _src_color.get(r.source, "#888888")
            cells.append(f"<td width='3' bgcolor='{col}'></td>")
        bar_html = (
            "<table cellspacing='0' cellpadding='0' border='0' width='100%'>"
            "<tr><td width='60' style='font-size:10px;color:#888;'>"
            "Coverage:</td>"
            + "".join(cells)
            + "</tr></table>")
        self._comp_coverage_bar.setHtml(bar_html)

        # ── Load 3Dmol viewer ─────────────────────────────────────────────────
        pred_resis    = [r.resi_composite for r in result.residues
                         if r.source == "predicted"]
        lowconf_resis = [r.resi_composite for r in result.residues
                         if r.source == "low_confidence"]
        import json as _json
        self._comp_js(
            f"loadComposite({_json.dumps(result.pdb_str)}, "
            f"{_json.dumps(pred_resis)}, "
            f"{_json.dumps(lowconf_resis)});")

        self._comp_export_btn.setEnabled(True)

    def _on_composite_error(self, msg: str) -> None:
        self._comp_set_build_btns_enabled(True)
        self._comp_status_lbl.setText(f"Error: {msg}")
        QMessageBox.warning(self, "Composite Structure Error", msg)

    def _comp_clear(self) -> None:
        """Reset all Fix PDB tab inputs and results to initial state."""
        # Stop any running workers
        for attr in ("_comp_pdb_fetch_worker", "_composite_worker",
                     "_composite_esm_worker"):
            w = getattr(self, attr, None)
            if w and w.isRunning():
                w.quit()
                w.wait(_WORKER_WAIT_MS)

        # Clear stored state
        self._comp_exp_pdb_str  = None
        self._composite_pdb_str = None

        # Reset input fields
        self._comp_pdb_input.clear()
        self._comp_acc_input.clear()

        # Reset labels / status
        self._comp_pdb_status_lbl.setText("")
        self._comp_stats_lbl.setText("")
        self._comp_status_lbl.setText(
            "Step 1: enter a PDB ID and fetch the experimental structure. "
            "Step 2: enter the UniProt accession for the AlphaFold model. "
            "Step 3: click Fix PDB.")

        # Reset issues browser
        self._comp_issues_browser.setHtml(
            "<p style='color:#888;font-size:11px;margin:4px'>"
            "Fetch a PDB to see structural issues.</p>")

        # Reset coverage bar
        self._comp_coverage_bar.setHtml(
            "<p style='color:#888;font-size:11px;margin:4px'>"
            "Coverage bar will appear here after build.</p>")

        # Reset buttons
        self._comp_build_btn.setEnabled(False)
        self._comp_esm_btn.setEnabled(False)
        self._comp_export_btn.setEnabled(False)
        self._comp_pdb_fetch_btn.setEnabled(True)

        # Clear 3Dmol viewer
        self._comp_js("if(typeof _viewer!=='undefined'&&_viewer&&_viewer.plugin){_viewer.plugin.clear().catch(function(){});}")

    def _export_composite_pdb(self) -> None:
        if not self._composite_pdb_str:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Composite PDB", "", "PDB files (*.pdb);;All files (*)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._composite_pdb_str)
            self.statusBar.showMessage(f"Exported: {path}", 3000)

    # ── New method callbacks ──────────────────────────────────────────────────

    def run_truncation_series(self):
        if not self.analysis_data:
            QMessageBox.warning(self, "Truncation", "Run analysis first.")
            return
        if self._trunc_worker and self._trunc_worker.isRunning():
            self._trunc_worker.cancel()
            self._trunc_worker.wait(_WORKER_WAIT_MS)
        seq  = self.analysis_data["seq"]
        step = self.trunc_step_spin.value()
        do_n = self.trunc_nterm_cb.isChecked()
        do_c = self.trunc_cterm_cb.isChecked()
        self._trunc_rows = []
        self.trunc_table.setRowCount(0)
        self.trunc_run_btn.setEnabled(False)
        # Progress dialog — stays open until worker finishes or is cancelled
        n_variants = ((len(range(step, 100, step)) + 1) *
                      ((1 if do_n else 0) + (1 if do_c else 0)))
        from PySide6.QtWidgets import QProgressDialog
        from PySide6.QtCore import Qt as _Qt
        self._trunc_progress = QProgressDialog(
            "Running truncation series…", "Cancel", 0, n_variants, self)
        self._trunc_progress.setWindowModality(_Qt.WindowModality.WindowModal)
        self._trunc_progress.setMinimumDuration(0)
        self._trunc_progress.setValue(0)
        from beer.network.workers import TruncationWorker
        self._trunc_worker = TruncationWorker(
            seq, self._embedder, step=step, do_n=do_n, do_c=do_c,
            ph=self.default_pH, window=self.default_window_size,
            reducing=self.use_reducing, pka=self.custom_pka, parent=self)
        self._trunc_worker.row_ready.connect(self._on_trunc_row)
        self._trunc_worker.finished.connect(self._on_trunc_finished)
        self._trunc_worker.error.connect(self._on_trunc_error)
        self._trunc_worker.progress.connect(
            lambda msg: self._trunc_progress.setLabelText(msg))
        self._trunc_progress.canceled.connect(self._trunc_worker.cancel)
        self._trunc_worker.start()

    def _on_trunc_row(self, ttype: str, pct: int, rem: int, d: dict):
        self._trunc_rows.append((ttype, pct, rem, d))
        row = self.trunc_table.rowCount()
        self.trunc_table.insertRow(row)
        for col, val in enumerate([
            ttype, f"{pct}%", str(rem),
            f"{d['mol_weight']:.2f}", f"{d['iso_point']:.2f}",
            f"{d['gravy']:.3f}", f"{d['fcr']:.3f}", f"{d['ncpr']:+.3f}",
        ]):
            self.trunc_table.setItem(row, col, QTableWidgetItem(val))
        prog = getattr(self, "_trunc_progress", None)
        if prog:
            prog.setValue(row + 1)

    def _on_trunc_finished(self):
        prog = getattr(self, "_trunc_progress", None)
        if prog:
            prog.close()
        self.trunc_run_btn.setEnabled(True)
        rows = getattr(self, "_trunc_rows", [])
        self.trunc_table.resizeColumnsToContents()
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

    def _on_trunc_error(self, msg: str):
        prog = getattr(self, "_trunc_progress", None)
        if prog:
            prog.close()
        self.trunc_run_btn.setEnabled(True)
        QMessageBox.warning(self, "Truncation Error", msg)

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
        self.msa_pssm_btn.setEnabled(True)
        # Display alignment preview
        preview_lines = []
        for name, aln_seq in zip(names, aligned):
            preview_lines.append(f"<b>{name[:20]}</b>  <tt>{aln_seq[:80]}{'…' if len(aln_seq)>80 else ''}</tt>")
        self.msa_viewer.setHtml("<br>".join(preview_lines))
        # Conservation graph
        if _HAS_NEW_GRAPHS:
            fig = create_msa_conservation_figure(
                aligned, names,
                label_font=self.label_font_size, tick_font=self.tick_font_size)
            self._replace_graph("MSA Conservation", fig)
            self.msa_viewer.append(
                "<hr><p>\u25b6 <a href='opengraph:MSA Conservation' "
                "style='color:#4361ee;font-weight:bold;text-decoration:none;'>"
                "Open MSA Conservation graph</a></p>")
        # MSA graphs are self-contained \u2014 make the Graphs tab reachable even
        # when the main Analysis tab has not been run.
        self._set_nav_tab_enabled(self.main_tabs.stack_for_name("Graphs"), True)
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
                "<p>\u25b6 <a href='opengraph:MSA Covariance' "
                "style='color:#4361ee;font-weight:bold;text-decoration:none;'>"
                "Open MSA Covariance graph</a></p>")
        self.statusBar.showMessage(
            f"MSA: {len(aligned)} sequences, {n_cols} alignment columns", 3000)

    def _clear_msa(self):
        self._msa_sequences = []
        self._msa_names     = []
        self._msa_mi_apc    = None
        self.msa_pssm_btn.setEnabled(False)
        self.msa_input.clear()
        self.msa_viewer.clear()
        self.statusBar.showMessage("MSA cleared.", 2000)

    def _export_pssm(self):
        """Write a position-specific scoring matrix for the current alignment to CSV."""
        if not self._msa_sequences:
            QMessageBox.warning(self, "PSSM", "Run an MSA analysis first.")
            return
        from beer.analysis.msa_pssm import (
            compute_pssm, pssm_to_csv, consensus_sequence)
        from beer.io.provenance import text_header
        try:
            rows, conservation, coverage = compute_pssm(self._msa_sequences)
        except ValueError as e:
            QMessageBox.warning(self, "PSSM", str(e))
            return
        consensus = consensus_sequence(self._msa_sequences)
        csv_text  = pssm_to_csv(rows, conservation, coverage, consensus)
        path, _ = QFileDialog.getSaveFileName(
            self, "Export PSSM", "pssm.csv", "CSV files (*.csv);;All files (*)")
        if not path:
            return
        try:
            header = text_header(
                comment="# ",
                title=(f"MSA PSSM (log2 odds vs. Swiss-Prot background) — "
                       f"{len(self._msa_sequences)} sequences, "
                       f"{len(rows)} alignment columns"))
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(header)
                fh.write(csv_text)
        except OSError as e:
            QMessageBox.warning(self, "PSSM", f"Could not write file:\n{e}")
            return
        self.statusBar.showMessage(f"PSSM exported: {path}", 4000)

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
        from beer.utils.biophysics import calc_isoelectric_point as _calc_pi
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
            pi  = _calc_pi(seq)
            n_cys_pairs = seq.count("C") // 2 if not self.use_reducing else 0
            ext = 5500 * seq.count("W") + 1490 * seq.count("Y") + 125 * n_cys_pairs
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
                    f"<tr><td>{_html_mod.escape(str(inst.get('elm_identifier','?')))}</td>"
                    f"<td>{inst.get('start','?')}</td><td>{inst.get('end','?')}</td>"
                    f"<td>{_html_mod.escape(str(inst.get('logic','?')))}</td></tr>")
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
        self._mark_chip_error(self.fetch_elm_btn)
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
            lines = [f"<h2>DisProt: {_html_mod.escape(data.get('disprot_id','?'))}</h2>"
                     f"<p>{_html_mod.escape(data.get('protein_name',''))}</p>"
                     f"<p>Fraction disordered: {frac:.3f}</p>"
                     "<table><tr><th>Start</th><th>End</th><th>Type</th></tr>"]
            for r in regions:
                lines.append(
                    f"<tr><td>{r['start']}</td><td>{r['end']}</td>"
                    f"<td>{_html_mod.escape(r.get('type','IDR'))}</td></tr>")
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
        self._mark_chip_error(self.fetch_disprot_btn)
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
                   f"<p>Source: {_html_mod.escape(str(data.get('source','PhaSepDB')))}</p>"
                   f"<p>Category: <b>{_html_mod.escape(str(data.get('category','?')))}</b></p>"
                   f"<p>Evidence type: {_html_mod.escape(str(data.get('evidence_type','?')))}</p>")
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
        self._mark_chip_error(self.fetch_phasepdb_btn)
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
        self._mark_chip_error(self.fetch_mobidb_btn)
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
            "secondary_structure_helix":  "Secondary Structure: Helix Profile",
            "secondary_structure_strand": "Secondary Structure: Strand Profile",
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
        # Enable/disable Clear Tracks button based on whether any features were loaded
        has_data = bool(n)
        if hasattr(self, "_graphs_clear_tracks_btn"):
            self._graphs_clear_tracks_btn.setEnabled(has_data)
        # Always rebuild graph generators with the new UniProt data so overlays appear
        if self.analysis_data:
            self.update_graph_tabs()

    def _clear_uniprot_features(self) -> None:
        """Remove all UniProt annotation overlays from graphs."""
        self._uniprot_features = {}
        self._update_current_snapshot()
        if hasattr(self, "_graphs_clear_tracks_btn"):
            self._graphs_clear_tracks_btn.setEnabled(False)
        self._mark_chip_normal(self.fetch_uniprot_tracks_btn)
        if self.analysis_data:
            self.update_graph_tabs()
        self.statusBar.showMessage("UniProt annotation overlays removed.", 3000)

    def _on_uniprot_features_error(self, msg: str):
        self.fetch_uniprot_tracks_btn.setEnabled(True)
        self._mark_chip_error(self.fetch_uniprot_tracks_btn)
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
        reply = QMessageBox.question(
            self, "UniProt Match Found",
            f"A matching UniProt Swiss-Prot entry was found:\n\n"
            f"    {acc}\n\n"
            f"Load this entry? (This will fetch the accession and populate all data.)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            self.statusBar.showMessage(f"UniProt match {acc} not loaded.", 3000)
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
            desc = _html_mod.escape(v.get("description", "")[:80])
            vtype = _html_mod.escape(str(v.get("type", "")))
            lines.append(
                f"<tr><td>{v.get('position','?')}</td>"
                f"<td>{_html_mod.escape(str(v.get('original','?')))}</td>"
                f"<td>{_html_mod.escape(str(v.get('variant','?')))}</td>"
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
        self._mark_chip_error(self.fetch_variants_btn)
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
                f"<td>{_html_mod.escape(str(ix.get('partner_id', '—')))}</td>"
                f"<td>{_html_mod.escape(str(ix.get('partner_name', '—')))}</td>"
                f"<td>{_html_mod.escape(str(ix.get('detection_method', '—')))}</td>"
                f"<td>{_html_mod.escape(str(ix.get('interaction_type', '—')))}</td>"
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
        self._mark_chip_error(self.fetch_intact_btn)
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
                        with self._figure_export_light(src_fig):
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
            with self._figure_export_light(fig_out):
                fig_out.savefig(fn, format=ext, dpi=300, bbox_inches="tight",
                                transparent=use_transparent, metadata=figure_metadata(ext),
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


    def _push_undo(self):
        """Snapshot the current full state before a destructive action (Ctrl+Z)."""
        if not self.analysis_data:
            return
        try:
            self._undo_stack.append(self._make_snapshot(self.analysis_data))
            self._undo_stack = self._undo_stack[-20:]   # cap depth
        except Exception:
            pass

    def _undo_mutation(self):
        """Ctrl+Z \u2014 restore the state before the last mutation / Clear All / load."""
        if self._undo_stack:
            snap = self._undo_stack.pop()
            self._restore_snapshot(snap)
            self.statusBar.showMessage(
                f"Undo \u2014 restored '{snap.get('name', 'previous state')}'", 3000)
            return
        # Legacy single-slot fallback (pre-analysis mutation)
        if self._undo_seq is not None:
            self.seq_text.setPlainText(self._undo_seq)
            self.sequence_name = self._undo_name
            self._undo_seq = None
            self._undo_name = None
            self.statusBar.showMessage("Mutation undone \u2014 re-running analysis\u2026", 2000)
            self.on_analyze()
            return
        self.statusBar.showMessage("Nothing to undo.", 2000)

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
            # Category row — toggle expand/collapse
            item.setExpanded(not item.isExpanded())
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
                banner = (f"<p class='callout-info' style='padding:4px 8px;"
                          f"margin:0 0 6px 0;font-size:9pt;'>"
                          f"<b>Summary:</b> {summary}</p>")
                if "Summary:" not in current_html:
                    browser.setHtml(banner + current_html)

    # ── Inline sparklines ────────────────────────────────────────────────────

    # Maps each report-section name to: data key in analysis_data, full graph
    # title (for navigation), sparkline colour, optional threshold value.
    _SPARKLINE_MAP: dict[str, tuple[str, str, str, float | None]] = {
        # section                     data_key                    graph_title                       colour     threshold
        "Disorder":               ("disorder_scores",         "Disorder Profile",               "#4361ee", 0.56235),
        "Hydrophobicity":         ("hydro_profile",           "Hydrophobicity Profile",         "#f77f00", 0.0),
        "Charge":                 ("ncpr_profile",            "Local Charge Profile",           "#e63946", 0.0),
        "Charge Decoration (SCD)":("scd_profile",             "SCD Profile",                    "#9b5de5", None),
        "Hydrophobicity Decoration (SHD)": ("shd_profile",   "SHD Profile",                    "#f77f00", None),
        "β-Aggregation & Solubility": ("aggr_profile",        "β-Aggregation Profile",          "#e07a5f", 1.0),
        "Signal Peptide & GPI":   ("sp_bilstm_profile",       "Signal Peptide Profile",          "#f72585", 0.70173),
        "TM Helices":             ("tm_bilstm_profile",       "Transmembrane Profile",           "#34d399", 0.81339),
        "Intramembrane":          ("intramem_bilstm_profile", "Intramembrane Profile",           "#6ee7b7", 0.67273),
        "Coiled-Coil":            ("cc_bilstm_profile",       "Coiled-Coil Profile",             "#fb923c", 0.55178),
        "DNA-Binding":            ("dna_bilstm_profile",      "DNA-Binding Profile",             "#60a5fa", 0.87760),
        "Active Site":            ("act_bilstm_profile",      "Active Site Profile",             "#f87171", 0.86688),
        "Binding Site":           ("bnd_bilstm_profile",      "Binding Site Profile",            "#a78bfa", 0.98014),
        "Phosphorylation":        ("phos_bilstm_profile",     "Phosphorylation Profile",         "#fbbf24", 0.79967),
        "Low-Complexity":         ("lcd_bilstm_profile",      "Low-Complexity Profile",          "#94a3b8", 0.65838),
        "Zinc Finger":            ("znf_bilstm_profile",      "Zinc Finger Profile",             "#4ade80", 0.5),
        "Glycosylation":          ("glyc_bilstm_profile",     "Glycosylation Profile",           "#f9a8d4", 0.80024),
        "Ubiquitination":         ("ubiq_bilstm_profile",     "Ubiquitination Profile",          "#fb7185", 0.83320),
        "Methylation":            ("meth_bilstm_profile",     "Methylation Profile",             "#a3e635", 0.5),
        "Acetylation":            ("acet_bilstm_profile",     "Acetylation Profile",             "#38bdf8", 0.5),
        "Lipidation":             ("lipid_bilstm_profile",    "Lipidation Profile",              "#e879f9", 0.5),
        "Disulfide Bonds":        ("disulf_bilstm_profile",   "Disulfide Bond Profile",          "#fde68a", 0.5),
        "Functional Motifs":      ("motif_bilstm_profile",    "Functional Motif Profile",        "#c4b5fd", 0.5),
        "Propeptide":             ("prop_bilstm_profile",     "Propeptide Profile",              "#fdba74", 0.5),
        "Repeat Regions":         ("rep_bilstm_profile",      "Repeat Region Profile",           "#67e8f9", 0.5),
        "Amphipathic Helices":    ("moment_alpha",            "Hydrophobic Moment",              "#7b9cff", None),
        "SS3: α-Helix":           ("ss3_h_profile",  "Secondary Structure: Helix Profile",  "#e63946", None),
        "SS3: β-Strand":          ("ss3_e_profile",  "Secondary Structure: Strand Profile",  "#457b9d", None),
        "SS3: Coil/Loop":         ("ss3_c_profile",  "Secondary Structure: Coil Profile",    "#adb5bd", None),
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
                notice = (
                    "<div class='callout-warn'>"
                    "<b>Model not yet trained.</b>"
                    "<p style='margin:4px 0 0'>This AI prediction head has not been trained yet. "
                    "The graph and profile will appear here automatically once the model file "
                    "is available.</p>"
                    "</div>"
                )
                current = browser.toHtml()
                if "Model not yet trained" not in current:
                    browser.setHtml(current + notice)
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

    # One-letter → three-letter amino acid code
    _AA1_TO_3: dict[str, str] = {
        "A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys",
        "Q": "Gln", "E": "Glu", "G": "Gly", "H": "His", "I": "Ile",
        "L": "Leu", "K": "Lys", "M": "Met", "F": "Phe", "P": "Pro",
        "S": "Ser", "T": "Thr", "W": "Trp", "Y": "Tyr", "V": "Val",
    }

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
        seq = (self.analysis_data or {}).get("seq", "")
        mean_score = sum(scores) / len(scores) if scores else 0.0
        regions = self._get_predicted_regions(scores, threshold)
        graph_href = "beer://graph/" + _up.quote(graph_title)

        def _res_label(pos: int) -> str:
            aa = seq[pos - 1] if seq and 0 < pos <= len(seq) else ""
            return f"{self._AA1_TO_3.get(aa, aa)}-{pos}" if aa else str(pos)

        # Build regions table
        if regions:
            region_rows = "".join(
                f"<tr><td>{k}</td><td>{_res_label(s)} – {_res_label(e)}</td><td>{e - s + 1}</td></tr>"
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
            f"<p class='note'>Method: ESMC 600M embeddings → 2-layer BiLSTM classifier → sigmoid. "
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
            "<div class='callout-info'>"
            "<b>Click this section in the sidebar to compute the prediction.</b>"
            "<p style='margin:6px 0 0'>BEER will run the ESMC 600M embedding and "
            "the corresponding BiLSTM head for this feature only. "
            "The embedding is cached after the first computation, so subsequent "
            "sections are fast.</p>"
            + (""
               if _embedder_ready
               else "<p style='color:#b45309'><b>⚠ AI model not available.</b> "
                    "Click to attempt loading; check that beer is fully installed.</p>")
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

        # Soft, one-time-per-sequence notice: the 24 AI heads were trained on
        # sequences truncated to 1024 residues. Longer proteins are still scored
        # end-to-end, but predictions beyond residue 1024 are extrapolation.
        _ai_seq = self.analysis_data.get("seq", "") or ""
        if len(_ai_seq) > AI_TRAIN_MAX_LEN and self._ai_longseq_warned_seq != _ai_seq:
            self._ai_longseq_warned_seq = _ai_seq
            QMessageBox.information(
                self, "Sequence Longer Than Training Length",
                f"<b>This sequence is {len(_ai_seq):,} residues long.</b><br><br>"
                f"BEER's 24 AI prediction heads were trained on sequences truncated "
                f"to <b>{AI_TRAIN_MAX_LEN} residues</b>. The full sequence is still scored "
                f"end-to-end, but per-residue predictions <b>beyond residue "
                f"{AI_TRAIN_MAX_LEN} are extrapolation</b> and should be interpreted with "
                f"caution.<br><br>Classical (non-AI) analyses are unaffected.",
            )

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
                "<div class='callout-warn'>"
                f"<b>⏳ Running ESMC BiLSTM for <em>{display_name}</em>…</b>"
                "<p style='margin:6px 0 0'>This may take a moment on first use "
                "(the embedding is cached for subsequent sections).</p>"
                "</div>"
            ))

        # Update the graph panel placeholder to show "computing" state
        if graph_title in self.graph_tabs and graph_title not in self._generated_graphs:
            _, _vb = self.graph_tabs[graph_title]
            for _i in range(_vb.count()):
                _w = _vb.itemAt(_i).widget()
                if isinstance(_w, QLabel) and _w.objectName() == "placeholder_lbl":
                    _w.setText(f"Computing AI predictions…\n{graph_title}")
                    break

        # Warn if ESMC model needs to be downloaded first
        from beer.embeddings import ESMC_AVAILABLE
        if ESMC_AVAILABLE and self._embedder is not None:
            import pathlib
            _mn = getattr(self._embedder, "model_name", "esmc_600m")
            _hf_hub = pathlib.Path.home() / ".cache/huggingface/hub"
            _downloaded = any(_hf_hub.glob("models--EvolutionaryScale--esmc-*"))
            if not _downloaded and not getattr(self, "_esm2_download_warned", False):
                self._esm2_download_warned = True
                _sizes = {
                    "esm2_t6_8M_UR50D": "~30 MB", "esm2_t12_35M_UR50D": "~140 MB",
                    "esmc_300m": "~1.2 GB", "esmc_600m": "~2.4 GB",
                }
                _sz = _sizes.get(_mn, "~2.6 GB")
                reply = QMessageBox.information(
                    self, "First-time ESMC Setup",
                    f"<b>One-time model download required</b><br><br>"
                    f"The ESMC 600M language model ({_sz}) will be downloaded "
                    f"from Meta's model hub on the first run.<br><br>"
                    f"<b>Estimated time:</b> 2–15 minutes depending on your connection.<br>"
                    f"<b>Location:</b> <code>~/.cache/huggingface/hub/</code><br><br>"
                    f"The download runs in the background — BEER remains responsive. "
                    f"The model is cached permanently; subsequent runs are instant.",
                    QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
                )
                if reply == QMessageBox.StandardButton.Cancel:
                    return

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
        thr = threshold if threshold is not None else 0.5
        html = _style + self._build_ai_head_html(
            display_name, scores, graph_title, auroc, threshold=thr,
            sparkline_uri=sparkline_uri)

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
                "<div class='callout-error'>"
                f"<b>Computation failed.</b><pre style='font-size:9pt;"
                f"white-space:pre-wrap'>{msg}</pre>"
                "<p>Check that ESMC and the model file are installed correctly, "
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
            copy_btn.setMaximumWidth(112)
            copy_btn.setMinimumHeight(26)
            copy_btn.clicked.connect(lambda _, s=sec_key: self._copy_section(s))
            btn_row.addWidget(copy_btn)
            vb.addLayout(btn_row)
            browser = QTextBrowser()
            _install_beer_link_filter(browser, self._on_report_link_clicked)
            thr = threshold if threshold is not None else 0.5
            html = _style + self._build_ai_head_html(
                display_name, scores, graph_title, auroc, threshold=thr,
                sparkline_uri=sparkline_uri)
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
        graphs_row = -1
        for i in range(self.main_tabs.nav_list.count()):
            if "Graphs" in self.main_tabs.nav_list.item(i).text():
                graphs_row = i
                break
        if graphs_row >= 0:
            self.main_tabs.nav_list.setCurrentRow(graphs_row)   # i is a row, not a stack idx

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


