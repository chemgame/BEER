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
    QColorDialog,
)
from PySide6.QtGui import QFont, QKeySequence, QAction, QShortcut, QImage, QIcon, QPixmap
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
from beer.gui.dialogs import MutationDialog, _FigureComposerDialog, FormatChooserDialog
from beer.io.structure_formats import pdb_to_mmcif, pdb_to_gro, pdb_to_xyz
from beer.io.graph_data_export import get_graph_data
from beer.constants import (
    NAMED_COLORS, NAMED_COLORMAPS, GRAPH_TITLES, GRAPH_CATEGORIES,
    REPORT_SECTIONS, VALID_AMINO_ACIDS, _AA_COLOURS,
    KYTE_DOOLITTLE, DEFAULT_PKA, DISORDER_PROPENSITY,
    CHOU_FASMAN_HELIX, CHOU_FASMAN_SHEET, LINEAR_MOTIFS,
    STICKER_AROMATIC, STICKER_ELECTROSTATIC,
    HYDROPHOBICITY_SCALES,
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
    MobiDBWorker, UniProtVariantsWorker, IntActWorker,
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
    create_linear_sequence_map_figure,
    create_domain_architecture_figure, create_cation_pi_map_figure,
    create_local_complexity_figure, create_ramachandran_figure,
    create_contact_network_figure, create_plddt_figure, create_distance_map_figure,
    create_msa_conservation_figure, create_complex_mw_figure,
    create_truncation_series_figure,
    create_saturation_mutagenesis_figure, create_uversky_phase_plot,
    create_annotation_track_figure, create_cleavage_map_figure,
    create_plaac_profile_figure,
    create_msa_covariance_figure,
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
    "Disorder":                  "ESM2 linear-probe disorder prediction (AUC 0.83 on DisProt 2024); classical propensity fallback.",
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
    ("IDP & Phase Separation", ["Low Complexity", "Disorder", "Repeat Motifs", "Sticker & Spacer",
                                 "Charge Decoration (SCD)", "LARKS", "RNA Binding"]),
    ("Post-Translational", ["PTM Sites", "Signal Peptide & GPI"]),
    ("Structure & Topology", ["Amphipathic Helices", "TM Helices"]),
    ("Functional Sites", ["Linear Motifs", "Tandem Repeats", "Proteolytic Map",
                           "\u03b2-Aggregation & Solubility"]),
]

# ---------------------------------------------------------------------------

def _make_hsep() -> "QFrame":
    sep = QFrame()
    sep.setFrameShape(QFrame.Shape.HLine)
    sep.setFrameShadow(QFrame.Shadow.Sunken)
    return sep

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
        self.use_esm2_aggregation   = _cfg.get("use_esm2_aggregation", False)
        self._esm2_missing_warned   = False   # show ESM2 missing notice at most once
        self._history             = []   # session-only: never restored from disk
        self.hydro_scale          = "Kyte-Doolittle"
        self.sequence_name       = ""
        self._tooltips: dict     = {}
        self._analysis_worker    = None
        self._progress_dlg       = None
        self._pending_pdb        = None   # stored when loadPDB is called before page ready

        # --- New state for AlphaFold / Pfam / BLAST ---
        self.current_accession   = ""   # last successfully fetched UniProt accession
        self.alphafold_data      = None # dict: pdb_str, plddt, dist_matrix, accession
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
        "Hydrophobicity Profile": (
            "Sliding-window average of residue hydrophobicity.\n\n"
            "Formula: H(i) = (1/w) \u00b7 \u03a3 h(j)  for j = i\u2013\u230aw/2\u230b to i+\u230aw/2\u230b\n"
            "where h(j) is the per-residue score and w is the window size (default 9).\n\n"
            "Kyte & Doolittle scale (J. Mol. Biol. 157:105, 1982): range \u22124.5 (Arg) to +4.5 (Ile). "
            "Values > 1.8 sustained over \u2265 20 residues suggest a transmembrane helix.\n\n"
            "Other scales available in Settings: Wimley\u2013White, Hessa, GES, Hopp\u2013Woods, "
            "Fauch\u00e8re\u2013Pliska, Urry, Moon\u2013Fleming."
        ),
        "Local Charge Profile": (
            "Sliding-window mean net charge per residue (NCPR).\n\n"
            "NCPR = (f\u207a \u2212 f\u207b) averaged over a window, "
            "where f\u207a = fraction positive (K, R) and f\u207b = fraction negative (D, E).\n\n"
            "Positive values (blue) = net positive; negative values (red) = net negative. "
            "Useful for identifying charged patches, NLS signals, and polyampholyte regions."
        ),
        "Local Complexity": (
            "Per-window Shannon sequence entropy (bits).\n\n"
            "Formula: H = \u2212\u03a3\u1d62 p\u1d62 \u00b7 log\u2082(p\u1d62)\n"
            "Maximum H = log\u2082(20) \u2248 4.32 bits (all residues equally frequent).\n\n"
            "Low-complexity regions (H < 2 bits) often indicate repetitive or disordered segments.\n"
            "Reference: Wootton & Federhen, Comput. Chem. 17:149, 1993."
        ),
        "Disorder Profile": (
            "Per-residue intrinsic disorder score (0 = ordered, 1 = disordered).\n\n"
            "Computed with metapredict v3 (Emenecker et al., eLife 2022), a deep-learning predictor "
            "trained on DisProt/PED experimental data. Threshold: 0.5 (dashed line).\n\n"
            "Regions consistently above 0.5 are predicted intrinsically disordered (IDRs). "
            "The score is a continuous probability, not a binary state."
        ),
        "Coiled-Coil Profile": (
            "Per-residue coiled-coil propensity based on heptad repeat scoring.\n\n"
            "Coiled coils follow an (a-b-c-d-e-f-g)\u2099 heptad; positions a and d are typically hydrophobic. "
            "Score = heptad-weighted sum of Chou\u2013Fasman \u03b1-helix propensities.\n\n"
            "For validated predictions use COILS (Lupas et al., Science 252:1162, 1991) or DeepCoil."
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
            "\u0394G \u2248 \u22121 to \u22123 kcal/mol. Enriched in phase-separating IDPs and RNA-binding proteins.\n"
            "Reference: Gallivan & Dougherty, PNAS 96:9459, 1999."
        ),
        "Bead Model (Hydrophobicity)": (
            "Bead-and-stick sequence representation coloured by Kyte\u2013Doolittle hydrophobicity.\n\n"
            "Blue = hydrophilic, warm = hydrophobic (colourmap selectable in Settings). "
            "Useful for visually identifying hydrophobic patches and amphipathic stretches."
        ),
        "Bead Model (Charge)": (
            "Bead-and-stick model coloured by residue charge state.\n\n"
            "Blue = positive (K, R); red = negative (D, E); grey = uncharged. Charge at neutral pH.\n\n"
            "Useful for visualising charge clusters, salt-bridge potential, and "
            "electrostatic patterning relevant to condensate behaviour."
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


    def _replace_graph(self, title: str, fig):
        """Swap graph canvas in the named tab."""
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
        if not getattr(fig, "get_constrained_layout", lambda: False)():
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
        _PROFILE_GRAPHS = {
            "Hydrophobicity Profile", "Disorder Profile", "Local Charge Profile",
            "Local Complexity", "SCD Profile", "RNA-Binding Profile",
            "Coiled-Coil Profile", "β-Aggregation Profile", "Solubility Profile",
            "pLDDT Profile", "Hydrophobic Moment",
        }
        if title in _PROFILE_GRAPHS and len(canvas.figure.axes) == 1:
            try:
                from matplotlib.widgets import Cursor as _MplCursor
                _MplCursor(canvas.figure.axes[0], useblit=True,
                           color="#4361ee", linewidth=0.7, linestyle="--", alpha=0.6)
            except Exception:
                pass
        hint = self._GRAPH_HINTS.get(title, "")
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
        _btn_bar = QWidget()
        _btn_row = QHBoxLayout(_btn_bar)
        _btn_row.setContentsMargins(0, 2, 0, 2)
        _btn_row.addStretch()
        btn = QPushButton("Save Graph")
        btn.clicked.connect(lambda _, t=title: self.save_graph(t))
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
        self.export_structure_btn.setEnabled(False)
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

        # ---- toolbar row 1: core actions ----
        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)
        self.import_fasta_btn = QPushButton("Import FASTA")
        self.import_fasta_btn.setToolTip("Load a .fasta / .fa file (single or multi-sequence)")
        self.import_fasta_btn.clicked.connect(self.import_fasta)
        self.import_pdb_btn = QPushButton("Import PDB")
        self.import_pdb_btn.setToolTip("Load sequences from a local PDB file")
        self.import_pdb_btn.clicked.connect(self.import_pdb)
        self.analyze_btn = QPushButton("Analyze  [Ctrl+↵]")
        self.analyze_btn.setToolTip("Run full biophysical analysis on the current sequence")
        self.analyze_btn.clicked.connect(self.on_analyze)
        self.export_analysis_btn = QPushButton("Export Analysis")
        self.export_analysis_btn.setToolTip(
            "Export analysis results — choose CSV, JSON, PDF, or DAT (run analysis first)")
        self.export_analysis_btn.setEnabled(False)
        self.export_analysis_btn.clicked.connect(self.export_analysis_dialog)
        self.mutate_btn = QPushButton("Mutate…")
        self.mutate_btn.setToolTip("Introduce a point mutation at any position (run analysis first)")
        self.mutate_btn.setEnabled(False)
        self.mutate_btn.clicked.connect(self.open_mutation_dialog)
        for w in (self.import_fasta_btn, self.import_pdb_btn, self.analyze_btn,
                  self.export_analysis_btn, self.mutate_btn):
            w.setMinimumHeight(32)
            toolbar.addWidget(w)
        toolbar.addStretch()
        outer.addLayout(toolbar)

        # ---- toolbar row 2: fetch box (left) + session/tools (right) ----
        tb1b = QHBoxLayout()
        tb1b.setSpacing(6)
        tb1b.addWidget(QLabel("Fetch:"))
        self.accession_input = QLineEdit()
        self.accession_input.setPlaceholderText("UniProt ID or PDB ID (e.g. P04637, 1ABC)")
        tb1b.addWidget(self.accession_input, 1)   # stretch=1 so input takes all spare space
        fetch_btn = QPushButton("Fetch")
        fetch_btn.setMinimumHeight(30)
        fetch_btn.clicked.connect(self.fetch_accession)
        tb1b.addWidget(fetch_btn)
        tb1b.addSpacing(20)
        self.session_save_btn = QPushButton("Save Session")
        self.session_save_btn.setToolTip("Save the current sequence and settings to a .beer file")
        self.session_save_btn.clicked.connect(self.session_save)
        self.session_load_btn = QPushButton("Load Session")
        self.session_load_btn.setToolTip("Restore a previously saved .beer session file")
        self.session_load_btn.clicked.connect(self.session_load)
        self.figure_composer_btn = QPushButton("Figure Composer")
        self.figure_composer_btn.setToolTip(
            "Compose a multi-panel publication figure from any combination of graphs.")
        self.figure_composer_btn.clicked.connect(self.open_figure_composer)
        for w in (self.session_save_btn, self.session_load_btn, self.figure_composer_btn):
            w.setMinimumHeight(30)
            tb1b.addWidget(w)
        outer.addLayout(tb1b)

        # ---- toolbar row 3: annotation chip buttons grouped by category + history ----
        tb2 = QHBoxLayout()
        tb2.setSpacing(4)


        def _sep():
            f = QFrame()
            f.setFrameShape(QFrame.Shape.VLine)
            f.setFrameShadow(QFrame.Shadow.Plain)
            f.setObjectName("v_sep")
            f.setMaximumHeight(20)
            return f

        # — Structure —
        grp_lbl = QLabel("Structure")
        grp_lbl.setObjectName("group_lbl")
        tb2.addWidget(grp_lbl)
        self.fetch_af_btn = QPushButton("AlphaFold")
        self.fetch_af_btn.setObjectName("chip_btn")
        self.fetch_af_btn.setProperty("chip_state", "normal")
        self.fetch_af_btn.setEnabled(False)
        self.fetch_af_btn.setToolTip("Fetch AlphaFold predicted structure (requires UniProt accession)")
        self.fetch_af_btn.clicked.connect(self.fetch_alphafold)
        tb2.addWidget(self.fetch_af_btn)
        self.fetch_pfam_btn = QPushButton("Pfam")
        self.fetch_pfam_btn.setObjectName("chip_btn")
        self.fetch_pfam_btn.setProperty("chip_state", "normal")
        self.fetch_pfam_btn.setEnabled(False)
        self.fetch_pfam_btn.setToolTip("Fetch Pfam domain annotations from InterPro")
        self.fetch_pfam_btn.clicked.connect(self.fetch_pfam)
        tb2.addWidget(self.fetch_pfam_btn)
        self.fetch_deeptmhmm_btn = QPushButton("DeepTMHMM")
        self.fetch_deeptmhmm_btn.setObjectName("chip_btn")
        self.fetch_deeptmhmm_btn.setProperty("chip_state", "normal")
        self.fetch_deeptmhmm_btn.setEnabled(False)
        self.fetch_deeptmhmm_btn.setToolTip(
            "Run DeepTMHMM transmembrane topology prediction (requires internet + pybiolib)")
        self.fetch_deeptmhmm_btn.clicked.connect(self._run_deeptmlhmm)
        tb2.addWidget(self.fetch_deeptmhmm_btn)
        self.fetch_signalp6_btn = QPushButton("SignalP 6")
        self.fetch_signalp6_btn.setObjectName("chip_btn")
        self.fetch_signalp6_btn.setProperty("chip_state", "normal")
        self.fetch_signalp6_btn.setEnabled(False)
        self.fetch_signalp6_btn.setToolTip(
            "Run SignalP 6.0 signal peptide prediction via BioLib (requires internet + pybiolib)")
        self.fetch_signalp6_btn.clicked.connect(self._run_signalp6)
        tb2.addWidget(self.fetch_signalp6_btn)

        tb2.addSpacing(4)
        tb2.addWidget(_sep())
        tb2.addSpacing(4)

        # — Disorder / IDP —
        grp_lbl2 = QLabel("Disorder / IDP")
        grp_lbl2.setObjectName("group_lbl")
        tb2.addWidget(grp_lbl2)
        self.fetch_elm_btn = QPushButton("ELM")
        self.fetch_elm_btn.setObjectName("chip_btn")
        self.fetch_elm_btn.setProperty("chip_state", "normal")
        self.fetch_elm_btn.setEnabled(False)
        self.fetch_elm_btn.setToolTip("Fetch experimentally validated linear motifs from ELM (UniProt only)")
        self.fetch_elm_btn.clicked.connect(self.fetch_elm)
        tb2.addWidget(self.fetch_elm_btn)
        self.fetch_disprot_btn = QPushButton("DisProt")
        self.fetch_disprot_btn.setObjectName("chip_btn")
        self.fetch_disprot_btn.setProperty("chip_state", "normal")
        self.fetch_disprot_btn.setEnabled(False)
        self.fetch_disprot_btn.setToolTip("Fetch disorder annotations from DisProt (UniProt only)")
        self.fetch_disprot_btn.clicked.connect(self.fetch_disprot)
        tb2.addWidget(self.fetch_disprot_btn)
        self.fetch_mobidb_btn = QPushButton("MobiDB")
        self.fetch_mobidb_btn.setObjectName("chip_btn")
        self.fetch_mobidb_btn.setProperty("chip_state", "normal")
        self.fetch_mobidb_btn.setEnabled(False)
        self.fetch_mobidb_btn.setToolTip("Fetch consensus disorder annotations from MobiDB (UniProt only)")
        self.fetch_mobidb_btn.clicked.connect(self.fetch_mobidb)
        tb2.addWidget(self.fetch_mobidb_btn)
        self.fetch_phasepdb_btn = QPushButton("PhaSepDB")
        self.fetch_phasepdb_btn.setObjectName("chip_btn")
        self.fetch_phasepdb_btn.setProperty("chip_state", "normal")
        self.fetch_phasepdb_btn.setEnabled(False)
        self.fetch_phasepdb_btn.setToolTip("Check phase-separation database PhaSepDB (UniProt only)")
        self.fetch_phasepdb_btn.clicked.connect(self.fetch_phasepdb)
        tb2.addWidget(self.fetch_phasepdb_btn)

        tb2.addSpacing(4)
        tb2.addWidget(_sep())
        tb2.addSpacing(4)

        # — Variants & Interactions —
        grp_lbl3 = QLabel("Variants & Interactions")
        grp_lbl3.setObjectName("group_lbl")
        tb2.addWidget(grp_lbl3)
        self.fetch_variants_btn = QPushButton("Variants")
        self.fetch_variants_btn.setObjectName("chip_btn")
        self.fetch_variants_btn.setProperty("chip_state", "normal")
        self.fetch_variants_btn.setEnabled(False)
        self.fetch_variants_btn.setToolTip("Fetch natural variants and mutagenesis data from UniProt")
        self.fetch_variants_btn.clicked.connect(self.fetch_variants)
        tb2.addWidget(self.fetch_variants_btn)
        self.fetch_alphafold_missense_btn = QPushButton("AlphaMissense")
        self.fetch_alphafold_missense_btn.setObjectName("chip_btn")
        self.fetch_alphafold_missense_btn.setProperty("chip_state", "normal")
        self.fetch_alphafold_missense_btn.setEnabled(False)
        self.fetch_alphafold_missense_btn.setToolTip(
            "Fetch AlphaMissense variant pathogenicity scores from EBI (UniProt only, requires internet)")
        self.fetch_alphafold_missense_btn.clicked.connect(
            lambda: self._run_alphafold_missense(self.current_accession))
        tb2.addWidget(self.fetch_alphafold_missense_btn)
        self.fetch_intact_btn = QPushButton("IntAct")
        self.fetch_intact_btn.setObjectName("chip_btn")
        self.fetch_intact_btn.setProperty("chip_state", "normal")
        self.fetch_intact_btn.setEnabled(False)
        self.fetch_intact_btn.setToolTip("Fetch curated binary interactions from IntAct / EBI (UniProt only)")
        self.fetch_intact_btn.clicked.connect(self.fetch_intact)
        tb2.addWidget(self.fetch_intact_btn)

        # Convenience list for bulk enable/disable
        self._db_fetch_btns = [
            self.fetch_af_btn, self.fetch_pfam_btn, self.fetch_elm_btn,
            self.fetch_disprot_btn, self.fetch_mobidb_btn, self.fetch_phasepdb_btn,
            self.fetch_variants_btn, self.fetch_intact_btn,
            self.fetch_alphafold_missense_btn, self.fetch_deeptmhmm_btn,
        ]

        tb2.addStretch()
        tb2.addWidget(QLabel("History:"))
        self.history_combo = QComboBox()
        self.history_combo.setMinimumWidth(200)
        self.history_combo.addItem("— recent sequences —")
        self.history_combo.currentIndexChanged.connect(self._on_history_selected)
        tb2.addWidget(self.history_combo)
        outer.addLayout(tb2)

        # ── Persistent sequence info bar ─────────────────────────────────────
        self._seq_info_label = QLabel("")
        self._seq_info_label.setObjectName("seq_info_lbl")
        self._seq_info_label.hide()
        outer.addWidget(self._seq_info_label)

        # ---- splitter: left input panel | right results panel ----
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(4)

        # Left panel: sequence input + chain selector + sequence viewer
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 4, 0)
        left_layout.setSpacing(5)

        self._seq_label = QLabel("Protein Sequence:")
        self._seq_label.setObjectName("accent_lbl")
        left_layout.addWidget(self._seq_label)

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
        self._seq_view_label = QLabel("Sequence Viewer:")
        self._seq_view_label.setObjectName("accent_lbl")
        sv_hdr.addWidget(self._seq_view_label)
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
        self.motif_match_lbl = QLabel("")
        self.motif_match_lbl.setObjectName("accent_lbl")
        sv_hdr.addWidget(self.motif_match_lbl)
        left_layout.addLayout(sv_hdr)
        self.seq_viewer = QTextBrowser()
        self.seq_viewer.setFont(QFont("Courier New", 10))
        left_layout.addWidget(self.seq_viewer, 1)

        # ── Sequence action row (copy / clear) ────────────────────────────
        seq_action_row = QHBoxLayout()
        seq_action_row.setSpacing(6)
        copy_seq_btn = QPushButton("Copy Sequence")
        copy_seq_btn.setToolTip("Copy the full sequence or a selected range to clipboard")
        copy_seq_btn.setMinimumHeight(28)
        copy_seq_btn.clicked.connect(self._copy_sequence_menu)
        seq_action_row.addWidget(copy_seq_btn)
        clear_protein_btn = QPushButton("Clear All")
        clear_protein_btn.setToolTip("Clear the loaded protein, analysis, graphs and structure")
        clear_protein_btn.setMinimumHeight(28)
        clear_protein_btn.setObjectName("delete_btn")
        clear_protein_btn.clicked.connect(self._clear_all)
        seq_action_row.addWidget(clear_protein_btn)
        seq_action_row.addStretch()
        left_layout.addLayout(seq_action_row)

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

        # ── Protein info bar (shown after accession fetch) ────────────────
        self._protein_info_bar = QTextBrowser()
        self._protein_info_bar.setOpenExternalLinks(True)
        self._protein_info_bar.setMaximumHeight(72)
        self._protein_info_bar.setObjectName("info_bar")
        self._protein_info_bar.hide()
        right_layout.addWidget(self._protein_info_bar)

        right_layout.addWidget(report_panel, 1)

        self.report_section_tabs = {}
        self._report_sec_to_idx: dict = {}
        _stack_idx = 0
        bold_font = QFont(); bold_font.setBold(True)

        # Build a set of all sections in groups
        _grouped_secs = {s for _, secs in _REPORT_SECTION_GROUPS for s in secs}

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
                # Build the tab widget
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
                self._report_sec_to_idx[sec] = _stack_idx
                _stack_idx += 1
            grp_item.setExpanded(True)

        # Any sections not in groups
        for sec in REPORT_SECTIONS:
            if sec not in _grouped_secs:
                leaf = QTreeWidgetItem([sec])
                leaf.setData(0, Qt.ItemDataRole.UserRole, sec)
                self.report_section_list.addTopLevelItem(leaf)
                tab = QWidget(); vb = QVBoxLayout(tab); vb.setContentsMargins(4, 4, 4, 4)
                btn_row = QHBoxLayout(); btn_row.setSpacing(4)
                btn_row.addStretch()
                copy_btn = QPushButton("Copy Table"); copy_btn.setMaximumWidth(100)
                copy_btn.setMinimumHeight(26)
                copy_btn.clicked.connect(lambda _, s=sec: self._copy_section(s))
                btn_row.addWidget(copy_btn); vb.addLayout(btn_row)
                browser = QTextBrowser(); vb.addWidget(browser)
                self.report_stack.addWidget(tab)
                self.report_section_tabs[sec] = browser
                self._report_sec_to_idx[sec] = _stack_idx
                _stack_idx += 1

        self.report_section_list.itemClicked.connect(self._on_report_section_clicked)
        self.report_section_list.setCurrentItem(
            self.report_section_list.topLevelItem(0).child(0)
            if self.report_section_list.topLevelItem(0) else None)

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

        # Top bar: Save All button
        top_bar = QHBoxLayout()
        top_bar.addStretch()
        save_all = QPushButton("Save All Graphs")
        save_all.setMaximumWidth(160)
        save_all.clicked.connect(self.save_all_graphs)
        top_bar.addWidget(save_all)
        right_v.addLayout(top_bar)

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
        "pLDDT / B-factor":    ["Red-White-Blue", "Blue-White-Red", "Rainbow", "Sinebow"],
        "Residue Type":         ["Amino Acid (UniProt)", "Shapely"],
        "Chain":                ["Chain Colors"],
        "Charge":               ["Blue / Red / Grey"],
        "Hydrophobicity":       ["Cyan-White-Orange", "Blue-White-Red", "Green-White-Red"],
        "Mass":                 ["Blue-to-Red", "Rainbow"],
        "Secondary Structure":  ["JMol", "PyMOL"],
        "Spectrum (N→C)":       ["Spectrum"],
    }
    _STRUCT_MODE_KEY = {
        "pLDDT / B-factor":    "plddt",
        "Residue Type":         "residue",
        "Chain":                "chain",
        "Charge":               "charge",
        "Hydrophobicity":       "hydrophobicity",
        "Mass":                 "mass",
        "Secondary Structure":  "secondary_structure",
        "Spectrum (N→C)":       "spectrum",
    }
    _STRUCT_PANEL_CSS_LIGHT = """
        QScrollArea { border: 1px solid #d1d9f0; border-radius: 8px; background: #f4f6fd; }
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
    """
    _STRUCT_PANEL_CSS_DARK = """
        QScrollArea { border: 1px solid #1a3a5c; border-radius: 8px; background: #0f3460; }
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

            # ── left control panel ────────────────────────────────────────────
            self.struct_ctrl_scroll = QScrollArea()
            self.struct_ctrl_scroll.setWidgetResizable(True)
            self.struct_ctrl_scroll.setFixedWidth(226)
            self.struct_ctrl_scroll.setStyleSheet(self._STRUCT_PANEL_CSS_LIGHT)
            ctrl_scroll = self.struct_ctrl_scroll
            ctrl_inner = QWidget()
            ctrl_inner.setObjectName("structCtrl")
            ctrl_layout = QVBoxLayout(ctrl_inner)
            ctrl_layout.setContentsMargins(6, 6, 6, 8)
            ctrl_layout.setSpacing(6)
            ctrl_scroll.setWidget(ctrl_inner)

            # ── panel title ───────────────────────────────────────────────────
            self._struct_title_lbl = QLabel("Visualization Controls")
            self._struct_title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._struct_title_lbl.setStyleSheet(
                "font-weight:700; font-size:10pt; color:#3b4fc8;"
                " padding:4px 0; background:transparent;"
            )
            ctrl_layout.addWidget(self._struct_title_lbl)

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

            ctrl_layout.addWidget(rep_grp)

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
            ctrl_layout.addWidget(color_grp)

            # ── Legend ────────────────────────────────────────────────────────
            legend_grp = QGroupBox("Legend")
            legend_gl = QVBoxLayout(legend_grp)
            legend_gl.setContentsMargins(8, 4, 8, 6)
            self.struct_colorbar_cb = QCheckBox("Show color bar")
            self.struct_colorbar_cb.setChecked(True)
            self.struct_colorbar_cb.toggled.connect(self._on_struct_colorbar_toggled)
            legend_gl.addWidget(self.struct_colorbar_cb)

            ctrl_layout.addWidget(legend_grp)

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
            ctrl_layout.addWidget(bg_grp)

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
            ctrl_layout.addWidget(motion_grp)

            # ── Export / Snapshot ─────────────────────────────────────────────
            snap_grp = QGroupBox("Export")
            snap_gl = QVBoxLayout(snap_grp)
            snap_gl.setContentsMargins(6, 4, 6, 6)
            reset_btn = QPushButton("Reset View")
            reset_btn.setToolTip("Reset representation, colour, background and camera to defaults")
            reset_btn.clicked.connect(self._reset_struct_view)
            snap_gl.addWidget(reset_btn)
            snapshot_btn = QPushButton("Snapshot PNG")
            snapshot_btn.setToolTip("Render the current view to a PNG file")
            snapshot_btn.clicked.connect(self._take_structure_snapshot)
            snap_gl.addWidget(snapshot_btn)
            ctrl_layout.addWidget(snap_grp)

            # ── Chain visibility ──────────────────────────────────────────
            self._chains_grp = QGroupBox("Chains")
            chains_gl = QVBoxLayout(self._chains_grp)
            chains_gl.setContentsMargins(6, 4, 6, 6)
            chains_gl.setSpacing(4)
            chain_btn_row = QHBoxLayout()
            chain_all_btn = QPushButton("All")
            chain_all_btn.setFixedHeight(22)
            chain_all_btn.clicked.connect(self._show_all_chains)
            chain_none_btn = QPushButton("None")
            chain_none_btn.setFixedHeight(22)
            chain_none_btn.clicked.connect(self._hide_all_chains)
            chain_btn_row.addWidget(chain_all_btn)
            chain_btn_row.addWidget(chain_none_btn)
            chains_gl.addLayout(chain_btn_row)
            self._chain_cbs_widget = QWidget()
            self._chain_cbs_layout = QVBoxLayout(self._chain_cbs_widget)
            self._chain_cbs_layout.setContentsMargins(0, 0, 0, 0)
            self._chain_cbs_layout.setSpacing(2)
            chains_gl.addWidget(self._chain_cbs_widget)
            self._chains_grp.setVisible(False)
            ctrl_layout.addWidget(self._chains_grp)
            self._chain_checkboxes: dict = {}

            ctrl_layout.addStretch()
            content_row.addWidget(ctrl_scroll)

            # ── 3-D viewer ────────────────────────────────────────────────────
            self.structure_viewer = QWebEngineView()
            self.structure_viewer.setMinimumHeight(500)
            # When the base page finishes loading, deliver any queued PDB
            self.structure_viewer.loadFinished.connect(self._on_structure_page_loaded)
            content_row.addWidget(self.structure_viewer, 1)

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
</div>
<script src="https://3dmol.org/build/3Dmol-min.js"></script>
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

// ── charge colorfunc ───────────────────────────────────────────────────────
function _charge(atom){{
    if(['ARG','LYS','HIS'].indexOf(atom.resn)>=0) return '#5588ff';
    if(['ASP','GLU'].indexOf(atom.resn)>=0)        return '#ff5555';
    return '#aaaaaa';
}}

// ── secondary structure colorfunc (JMol palette) ──────────────────────────
function _ss_jmol(atom){{
    var s=atom.ss;
    if(s==='h'||s==='H') return '#FF0080';   // helix  — hot pink
    if(s==='s'||s==='S') return '#FFFF00';   // sheet  — yellow
    return '#FFFFFF';                         // coil   — white
}}
// ── secondary structure colorfunc (PyMOL palette) ─────────────────────────
function _ss_pymol(atom){{
    var s=atom.ss;
    if(s==='h'||s==='H') return '#FF6666';   // helix  — salmon
    if(s==='s'||s==='S') return '#6699FF';   // sheet  — cornflower blue
    return '#CCCCCC';                         // coil   — grey
}}

function _getColorFunc(){{
    if(colorMode==='hydrophobicity'){{
        if(colorScheme==='Blue-White-Red')  return _hydro_BWR;
        if(colorScheme==='Green-White-Red') return _hydro_GWR;
        return _hydro_CWO;
    }}
    if(colorMode==='mass'){{
        if(colorScheme==='Rainbow') return _mass_RB;
        return _mass_BR;
    }}
    if(colorMode==='charge') return _charge;
    if(colorMode==='secondary_structure'){{
        return colorScheme==='PyMOL' ? _ss_pymol : _ss_jmol;
    }}
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
        o[rep]={{colorscheme:_plddtScheme(),opacity:op}};
    }} else if(colorMode==='residue'){{
        o[rep]={{colorscheme:colorScheme==='Shapely'?'shapely':'amino',opacity:op}};
    }} else if(colorMode==='chain'){{
        o[rep]={{colorscheme:'chain',opacity:op}};
    }} else if(colorMode==='spectrum'){{
        o[rep]={{colorscheme:'spectrum',opacity:op}};
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
        if(colorMode==='plddt')        sOpts.colorscheme=_plddtScheme();
        else if(colorMode==='residue') sOpts.colorscheme=colorScheme==='Shapely'?'shapely':'amino';
        else if(colorMode==='chain')   sOpts.colorscheme='chain';
        else if(colorMode==='spectrum') sOpts.colorscheme='spectrum';
        else                           sOpts.colorfunc=_getColorFunc();
        // addSurface is async — store the ID only when the Promise resolves.
        viewer.addSurface($3Dmol.SurfaceType.MS,sOpts).then(function(id){{
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
    'Spectrum':{{css:'linear-gradient(to top,#0000ff,#00ffff,#00ff00,#ffff00,#ff0000)',min:'N-term',mid:'middle',max:'C-term',unit:'Sequence position'}},
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
        title.textContent=colorMode==='plddt'?'pLDDT':colorMode==='hydrophobicity'?'Hydrophobicity':'Mass';
        grad.style.background=cfg.css;
        tmax.textContent=cfg.max; tmid.textContent=cfg.mid; tmin.textContent=cfg.min;
        unit.textContent=cfg.unit;
    }}
}}

// ── public API ─────────────────────────────────────────────────────────────
function setRepresentation(r)  {{ repMode=r; applyStyle(); }}
function setColorMode(m,s)     {{ colorMode=m; if(s) colorScheme=s; applyStyle(); }}
function setScheme(s)          {{ colorScheme=s; applyStyle(); }}
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

// ── loadPDB: swap in a new structure without reloading the page ────────────
function loadPDB(data){{
    pdbData = data || null;   // always store first — init() picks this up if viewer not ready yet
    hiddenChains = {{}};       // reset chain visibility on every new structure
    if(!viewer) return;        // CDN still loading; init() will load pdbData when ready
    viewer.clear();
    if(pdbData){{
        viewer.addModel(pdbData, "pdb");
        viewer.zoomTo();   // set camera BEFORE rendering
        applyStyle();      // renders with correct camera
    }} else {{
        viewer.render();
        updateColorBar();
    }}
}}

function init(){{
    viewer=$3Dmol.createViewer("vp",{{backgroundColor:"#ffffff",antialias:true}});
    if(pdbData){{          // null on the initial empty-page load
        viewer.addModel(pdbData,"pdb");
        viewer.zoomTo();   // set camera BEFORE rendering
        applyStyle();      // renders with correct camera
    }}
    viewer.render();
    updateColorBar();
}}
window.addEventListener("load",init);
</script>
</body></html>"""

    def _js(self, code: str) -> None:
        """Run JavaScript in the structure viewer (no-op if unavailable)."""
        if self.structure_viewer is not None:
            self.structure_viewer.page().runJavaScript(code)

    def _update_scheme_combo(self, mode: str) -> None:
        schemes = self._STRUCT_SCHEMES.get(mode, ["Default"])
        self.struct_scheme_combo.blockSignals(True)
        self.struct_scheme_combo.clear()
        self.struct_scheme_combo.addItems(schemes)
        self.struct_scheme_combo.blockSignals(False)

    def _on_struct_rep_changed(self, rep_label: str) -> None:
        self._js(f"setRepresentation('{rep_label.lower()}');")

    def _on_struct_color_mode_changed(self, mode: str) -> None:
        self._update_scheme_combo(mode)
        key = self._STRUCT_MODE_KEY.get(mode, "plddt")
        scheme = self.struct_scheme_combo.currentText()
        self._js(f"setColorMode('{key}','{scheme}');")

    def _on_struct_scheme_changed(self, scheme: str) -> None:
        if scheme:
            self._js(f"setScheme('{scheme}');")

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
        self._js("resetView();")

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

    def _load_structure_viewer(self, pdb_str: str) -> None:
        """Swap in a new structure without reloading the 3Dmol page."""
        if not _WEBENGINE_AVAILABLE or self.structure_viewer is None:
            return
        pdb_json = json.dumps(pdb_str)
        # Keep as pending so loadFinished can retry if the page is still loading.
        self._pending_pdb = pdb_json
        # 1-arg form is the only safe form in PySide6 (no 2-arg callback variant).
        self._js(f"loadPDB({pdb_json});")
        self._populate_chain_controls(pdb_str)
        # Annotate disorder regions and signal peptide in 3D viewer after a short delay
        from PySide6.QtCore import QTimer as _QT
        _QT.singleShot(800, self._annotate_structure_viewer)

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
        self.hydro_scale_combo.setCurrentText("Kyte-Doolittle")
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

        self.graph_format_combo = QComboBox()
        self.graph_format_combo.addItems(["PNG", "SVG", "PDF"])
        self._set_tooltip(self.graph_format_combo, "Default file format when saving graphs.")
        form3.addRow("Default Graph Format:", self.graph_format_combo)

        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(NAMED_COLORMAPS)
        self.colormap_combo.setCurrentText(self.colormap)
        self._set_tooltip(self.colormap_combo, "Colour map for the bead hydrophobicity model.")
        form3.addRow("Bead Colormap:", self.colormap_combo)

        self.heatmap_cmap_combo = QComboBox()
        self.heatmap_cmap_combo.addItems(NAMED_COLORMAPS)
        self.heatmap_cmap_combo.setCurrentText(self.heatmap_cmap)
        self._set_tooltip(self.heatmap_cmap_combo,
            "Colour map for heatmaps: Distance Map, Cation\u2013\u03c0 Map, "
            "MSA Covariance, Single-Residue Perturbation Map, Residue Contact Network.")
        form3.addRow("Heatmap Colormap:", self.heatmap_cmap_combo)

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
            "ESM2 model for per-residue embeddings (disorder, variant effect). "
            "Requires 'pip install fair-esm torch'. Larger models are more accurate but slower."
        )
        form5.addRow("ESM2 model:", self.esm2_combo)

        self.esm2_aggr_checkbox = QCheckBox(
            "Use ESM2 logistic probe for \u03b2-aggregation (requires ESM2)"
        )
        self.esm2_aggr_checkbox.setChecked(self.use_esm2_aggregation)
        self._set_tooltip(
            self.esm2_aggr_checkbox,
            "When enabled, the \u03b2-Aggregation Profile uses the ESM2 logistic probe "
            "instead of ZYGGREGATOR.\n"
            "Default (unchecked): ZYGGREGATOR (Tartaglia & Vendruscolo 2008) — "
            "peer-reviewed, no ML install required.\n"
            "ESM2 option: in-house logistic probe; requires ESM2 to be installed."
        )
        form5.addRow("", self.esm2_aggr_checkbox)
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
  <tr><td>Ctrl+2</td><td>Switch to Structure tab</td></tr>
  <tr><td>Ctrl+3</td><td>Switch to BLAST tab</td></tr>
  <tr><td>Ctrl+7</td><td>Switch to MSA tab</td></tr>
  <tr><td>Ctrl+Z</td><td>Undo last mutation</td></tr>
  <tr><td>Ctrl+E</td><td>Export PDF report</td></tr>
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
  <li><b>Coiled-Coil Profile</b> — heptad-weighted propensity profile; score is normalised relative to the sequence maximum. Not a validated coiled-coil predictor — interpret as a relative propensity indicator only.</li>
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

        methods_btn = QPushButton("Generate Methods Paragraph")
        methods_btn.setMinimumHeight(32)
        methods_btn.setToolTip("Auto-generate a methods paragraph for your paper")
        methods_btn.clicked.connect(self._generate_methods)
        cite_bar.addWidget(methods_btn)

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

        self.analyze_btn.setEnabled(False)
        self.statusBar.showMessage("Analyzing…")

        _model_tag = ""
        if self._embedder is not None:
            _mn = getattr(self._embedder, "model_name", "")
            _parts = _mn.split("_")
            try:
                _model_tag = f"  ·  ESM2 {next(p for p in _parts if p.endswith('M') or p.endswith('B'))}"
            except StopIteration:
                _model_tag = "  ·  ESM2"
        self._progress_dlg = QProgressDialog(
            f"Running analysis{_model_tag}…", "Cancel", 0, 0, self)
        self._progress_dlg.setWindowTitle("BEER Analysis")
        self._progress_dlg.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress_dlg.setMinimumDuration(500)
        self._progress_dlg.canceled.connect(self._cancel_analysis)
        self._progress_dlg.show()

        self._analysis_worker = AnalysisWorker(
            seq, pH, self.default_window_size, self.use_reducing, self.custom_pka,
            hydro_scale=self.hydro_scale,
            embedder=self._embedder,
            use_esm2_aggregation=self.use_esm2_aggregation,
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
            return fig

        gens = {}
        gens["Amino Acid Composition (Bar)"] = lambda: _wrap(lambda: create_amino_acid_composition_figure(
            ad["aa_counts"], ad["aa_freq"], label_font=lf, tick_font=tf))
        gens["Amino Acid Composition (Pie)"] = lambda: _wrap(lambda: create_amino_acid_composition_pie_figure(
            ad["aa_counts"], label_font=lf))
        gens["Hydrophobicity Profile"] = lambda: _wrap(lambda: create_hydrophobicity_figure(
            ad["hydro_profile"], ad["window_size"], hs, label_font=lf, tick_font=tf))
        gens["Bead Model (Hydrophobicity)"] = lambda: _wrap(lambda: create_bead_model_hydrophobicity_figure(
            seq, sbl, label_font=lf, tick_font=tf, cmap=cm))
        gens["Bead Model (Charge)"] = lambda: _wrap(lambda: create_bead_model_charge_figure(
            seq, sbl, label_font=lf, tick_font=tf))
        gens["Sticker Map"] = lambda: _wrap(lambda: create_sticker_map_figure(
            seq, sbl, label_font=lf, tick_font=tf))
        gens["Local Charge Profile"] = lambda: _wrap(lambda: create_local_charge_figure(
            ad["ncpr_profile"], ad["window_size"], label_font=lf, tick_font=tf))
        gens["Local Complexity"] = lambda: _wrap(lambda: create_local_complexity_figure(
            ad["entropy_profile"], ad["window_size"], label_font=lf, tick_font=tf))
        gens["Cation\u2013\u03c0 Map"] = lambda: _wrap(lambda: create_cation_pi_map_figure(
            seq, label_font=lf, tick_font=tf, cmap=hcm))
        gens["Isoelectric Focus"] = lambda: _wrap(lambda: create_isoelectric_focus_figure(
            seq, label_font=lf, tick_font=tf, pka=pk))
        gens["Helical Wheel"] = lambda: _wrap(lambda: create_helical_wheel_figure(seq, label_font=lf))
        gens["Charge Decoration"] = lambda: _wrap(lambda: create_charge_decoration_figure(
            ad["fcr"], ad["ncpr"], label_font=lf, tick_font=tf))
        gens["Linear Sequence Map"] = lambda: _wrap(lambda: create_linear_sequence_map_figure(
            seq, ad["hydro_profile"], ad["ncpr_profile"], ad["disorder_scores"],
            label_font=lf, tick_font=tf))
        gens["Disorder Profile"] = lambda: _wrap(lambda: create_disorder_profile_figure(
            ad["disorder_scores"], label_font=lf, tick_font=tf))
        gens["TM Topology"] = lambda: _wrap(lambda: create_tm_topology_figure(
            seq, ad.get("tm_helices", []), label_font=lf, tick_font=tf))
        gens["Uversky Phase Plot"] = lambda: _wrap(lambda: create_uversky_phase_plot(
            seq, label_font=lf, tick_font=tf))
        gens["Single-Residue Perturbation Map"] = lambda: _wrap(lambda: create_saturation_mutagenesis_figure(
            seq, label_font=lf, tick_font=tf, cmap=hcm))
        gens["Domain Architecture"] = lambda: _wrap(lambda: create_domain_architecture_figure(
            len(seq), self.pfam_domains, seq=seq,
            disorder_scores=ad.get("disorder_scores"),
            tm_helices=ad.get("tm_helices"),
            label_font=lf, tick_font=tf))
        # Use ESM2 aggregation profile for Annotation Track when available
        _annot_aggr = ad.get("aggr_profile_esm2") if (
            self.use_esm2_aggregation and ad.get("aggr_profile_esm2")
        ) else ad.get("aggr_profile", calc_aggregation_profile(seq))
        gens["Annotation Track"] = lambda: _wrap(lambda: create_annotation_track_figure(
            seq, ad.get("disorder_scores", []), ad.get("hydro_profile", []),
            _annot_aggr,
            ad.get("tm_helices", []),
            ad.get("larks", []), ad.get("sp_result", {}),
            label_font=lf, tick_font=tf))
        gens["Cleavage Map"] = lambda: _wrap(lambda: create_cleavage_map_figure(
            seq, ad.get("prot_sites", {}), label_font=lf, tick_font=tf))
        if ad.get("cc_profile"):
            gens["Coiled-Coil Profile"] = lambda: _wrap(lambda: create_coiled_coil_profile_figure(
                ad["cc_profile"], label_font=lf, tick_font=tf))
        if _HAS_AGGREGATION:
            # Use ESM2 profile if enabled in settings and available, else ZYGGREGATOR
            _aggr_prof = ad.get("aggr_profile_esm2") if (
                self.use_esm2_aggregation and ad.get("aggr_profile_esm2")
            ) else ad.get("aggr_profile", calc_aggregation_profile(seq))
            gens["\u03b2-Aggregation Profile"] = lambda: _wrap(lambda: create_aggregation_profile_figure(
                seq, _aggr_prof, predict_aggregation_hotspots(seq),
                label_font=lf, tick_font=tf))
            gens["Solubility Profile"] = lambda: _wrap(lambda: create_solubility_profile_figure(
                seq, calc_camsolmt_score(seq), label_font=lf, tick_font=tf))
        _am_data = getattr(self, "_alphafold_missense_data", None)
        if _am_data:
            gens["AlphaMissense"] = lambda: _wrap(lambda: create_alphafold_missense_figure(
                _am_data, seq=seq, label_font=lf, tick_font=tf))
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
        if ad.get("plaac"):
            gens["PLAAC Profile"] = lambda: _wrap(lambda: create_plaac_profile_figure(
                ad["plaac"], label_font=lf, tick_font=tf))

        # Structure-dependent
        afd = self.alphafold_data
        if afd:
            plddt = afd.get("plddt")
            if plddt and len(plddt) == len(seq):
                gens["pLDDT Profile"] = lambda: _wrap(lambda: create_plddt_figure(
                    afd["plddt"], label_font=lf, tick_font=tf))
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
                seq, lf, tf)

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

    def _gen_variant_effect_fig(self, seq: str, lf: int, tf: int):
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
        return create_variant_effect_figure(seq, llr, label_font=lf, tick_font=tf)

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
        if title and title in self._graph_title_to_stack_idx:
            self.graph_stack.setCurrentIndex(self._graph_title_to_stack_idx[title])
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

    def export_analysis_dialog(self):
        """Show format chooser then export analysis data (CSV / JSON / PDF / DAT)."""
        if not self.analysis_data:
            QMessageBox.warning(self, "Export Analysis", "Run analysis first.")
            return
        dlg = FormatChooserDialog(
            "Export Analysis",
            [
                ("CSV — comma-separated values (.csv)",  "csv",  True),
                ("JSON — structured key-value file (.json)", "json", True),
                ("PDF — formatted report (.pdf)",         "pdf",  True),
                ("DAT — tab-separated text (.dat)",       "dat",  True),
            ],
            self,
        )
        if dlg.exec() != QDialog.Accepted:
            return
        self._export_analysis_as(dlg.selected_key())

    def _export_analysis_as(self, fmt: str):
        """Dispatch analysis export to the chosen format."""
        name = self.sequence_name or "analysis"
        filters = {
            "csv":  ("CSV Files (*.csv)",  f"{name}.csv"),
            "json": ("JSON Files (*.json)", f"{name}.json"),
            "pdf":  ("PDF Files (*.pdf)",   f"{name}.pdf"),
            "dat":  ("DAT Files (*.dat)",   f"{name}.dat"),
        }
        flt, default = filters[fmt]
        fn, _ = QFileDialog.getSaveFileName(self, "Export Analysis", default, flt)
        if not fn:
            return

        d = self.analysis_data
        rows = [
            ("Sequence Name",                  self.sequence_name or ""),
            ("Length (aa)",                    len(d.get("seq", ""))),
            ("Molecular Weight (Da)",          f"{d.get('mol_weight', 0):.2f}"),
            ("Isoelectric Point",              f"{d.get('iso_point', 0):.2f}"),
            ("GRAVY",                          f"{d.get('gravy', 0):.3f}"),
            ("Net Charge (pH 7)",              f"{d.get('net_charge_7', 0):.2f}"),
            ("FCR",                            f"{d.get('fcr', 0):.3f}"),
            ("NCPR",                           f"{d.get('ncpr', 0):+.3f}"),
            ("Aromaticity",                    f"{d.get('aromaticity', 0):.3f}"),
            ("Extinction Coeff. (reduced)",    d.get('extinction', ('', ''))[0]
                                               if isinstance(d.get('extinction'), tuple)
                                               else d.get('extinction', '')),
            ("Extinction Coeff. (non-reduced)", d.get('extinction', ('', ''))[1]
                                                if isinstance(d.get('extinction'), tuple)
                                                else ''),
            ("Kappa (κ)",                      f"{d.get('kappa', 0):.4f}"),
            ("Omega (Ω)",                      f"{d.get('omega', 0):.4f}"),
            ("SCD",                            f"{d.get('scd', 0):.3f}"),
            ("Fraction Disorder",              f"{d.get('disorder_f', 0):.3f}"),
            ("% Aggregation-prone",            f"{d.get('solub_stats', {}).get('pct_aggregation_prone', 0):.1f}"),
            ("RNA-binding propensity",         f"{d.get('rbp', {}).get('mean_propensity', 0):.3f}"),
            ("Signal peptide h-region KD",     f"{d.get('sp_result', {}).get('h_region_score', 0):.3f}"),
        ]

        try:
            if fmt == "csv":
                with open(fn, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Metric", "Value"])
                    writer.writerows(rows)

            elif fmt == "json":
                with open(fn, "w") as f:
                    json.dump({"sequence_name": self.sequence_name,
                               "metrics": {k: v for k, v in rows}}, f, indent=2)

            elif fmt == "pdf":
                ExportTools.export_pdf(self.analysis_data, fn, self,
                                       seq_name=self.sequence_name)
                return  # ExportTools shows its own success dialog

            elif fmt == "dat":
                with open(fn, "w") as f:
                    f.write("Metric\tValue\n")
                    for k, v in rows:
                        f.write(f"{k}\t{v}\n")

            self.statusBar.showMessage(
                f"Analysis exported as {fmt.upper()} to {os.path.basename(fn)}", 4000)
        except OSError as e:
            QMessageBox.critical(self, "Export Error", str(e))

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

    def save_all_graphs(self):
        d = QFileDialog.getExistingDirectory(self, "Select Directory")
        if not d:
            return
        ext = self.default_graph_format.lower()
        try:
            # Ensure every registered graph is rendered before saving
            for title in list(self._graph_generators.keys()):
                self._render_graph(title)
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
        self.colormap         = self.colormap_combo.currentText()
        self.heatmap_cmap     = self.heatmap_cmap_combo.currentText()
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
        self.hydro_scale          = self.hydro_scale_combo.currentText()

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
        self.use_esm2_aggregation = self.esm2_aggr_checkbox.isChecked()

        # Re-initialise ESM2 embedder if model changed
        new_esm2_model = self.esm2_combo.currentText()
        current_model = getattr(self._embedder, "model_name", None)
        if new_esm2_model != current_model:
            try:
                from beer.embeddings import get_embedder
                self._embedder = get_embedder(new_esm2_model)
                self._update_esm2_indicator("ready")
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
            "esm2_model":           self.esm2_combo.currentText(),
            "use_esm2_aggregation": self.use_esm2_aggregation,
        })
        self.statusBar.showMessage("Settings applied and saved.", 5000)

    def reset_defaults(self):
        self.window_size_input.setText("9")
        self.hydro_scale_combo.setCurrentText("Kyte-Doolittle")
        self.ph_input.setText("7.0")
        self.pka_input.setText("")
        self.reducing_checkbox.setChecked(False)
        self.label_checkbox.setChecked(True)
        self.colormap_combo.setCurrentText("coolwarm")
        self.heatmap_cmap_combo.setCurrentText("viridis")
        self.label_font_input.setText("11")
        self.tick_font_input.setText("9")
        self.marker_size_input.setText("10")
        self.graph_color_combo.setCurrentText("Royal Blue")
        self.graph_format_combo.setCurrentText("PNG")
        self.heading_checkbox.setChecked(True)
        self.grid_checkbox.setChecked(True)
        self.transparent_bg_checkbox.setChecked(True)
        self.theme_toggle.setChecked(False)
        self.tooltips_checkbox.setChecked(True)
        self.esm2_aggr_checkbox.setChecked(False)
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
        QShortcut(QKeySequence("Ctrl+E"),      self, self.export_analysis_dialog)
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
            ("Ctrl+E",      "Export PDF report"),
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
        self._add_to_history(self.sequence_name, seq)
        for sec, browser in self.report_section_tabs.items():
            if sec in data["report_sections"]:
                browser.setHtml(data["report_sections"][sec])
        self._update_seq_viewer()
        self.update_graph_tabs()
        self.analyze_btn.setEnabled(True)
        # Enable all analysis-dependent buttons
        for btn in (self.export_analysis_btn, self.mutate_btn, self.trunc_run_btn,
                    self.fetch_deeptmhmm_btn, self.fetch_signalp6_btn):
            btn.setEnabled(True)
        self.trunc_run_btn.setToolTip("Run truncation series analysis")
        # Update window title with current sequence name
        title = self.sequence_name or "Untitled"
        self.setWindowTitle(f"BEER — {title}")
        self.statusBar.showMessage(
            f"Analysis complete  |  {len(seq)} aa  |  {self.sequence_name}", 4000
        )
        mw   = self.analysis_data.get("mol_weight", 0)
        pi   = self.analysis_data.get("iso_point", 0)
        info = f"{self.sequence_name or 'Sequence'}  \u00b7  {len(seq)} aa  \u00b7  MW {mw:.1f} Da  \u00b7  pI {pi:.2f}"
        self._seq_info_label.setText(info)
        self._seq_info_label.show()
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

    def _add_to_history(self, name: str, seq: str):
        # Avoid duplicates by sequence
        self._history = [(n, s) for n, s in self._history if s != seq]
        self._history.insert(0, (name or "Sequence", seq))
        self._history = self._history[:10]
        # Rebuild combo and show the just-analyzed sequence as selected
        self.history_combo.blockSignals(True)
        self.history_combo.clear()
        self.history_combo.addItem("— recent sequences —")
        for n, _ in self._history:
            self.history_combo.addItem(n)
        self.history_combo.setCurrentIndex(1)  # show current sequence name
        self.history_combo.blockSignals(False)

    def _on_history_selected(self, idx: int):
        if idx <= 0:
            return
        name, seq = self._history[idx - 1]
        self.seq_text.setPlainText(seq)
        self.sequence_name = name
        # Clear any previously loaded structure since it belongs to a different protein
        self._js("loadPDB(null);")
        self.alphafold_data = None
        self.current_accession = ""
        self.on_analyze()

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

        # Fetch and display protein summary (best-effort, non-blocking)
        self._fetch_and_show_protein_summary(acc, is_pdb)

    def _fetch_and_show_protein_summary(self, acc: str, is_pdb: bool) -> None:
        """Fetch brief metadata from UniProt or RCSB and display in the info bar."""
        try:
            if is_pdb:
                url = f"https://data.rcsb.org/rest/v1/core/entry/{acc.upper()}"
                req = urllib.request.Request(url, headers={
                    "Accept": "application/json", "User-Agent": "BEER/2.0"})
                with urllib.request.urlopen(req, timeout=8) as resp:
                    data = json.loads(resp.read().decode())
                title = data.get("struct", {}).get("title", "")
                pdb_id = acc.upper()
                html = (
                    f"<b>PDB {pdb_id}</b> &nbsp;|&nbsp; {title}"
                )
            else:
                url = f"https://rest.uniprot.org/uniprotkb/{acc}.json"
                req = urllib.request.Request(url, headers={
                    "Accept": "application/json", "User-Agent": "BEER/2.0"})
                with urllib.request.urlopen(req, timeout=8) as resp:
                    data = json.loads(resp.read().decode())
                # Protein name
                pd = data.get("proteinDescription", {})
                rec = pd.get("recommendedName") or (pd.get("submittedNames") or [{}])[0]
                prot_name = (rec.get("fullName") or {}).get("value", "")
                # Gene name
                genes = data.get("genes", [])
                gene = (genes[0].get("geneName") or {}).get("value", "") if genes else ""
                # Organism
                organism = data.get("organism", {}).get("scientificName", "")
                # Function comment (first sentence only)
                func = ""
                for c in data.get("comments", []):
                    if c.get("commentType") == "FUNCTION":
                        texts = c.get("texts", [])
                        if texts:
                            raw = texts[0].get("value", "")
                            func = raw.split(".")[0] + "." if raw else ""
                        break
                parts = []
                if prot_name:
                    parts.append(f"<b>{prot_name}</b>")
                meta = " | ".join(filter(None, [gene, f"<i>{organism}</i>" if organism else ""]))
                if meta:
                    parts.append(meta)
                if func:
                    parts.append(func)
                html = " &nbsp;·&nbsp; ".join(parts)
            if html:
                self._protein_info_bar.setHtml(html)
                self._protein_info_bar.show()
        except Exception:
            pass  # summary is informational only

    def _fetch_pdb_fasta(self, pdb_id: str) -> str:
        """Fetch FASTA sequence(s) from RCSB PDB for a given 4-char PDB ID."""
        url = f"https://www.rcsb.org/fasta/entry/{pdb_id.upper()}"
        req = urllib.request.Request(url, headers={"User-Agent": "BEER/2.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode()

    def _fetch_pdb_structure(self, pdb_id: str) -> str:
        """Download the PDB coordinate file from RCSB for a given 4-char PDB ID."""
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        req = urllib.request.Request(url, headers={"User-Agent": "BEER/2.0"})
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
        self._mark_chip_loading(self.fetch_af_btn)
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

        # ── Protein info bar ──────────────────────────────────────────────
        self._protein_info_bar.hide()

        # ── Sequence info label ───────────────────────────────────────────
        self._seq_info_label.setText("")
        self._seq_info_label.hide()

        # ── Undo state ────────────────────────────────────────────────────
        self._undo_seq  = None
        self._undo_name = None

        # ── Sequence & identity ───────────────────────────────────────────
        self.seq_text.clear()
        self.seq_viewer.clear()
        self.sequence_name = ""
        self.analysis_data = None
        self.batch_data.clear()
        self.current_accession = ""
        self.alphafold_data = None
        self.pfam_domains = []
        self.motif_input.clear()
        self.motif_match_lbl.setText("")

        # ── Chain selector ────────────────────────────────────────────────
        self.chain_combo.blockSignals(True)
        self.chain_combo.clear()
        self.chain_combo.setEnabled(False)
        self.chain_combo.blockSignals(False)

        # ── Analysis report browsers ──────────────────────────────────────
        for browser in self.report_section_tabs.values():
            browser.clear()

        # ── Graphs ───────────────────────────────────────────────────────
        for _tab, vb in self.graph_tabs.values():
            self._clear_layout(vb)

        # ── Structure viewer ──────────────────────────────────────────────
        if self.structure_viewer is not None:
            self._js("loadPDB(null);")

        # ── MSA state ─────────────────────────────────────────────────────
        self._msa_sequences = []
        self._msa_names     = []
        self._msa_mi_apc    = None

        # ── Annotation data ───────────────────────────────────────────────
        self.elm_data       = []
        self.disprot_data   = {}
        self.phasepdb_data  = {}
        self.mobidb_data    = {}
        self.variants_data  = []
        self.intact_data    = {}

        # ── Toolbar buttons ───────────────────────────────────────────────
        self.export_analysis_btn.setEnabled(False)
        self.mutate_btn.setEnabled(False)
        self.export_structure_btn.setEnabled(False)
        for btn in self._db_fetch_btns:
            btn.setEnabled(False)
            btn.setProperty("chip_state", "normal")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

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
        if sec and sec in self._report_sec_to_idx:
            self.report_stack.setCurrentIndex(self._report_sec_to_idx[sec])

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
            summaries["Signal Peptide & GPI"] = (
                ("Signal peptide detected" if has_sp else "No signal peptide predicted")
                + f" (ESM2 score {prob:.2f}).")
        for sec, summary in summaries.items():
            browser = self.report_section_tabs.get(sec)
            if browser:
                current_html = browser.toHtml()
                banner = (f"<p style='background:#f0f4ff;border-left:3px solid #4361ee;"
                          f"padding:4px 8px;margin:0 0 6px 0;font-size:9pt;color:#2d3748;'>"
                          f"<b>Summary:</b> {summary}</p>")
                if "Summary:" not in current_html:
                    browser.setHtml(banner + current_html)

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


