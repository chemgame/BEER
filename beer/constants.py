"""BEER constants — all scales, lookup tables, and UI/report configuration."""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Aggregation / solubility scales
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
"""Diagonal of PASTA pairwise β-strand interaction energy matrix (Trovato et al. 2007)."""

CAMSOLMT_SCALE: dict[str, float] = {
    'A':  0.238, 'R': -0.132, 'N':  0.047, 'D':  0.191,
    'C':  0.238, 'Q':  0.047, 'E':  0.191, 'G':  0.024,
    'H': -0.083, 'I': -0.387, 'L': -0.387, 'K': -0.065,
    'M': -0.265, 'F': -0.386, 'P':  0.190, 'S':  0.264,
    'T':  0.209, 'W': -0.380, 'Y': -0.241, 'V': -0.322,
}
"""CamSol intrinsic solubility scale (Sormanni et al. 2015, J Mol Biol)."""

# ---------------------------------------------------------------------------
# Hydrophobicity scales
# ---------------------------------------------------------------------------

KYTE_DOOLITTLE: dict[str, float] = {
    'A':  1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5,
    'C':  2.5, 'Q': -3.5, 'E': -3.5, 'G': -0.4,
    'H': -3.2, 'I':  4.5, 'L':  3.8, 'K': -3.9,
    'M':  1.9, 'F':  2.8, 'P': -1.6, 'S': -0.8,
    'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V':  4.2,
}
"""Kyte-Doolittle hydropathy scale (Kyte & Doolittle 1982)."""

EISENBERG_SCALE: dict[str, float] = {
    'A':  0.620, 'R': -2.530, 'N': -0.780, 'D': -0.900,
    'C':  0.290, 'Q': -0.850, 'E': -0.740, 'G':  0.480,
    'H': -0.400, 'I':  1.380, 'L':  1.060, 'K': -1.500,
    'M':  0.640, 'F':  1.190, 'P':  0.120, 'S': -0.180,
    'T': -0.050, 'W':  0.810, 'Y':  0.260, 'V':  1.080,
}
"""Eisenberg normalised consensus hydrophobicity scale (Eisenberg et al. 1984 PNAS)."""

WIMLEY_WHITE: dict[str, float] = {
    'A':  0.17, 'R': -0.81, 'N': -0.42, 'D': -1.23, 'C':  0.24,
    'Q': -0.58, 'E': -2.02, 'G': -0.01, 'H': -0.17, 'I':  1.25,
    'L':  1.22, 'K': -2.80, 'M':  0.67, 'F':  1.13, 'P': -0.45,
    'S': -0.13, 'T':  0.14, 'W':  1.85, 'Y':  0.94, 'V':  0.07,
}
"""Wimley-White whole-residue interfacial hydrophobicity (negated ΔGwif, kcal/mol).
Wimley & White 1996, Nature Structural Biology 3:842. Positive = hydrophobic."""

HESSA_SCALE: dict[str, float] = {
    'A':  0.11, 'R': -2.58, 'N': -0.84, 'D': -3.49, 'C':  0.13,
    'Q': -0.87, 'E': -3.45, 'G':  0.04, 'H': -0.97, 'I':  0.60,
    'L':  1.25, 'K': -2.60, 'M':  0.10, 'F':  0.32, 'P': -0.84,
    'S': -0.46, 'T': -0.25, 'W': -0.30, 'Y': -0.73, 'V':  0.31,
}
"""Hessa et al. biological hydrophobicity (negated ΔGapp, kcal/mol).
Hessa et al. 2005, Nature 433:377. Positive = hydrophobic (favorable TM insertion)."""

MOON_FLEMING_SCALE: dict[str, float] = {
    'A':  0.92, 'R': -2.45, 'N': -1.60, 'D': -3.17, 'C':  0.85,
    'Q': -1.31, 'E': -3.18, 'G':  0.35, 'H': -0.49, 'I':  1.22,
    'L':  1.64, 'K': -2.49, 'M':  1.03, 'F':  1.27, 'P': -2.85,
    'S': -0.19, 'T': -0.06, 'W':  0.87, 'Y': -0.73, 'V':  1.06,
}
"""Moon-Fleming biological hydrophobicity (negated ΔGapp, kcal/mol).
Moon & Fleming 2011, PNAS 108:10174. Positive = hydrophobic (translocon scale)."""

GES_SCALE: dict[str, float] = {
    'A':  1.20, 'R': -3.07, 'N': -0.53, 'D': -3.82, 'C':  1.56,
    'Q': -0.53, 'E': -3.63, 'G':  0.84, 'H':  0.25, 'I':  2.23,
    'L':  2.32, 'K': -4.24, 'M':  1.69, 'F':  2.51, 'P': -0.90,
    'S':  0.41, 'T':  0.68, 'W':  3.00, 'Y':  2.27, 'V':  1.67,
}
"""Goldman-Engelman-Steitz (GES) hydrophobicity (negated ΔG transfer to bilayer, kcal/mol).
Engelman et al. 1986, Annual Review of Biophysics 15:321. Positive = hydrophobic."""

HOPP_WOODS: dict[str, float] = {
    'A':  0.5, 'R': -3.0, 'N': -0.2, 'D': -3.0, 'C':  1.0,
    'Q': -0.2, 'E': -3.0, 'G':  0.0, 'H':  0.5, 'I':  1.8,
    'L':  1.8, 'K': -3.0, 'M':  1.3, 'F':  2.5, 'P':  0.0,
    'S': -0.3, 'T':  0.4, 'W':  3.4, 'Y':  2.3, 'V':  1.5,
}
"""Hopp-Woods scale (negated hydrophilicity). Hopp & Woods 1981, PNAS 78:3824.
Positive = hydrophobic. Original is a hydrophilicity scale for surface/antigenicity prediction."""

FAUCHERE_PLISKA: dict[str, float] = {
    'A':  0.31, 'R': -1.01, 'N': -0.60, 'D': -0.77, 'C':  1.54,
    'Q': -0.22, 'E': -0.64, 'G':  0.00, 'H':  0.13, 'I':  1.80,
    'L':  1.70, 'K': -0.99, 'M':  1.23, 'F':  1.79, 'P':  0.72,
    'S': -0.04, 'T':  0.26, 'W':  2.25, 'Y':  0.96, 'V':  1.22,
}
"""Fauche-Pliska octanol/water partition coefficients (logP).
Fauche & Pliska 1983, Eur. J. Med. Chem. 18:369. Positive = hydrophobic (lipophilic)."""

URRY_SCALE: dict[str, float] = {
    'A':  0.12, 'R': -1.84, 'N': -0.89, 'D': -1.27, 'C':  0.93,
    'Q': -0.97, 'E': -2.04, 'G':  0.00, 'H':  0.50, 'I':  1.40,
    'L':  1.80, 'K': -1.38, 'M':  1.30, 'F':  2.00, 'P':  0.46,
    'S': -0.46, 'T': -0.12, 'W':  2.08, 'Y':  1.80, 'V':  1.08,
}
"""Urry scale — inverse temperature transition of elastin-like polypeptides (negated ΔTt, kcal/mol).
Urry et al. 1992, Biopolymers 32:1243. Positive = hydrophobic (drives LLPS/coacervation).
Particularly relevant for IDP/phase-separation research."""

HYDROPHOBICITY_SCALES: dict[str, dict] = {
    "Kyte-Doolittle":   {"values": KYTE_DOOLITTLE,     "ylabel": "KD Score",      "ref": "Kyte & Doolittle 1982"},
    "Eisenberg":        {"values": EISENBERG_SCALE,     "ylabel": "Eisenberg",     "ref": "Eisenberg et al. 1984"},
    "Wimley-White":     {"values": WIMLEY_WHITE,        "ylabel": "ΔG (WW)",       "ref": "Wimley & White 1996"},
    "Hessa":            {"values": HESSA_SCALE,         "ylabel": "ΔG (Hessa)",    "ref": "Hessa et al. 2005"},
    "Moon-Fleming":     {"values": MOON_FLEMING_SCALE,  "ylabel": "ΔG (MF)",       "ref": "Moon & Fleming 2011"},
    "GES":              {"values": GES_SCALE,           "ylabel": "ΔG (GES)",      "ref": "Engelman et al. 1986"},
    "Hopp-Woods":       {"values": HOPP_WOODS,          "ylabel": "HW Score",      "ref": "Hopp & Woods 1981"},
    "Fauche-Pliska":    {"values": FAUCHERE_PLISKA,     "ylabel": "logP (FP)",     "ref": "Fauche & Pliska 1983"},
    "Urry":             {"values": URRY_SCALE,          "ylabel": "Urry Score",    "ref": "Urry et al. 1992"},
}
"""Registry of all hydrophobicity scales available in BEER.
All scales stored with positive = hydrophobic for display consistency."""

# ---------------------------------------------------------------------------
# Amino acid sets
# ---------------------------------------------------------------------------

VALID_AMINO_ACIDS: set[str] = set("ACDEFGHIKLMNPQRSTVWY")

# ---------------------------------------------------------------------------
# pKa values
# ---------------------------------------------------------------------------

# Classic free amino-acid pKa values from:
# Dawson, R.M.C. et al. (1986) Data for Biochemical Research, 3rd ed.
# Oxford: Clarendon Press. Table of pKa values.
# These are the same values used by Biopython ProteinAnalysis and the
# ExPASy ProtParam server. NTERM and CTERM refer to the alpha-amino and
# alpha-carboxyl groups of free amino acids; protein-context terminal pKa
# values differ (typically N-term ~8.0, C-term ~3.1) but these standard
# values are the accepted bioinformatics approximation.
DEFAULT_PKA: dict[str, float] = {
    'NTERM': 9.69, 'CTERM': 2.34,
    'D': 3.90, 'E': 4.07, 'C': 8.18, 'Y': 10.46,
    'H': 6.04, 'K': 10.54, 'R': 12.48,
}

# ---------------------------------------------------------------------------
# UI colour/colormap registries
# ---------------------------------------------------------------------------

NAMED_COLORS: dict[str, str] = {
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

NAMED_COLORMAPS: list[str] = [
    "coolwarm", "RdBu", "RdYlBu", "RdYlGn", "Spectral",
    "viridis", "plasma", "inferno", "magma", "cividis", "turbo",
    "PiYG", "PRGn", "BrBG", "puOr",
    "Blues", "Reds", "Greens", "Purples", "Oranges",
    "YlOrRd", "YlGnBu", "GnBu", "OrRd",
    "hot", "copper", "cool", "autumn", "winter", "spring", "summer",
    "twilight", "twilight_shifted", "hsv",
]

# ---------------------------------------------------------------------------
# Disorder / order residue classification (Uversky)
# ---------------------------------------------------------------------------

DISORDER_PROMOTING: set[str] = set("AEGKPQRS")
ORDER_PROMOTING: set[str] = set("CFHILMVWY")

# ---------------------------------------------------------------------------
# Sticker residue sets for phase separation analysis
# ---------------------------------------------------------------------------

STICKER_AROMATIC: set[str] = set("FWY")
STICKER_ELECTROSTATIC: set[str] = set("KRDE")
STICKER_ALL: set[str] = STICKER_AROMATIC | STICKER_ELECTROSTATIC

# ---------------------------------------------------------------------------
# Prion-like domain composition residues (PLAAC/Lancaster)
# ---------------------------------------------------------------------------

PRION_LIKE: set[str] = set("NQSGY")

# ---------------------------------------------------------------------------
# LARKS residues
# ---------------------------------------------------------------------------

LARKS_AROMATIC: set[str] = set("FWY")
LARKS_LC: set[str] = set("GASTNQ")  # low-complexity residues for LARKS windows

# ---------------------------------------------------------------------------
# Coiled-coil heptad periodicity (Lupas/Berger matrix — simplified mean propensity)
# Position-specific heptad weighting (a/d > e/g > b/c/f) is applied in
# beer/analysis/structure.py:_score_coiled_coil_window(), not stored here.
# ---------------------------------------------------------------------------

COILED_COIL_PROPENSITY: dict[str, float] = {
    'A': 1.29, 'R': 0.96, 'N': 0.90, 'D': 0.72, 'C': 0.77,
    'Q': 1.23, 'E': 1.44, 'G': 0.56, 'H': 0.92, 'I': 0.98,
    'L': 1.36, 'K': 1.17, 'M': 1.16, 'F': 0.73, 'P': 0.40,
    'S': 0.82, 'T': 0.89, 'W': 0.72, 'Y': 0.70, 'V': 1.09,
}

# ---------------------------------------------------------------------------
# Linear motif regex library (name, pattern, description)
# ---------------------------------------------------------------------------

LINEAR_MOTIFS: list[tuple[str, str, str]] = [
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
    ("WW domain ligand", r"PP.Y", "WW-domain binding (PPxY)"),
    ("Caspase-3 cleavage", r"DEVD|DMQD", "Caspase-3/7 cleavage motifs DEVD and DMQD; cleavage occurs after the C-terminal Asp"),
    ("Glycosylation (N-linked)", r"N[^P][ST]", "N-linked glycosylation sequon (NxS/T, x\u2260P)"),
    ("SUMOylation", r"[VILMF]K.E", "SUMOylation consensus (\u03a8KxE)"),
    ("Phospho (CK2)", r"[ST].{2}[DE]", "CK2 phosphorylation consensus (S/TxxE/D)"),
]

# ---------------------------------------------------------------------------
# Chou-Fasman propensities
# ---------------------------------------------------------------------------

CHOU_FASMAN_HELIX: dict[str, float] = {
    'A': 1.42, 'R': 0.98, 'N': 0.67, 'D': 1.01, 'C': 0.70,
    'Q': 1.11, 'E': 1.51, 'G': 0.57, 'H': 1.00, 'I': 1.08,
    'L': 1.21, 'K': 1.16, 'M': 1.45, 'F': 1.13, 'P': 0.57,
    'S': 0.77, 'T': 0.83, 'W': 1.08, 'Y': 0.69, 'V': 1.06,
}
CHOU_FASMAN_SHEET: dict[str, float] = {
    'A': 0.83, 'R': 0.93, 'N': 0.89, 'D': 0.54, 'C': 1.19,
    'Q': 1.10, 'E': 0.37, 'G': 0.75, 'H': 0.87, 'I': 1.60,
    'L': 1.30, 'K': 0.74, 'M': 1.05, 'F': 1.38, 'P': 0.55,
    'S': 0.75, 'T': 1.19, 'W': 1.37, 'Y': 1.47, 'V': 1.70,
}

# ---------------------------------------------------------------------------
# Per-residue disorder propensity (classical fallback scale)
# These values are a classical physicochemical composite derived from
# amino-acid flexibility, hydrophobicity, and secondary-structure tendencies
# as a fallback when ESM2/metapredict is unavailable. They do not reproduce
# IUPred2A (Mészáros et al. 2018) or any single published scale exactly;
# they are used only for qualitative profile visualisation. When ESM2 or
# metapredict is available those predictions take precedence.
# ---------------------------------------------------------------------------

DISORDER_PROPENSITY: dict[str, float] = {
    'A':  0.060, 'R': -0.260, 'N':  0.007, 'D':  0.192, 'C': -0.020,
    'Q': -0.091, 'E':  0.736, 'G':  0.166, 'H': -0.303, 'I': -0.486,
    'L': -0.326, 'K':  0.586, 'M': -0.397, 'F': -0.697, 'P':  0.987,
    'S':  0.341, 'T':  0.059, 'W': -0.884, 'Y': -0.510, 'V': -0.386,
}

# ---------------------------------------------------------------------------
# Residue colours for the colour-coded sequence viewer
# ---------------------------------------------------------------------------

_AA_COLOURS: dict[str, str] = {
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

# ---------------------------------------------------------------------------
# Report / graph registries
# ---------------------------------------------------------------------------

REPORT_SECTIONS: list[str] = [
    "Composition",
    "Properties",
    "Hydrophobicity",
    "Charge",
    "Aromatic & \u03c0",
    "Low Complexity",
    "Disorder",
    "Repeat Motifs",
    "Sticker & Spacer",
    "TM Helices",
    "LARKS",
    "Linear Motifs",
    # --- New sections ---
    "\u03b2-Aggregation & Solubility",
    "Signal Peptide & GPI",
    "Amphipathic Helices",
    "Charge Decoration (SCD)",
    "RNA Binding",
    "Tandem Repeats",
    "Proteolytic Map",
]

GRAPH_TITLES: list[str] = [
    "Amino Acid Composition (Bar)",
    "Amino Acid Composition (Pie)",
    "Hydrophobicity Profile",
    "Bead Model (Hydrophobicity)",
    "Bead Model (Charge)",
    "Sticker Map",
    "Local Charge Profile",
    "Local Complexity",
    "Cation\u2013\u03c0 Map",
    "Isoelectric Focus",
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
    "Single-Residue Perturbation Map",
    # --- New graphs ---
    "\u03b2-Aggregation Profile",
    "Solubility Profile",
    "Hydrophobic Moment",
    "AlphaMissense",
    "RNA-Binding Profile",
    "SCD Profile",
    "Truncation Series",
    "Ramachandran Plot",
    "Residue Contact Network",
    "MSA Conservation",
    "MSA Covariance",
    "Complex Mass",
    "Annotation Track",
    "Cleavage Map",
    "PLAAC Profile",
]

# Graph categories for the tree browser (order matters; every GRAPH_TITLES entry must appear here)
GRAPH_CATEGORIES: list[tuple[str, list[str]]] = [
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
    ]),
    ("Charge & \u03c0-Interactions", [
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
        "Single-Residue Perturbation Map",
        "SCD Profile",
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
        "AlphaMissense",
    ]),
    ("Post-Translational & Binding", [
        "RNA-Binding Profile",
    ]),
    ("Variant Effects", [
        "Variant Effect Map",
    ]),
    ("Evolutionary & Comparative", [
        "Truncation Series",
        "MSA Conservation",
        "MSA Covariance",
        "Complex Mass",
    ]),
    ("Sequence Analysis", [
        "Annotation Track",
        "Cleavage Map",
        "PLAAC Profile",
    ]),
]

# ---------------------------------------------------------------------------
# RNA-binding propensity scores and motifs
# ---------------------------------------------------------------------------

RBP_RESIDUE_PROPENSITY: dict[str, float] = {
    'K':  0.72, 'R':  0.80, 'Y':  0.44, 'F':  0.36,
    'W':  0.51, 'G':  0.25, 'S':  0.10, 'T':  0.08,
    'N':  0.06, 'H':  0.35, 'D': -0.15, 'E': -0.42,
    'L': -0.20, 'I': -0.18, 'V': -0.12, 'A': -0.05,
    'M':  0.12, 'C':  0.15, 'P': -0.25, 'Q': -0.08,
}
"""Per-residue RNA-binding propensity scores (Jeong et al. 2012, scaled to [-1, 1])."""

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
