"""Publication-quality styling helpers for BEER graphs."""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt  # noqa: F401 (for callers that import plt from here)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_PALETTE = [
    "#4361ee", "#f72585", "#4cc9f0", "#7209b7", "#3a0ca3",
    "#f3722c", "#43aa8b", "#277da1", "#577590", "#90be6d",
]

_ACCENT   = "#4361ee"
_FILL     = "#4361ee"
_NEG_COL  = "#f72585"
_POS_COL  = "#4361ee"
_NEUT_COL = "#adb5bd"

# Uniform palette for all sequence profile plots (line + fills)
_PROFILE_LINE  = "#1e2640"   # near-black curve for all profiles
_FILL_ABOVE    = "#4361ee"   # blue  — hydrophobic / positive charge / BiLSTM predicted
_FILL_BELOW    = "#f72585"   # pink  — hydrophilic / negative charge
_FILL_NEUTRAL  = "#d0d8e8"   # light blue-gray — BiLSTM "below threshold" (not predicted)

# ---------------------------------------------------------------------------
# Per-feature colour map (21 BiLSTM heads + aggregation)
# Used consistently across profile plots, dual-track figures, and report badges.
# ---------------------------------------------------------------------------

FEATURE_COLORS: dict[str, str] = {
    "disorder":        "#f3722c",  # warm orange — flexibility
    "signal_peptide":  "#4361ee",  # blue — secretory
    "transmembrane":   "#7209b7",  # purple — membrane
    "coiled_coil":     "#f72585",  # magenta — structural repeat
    "dna_binding":     "#3a0ca3",  # deep violet — nucleic acid
    "active_site":     "#e63946",  # red — catalytic
    "binding_site":    "#2a9d8f",  # teal — ligand interaction
    "phosphorylation": "#e9c46a",  # amber — PTM
    "lcd":             "#90be6d",  # green — low complexity
    "zinc_finger":     "#457b9d",  # steel blue — metal binding
    "glycosylation":   "#f4a261",  # peach — sugar modification
    "ubiquitination":  "#e76f51",  # terracotta — degradation
    "methylation":     "#84a98c",  # sage green — epigenetic
    "acetylation":     "#52b788",  # emerald — histone mark
    "lipidation":      "#9b5de5",  # lavender — membrane anchor
    "disulfide":       "#fcbf49",  # gold — structural bond
    "intramembrane":   "#9b2226",  # dark red — membrane interior
    "motif":           "#606c38",  # olive — functional sequence
    "propeptide":      "#bc6c25",  # brown — processing
    "repeat":          "#577590",  # slate blue — structural repeat
    "aggregation":     "#4682b4",  # steel blue — aggregation
}


# ---------------------------------------------------------------------------
# Axes styling helpers
# ---------------------------------------------------------------------------

def _pub_style_ax(
    ax,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    grid: bool = True,
    despine: bool = True,
    title_size: int = 11,
    label_size: int = 11,
    tick_size: int = 10,
) -> None:
    """Apply publication-quality styling to a matplotlib Axes object."""
    if title:
        ax.set_title(title, fontsize=title_size, fontweight="bold", pad=10,
                     color="#1a1a2e")
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
        ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.5, color="#c8cdd8")
        ax.set_axisbelow(True)
    ax.set_facecolor("#fafbff")


def _apply_font_sizes(ax, label_font: int, tick_font: int) -> None:
    """Apply consistent label and tick font sizes to an Axes object."""
    ax.xaxis.label.set_fontsize(label_font)
    ax.yaxis.label.set_fontsize(label_font)
    ax.title.set_fontsize(max(label_font - 2, 8))
    ax.tick_params(axis="both", labelsize=tick_font)


def _residue_x(seq: str):
    """Return 1-based residue position array for a sequence string."""
    import numpy as np
    return np.arange(1, len(seq) + 1, dtype=float)
