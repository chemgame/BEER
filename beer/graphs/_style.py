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
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, color="#c8cdd8")
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
