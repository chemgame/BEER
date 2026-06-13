"""Publication-quality styling helpers for BEER graphs."""
from __future__ import annotations

import matplotlib

# Publication font: prefer Arial/Helvetica (journal standard); fall back to the
# bundled Liberation Sans / DejaVu Sans so this never fails when Arial is absent.
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = [
    "Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans",
]
matplotlib.rcParams["mathtext.fontset"] = "dejavusans"
matplotlib.rcParams["axes.unicode_minus"] = False

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
    "aggregation":                  "#4682b4",  # steel blue — aggregation
    "secondary_structure_helix":   "#e63946",  # red — α-helix
    "secondary_structure_strand":  "#457b9d",  # steel blue — β-strand
    "secondary_structure_coil":    "#adb5bd",  # grey — coil/loop
    "rna_binding":                 "#2dc653",  # green — RNA interaction
    "nucleotide_binding":          "#0ea5e9",  # sky blue — nucleotide
    "transit_peptide":             "#d946ef",  # fuchsia — targeting sequence
}


# ---------------------------------------------------------------------------
# Figure theme palette (screen). Export always forces LIGHT (see apply_figure_theme).
# ---------------------------------------------------------------------------

_LIGHT_THEME = {
    "fig_bg": "#ffffff", "ax_bg": "#fafbff", "title": "#1a1a2e",
    "label": "#2d3748", "tick": "#4a5568", "spine": "#c0c4d0",
    "grid": "#c8cdd8", "legend_edge": "#d0d4e0",
}
_DARK_THEME = {
    "fig_bg": "#1a1a2e", "ax_bg": "#222742", "title": "#e8eaf0",
    "label": "#c7cbd9", "tick": "#aab0c4", "spine": "#3a4060",
    "grid": "#3a4060", "legend_edge": "#3a4060",
}
_ACTIVE_THEME = dict(_LIGHT_THEME)


def set_figure_dark(dark: bool) -> None:
    """Switch the palette new figures are styled with (called by the theme toggle)."""
    _ACTIVE_THEME.clear()
    _ACTIVE_THEME.update(_DARK_THEME if dark else _LIGHT_THEME)


def apply_figure_theme(fig, dark: "bool | None" = None) -> None:
    """Recolour an existing figure's chrome (bg, title, labels, ticks, spines,
    grid, legend) to the light or dark palette. Data colours are untouched.
    Pass dark=False before export so saved figures are always publication-light;
    pass the live theme when embedding a figure in the on-screen canvas."""
    t = _ACTIVE_THEME if dark is None else (_DARK_THEME if dark else _LIGHT_THEME)
    fig.patch.set_facecolor(t["fig_bg"])
    for ax in fig.get_axes():
        ax.set_facecolor(t["ax_bg"])
        ax.title.set_color(t["title"])
        ax.xaxis.label.set_color(t["label"])
        ax.yaxis.label.set_color(t["label"])
        ax.tick_params(colors=t["tick"])
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_color(t["tick"])
        for sp in ax.spines.values():
            sp.set_color(t["spine"])
        for gl in ax.get_xgridlines() + ax.get_ygridlines():
            gl.set_color(t["grid"])
        leg = ax.get_legend()
        if leg is not None:
            leg.get_frame().set_edgecolor(t["legend_edge"])
            leg.get_frame().set_facecolor(t["fig_bg"])
            for txt in leg.get_texts():
                txt.set_color(t["label"])


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
    """Apply publication-quality styling to a matplotlib Axes object.

    Chrome colours come from the active theme palette (see set_figure_dark), so a
    figure built while dark mode is active is themed consistently; export forces
    the light palette via apply_figure_theme."""
    t = _ACTIVE_THEME
    if title:
        ax.set_title(title, fontsize=title_size, fontweight="bold", pad=10,
                     color=t["title"])
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=label_size, labelpad=6, color=t["label"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=label_size, labelpad=6, color=t["label"])
    ax.tick_params(labelsize=tick_size, length=4, width=0.8, colors=t["tick"])
    if despine:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.spines["left"].set_color(t["spine"])
        ax.spines["bottom"].set_color(t["spine"])
    if grid:
        ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.5, color=t["grid"])
        ax.set_axisbelow(True)
    ax.set_facecolor(t["ax_bg"])


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
