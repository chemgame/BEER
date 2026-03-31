"""Structure-related figures: bead models, helical wheel, TM topology, etc."""
from __future__ import annotations

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from beer.constants import KYTE_DOOLITTLE
from beer.graphs._style import (
    _pub_style_ax, _PALETTE, _ACCENT,
)

_STICKER_AROMATIC     = set("FWY")
_STICKER_ELECTROSTATIC = set("KRDE")
_STICKER_ALL           = _STICKER_AROMATIC | _STICKER_ELECTROSTATIC


def _bead_width(n: int) -> float:
    """Figure width that scales with sequence length for bead models."""
    return max(10, min(22, 0.25 * n + 4))


def _x_tick_step(n: int) -> int:
    """Adaptive x-axis tick spacing so bead/linear plots stay readable at any length."""
    for s in (5, 10, 20, 25, 50, 100, 200, 250, 500):
        if n // s <= 20:
            return s
    return 500


def create_bead_model_hydrophobicity_figure(
    seq: str,
    show_labels: bool,
    label_font: int = 14,
    tick_font: int = 12,
    cmap: str = "coolwarm",
) -> Figure:
    """Linear bead model coloured by Kyte-Doolittle hydrophobicity."""
    n = len(seq)
    fig = Figure(figsize=(_bead_width(n), 2.4), dpi=120)
    fig.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)
    xs = list(range(1, n + 1))
    vals = [KYTE_DOOLITTLE.get(aa, 0.0) for aa in seq]
    sc = ax.scatter(xs, [1] * n, c=vals, cmap=cmap,
                    s=200, linewidths=0.4, edgecolors="white",
                    vmin=-4.5, vmax=4.5, zorder=4)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.65, aspect=12, pad=0.02)
    cbar.set_label("Hydrophobicity", fontsize=tick_font - 1, color="#4a5568")
    cbar.ax.tick_params(labelsize=tick_font - 2, colors="#4a5568")
    ax.set_yticks([])
    ax.set_xlim(0, n + 1)
    ax.set_ylim(0.5, 1.5)
    _pub_style_ax(ax, title="Bead Model: Hydrophobicity",
                  xlabel="Residue", grid=False, despine=False,
                  title_size=label_font - 1, label_size=label_font - 2,
                  tick_size=tick_font - 2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    if show_labels and n <= 60:
        for i, aa in enumerate(seq):
            ax.text(xs[i], 1, aa, ha="center", va="center",
                    fontsize=max(5, label_font - 5), color="white",
                    fontweight="bold")
    _step = _x_tick_step(n)
    ax.set_xticks(range(_step, n + 1, _step))
    ax.tick_params(labelsize=tick_font - 2)
    fig.tight_layout(pad=1.2)
    return fig


def create_bead_model_charge_figure(
    seq: str,
    show_labels: bool,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Linear bead model coloured by charge (K/R/D/E/H)."""
    n = len(seq)
    fig = Figure(figsize=(_bead_width(n), 2.4), dpi=120)
    fig.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)
    xs = list(range(1, n + 1))
    pos_c = "#4361ee"; neg_c = "#f72585"; neu_c = "#adb5bd"; his_c = "#4cc9f0"
    cols = []
    for aa in seq:
        if aa in "KR":    cols.append(pos_c)
        elif aa in "DE":  cols.append(neg_c)
        elif aa == "H":   cols.append(his_c)
        else:             cols.append(neu_c)
    ax.scatter(xs, [1] * n, c=cols, s=200, linewidths=0.4,
               edgecolors="white", zorder=4)
    ax.legend(handles=[
        Patch(color=pos_c, label="Positive (K,R)"),
        Patch(color=neg_c, label="Negative (D,E)"),
        Patch(color=his_c, label="His (H)"),
        Patch(color=neu_c, label="Neutral"),
    ], loc="upper right", fontsize=max(7, label_font - 5),
        framealpha=0.85, edgecolor="#d0d4e0")
    ax.set_yticks([])
    ax.set_xlim(0, n + 1)
    ax.set_ylim(0.5, 1.5)
    _pub_style_ax(ax, title="Bead Model: Charge",
                  xlabel="Residue", grid=False, despine=False,
                  title_size=label_font - 1, label_size=label_font - 2,
                  tick_size=tick_font - 2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    if show_labels and n <= 60:
        for i, aa in enumerate(seq):
            ax.text(xs[i], 1, aa, ha="center", va="center",
                    fontsize=max(5, label_font - 5), color="white",
                    fontweight="bold")
    _step = _x_tick_step(n)
    ax.set_xticks(range(_step, n + 1, _step))
    ax.tick_params(labelsize=tick_font - 2)
    fig.tight_layout(pad=1.2)
    return fig


def create_helical_wheel_figure(
    seq: str,
    label_font: int = 14,
) -> Figure:
    """Helical wheel projection (first ≤18 residues)."""
    seg = seq[:18]
    n = len(seg)
    fig = Figure(figsize=(6.0, 6.0), dpi=120)
    fig.set_facecolor("#ffffff")
    ax = fig.add_axes([0.06, 0.06, 0.78, 0.84])
    ax.set_facecolor("#fafbff")
    ax.set_aspect("equal")
    ax.axis("off")

    cmap = plt.get_cmap("RdYlBu_r")
    kd_min, kd_max = -4.5, 4.5
    norm = Normalize(vmin=kd_min, vmax=kd_max)
    DOT_R = 0.13
    RING_R = 1.0

    def _pos(i):
        theta = math.radians(90.0 - i * 100.0)
        return math.cos(theta) * RING_R, math.sin(theta) * RING_R

    xs = [_pos(i)[0] for i in range(n)]
    ys = [_pos(i)[1] for i in range(n)]

    for i in range(n - 1):
        ax.plot([xs[i], xs[i + 1]], [ys[i], ys[i + 1]],
                color="#b0bac8", linewidth=1.0, zorder=2, solid_capstyle="round")

    for i, aa in enumerate(seg):
        kd = KYTE_DOOLITTLE.get(aa, 0.0)
        col = cmap(norm(kd))
        r, g, b, _ = col
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        txt_col = "#1a1a2e" if lum > 0.45 else "white"
        circle = plt.Circle((xs[i], ys[i]), DOT_R, color=col, zorder=4,
                             linewidth=1.0, edgecolor="#718096")
        ax.add_patch(circle)
        ax.text(xs[i], ys[i], aa, ha="center", va="center",
                fontsize=label_font - 2, fontweight="bold",
                color=txt_col, zorder=5)
        nx = xs[i] * (1.0 + (DOT_R + 0.07) / RING_R)
        ny = ys[i] * (1.0 + (DOT_R + 0.07) / RING_R)
        ax.text(nx, ny, str(i + 1), ha="center", va="center",
                fontsize=max(6, label_font - 6), color="#718096", zorder=5)

    pad = RING_R + DOT_R + 0.28
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-pad, pad)
    ax.set_title(f"Helical Wheel  (1–{n})",
                 fontsize=label_font - 1, fontweight="bold", color="#1a1a2e", pad=8)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.87, 0.12, 0.03, 0.68])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("KD Score", fontsize=label_font - 4, color="#4a5568")
    cbar.ax.tick_params(labelsize=label_font - 5, colors="#4a5568")
    return fig


def create_tm_topology_figure(
    seq: str,
    helices: list,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Simplified transmembrane topology diagram (snake-plot style)."""
    n = len(seq)
    w = max(9, min(18, n * 0.06 + 4))
    fig = Figure(figsize=(w, 4.5), dpi=120)
    fig.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#fafbff")
    ax.axhspan(-0.5, 0.5, alpha=0.12, color="#f59e0b")
    ax.text(2, 0.65, "Extracellular", fontsize=tick_font - 2,
            color="#6b7280", style="italic")
    ax.text(2, -0.85, "Cytoplasmic", fontsize=tick_font - 2,
            color="#6b7280", style="italic")
    side = 1
    prev_end = 0
    for h in helices:
        s, e = h["start"], h["end"]
        y = side * 1.15
        if s > prev_end:
            ax.plot([prev_end + 1, s], [y, y],
                    color="#4361ee", linewidth=1.6,
                    solid_capstyle="round", zorder=3)
        rect = Rectangle((s + 1, -0.5), e - s, 1.0,
                          color="#4361ee", alpha=0.75, zorder=4, linewidth=0)
        ax.add_patch(rect)
        mid = (s + e) / 2 + 1
        ax.text(mid, 0, f"{s+1}–{e+1}",
                ha="center", va="center",
                fontsize=max(5, tick_font - 5), color="white",
                fontweight="bold", zorder=5)
        side = -side
        prev_end = e
    y = side * 1.15
    ax.plot([prev_end + 1, n], [y, y],
            color="#4361ee", linewidth=1.6, solid_capstyle="round", zorder=3)
    _pub_style_ax(ax, title=f"TM Topology  ({len(helices)} helices)",
                  xlabel="Residue", ylabel="",
                  grid=False, title_size=label_font - 1,
                  label_size=label_font - 1, tick_size=tick_font - 1)
    ax.set_xlim(0, n + 2)
    ax.set_ylim(-1.6, 1.8)
    ax.set_yticks([])
    ax.legend(handles=[Patch(color="#f59e0b", alpha=0.3, label="Membrane"),
                        Patch(color="#4361ee", alpha=0.75, label="TM helix")],
              fontsize=tick_font - 3, framealpha=0.85, edgecolor="#d0d4e0",
              loc="upper right")
    fig.tight_layout(pad=1.5)
    return fig


def create_sticker_map_figure(
    seq: str,
    show_labels: bool,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Sticker map showing aromatic, basic, and acidic residues."""
    n = len(seq)
    fig = Figure(figsize=(_bead_width(n), 2.4), dpi=120)
    fig.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)
    xs = list(range(1, n + 1))
    arom_c = "#f59e0b"; basic_c = "#4361ee"
    acid_c = "#f72585"; space_c = "#e2e8f0"
    cols = []
    for aa in seq:
        if aa in _STICKER_AROMATIC: cols.append(arom_c)
        elif aa in "KR":            cols.append(basic_c)
        elif aa in "DE":            cols.append(acid_c)
        else:                       cols.append(space_c)
    ax.scatter(xs, [1] * n, c=cols, s=200, linewidths=0.4,
               edgecolors="white", zorder=4)
    ax.legend(handles=[
        Patch(color=arom_c,  label="Aromatic (F,W,Y)"),
        Patch(color=basic_c, label="Basic (K,R)"),
        Patch(color=acid_c,  label="Acidic (D,E)"),
        Patch(color=space_c, label="Spacer"),
    ], loc="upper right", fontsize=max(7, label_font - 5),
        framealpha=0.85, edgecolor="#d0d4e0")
    ax.set_yticks([])
    ax.set_xlim(0, n + 1)
    ax.set_ylim(0.5, 1.5)
    _pub_style_ax(ax, title="Sticker Map",
                  xlabel="Residue", grid=False, despine=False,
                  title_size=label_font - 1, label_size=label_font - 2,
                  tick_size=tick_font - 2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    if show_labels and n <= 60:
        for i, aa in enumerate(seq):
            tc = "#ffffff" if aa in _STICKER_ALL else "#6b7280"
            ax.text(xs[i], 1, aa, ha="center", va="center",
                    fontsize=max(5, label_font - 5), color=tc, fontweight="bold")
    _step = _x_tick_step(n)
    ax.set_xticks(range(_step, n + 1, _step))
    ax.tick_params(labelsize=tick_font - 2)
    fig.tight_layout(pad=1.2)
    return fig


def create_hydrophobic_moment_figure(
    seq: str,
    moment_alpha: list,
    moment_beta: list,
    amphipathic_regions: list,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Hydrophobic moment profile for alpha-helix and beta-strand windows."""
    from beer.graphs._style import _residue_x
    n = len(seq)
    x = _residue_x(seq)
    ya = np.asarray(moment_alpha, dtype=float)
    yb = np.asarray(moment_beta, dtype=float)

    w = max(9, min(16, 9 + n * 0.015))
    fig = Figure(figsize=(w, 4), dpi=120)
    fig.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)

    ax.plot(x, ya, color="#4361ee", linewidth=1.4, label="α-helix (δ=100°)")
    ax.plot(x, yb, color="#f72585", linewidth=1.4, label="β-strand (δ=160°)")

    HM_THRESHOLD = 0.35
    ax.axhline(HM_THRESHOLD, color="#374151", linestyle="--", linewidth=0.9,
               label=f"Threshold ({HM_THRESHOLD})")

    y_max = max(np.max(ya), np.max(yb)) * 1.15 + 0.1
    for idx, region in enumerate(amphipathic_regions):
        start, end = region["start"], region["end"]
        ax.add_patch(Rectangle(
            (start - 0.5, 0), (end - start + 1), y_max,
            linewidth=0, facecolor="#43aa8b", alpha=0.18, zorder=0,
            label="Amphipathic" if idx == 0 else "_nolegend_",
        ))

    _pub_style_ax(ax, title="Hydrophobic Moment",
                  xlabel="Residue", ylabel="μH",
                  grid=True, title_size=label_font - 1,
                  label_size=label_font - 1, tick_size=tick_font - 1)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=tick_font - 2, loc="upper right",
              framealpha=0.85, edgecolor="#d0d4e0")
    fig.tight_layout(pad=1.5)
    return fig


def create_coiled_coil_profile_figure(
    cc_profile: list,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Per-residue coiled-coil propensity profile."""
    n = len(cc_profile)
    xs = list(range(1, n + 1))
    w = max(9, min(16, 9 + n * 0.015))
    fig = Figure(figsize=(w, 4), dpi=120)
    fig.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#fafbff")

    ax.plot(xs, cc_profile, color=_ACCENT, linewidth=1.4, zorder=3)
    ax.fill_between(xs, cc_profile, 0.5,
                    where=[v > 0.5 for v in cc_profile],
                    alpha=0.22, color=_ACCENT, zorder=2,
                    label="Above threshold")
    ax.axhline(0.5, color="#374151", linestyle="--", linewidth=0.9,
               alpha=0.7, label="Threshold (0.5)", zorder=4)
    _pub_style_ax(ax, title="Coiled-Coil Propensity",
                  xlabel="Residue", ylabel="CC Score",
                  grid=True, title_size=label_font - 1,
                  label_size=label_font - 1, tick_size=tick_font - 1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=tick_font - 3, framealpha=0.85,
              edgecolor="#d0d4e0", loc="upper right")
    ax.annotate(
        "Lupas et al. (1991) Science 252:1162 · Berger et al. (1995) PNAS 92:8259",
        xy=(0.01, 0.01), xycoords="axes fraction",
        fontsize=max(6, tick_font - 5), color="#6b7280", ha="left", va="bottom",
    )
    fig.tight_layout(pad=1.5)
    return fig
