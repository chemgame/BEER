"""Sequence map figures: linear map, PTM, domain architecture, cation-pi, complexity."""
from __future__ import annotations

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle

from beer.graphs._style import (
    _pub_style_ax, _PALETTE, _ACCENT, _NEG_COL, _POS_COL,
)

# PTM colour palette
PTM_COLORS = {
    "phospho": "#1f77b4",
    "glycosylation": "#2ca02c",
    "ubiquitination": "#ff7f0e",
    "sumo": "#9467bd",
    "acetylation": "#17becf",
    "methylation": "#e377c2",
    "palmitoylation": "#8c564b",
}

# Sticker sets (needed locally)
_STICKER_AROMATIC = set("FWY")


def create_linear_sequence_map_figure(
    seq: str,
    hydro_profile: list,
    ncpr_profile: list,
    disorder_scores: list,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """3-track linear sequence map: hydrophobicity, NCPR, and disorder."""
    n = len(seq)
    xs_win = list(range(1, len(hydro_profile) + 1))
    xs_all = list(range(1, n + 1))
    fig = Figure(figsize=(10, 6), dpi=120)
    fig.set_facecolor("#ffffff")
    axs = fig.subplots(3, 1, sharex=False)
    fig.subplots_adjust(hspace=0.55, left=0.10, right=0.97, top=0.93, bottom=0.08)
    fig.suptitle("Linear Sequence Map", fontsize=label_font + 1,
                 fontweight="bold", color="#1a1a2e")

    def _track(ax, xs, ys, col, zero, ylabel_txt, fill_above=True, fill_below=True):
        if fill_above:
            ax.fill_between(xs, ys, zero,
                            where=[v >= zero for v in ys],
                            alpha=0.28, color=col, interpolate=True)
        if fill_below:
            ax.fill_between(xs, ys, zero,
                            where=[v < zero for v in ys],
                            alpha=0.28, color="#f72585", interpolate=True)
        ax.plot(xs, ys, color=col, linewidth=1.3)
        ax.axhline(zero, color="#aaa", linewidth=0.6, linestyle="--")
        ax.set_ylabel(ylabel_txt, fontsize=tick_font - 2, color="#4a5568")
        ax.tick_params(labelsize=tick_font - 3, length=3)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.set_facecolor("#fafbff")
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, color="#c8cdd8")
        ax.set_axisbelow(True)

    _track(axs[0], xs_win, hydro_profile, "#4361ee", 0, "Hydrophobicity")
    _track(axs[1], xs_win, ncpr_profile, "#7209b7", 0, "NCPR")
    _track(axs[2], xs_all, disorder_scores, "#f3722c", 0.5, "Disorder",
           fill_above=True, fill_below=False)
    axs[2].set_xlabel("Residue Position", fontsize=tick_font - 1, color="#4a5568")
    return fig


def create_ptm_profile_figure(
    seq: str,
    ptm_sites: list,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Lollipop/stem plot of predicted PTM sites."""
    import matplotlib.patches as mpatches

    ptm_types_present = []
    for site in ptm_sites:
        pt = site.get("ptm_type", "unknown")
        if pt not in ptm_types_present:
            ptm_types_present.append(pt)

    ptm_y_map = {pt: i for i, pt in enumerate(ptm_types_present)}
    n_types = max(len(ptm_types_present), 1)

    fig = Figure(figsize=(12, max(3, n_types * 1.2 + 1.5)), tight_layout=True)
    ax = fig.add_subplot(111)

    for site in ptm_sites:
        pos = site.get("position", 1)
        pt = site.get("ptm_type", "unknown")
        y_idx = ptm_y_map.get(pt, 0)
        color = PTM_COLORS.get(pt, "#7f7f7f")
        ax.plot([pos, pos], [y_idx - 0.35, y_idx], color=color,
                linewidth=1.0, zorder=2)
        ax.scatter([pos], [y_idx], color=color, s=60, zorder=3,
                   edgecolors="black", linewidths=0.4)

    ax.set_yticks(list(ptm_y_map.values()))
    ax.set_yticklabels(list(ptm_y_map.keys()), fontsize=tick_font)
    ax.set_ylim(-0.8, n_types - 0.2)

    ax.set_xlabel("Residue Position", fontsize=label_font)
    ax.set_ylabel("PTM Type", fontsize=label_font)
    ax.set_title("Post-Translational Modification Sites (Predicted)", fontsize=label_font)
    ax.set_xlim(0, len(seq) + 1)
    ax.tick_params(axis="x", labelsize=tick_font)

    legend_patches = [
        mpatches.Patch(color=PTM_COLORS.get(pt, "#7f7f7f"), label=pt)
        for pt in ptm_types_present
    ]
    if legend_patches:
        ax.legend(handles=legend_patches, fontsize=tick_font,
                  loc="upper right", title="PTM Type",
                  title_fontsize=tick_font)

    return fig


def create_domain_architecture_figure(
    seq_len: int,
    domains: list,
    seq: str = "",
    disorder_scores=None,
    tm_helices=None,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Multi-track domain architecture figure."""
    def _lc_mask(s, window=12, thr=2.0):
        n = len(s)
        covered = [False] * n
        for i in range(n - min(window, n) + 1):
            win = s[i:i + window]
            counts = {}
            for aa in win:
                counts[aa] = counts.get(aa, 0) + 1
            L = len(win)
            ent = -sum((c / L) * math.log2(c / L) for c in counts.values() if c > 0)
            if ent < thr:
                for j in range(i, min(i + window, n)):
                    covered[j] = True
        return covered

    def _runs(bools):
        segs, in_seg, start = [], False, 0
        for i, v in enumerate(bools):
            if v and not in_seg:
                in_seg, start = True, i + 1
            elif not v and in_seg:
                segs.append((start, i))
                in_seg = False
        if in_seg:
            segs.append((start, len(bools)))
        return segs

    tracks = []

    if domains:
        pfam_regions = [(d["start"], d["end"]) for d in domains]
        tracks.append(("Pfam Domains", None, pfam_regions, domains))

    if disorder_scores is not None and len(disorder_scores) == seq_len:
        dis_runs = _runs([v > 0.5 for v in disorder_scores])
        tracks.append(("Disorder", "#f3722c", dis_runs, None))

    if seq and len(seq) == seq_len:
        lc_runs = _runs(_lc_mask(seq))
        tracks.append(("Low Complexity", "#a8dadc", lc_runs, None))

    if tm_helices:
        tm_regions = [(h["start"] + 1, h["end"] + 1) for h in tm_helices]
        tracks.append(("TM Helices", "#6a4c93", tm_regions, None))

    if not tracks:
        tracks.append(("Sequence", "#94a3b8", [], None))

    n_tracks = len(tracks)
    fig_h = max(2.5, 1.2 + n_tracks * 1.0)
    fig = Figure(figsize=(10, fig_h), dpi=120)
    fig.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#fafbff")

    track_ys = list(range(n_tracks - 1, -1, -1))
    half = 0.32
    legend_patches = []

    for tidx, (label, colour, regions, meta) in enumerate(tracks):
        ty = track_ys[tidx]
        ax.plot([1, seq_len], [ty, ty], color="#cbd5e0", linewidth=2.5,
                solid_capstyle="round", zorder=2)
        if tidx == 0:
            ax.text(1, ty + half + 0.06, "N",
                    ha="center", fontsize=tick_font - 3, color="#718096")
            ax.text(seq_len, ty + half + 0.06, "C",
                    ha="center", fontsize=tick_font - 3, color="#718096")

        if meta is not None:
            for i, (dom, (s, e)) in enumerate(zip(meta, regions)):
                col = _PALETTE[i % len(_PALETTE)]
                w = e - s + 1
                rect = Rectangle((s, ty - half), w, 2 * half,
                                  color=col, alpha=0.85, zorder=4, linewidth=0)
                ax.add_patch(rect)
                mid = (s + e) / 2.0
                ax.text(mid, ty, dom["name"][:14],
                        ha="center", va="center",
                        fontsize=max(5, tick_font - 5), color="white",
                        fontweight="bold", zorder=5)
                legend_patches.append(Patch(color=col, label=dom["name"]))
        else:
            for (s, e) in regions:
                w = e - s + 1
                rect = Rectangle((s, ty - half), w, 2 * half,
                                  color=colour, alpha=0.80, zorder=4, linewidth=0)
                ax.add_patch(rect)
            if regions:
                legend_patches.append(Patch(color=colour, label=label))

    ax.set_yticks(track_ys)
    ax.set_yticklabels([t[0] for t in tracks],
                       fontsize=max(6, tick_font - 3), color="#4a5568",
                       fontweight="600")
    ax.tick_params(axis="y", length=0, pad=6)

    _pub_style_ax(ax,
                  title="Domain Architecture",
                  xlabel="Residue Position", ylabel="",
                  grid=False, title_size=label_font + 1,
                  label_size=label_font - 1, tick_size=tick_font - 1)
    ax.set_xlim(1, seq_len)
    ax.set_ylim(-0.7, n_tracks - 0.3)
    ax.set_yticks(track_ys)
    ax.set_yticklabels([t[0] for t in tracks],
                       fontsize=max(6, tick_font - 3), color="#4a5568",
                       fontweight="600")
    ax.tick_params(axis="y", length=0, pad=6)

    if legend_patches:
        ax.legend(handles=legend_patches,
                  fontsize=max(6, tick_font - 4), framealpha=0.85,
                  edgecolor="#d0d4e0", loc="upper right",
                  ncol=max(1, len(legend_patches) // 6))
    fig.tight_layout(pad=1.5)
    return fig


def create_cation_pi_map_figure(
    seq: str,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Cation-pi proximity heatmap."""
    n = len(seq)
    window = 8
    mat = np.zeros((n, n))
    arom = _STICKER_AROMATIC
    basic = set("KR")
    for i in range(n):
        if seq[i] in basic or seq[i] in arom:
            for j in range(max(0, i - window), min(n, i + window + 1)):
                if j != i:
                    is_cp = (seq[i] in basic and seq[j] in arom) or \
                            (seq[i] in arom and seq[j] in basic)
                    if is_cp:
                        mat[i, j] = 1.0 / abs(i - j)
    fig = Figure(figsize=(6, 5), dpi=120)
    fig.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#fafbff")
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto", origin="upper",
                   interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, aspect=20, pad=0.02)
    cbar.set_label("Proximity (1 / distance)", fontsize=tick_font - 1, color="#4a5568")
    cbar.ax.tick_params(labelsize=tick_font - 2, colors="#4a5568")
    _pub_style_ax(ax,
                  title="Cation\u2013\u03c0 Proximity Map",
                  xlabel="Residue Position",
                  ylabel="Residue Position",
                  grid=False,
                  title_size=label_font + 1,
                  label_size=label_font - 1,
                  tick_size=tick_font - 1)
    fig.tight_layout(pad=1.5)
    return fig


def create_local_complexity_figure(
    entropy_profile: list,
    window_size: int,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Shannon entropy sliding-window local complexity profile."""
    fig = Figure(figsize=(9, 4), dpi=120)
    fig.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)
    xs = list(range(1, len(entropy_profile) + 1))
    ax.fill_between(xs, entropy_profile, 2.0,
                    where=[v < 2.0 for v in entropy_profile],
                    alpha=0.18, color=_NEG_COL, interpolate=True,
                    label="Low complexity region")
    ax.plot(xs, entropy_profile, color=_ACCENT, linewidth=1.8,
            marker="o", markersize=3.5, markeredgewidth=0, zorder=4,
            label="Entropy")
    ax.axhline(2.0, color="#f72585", linewidth=1.2, linestyle="--",
               zorder=3, label="LC threshold (2.0 bits)")
    _pub_style_ax(ax,
                  title=f"Local Complexity  (window = {window_size})",
                  xlabel="Residue Position",
                  ylabel="Shannon Entropy (bits)",
                  grid=True,
                  title_size=label_font + 1,
                  label_size=label_font - 1,
                  tick_size=tick_font - 1)
    ax.legend(fontsize=tick_font - 2, framealpha=0.85, edgecolor="#d0d4e0")
    fig.tight_layout(pad=1.5)
    import mplcursors
    mplcursors.cursor(ax)
    return fig
