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


def create_annotation_track_figure(
    seq: str,
    disorder_scores: list,
    hydro_profile: list,
    aggr_profile: list,
    ptm_sites: list,
    tm_helices: list,
    larks: list,
    sp_result: dict,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Genome-browser-style multi-track annotation overview figure.

    Five stacked horizontal tracks share the same x-axis (residue position):
      1. Disorder score filled-area plot.
      2. Hydrophobicity sliding-window filled-area plot.
      3. Aggregation propensity filled-area plot.
      4. Categorical features: TM helices, signal peptide, LARKS, PTM sites.
      5. Sequence ruler (x-axis only).
    """
    n = len(seq)
    xs_all = np.arange(1, n + 1, dtype=float)

    # Decide which continuous tracks are present
    has_disorder = bool(disorder_scores)
    has_hydro    = bool(hydro_profile)
    has_aggr     = bool(aggr_profile)

    n_tracks = 5  # fixed layout: disorder / hydro / aggr / features / ruler

    fig = Figure(figsize=(12, 8), dpi=120)
    fig.set_facecolor("#ffffff")
    # height_ratios: continuous tracks get more space; features track medium; ruler minimal
    height_ratios = [2, 2, 2, 1.4, 0.5]
    axs = fig.subplots(n_tracks, 1, sharex=True,
                       gridspec_kw={"height_ratios": height_ratios})
    fig.subplots_adjust(hspace=0.15, left=0.11, right=0.97, top=0.93, bottom=0.07)
    fig.suptitle("Feature Annotation Track", fontsize=label_font + 1,
                 fontweight="bold", color="#1a1a2e")

    # ------------------------------------------------------------------ #
    # Shared spine / tick helper
    # ------------------------------------------------------------------ #
    def _style_track(ax, ylabel_txt, hide_xticks=True):
        ax.set_ylabel(ylabel_txt, fontsize=tick_font - 1, color="#4a5568",
                      labelpad=4, rotation=90, va="center")
        ax.tick_params(labelsize=tick_font - 3, length=3, width=0.7,
                       colors="#4a5568")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.spines["left"].set_linewidth(0.7)
        ax.spines["left"].set_color("#c0c4d0")
        ax.spines["bottom"].set_linewidth(0.7)
        ax.spines["bottom"].set_color("#c0c4d0")
        ax.set_facecolor("#fafbff")
        ax.grid(True, axis="x", linestyle="--", linewidth=0.35,
                alpha=0.45, color="#c8cdd8")
        ax.set_axisbelow(True)
        if hide_xticks:
            ax.tick_params(axis="x", which="both", bottom=False,
                           labelbottom=False)

    # ------------------------------------------------------------------ #
    # Track 1 – Disorder
    # ------------------------------------------------------------------ #
    ax_dis = axs[0]
    if has_disorder:
        dis = np.asarray(disorder_scores, dtype=float)
        xs_dis = np.arange(1, len(dis) + 1, dtype=float)
        above = np.where(dis > 0.5, dis, np.nan)
        below = np.where(dis <= 0.5, dis, np.nan)
        ax_dis.fill_between(xs_dis, above, 0, color="#e63946",
                            alpha=0.35, interpolate=True)
        ax_dis.fill_between(xs_dis, below, 0, color="#4361ee",
                            alpha=0.35, interpolate=True)
        ax_dis.plot(xs_dis, dis, color="#333333", linewidth=0.8, zorder=3)
        ax_dis.axhline(0.5, color="#666666", linewidth=0.9,
                       linestyle=":", zorder=4)
        ax_dis.set_ylim(0, 1.05)
    else:
        ax_dis.text(0.5, 0.5, "No disorder data", transform=ax_dis.transAxes,
                    ha="center", va="center", fontsize=tick_font - 2,
                    color="#aaa")
        ax_dis.set_ylim(0, 1)
    _style_track(ax_dis, "Disorder")

    # ------------------------------------------------------------------ #
    # Track 2 – Hydrophobicity
    # ------------------------------------------------------------------ #
    ax_hyd = axs[1]
    if has_hydro:
        hyd = np.asarray(hydro_profile, dtype=float)
        offset = (n - len(hyd)) // 2
        xs_hyd = np.arange(offset + 1, offset + len(hyd) + 1, dtype=float)
        ax_hyd.fill_between(xs_hyd, hyd, 0,
                            where=(hyd >= 0), color="#ff8800",
                            alpha=0.35, interpolate=True)
        ax_hyd.fill_between(xs_hyd, hyd, 0,
                            where=(hyd < 0), color="#4361ee",
                            alpha=0.35, interpolate=True)
        ax_hyd.plot(xs_hyd, hyd, color="#333333", linewidth=0.8, zorder=3)
        ax_hyd.axhline(0, color="#888888", linewidth=0.7,
                       linestyle="--", zorder=4)
    else:
        ax_hyd.text(0.5, 0.5, "No hydrophobicity data",
                    transform=ax_hyd.transAxes,
                    ha="center", va="center", fontsize=tick_font - 2,
                    color="#aaa")
    _style_track(ax_hyd, "Hydrophob.")

    # ------------------------------------------------------------------ #
    # Track 3 – Aggregation
    # ------------------------------------------------------------------ #
    ax_agg = axs[2]
    if has_aggr:
        agg = np.asarray(aggr_profile, dtype=float)
        xs_agg = np.arange(1, len(agg) + 1, dtype=float)
        hot  = np.where(agg > 1.0, agg, np.nan)
        norm = np.where(agg <= 1.0, agg, np.nan)
        ax_agg.fill_between(xs_agg, norm, 0, color="#4682b4",
                            alpha=0.35, interpolate=True)
        ax_agg.fill_between(xs_agg, hot, 0, color="#e63946",
                            alpha=0.40, interpolate=True)
        ax_agg.plot(xs_agg, agg, color="#333333", linewidth=0.8, zorder=3)
        ax_agg.axhline(1.0, color="#888888", linewidth=0.7,
                       linestyle=":", zorder=4)
    else:
        ax_agg.text(0.5, 0.5, "No aggregation data",
                    transform=ax_agg.transAxes,
                    ha="center", va="center", fontsize=tick_font - 2,
                    color="#aaa")
    _style_track(ax_agg, "Aggregation")

    # ------------------------------------------------------------------ #
    # Track 4 – Categorical features
    # ------------------------------------------------------------------ #
    ax_feat = axs[3]
    ax_feat.set_ylim(0, 1)
    ax_feat.set_yticks([])

    # TM helices – dark green rectangles
    for helix in (tm_helices or []):
        h_s = helix.get("start", 0)
        h_e = helix.get("end", h_s)
        # tolerate 0-based or 1-based; normalise to 1-based width
        h_s_1 = h_s + 1 if h_s == 0 or h_s < h_e else h_s
        h_e_1 = h_e + 1 if h_s == 0 or h_s < h_e else h_e
        w = max(1, h_e_1 - h_s_1 + 1)
        rect = Rectangle((h_s_1, 0.3), w, 0.4,
                          color="#2d6a4f", alpha=0.85, zorder=4,
                          linewidth=0)
        ax_feat.add_patch(rect)
        mid = h_s_1 + w / 2.0
        ax_feat.text(mid, 0.50, "TM", ha="center", va="center",
                     fontsize=max(5, tick_font - 5), color="white",
                     fontweight="bold", zorder=5)

    # Signal peptide – violet rectangle
    sp = sp_result or {}
    sp_start = sp.get("h_start", None)
    sp_end   = sp.get("h_end",   None)
    if sp_start is None:
        clv = sp.get("cleavage_site", None)
        if clv is not None:
            sp_start, sp_end = 1, int(clv)
    if sp_start is not None and sp_end is not None:
        sp_w = max(1, int(sp_end) - int(sp_start) + 1)
        rect_sp = Rectangle((int(sp_start), 0.55), sp_w, 0.35,
                             color="#7b2d8b", alpha=0.80, zorder=3,
                             linewidth=0)
        ax_feat.add_patch(rect_sp)
        ax_feat.text(int(sp_start) + sp_w / 2.0, 0.725, "SP",
                     ha="center", va="center",
                     fontsize=max(5, tick_font - 5), color="white",
                     fontweight="bold", zorder=5)

    # LARKS – orange vertical tick marks at center position
    for lark in (larks or []):
        if isinstance(lark, dict):
            lk_s = lark.get("start", lark.get("position", 1))
            lk_e = lark.get("end", lk_s)
            center = (int(lk_s) + int(lk_e)) / 2.0
        else:
            center = float(lark)
        ax_feat.plot([center, center], [0.05, 0.25],
                     color="#ff8800", linewidth=1.4, zorder=4,
                     solid_capstyle="round")

    # PTM sites – small colored diamonds
    for site in (ptm_sites or []):
        pos = site.get("position", 1)
        pt  = site.get("ptm_type", "unknown")
        col = PTM_COLORS.get(pt, "#7f7f7f")
        ax_feat.scatter([pos], [0.15], marker="D", s=28, color=col,
                        edgecolors="#333333", linewidths=0.4, zorder=5)

    _style_track(ax_feat, "Features")
    # hide y spine for feature track
    ax_feat.spines["left"].set_visible(False)

    # ------------------------------------------------------------------ #
    # Track 5 – Ruler (x-axis only)
    # ------------------------------------------------------------------ #
    ax_rul = axs[4]
    ax_rul.set_ylim(0, 1)
    ax_rul.set_yticks([])
    for sp_name in ("top", "right", "left"):
        ax_rul.spines[sp_name].set_visible(False)
    ax_rul.spines["bottom"].set_linewidth(0.7)
    ax_rul.spines["bottom"].set_color("#c0c4d0")
    ax_rul.set_facecolor("#ffffff")
    ax_rul.tick_params(axis="x", labelsize=tick_font - 3, length=4,
                       width=0.7, colors="#4a5568")
    ax_rul.set_xlabel("Residue Position", fontsize=tick_font - 1,
                      color="#4a5568", labelpad=3)
    # tick every 50 residues
    tick_step = 50
    major_ticks = list(range(1, n + 1, tick_step))
    if n not in major_ticks:
        major_ticks.append(n)
    ax_rul.set_xticks(major_ticks)

    # Shared x limits
    for ax in axs:
        ax.set_xlim(1, n)

    return fig


def create_cleavage_map_figure(
    seq: str,
    cleavage_data: dict,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Proteolytic cleavage site map — one horizontal track per enzyme.

    cleavage_data: dict of enzyme_name -> list of 1-based cut positions,
    as returned by calc_proteolytic_sites(seq).
    """
    _CLEAVAGE_PALETTE = [
        "#4361ee", "#e63946", "#2a9d8f", "#e9c46a",
        "#f4a261", "#6d6875", "#264653", "#023047",
    ]

    if not cleavage_data:
        fig = Figure(figsize=(12, 4), dpi=120)
        fig.set_facecolor("#ffffff")
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No cleavage data available",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=label_font - 1, color="#aaa")
        ax.set_axis_off()
        fig.suptitle("Proteolytic Cleavage Map", fontsize=label_font + 1,
                     fontweight="bold", color="#1a1a2e")
        return fig

    n = len(seq)
    enzymes   = list(cleavage_data.keys())
    n_enzymes = len(enzymes)

    fig_h = max(4.0, 1.2 * n_enzymes)
    fig = Figure(figsize=(12, fig_h), dpi=120)
    fig.set_facecolor("#ffffff")
    fig.suptitle("Proteolytic Cleavage Map", fontsize=label_font + 1,
                 fontweight="bold", color="#1a1a2e", y=0.98)

    # Reserve bottom space for the trypsin summary annotation
    fig.subplots_adjust(left=0.14, right=0.88, top=0.90, bottom=0.14,
                        hspace=0.0)
    ax = fig.add_subplot(111)
    ax.set_facecolor("#fafbff")
    ax.set_xlim(0, n)
    ax.set_ylim(-0.5, n_enzymes - 0.5)

    for idx, enzyme in enumerate(enzymes):
        cuts  = cleavage_data[enzyme] or []
        color = _CLEAVAGE_PALETTE[idx % len(_CLEAVAGE_PALETTE)]
        y     = idx  # one integer y level per enzyme

        # Gray backbone line
        ax.plot([0, n], [y, y], color="#cccccc", linewidth=1.5,
                solid_capstyle="round", zorder=2)

        # Vertical tick marks at each cut position
        tick_h = 0.30
        for pos in cuts:
            ax.plot([pos, pos], [y - tick_h, y + tick_h],
                    color=color, linewidth=1.1, zorder=4,
                    solid_capstyle="round")

        # Cut count label on the right
        n_cuts = len(cuts)
        ax.text(n + n * 0.01, y, f"{n_cuts}",
                va="center", ha="left",
                fontsize=tick_font - 2, color=color, fontweight="600")

    # y-axis: enzyme names
    ax.set_yticks(list(range(n_enzymes)))
    ax.set_yticklabels(enzymes, fontsize=tick_font - 1, color="#4a5568")
    ax.tick_params(axis="y", length=0, pad=5)

    # x-axis styling
    ax.set_xlabel("Residue Position", fontsize=label_font - 1,
                  color="#2d3748", labelpad=5)
    tick_step = 50
    major_ticks = list(range(0, n + 1, tick_step))
    if n not in major_ticks:
        major_ticks.append(n)
    ax.set_xticks(major_ticks)
    ax.tick_params(axis="x", labelsize=tick_font - 2, length=3,
                   width=0.7, colors="#4a5568")

    # Spine cleanup
    for sp_name in ("top", "right"):
        ax.spines[sp_name].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["left"].set_color("#c0c4d0")
    ax.spines["bottom"].set_linewidth(0.7)
    ax.spines["bottom"].set_color("#c0c4d0")

    # Subtle vertical grid
    ax.grid(True, axis="x", linestyle="--", linewidth=0.35,
            alpha=0.45, color="#c8cdd8")
    ax.set_axisbelow(True)

    # Right-edge label header
    ax.text(n + n * 0.01, n_enzymes - 0.5 + 0.1, "cuts",
            va="bottom", ha="left",
            fontsize=tick_font - 3, color="#888888")

    # Bottom summary for trypsin (most commonly used enzyme)
    trypsin_key = next(
        (k for k in enzymes if "trypsin" in k.lower()), None
    )
    if trypsin_key is not None:
        t_cuts = cleavage_data[trypsin_key] or []
        n_cuts = len(t_cuts)
        n_peptides = n_cuts + 1
        avg_len = round(n / n_peptides, 1) if n_peptides > 0 else 0
        summary = (
            f"Trypsin: {n_cuts} cut{'s' if n_cuts != 1 else ''}  "
            f"\u2192  {n_peptides} peptide{'s' if n_peptides != 1 else ''}"
            f"  (avg {avg_len} aa)"
        )
        fig.text(0.50, 0.03, summary, ha="center", va="bottom",
                 fontsize=tick_font - 1, color="#2d3748",
                 style="italic")

    return fig
