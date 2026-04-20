"""Sequence map figures: linear map, domain architecture, cation-pi, complexity, annotation track."""
from __future__ import annotations

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator

from beer.graphs._style import (
    _pub_style_ax, _PALETTE, _ACCENT, _NEG_COL, _POS_COL,
)


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
    w = max(10, min(18, 10 + n * 0.015))
    fig = Figure(figsize=(w, 7), dpi=120, layout="constrained")
    fig.set_facecolor("#ffffff")
    axs = fig.subplots(3, 1, sharex=False)
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
        ax.set_ylabel(ylabel_txt, fontsize=tick_font - 1, color="#4a5568")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
        ax.tick_params(labelsize=tick_font - 2, length=3)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.set_facecolor("#fafbff")
        ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.5, color="#c8cdd8")
        ax.set_axisbelow(True)

    _track(axs[0], xs_win, hydro_profile, "#4361ee", 0, "Hydrophobicity")
    _track(axs[1], xs_win, ncpr_profile, "#7209b7", 0, "NCPR")
    _track(axs[2], xs_all, disorder_scores, "#f3722c", 0.5, "Disorder",
           fill_above=True, fill_below=False)
    axs[2].set_xlabel("Residue", fontsize=tick_font - 1, color="#4a5568")
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

    def _assign_lanes(regions):
        lane_ends = []
        lanes = []
        for (s, e) in regions:
            placed = False
            for li, end in enumerate(lane_ends):
                if s > end:
                    lane_ends[li] = e
                    lanes.append(li)
                    placed = True
                    break
            if not placed:
                lane_ends.append(e)
                lanes.append(len(lane_ends) - 1)
        return lanes

    lane_counts = []
    for (label, colour, regions, meta) in tracks:
        if regions:
            lane_counts.append(max(_assign_lanes(regions)) + 1)
        else:
            lane_counts.append(1)

    slot = 0.55
    gap  = 0.30

    track_y0 = []
    y_cursor = 0.0
    for lc in reversed(lane_counts):
        track_y0.insert(0, y_cursor + (lc - 1) * slot / 2.0)
        y_cursor += lc * slot + gap

    total_height = y_cursor - gap
    half = slot * 0.42
    legend_patches = []

    for tidx, (label, colour, regions, meta) in enumerate(tracks):
        ty_base = track_y0[tidx]
        lc = lane_counts[tidx]

        ax_ref = None  # assigned below after fig/ax creation

        if tidx == 0:
            _ty_first = ty_base
            _lc_first = lc

        if not regions:
            continue

        lanes = _assign_lanes(regions)

        if meta is not None:
            for i, (dom, (s, e), li) in enumerate(zip(meta, regions, lanes)):
                col = _PALETTE[i % len(_PALETTE)]
                legend_patches.append((col, dom["name"], ty_base, li))
        else:
            legend_patches.append((colour, label, None, None))

    # Build legend entries
    seen, unique_leg = set(), []
    raw_patches = []
    for item in legend_patches:
        col, lbl = item[0], item[1]
        if lbl not in seen:
            seen.add(lbl)
            raw_patches.append(Patch(color=col, label=lbl))
    legend_patches = raw_patches

    # Decide figure height based on legend size
    _MAX_LEGEND = 12
    n_extra = 0
    if len(legend_patches) > _MAX_LEGEND:
        n_extra = len(legend_patches) - _MAX_LEGEND
        legend_patches = legend_patches[:_MAX_LEGEND]
        legend_patches.append(Patch(color="none", label=f"(+{n_extra} more)"))

    many_legend = len(legend_patches) > 8
    leg_rows = math.ceil(len(legend_patches) / 5) if many_legend else 0
    extra_bottom = 0.06 + leg_rows * 0.06 if many_legend else 0.0

    fig_h = max(3.0, 1.5 + total_height * 1.4 + extra_bottom * 6)
    fig = Figure(figsize=(10, fig_h), dpi=120)
    fig.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#fafbff")

    for tidx, (label, colour, regions, meta) in enumerate(tracks):
        ty_base = track_y0[tidx]
        lc = lane_counts[tidx]

        ax.plot([1, seq_len], [ty_base, ty_base], color="#cbd5e0",
                linewidth=2.5, solid_capstyle="round", zorder=2)

        if tidx == 0:
            ax.text(1, ty_base + lc * slot * 0.5 + 0.05, "N",
                    ha="center", fontsize=tick_font - 3, color="#718096")
            ax.text(seq_len, ty_base + lc * slot * 0.5 + 0.05, "C",
                    ha="center", fontsize=tick_font - 3, color="#718096")

        if not regions:
            continue

        lanes = _assign_lanes(regions)

        if meta is not None:
            for i, (dom, (s, e), li) in enumerate(zip(meta, regions, lanes)):
                col = _PALETTE[i % len(_PALETTE)]
                w_rect = e - s + 1
                ty = ty_base + li * slot
                rect = Rectangle((s, ty - half), w_rect, 2 * half,
                                  color=col, alpha=0.85, zorder=4, linewidth=0)
                ax.add_patch(rect)
        else:
            for (s, e), li in zip(regions, lanes):
                w_rect = e - s + 1
                ty = ty_base + li * slot
                rect = Rectangle((s, ty - half), w_rect, 2 * half,
                                  color=colour, alpha=0.80, zorder=4, linewidth=0)
                ax.add_patch(rect)

    ax.set_yticks(track_y0)
    ax.set_yticklabels([t[0] for t in tracks],
                       fontsize=max(6, tick_font - 2), color="#4a5568",
                       fontweight="600")
    ax.tick_params(axis="y", length=0, pad=6)

    _pub_style_ax(ax,
                  title="Domain Architecture",
                  xlabel="Residue", ylabel="",
                  grid=False, title_size=label_font - 1,
                  label_size=label_font - 1, tick_size=tick_font - 1)
    ax.set_xlim(1, seq_len)
    ax.set_ylim(-half - 0.15, total_height + 0.25)

    if legend_patches:
        ncols = max(1, min(len(legend_patches), 5))
        leg_font = max(6, tick_font - 3)
        if many_legend:
            ax.legend(handles=legend_patches,
                      fontsize=leg_font, framealpha=0.85,
                      edgecolor="#d0d4e0",
                      loc="upper center",
                      bbox_to_anchor=(0.5, -0.18),
                      ncol=ncols,
                      handlelength=1.2, borderpad=0.5, labelspacing=0.3)
            fig.subplots_adjust(bottom=max(0.20, extra_bottom + 0.08))
        else:
            ncols = max(1, min(len(legend_patches), 4))
            ax.legend(handles=legend_patches,
                      fontsize=leg_font, framealpha=0.85,
                      edgecolor="#d0d4e0", loc="upper right", ncol=ncols,
                      handlelength=1.2, borderpad=0.5, labelspacing=0.3)
            fig.tight_layout(pad=1.8)
    else:
        fig.tight_layout(pad=1.8)
    return fig


def create_cation_pi_map_figure(
    seq: str,
    label_font: int = 14,
    tick_font: int = 12,
    cmap: str = "YlOrRd",
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
    dim = max(5.5, min(9.0, 5.0 + n * 0.025))
    fig = Figure(figsize=(dim + 1.2, dim + 0.4), dpi=120)
    fig.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#fafbff")
    im = ax.imshow(mat, cmap=cmap, aspect="auto", origin="upper",
                   interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, shrink=0.80, aspect=22, pad=0.03)
    cbar.set_label("Proximity (1/|i−j|)", fontsize=tick_font - 1, color="#4a5568")
    cbar.ax.tick_params(labelsize=tick_font - 2, colors="#4a5568")
    _pub_style_ax(ax,
                  title="Cation–π Proximity Map",
                  xlabel="Residue",
                  ylabel="Residue",
                  grid=False,
                  title_size=label_font - 1,
                  label_size=label_font - 1,
                  tick_size=tick_font - 1)
    fig.tight_layout(pad=1.8)
    return fig


def create_local_complexity_figure(
    entropy_profile: list,
    window_size: int,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Shannon entropy sliding-window local complexity profile."""
    n = len(entropy_profile)
    w = max(9, min(16, 9 + n * 0.015))
    fig = Figure(figsize=(w, 4.5), dpi=120)
    fig.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)
    xs = list(range(1, len(entropy_profile) + 1))
    ax.fill_between(xs, entropy_profile, 2.0,
                    where=[v < 2.0 for v in entropy_profile],
                    alpha=0.18, color=_NEG_COL, interpolate=True,
                    label="Low complexity")
    ax.plot(xs, entropy_profile, color=_ACCENT, linewidth=1.6,
            marker="o", markersize=3.0, markeredgewidth=0, zorder=4,
            label="Entropy")
    ax.axhline(2.0, color="#f72585", linewidth=1.2, linestyle="--",
               zorder=3, label="LC threshold (2.0 bits)")
    _pub_style_ax(ax,
                  title=f"Local Complexity  (w = {window_size})",
                  xlabel="Residue",
                  ylabel="Entropy (bits)",
                  grid=True,
                  title_size=label_font - 1,
                  label_size=label_font - 1,
                  tick_size=tick_font - 1)
    ax.legend(fontsize=tick_font - 2, framealpha=0.90,
              edgecolor="#d0d4e0", loc="lower right", borderpad=0.6)
    fig.tight_layout(pad=1.8)
    return fig


def create_annotation_track_figure(
    seq: str,
    disorder_scores: list,
    hydro_profile: list,
    aggr_profile: list,
    tm_helices: list,
    larks: list,
    sp_result: dict,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Genome-browser-style multi-track annotation overview figure."""
    n = len(seq)

    has_disorder = bool(disorder_scores)
    has_hydro    = bool(hydro_profile)
    has_aggr     = bool(aggr_profile)

    n_tracks = 5

    fig = Figure(figsize=(12, 9), dpi=120)
    fig.set_facecolor("#ffffff")
    height_ratios = [2, 2, 2, 1.6, 0.6]
    axs = fig.subplots(n_tracks, 1, sharex=True,
                       gridspec_kw={"height_ratios": height_ratios})
    fig.subplots_adjust(hspace=0.35, left=0.11, right=0.97, top=0.92, bottom=0.07)
    fig.suptitle("Feature Annotation Track", fontsize=label_font + 1,
                 fontweight="bold", color="#1a1a2e")

    def _style_track(ax, ylabel_txt, hide_xticks=True):
        ax.set_ylabel(ylabel_txt, fontsize=max(7, tick_font - 1), color="#4a5568",
                      labelpad=4, rotation=90, va="center")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
        ax.tick_params(labelsize=tick_font - 2, length=3, width=0.7,
                       colors="#4a5568")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.spines["left"].set_linewidth(0.7)
        ax.spines["left"].set_color("#c0c4d0")
        ax.spines["bottom"].set_linewidth(0.7)
        ax.spines["bottom"].set_color("#c0c4d0")
        ax.set_facecolor("#fafbff")
        ax.grid(True, axis="x", linestyle="--", linewidth=0.3,
                alpha=0.45, color="#c8cdd8")
        ax.set_axisbelow(True)
        if hide_xticks:
            ax.tick_params(axis="x", which="both", bottom=False,
                           labelbottom=False)

    # Track 1 – Disorder
    ax_dis = axs[0]
    if has_disorder:
        dis = np.asarray(disorder_scores, dtype=float)
        xs_dis = np.arange(1, len(dis) + 1, dtype=float)
        ax_dis.fill_between(xs_dis, np.where(dis > 0.5, dis, np.nan), 0,
                            color="#e63946", alpha=0.35, interpolate=True)
        ax_dis.fill_between(xs_dis, np.where(dis <= 0.5, dis, np.nan), 0,
                            color="#4361ee", alpha=0.35, interpolate=True)
        ax_dis.plot(xs_dis, dis, color="#333333", linewidth=0.8, zorder=3)
        ax_dis.axhline(0.5, color="#666666", linewidth=0.9,
                       linestyle=":", zorder=4)
        ax_dis.set_ylim(0, 1.05)
    else:
        ax_dis.text(0.5, 0.5, "No disorder data", transform=ax_dis.transAxes,
                    ha="center", va="center", fontsize=tick_font - 2, color="#aaa")
        ax_dis.set_ylim(0, 1)
    _style_track(ax_dis, "Disorder")

    # Track 2 – Hydrophobicity
    ax_hyd = axs[1]
    if has_hydro:
        hyd = np.asarray(hydro_profile, dtype=float)
        offset = (n - len(hyd)) // 2
        xs_hyd = np.arange(offset + 1, offset + len(hyd) + 1, dtype=float)
        ax_hyd.fill_between(xs_hyd, hyd, 0, where=(hyd >= 0),
                            color="#ff8800", alpha=0.35, interpolate=True)
        ax_hyd.fill_between(xs_hyd, hyd, 0, where=(hyd < 0),
                            color="#4361ee", alpha=0.35, interpolate=True)
        ax_hyd.plot(xs_hyd, hyd, color="#333333", linewidth=0.8, zorder=3)
        ax_hyd.axhline(0, color="#888888", linewidth=0.7, linestyle="--", zorder=4)
    else:
        ax_hyd.text(0.5, 0.5, "No hydrophobicity data",
                    transform=ax_hyd.transAxes,
                    ha="center", va="center", fontsize=tick_font - 2, color="#aaa")
    _style_track(ax_hyd, "Hydrophob.")

    # Track 3 – Aggregation
    ax_agg = axs[2]
    if has_aggr:
        agg = np.asarray(aggr_profile, dtype=float)
        xs_agg = np.arange(1, len(agg) + 1, dtype=float)
        ax_agg.fill_between(xs_agg, np.where(agg <= 1.0, agg, np.nan), 0,
                            color="#4682b4", alpha=0.35, interpolate=True)
        ax_agg.fill_between(xs_agg, np.where(agg > 1.0, agg, np.nan), 0,
                            color="#e63946", alpha=0.40, interpolate=True)
        ax_agg.plot(xs_agg, agg, color="#333333", linewidth=0.8, zorder=3)
        ax_agg.axhline(1.0, color="#888888", linewidth=0.7, linestyle=":", zorder=4)
    else:
        ax_agg.text(0.5, 0.5, "No aggregation data",
                    transform=ax_agg.transAxes,
                    ha="center", va="center", fontsize=tick_font - 2, color="#aaa")
    _style_track(ax_agg, "Aggregation")

    # Track 4 – Categorical features
    ax_feat = axs[3]
    ax_feat.set_ylim(0, 1)
    ax_feat.set_yticks([])

    for helix in (tm_helices or []):
        h_s = helix.get("start", 0)
        h_e = helix.get("end", h_s)
        h_s_1 = h_s + 1 if h_s == 0 or h_s < h_e else h_s
        h_e_1 = h_e + 1 if h_s == 0 or h_s < h_e else h_e
        w = max(1, h_e_1 - h_s_1 + 1)
        ax_feat.add_patch(Rectangle((h_s_1, 0.3), w, 0.4,
                                     color="#2d6a4f", alpha=0.85, zorder=4,
                                     linewidth=0))
        mid = h_s_1 + w / 2.0
        ax_feat.text(mid, 0.50, "TM", ha="center", va="center",
                     fontsize=max(5, tick_font - 5), color="white",
                     fontweight="bold", zorder=5)

    sp = sp_result or {}
    sp_start = sp.get("h_start")
    sp_end   = sp.get("h_end")
    if sp_start is None:
        clv = sp.get("cleavage_site")
        if clv is not None:
            sp_start, sp_end = 1, int(clv)
    if sp_start is not None and sp_end is not None:
        sp_w = max(1, int(sp_end) - int(sp_start) + 1)
        ax_feat.add_patch(Rectangle((int(sp_start), 0.55), sp_w, 0.35,
                                     color="#7b2d8b", alpha=0.80, zorder=3,
                                     linewidth=0))
        ax_feat.text(int(sp_start) + sp_w / 2.0, 0.725, "SP",
                     ha="center", va="center",
                     fontsize=max(5, tick_font - 5), color="white",
                     fontweight="bold", zorder=5)

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

    ax_feat.legend(handles=[
        Patch(color="#2d6a4f", alpha=0.85, label="TM helix"),
        Patch(color="#7b2d8b", alpha=0.80, label="Signal peptide"),
        Patch(color="#ff8800", alpha=0.85, label="LARKS"),
    ], fontsize=max(6, tick_font - 3), loc="upper right",
       framealpha=0.90, edgecolor="#d0d4e0",
       handlelength=1.2, handleheight=0.9, borderpad=0.5, labelspacing=0.3)

    _style_track(ax_feat, "Features")
    ax_feat.spines["left"].set_visible(False)

    # Track 5 – Ruler
    ax_rul = axs[4]
    ax_rul.set_ylim(0, 1)
    ax_rul.set_yticks([])
    for sp_name in ("top", "right", "left"):
        ax_rul.spines[sp_name].set_visible(False)
    ax_rul.spines["bottom"].set_linewidth(0.7)
    ax_rul.spines["bottom"].set_color("#c0c4d0")
    ax_rul.set_facecolor("#ffffff")
    ax_rul.tick_params(axis="x", labelsize=tick_font - 2, length=4,
                       width=0.7, colors="#4a5568")
    ax_rul.set_xlabel("Residue", fontsize=tick_font - 1, color="#4a5568", labelpad=3)
    tick_step = max(50, (n // 10 // 50) * 50) if n > 100 else 10
    major_ticks = list(range(1, n + 1, tick_step))
    if n not in major_ticks:
        major_ticks.append(n)
    ax_rul.set_xticks(major_ticks)

    for ax in axs:
        ax.set_xlim(1, n)

    return fig


def create_cleavage_map_figure(
    seq: str,
    cleavage_data: dict,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Proteolytic cleavage site map — one horizontal track per enzyme."""
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

    # Extra bottom space for the trypsin summary line
    fig_h = max(4.5, 1.3 * n_enzymes + 1.0)
    fig = Figure(figsize=(12, fig_h), dpi=120)
    fig.set_facecolor("#ffffff")
    fig.suptitle("Proteolytic Cleavage Map", fontsize=label_font + 1,
                 fontweight="bold", color="#1a1a2e", y=0.98)

    fig.subplots_adjust(left=0.14, right=0.88, top=0.88, bottom=0.18,
                        hspace=0.0)
    ax = fig.add_subplot(111)
    ax.set_facecolor("#fafbff")
    ax.set_xlim(0, n)
    ax.set_ylim(-0.5, n_enzymes - 0.5)

    for idx, enzyme in enumerate(enzymes):
        cuts  = cleavage_data[enzyme] or []
        color = _CLEAVAGE_PALETTE[idx % len(_CLEAVAGE_PALETTE)]
        y     = idx

        ax.plot([0, n], [y, y], color="#cccccc", linewidth=1.5,
                solid_capstyle="round", zorder=2)

        tick_h = 0.30
        for pos in cuts:
            ax.plot([pos, pos], [y - tick_h, y + tick_h],
                    color=color, linewidth=1.1, zorder=4,
                    solid_capstyle="round")

        n_cuts = len(cuts)
        ax.text(n + n * 0.01, y, f"{n_cuts}",
                va="center", ha="left",
                fontsize=tick_font - 2, color=color, fontweight="600")

    ax.set_yticks(list(range(n_enzymes)))
    ax.set_yticklabels(enzymes, fontsize=tick_font - 1, color="#4a5568")
    ax.tick_params(axis="y", length=0, pad=5)

    ax.set_xlabel("Residue", fontsize=label_font - 1, color="#2d3748", labelpad=5)
    tick_step = max(50, (n // 10 // 50) * 50) if n > 100 else 10
    major_ticks = list(range(0, n + 1, tick_step))
    if n not in major_ticks:
        major_ticks.append(n)
    ax.set_xticks(major_ticks)
    ax.tick_params(axis="x", labelsize=tick_font - 2, length=3,
                   width=0.7, colors="#4a5568")

    for sp_name in ("top", "right"):
        ax.spines[sp_name].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["left"].set_color("#c0c4d0")
    ax.spines["bottom"].set_linewidth(0.7)
    ax.spines["bottom"].set_color("#c0c4d0")

    ax.grid(True, axis="x", linestyle="--", linewidth=0.3,
            alpha=0.45, color="#c8cdd8")
    ax.set_axisbelow(True)

    ax.text(n + n * 0.01, n_enzymes - 0.5 + 0.1, "cuts",
            va="bottom", ha="left",
            fontsize=tick_font - 3, color="#888888")

    # Trypsin summary — placed inside the figure at axes fraction
    trypsin_key = next((k for k in enzymes if "trypsin" in k.lower()), None)
    if trypsin_key is not None:
        t_cuts = cleavage_data[trypsin_key] or []
        n_cuts = len(t_cuts)
        n_peptides = n_cuts + 1
        avg_len = round(n / n_peptides, 1) if n_peptides > 0 else 0
        summary = (
            f"Trypsin: {n_cuts} cut{'s' if n_cuts != 1 else ''}  "
            f"→  {n_peptides} peptide{'s' if n_peptides != 1 else ''}"
            f"  (avg {avg_len} aa)"
        )
        ax.annotate(summary,
                    xy=(0.5, -0.14), xycoords="axes fraction",
                    fontsize=tick_font - 1, color="#2d3748",
                    style="italic", ha="center", va="top")

    return fig
