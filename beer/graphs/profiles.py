"""Per-residue profile figures (hydrophobicity, aggregation, solubility, etc.)."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from beer.graphs._style import (
    _pub_style_ax, _apply_font_sizes, _residue_x,
    _ACCENT, _NEG_COL,
)
from beer.constants import HYDROPHOBICITY_SCALES

AGGREGATION_THRESHOLD = 1.0
SOLUBILITY_NEUTRAL = 0.0
HM_THRESHOLD = 0.35
RBP_THRESHOLD = 0.3


def _maybe_downsample(x_arr, y_arr, max_pts: int = 800):
    """Thin (x, y) arrays to at most max_pts points using uniform stride."""
    if len(y_arr) <= max_pts:
        return x_arr, y_arr
    stride = max(1, len(y_arr) // max_pts)
    return x_arr[::stride], y_arr[::stride]


def _adaptive_width(n: int, base: float = 9.0, scale: float = 0.015,
                    lo: float = 9.0, hi: float = 16.0) -> float:
    """Return figure width that grows gently with sequence length."""
    return max(lo, min(hi, base + n * scale))


def create_hydrophobicity_figure(
    hydro_profile: list,
    window_size: int,
    scale_name: str = "Kyte-Doolittle",
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Sliding-window hydrophobicity plot."""
    n = len(hydro_profile)
    fig = Figure(figsize=(_adaptive_width(n), 4), dpi=120)
    fig.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)
    xs = np.arange(1, n + 1, dtype=float)
    ys = np.asarray(hydro_profile, dtype=float)
    xs, ys = _maybe_downsample(xs, ys)
    ax.fill_between(xs, ys, 0, where=(ys >= 0),
                    alpha=0.18, color=_ACCENT, interpolate=True)
    ax.fill_between(xs, ys, 0, where=(ys < 0),
                    alpha=0.18, color=_NEG_COL, interpolate=True)
    ax.plot(xs, ys, color=_ACCENT, linewidth=1.6,
            marker="o", markersize=3.0, markerfacecolor=_ACCENT,
            markeredgewidth=0, zorder=4)
    ax.axhline(0, color="#888", linewidth=0.7, linestyle="--", zorder=3)
    _ylabel = HYDROPHOBICITY_SCALES.get(scale_name, {}).get("ylabel", "Score")
    _pub_style_ax(ax,
                  title=f"Hydrophobicity  (w={window_size},  {scale_name})",
                  xlabel="Residue", ylabel=_ylabel,
                  grid=True, title_size=label_font - 1,
                  label_size=label_font - 1, tick_size=tick_font - 1)
    fig.tight_layout(pad=1.5)
    return fig


def create_aggregation_profile_figure(
    seq: str,
    aggregation_profile: list,
    hotspots: list,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Line plot of per-residue beta-aggregation propensity (Zyggregator)."""
    n = len(seq)
    x = _residue_x(seq)
    y = np.asarray(aggregation_profile, dtype=float)
    x, y = _maybe_downsample(x, y)

    fig = Figure(figsize=(_adaptive_width(n), 4), dpi=120)
    fig.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)

    ax.plot(x, y, color="#4682b4", linewidth=1.4, label="Aggregation")
    ax.fill_between(x, AGGREGATION_THRESHOLD, y, where=(y > AGGREGATION_THRESHOLD),
                    interpolate=True, color="#f3722c", alpha=0.50,
                    label=f"Hotspot (>{AGGREGATION_THRESHOLD})")
    ax.axhline(AGGREGATION_THRESHOLD, color="#374151", linestyle="--",
               linewidth=0.9, label=f"Threshold ({AGGREGATION_THRESHOLD})")

    y_min, y_max = float(y.min()), float(y.max()) * 1.1 + 0.2
    for idx, hs in enumerate(hotspots):
        start, end = hs[0], hs[1]
        rect = Rectangle((start - 0.5, y_min), (end - start + 1), y_max - y_min,
                          linewidth=0, facecolor="#f72585", alpha=0.18, zorder=0,
                          label="Hotspot" if idx == 0 else "_nolegend_")
        ax.add_patch(rect)

    _pub_style_ax(ax, title="β-Aggregation Propensity",
                  xlabel="Residue", ylabel="Aggregation",
                  grid=True, title_size=label_font - 1,
                  label_size=label_font - 1, tick_size=tick_font - 1)
    ax.set_xlim(x[0], x[-1])
    ax.legend(fontsize=tick_font - 2, loc="upper right",
              framealpha=0.85, edgecolor="#d0d4e0")
    fig.tight_layout(pad=1.5)
    return fig


def create_solubility_profile_figure(
    seq: str,
    camsolmt_profile: list,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Per-residue CamSol intrinsic solubility profile."""
    n = len(seq)
    x = _residue_x(seq)
    y = np.asarray(camsolmt_profile, dtype=float)
    x, y = _maybe_downsample(x, y)

    fig = Figure(figsize=(_adaptive_width(n), 4), dpi=120)
    fig.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)

    ax.plot(x, y, color="#2d3748", linewidth=1.2)
    ax.fill_between(x, 0, y, where=(y >= 0), interpolate=True,
                    color="#43aa8b", alpha=0.45, label="Soluble (>0)")
    ax.fill_between(x, 0, y, where=(y < 0), interpolate=True,
                    color="#f72585", alpha=0.45, label="Insoluble (<0)")
    ax.axhline(0.0, color="#888", linestyle="--", linewidth=0.8)

    _pub_style_ax(ax, title="CamSol Solubility",
                  xlabel="Residue", ylabel="CamSol Score",
                  grid=True, title_size=label_font - 1,
                  label_size=label_font - 1, tick_size=tick_font - 1)
    ax.set_xlim(x[0], x[-1])
    ax.legend(fontsize=tick_font - 2, loc="upper right",
              framealpha=0.85, edgecolor="#d0d4e0")
    fig.tight_layout(pad=1.5)
    return fig


def create_scd_profile_figure(
    seq: str,
    scd_profile: list,
    window: int,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Sliding-window SCD (Sequence Charge Decoration) profile."""
    n = len(scd_profile)
    x = np.arange(1, n + 1, dtype=float)
    y = np.asarray(scd_profile, dtype=float)
    x, y = _maybe_downsample(x, y)

    fig = Figure(figsize=(_adaptive_width(n), 4), dpi=120)
    fig.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)

    ax.plot(x, y, color="#2d3748", linewidth=1.2)
    ax.fill_between(x, 0, y, where=(y > 0), interpolate=True,
                    color="#f72585", alpha=0.45, label="Segregated (+)")
    ax.fill_between(x, 0, y, where=(y < 0), interpolate=True,
                    color="#4361ee", alpha=0.45, label="Mixed (−)")
    ax.axhline(0.0, color="#888", linestyle="--", linewidth=0.8)

    _pub_style_ax(ax, title=f"Charge Decoration  (w={window})",
                  xlabel="Residue", ylabel="SCD",
                  grid=True, title_size=label_font - 1,
                  label_size=label_font - 1, tick_size=tick_font - 1)
    ax.set_xlim(x[0], x[-1])
    ax.legend(fontsize=tick_font - 2, loc="upper right",
              framealpha=0.85, edgecolor="#d0d4e0")
    fig.tight_layout(pad=1.5)
    return fig


def create_rbp_profile_figure(
    seq: str,
    rbp_profile: list,
    rbp_motifs: list,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Sliding-window RNA-binding propensity profile."""
    n = len(seq)
    x = _residue_x(seq)
    y = np.asarray(rbp_profile, dtype=float)
    if len(x) != len(y):
        x = x[:len(y)]
    x, y = _maybe_downsample(x, y)

    fig = Figure(figsize=(_adaptive_width(n), 4), dpi=120)
    fig.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)

    ax.plot(x, y, color="#2a9d8f", linewidth=1.4, label="RBP propensity")
    ax.fill_between(x, 0, y, where=(y > 0), interpolate=True,
                    color="#2a9d8f", alpha=0.22)

    motif_colors = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8",
                    "#f58231", "#911eb4", "#42d4f4", "#f032e6"]
    for i, motif in enumerate(rbp_motifs):
        start = motif.get("start", 1)
        end = motif.get("end", start)
        mcolor = motif.get("color", motif_colors[i % len(motif_colors)])
        mname = motif.get("motif_name", f"Motif {i+1}")
        ax.axvspan(start - 0.5, end + 0.5, color=mcolor, alpha=0.22,
                   label=mname, zorder=0)

    _pub_style_ax(ax, title="RNA-Binding Propensity",
                  xlabel="Residue", ylabel="RBP Score",
                  grid=True, title_size=label_font - 1,
                  label_size=label_font - 1, tick_size=tick_font - 1)
    ax.set_xlim(x[0], x[-1])
    ax.legend(fontsize=tick_font - 2, loc="upper right",
              framealpha=0.85, edgecolor="#d0d4e0")
    fig.tight_layout(pad=1.5)
    return fig


def create_disorder_profile_figure(
    disorder_scores: list,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """IUPred-style per-residue disorder score plot."""
    n = len(disorder_scores)
    xs = list(range(1, n + 1))
    fig = Figure(figsize=(_adaptive_width(n), 4), dpi=120)
    fig.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)
    ax.fill_between(xs, disorder_scores, 0.5,
                    where=[v > 0.5 for v in disorder_scores],
                    alpha=0.28, color="#f3722c", interpolate=True,
                    label="Disordered (>0.5)")
    ax.fill_between(xs, disorder_scores, 0.5,
                    where=[v <= 0.5 for v in disorder_scores],
                    alpha=0.12, color="#4361ee", interpolate=True,
                    label="Ordered (≤0.5)")
    ax.plot(xs, disorder_scores, color="#f3722c", linewidth=1.6,
            marker="o", markersize=2.5, markeredgewidth=0, zorder=4)
    ax.axhline(0.5, color="#888", linewidth=0.9, linestyle="--",
               zorder=3, label="Threshold (0.5)")
    _pub_style_ax(ax, title="Disorder Profile",
                  xlabel="Residue", ylabel="Disorder Score",
                  grid=True, title_size=label_font - 1,
                  label_size=label_font - 1, tick_size=tick_font - 1)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=tick_font - 2, framealpha=0.85, edgecolor="#d0d4e0")
    fig.tight_layout(pad=1.5)
    return fig


def create_plaac_profile_figure(
    plaac_data: dict,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Per-residue PLAAC log-odds profile (Lancaster et al. 2014)."""
    profile = plaac_data.get("profile", [])
    regions = plaac_data.get("prion_like_regions", [])
    n = len(profile)
    if n == 0:
        fig = Figure(figsize=(9, 3), dpi=120)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No PLAAC data", ha="center", va="center",
                transform=ax.transAxes, color="#718096")
        return fig

    xs = list(range(1, n + 1))
    fig = Figure(figsize=(_adaptive_width(n, base=9.0, scale=0.012), 3.5), dpi=120)
    fig.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)

    pos = [v if v > 0 else 0 for v in profile]
    neg = [v if v < 0 else 0 for v in profile]
    ax.fill_between(xs, pos, 0, alpha=0.45, color="#e63946", label="Prion-like (>0)")
    ax.fill_between(xs, neg, 0, alpha=0.35, color="#4361ee", label="Background (<0)")
    ax.plot(xs, profile, color="#2d3748", linewidth=0.8, alpha=0.7)
    ax.axhline(0, color="#aaa", linewidth=0.8, linestyle="--")

    for r in regions:
        ax.axvspan(r["start_1based"], r["end_1based"],
                   alpha=0.12, color="#e63946", zorder=0)

    _pub_style_ax(ax, title="PLAAC Prion-like Profile",
                  xlabel="Residue", ylabel="Log-odds",
                  grid=True, title_size=label_font - 1,
                  label_size=label_font - 1, tick_size=tick_font - 1)
    ax.set_xlim(1, n)
    ax.legend(fontsize=tick_font - 2, framealpha=0.85, edgecolor="#d0d4e0")
    fig.tight_layout(pad=1.5)
    return fig


def create_tango_figure(
    seq: str,
    tango_profile: list,
    hotspots: list,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """TANGO-style β-aggregation profile (0–100 %)."""
    n = len(seq)
    if not tango_profile:
        tango_profile = [0.0] * n
    x = np.arange(1, len(tango_profile) + 1, dtype=float)
    y = np.asarray(tango_profile, dtype=float)
    x, y = _maybe_downsample(x, y)

    fig = Figure(figsize=(_adaptive_width(n), 4), dpi=120)
    fig.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)

    ax.plot(x, y, color="#7209b7", linewidth=1.4, label="TANGO score")
    ax.fill_between(x, 5.0, y, where=(y > 5.0), interpolate=True,
                    color="#f72585", alpha=0.45, label="Hotspot (>5%)")
    ax.axhline(5.0, color="#374151", linestyle="--", linewidth=0.9,
               label="Threshold (5%)")

    y_min = float(y.min())
    y_max = max(float(y.max()) * 1.1 + 2, 10)
    for idx, (start, end) in enumerate(hotspots):
        rect = Rectangle((start - 0.5, y_min), end - start + 1, y_max - y_min,
                          linewidth=0, facecolor="#f72585", alpha=0.15, zorder=0,
                          label="Hotspot region" if idx == 0 else "_nolegend_")
        ax.add_patch(rect)

    _pub_style_ax(ax, title="TANGO \u03b2-Aggregation",
                  xlabel="Residue", ylabel="Score (%)",
                  grid=True, title_size=label_font - 1,
                  label_size=label_font - 1, tick_size=tick_font - 1)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(y_min, y_max)
    ax.legend(fontsize=tick_font - 2, loc="upper right",
              framealpha=0.85, edgecolor="#d0d4e0")
    fig.tight_layout(pad=1.5)
    return fig
