"""Charge-related figures (isoelectric focus, local charge, charge decoration)."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure

from beer.constants import DEFAULT_PKA
from beer.graphs._style import (
    _pub_style_ax,
    _ACCENT, _NEG_COL, _POS_COL,
)


def _calc_net_charge(seq: str, pH: float = 7.0, pka: dict = None) -> float:
    """Henderson-Hasselbalch net charge."""
    p = pka or DEFAULT_PKA
    net = 1 / (1 + 10 ** (pH - p['NTERM'])) - 1 / (1 + 10 ** (p['CTERM'] - pH))
    for aa in seq:
        if aa in ('D', 'E', 'C', 'Y'):
            net -= 1 / (1 + 10 ** (p[aa] - pH))
        elif aa in ('K', 'R', 'H'):
            net += 1 / (1 + 10 ** (pH - p[aa]))
    return net


def create_isoelectric_focus_figure(
    seq: str,
    label_font: int = 14,
    tick_font: int = 12,
    pka: dict = None,
) -> Figure:
    """Enhanced isoelectric focusing simulation."""
    fig = Figure(figsize=(9, 4.5), dpi=120)
    fig.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)
    phs = [i / 20 for i in range(281)]
    nets = [_calc_net_charge(seq, p, pka) for p in phs]
    ax.plot(phs, nets, color=_ACCENT, linewidth=2.0, zorder=5)
    ax.fill_between(phs, nets, 0, where=[v >= 0 for v in nets],
                    alpha=0.15, color=_POS_COL, interpolate=True, label="+")
    ax.fill_between(phs, nets, 0, where=[v < 0 for v in nets],
                    alpha=0.15, color=_NEG_COL, interpolate=True, label="−")
    ax.axhline(0, color="#888", linewidth=0.8, linestyle="--", zorder=3)
    pI_idx = min(range(len(nets)), key=lambda i: abs(nets[i]))
    pI = phs[pI_idx]
    ax.axvline(pI, color="#f72585", linewidth=1.6, linestyle="-", alpha=0.9, zorder=4)
    y_top = max(nets) if max(nets) > 0 else 1
    ax.annotate(f"  pI={pI:.2f}", xy=(pI, 0),
                xytext=(pI + 0.5, y_top * 0.65),
                fontsize=tick_font, color="#f72585", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#f72585", lw=1.0))
    ch74 = _calc_net_charge(seq, 7.4, pka)
    ax.axvline(7.4, color="#43aa8b", linewidth=0.9, linestyle=":", alpha=0.8, zorder=3)
    y_bot = min(nets) if min(nets) < 0 else -1
    ax.text(7.55, y_bot * 0.65, f"pH 7.4\n({ch74:+.1f})",
            fontsize=tick_font - 2, color="#43aa8b")
    _pub_style_ax(ax, title="Isoelectric Focus",
                  xlabel="pH", ylabel="Net Charge",
                  grid=True, title_size=label_font - 1,
                  label_size=label_font - 1, tick_size=tick_font - 1)
    ax.set_xlim(0, 14)
    ax.legend(fontsize=tick_font - 2, framealpha=0.85, edgecolor="#d0d4e0",
              loc="upper right")
    fig.tight_layout(pad=1.5)
    return fig


def create_local_charge_figure(
    ncpr_profile: list,
    window_size: int,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Sliding-window NCPR (local charge) profile."""
    n = len(ncpr_profile)
    w = max(9, min(16, 9 + n * 0.015))
    fig = Figure(figsize=(w, 4), dpi=120)
    fig.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)
    xs = list(range(1, n + 1))
    ax.fill_between(xs, ncpr_profile, 0, where=[v >= 0 for v in ncpr_profile],
                    alpha=0.18, color=_POS_COL, interpolate=True)
    ax.fill_between(xs, ncpr_profile, 0, where=[v < 0 for v in ncpr_profile],
                    alpha=0.18, color=_NEG_COL, interpolate=True)
    ax.plot(xs, ncpr_profile, color=_ACCENT, linewidth=1.6,
            marker="o", markersize=3.0, markeredgewidth=0, zorder=4)
    ax.axhline(0, color="#888", linewidth=0.8, linestyle="--", zorder=3)
    _pub_style_ax(ax, title=f"Local Charge  (w={window_size})",
                  xlabel="Residue", ylabel="NCPR",
                  grid=True, title_size=label_font - 1,
                  label_size=label_font - 1, tick_size=tick_font - 1)
    fig.tight_layout(pad=1.5)
    return fig


def create_charge_decoration_figure(
    fcr: float,
    ncpr: float,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Das-Pappu FCR vs |NCPR| phase diagram."""
    abs_ncpr = abs(ncpr)
    fig = Figure(figsize=(6, 5.5), dpi=120)
    fig.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#fafbff")
    fcr_arr = np.linspace(0, 0.72, 200)
    ax.plot(fcr_arr, fcr_arr, color="#c0c4d0", linewidth=1.2,
            linestyle="--", zorder=2, label="|NCPR|=FCR")
    ax.fill_between(fcr_arr, 0, fcr_arr, alpha=0.04, color="#4361ee")
    ax.axvspan(0,    0.25, alpha=0.09, color="#43aa8b")
    ax.axvspan(0.25, 0.35, alpha=0.09, color="#f3722c")
    ax.axvspan(0.35, 0.72, alpha=0.09, color="#f72585")
    ax.axvline(0.25, color="#888", linewidth=0.7, linestyle=":", zorder=3)
    ax.axvline(0.35, color="#888", linewidth=0.7, linestyle=":", zorder=3)
    ax.text(0.12, 0.62, "Globule /\nCollapsed", ha="center",
            fontsize=tick_font - 3, color="#43aa8b", fontweight="600")
    ax.text(0.30, 0.62, "Coil /\nTadpole", ha="center",
            fontsize=tick_font - 3, color="#f3722c", fontweight="600")
    ax.text(0.54, 0.62, "Strong\nPolyelectrolyte", ha="center",
            fontsize=tick_font - 3, color="#f72585", fontweight="600")
    ax.scatter([fcr], [abs_ncpr], marker="*", s=360, color="#f72585",
               zorder=10, edgecolors="white", linewidths=0.8, label="This protein")
    ax.annotate(f"  ({fcr:.2f}, {abs_ncpr:.2f})", xy=(fcr, abs_ncpr),
                fontsize=tick_font - 1, color="#f72585", va="bottom")
    _pub_style_ax(ax, title="Das-Pappu Phase Diagram",
                  xlabel="FCR", ylabel="|NCPR|",
                  grid=True, title_size=label_font - 1,
                  label_size=label_font - 1, tick_size=tick_font - 1)
    ax.set_xlim(0, 0.72)
    ax.set_ylim(0, 0.72)
    ax.legend(fontsize=tick_font - 3, framealpha=0.85, edgecolor="#d0d4e0",
              loc="upper left")
    fig.tight_layout(pad=1.5)
    return fig
