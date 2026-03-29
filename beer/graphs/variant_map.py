"""ESM2 variant effect heatmap and per-position mean effect plot."""
from __future__ import annotations
import numpy as np
from matplotlib.figure import Figure
from beer.graphs._style import _pub_style_ax, _apply_font_sizes

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")


def create_variant_effect_figure(
    seq: str,
    llr_matrix: np.ndarray,
    label_font: int = 12,
    tick_font: int = 10,
) -> Figure:
    """Two-panel figure: heatmap (LxAA) + per-position mean LLR profile.

    Parameters
    ----------
    seq:        Wild-type sequence.
    llr_matrix: (L x 20) LLR array from variant_scoring.compute_single_mutant_llr.
    """
    L = len(seq)
    fig = Figure(figsize=(max(10, L * 0.18), 7))
    gs  = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.4)

    # --- Heatmap ---
    ax_heat = fig.add_subplot(gs[0])
    import matplotlib.colors as mcolors
    vmax = max(2.0, float(np.abs(llr_matrix).max()))
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax_heat.imshow(
        llr_matrix.T,
        aspect="auto",
        cmap="RdBu_r",
        norm=norm,
        interpolation="nearest",
    )
    ax_heat.set_yticks(range(20))
    ax_heat.set_yticklabels(AA_ORDER, fontsize=max(tick_font - 2, 7))
    # Sparse x-ticks
    step = max(1, L // 20)
    xticks = list(range(0, L, step))
    ax_heat.set_xticks(xticks)
    ax_heat.set_xticklabels([str(i + 1) for i in xticks], fontsize=max(tick_font - 2, 7))
    _pub_style_ax(ax_heat, title="ESM2 Variant Effect Map (LLR)",
                  xlabel="", ylabel="Mutant amino acid",
                  grid=False, despine=False,
                  title_size=label_font - 1, label_size=label_font - 2,
                  tick_size=tick_font - 2)
    fig.colorbar(im, ax=ax_heat, fraction=0.02, pad=0.02,
                 label="Log-likelihood ratio (mut \u2212 WT)")

    # --- Per-position mean LLR ---
    ax_mean = fig.add_subplot(gs[1])
    from beer.analysis.variant_scoring import mean_effect_per_position
    mean_llr = mean_effect_per_position(llr_matrix)
    positions = np.arange(1, L + 1)
    ax_mean.bar(positions, mean_llr,
                color=["#f72585" if v < 0 else "#4361ee" for v in mean_llr],
                width=0.8, alpha=0.85)
    ax_mean.axhline(0, color="#1a1a2e", linewidth=0.8)
    _pub_style_ax(ax_mean, title="",
                  xlabel="Residue position",
                  ylabel="Mean LLR",
                  grid=True, despine=True,
                  title_size=label_font - 1, label_size=label_font - 2,
                  tick_size=tick_font - 2)
    _apply_font_sizes(ax_mean, label_font, tick_font)

    fig.patch.set_facecolor("#f8f9ff")
    fig.tight_layout(pad=2.0)
    return fig


def create_pocket_proxy_figure(
    seq: str,
    scores: np.ndarray,
    regions: list,
    label_font: int = 12,
    tick_font: int = 10,
) -> Figure:
    """Single-panel pocket proxy score profile with highlighted regions."""
    L = len(seq)
    fig = Figure(figsize=(max(8, L * 0.08), 3.5))
    ax  = fig.add_subplot(111)
    positions = np.arange(1, L + 1)
    ax.plot(positions, scores, color="#7209b7", linewidth=1.2, alpha=0.9)
    ax.fill_between(positions, scores, alpha=0.2, color="#7209b7")
    threshold = 0.65
    ax.axhline(threshold, color="#f72585", linewidth=0.8, linestyle="--", alpha=0.7,
               label=f"Threshold ({threshold})")
    for start, end in regions:
        ax.axvspan(start + 1, end + 1, alpha=0.15, color="#f72585")
    _pub_style_ax(ax,
                  title="Binding Pocket Proxy Score",
                  xlabel="Residue position",
                  ylabel="Pocket score",
                  grid=True, despine=True,
                  title_size=label_font - 1, label_size=label_font,
                  tick_size=tick_font)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=tick_font - 1, loc="upper right")
    _apply_font_sizes(ax, label_font, tick_font)
    fig.patch.set_facecolor("#f8f9ff")
    fig.tight_layout(pad=2.0)
    return fig
