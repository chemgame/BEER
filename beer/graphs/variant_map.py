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
    fig = Figure(figsize=(max(10, L * 0.18), 7), layout="constrained")
    gs  = fig.add_gridspec(2, 1, height_ratios=[3, 1])

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
    ax_heat.tick_params(labelbottom=False)
    _pub_style_ax(ax_heat, title="Variant Effect Map",
                  xlabel="", ylabel="Mutant AA",
                  grid=False, despine=False,
                  title_size=label_font - 1, label_size=label_font - 2,
                  tick_size=tick_font - 2)
    fig.colorbar(im, ax=ax_heat, fraction=0.02, pad=0.02,
                 label="LLR (mut\u2212WT)")

    # --- Per-position mean LLR (shares x-axis with heatmap) ---
    ax_mean = fig.add_subplot(gs[1], sharex=ax_heat)
    from beer.analysis.variant_scoring import mean_effect_per_position
    mean_llr = mean_effect_per_position(llr_matrix)
    positions = np.arange(1, L + 1)
    ax_mean.bar(positions, mean_llr,
                color=["#f72585" if v < 0 else "#4361ee" for v in mean_llr],
                width=0.8, alpha=0.85)
    ax_mean.axhline(0, color="#1a1a2e", linewidth=0.8)
    # Sparse, adaptive x-ticks only on the bottom panel
    step = max(1, L // 20)
    xticks = list(range(step, L + 1, step))
    ax_mean.set_xticks([x - 1 for x in xticks])  # imshow uses 0-based indices
    ax_mean.set_xticklabels([str(x) for x in xticks], fontsize=max(tick_font - 2, 7))
    _pub_style_ax(ax_mean, title="",
                  xlabel="Residue Position",
                  ylabel="Mean Log-Likelihood Ratio",
                  grid=True, despine=True,
                  title_size=label_font - 1, label_size=label_font - 2,
                  tick_size=tick_font - 2)
    _apply_font_sizes(ax_mean, label_font, tick_font)

    fig.patch.set_facecolor("#f8f9ff")
    return fig



def create_alphafold_missense_figure(
    am_data: dict,
    seq: str = "",
    label_font: int = 12,
    tick_font: int = 10,
) -> Figure:
    """Two-panel AlphaMissense figure: heatmap (positions × mutants) + mean pathogenicity profile.

    am_data: dict from fetch_alphafold_missense_scores().
    """
    import matplotlib.colors as mcolors
    from beer.graphs._style import _pub_style_ax

    AA_ORDER_LOCAL = list("ACDEFGHIKLMNPQRSTVWY")
    scores_dict = am_data.get("scores", {})
    mean_profile = am_data.get("mean_per_position", [])
    L = am_data.get("seq_length", len(mean_profile))

    # Build LxAA matrix
    mat = np.full((L, 20), 0.5)
    for pos, mut_dict in scores_dict.items():
        if 1 <= pos <= L:
            for aa, val in mut_dict.items():
                if aa in AA_ORDER_LOCAL:
                    mat[pos - 1, AA_ORDER_LOCAL.index(aa)] = val

    fig = Figure(figsize=(max(10, L * 0.18), 7), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])

    ax_heat = fig.add_subplot(gs[0])
    norm = mcolors.Normalize(vmin=0, vmax=1)
    im = ax_heat.imshow(mat.T, aspect="auto", cmap="RdYlGn_r",
                        norm=norm, interpolation="nearest")
    ax_heat.set_yticks(range(20))
    ax_heat.set_yticklabels(AA_ORDER_LOCAL, fontsize=max(tick_font - 2, 7))
    ax_heat.tick_params(labelbottom=False)
    _pub_style_ax(ax_heat, title="AlphaMissense Pathogenicity",
                  xlabel="", ylabel="Mutant AA",
                  grid=False, despine=False,
                  title_size=label_font - 1, label_size=label_font - 2,
                  tick_size=tick_font - 2)
    fig.colorbar(im, ax=ax_heat, fraction=0.02, pad=0.02,
                 label="Pathogenicity (0=benign, 1=pathogenic)")

    # Bottom panel shares x-axis with heatmap
    ax_mean = fig.add_subplot(gs[1], sharex=ax_heat)
    pos_arr = np.arange(1, L + 1)
    mean_arr = np.array(mean_profile[:L])
    ax_mean.bar(pos_arr, mean_arr,
                color=["#d62728" if v > 0.564 else "#2ca02c" if v < 0.340 else "#ff7f0e"
                       for v in mean_arr],
                width=0.9, alpha=0.85)
    ax_mean.axhline(0.564, color="#d62728", linewidth=0.8, linestyle="--", alpha=0.7)
    ax_mean.axhline(0.340, color="#2ca02c", linewidth=0.8, linestyle="--", alpha=0.7)
    # Sparse, adaptive x-ticks only on the bottom panel
    step = max(1, L // 20)
    xticks = list(range(step, L + 1, step))
    ax_mean.set_xticks(xticks)
    ax_mean.set_xticklabels([str(x) for x in xticks], fontsize=max(tick_font - 2, 7))
    _pub_style_ax(ax_mean, title="",
                  xlabel="Residue Position", ylabel="Mean Pathogenicity Score",
                  grid=True, despine=True,
                  title_size=label_font - 1, label_size=label_font - 2,
                  tick_size=tick_font - 2)
    ax_mean.set_ylim(0, 1.05)
    ax_mean.set_xlim(0.5, L + 0.5)

    fig.patch.set_facecolor("#f8f9ff")
    return fig
