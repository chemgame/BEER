"""Amino acid composition figures."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from beer.graphs._style import _pub_style_ax, _PALETTE


def create_amino_acid_composition_figure(
    aa_counts: dict,
    aa_freq: dict,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Bar chart of amino acid counts with frequency labels."""
    fig = Figure(figsize=(9, 4.5), dpi=120)
    fig.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)
    aas = sorted(aa_counts)
    cnts = [aa_counts[a] for a in aas]
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(aas))]
    bars = ax.bar(aas, cnts, color=colors, width=0.65, zorder=3,
                  edgecolor="white", linewidth=0.5)
    _pub_style_ax(ax,
                  title="Amino Acid Composition",
                  xlabel="Amino Acid",
                  ylabel="Count",
                  grid=True,
                  title_size=label_font,
                  label_size=label_font,
                  tick_size=tick_font)
    fig.tight_layout(pad=1.8)
    return fig


