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
                  title_size=label_font - 1,
                  label_size=label_font - 1,
                  tick_size=tick_font - 1)
    fig.tight_layout(pad=1.8)
    return fig


def create_amino_acid_composition_pie_figure(
    aa_counts: dict,
    label_font: int = 14,
) -> Figure:
    """Pie chart of amino acid composition."""
    # Filter out zero-count amino acids
    items = [(aa, cnt) for aa, cnt in aa_counts.items() if cnt > 0]
    labels = [x[0] for x in items]
    values = [x[1] for x in items]
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(labels))]

    fig = Figure(figsize=(7, 5), dpi=120)
    fig.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)
    wedges, texts = ax.pie(
        values, labels=labels, colors=colors,
        startangle=140,
        wedgeprops=dict(linewidth=0.8, edgecolor="white"),
    )
    for t in texts:
        t.set_fontsize(label_font - 3)
        t.set_color("#2d3748")
    ax.set_title("Amino Acid Composition", fontsize=label_font - 1,
                 fontweight="bold", color="#1a1a2e", pad=12)
    fig.tight_layout(pad=1.8)
    return fig
