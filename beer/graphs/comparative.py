"""Comparative/multi-protein figures: MSA conservation, complex MW, truncation series, etc."""
from __future__ import annotations

import math
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
import mplcursors

from beer.constants import KYTE_DOOLITTLE, DEFAULT_PKA
from beer.graphs._style import _pub_style_ax, _ACCENT, _NEG_COL, _POS_COL

from collections import Counter

# MW ladder standards for pI/MW gel
MW_STANDARDS_KDA = [10, 15, 20, 25, 37, 50, 75, 100, 150, 250]


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


def _shannon_entropy(column_chars: list) -> float:
    """Shannon entropy (bits) for a list of characters (including gaps)."""
    total = len(column_chars)
    if total == 0:
        return 0.0
    counts = Counter(column_chars)
    probs = [c / total for c in counts.values()]
    h = -sum(p * math.log2(p) for p in probs if p > 0)
    return h


def create_msa_conservation_figure(
    sequences: list,
    names: list,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Per-column conservation bar chart from a multiple sequence alignment."""
    if not sequences:
        fig = Figure(figsize=(10, 4))
        ax = fig.add_subplot(111)
        ax.set_title("MSA Conservation Profile (no data)", fontsize=label_font - 2)
        return fig

    n_seq = len(sequences)
    aln_len = len(sequences[0])
    positions = np.arange(1, aln_len + 1)

    conservation = np.zeros(aln_len)
    dominant_aa = [""] * aln_len

    for col in range(aln_len):
        chars = [sequences[s][col] for s in range(n_seq)
                 if col < len(sequences[s])]
        h = _shannon_entropy(chars)
        n_distinct = len(set(chars))
        if n_distinct > 1:
            conservation[col] = max(0.0, 1.0 - h / math.log2(n_distinct))
        else:
            conservation[col] = 1.0

        non_gap = [c for c in chars if c != "-"]
        if non_gap:
            dominant_aa[col] = Counter(non_gap).most_common(1)[0][0]
        else:
            dominant_aa[col] = "-"

    fig = Figure(figsize=(max(10, aln_len * 0.15), 4), dpi=120)
    fig.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)

    ax.bar(positions, conservation, color="#4361ee", width=0.8, zorder=2, alpha=0.8)

    for col in range(aln_len):
        if conservation[col] > 0.7 and dominant_aa[col] not in ("-", ""):
            ax.text(
                positions[col], conservation[col] + 0.02,
                dominant_aa[col],
                ha="center", va="bottom",
                fontsize=max(tick_font - 4, 6),
                color="#1a1a2e",
            )

    ax.axhline(0.7, color="#f72585", linestyle="--", linewidth=0.8,
               label="Threshold (0.7)")

    _pub_style_ax(ax, title="MSA Conservation",
                  xlabel="Alignment Position", ylabel="Conservation",
                  grid=True, title_size=label_font - 1,
                  label_size=label_font - 1, tick_size=tick_font - 1)
    ax.set_xlim(0.5, aln_len + 0.5)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=tick_font - 2, loc="upper right",
              framealpha=0.85, edgecolor="#d0d4e0")
    fig.tight_layout(pad=1.5)
    return fig


def create_complex_mw_figure(
    chains_data: list,
    stoichiometry_str: str,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Bar chart of individual chain MWs and the assembled complex MW."""
    stoich_map: dict = {}
    for match in re.finditer(r"([A-Za-z]+)(\d*)", stoichiometry_str):
        chain_id = match.group(1)
        count_str = match.group(2)
        count = int(count_str) if count_str else 1
        if chain_id:
            stoich_map[chain_id] = count

    chain_ids = [c["chain_id"] for c in chains_data]
    chain_mws = [float(c["mol_weight"]) for c in chains_data]
    default_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    ]
    chain_colors = [
        c.get("color", default_colors[i % len(default_colors)])
        for i, c in enumerate(chains_data)
    ]

    total_mw = 0.0
    for chain in chains_data:
        cid = chain["chain_id"]
        n_copies = stoich_map.get(cid, 1)
        total_mw += n_copies * float(chain["mol_weight"])

    bar_labels = chain_ids + ["Complex"]
    bar_heights = chain_mws + [total_mw]
    bar_colors_all = chain_colors + ["#333333"]

    fig = Figure(figsize=(max(6, len(bar_labels) * 1.2 + 2), 5), dpi=120)
    fig.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)

    bars = ax.bar(
        range(len(bar_labels)),
        [mw / 1000 for mw in bar_heights],
        color=bar_colors_all,
        edgecolor="black",
        linewidth=0.6,
    )

    for bar_obj, height_kda in zip(bars, [h / 1000 for h in bar_heights]):
        ax.text(
            bar_obj.get_x() + bar_obj.get_width() / 2,
            bar_obj.get_height() + 0.5,
            f"{height_kda:.1f} kDa",
            ha="center", va="bottom",
            fontsize=max(tick_font - 2, 8),
        )

    ax.annotate(
        f"Total Complex: {total_mw/1000:.1f} kDa\n(Stoichiometry: {stoichiometry_str})",
        xy=(len(bar_labels) - 1, total_mw / 1000),
        xytext=(-60, 20),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=tick_font,
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="grey"),
    )

    ax.set_xticks(range(len(bar_labels)))
    ax.set_xticklabels(bar_labels, fontsize=tick_font - 1)
    _pub_style_ax(ax, title="Complex Mass Composition",
                  xlabel="", ylabel="MW (kDa)",
                  grid=True, title_size=label_font - 1,
                  label_size=label_font - 1, tick_size=tick_font - 1)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(pad=1.5)
    return fig


def create_truncation_series_figure(
    truncation_data: dict,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Multi-panel (2x3) figure showing how 6 properties change with truncation."""
    panels = [
        ("pI", "pI"),
        ("gravy", "GRAVY"),
        ("fcr", "FCR"),
        ("ncpr", "NCPR"),
        ("net_charge_7", "Net Charge (pH 7)"),
        ("disorder_frac", "Disorder Fraction"),
    ]

    n_trunc = truncation_data.get("n_trunc", [])
    c_trunc = truncation_data.get("c_trunc", [])

    def extract(data_list, key):
        return (
            [d.get("pct", 0) for d in data_list],
            [d.get(key, 0) for d in data_list],
        )

    fig = Figure(figsize=(14, 8), dpi=120)
    fig.set_facecolor("#ffffff")
    fig.suptitle("Truncation Series", fontsize=label_font,
                 fontweight="bold", color="#1a1a2e")
    fig.subplots_adjust(hspace=0.45, wspace=0.38,
                        left=0.08, right=0.97, top=0.91, bottom=0.09)

    for panel_idx, (key, ylabel) in enumerate(panels):
        ax = fig.add_subplot(2, 3, panel_idx + 1)

        nx, ny = extract(n_trunc, key)
        cx, cy = extract(c_trunc, key)

        if nx:
            ax.plot(nx, ny, color="#4361ee", linewidth=1.4, marker="o",
                    markersize=3.5, label="N-term")
        if cx:
            ax.plot(cx, cy, color="#f72585", linewidth=1.4, marker="s",
                    markersize=3.5, label="C-term")

        _pub_style_ax(ax, title="", xlabel="Truncation (%)", ylabel=ylabel,
                      grid=True, title_size=label_font - 2,
                      label_size=label_font - 3, tick_size=tick_font - 2)
        ax.set_xlim(0, 90)

        if panel_idx == 0:
            ax.legend(fontsize=tick_font - 3, loc="best",
                      framealpha=0.85, edgecolor="#d0d4e0")

    return fig


def create_pI_MW_gel_figure(
    proteins_data: list,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """2D scatter plot of pI vs log10(MW) - SDS-PAGE proxy."""
    fig = Figure(figsize=(9, 6), dpi=120)
    fig.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)

    single = len(proteins_data) == 1

    for mw_kda in MW_STANDARDS_KDA:
        log_mw = math.log10(mw_kda * 1000)
        ax.axhline(log_mw, color="lightgrey", linestyle="--", linewidth=0.8, zorder=0)
        ax.text(13.8, log_mw, f"{mw_kda} kDa", va="center", ha="right",
                fontsize=max(tick_font - 3, 7), color="grey")

    ax.axvline(7.0, color="grey", linestyle="--", linewidth=0.8, zorder=0)
    ax.axvline(4.0, color="lightblue", linestyle=":", linewidth=0.8, zorder=0)
    ax.axvline(10.0, color="lightcoral", linestyle=":", linewidth=0.8, zorder=0)

    scatter_artists = []
    for pdata in proteins_data:
        pI_val = float(pdata["pI"])
        mw_val = float(pdata["mol_weight"])
        log_mw = math.log10(mw_val)
        color = pdata.get("color", "steelblue")
        marker = "*" if single else "o"
        ms = 18 if single else 10
        sc = ax.scatter(pI_val, log_mw, color=color, marker=marker,
                        s=ms ** 2, zorder=5, edgecolors="black", linewidths=0.5)
        ax.annotate(
            pdata["name"],
            xy=(pI_val, log_mw),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=max(tick_font - 2, 8),
        )
        scatter_artists.append(sc)

    if scatter_artists:
        try:
            cursor = mplcursors.cursor(scatter_artists, hover=True)

            @cursor.connect("add")
            def on_add(sel):
                idx = sel.index
                if idx < len(proteins_data):
                    p = proteins_data[idx]
                    sel.annotation.set_text(
                        f"{p['name']}\npI={p['pI']:.2f}\nMW={p['mol_weight']:.0f} Da"
                    )
        except Exception:
            pass

    std_log_ticks = [math.log10(mw * 1000) for mw in MW_STANDARDS_KDA]
    ax.set_yticks(std_log_ticks)
    ax.set_yticklabels([f"{mw} kDa" for mw in MW_STANDARDS_KDA],
                       fontsize=tick_font - 1)

    _pub_style_ax(ax, title="pI–MW Map",
                  xlabel="pI", ylabel="MW",
                  grid=True, title_size=label_font - 1,
                  label_size=label_font - 1, tick_size=tick_font - 1)
    ax.set_xlim(0, 14)
    fig.tight_layout(pad=1.5)
    return fig


def create_saturation_mutagenesis_figure(
    seq: str,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """In silico saturation mutagenesis heatmap."""
    n = len(seq)
    if n == 0 or n > 500:
        fig = Figure(figsize=(9, 5), dpi=120)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5,
                "Sequence too long for saturation mutagenesis\n(limit: 500 aa)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=label_font - 2, color="#718096")
        ax.set_axis_off()
        return fig

    AAS_BY_HYDRO = list("RNDQEKSHPYTGACMWFLIV")
    wt_gravy = sum(KYTE_DOOLITTLE.get(aa, 0.0) for aa in seq) / n
    pos_n = sum(1 for aa in seq if aa in "KR")
    neg_n = sum(1 for aa in seq if aa in "DE")
    wt_ncpr = (pos_n - neg_n) / n

    mat = np.zeros((20, n), dtype=float)
    for col_i, pos in enumerate(range(n)):
        wt_aa = seq[pos]
        wt_kd = KYTE_DOOLITTLE.get(wt_aa, 0.0)
        wt_is_pos = 1 if wt_aa in "KR" else 0
        wt_is_neg = 1 if wt_aa in "DE" else 0
        for row_i, mut_aa in enumerate(AAS_BY_HYDRO):
            if mut_aa == wt_aa:
                mat[row_i, col_i] = 0.0
                continue
            mut_kd = KYTE_DOOLITTLE.get(mut_aa, 0.0)
            delta_gravy = (mut_kd - wt_kd) / n
            mut_is_pos = 1 if mut_aa in "KR" else 0
            mut_is_neg = 1 if mut_aa in "DE" else 0
            delta_ncpr = ((mut_is_pos - wt_is_pos) - (mut_is_neg - wt_is_neg)) / n
            mat[row_i, col_i] = abs(delta_gravy) + abs(delta_ncpr)

    fig = Figure(figsize=(max(9, n * 0.08 + 1), 5.5), dpi=120)
    fig.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)

    im = ax.imshow(mat, aspect="auto", cmap="hot_r", origin="upper",
                   interpolation="nearest",
                   vmin=0, vmax=np.percentile(mat[mat > 0], 95) if mat.max() > 0 else 1)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, aspect=20, pad=0.02)
    cbar.set_label("|ΔGRAVY| + |ΔNCPR|", fontsize=tick_font - 1, color="#4a5568")
    cbar.ax.tick_params(labelsize=tick_font - 2, colors="#4a5568")

    for col_i, wt_aa in enumerate(seq):
        if wt_aa in AAS_BY_HYDRO:
            row_i = AAS_BY_HYDRO.index(wt_aa)
            ax.plot(col_i, row_i, "w.", markersize=3, alpha=0.8)

    ax.set_yticks(range(20))
    ax.set_yticklabels(AAS_BY_HYDRO, fontsize=max(6, tick_font - 4))
    ax.set_xlabel("Residue", fontsize=label_font - 1, color="#4a5568")
    ax.set_ylabel("Substitution", fontsize=label_font - 1, color="#4a5568")
    ax.set_title("Single-Residue Perturbation Map",
                 fontsize=label_font, fontweight="bold", color="#1a1a2e", pad=8)
    ax.tick_params(axis="x", labelsize=tick_font - 2)
    fig.tight_layout(pad=1.5)
    return fig


def create_uversky_phase_plot(
    seq: str,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Uversky charge-hydrophobicity phase diagram."""
    n = len(seq)
    if n == 0:
        fig = Figure(figsize=(6, 5), dpi=120)
        return fig

    pos_n = sum(1 for aa in seq if aa in "KR")
    neg_n = sum(1 for aa in seq if aa in "DE")
    mean_charge = abs(pos_n - neg_n) / n
    mean_hydro = sum(KYTE_DOOLITTLE.get(aa, 0.0) for aa in seq) / n
    h_norm = (mean_hydro + 4.5) / 9.0

    fig = Figure(figsize=(6, 5), dpi=120)
    fig.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#fafbff")

    r_vals = np.linspace(0, 0.5, 200)
    h_boundary = 2.785 * r_vals + 0.446
    h_boundary = np.clip(h_boundary, 0, 1)
    ax.plot(r_vals, h_boundary, color="#374151", linewidth=1.8,
            linestyle="--", label="Uversky boundary", zorder=3)

    ax.fill_between(r_vals, h_boundary, 1.0, alpha=0.10,
                    color="#4361ee", label="Ordered / compact")
    ax.fill_between(r_vals, 0, h_boundary, alpha=0.10,
                    color="#f72585", label="Disordered / IDP")

    ax.text(0.05, 0.75, "Ordered / Folded", fontsize=tick_font - 3,
            color="#4361ee", alpha=0.8, style="italic")
    ax.text(0.25, 0.15, "Disordered / IDP", fontsize=tick_font - 3,
            color="#f72585", alpha=0.8, style="italic")

    # Plot the protein's position without a binary verdict label
    is_below = h_norm < (2.785 * mean_charge + 0.446)
    pt_color = "#f72585" if is_below else "#4361ee"
    ax.scatter([mean_charge], [h_norm], color=pt_color, s=120, zorder=5,
               edgecolors="white", linewidths=1.2)
    ax.annotate(f"  ({mean_charge:.3f}, {h_norm:.3f})",
                xy=(mean_charge, h_norm),
                fontsize=tick_font - 3, color=pt_color,
                xytext=(mean_charge + 0.02, h_norm + 0.04))

    _pub_style_ax(ax,
                  title="Uversky Phase Diagram",
                  xlabel="|Mean Charge|",
                  ylabel="Mean Hydrophobicity",
                  grid=True, title_size=label_font - 1,
                  label_size=label_font - 1, tick_size=tick_font - 1)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=tick_font - 3, framealpha=0.85,
              edgecolor="#d0d4e0", loc="upper right")
    fig.tight_layout(pad=1.5)
    return fig


def create_msa_covariance_figure(
    mi_apc: "list[list[float]]",
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Residue covariance heatmap from MSA mutual information (APC-corrected).

    Parameters
    ----------
    mi_apc:
        *n_col × n_col* matrix of APC-corrected mutual information values
        (bits), as returned by :func:`beer.analysis.msa_covariance.calc_msa_mutual_information`.
    label_font, tick_font:
        Font sizes.
    """
    mat = np.array(mi_apc, dtype=float)
    n = mat.shape[0]

    if n == 0:
        fig = Figure(figsize=(6, 5), dpi=120)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No alignment data", ha="center", va="center",
                transform=ax.transAxes, fontsize=label_font - 2, color="#718096")
        ax.set_axis_off()
        return fig

    # Figure size scales with n but caps to keep it renderable
    dim = min(9.0, max(5.0, n * 0.025 + 3.0))
    fig = Figure(figsize=(dim + 1.2, dim), dpi=120)
    fig.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)

    pos_vals = mat[mat > 0]
    vmax = float(np.percentile(pos_vals, 95)) if pos_vals.size > 0 else 1.0
    im = ax.imshow(mat, cmap="viridis", aspect="auto", origin="upper",
                   vmin=0.0, vmax=vmax, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, aspect=22, pad=0.02)
    cbar.set_label("MI-APC (bits)", fontsize=tick_font - 1, color="#4a5568")
    cbar.ax.tick_params(labelsize=tick_font - 2, colors="#4a5568")

    _pub_style_ax(ax, title="Residue Covariance  (MI-APC)",
                  xlabel="Column", ylabel="Column",
                  grid=False, title_size=label_font - 1,
                  label_size=label_font - 1, tick_size=tick_font - 2)
    ax.tick_params(colors="#4a5568")
    fig.tight_layout(pad=1.5)
    return fig
