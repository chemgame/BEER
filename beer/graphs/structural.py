"""Structural analysis figures: Ramachandran, contact network, pLDDT, distance map."""
from __future__ import annotations

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import mplcursors

from beer.graphs._style import _pub_style_ax


# ---------------------------------------------------------------------------
# Internal layout helpers for contact network
# ---------------------------------------------------------------------------

def _circular_layout(n: int) -> np.ndarray:
    """Return (n, 2) xy positions for n nodes evenly spaced on a unit circle."""
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _spring_layout(adj: np.ndarray, n_iter: int = 200, seed: int = 42) -> np.ndarray:
    """Fruchterman-Reingold spring layout (pure numpy)."""
    n = adj.shape[0]
    rng = np.random.default_rng(seed)
    pos = rng.random((n, 2)) * 2 - 1

    k = 1.0 / math.sqrt(n) if n > 0 else 1.0
    t = 0.1

    for _ in range(n_iter):
        delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.linalg.norm(delta, axis=2)
        np.fill_diagonal(dist, 1e-9)

        rep = k ** 2 / dist
        rep_force = np.einsum("ij,ijk->ik", rep / dist, delta)

        att = (dist ** 2 / k) * adj
        att_force = np.einsum("ij,ijk->ik", att / dist, -delta)

        displacement = rep_force + att_force
        disp_len = np.linalg.norm(displacement, axis=1, keepdims=True)
        disp_len = np.maximum(disp_len, 1e-9)
        pos += displacement / disp_len * np.minimum(disp_len, t)
        t *= 0.95

    pos -= pos.min(axis=0)
    pos /= pos.max(axis=0) + 1e-9
    pos = pos * 2 - 1

    return pos


# ---------------------------------------------------------------------------
# Public figure functions
# ---------------------------------------------------------------------------

def create_ramachandran_figure(
    phi_psi_data: list,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Classical Ramachandran plot coloured by secondary structure."""
    SS_COLORS = {"H": "#1f77b4", "E": "#d62728", "C": "#aaaaaa"}
    SS_LABELS = {"H": "\u03b1-Helix", "E": "\u03b2-Sheet", "C": "Coil"}

    fig = Figure(figsize=(7, 6), tight_layout=True)
    ax = fig.add_subplot(111)

    ax.add_patch(Rectangle((-80, -60), 32, 40, color="#555555", alpha=0.25,
                            zorder=0, label="_helix_region"))
    ax.add_patch(Rectangle((-150, 90), 60, 70, color="#888888", alpha=0.20,
                            zorder=0, label="_sheet_region"))
    ax.add_patch(Rectangle((40, 20), 40, 40, color="#bbbbbb", alpha=0.20,
                            zorder=0, label="_lh_region"))

    ax.axhline(0, color="grey", linewidth=0.6, linestyle="-")
    ax.axvline(0, color="grey", linewidth=0.6, linestyle="-")

    plotted_ss = set()
    for res in phi_psi_data:
        phi = res.get("phi")
        psi = res.get("psi")
        ss = res.get("ss", "C")
        if phi is None or psi is None:
            continue
        color = SS_COLORS.get(ss, "#aaaaaa")
        label = SS_LABELS.get(ss, ss) if ss not in plotted_ss else "_nolegend_"
        plotted_ss.add(ss)
        ax.scatter(phi, psi, color=color, s=12, alpha=0.7,
                   edgecolors="none", label=label, zorder=3)

    helix_patch = mpatches.Patch(color="#555555", alpha=0.4, label="Core \u03b1-helix region")
    sheet_patch = mpatches.Patch(color="#888888", alpha=0.35, label="Core \u03b2-sheet region")
    lh_patch = mpatches.Patch(color="#bbbbbb", alpha=0.35, label="Left-handed helix")

    handles, labels = ax.get_legend_handles_labels()
    visible = [(h, l) for h, l in zip(handles, labels) if not l.startswith("_")]
    visible_h, visible_l = zip(*visible) if visible else ([], [])
    ax.legend(
        list(visible_h) + [helix_patch, sheet_patch, lh_patch],
        list(visible_l) + ["Core \u03b1-helix", "Core \u03b2-sheet", "Left-handed helix"],
        fontsize=tick_font,
        loc="upper right",
        markerscale=1.5,
    )

    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xlabel("\u03c6 (\u00b0)", fontsize=label_font)
    ax.set_ylabel("\u03c8 (\u00b0)", fontsize=label_font)
    ax.set_title("Ramachandran Plot (from PDB structure)", fontsize=label_font)
    ax.tick_params(axis="both", labelsize=tick_font)
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks(range(-180, 181, 60))

    return fig


def create_contact_network_figure(
    seq: str,
    dist_matrix: np.ndarray,
    cutoff_angstrom: float = 8.0,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Residue contact network derived from Ca distance matrix."""
    n = len(seq)
    dist_matrix = np.asarray(dist_matrix, dtype=float)

    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 2, n):
            if dist_matrix[i, j] <= cutoff_angstrom:
                adj[i, j] = True
                adj[j, i] = True

    degree = adj.sum(axis=1).astype(float)
    max_deg = degree.max() if degree.max() > 0 else 1.0

    large_protein = n > 100
    top_n = 30

    if large_protein:
        top_idx = np.argsort(degree)[-top_n:]
        sub_n = len(top_idx)
        sub_degree = degree[top_idx]
        sub_adj = adj[np.ix_(top_idx, top_idx)]
        pos = _circular_layout(sub_n)
        node_degrees = sub_degree
        plot_adj = sub_adj
        residue_labels = [f"{top_idx[i]+1}" for i in range(sub_n)]
    else:
        pos = _spring_layout(adj.astype(float))
        node_degrees = degree
        plot_adj = adj
        sub_n = n
        residue_labels = [str(i + 1) for i in range(n)]

    norm_n = (n - 1) if n > 1 else 1
    centrality = node_degrees / norm_n

    fig = Figure(figsize=(9, 8), tight_layout=True)
    ax = fig.add_subplot(111)

    cmap = cm.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=0, vmax=centrality.max() if centrality.max() > 0 else 1)

    for i in range(sub_n):
        for j in range(i + 1, sub_n):
            if plot_adj[i, j]:
                orig_i = top_idx[i] if large_protein else i
                orig_j = top_idx[j] if large_protein else j
                d = dist_matrix[orig_i, orig_j]
                lw = max(0.2, 2.0 / (d + 1e-9) * cutoff_angstrom)
                ax.plot(
                    [pos[i, 0], pos[j, 0]],
                    [pos[i, 1], pos[j, 1]],
                    color="lightgrey",
                    linewidth=min(lw, 2.0),
                    zorder=1,
                )

    node_sizes = 50 + (node_degrees / max_deg) * 250
    scatter = ax.scatter(
        pos[:, 0], pos[:, 1],
        c=centrality, cmap="viridis", norm=norm,
        s=node_sizes, zorder=3, edgecolors="black", linewidths=0.4,
    )

    if not large_protein and n <= 50:
        for i in range(sub_n):
            ax.text(pos[i, 0], pos[i, 1], residue_labels[i],
                    ha="center", va="center",
                    fontsize=max(tick_font - 4, 6), zorder=4)

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Degree Centrality", fontsize=label_font - 2)
    cbar.ax.tick_params(labelsize=tick_font - 2)

    ax.set_aspect("equal")
    ax.axis("off")
    subtitle = f" \u2014 Top {top_n} residues by degree" if large_protein else ""
    ax.set_title(
        f"Residue Contact Network (C\u03b1 \u2264 {cutoff_angstrom} \u00c5){subtitle}",
        fontsize=label_font,
    )

    return fig


def create_plddt_figure(
    plddt: list,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Per-residue pLDDT confidence score with coloured confidence zones."""
    import matplotlib.pyplot as plt

    n = len(plddt)
    xs = list(range(1, n + 1))
    fig = Figure(figsize=(9, 4), dpi=120)
    fig.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)
    ax.axhspan(90, 100, alpha=0.07, color="#0053D6")
    ax.axhspan(70, 90, alpha=0.07, color="#65CBF3")
    ax.axhspan(50, 70, alpha=0.07, color="#FFDB13")
    ax.axhspan(0, 50, alpha=0.07, color="#FF7D45")
    cmap = plt.get_cmap("RdYlBu")
    norm = mcolors.Normalize(vmin=0, vmax=100)
    for i in range(n - 1):
        ax.plot([xs[i], xs[i + 1]], [plddt[i], plddt[i + 1]],
                color=cmap(norm((plddt[i] + plddt[i + 1]) / 2)),
                linewidth=1.8, zorder=4, solid_capstyle="round")
    for thresh, col, lbl in [
        (90, "#0053D6", ">90 Very high"),
        (70, "#65CBF3", "70\u201390 Confident"),
        (50, "#FFDB13", "50\u201370 Low"),
    ]:
        ax.axhline(thresh, color=col, linewidth=0.8, linestyle="--", alpha=0.8)
    _pub_style_ax(ax,
                  title="pLDDT / B-factor Confidence",
                  xlabel="Residue Position", ylabel="pLDDT Score",
                  grid=False, title_size=label_font + 1,
                  label_size=label_font - 1, tick_size=tick_font - 1)
    ax.set_ylim(0, 100)
    ax.set_xlim(1, n)
    ax.legend(handles=[
        Patch(color="#0053D6", alpha=0.5, label=">90  Very high"),
        Patch(color="#65CBF3", alpha=0.5, label="70\u201390  Confident"),
        Patch(color="#FFDB13", alpha=0.5, label="50\u201370  Low"),
        Patch(color="#FF7D45", alpha=0.5, label="<50  Very low"),
    ], fontsize=tick_font - 3, framealpha=0.85, edgecolor="#d0d4e0", loc="lower right")
    fig.tight_layout(pad=1.5)
    mplcursors.cursor(ax)
    return fig


def create_distance_map_figure(
    dist_matrix: np.ndarray,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Ca pairwise distance heatmap from a loaded PDB structure."""
    n = dist_matrix.shape[0]
    fig = Figure(figsize=(6.5, 5.5), dpi=120)
    fig.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#fafbff")
    im = ax.imshow(dist_matrix, cmap="viridis_r", aspect="auto",
                   origin="upper", interpolation="nearest",
                   vmin=0, vmax=min(40, dist_matrix.max()))
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, aspect=20, pad=0.02)
    cbar.set_label("C\u03b1 distance (\u00c5)", fontsize=tick_font - 1, color="#4a5568")
    cbar.ax.tick_params(labelsize=tick_font - 2, colors="#4a5568")
    ax.contour(dist_matrix, levels=[8.0], colors=["#f72585"],
               linewidths=[0.6], alpha=0.7)
    _pub_style_ax(ax,
                  title=f"C\u03b1 Distance Map  ({n} residues)  \u2014 pink contour = 8 \u00c5 contact",
                  xlabel="Residue Position", ylabel="Residue Position",
                  grid=False, title_size=label_font,
                  label_size=label_font - 1, tick_size=tick_font - 1)
    fig.tight_layout(pad=1.5)
    return fig
