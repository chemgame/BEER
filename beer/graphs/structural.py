"""Structural analysis figures: Ramachandran, contact network, pLDDT, distance map, SS bead."""
from __future__ import annotations

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.patches import Patch
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.lines as mlines

from beer.graphs._style import _pub_style_ax


def _circular_layout(n: int) -> np.ndarray:
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _spring_layout(adj: np.ndarray, n_iter: int = 200, seed: int = 42) -> np.ndarray:
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


import functools
import pathlib


@functools.lru_cache(maxsize=1)
def _load_rama_grids():
    """Load the bundled Top8000 per-class Ramachandran density grids, or None.

    Built from the MolProbity Top8000 reference set (Richardson lab; 1.9 M
    residues) — per class: a smoothed φ/ψ density grid plus favored/allowed
    contour levels calibrated to the standard enclosed fractions (98% favored;
    99.95% general / 99.8% Gly·Pro·prePro allowed)."""
    try:
        p = pathlib.Path(__file__).parent / "data" / "ramachandran_top8000.npz"
        d = np.load(p)
        return {k: d[k] for k in d.files}
    except Exception:
        return None


def _rama_classes(data: list) -> list:
    """MolProbity Ramachandran class per residue (order preserved)."""
    n = len(data)
    out = []
    for i, res in enumerate(data):
        rn = (res.get("resname") or "").upper()
        nxt = data[i + 1] if i + 1 < n else None
        prepro = (nxt is not None
                  and (nxt.get("resname") or "").upper() == "PRO"
                  and nxt.get("chain_id") == res.get("chain_id"))
        if rn == "GLY":
            out.append("Glycine")
        elif rn == "PRO":
            out.append("Trans-proline")   # cis-Pro needs ω; default to trans
        elif prepro:
            out.append("Pre-proline")
        elif rn in ("ILE", "VAL"):
            out.append("Ile-Val")
        else:
            out.append("General")
    return out


def _rama_eval(phi: float, psi: float, cls: str, grids: dict) -> str:
    """'favored' | 'allowed' | 'outlier' for one residue vs its class grid."""
    edges = grids["edges"]
    nb = int(grids["nbins"])
    g = grids.get(f"{cls}__grid")
    if g is None:
        cls, g = "General", grids["General__grid"]
    ix = min(max(int(np.digitize(phi, edges) - 1), 0), nb - 1)
    iy = min(max(int(np.digitize(psi, edges) - 1), 0), nb - 1)
    d = float(g[ix, iy])
    if d >= float(grids[f"{cls}__fav"]):
        return "favored"
    if d >= float(grids[f"{cls}__allow"]):
        return "allowed"
    return "outlier"


def create_ramachandran_figure(
    phi_psi_data: list,
    label_font: int = 14,
    tick_font: int = 12,
    other_phi_psi: "list | None" = None,
    other_label: str = "ESMFold2",
) -> Figure:
    """Classical Ramachandran plot coloured by secondary structure.
    When other_phi_psi given, overlays a second structure (e.g. ESMFold2) as hollow markers.
    """
    SS_COLORS = {"H": "#1f77b4", "E": "#d62728", "C": "#aaaaaa"}
    SS_LABELS = {"H": "α-Helix", "E": "β-Sheet", "C": "Coil"}

    fig = Figure(figsize=(6.5, 5.5), dpi=120)
    fig.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)

    # Favored / allowed background. Prefer the real MolProbity Top8000 per-class
    # density contours (general-case envelope drawn here; each residue is scored
    # against ITS OWN class grid for outlier detection below). Fall back to an
    # analytical Gaussian-mixture envelope if the bundled grids are unavailable.
    grids = _load_rama_grids()
    if grids is not None:
        _edges = grids["edges"]
        _ctr = (_edges[:-1] + _edges[1:]) / 2
        _PX, _PY = np.meshgrid(_ctr, _ctr, indexing="ij")
        _G = grids["General__grid"]
        _lv = [float(grids["General__allow"]), float(grids["General__fav"])]
        ax.contourf(_PX, _PY, _G, levels=_lv + [1.01],
                    colors=["#c7d0e8", "#9aa8d0"], alpha=0.55, zorder=0)
        ax.contour(_PX, _PY, _G, levels=_lv, colors="#8a98c8",
                   linewidths=0.5, alpha=0.7, zorder=0)
    else:
        _grid = np.linspace(-180, 180, 200)
        _PX, _PY = np.meshgrid(_grid, _grid)
        _basins = [
            (-63, -42, 1.00, 20, 20), (-120, 130, 0.95, 26, 24),
            (-70, 150, 0.55, 20, 18), (-90, -10, 0.35, 24, 22),
            (60, 45, 0.30, 18, 18),
        ]
        _dens = np.zeros_like(_PX)
        for _p0, _q0, _w, _sp, _sq in _basins:
            _dens += _w * np.exp(-(((_PX - _p0) / _sp) ** 2 + ((_PY - _q0) / _sq) ** 2) / 2)
        _dens /= _dens.max()
        ax.contourf(_PX, _PY, _dens, levels=[0.02, 0.12, 1.01],
                    colors=["#c7d0e8", "#9aa8d0"], alpha=0.5, zorder=0)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="-", zorder=1)
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="-", zorder=1)

    def _plot_set(data, filled, label_prefix):
        plotted_ss = set()
        classes = _rama_classes(data) if grids is not None else None
        ox, oy = [], []
        for i, res in enumerate(data):
            phi = res.get("phi")
            psi = res.get("psi")
            ss  = res.get("ss", "C")
            if phi is None or psi is None:
                continue
            color = SS_COLORS.get(ss, "#aaaaaa")
            lbl = f"{label_prefix} {SS_LABELS.get(ss, ss)}" if ss not in plotted_ss else "_nolegend_"
            plotted_ss.add(ss)
            if filled:
                ax.scatter(phi, psi, color=color, s=10, alpha=0.65,
                           edgecolors="none", label=lbl, zorder=3)
            else:
                ax.scatter(phi, psi, facecolors="none", edgecolors=color,
                           s=10, alpha=0.55, linewidths=0.6, label=lbl, zorder=4)
            if classes is not None and _rama_eval(phi, psi, classes[i], grids) == "outlier":
                ox.append(phi); oy.append(psi)
        if ox:
            ax.scatter(ox, oy, facecolors="none", edgecolors="#e63946",
                       s=48, linewidths=1.4, zorder=6, label="_nolegend_")

    _plot_set(phi_psi_data, filled=True,
              label_prefix="AF" if other_phi_psi else "")
    if other_phi_psi:
        _plot_set(other_phi_psi, filled=False, label_prefix=other_label)

    # Per-class outlier statistics for the primary structure (annotation).
    if grids is not None:
        _cl = _rama_classes(phi_psi_data)
        _ev = [_rama_eval(r["phi"], r["psi"], _cl[i], grids)
               for i, r in enumerate(phi_psi_data)
               if r.get("phi") is not None and r.get("psi") is not None]
        if _ev:
            _tot = len(_ev)
            _nf, _na, _no = _ev.count("favored"), _ev.count("allowed"), _ev.count("outlier")
            ax.text(0.02, 0.02,
                    f"Favored {100*_nf/_tot:.1f}%  ·  Allowed {100*_na/_tot:.1f}%  ·  "
                    f"Outliers {_no} ({100*_no/_tot:.1f}%)",
                    transform=ax.transAxes, fontsize=max(6, tick_font - 3),
                    va="bottom", ha="left", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.3", fc="#ffffff",
                              ec="#d0d4e0", alpha=0.85))

    fav_patch = mpatches.Patch(color="#9aa8d0", alpha=0.7, label="Favored")
    alw_patch = mpatches.Patch(color="#c7d0e8", alpha=0.7, label="Allowed")
    out_handle = mlines.Line2D([], [], marker="o", linestyle="none",
                               markerfacecolor="none", markeredgecolor="#e63946",
                               markersize=7, markeredgewidth=1.4, label="Outlier")

    handles, labels = ax.get_legend_handles_labels()
    visible = [(h, l) for h, l in zip(handles, labels) if not l.startswith("_")]
    visible_h, visible_l = zip(*visible) if visible else ([], [])
    _extra_h = [fav_patch, alw_patch] + ([out_handle] if grids is not None else [])
    _extra_l = ["Favored", "Allowed"] + (["Outlier"] if grids is not None else [])
    ax.legend(
        list(visible_h) + _extra_h,
        list(visible_l) + _extra_l,
        fontsize=max(6, tick_font - 3), loc="lower right", markerscale=1.2,
        framealpha=0.85, edgecolor="#d0d4e0",
        handlelength=1.2, borderpad=0.5, labelspacing=0.3,
    )

    title = "Ramachandran Map"
    if other_phi_psi:
        title = "Ramachandran Map — AlphaFold (filled) vs ESMFold2 (hollow)"
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    _pub_style_ax(ax, title=title,
                  xlabel=r"$\phi$ (°)", ylabel=r"$\psi$ (°)",
                  grid=True, despine=True,
                  title_size=label_font,
                  label_size=label_font, tick_size=tick_font)
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks(range(-180, 181, 60))
    fig.tight_layout(pad=1.8)
    return fig


def _draw_contact_network(ax, fig, seq, dist_matrix, cutoff_angstrom, label_font, tick_font, cmap, title):
    """Draw a single contact network onto ax. Returns colorbar mappable."""
    n = len(seq)
    dist_matrix = np.asarray(dist_matrix, dtype=float)
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 2, n):
            if dist_matrix[i, j] <= cutoff_angstrom:
                adj[i, j] = True
                adj[j, i] = True
    degree = adj.sum(axis=1).astype(float)
    _max_d = degree.max() if n > 0 else 0.0
    max_deg = _max_d if _max_d > 0 else 1.0
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
    _cmap = matplotlib.colormaps[cmap]
    norm = mcolors.Normalize(vmin=0, vmax=centrality.max() if centrality.max() > 0 else 1)
    for i in range(sub_n):
        for j in range(i + 1, sub_n):
            if plot_adj[i, j]:
                orig_i = top_idx[i] if large_protein else i
                orig_j = top_idx[j] if large_protein else j
                d = dist_matrix[orig_i, orig_j]
                lw = max(0.2, 2.0 / (d + 1e-9) * cutoff_angstrom)
                ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                        color="lightgrey", linewidth=min(lw, 1.6), zorder=1)
    node_sizes = 45 + (node_degrees / max_deg) * 200
    scatter = ax.scatter(pos[:, 0], pos[:, 1], c=centrality, cmap=cmap, norm=norm,
                         s=node_sizes, zorder=3, edgecolors="black", linewidths=0.3)
    if not large_protein and n <= 50:
        for i in range(sub_n):
            ax.text(pos[i, 0], pos[i, 1], residue_labels[i],
                    ha="center", va="center", fontsize=max(tick_font - 4, 6), zorder=4)
    ax.set_aspect("equal")
    ax.axis("off")
    subtitle = f" — top {top_n} by degree" if large_protein else ""
    ax.set_title(f"{title}{subtitle}", fontsize=label_font - 1, fontweight="bold",
                 color="#1a1a2e", pad=8)
    return scatter


def create_contact_network_figure(
    seq: str,
    dist_matrix: np.ndarray,
    cutoff_angstrom: float = 8.0,
    label_font: int = 14,
    tick_font: int = 12,
    cmap: str = "viridis",
    other_dist_matrix: "np.ndarray | None" = None,
    other_label: str = "ESMFold2",
) -> Figure:
    """Residue contact network derived from Ca distance matrix."""
    n = len(seq)
    if n < 2:
        fig = Figure(figsize=(7, 4), dpi=120)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "Sequence too short for contact network",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        return fig
    dim = max(6, min(9, 6 + n * 0.015))
    if other_dist_matrix is not None:
        other_dist_matrix = np.asarray(other_dist_matrix, dtype=float)
        fig = Figure(figsize=(dim * 2 + 0.5, dim), dpi=120)
        fig.set_facecolor("#ffffff")
        fig._beer_manual_layout = True
        ax1 = fig.add_axes([0.03, 0.05, 0.42, 0.88])
        ax2 = fig.add_axes([0.53, 0.05, 0.42, 0.88])
        sc1 = _draw_contact_network(ax1, fig, seq, dist_matrix,       cutoff_angstrom, label_font, tick_font, cmap, "AlphaFold — Contact Network")
        sc2 = _draw_contact_network(ax2, fig, seq, other_dist_matrix, cutoff_angstrom, label_font, tick_font, cmap, f"{other_label} — Contact Network")
        cax = fig.add_axes([0.96, 0.15, 0.015, 0.65])
        cbar = fig.colorbar(sc2, cax=cax)
        cbar.set_label("Degree centrality", fontsize=label_font - 2)
        cbar.ax.tick_params(labelsize=tick_font - 2)
        return fig
    fig = Figure(figsize=(dim, dim), dpi=120)
    fig.set_facecolor("#ffffff")
    fig._beer_manual_layout = True
    ax = fig.add_axes([0.05, 0.05, 0.80, 0.88])
    scatter = _draw_contact_network(ax, fig, seq, dist_matrix, cutoff_angstrom,
                                    label_font, tick_font, cmap, "Contact Map")
    cax = fig.add_axes([0.87, 0.15, 0.03, 0.65])
    cbar = fig.colorbar(scatter, cax=cax)
    cbar.set_label("Degree centrality", fontsize=label_font - 2)
    cbar.ax.tick_params(labelsize=tick_font - 2)
    return fig


def create_plddt_figure(
    plddt: list,
    label_font: int = 14,
    tick_font: int = 12,
    use_bfactor: bool = False,
    source: str = "alphafold",
    other_plddt: "list | None" = None,
    other_label: str = "ESMFold2",
) -> Figure:
    """Per-residue confidence/B-factor profile with coloured zones.

    source: "alphafold" → pLDDT title/ylabel
            "esmfold2"  → ESMFold2 Confidence title/ylabel (same 0-100 scale)
            "pdb"       → B-Factor title/ylabel (Å²)
    use_bfactor is kept for backward compatibility (True == source "pdb").
    """
    if use_bfactor:
        source = "pdb"
    n = len(plddt)
    xs = list(range(1, n + 1))
    w = max(9, min(16, 9 + n * 0.015))
    fig = Figure(figsize=(w, 4.5), dpi=120)
    fig.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)

    if source == "pdb":
        title  = "B-Factor Profile"
        ylabel = "B-Factor (Å²)"
        vmax   = max(100.0, float(max(plddt))) if plddt else 100.0
        ax.axhspan(0,    20,   alpha=0.14, color="#0053D6")
        ax.axhspan(20,   40,   alpha=0.14, color="#65CBF3")
        ax.axhspan(40,   60,   alpha=0.14, color="#FFDB13")
        ax.axhspan(60,   vmax, alpha=0.14, color="#FF7D45")
        for i in range(n - 1):
            mid = (plddt[i] + plddt[i + 1]) / 2
            if mid < 20:    seg_col = "#0053D6"
            elif mid < 40:  seg_col = "#65CBF3"
            elif mid < 60:  seg_col = "#FFDB13"
            else:           seg_col = "#FF7D45"
            ax.plot([xs[i], xs[i + 1]], [plddt[i], plddt[i + 1]],
                    color=seg_col, linewidth=2.5, zorder=4, solid_capstyle="round")
        ax.plot(xs, plddt, color="black", linewidth=0.5, alpha=0.35, zorder=5)
        for thresh, col in [(20, "#0053D6"), (40, "#65CBF3"), (60, "#FFDB13")]:
            ax.axhline(thresh, color=col, linewidth=0.9, linestyle="--", alpha=0.9)
        ax.set_ylim(0, vmax * 1.05)
        ax.legend(handles=[
            Patch(color="#0053D6", alpha=0.5, label="Rigid / well-ordered (< 20 Å²)"),
            Patch(color="#65CBF3", alpha=0.5, label="Ordered (20–40 Å²)"),
            Patch(color="#FFDB13", alpha=0.5, label="Mobile (40–60 Å²)"),
            Patch(color="#FF7D45", alpha=0.5, label="Flexible / disordered (> 60 Å²)"),
        ], fontsize=tick_font - 3, framealpha=0.85, edgecolor="#d0d4e0",
           loc="upper right")
    elif source == "esmfold2":
        title  = "ESMFold2 Confidence Score"
        ylabel = "Confidence (0–100)"
        ax.axhspan(90, 100, alpha=0.14, color="#0053D6")
        ax.axhspan(70,  90, alpha=0.14, color="#65CBF3")
        ax.axhspan(50,  70, alpha=0.14, color="#FFDB13")
        ax.axhspan(0,   50, alpha=0.14, color="#FF7D45")
        for i in range(n - 1):
            mid = (plddt[i] + plddt[i + 1]) / 2
            if mid >= 90:   seg_col = "#0053D6"
            elif mid >= 70: seg_col = "#65CBF3"
            elif mid >= 50: seg_col = "#FFDB13"
            else:           seg_col = "#FF7D45"
            ax.plot([xs[i], xs[i + 1]], [plddt[i], plddt[i + 1]],
                    color=seg_col, linewidth=2.5, zorder=4, solid_capstyle="round")
        ax.plot(xs, plddt, color="black", linewidth=0.5, alpha=0.35, zorder=5)
        for thresh, col in [(90, "#0053D6"), (70, "#65CBF3"), (50, "#FFDB13")]:
            ax.axhline(thresh, color=col, linewidth=0.9, linestyle="--", alpha=0.9)
        ax.set_ylim(0, 100)
        ax.legend(handles=[
            Patch(color="#0053D6", alpha=0.5, label="Very high (≥ 90)"),
            Patch(color="#65CBF3", alpha=0.5, label="Confident (70–90)"),
            Patch(color="#FFDB13", alpha=0.5, label="Low (50–70)"),
            Patch(color="#FF7D45", alpha=0.5, label="Very low (< 50)"),
        ], fontsize=tick_font - 3, framealpha=0.85, edgecolor="#d0d4e0",
           loc="lower right")
    else:  # "alphafold" (default)
        title  = "pLDDT Confidence Score"
        ylabel = "pLDDT Score"
        ax.axhspan(90, 100, alpha=0.14, color="#0053D6")
        ax.axhspan(70,  90, alpha=0.14, color="#65CBF3")
        ax.axhspan(50,  70, alpha=0.14, color="#FFDB13")
        ax.axhspan(0,   50, alpha=0.14, color="#FF7D45")
        for i in range(n - 1):
            mid = (plddt[i] + plddt[i + 1]) / 2
            if mid >= 90:   seg_col = "#0053D6"
            elif mid >= 70: seg_col = "#65CBF3"
            elif mid >= 50: seg_col = "#FFDB13"
            else:           seg_col = "#FF7D45"
            ax.plot([xs[i], xs[i + 1]], [plddt[i], plddt[i + 1]],
                    color=seg_col, linewidth=2.5, zorder=4, solid_capstyle="round")
        ax.plot(xs, plddt, color="black", linewidth=0.5, alpha=0.35, zorder=5)
        for thresh, col in [(90, "#0053D6"), (70, "#65CBF3"), (50, "#FFDB13")]:
            ax.axhline(thresh, color=col, linewidth=0.9, linestyle="--", alpha=0.9)
        ax.set_ylim(0, 100)
        ax.legend(handles=[
            Patch(color="#0053D6", alpha=0.5, label="Very high (≥ 90)"),
            Patch(color="#65CBF3", alpha=0.5, label="Confident (70–90)"),
            Patch(color="#FFDB13", alpha=0.5, label="Low (50–70)"),
            Patch(color="#FF7D45", alpha=0.5, label="Very low (< 50)"),
        ], fontsize=tick_font - 3, framealpha=0.85, edgecolor="#d0d4e0",
           loc="lower right")

    # Overlay second series when both structures loaded
    if other_plddt and source != "pdb":
        _o = _norm100(list(other_plddt))
        _ox = list(range(1, len(_o) + 1))
        ax.plot(_ox, _o, color="#E65100", linewidth=1.8, alpha=0.80,
                linestyle="--", zorder=6, label=other_label)
        ax.plot(xs, plddt, color="#1565C0", linewidth=1.8, alpha=0.80,
                zorder=6, label="AlphaFold")
        ax.legend(fontsize=tick_font - 3, framealpha=0.88,
                  edgecolor="#d0d4e0", loc="lower right")
        title = "Confidence Score — AlphaFold vs ESMFold2"

    _pub_style_ax(ax, title=title,
                  xlabel="Residue Position", ylabel=ylabel,
                  grid=False, title_size=label_font,
                  label_size=label_font, tick_size=tick_font)
    ax.set_xlim(1, n)
    fig.tight_layout(pad=1.8)
    return fig


def _norm100(vals: list) -> list:
    """Normalise a confidence list to 0-100. Handles both 0-1 and 0-100 inputs."""
    if not vals:
        return vals
    return [v * 100.0 for v in vals] if max(vals) <= 1.0 else list(vals)


def create_structure_comparison_figure(
    af_plddt: list,
    esm_plddt: list,
    rmsd_per_res: list | None = None,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Per-residue Cα RMSD between AlphaFold and ESMFold2 after Kabsch superposition."""
    if not rmsd_per_res:
        fig = Figure(figsize=(9, 4), dpi=120)
        fig.set_facecolor("#ffffff")
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No RMSD data — load both AlphaFold and ESMFold2 structures",
                ha="center", va="center", transform=ax.transAxes,
                color="#888888", fontsize=12)
        fig.tight_layout(pad=1.8)
        return fig

    n = len(rmsd_per_res)
    w = max(9, min(16, 9 + n * 0.015))
    fig = Figure(figsize=(w, 4.5), dpi=120)
    fig.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)

    rmsd_xs = list(range(1, n + 1))

    def _isnan(x): return x != x

    for i in range(n - 1):
        a, b = rmsd_per_res[i], rmsd_per_res[i + 1]
        if _isnan(a) or _isnan(b):
            continue
        v = (a + b) / 2
        col = "#2e7d32" if v < 2.0 else ("#f9a825" if v < 4.0 else "#c62828")
        ax.plot([rmsd_xs[i], rmsd_xs[i + 1]], [a, b],
                color=col, linewidth=2.2, solid_capstyle="round", zorder=3)

    ax.fill_between(rmsd_xs,
                    [0 if _isnan(v) else v for v in rmsd_per_res],
                    alpha=0.12, color="#607d8b", zorder=2)
    ax.axhline(2.0, color="#f9a825", linewidth=0.9, linestyle="--", alpha=0.85)
    ax.axhline(4.0, color="#c62828", linewidth=0.9, linestyle="--", alpha=0.85)
    ax.set_xlim(1, n)
    ax.set_ylim(bottom=0)

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], color="#2e7d32", lw=2.2, label="< 2 Å  (similar)"),
        Line2D([0], [0], color="#f9a825", lw=2.2, label="2–4 Å  (moderate)"),
        Line2D([0], [0], color="#c62828", lw=2.2, label="> 4 Å  (divergent)"),
    ], fontsize=tick_font - 3, framealpha=0.85, edgecolor="#d0d4e0", loc="upper right")

    _pub_style_ax(ax, title="Cα RMSD — AlphaFold vs ESMFold2 (Kabsch superposition)",
                  xlabel="Residue Position", ylabel="Cα RMSD (Å)",
                  grid=False, title_size=label_font,
                  label_size=label_font, tick_size=tick_font)
    fig.tight_layout(pad=1.8)
    return fig


def create_sasa_figure(
    rsa_dict: dict[int, float],
    asa_dict: dict[int, float],
    window: int = 9,
    show_asa: bool = False,
    label_font: int = 14,
    tick_font: int = 12,
    other_rsa: "dict | None" = None,
    other_asa: "dict | None" = None,
    other_label: str = "ESMFold2",
) -> Figure:
    """Per-residue solvent accessibility profile with smoothing.

    Args:
        rsa_dict: {PDB resi number: RSA 0..1}
        asa_dict: {PDB resi number: absolute ASA in Å²}
        window:   smoothing window (from Settings, same as other profiles)
        show_asa: if True plot raw ASA (Å²); if False plot RSA (dimensionless 0–1)
    """
    if not rsa_dict:
        fig = Figure(figsize=(9, 4), dpi=120)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No structure loaded", ha="center", va="center",
                transform=ax.transAxes, color="#888888", fontsize=12)
        fig.tight_layout()
        return fig

    src = asa_dict if show_asa else rsa_dict
    if not src:
        # e.g. ASA requested but only RSA available — fall back to RSA.
        src = rsa_dict
        show_asa = False
    resi_nums = sorted(src.keys())
    vals = np.array([src[r] for r in resi_nums], dtype=float)
    xs   = np.arange(1, len(vals) + 1)   # sequential 1-based position for x-axis

    # Colour each segment by burial level using a vivid blue→orange diverging map
    _CMAP = matplotlib.colormaps["RdYlBu_r"]   # blue=buried, red=exposed — intuitive for accessibility

    w = max(9, min(16, 9 + len(vals) * 0.015))
    fig = Figure(figsize=(w, 4.5), dpi=120)
    fig.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)

    if show_asa:
        ylabel = "ASA (Å²)"
        title  = "Solvent-Accessible Surface Area"
        vmax = float(np.percentile(vals, 98)) if len(vals) > 2 else float(vals.max()) or 1.0
        norm = mcolors.Normalize(vmin=0, vmax=vmax)
        ax.fill_between(xs, vals, alpha=0.15, color="#3b82f6", linewidth=0)
    else:
        ylabel = "Relative Solvent Accessibility (RSA)"
        title  = "Relative Solvent Accessibility"
        norm = mcolors.Normalize(vmin=0, vmax=1)
        # Clear burial-zone bands
        ax.axhspan(0.0,  0.20, alpha=0.12, color="#1e3a8a", zorder=0)   # buried  (deep blue)
        ax.axhspan(0.20, 0.50, alpha=0.08, color="#fbbf24", zorder=0)   # partial (amber)
        ax.axhspan(0.50, 1.0,  alpha=0.10, color="#dc2626", zorder=0)   # exposed (red)

    # Raw profile (thin, grey)
    ax.plot(xs, vals, color="#94a3b8", linewidth=0.8, alpha=0.45, zorder=2)

    # Smoothed profile — coloured per-segment by burial level
    half = max(1, window // 2)
    kern = np.ones(window) / window
    if len(vals) >= window:
        smooth = np.convolve(vals, kern, mode="same")
        for i in range(half):
            smooth[i]        = vals[:i + half + 1].mean()
            smooth[-(i + 1)] = vals[-(i + half + 1):].mean()
    else:
        smooth = vals.copy()

    # Draw coloured line segments
    ref = smooth if not show_asa else vals
    for i in range(len(xs) - 1):
        seg_val = (ref[i] + ref[i + 1]) / 2
        seg_col = _CMAP(norm(seg_val))
        ax.plot(xs[i:i+2], smooth[i:i+2], color=seg_col,
                linewidth=2.5, zorder=3, solid_capstyle="round")
    # Narrow dark overlay (matches pLDDT style — clarifies profile against coloured background)
    ax.plot(xs, smooth, color="black", linewidth=0.5, alpha=0.30, zorder=4)

    if not show_asa:
        ax.axhline(0.20, color="#1e3a8a", linewidth=1.0, linestyle="--", alpha=0.8)
        ax.axhline(0.50, color="#dc2626", linewidth=1.0, linestyle="--", alpha=0.8)
        ax.set_ylim(-0.02, 1.05)

    # Overlay second structure if provided
    if other_rsa or other_asa:
        _o_src = (other_asa if show_asa else other_rsa) or {}
        if _o_src:
            _o_resi = sorted(_o_src.keys())
            _o_vals = np.array([_o_src[r] for r in _o_resi], dtype=float)
            _o_xs   = np.arange(1, len(_o_vals) + 1)
            _o_kern = np.ones(window) / window
            _o_sm   = np.convolve(_o_vals, _o_kern, mode="same") if len(_o_vals) >= window else _o_vals.copy()
            ax.plot(_o_xs, _o_sm, color="#E65100", linewidth=1.8,
                    linestyle="--", alpha=0.80, zorder=5, label=other_label)
            ax.plot(xs, smooth, color="#1565C0", linewidth=1.8,
                    alpha=0.80, zorder=5, label="AlphaFold")
            title = f"{title} — AlphaFold vs {other_label}"

    if not show_asa:
        ax.legend(handles=[
            Patch(color="#1e3a8a", alpha=0.55, label="Buried (RSA < 0.20)"),
            Patch(color="#fbbf24", alpha=0.65, label="Partial (0.20 – 0.50)"),
            Patch(color="#dc2626", alpha=0.50, label="Exposed (RSA > 0.50)"),
        ], fontsize=tick_font - 3, framealpha=0.88, edgecolor="#d0d4e0", loc="upper right")

    _pub_style_ax(ax, title=title, xlabel="Residue Position", ylabel=ylabel,
                  grid=False, title_size=label_font,
                  label_size=label_font, tick_size=tick_font)
    ax.set_xlim(1, len(vals))
    fig.tight_layout(pad=1.8)
    return fig


def _draw_dist_map(ax, dist_matrix, cmap, tick_font, title, vmin=0, vmax=None):
    """Draw a single Cα distance heatmap onto ax."""
    if vmax is None:
        vmax = min(40, float(dist_matrix.max())) if dist_matrix.size > 0 else 40.0
    im = ax.imshow(dist_matrix, cmap=cmap, aspect="auto",
                   origin="upper", interpolation="nearest",
                   vmin=vmin, vmax=vmax)
    ax.contour(dist_matrix, levels=[8.0], colors=["#f72585"],
               linewidths=[0.6], alpha=0.7)
    ax.set_title(title, fontsize=tick_font, fontweight="bold", color="#1a1a2e", pad=6)
    ax.set_xlabel("Residue", fontsize=tick_font - 1, color="#4a5568")
    ax.set_ylabel("Residue", fontsize=tick_font - 1, color="#4a5568")
    ax.tick_params(labelsize=tick_font - 2, colors="#4a5568")
    return im


def create_distance_map_figure(
    dist_matrix: np.ndarray,
    label_font: int = 14,
    tick_font: int = 12,
    cmap: str = "viridis_r",
    other_dist_matrix: "np.ndarray | None" = None,
    other_label: str = "ESMFold2",
) -> Figure:
    """Cα pairwise distance heatmap. When other_dist_matrix given, shows side-by-side."""
    dist_matrix = np.asarray(dist_matrix, dtype=float)
    n = dist_matrix.shape[0] if dist_matrix.ndim == 2 else 0
    if n < 2:
        fig = Figure(figsize=(7, 4), dpi=120)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "Structure too short for a distance map",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        return fig
    dim = max(5.5, min(9.0, 5.0 + n * 0.025))
    if other_dist_matrix is not None:
        other_dist_matrix = np.asarray(other_dist_matrix, dtype=float)
        shared_vmax = min(40, max(
            float(dist_matrix.max()) if dist_matrix.size > 0 else 0,
            float(other_dist_matrix.max()) if other_dist_matrix.size > 0 else 0,
        ))
        from matplotlib.gridspec import GridSpec
        fig = Figure(figsize=(dim * 2 + 0.7, dim), dpi=120)
        fig.set_facecolor("#ffffff")
        gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 0.045],
                      wspace=0.08, left=0.07, right=0.92, top=0.92, bottom=0.1)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1, sharey=ax1)
        cax = fig.add_subplot(gs[2])
        _draw_dist_map(ax1, dist_matrix,       cmap, tick_font, "AlphaFold — Cα Distance Map",
                       vmax=shared_vmax)
        im2 = _draw_dist_map(ax2, other_dist_matrix, cmap, tick_font,
                              f"{other_label} — Cα Distance Map", vmax=shared_vmax)
        ax2.tick_params(labelleft=False)
        ax2.set_ylabel("")
        cbar = fig.colorbar(im2, cax=cax)
        cbar.set_label("Cα Distance (Å)", fontsize=tick_font - 1, color="#4a5568")
        cbar.ax.tick_params(labelsize=tick_font - 2, colors="#4a5568")
        return fig
    fig = Figure(figsize=(dim + 1.0, dim), dpi=120)
    fig.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#fafbff")
    im = _draw_dist_map(ax, dist_matrix, cmap, tick_font, "Cα Distance Map")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, aspect=20, pad=0.02)
    cbar.set_label("Cα Pairwise Distance (Å)", fontsize=tick_font - 1, color="#4a5568")
    cbar.ax.tick_params(labelsize=tick_font - 2, colors="#4a5568")
    fig.tight_layout(pad=1.8)
    return fig
