"""Structural analysis figures: Ramachandran, contact network, pLDDT, distance map, SS bead."""
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

from beer.graphs._style import _pub_style_ax


def _bead_width_struct(n: int) -> float:
    return max(8.0, min(22.0, 4.0 + n * 0.08))


def _x_tick_step_struct(n: int) -> int:
    if n <= 50:   return 10
    if n <= 150:  return 25
    if n <= 400:  return 50
    return 100


def create_bead_model_ss_figure(
    pdb_str: str,
    show_labels: bool = True,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Linear bead model coloured by secondary structure (helix/sheet/coil) from PDB records."""
    # Parse HELIX and SHEET records
    helix_res: set[tuple[str, int]] = set()
    sheet_res: set[tuple[str, int]] = set()
    for line in pdb_str.splitlines():
        if line.startswith("HELIX "):
            try:
                chain = line[19:20].strip()
                start = int(line[21:25])
                end   = int(line[33:37])
                for r in range(start, end + 1):
                    helix_res.add((chain, r))
            except (ValueError, IndexError):
                pass
        elif line.startswith("SHEET "):
            try:
                chain = line[21:22].strip()
                start = int(line[22:26])
                end   = int(line[33:37])
                for r in range(start, end + 1):
                    sheet_res.add((chain, r))
            except (ValueError, IndexError):
                pass

    # Collect ordered residues from ATOM records (first model, all chains)
    seen: dict[tuple[str, int], str] = {}
    for line in pdb_str.splitlines():
        if line.startswith("END"):
            break
        if not line.startswith("ATOM  "):
            continue
        try:
            chain = line[21:22].strip()
            resi  = int(line[22:26])
            resn  = line[17:20].strip()
            key   = (chain, resi)
            if key not in seen:
                seen[key] = resn
        except (ValueError, IndexError):
            pass

    if not seen:
        fig = Figure(figsize=(8, 3), dpi=120)
        fig.set_facecolor("#ffffff")
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No ATOM records found in PDB",
                ha="center", va="center", transform=ax.transAxes, fontsize=label_font)
        return fig

    keys   = list(seen.keys())
    resns  = [seen[k] for k in keys]
    n      = len(keys)
    xs     = list(range(1, n + 1))

    HELIX_C = "#FF6666"
    SHEET_C = "#FFD700"
    COIL_C  = "#aaaaaa"

    cols = []
    for key in keys:
        if key in helix_res:
            cols.append(HELIX_C)
        elif key in sheet_res:
            cols.append(SHEET_C)
        else:
            cols.append(COIL_C)

    fig = Figure(figsize=(_bead_width_struct(n), 4.5), dpi=120)
    fig.set_facecolor("#ffffff")
    ax  = fig.add_subplot(111)
    ax.scatter(xs, [1] * n, c=cols, s=220, linewidths=0.5, edgecolors="white", zorder=4)

    legend_patches = [
        Patch(color=HELIX_C, label="α-Helix"),
        Patch(color=SHEET_C, label="β-Sheet"),
        Patch(color=COIL_C,  label="Coil / Loop"),
    ]
    ax.legend(handles=legend_patches, loc="upper right",
              fontsize=max(7, tick_font - 3), framealpha=0.85, edgecolor="#d1d5db")

    ax.set_yticks([])
    ax.set_xlim(0, n + 1)
    ax.set_ylim(0.3, 1.7)

    if show_labels and n <= 60:
        for i, resn in enumerate(resns):
            aa = resn[0] if len(resn) == 1 else resn
            ax.text(xs[i], 1, aa[:1], ha="center", va="center",
                    fontsize=max(5, label_font - 5),
                    color="white" if cols[i] == HELIX_C else "#333333",
                    fontweight="bold")

    _step = _x_tick_step_struct(n)
    ax.set_xticks(range(_step, n + 1, _step))
    ax.tick_params(labelsize=tick_font - 2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    _pub_style_ax(ax, title="Secondary Structure Bead Model",
                  xlabel="Residue Position", grid=False, despine=False,
                  title_size=label_font - 1, label_size=label_font - 2,
                  tick_size=tick_font - 2)
    fig.tight_layout(pad=1.8)
    return fig


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


def create_ramachandran_figure(
    phi_psi_data: list,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Classical Ramachandran plot coloured by secondary structure."""
    SS_COLORS = {"H": "#1f77b4", "E": "#d62728", "C": "#aaaaaa"}
    SS_LABELS = {"H": "α-Helix", "E": "β-Sheet", "C": "Coil"}

    fig = Figure(figsize=(6.5, 5.5), dpi=120)
    fig.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)

    ax.add_patch(Rectangle((-80, -60), 32, 40, color="#555555", alpha=0.25,
                            zorder=0))
    ax.add_patch(Rectangle((-150, 90), 60, 70, color="#888888", alpha=0.20,
                            zorder=0))
    ax.add_patch(Rectangle((40, 20), 40, 40, color="#bbbbbb", alpha=0.20,
                            zorder=0))

    ax.axhline(0, color="grey", linewidth=0.5, linestyle="-")
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="-")

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
        ax.scatter(phi, psi, color=color, s=10, alpha=0.65,
                   edgecolors="none", label=label, zorder=3)

    helix_patch = mpatches.Patch(color="#555555", alpha=0.4, label="Core α-helix")
    sheet_patch = mpatches.Patch(color="#888888", alpha=0.35, label="Core β-sheet")
    lh_patch = mpatches.Patch(color="#bbbbbb", alpha=0.35, label="LH helix")

    handles, labels = ax.get_legend_handles_labels()
    visible = [(h, l) for h, l in zip(handles, labels) if not l.startswith("_")]
    visible_h, visible_l = zip(*visible) if visible else ([], [])
    ax.legend(
        list(visible_h) + [helix_patch, sheet_patch, lh_patch],
        list(visible_l) + ["Core α-helix", "Core β-sheet", "LH helix"],
        fontsize=max(6, tick_font - 3), loc="lower right", markerscale=1.2,
        framealpha=0.85, edgecolor="#d0d4e0",
        handlelength=1.2, borderpad=0.5, labelspacing=0.3,
    )

    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    _pub_style_ax(ax, title="Ramachandran Plot",
                  xlabel=r"$\phi$ (°)", ylabel=r"$\psi$ (°)",
                  grid=True, despine=True,
                  title_size=label_font - 1,
                  label_size=label_font - 1, tick_size=tick_font - 1)
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks(range(-180, 181, 60))
    fig.tight_layout(pad=1.8)
    return fig


def create_contact_network_figure(
    seq: str,
    dist_matrix: np.ndarray,
    cutoff_angstrom: float = 8.0,
    label_font: int = 14,
    tick_font: int = 12,
    cmap: str = "viridis",
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

    # Scale figure with n, capped; extra width for colorbar via subplots_adjust
    dim = max(6, min(9, 6 + n * 0.015))
    fig = Figure(figsize=(dim, dim), dpi=120)
    fig.set_facecolor("#ffffff")
    ax = fig.add_axes([0.05, 0.05, 0.80, 0.88])

    _cmap = cm.get_cmap(cmap)
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
    scatter = ax.scatter(
        pos[:, 0], pos[:, 1],
        c=centrality, cmap=cmap, norm=norm,
        s=node_sizes, zorder=3, edgecolors="black", linewidths=0.3,
    )

    if not large_protein and n <= 50:
        for i in range(sub_n):
            ax.text(pos[i, 0], pos[i, 1], residue_labels[i],
                    ha="center", va="center",
                    fontsize=max(tick_font - 4, 6), zorder=4)

    cax = fig.add_axes([0.87, 0.15, 0.03, 0.65])
    cbar = fig.colorbar(scatter, cax=cax)
    cbar.set_label("Degree centrality", fontsize=label_font - 2)
    cbar.ax.tick_params(labelsize=tick_font - 2)

    ax.set_aspect("equal")
    ax.axis("off")
    subtitle = f" — top {top_n} by degree" if large_protein else ""
    ax.set_title(
        f"Contact Network  (Cα ≤ {cutoff_angstrom} Å){subtitle}",
        fontsize=label_font - 1, fontweight="bold", color="#1a1a2e", pad=8,
    )
    return fig


def create_plddt_figure(
    plddt: list,
    label_font: int = 14,
    tick_font: int = 12,
    use_bfactor: bool = False,
) -> Figure:
    """Per-residue pLDDT or B-factor profile with coloured confidence zones.

    Set use_bfactor=True when the structure comes from PDB (crystallographic B-factors).
    """
    n = len(plddt)
    xs = list(range(1, n + 1))
    w = max(9, min(16, 9 + n * 0.015))
    fig = Figure(figsize=(w, 4.5), dpi=120)
    fig.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)

    if use_bfactor:
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
            Patch(color="#0053D6", alpha=0.5, label="Very low (< 20 Å²)"),
            Patch(color="#65CBF3", alpha=0.5, label="Low (20–40 Å²)"),
            Patch(color="#FFDB13", alpha=0.5, label="Medium (40–60 Å²)"),
            Patch(color="#FF7D45", alpha=0.5, label="High (> 60 Å²)"),
        ], fontsize=tick_font - 3, framealpha=0.85, edgecolor="#d0d4e0",
           loc="upper right")
    else:
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

    _pub_style_ax(ax, title=title,
                  xlabel="Residue Position", ylabel=ylabel,
                  grid=False, title_size=label_font - 1,
                  label_size=label_font - 1, tick_size=tick_font - 1)
    ax.set_xlim(1, n)
    fig.tight_layout(pad=1.8)
    return fig


def create_sasa_figure(
    rsa_dict: dict[int, float],
    asa_dict: dict[int, float],
    window: int = 9,
    show_asa: bool = False,
    label_font: int = 14,
    tick_font: int = 12,
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
    resi_nums = sorted(src.keys())
    vals = np.array([src[r] for r in resi_nums], dtype=float)
    xs   = np.arange(1, len(vals) + 1)   # sequential 1-based position for x-axis

    # Colour each segment by burial level using a vivid blue→orange diverging map
    _CMAP = cm.get_cmap("RdYlBu_r")   # blue=buried, red=exposed — intuitive for accessibility

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
        ax.legend(handles=[
            Patch(color="#1e3a8a", alpha=0.55, label="Buried (RSA < 0.20)"),
            Patch(color="#fbbf24", alpha=0.65, label="Partial (0.20 – 0.50)"),
            Patch(color="#dc2626", alpha=0.50, label="Exposed (RSA > 0.50)"),
        ], fontsize=tick_font - 3, framealpha=0.88, edgecolor="#d0d4e0", loc="upper right")

    _pub_style_ax(ax, title=title, xlabel="Residue Position", ylabel=ylabel,
                  grid=False, title_size=label_font - 1,
                  label_size=label_font - 1, tick_size=tick_font - 1)
    ax.set_xlim(1, len(vals))
    fig.tight_layout(pad=1.8)
    return fig


def create_distance_map_figure(
    dist_matrix: np.ndarray,
    label_font: int = 14,
    tick_font: int = 12,
    cmap: str = "viridis_r",
) -> Figure:
    """Ca pairwise distance heatmap from a loaded PDB structure."""
    n = dist_matrix.shape[0]
    # Scale figure with sequence length, capped
    dim = max(5.5, min(9.0, 5.0 + n * 0.025))
    fig = Figure(figsize=(dim + 1.0, dim), dpi=120)
    fig.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#fafbff")
    im = ax.imshow(dist_matrix, cmap=cmap, aspect="auto",
                   origin="upper", interpolation="nearest",
                   vmin=0, vmax=min(40, dist_matrix.max()))
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, aspect=20, pad=0.02)
    cbar.set_label("Cα Pairwise Distance (Å)", fontsize=tick_font - 1, color="#4a5568")
    cbar.ax.tick_params(labelsize=tick_font - 2, colors="#4a5568")
    ax.contour(dist_matrix, levels=[8.0], colors=["#f72585"],
               linewidths=[0.6], alpha=0.7)
    _pub_style_ax(ax, title=f"Cα Distance Map  ({n} aa)",
                  xlabel="Residue Position", ylabel="Residue Position",
                  grid=False, title_size=label_font - 1,
                  label_size=label_font - 1, tick_size=tick_font - 1)
    fig.tight_layout(pad=1.8)
    return fig
