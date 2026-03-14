"""BEER new_graphs.py — matplotlib Figure-based graph functions.

All functions return a ``matplotlib.figure.Figure`` and accept at minimum
``label_font`` and ``tick_font`` keyword arguments.  No imports from beer.py.
"""

import math
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, FancyArrowPatch, Arc
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import mplcursors

# ---------------------------------------------------------------------------
# Module-level constants (no imports from beer.py)
# ---------------------------------------------------------------------------
AGGREGATION_THRESHOLD = 1.0
SOLUBILITY_NEUTRAL = 0.0
HM_THRESHOLD = 0.35
RBP_THRESHOLD = 0.3

PTM_COLORS = {
    "phospho": "#1f77b4",
    "glycosylation": "#2ca02c",
    "ubiquitination": "#ff7f0e",
    "sumo": "#9467bd",
    "acetylation": "#17becf",
    "methylation": "#e377c2",
    "palmitoylation": "#8c564b",
}

MW_STANDARDS_KDA = [10, 15, 20, 25, 37, 50, 75, 100, 150, 250]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _apply_font_sizes(ax, label_font: int, tick_font: int) -> None:
    """Apply consistent label and tick font sizes to an Axes object."""
    ax.xaxis.label.set_fontsize(label_font)
    ax.yaxis.label.set_fontsize(label_font)
    ax.title.set_fontsize(label_font)
    ax.tick_params(axis="both", labelsize=tick_font)


def _residue_x(seq: str) -> np.ndarray:
    """Return 1-based residue position array for a sequence string."""
    return np.arange(1, len(seq) + 1, dtype=float)


# ---------------------------------------------------------------------------
# 1. Aggregation profile
# ---------------------------------------------------------------------------

def create_aggregation_profile_figure(
    seq: str,
    aggregation_profile: list,
    hotspots: list,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Line plot of per-residue β-aggregation propensity (Zyggregator).

    Parameters
    ----------
    seq:
        Protein sequence string (used for x-axis length).
    aggregation_profile:
        Per-residue aggregation score (same length as seq).
    hotspots:
        List of (start, end) tuples (1-based, inclusive) marking hotspot
        regions to shade in red.
    label_font, tick_font:
        Font sizes for axis labels/titles and tick labels.

    Returns
    -------
    matplotlib.figure.Figure
    """
    x = _residue_x(seq)
    y = np.asarray(aggregation_profile, dtype=float)

    fig = Figure(figsize=(10, 4), tight_layout=True)
    ax = fig.add_subplot(111)

    ax.plot(x, y, color="steelblue", linewidth=1.5, label="Aggregation propensity")

    # Fill orange where y > threshold
    ax.fill_between(x, AGGREGATION_THRESHOLD, y, where=(y > AGGREGATION_THRESHOLD),
                    interpolate=True, color="orange", alpha=0.6,
                    label=f"Above threshold ({AGGREGATION_THRESHOLD})")

    # Dashed threshold line
    ax.axhline(AGGREGATION_THRESHOLD, color="black", linestyle="--", linewidth=1.0,
               label=f"Threshold ({AGGREGATION_THRESHOLD})")

    # Red hotspot rectangles
    y_min, y_max = ax.get_ylim()
    for idx, hs in enumerate(hotspots):
        start, end = hs[0], hs[1]
        label_hs = "Hotspot" if idx == 0 else "_nolegend_"
        rect = Rectangle(
            (start - 0.5, y_min),
            (end - start + 1),
            y_max - y_min,
            linewidth=0,
            edgecolor="none",
            facecolor="red",
            alpha=0.25,
            label=label_hs,
            zorder=0,
        )
        ax.add_patch(rect)

    ax.set_xlabel("Residue Position", fontsize=label_font)
    ax.set_ylabel("β-Aggregation Propensity", fontsize=label_font)
    ax.set_title("β-Aggregation Propensity Profile (Zyggregator)", fontsize=label_font)
    ax.set_xlim(x[0], x[-1])
    ax.tick_params(axis="both", labelsize=tick_font)
    ax.legend(fontsize=tick_font, loc="upper right")

    return fig


# ---------------------------------------------------------------------------
# 2. CamSol solubility profile
# ---------------------------------------------------------------------------

def create_solubility_profile_figure(
    seq: str,
    camsolmt_profile: list,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Per-residue CamSol intrinsic solubility profile.

    Parameters
    ----------
    seq:
        Protein sequence string.
    camsolmt_profile:
        Per-residue CamSol scores (same length as seq).
    label_font, tick_font:
        Font sizes.

    Returns
    -------
    matplotlib.figure.Figure
    """
    x = _residue_x(seq)
    y = np.asarray(camsolmt_profile, dtype=float)

    fig = Figure(figsize=(10, 4), tight_layout=True)
    ax = fig.add_subplot(111)

    ax.plot(x, y, color="black", linewidth=1.2, label="CamSol score")

    ax.fill_between(x, 0, y, where=(y >= 0), interpolate=True,
                    color="green", alpha=0.5, label="Soluble (>0)")
    ax.fill_between(x, 0, y, where=(y < 0), interpolate=True,
                    color="red", alpha=0.5, label="Insoluble (<0)")

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)

    ax.set_xlabel("Residue Position", fontsize=label_font)
    ax.set_ylabel("CamSol Score", fontsize=label_font)
    ax.set_title("CamSol Intrinsic Solubility Profile", fontsize=label_font)
    ax.set_xlim(x[0], x[-1])
    ax.tick_params(axis="both", labelsize=tick_font)
    ax.legend(fontsize=tick_font, loc="upper right")

    return fig


# ---------------------------------------------------------------------------
# 3. Hydrophobic moment profile
# ---------------------------------------------------------------------------

def create_hydrophobic_moment_figure(
    seq: str,
    moment_alpha: list,
    moment_beta: list,
    amphipathic_regions: list,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Hydrophobic moment profile for α-helix and β-strand windows.

    Parameters
    ----------
    seq:
        Protein sequence string.
    moment_alpha:
        μH values for α-helix (δ=100°) per residue position.
    moment_beta:
        μH values for β-strand (δ=160°) per residue position.
    amphipathic_regions:
        List of (start, end) tuples (1-based) marking amphipathic regions.
    label_font, tick_font:
        Font sizes.

    Returns
    -------
    matplotlib.figure.Figure
    """
    x = _residue_x(seq)
    ya = np.asarray(moment_alpha, dtype=float)
    yb = np.asarray(moment_beta, dtype=float)

    fig = Figure(figsize=(10, 4), tight_layout=True)
    ax = fig.add_subplot(111)

    ax.plot(x, ya, color="blue", linewidth=1.5,
            label="μH (α-helix, δ=100°)")
    ax.plot(x, yb, color="red", linewidth=1.5,
            label="μH (β-strand, δ=160°)")

    ax.axhline(HM_THRESHOLD, color="grey", linestyle="--", linewidth=1.0,
               label=f"Threshold ({HM_THRESHOLD})")

    # Green rectangles for amphipathic regions
    y_min, y_max = 0, max(np.max(ya), np.max(yb)) * 1.15 + 0.1
    for idx, region in enumerate(amphipathic_regions):
        start, end = region[0], region[1]
        label_amp = "Amphipathic" if idx == 0 else "_nolegend_"
        rect = Rectangle(
            (start - 0.5, 0),
            (end - start + 1),
            y_max,
            linewidth=0,
            facecolor="green",
            alpha=0.20,
            label=label_amp,
            zorder=0,
        )
        ax.add_patch(rect)

    ax.set_xlabel("Residue Position", fontsize=label_font)
    ax.set_ylabel("μH", fontsize=label_font)
    ax.set_title("Hydrophobic Moment Profile", fontsize=label_font)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="both", labelsize=tick_font)
    ax.legend(fontsize=tick_font, loc="upper right")

    return fig


# ---------------------------------------------------------------------------
# 4. pI / MW 2D gel map
# ---------------------------------------------------------------------------

def create_pI_MW_gel_figure(
    proteins_data: list,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """2D scatter plot of pI vs log10(MW) — SDS-PAGE proxy.

    Parameters
    ----------
    proteins_data:
        List of dicts with keys: ``name``, ``pI``, ``mol_weight`` (Da),
        and optional ``color``.
    label_font, tick_font:
        Font sizes.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = Figure(figsize=(9, 6), tight_layout=True)
    ax = fig.add_subplot(111)

    single = len(proteins_data) == 1

    # MW standard dashed lines
    for mw_kda in MW_STANDARDS_KDA:
        log_mw = math.log10(mw_kda * 1000)
        ax.axhline(log_mw, color="lightgrey", linestyle="--", linewidth=0.8, zorder=0)
        ax.text(13.8, log_mw, f"{mw_kda} kDa", va="center", ha="right",
                fontsize=max(tick_font - 3, 7), color="grey")

    # Vertical reference lines
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
                        s=ms**2, zorder=5, edgecolors="black", linewidths=0.5)
        ax.annotate(
            pdata["name"],
            xy=(pI_val, log_mw),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=max(tick_font - 2, 8),
        )
        scatter_artists.append(sc)

    # mplcursors tooltip showing name, pI, MW
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
            pass  # mplcursors may not function outside a GUI event loop

    # Y-axis ticks: show kDa labels at standard positions
    std_log_ticks = [math.log10(mw * 1000) for mw in MW_STANDARDS_KDA]
    ax.set_yticks(std_log_ticks)
    ax.set_yticklabels([f"{mw} kDa" for mw in MW_STANDARDS_KDA], fontsize=tick_font)

    ax.set_xlabel("Isoelectric Point (pI)", fontsize=label_font)
    ax.set_ylabel("Molecular Weight", fontsize=label_font)
    ax.set_title("pI / MW 2D Map (SDS-PAGE Proxy)", fontsize=label_font)
    ax.set_xlim(0, 14)
    ax.tick_params(axis="x", labelsize=tick_font)

    return fig


# ---------------------------------------------------------------------------
# 5. PTM profile (lollipop)
# ---------------------------------------------------------------------------

def create_ptm_profile_figure(
    seq: str,
    ptm_sites: list,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Lollipop/stem plot of predicted PTM sites.

    Parameters
    ----------
    seq:
        Protein sequence string.
    ptm_sites:
        List of dicts: {``position`` (1-based int), ``ptm_type`` (str)}.
        PTM types: phospho, glycosylation, ubiquitination, sumo,
        acetylation, methylation, palmitoylation.
    label_font, tick_font:
        Font sizes.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Collect all PTM types present
    ptm_types_present = []
    for site in ptm_sites:
        pt = site.get("ptm_type", "unknown")
        if pt not in ptm_types_present:
            ptm_types_present.append(pt)

    # Assign y-index per PTM type
    ptm_y_map = {pt: i for i, pt in enumerate(ptm_types_present)}
    n_types = max(len(ptm_types_present), 1)

    fig = Figure(figsize=(12, max(3, n_types * 1.2 + 1.5)), tight_layout=True)
    ax = fig.add_subplot(111)

    for site in ptm_sites:
        pos = site.get("position", 1)
        pt = site.get("ptm_type", "unknown")
        y_idx = ptm_y_map.get(pt, 0)
        color = PTM_COLORS.get(pt, "#7f7f7f")

        # Vertical stem
        ax.plot([pos, pos], [y_idx - 0.35, y_idx], color=color,
                linewidth=1.0, zorder=2)
        # Dot
        ax.scatter([pos], [y_idx], color=color, s=60, zorder=3,
                   edgecolors="black", linewidths=0.4)

    # Y-axis labels = PTM types
    ax.set_yticks(list(ptm_y_map.values()))
    ax.set_yticklabels(list(ptm_y_map.keys()), fontsize=tick_font)
    ax.set_ylim(-0.8, n_types - 0.2)

    ax.set_xlabel("Residue Position", fontsize=label_font)
    ax.set_ylabel("PTM Type", fontsize=label_font)
    ax.set_title("Post-Translational Modification Sites (Predicted)", fontsize=label_font)
    ax.set_xlim(0, len(seq) + 1)
    ax.tick_params(axis="x", labelsize=tick_font)

    # Legend patches
    legend_patches = [
        mpatches.Patch(color=PTM_COLORS.get(pt, "#7f7f7f"), label=pt)
        for pt in ptm_types_present
    ]
    if legend_patches:
        ax.legend(handles=legend_patches, fontsize=tick_font,
                  loc="upper right", title="PTM Type",
                  title_fontsize=tick_font)

    return fig


# ---------------------------------------------------------------------------
# 6. RNA-binding propensity profile
# ---------------------------------------------------------------------------

def create_rbp_profile_figure(
    seq: str,
    rbp_profile: list,
    rbp_motifs: list,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Sliding-window RNA-binding propensity profile.

    Parameters
    ----------
    seq:
        Protein sequence string.
    rbp_profile:
        Per-residue RBP propensity scores (same length as seq).
    rbp_motifs:
        List of dicts: {``start``, ``end``, ``motif_name``, optional ``color``}.
    label_font, tick_font:
        Font sizes.

    Returns
    -------
    matplotlib.figure.Figure
    """
    x = _residue_x(seq)
    y = np.asarray(rbp_profile, dtype=float)

    fig = Figure(figsize=(10, 4), tight_layout=True)
    ax = fig.add_subplot(111)

    ax.plot(x, y, color="teal", linewidth=1.4, label="RBP propensity")

    ax.fill_between(x, RBP_THRESHOLD, y, where=(y > RBP_THRESHOLD),
                    interpolate=True, color="teal", alpha=0.4,
                    label=f"Above threshold ({RBP_THRESHOLD})")

    ax.axhline(RBP_THRESHOLD, color="black", linestyle="--", linewidth=1.0,
               label=f"Threshold ({RBP_THRESHOLD})")

    # RBP motif colored spans
    motif_colors = [
        "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
        "#911eb4", "#42d4f4", "#f032e6",
    ]
    for i, motif in enumerate(rbp_motifs):
        start = motif.get("start", 1)
        end = motif.get("end", start)
        mcolor = motif.get("color", motif_colors[i % len(motif_colors)])
        mname = motif.get("motif_name", f"Motif {i+1}")
        ax.axvspan(start - 0.5, end + 0.5, color=mcolor, alpha=0.25,
                   label=mname, zorder=0)

    ax.set_xlabel("Residue Position", fontsize=label_font)
    ax.set_ylabel("RBP Propensity", fontsize=label_font)
    ax.set_title("RNA-Binding Propensity Profile", fontsize=label_font)
    ax.set_xlim(x[0], x[-1])
    ax.tick_params(axis="both", labelsize=tick_font)
    ax.legend(fontsize=tick_font, loc="upper right")

    return fig


# ---------------------------------------------------------------------------
# 7. Truncation series analysis (2×3 multi-panel)
# ---------------------------------------------------------------------------

def create_truncation_series_figure(
    truncation_data: dict,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Multi-panel (2×3) figure showing how 6 properties change with truncation.

    Parameters
    ----------
    truncation_data:
        Dict with keys ``n_trunc`` and ``c_trunc``, each a list of dicts:
        {``pct``, ``pI``, ``gravy``, ``fcr``, ``ncpr``, ``net_charge_7``,
        ``disorder_frac``}.
    label_font, tick_font:
        Font sizes.

    Returns
    -------
    matplotlib.figure.Figure
    """
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

    fig = Figure(figsize=(14, 8), tight_layout=True)
    fig.suptitle("Truncation Series Analysis", fontsize=label_font + 2)

    for panel_idx, (key, ylabel) in enumerate(panels):
        ax = fig.add_subplot(2, 3, panel_idx + 1)

        nx, ny = extract(n_trunc, key)
        cx, cy = extract(c_trunc, key)

        if nx:
            ax.plot(nx, ny, color="blue", linewidth=1.5, marker="o",
                    markersize=4, label="N-terminal")
        if cx:
            ax.plot(cx, cy, color="red", linewidth=1.5, marker="s",
                    markersize=4, label="C-terminal")

        ax.set_xlabel("Truncation (%)", fontsize=label_font - 2)
        ax.set_ylabel(ylabel, fontsize=label_font - 2)
        ax.tick_params(axis="both", labelsize=tick_font - 1)
        ax.set_xlim(0, 90)

        if panel_idx == 0:
            ax.legend(fontsize=tick_font - 2, loc="best")

    return fig


# ---------------------------------------------------------------------------
# 8. SCD profile
# ---------------------------------------------------------------------------

def create_scd_profile_figure(
    seq: str,
    scd_profile: list,
    window: int,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Sliding-window SCD (Sequence Charge Decoration) profile.

    Parameters
    ----------
    seq:
        Protein sequence string.
    scd_profile:
        Per-window SCD values.  Length may be <= len(seq).
    window:
        Window size used for the SCD calculation (for title display).
    label_font, tick_font:
        Font sizes.

    Returns
    -------
    matplotlib.figure.Figure
    """
    x = np.arange(1, len(scd_profile) + 1, dtype=float)
    y = np.asarray(scd_profile, dtype=float)

    fig = Figure(figsize=(10, 4), tight_layout=True)
    ax = fig.add_subplot(111)

    ax.plot(x, y, color="black", linewidth=1.2)

    ax.fill_between(x, 0, y, where=(y > 0), interpolate=True,
                    color="red", alpha=0.5, label="Segregated charges (+)")
    ax.fill_between(x, 0, y, where=(y < 0), interpolate=True,
                    color="blue", alpha=0.5, label="Mixed charges (−)")

    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)

    ax.set_xlabel("Residue Position", fontsize=label_font)
    ax.set_ylabel("SCD", fontsize=label_font)
    ax.set_title(
        f"SCD Profile (Sequence Charge Decoration, window={window})",
        fontsize=label_font,
    )
    ax.set_xlim(x[0], x[-1])
    ax.tick_params(axis="both", labelsize=tick_font)
    ax.legend(fontsize=tick_font, loc="upper right")

    return fig


# ---------------------------------------------------------------------------
# 9. Ramachandran plot
# ---------------------------------------------------------------------------

def create_ramachandran_figure(
    phi_psi_data: list,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Classical Ramachandran plot coloured by secondary structure.

    Parameters
    ----------
    phi_psi_data:
        List of dicts: {``phi``, ``psi``, ``resname``, ``resnum``, ``ss``}.
        ``ss`` values: 'H' = helix, 'E' = sheet, 'C' = coil.
    label_font, tick_font:
        Font sizes.

    Returns
    -------
    matplotlib.figure.Figure
    """
    SS_COLORS = {"H": "#1f77b4", "E": "#d62728", "C": "#aaaaaa"}
    SS_LABELS = {"H": "α-Helix", "E": "β-Sheet", "C": "Coil"}

    fig = Figure(figsize=(7, 6), tight_layout=True)
    ax = fig.add_subplot(111)

    # ---- Allowed/generous regions as filled rectangles ----
    # Core α-helix
    ax.add_patch(Rectangle((-80, -60), 32, 40, color="#555555", alpha=0.25,
                            zorder=0, label="_helix_region"))
    # Core β-sheet
    ax.add_patch(Rectangle((-150, 90), 60, 70, color="#888888", alpha=0.20,
                            zorder=0, label="_sheet_region"))
    # Left-handed helix (positive phi)
    ax.add_patch(Rectangle((40, 20), 40, 40, color="#bbbbbb", alpha=0.20,
                            zorder=0, label="_lh_region"))

    # Crosshairs
    ax.axhline(0, color="grey", linewidth=0.6, linestyle="-")
    ax.axvline(0, color="grey", linewidth=0.6, linestyle="-")

    # Plot residues
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

    # Allowed region annotation patches for legend
    helix_patch = mpatches.Patch(color="#555555", alpha=0.4, label="Core α-helix region")
    sheet_patch = mpatches.Patch(color="#888888", alpha=0.35, label="Core β-sheet region")
    lh_patch = mpatches.Patch(color="#bbbbbb", alpha=0.35, label="Left-handed helix")

    # Build legend
    handles, labels = ax.get_legend_handles_labels()
    # Filter out internal labels
    visible = [(h, l) for h, l in zip(handles, labels) if not l.startswith("_")]
    visible_h, visible_l = zip(*visible) if visible else ([], [])
    ax.legend(
        list(visible_h) + [helix_patch, sheet_patch, lh_patch],
        list(visible_l) + ["Core α-helix", "Core β-sheet", "Left-handed helix"],
        fontsize=tick_font,
        loc="upper right",
        markerscale=1.5,
    )

    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xlabel("φ (°)", fontsize=label_font)
    ax.set_ylabel("ψ (°)", fontsize=label_font)
    ax.set_title("Ramachandran Plot (from PDB structure)", fontsize=label_font)
    ax.tick_params(axis="both", labelsize=tick_font)
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks(range(-180, 181, 60))

    return fig


# ---------------------------------------------------------------------------
# 10. Residue contact network
# ---------------------------------------------------------------------------

def _circular_layout(n: int) -> np.ndarray:
    """Return (n, 2) xy positions for n nodes evenly spaced on a unit circle."""
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _spring_layout(adj: np.ndarray, n_iter: int = 200,
                   seed: int = 42) -> np.ndarray:
    """Fruchterman-Reingold spring layout (pure numpy).

    Parameters
    ----------
    adj:
        (n, n) boolean adjacency matrix.
    n_iter:
        Number of iterations.

    Returns
    -------
    pos : np.ndarray of shape (n, 2)
    """
    n = adj.shape[0]
    rng = np.random.default_rng(seed)
    pos = rng.random((n, 2)) * 2 - 1  # uniform in [-1, 1]

    k = 1.0 / math.sqrt(n) if n > 0 else 1.0
    t = 0.1  # initial temperature

    for _ in range(n_iter):
        delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (n,n,2)
        dist = np.linalg.norm(delta, axis=2)  # (n,n)
        np.fill_diagonal(dist, 1e-9)

        # Repulsive forces
        rep = k**2 / dist  # (n, n)
        rep_force = np.einsum("ij,ijk->ik", rep / dist, delta)  # (n, 2)

        # Attractive forces (only connected pairs)
        att = (dist**2 / k) * adj
        att_force = np.einsum("ij,ijk->ik", att / dist, -delta)  # (n, 2)

        displacement = rep_force + att_force
        # Cap displacement by temperature
        disp_len = np.linalg.norm(displacement, axis=1, keepdims=True)
        disp_len = np.maximum(disp_len, 1e-9)
        pos += displacement / disp_len * np.minimum(disp_len, t)

        t *= 0.95  # cool down

    # Normalise to [-1, 1]
    pos -= pos.min(axis=0)
    pos /= pos.max(axis=0) + 1e-9
    pos = pos * 2 - 1

    return pos


def create_contact_network_figure(
    seq: str,
    dist_matrix: np.ndarray,
    cutoff_angstrom: float = 8.0,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Residue contact network derived from Cα distance matrix.

    Parameters
    ----------
    seq:
        Protein sequence string.
    dist_matrix:
        n×n numpy array of Cα pairwise distances (Å).
    cutoff_angstrom:
        Distance cutoff for defining contacts.
    label_font, tick_font:
        Font sizes.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = len(seq)
    dist_matrix = np.asarray(dist_matrix, dtype=float)

    # Build adjacency (ignore self-contacts and sequence neighbours ±1)
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 2, n):
            if dist_matrix[i, j] <= cutoff_angstrom:
                adj[i, j] = True
                adj[j, i] = True

    degree = adj.sum(axis=1).astype(float)  # (n,)
    max_deg = degree.max() if degree.max() > 0 else 1.0

    large_protein = n > 100
    top_n = 30

    if large_protein:
        # Show only top-30 by degree
        top_idx = np.argsort(degree)[-top_n:]
        sub_n = len(top_idx)
        idx_map = {old: new for new, old in enumerate(top_idx)}
        sub_degree = degree[top_idx]
        sub_adj = adj[np.ix_(top_idx, top_idx)]
        sub_seq = [seq[i] for i in top_idx]
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

    # Degree centrality: normalise by (n-1)
    norm_n = (n - 1) if n > 1 else 1
    centrality = node_degrees / norm_n

    fig = Figure(figsize=(9, 8), tight_layout=True)
    ax = fig.add_subplot(111)

    cmap = cm.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=0, vmax=centrality.max() if centrality.max() > 0 else 1)

    # Draw edges
    for i in range(sub_n):
        for j in range(i + 1, sub_n):
            if plot_adj[i, j]:
                # Get original indices for distance lookup
                if large_protein:
                    orig_i = top_idx[i]
                    orig_j = top_idx[j]
                else:
                    orig_i, orig_j = i, j
                d = dist_matrix[orig_i, orig_j]
                lw = max(0.2, 2.0 / (d + 1e-9) * cutoff_angstrom)
                ax.plot(
                    [pos[i, 0], pos[j, 0]],
                    [pos[i, 1], pos[j, 1]],
                    color="lightgrey",
                    linewidth=min(lw, 2.0),
                    zorder=1,
                )

    # Draw nodes
    node_sizes = 50 + (node_degrees / max_deg) * 250
    scatter = ax.scatter(
        pos[:, 0], pos[:, 1],
        c=centrality, cmap="viridis", norm=norm,
        s=node_sizes, zorder=3, edgecolors="black", linewidths=0.4,
    )

    # Node labels (only if small protein)
    if not large_protein and n <= 50:
        for i in range(sub_n):
            ax.text(pos[i, 0], pos[i, 1], residue_labels[i],
                    ha="center", va="center",
                    fontsize=max(tick_font - 4, 6), zorder=4)

    # Colorbar
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Degree Centrality", fontsize=label_font - 2)
    cbar.ax.tick_params(labelsize=tick_font - 2)

    ax.set_aspect("equal")
    ax.axis("off")
    subtitle = ""
    if large_protein:
        subtitle = f" — Top {top_n} residues by degree"
    ax.set_title(
        f"Residue Contact Network (Cα ≤ {cutoff_angstrom} Å){subtitle}",
        fontsize=label_font,
    )

    return fig


# ---------------------------------------------------------------------------
# 11. MSA conservation profile
# ---------------------------------------------------------------------------

def _shannon_entropy(column_chars: list) -> float:
    """Shannon entropy (bits) for a list of characters (including gaps)."""
    total = len(column_chars)
    if total == 0:
        return 0.0
    from collections import Counter
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
    """Per-column conservation bar chart from a multiple sequence alignment.

    Parameters
    ----------
    sequences:
        List of equal-length strings (aligned sequences; gaps = '-').
    names:
        Sequence names (same length as sequences; used in tooltip/legend).
    label_font, tick_font:
        Font sizes.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not sequences:
        fig = Figure(figsize=(10, 4))
        ax = fig.add_subplot(111)
        ax.set_title("MSA Conservation Profile (no data)", fontsize=label_font)
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
            conservation[col] = 1.0  # perfectly conserved

        # Most common character (excluding gaps for dominant AA display)
        non_gap = [c for c in chars if c != "-"]
        if non_gap:
            from collections import Counter
            dominant_aa[col] = Counter(non_gap).most_common(1)[0][0]
        else:
            dominant_aa[col] = "-"

    fig = Figure(figsize=(max(10, aln_len * 0.15), 4), tight_layout=True)
    ax = fig.add_subplot(111)

    ax.bar(positions, conservation, color="steelblue", width=0.8, zorder=2)

    # Dominant AA text overlay where conservation > 0.7
    for col in range(aln_len):
        if conservation[col] > 0.7 and dominant_aa[col] not in ("-", ""):
            ax.text(
                positions[col], conservation[col] + 0.02,
                dominant_aa[col],
                ha="center", va="bottom",
                fontsize=max(tick_font - 4, 6),
                color="darkblue",
            )

    ax.axhline(0.7, color="red", linestyle="--", linewidth=0.8,
               label="Conservation = 0.7")

    ax.set_xlabel("Alignment Position", fontsize=label_font)
    ax.set_ylabel("Conservation", fontsize=label_font)
    ax.set_title("MSA Conservation Profile", fontsize=label_font)
    ax.set_xlim(0.5, aln_len + 0.5)
    ax.set_ylim(0, 1.15)
    ax.tick_params(axis="both", labelsize=tick_font)
    ax.legend(fontsize=tick_font, loc="upper right")

    return fig


# ---------------------------------------------------------------------------
# 12. Protein complex MW bar chart
# ---------------------------------------------------------------------------

def create_complex_mw_figure(
    chains_data: list,
    stoichiometry_str: str,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Bar chart of individual chain MWs and the assembled complex MW.

    Parameters
    ----------
    chains_data:
        List of dicts: {``chain_id`` (str), ``mol_weight`` (Da), ``color`` (optional str)}.
    stoichiometry_str:
        Stoichiometry string, e.g. ``"A2B1"``.  Used for title/annotation.
    label_font, tick_font:
        Font sizes.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import re

    # Parse stoichiometry string like "A2B1" → {A: 2, B: 1}
    stoich_map: dict = {}
    for match in re.finditer(r"([A-Za-z]+)(\d*)", stoichiometry_str):
        chain_id = match.group(1)
        count_str = match.group(2)
        count = int(count_str) if count_str else 1
        if chain_id:
            stoich_map[chain_id] = count

    # Build labels and heights
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

    # Total complex MW = sum(n_copies * mw) for each chain
    total_mw = 0.0
    for chain in chains_data:
        cid = chain["chain_id"]
        n_copies = stoich_map.get(cid, 1)
        total_mw += n_copies * float(chain["mol_weight"])

    bar_labels = chain_ids + ["Complex"]
    bar_heights = chain_mws + [total_mw]
    bar_colors_all = chain_colors + ["#333333"]

    fig = Figure(figsize=(max(6, len(bar_labels) * 1.2 + 2), 5), tight_layout=True)
    ax = fig.add_subplot(111)

    bars = ax.bar(
        range(len(bar_labels)),
        [mw / 1000 for mw in bar_heights],  # convert to kDa
        color=bar_colors_all,
        edgecolor="black",
        linewidth=0.6,
    )

    # Annotate bars with values
    for bar_obj, height_kda in zip(bars, [h / 1000 for h in bar_heights]):
        ax.text(
            bar_obj.get_x() + bar_obj.get_width() / 2,
            bar_obj.get_height() + 0.5,
            f"{height_kda:.1f} kDa",
            ha="center", va="bottom",
            fontsize=max(tick_font - 2, 8),
        )

    # Total complex annotation
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
    ax.set_xticklabels(bar_labels, fontsize=tick_font)
    ax.set_ylabel("Molecular Weight (kDa)", fontsize=label_font)
    ax.set_title("Protein Complex Mass Composition", fontsize=label_font)
    ax.tick_params(axis="y", labelsize=tick_font)

    return fig
