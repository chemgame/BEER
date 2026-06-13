"""Multi-feature overlay: plot N per-residue profiles on a shared x-axis."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib.figure import Figure
from matplotlib import colormaps as _cm
from ._style import _pub_style_ax


def _distinct_colors(n: int) -> "list[str]":
    """Return *n* maximally-distinct hex colors using qualitative colormaps."""
    if n == 0:
        return []
    # tab10 (10) + tab20b (20) + tab20c (20) give 50 distinct colors
    pools = [
        list(_cm["tab10"].colors),
        list(_cm["tab20b"].colors),
        list(_cm["tab20c"].colors),
    ]
    palette: list = []
    for pool in pools:
        palette.extend(pool)
    # Convert RGBA tuples to hex
    def _hex(c):
        r, g, b = (int(v * 255) for v in c[:3])
        return f"#{r:02x}{g:02x}{b:02x}"
    colors = [_hex(c) for c in palette]
    # If n > palette length, cycle
    return [colors[i % len(colors)] for i in range(n)]


def create_overlay_figure(
    profiles: "dict[str, list]",
    normalize: bool = True,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Return a Figure with all profiles overlaid on residue-position x-axis."""
    fig = Figure(figsize=(10, 4.5), dpi=120)
    ax = fig.add_subplot(111)
    ax.set_facecolor("#fafbff")

    colors = _distinct_colors(len(profiles))

    for idx, (name, data) in enumerate(profiles.items()):
        arr = np.array(data, dtype=float)
        if not np.any(np.isfinite(arr)):
            continue  # all-NaN profile — nothing to plot
        if normalize:
            lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
            arr = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)

        x = np.arange(1, len(arr) + 1)
        ax.plot(x, arr, label=name, color=colors[idx], linewidth=1.4, alpha=0.85)

    ylabel = "Normalized Score (0–1)" if normalize else "Score"
    _pub_style_ax(ax, xlabel="Residue Position", ylabel=ylabel,
                  label_size=label_font, tick_size=tick_font)

    n = len(profiles)
    if n:
        ax.legend(fontsize=max(7, tick_font - 2), framealpha=0.75,
                  loc="upper right", ncol=max(1, n // 8))

    fig.tight_layout(pad=1.5)
    return fig


def create_correlation_figure(
    profiles: "dict[str, list]",
    cmap: str = "coolwarm",
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Return a Figure showing pairwise Pearson r heatmap across selected profiles."""
    names = list(profiles.keys())
    n = len(names)
    arrays = [np.array(profiles[nm], dtype=float) for nm in names]

    min_len = min(len(a) for a in arrays)
    arrays = [a[:min_len] for a in arrays]

    corr = np.ones((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            mask = np.isfinite(arrays[i]) & np.isfinite(arrays[j])
            ok = (mask.sum() >= 3
                  and np.std(arrays[i][mask]) > 0
                  and np.std(arrays[j][mask]) > 0)
            r = float(np.corrcoef(arrays[i][mask], arrays[j][mask])[0, 1]) if ok else np.nan
            corr[i, j] = corr[j, i] = r

    # Uniform small font for everything — scales down with grid size
    base_font = max(5, min(8, int(90 / max(n, 1))))

    size = max(5, n * 0.6)
    fig = Figure(figsize=(size, size * 0.9), dpi=120)
    ax = fig.add_subplot(111)

    try:
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap=cmap, aspect="auto")
    except ValueError:
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=base_font)
    ax.set_yticklabels(names, fontsize=base_font)
    ax.tick_params(length=2, pad=2)

    for i in range(n):
        for j in range(n):
            val = corr[i, j]
            if not np.isfinite(val):
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=base_font, color="#888888")
                continue
            txt_color = "white" if abs(val) > 0.65 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=base_font, color=txt_color)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Pearson r", fontsize=base_font)
    cb.ax.tick_params(labelsize=base_font)
    fig.tight_layout(pad=1.5)
    return fig
