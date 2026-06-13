"""Per-residue profile figures (hydrophobicity, aggregation, solubility, etc.)."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from beer.graphs._style import (
    _pub_style_ax, _apply_font_sizes, _residue_x,
    _ACCENT, _NEG_COL, FEATURE_COLORS,
    _PROFILE_LINE, _FILL_ABOVE, _FILL_BELOW, _FILL_NEUTRAL,
)
from beer.constants import HYDROPHOBICITY_SCALES

# ZYGGREGATOR Z_agg^i threshold: one standard deviation above the random-sequence
# mean, as published in Tartaglia et al. (2008) J. Mol. Biol. 380:425 and
# Tartaglia & Vendruscolo (2008) Chem. Soc. Rev. 37:1395.  calc_aggregation_profile()
# now returns fully normalised Z-scores, so this threshold is the original Z > 1.
AGGREGATION_THRESHOLD = 1.0
SOLUBILITY_NEUTRAL = 0.0
# Eisenberg et al. (1984) PNAS 81:140 — µH ≥ 0.35 defines amphipathic helix
HM_THRESHOLD = 0.35


def _maybe_downsample(x_arr, y_arr, max_pts: int = 800):
    """Thin (x, y) arrays to at most max_pts points using uniform stride."""
    if len(y_arr) <= max_pts:
        return x_arr, y_arr
    stride = max(1, len(y_arr) // max_pts)
    return x_arr[::stride], y_arr[::stride]


def _adaptive_width(n: int, base: float = 9.0, scale: float = 0.015,
                    lo: float = 9.0, hi: float = 16.0) -> float:
    """Return figure width that grows gently with sequence length."""
    return max(lo, min(hi, base + n * scale))


def create_hydrophobicity_figure(
    hydro_profile: list,
    window_size: int,
    scale_name: str = "Kyte-Doolittle",
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Sliding-window hydrophobicity plot."""
    n = len(hydro_profile)
    fig = Figure(figsize=(_adaptive_width(n), 4.5), dpi=120)
    fig.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)
    xs = np.arange(1, n + 1, dtype=float)
    ys = np.asarray(hydro_profile, dtype=float)
    xs, ys = _maybe_downsample(xs, ys)
    ax.fill_between(xs, ys, 0, where=(ys >= 0),
                    alpha=0.25, color=_FILL_ABOVE, interpolate=True)
    ax.fill_between(xs, ys, 0, where=(ys < 0),
                    alpha=0.25, color=_FILL_BELOW, interpolate=True)
    ax.plot(xs, ys, color=_PROFILE_LINE, linewidth=1.4, zorder=4)
    ax.axhline(0, color="#888", linewidth=0.7, linestyle="--", zorder=3)
    _ylabel = HYDROPHOBICITY_SCALES.get(scale_name, {}).get("ylabel", "Score")
    _pub_style_ax(ax,
                  title="Hydrophobicity Profile",
                  xlabel="Residue Position", ylabel=_ylabel,
                  grid=True, title_size=label_font,
                  label_size=label_font, tick_size=tick_font)
    fig.tight_layout(pad=1.5)
    return fig


def create_aggregation_profile_figure(
    seq: str,
    aggregation_profile: list,
    hotspots: list,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Line plot of per-residue beta-aggregation propensity (Zyggregator)."""
    n = len(seq)
    x = _residue_x(seq)
    y = np.asarray(aggregation_profile, dtype=float)
    x, y = _maybe_downsample(x, y)

    fig = Figure(figsize=(_adaptive_width(n), 4.5), dpi=120)
    fig.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)

    ax.plot(x, y, color=_PROFILE_LINE, linewidth=1.4)
    ax.fill_between(x, AGGREGATION_THRESHOLD, y, where=(y > AGGREGATION_THRESHOLD),
                    interpolate=True, color=_FILL_BELOW, alpha=0.40)
    ax.fill_between(x, y, AGGREGATION_THRESHOLD, where=(y <= AGGREGATION_THRESHOLD),
                    interpolate=True, color=_FILL_ABOVE, alpha=0.15)
    ax.axhline(AGGREGATION_THRESHOLD, color="#374151", linestyle="--",
               linewidth=0.9)

    y_min, y_max = float(y.min()), float(y.max()) * 1.1 + 0.2
    for idx, hs in enumerate(hotspots):
        # hs is a dict with 0-based 'start'/'end'; x-axis is 1-based
        start = hs['start'] + 1
        end = hs['end']
        rect = Rectangle((start - 0.5, y_min), (end - start + 1), y_max - y_min,
                          linewidth=0, facecolor="#f72585", alpha=0.18, zorder=0,
                          label="Hotspot region" if idx == 0 else "_nolegend_")
        ax.add_patch(rect)

    _pub_style_ax(ax, title="β-Aggregation Profile",
                  xlabel="Residue Position", ylabel="β-Aggregation Z-Score",
                  grid=True, title_size=label_font,
                  label_size=label_font, tick_size=tick_font)
    ax.set_xlim(x[0], x[-1])
    fig.tight_layout(pad=1.5)
    return fig


def create_solubility_profile_figure(
    seq: str,
    camsolmt_profile: list,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Per-residue CamSol intrinsic solubility profile."""
    n = len(seq)
    x = _residue_x(seq)
    y = np.asarray(camsolmt_profile, dtype=float)
    x, y = _maybe_downsample(x, y)

    fig = Figure(figsize=(_adaptive_width(n), 4.5), dpi=120)
    fig.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)

    ax.plot(x, y, color=_PROFILE_LINE, linewidth=0.8)
    ax.fill_between(x, 0, y, where=(y >= 0), interpolate=True,
                    color=_FILL_ABOVE, alpha=0.30)
    ax.fill_between(x, 0, y, where=(y < 0), interpolate=True,
                    color=_FILL_BELOW, alpha=0.30)
    ax.axhline(0.0, color="#888", linestyle="--", linewidth=0.8)

    _pub_style_ax(ax, title="Solubility Profile",
                  xlabel="Residue Position", ylabel="CamSol Solubility Score",
                  grid=True, title_size=label_font,
                  label_size=label_font, tick_size=tick_font)
    ax.set_xlim(x[0], x[-1])
    fig.tight_layout(pad=1.5)
    return fig


def create_scd_profile_figure(
    seq: str,
    scd_profile: list,
    window: int,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Sliding-window SCD (Sequence Charge Decoration) profile."""
    seq_len = len(seq)
    n = len(scd_profile)
    # Centre each window value on the middle residue of its window so that the
    # x-axis aligns with sequence position rather than window-start position.
    half = window // 2
    x_start = half + 1  # window i=0 spans positions 1..W; 1-based center = W//2 + 1
    x = np.arange(x_start, x_start + n, dtype=float)
    y = np.asarray(scd_profile, dtype=float)
    x, y = _maybe_downsample(x, y)

    fig = Figure(figsize=(_adaptive_width(n), 4.5), dpi=120)
    fig.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)

    ax.plot(x, y, color=_PROFILE_LINE, linewidth=1.2)
    ax.fill_between(x, 0, y, where=(y > 0), interpolate=True,
                    color=_FILL_ABOVE, alpha=0.30)
    ax.fill_between(x, 0, y, where=(y < 0), interpolate=True,
                    color=_FILL_BELOW, alpha=0.30)
    ax.axhline(0.0, color="#888", linestyle="--", linewidth=0.8)

    _pub_style_ax(ax, title="Charge Decoration Profile",
                  xlabel="Residue Position", ylabel="Charge Decoration (SCD)",
                  grid=True, title_size=label_font,
                  label_size=label_font, tick_size=tick_font)
    ax.set_xlim(1, seq_len)
    fig.tight_layout(pad=1.5)
    return fig


def create_plaac_profile_figure(
    plaac_data: dict,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Per-residue PLAAC log-odds profile (Lancaster et al. 2014)."""
    profile = plaac_data.get("profile", [])
    n = len(profile)
    if n == 0:
        fig = Figure(figsize=(9, 4.5), dpi=120)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No PLAAC data", ha="center", va="center",
                transform=ax.transAxes, color="#718096")
        return fig

    xs = list(range(1, n + 1))
    fig = Figure(figsize=(_adaptive_width(n, base=9.0, scale=0.012), 4.5), dpi=120)
    fig.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)

    pos = [v if v > 0 else 0 for v in profile]
    neg = [v if v < 0 else 0 for v in profile]
    _lcd_col = FEATURE_COLORS.get("lcd", "#90be6d")
    ax.fill_between(xs, pos, 0, alpha=0.50, color=_lcd_col)
    ax.fill_between(xs, neg, 0, alpha=0.30, color=_FILL_BELOW)
    ax.plot(xs, profile, color=_PROFILE_LINE, linewidth=0.8, alpha=0.7)
    ax.axhline(0, color="#aaa", linewidth=0.8, linestyle="--")

    _pub_style_ax(ax, title="Prion-like Domain Profile",
                  xlabel="Residue Position", ylabel="PLAAC Log-Odds Score",
                  grid=True, title_size=label_font,
                  label_size=label_font, tick_size=tick_font)
    ax.set_xlim(1, n)
    fig.tight_layout(pad=1.5)
    return fig


# ---------------------------------------------------------------------------
# BiLSTM profile (single-track with optional uncertainty band)
# ---------------------------------------------------------------------------

def _bilstm_predicted_regions(scores_np, threshold: float = 0.5) -> list[tuple[int, int]]:
    """Return (start, end) 1-based inclusive spans where score > threshold."""
    regions, in_region, start = [], False, 0
    for i, v in enumerate(scores_np):
        if v > threshold and not in_region:
            in_region, start = True, i + 1
        elif v <= threshold and in_region:
            regions.append((start, i))
            in_region = False
    if in_region:
        regions.append((start, len(scores_np)))
    return regions


_BILSTM_THRESHOLDS: dict[str, float] = {
    "disorder":          0.63054,   # v3b CNN-BiLSTM
    "signal_peptide":    0.77552,
    "transmembrane":     0.81339,   # CRF head — no threshold_at_f1max; kept from v2
    "intramembrane":     0.76351,
    "coiled_coil":       0.55178,   # v3b improved model
    "dna_binding":       0.76365,
    "active_site":       0.86000,
    "binding_site":      0.74051,
    "phosphorylation":   0.81573,
    "lcd":               0.67142,   # v3b CNN-BiLSTM
    "glycosylation":     0.84581,
    "ubiquitination":    0.81069,
    "nucleotide_binding":0.76781,
    "acetylation":       0.86891,
    "methylation":       0.88184,   # v3b CNN-BiLSTM
    "lipidation":        0.81459,
    "disulfide":         0.85226,
    "motif":             0.76468,   # v3b CNN-BiLSTM
    "propeptide":        0.66717,   # v3b PosEnc-BiLSTM
    "repeat":            0.52704,   # v3b CNN-BiLSTM
    "rna_binding":       0.63172,   # v3b CNN-BiLSTM
    "transit_peptide":   0.69661,
    "zinc_finger":       0.62258,
}


def create_bilstm_profile_figure(
    feature: str,
    scores: "list[float]",
    uncertainty: "list[float] | None" = None,
    uniprot_regions: "list[dict] | None" = None,
    threshold: "float | None" = None,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Per-residue BiLSTM probability profile.

    BiLSTM prediction is the primary curve.  When *uniprot_regions* is
    supplied (after clicking "UniProt Tracks"), the curated Swiss-Prot
    annotation is overlaid on the **same axes** as:

    * A semi-transparent coloured background span (α = 0.18) across the
      full y-range for each annotated region — gives an at-a-glance view.
    * A solid rug strip pinned just below y = 0 (α = 0.85) — marks the
      precise annotated positions clearly without obscuring the curve.

    Both layers share the feature colour so the visual language is consistent
    across all 20 heads.
    """
    from matplotlib.patches import Patch
    color     = FEATURE_COLORS.get(feature, _ACCENT)
    has_uniprot = bool(uniprot_regions)
    n  = len(scores)
    xs = list(range(1, n + 1))
    thr = threshold if threshold is not None else _BILSTM_THRESHOLDS.get(feature, 0.5)

    fig = Figure(figsize=(_adaptive_width(n), 4.5), dpi=120)
    fig.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)

    scores_np = np.array(scores[:n], dtype=float)
    feat_display = feature.replace("_", " ").title()
    # Fix abbreviations that title() lowercases incorrectly
    for _wrong, _right in (("Rna", "RNA"), ("Dna", "DNA"), ("Tm ", "TM ")):
        feat_display = feat_display.replace(_wrong, _right)

    # ── UniProt annotation overlay (behind BiLSTM curve) ─────────────────
    if has_uniprot:
        ref_np = np.zeros(n, dtype=float)
        for reg in uniprot_regions:
            s = max(0, int(reg.get("start", 1)) - 1)
            e = min(n, int(reg.get("end", 1)))
            ref_np[s:e] = 1.0
            ax.axvspan(s + 0.5, e + 0.5, alpha=0.18, color=color,
                       zorder=1, linewidth=0)
        ax.fill_between(xs, -0.08, np.where(ref_np > 0.5, -0.02, -0.08),
                        step="pre", color=color, alpha=0.85, zorder=3,
                        linewidth=0)

    # ── BiLSTM curve (primary) ────────────────────────────────────────────
    ax.fill_between(xs, scores_np, thr,
                    where=scores_np > thr,
                    alpha=0.28, color=_FILL_ABOVE, interpolate=True, zorder=4)
    ax.fill_between(xs, scores_np, thr,
                    where=scores_np <= thr,
                    alpha=0.18, color=_FILL_BELOW, interpolate=True, zorder=4)
    if uncertainty is not None:
        unc_np = np.array(uncertainty[:n], dtype=float)
        lo = np.clip(scores_np - unc_np, 0.0, 1.0)
        hi = np.clip(scores_np + unc_np, 0.0, 1.0)
        # Draw behind the main fills (zorder=2) so band edges peek out on both sides.
        ax.fill_between(xs, lo, hi, alpha=0.30, color="#94a3b8",
                        linewidth=0, zorder=2)
        # Thin dashed border lines to make the band boundaries legible.
        ax.plot(xs, hi, color="#64748b", linewidth=0.6,
                linestyle="--", alpha=0.7, zorder=2)
        ax.plot(xs, lo, color="#64748b", linewidth=0.6,
                linestyle="--", alpha=0.7, zorder=2)
    ax.plot(xs, scores_np, color=_PROFILE_LINE, linewidth=1.6, zorder=5)
    ax.axhline(thr, color="#888888", linewidth=0.9, linestyle="--", zorder=3)

    # ── Axes styling ─────────────────────────────────────────────────────
    title = f"{feat_display} Profile"
    ylabel = f"{feat_display} Prediction Score"
    _pub_style_ax(ax, title=title, xlabel="Residue Position", ylabel=ylabel,
                  grid=True, title_size=label_font,
                  label_size=label_font, tick_size=tick_font)

    ymin = -0.10 if has_uniprot else -0.03
    ax.set_ylim(ymin, 1.05)

    # Legend — only shown when UniProt annotation is overlaid (necessary context)
    if has_uniprot:
        ax.add_artist(ax.legend(
            handles=[
                ax.get_lines()[0] if ax.get_lines() else Patch(color=color),
                Patch(facecolor=color, alpha=0.45, label="UniProt Swiss-Prot"),
            ],
            labels=[
                "BiLSTM (ESMC 600M)",
                "UniProt Swiss-Prot",
            ],
            fontsize=tick_font - 2, loc="upper right",
            framealpha=0.92, edgecolor="#d0d4e0",
        ))

    fig.tight_layout(pad=1.5)
    return fig


def create_bilstm_dual_track_figure(
    feature: str,
    seq: str,
    scores: "list[float]",
    uniprot_regions: "list[dict]",
    uncertainty: "list[float] | None" = None,
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Unified overlay figure — delegates to create_bilstm_profile_figure."""
    return create_bilstm_profile_figure(
        feature, scores,
        uncertainty=uncertainty,
        uniprot_regions=uniprot_regions,
        label_font=label_font,
        tick_font=tick_font,
    )


def create_shd_profile_figure(
    seq: str,
    shd_profile: list,
    window: int = 20,
    scale_name: str = "Kyte-Doolittle",
    label_font: int = 14,
    tick_font: int = 12,
) -> Figure:
    """Sliding-window SHD (Sequence Hydrophobicity Decoration) profile.

    Analogous to SCD but based on hydrophobicity instead of charge.
    SHD(window) = (1/W) * Σ_{i<j} hᵢ · hⱼ · |i−j|^0.5
    Positive: hydrophobic and hydrophilic residues cluster (amphipathic blocks).
    Negative: alternating hydrophobic/hydrophilic (mixed pattern).
    """
    seq_len = len(seq)
    n = len(shd_profile)
    half = window // 2
    x_start = half + 1  # same centering convention as SCD (1-based)
    x = np.arange(x_start, x_start + n, dtype=float)
    y = np.asarray(shd_profile, dtype=float)
    x, y = _maybe_downsample(x, y)

    fig = Figure(figsize=(_adaptive_width(n), 4.5), dpi=120)
    fig.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)

    ax.plot(x, y, color=_PROFILE_LINE, linewidth=1.2)
    ax.fill_between(x, 0, y, where=(y > 0), interpolate=True,
                    color=_FILL_ABOVE, alpha=0.30)
    ax.fill_between(x, 0, y, where=(y < 0), interpolate=True,
                    color=_FILL_BELOW, alpha=0.30)
    ax.axhline(0.0, color="#888", linestyle="--", linewidth=0.8)

    _pub_style_ax(ax,
                  title="Hydrophobicity Decoration Profile",
                  xlabel="Residue Position",
                  ylabel="Hydrophobicity Decoration (SHD)",
                  grid=True, title_size=label_font,
                  label_size=label_font, tick_size=tick_font)
    ax.set_xlim(1, seq_len)
    fig.tight_layout(pad=1.5)
    return fig


# ---------------------------------------------------------------------------
# AI Predictions Overview (domain-architecture style)
# ---------------------------------------------------------------------------

# (data_key, label, color, threshold_at_f1max)
_AI_OVERVIEW_HEADS: list[tuple[str, str, str, float]] = [
    ("disorder_scores",          "Disorder",           "#4361ee", 0.63054),   # v3b
    ("sp_bilstm_profile",        "Signal Peptide",     "#f72585", 0.77552),
    ("tm_bilstm_profile",        "Transmembrane",      "#34d399", 0.81339),
    ("intramem_bilstm_profile",  "Intramembrane",      "#6ee7b7", 0.76351),
    ("cc_bilstm_profile",        "Coiled-Coil",        "#fb923c", 0.55178),
    ("dna_bilstm_profile",       "DNA-Binding",        "#60a5fa", 0.76365),
    ("rnabind_bilstm_profile",   "RNA-Binding",        "#2dc653", 0.63172),   # v3b
    ("act_bilstm_profile",       "Active Site",        "#f87171", 0.86000),
    ("bnd_bilstm_profile",       "Binding Site",       "#a78bfa", 0.74051),
    ("phos_bilstm_profile",      "Phosphorylation",    "#fbbf24", 0.81573),
    ("lcd_bilstm_profile",       "Low-Complexity",     "#94a3b8", 0.67142),   # v3b
    ("znf_bilstm_profile",       "Zinc Finger",        "#4ade80", 0.62258),
    ("glyc_bilstm_profile",      "Glycosylation",      "#f9a8d4", 0.84581),
    ("ubiq_bilstm_profile",      "Ubiquitination",     "#fb7185", 0.81069),
    ("meth_bilstm_profile",      "Methylation",        "#a3e635", 0.88184),   # v3b
    ("acet_bilstm_profile",      "Acetylation",        "#38bdf8", 0.86891),
    ("lipid_bilstm_profile",     "Lipidation",         "#e879f9", 0.81459),
    ("disulf_bilstm_profile",    "Disulfide Bond",     "#fde68a", 0.85226),
    ("motif_bilstm_profile",     "Functional Motif",   "#c4b5fd", 0.76468),   # v3b
    ("prop_bilstm_profile",      "Propeptide",         "#fdba74", 0.66717),   # v3b
    ("rep_bilstm_profile",       "Repeat Region",      "#67e8f9", 0.52704),   # v3b
    ("nucbind_bilstm_profile",   "Nucleotide-Binding", "#0ea5e9", 0.76781),
    ("transit_bilstm_profile",   "Transit Peptide",    "#d946ef", 0.69661),
]
