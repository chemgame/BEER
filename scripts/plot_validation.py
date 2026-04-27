#!/usr/bin/env python3
"""Generate publication-quality validation figures for all BEER ESM2 heads.

Figures produced in manuscript/figures/
  fig_roc_all.png          — Multi-panel ROC curves (all tasks)
  fig_pr_all.png           — Multi-panel Precision-Recall curves
  fig_learning_curves.png  — Train loss + val AUROC vs epoch (all tasks)
  fig_auroc_violin.png     — Per-protein AUROC distributions (violin + box)
  fig_calibration.png      — Reliability diagrams (all tasks)
  fig_auroc_summary.png    — Summary bar chart with 95% CI error bars
  fig_task_stats.png       — Dataset statistics (class balance, n_proteins)

Usage
-----
    conda run -n beer python scripts/plot_validation.py
    conda run -n beer python scripts/plot_validation.py --tasks signal_peptide transmembrane
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

ROOT      = pathlib.Path(__file__).parent.parent
CACHE_DIR = ROOT / "scripts" / ".head_caches"
FIG_DIR   = ROOT / "manuscript" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Publication style
plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        9,
    "axes.labelsize":   10,
    "axes.titlesize":   10,
    "legend.fontsize":  8,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "figure.dpi":       300,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.linewidth":   0.8,
})

TASK_LABELS = {
    "disorder":        "Disorder",
    "signal_peptide":  "Signal peptide",
    "transmembrane":   "Transmembrane helix",
    "coiled_coil":     "Coiled coil",
    "dna_binding":     "DNA binding",
    "active_site":     "Active site",
    "binding_site":    "Binding site",
    "phosphorylation": "Phosphorylation",
    "lcd":             "Low complexity (LCD)",
    "zinc_finger":     "Zinc finger",
    "glycosylation":   "Glycosylation",
    "ubiquitination":  "Ubiquitination",
    "methylation":     "Methylation",
    "acetylation":     "Acetylation",
    "lipidation":      "Lipidation",
    "disulfide":       "Disulfide bond",
    "intramembrane":   "Intramembrane",
    "motif":           "Short linear motif",
    "propeptide":      "Propeptide",
    "repeat":          "Tandem repeat",
}

TASK_COLORS = {
    "signal_peptide":  "#e41a1c",
    "transmembrane":   "#377eb8",
    "coiled_coil":     "#4daf4a",
    "dna_binding":     "#984ea3",
    "active_site":     "#ff7f00",
    "binding_site":    "#a65628",
    "phosphorylation": "#f781bf",
    "disorder":        "#999999",
    "lcd":             "#66c2a5",
    "zinc_finger":     "#fc8d62",
    "glycosylation":   "#8da0cb",
    "ubiquitination":  "#e78ac3",
    "methylation":     "#a6d854",
    "acetylation":     "#ffd92f",
    "lipidation":      "#e5c494",
    "disulfide":       "#b3b3b3",
    "intramembrane":   "#1b9e77",
    "motif":           "#d95f02",
    "propeptide":      "#7570b3",
    "repeat":          "#e7298a",
}


def load_results(tasks: list[str]) -> dict[str, dict]:
    results = {}
    for task in tasks:
        p = CACHE_DIR / f"{task}_results.json"
        if p.exists():
            with open(p) as f:
                results[task] = json.load(f)
        else:
            print(f"  WARNING: {p} not found — skipping {task}")
    return results


# ---------------------------------------------------------------------------
# Figure 1 — ROC curves (all tasks, one panel each)
# ---------------------------------------------------------------------------

def fig_roc_all(results: dict, out_path: pathlib.Path):
    n = len(results)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.2 * ncols, 3.0 * nrows),
                             squeeze=False)

    for ax_idx, (task, res) in enumerate(results.items()):
        ax  = axes[ax_idx // ncols][ax_idx % ncols]
        fpr = res["roc_curve"]["fpr"]
        tpr = res["roc_curve"]["tpr"]
        m   = res["test_metrics"]
        col = TASK_COLORS.get(task, "#333333")
        ax.plot(fpr, tpr, color=col, lw=1.8,
                label=f"AUC = {m['auroc']:.3f}\n"
                      f"[{m['auroc_ci_lo']:.3f}–{m['auroc_ci_hi']:.3f}]")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
        ax.set_xlim(-0.02, 1.02);  ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(TASK_LABELS.get(task, task))
        ax.legend(loc="lower right", frameon=False)
        ax.set_aspect("equal")

    # Hide unused panels
    for i in range(ax_idx + 1, nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    fig.suptitle("ROC curves — ESM2 650M + BiLSTM heads", fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ---------------------------------------------------------------------------
# Figure 2 — Precision-Recall curves
# ---------------------------------------------------------------------------

def fig_pr_all(results: dict, out_path: pathlib.Path):
    n = len(results)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.2 * ncols, 3.0 * nrows),
                             squeeze=False)

    for ax_idx, (task, res) in enumerate(results.items()):
        ax   = axes[ax_idx // ncols][ax_idx % ncols]
        prec = res["pr_curve"]["precision"]
        rec  = res["pr_curve"]["recall"]
        m    = res["test_metrics"]
        col  = TASK_COLORS.get(task, "#333333")
        pos_rate = res["dataset"]["pos_rate_test"]
        ax.plot(rec, prec, color=col, lw=1.8,
                label=f"AUPRC = {m['auprc']:.3f}\nF1_max = {m['f1_max']:.3f}")
        ax.axhline(pos_rate, color="k", lw=0.8, ls="--", alpha=0.5,
                   label=f"Baseline = {pos_rate:.3f}")
        ax.scatter([m["recall_at_f1max"]], [m["precision_at_f1max"]],
                   color=col, s=40, zorder=5)
        ax.set_xlim(-0.02, 1.02);  ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel("Recall");   ax.set_ylabel("Precision")
        ax.set_title(TASK_LABELS.get(task, task))
        ax.legend(loc="upper right", frameon=False)

    for i in range(ax_idx + 1, nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    fig.suptitle("Precision-Recall curves — ESM2 650M + BiLSTM heads",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ---------------------------------------------------------------------------
# Figure 3 — Learning curves
# ---------------------------------------------------------------------------

def fig_learning_curves(results: dict, out_path: pathlib.Path):
    n = len(results)
    fig, axes = plt.subplots(n, 2,
                             figsize=(7.0, 2.5 * n),
                             squeeze=False)

    for row, (task, res) in enumerate(results.items()):
        hist  = res["training_history"]
        retr  = res.get("retrain_history", [])
        col   = TASK_COLORS.get(task, "#333333")
        label = TASK_LABELS.get(task, task)

        epochs    = [h["epoch"] for h in hist]
        losses    = [h["train_loss"] for h in hist]
        val_aucs  = [h["val_auroc"] for h in hist]

        # Loss
        ax = axes[row][0]
        ax.plot(epochs, losses, color=col, lw=1.5)
        ax.set_ylabel("Train loss")
        ax.set_xlabel("Epoch")
        ax.set_title(f"{label} — training loss")

        # Val AUROC
        ax = axes[row][1]
        ax.plot(epochs, val_aucs, color=col, lw=1.5, label="train phase")
        if retr:
            retr_epochs = [h["epoch"] for h in retr]
            retr_aucs   = [h["val_auroc"] for h in retr]
            ax.plot(retr_epochs, retr_aucs, color=col, lw=1.5,
                    ls="--", alpha=0.6, label="retrain phase")
        best_auc = max(val_aucs)
        ax.axhline(best_auc, color="k", lw=0.7, ls=":", alpha=0.5)
        ax.set_ylabel("Validation AUROC")
        ax.set_xlabel("Epoch")
        ax.set_title(f"{label} — validation AUROC")
        ax.legend(frameon=False)

    fig.suptitle("Learning curves — ESM2 650M + BiLSTM heads",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ---------------------------------------------------------------------------
# Figure 4 — Per-protein AUROC violin plots
# ---------------------------------------------------------------------------

def fig_auroc_violin(results: dict, out_path: pathlib.Path):
    tasks  = list(results.keys())
    labels = [TASK_LABELS.get(t, t) for t in tasks]
    data   = [res["per_protein_aurocs"] for res in results.values()]
    colors = [TASK_COLORS.get(t, "#333333") for t in tasks]

    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(tasks)), 4.5))

    parts = ax.violinplot(data, positions=range(len(tasks)),
                          showmedians=True, showextrema=True)
    for i, (pc, col) in enumerate(zip(parts["bodies"], colors)):
        pc.set_facecolor(col);  pc.set_alpha(0.7)
    parts["cmedians"].set_color("black");  parts["cmedians"].set_linewidth(1.5)

    # Overlay mean AUROC dots
    for i, (task, res) in enumerate(results.items()):
        pp = res["per_protein_aurocs"]
        mean_a = np.mean(pp)
        overall_a = res["test_metrics"]["auroc"]
        ax.scatter([i], [overall_a], color="black", s=30, zorder=5,
                   marker="D", label="Overall" if i == 0 else "")
        ax.scatter([i], [mean_a], color="white", edgecolors="black",
                   s=20, zorder=6, label="Mean per-protein" if i == 0 else "")

    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Per-protein AUROC")
    ax.set_title("Per-protein AUROC distribution on held-out test proteins\n"
                 "ESM2 650M + BiLSTM — Swiss-Prot reviewed annotations")
    ax.axhline(0.5, color="k", lw=0.7, ls="--", alpha=0.4)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ---------------------------------------------------------------------------
# Figure 5 — Calibration (reliability diagrams)
# ---------------------------------------------------------------------------

def fig_calibration(results: dict, out_path: pathlib.Path):
    n = len(results)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.2 * ncols, 3.0 * nrows),
                             squeeze=False)

    for ax_idx, (task, res) in enumerate(results.items()):
        ax   = axes[ax_idx // ncols][ax_idx % ncols]
        cal  = res["calibration"]
        mp   = np.array(cal["mean_predicted"])
        fp   = np.array(cal["fraction_positive"], dtype=float)
        cnt  = np.array(cal["counts"])
        col  = TASK_COLORS.get(task, "#333333")
        mask = cnt > 0

        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Perfect")
        ax.plot(mp[mask], fp[mask], "o-", color=col, lw=1.5,
                ms=4, label="Model")
        # Bar showing data density
        ax2 = ax.twinx()
        ax2.bar(mp[mask], cnt[mask], width=0.04, alpha=0.2, color=col)
        ax2.set_ylabel("Count", fontsize=7, color="grey")
        ax2.tick_params(axis="y", labelsize=6, colors="grey")
        ax2.spines["right"].set_visible(True)

        ax.set_xlim(-0.02, 1.02);  ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction positive")
        ax.set_title(TASK_LABELS.get(task, task))
        ax.legend(frameon=False, fontsize=7)

    for i in range(ax_idx + 1, nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    fig.suptitle("Calibration (reliability diagrams) — ESM2 650M + BiLSTM",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ---------------------------------------------------------------------------
# Figure 6 — Summary AUROC bar chart with CI
# ---------------------------------------------------------------------------

def fig_auroc_summary(results: dict, out_path: pathlib.Path):
    tasks   = list(results.keys())
    labels  = [TASK_LABELS.get(t, t) for t in tasks]
    aucs    = [r["test_metrics"]["auroc"]      for r in results.values()]
    ci_lo   = [r["test_metrics"]["auroc_ci_lo"] for r in results.values()]
    ci_hi   = [r["test_metrics"]["auroc_ci_hi"] for r in results.values()]
    auprc   = [r["test_metrics"]["auprc"]       for r in results.values()]
    colors  = [TASK_COLORS.get(t, "#333333")    for t in tasks]
    err_lo  = [a - l for a, l in zip(aucs, ci_lo)]
    err_hi  = [h - a for a, h in zip(aucs, ci_hi)]
    x = np.arange(len(tasks))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(7, 1.3 * len(tasks)), 4.5))

    # AUROC
    bars = ax1.bar(x, aucs, color=colors, alpha=0.85, width=0.6,
                   yerr=[err_lo, err_hi], capsize=4, ecolor="black", error_kw={"lw": 1.2})
    ax1.axhline(0.5, color="k", lw=0.8, ls="--", alpha=0.4, label="Random")
    ax1.axhline(0.9, color="grey", lw=0.8, ls=":", alpha=0.4)
    for bar, auc in zip(bars, aucs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
                 f"{auc:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax1.set_xticks(x);  ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1.set_ylim(0, 1.10);  ax1.set_ylabel("AUROC (95% CI)")
    ax1.set_title("AUROC — held-out test set")
    ax1.legend(frameon=False)

    # AUPRC
    bars2 = ax2.bar(x, auprc, color=colors, alpha=0.85, width=0.6)
    for bar, ap in zip(bars2, auprc):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                 f"{ap:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax2.set_xticks(x);  ax2.set_xticklabels(labels, rotation=30, ha="right")
    ax2.set_ylim(0, 1.10);  ax2.set_ylabel("AUPRC")
    ax2.set_title("AUPRC — held-out test set")

    fig.suptitle("BEER v3.0 — ESM2 650M + BiLSTM prediction performance",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ---------------------------------------------------------------------------
# Figure 7 — Dataset statistics
# ---------------------------------------------------------------------------

def fig_dataset_stats(results: dict, out_path: pathlib.Path):
    tasks   = list(results.keys())
    labels  = [TASK_LABELS.get(t, t) for t in tasks]
    n_train = [r["dataset"]["n_train_proteins"] for r in results.values()]
    n_test  = [r["dataset"]["n_test_proteins"]  for r in results.values()]
    pos_r   = [r["dataset"]["pos_rate_test"] * 100 for r in results.values()]
    x       = np.arange(len(tasks))
    colors  = [TASK_COLORS.get(t, "#333333") for t in tasks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(7, 1.3 * len(tasks)), 4))

    # Protein counts
    ax1.bar(x - 0.2, n_train, 0.38, label="Train", color=colors, alpha=0.9)
    ax1.bar(x + 0.2, n_test,  0.38, label="Test",  color=colors, alpha=0.45)
    ax1.set_xticks(x);  ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1.set_ylabel("Number of proteins")
    ax1.set_title("Dataset size per task")
    ax1.legend(frameon=False)

    # Positive residue rate
    ax2.bar(x, pos_r, color=colors, alpha=0.85, width=0.6)
    for xi, pr in zip(x, pos_r):
        ax2.text(xi, pr + 0.3, f"{pr:.1f}%", ha="center", va="bottom", fontsize=7.5)
    ax2.set_xticks(x);  ax2.set_xticklabels(labels, rotation=30, ha="right")
    ax2.set_ylabel("Positive residue rate (%)")
    ax2.set_title("Class balance (% positive residues, test set)")

    fig.suptitle("BEER v3.0 training dataset statistics", fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ---------------------------------------------------------------------------
# Combined ROC+PR two-panel figure (compact, suitable for paper main figure)
# ---------------------------------------------------------------------------

def fig_combined_compact(results: dict, out_path: pathlib.Path):
    """Single figure: ROC on left, PR on right, all tasks overlaid."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.5))

    for task, res in results.items():
        col   = TASK_COLORS.get(task, "#333333")
        label = TASK_LABELS.get(task, task)
        m     = res["test_metrics"]
        fpr   = res["roc_curve"]["fpr"]
        tpr   = res["roc_curve"]["tpr"]
        prec  = res["pr_curve"]["precision"]
        rec   = res["pr_curve"]["recall"]

        ax1.plot(fpr, tpr, color=col, lw=1.5,
                 label=f"{label} ({m['auroc']:.3f})")
        ax2.plot(rec, prec, color=col, lw=1.5,
                 label=f"{label} ({m['auprc']:.3f})")

    ax1.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
    ax1.set_xlim(-0.02, 1.02);  ax1.set_ylim(-0.02, 1.02)
    ax1.set_xlabel("False positive rate")
    ax1.set_ylabel("True positive rate")
    ax1.set_title("ROC curves")
    ax1.legend(frameon=False, fontsize=7, loc="lower right")
    ax1.set_aspect("equal")

    ax2.set_xlim(-0.02, 1.02);  ax2.set_ylim(-0.02, 1.05)
    ax2.set_xlabel("Recall");   ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall curves")
    ax2.legend(frameon=False, fontsize=7, loc="upper right")

    fig.suptitle("BEER v3.0 ESM2 650M + BiLSTM — held-out test performance",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

import math

ALL_TASKS = [
    "disorder", "signal_peptide", "transmembrane", "coiled_coil",
    "dna_binding", "active_site", "binding_site", "phosphorylation",
    "lcd", "zinc_finger", "glycosylation", "ubiquitination",
    "methylation", "acetylation", "lipidation", "disulfide",
    "intramembrane", "motif", "propeptide", "repeat",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="*", default=ALL_TASKS,
                        help="Tasks to include (default: all)")
    args = parser.parse_args()

    print(f"\nLoading results …", flush=True)
    results = load_results(args.tasks)
    if not results:
        print("No results found. Run train_all_heads.py first.")
        sys.exit(1)

    print(f"Loaded {len(results)} task result(s): {list(results.keys())}\n")

    fig_roc_all(results,    FIG_DIR / "fig_roc_all.png")
    fig_pr_all(results,     FIG_DIR / "fig_pr_all.png")
    fig_learning_curves(results, FIG_DIR / "fig_learning_curves.png")
    fig_auroc_violin(results,    FIG_DIR / "fig_auroc_violin.png")
    fig_calibration(results,     FIG_DIR / "fig_calibration.png")
    fig_auroc_summary(results,   FIG_DIR / "fig_auroc_summary.png")
    fig_dataset_stats(results,   FIG_DIR / "fig_dataset_stats.png")
    fig_combined_compact(results, FIG_DIR / "fig_combined_roc_pr.png")

    print(f"\nAll figures saved to {FIG_DIR}/")
    print("\nFigure inventory:")
    for f in sorted(FIG_DIR.glob("fig_*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
