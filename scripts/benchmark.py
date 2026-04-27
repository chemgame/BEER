#!/usr/bin/env python3
"""BEER v2.0 benchmark script.

Evaluates BEER's disorder and signal-peptide classifiers against standard
benchmark datasets and prints AUROC + 95% bootstrap confidence intervals.

Outputs
-------
- benchmark_results.json        : machine-readable results
- figures/fig_benchmark_roc.png : 2-panel ROC figure (disorder + SP)

Dependencies
------------
numpy, scipy, matplotlib, requests
Optional: torch, esm  (for ESM2 classifier evaluation)

Usage
-----
    python scripts/benchmark.py
"""
from __future__ import annotations

import json
import sys
import urllib.request
import urllib.parse
import io
import pathlib
import random
import time
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from beer.analysis.aggregation import calc_aggregation_profile
from beer.analysis.signal_peptide import predict_signal_peptide


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

# ---------------------------------------------------------------------------
# DisProt 2024 REST API
# ---------------------------------------------------------------------------
DISPROT_API = "https://disprot.org/api/search?format=json&page_size=200"


def fetch_disprot() -> list[dict]:
    """Fetch DisProt entries via REST API (current schema: data[], size)."""
    print("Fetching DisProt …")
    entries = []
    page = 1
    while True:
        url = f"{DISPROT_API}&page={page}"
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = json.loads(resp.read())
        batch = data.get("data", [])
        entries.extend(batch)
        total = data.get("size", len(entries))
        print(f"  page {page}: {len(batch)} entries (total so far {len(entries)}/{total})")
        if len(entries) >= total or len(batch) == 0:
            break
        page += 1
        time.sleep(0.15)
    print(f"  {len(entries)} DisProt entries fetched.")
    return entries


def build_disorder_labels(entries: list[dict]) -> tuple[list[str], list[np.ndarray]]:
    """Return (sequences, per-residue binary label arrays) from DisProt entries."""
    seqs, labels = [], []
    for entry in entries:
        seq = entry.get("sequence", "")
        if not seq or len(seq) < 20:
            continue
        lab = np.zeros(len(seq), dtype=np.int8)
        # disprot_consensus.full holds {start, end, type} with type="D" for disorder
        for region in entry.get("disprot_consensus", {}).get("full", []):
            if region.get("type", "D") != "D":
                continue
            start = region["start"] - 1  # 1-based → 0-based
            end = region["end"]          # end is inclusive in DisProt
            lab[start:end] = 1
        if lab.sum() == 0 or lab.sum() == len(lab):
            continue
        seqs.append(seq)
        labels.append(lab)
    return seqs, labels


# ---------------------------------------------------------------------------
# UniProt signal-peptide benchmark set
# ---------------------------------------------------------------------------
UNIPROT_SP_QUERY = (
    "https://rest.uniprot.org/uniprotkb/search?query="
    "reviewed:true+AND+ft_signal:*&format=fasta&size=500"
)
UNIPROT_NOSP_QUERY = (
    "https://rest.uniprot.org/uniprotkb/search?query="
    "reviewed:true+AND+organism_id:9606+AND+cc_subcellular_location:cytoplasm"
    "+AND+NOT+ft_signal:*&format=fasta&size=500"
)


def fetch_fasta_sequences(url: str) -> list[tuple[str, str]]:
    """Fetch FASTA sequences from a UniProt REST URL.  Returns (accession, seq) pairs."""
    print(f"  GET {url[:80]}…")
    with urllib.request.urlopen(url, timeout=90) as resp:
        text = resp.read().decode()
    pairs = []
    acc, buf = None, []
    for line in text.splitlines():
        if line.startswith(">"):
            if acc and buf:
                pairs.append((acc, "".join(buf)))
            acc = line.split("|")[1] if "|" in line else line[1:].split()[0]
            buf = []
        else:
            buf.append(line.strip())
    if acc and buf:
        pairs.append((acc, "".join(buf)))
    return pairs


# ---------------------------------------------------------------------------
# Classical IUPred2-style disorder baseline: hydrophilicity proxy
# (mean KD over 21-aa window, inverted; used only if ESM2 unavailable)
# ---------------------------------------------------------------------------

def iupred_proxy(seq: str, window: int = 21) -> np.ndarray:
    """Very rough disorder proxy: negative mean Kyte-Doolittle (inverted, scaled 0-1)."""
    from beer.constants import KYTE_DOOLITTLE
    n = len(seq)
    scores = np.zeros(n)
    for i in range(n):
        lo = max(0, i - window // 2)
        hi = min(n, i + window // 2 + 1)
        s = [KYTE_DOOLITTLE.get(aa, 0.0) for aa in seq[lo:hi]]
        scores[i] = -np.mean(s)
    # Normalise to [0,1]
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    return scores


# ---------------------------------------------------------------------------
# BEER aggregation-based disorder proxy (Z_agg, inverted as anti-aggregation)
# This is deliberately distinct from the ESM2 head to serve as a classical
# comparison even when ESM2 is not available.
# ---------------------------------------------------------------------------

def zagg_disorder_proxy(seq: str) -> np.ndarray:
    """Use negative Z_agg as a coarse classical disorder proxy."""
    z = np.array(calc_aggregation_profile(seq))
    scores = -z
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    return scores


# ---------------------------------------------------------------------------
# ESM2 disorder head (if available)
# ---------------------------------------------------------------------------

def esm2_disorder_scores(seqs: list[str]) -> Optional[list[np.ndarray]]:
    """Run BEER's ESM2 disorder head on a list of sequences."""
    try:
        from beer.embeddings.esm2_embedder import ESM2Embedder
    except ImportError:
        return None
    embedder = ESM2Embedder()
    if not embedder.is_available():
        return None
    try:
        head_data = np.load(ROOT / "beer" / "models" / "disorder_head.npz",
                            allow_pickle=True)
        W = head_data["coef"]    # shape (1, n_features) or (n_features,)
        b = head_data["intercept"]
    except Exception as e:
        print(f"  Could not load disorder head: {e}")
        return None

    W = np.atleast_2d(W)  # ensure (1, n_features)

    results = []
    for i, seq in enumerate(seqs):
        if i % 20 == 0:
            print(f"  ESM2 scoring {i}/{len(seqs)} …", flush=True)
        emb = embedder.embed(seq)
        if emb is None:
            return None
        logits = emb @ W.T + b   # (L, 1)
        probs = 1.0 / (1.0 + np.exp(-logits[:, 0]))
        results.append(probs)
    return results


# ---------------------------------------------------------------------------
# AUROC and bootstrap CI
# ---------------------------------------------------------------------------

def compute_roc(y_true: np.ndarray, y_score: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (fpr, tpr, auroc) arrays."""
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.array([0, 1]), np.array([0, 1]), float("nan")
    tp = np.cumsum(y_true_sorted)
    fp = np.cumsum(1 - y_true_sorted)
    tpr = np.concatenate([[0], tp / n_pos])
    fpr = np.concatenate([[0], fp / n_neg])
    auroc = float(np.trapz(tpr, fpr))
    return fpr, tpr, auroc


def compute_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    _, _, auroc = compute_roc(y_true, y_score)
    return auroc


def bootstrap_auroc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, float, float]:
    """Return (auroc, lower_ci, upper_ci) via stratified bootstrap."""
    rng = rng or np.random.default_rng(RNG_SEED)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    aurocs = []
    for _ in range(n_boot):
        b_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        b_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([b_pos, b_neg])
        aurocs.append(compute_auroc(y_true[idx], y_score[idx]))
    aurocs = np.array(aurocs)
    alpha = (1 - ci) / 2
    lo = float(np.nanpercentile(aurocs, alpha * 100))
    hi = float(np.nanpercentile(aurocs, (1 - alpha) * 100))
    return compute_auroc(y_true, y_score), lo, hi


# ---------------------------------------------------------------------------
# Signal peptide benchmark
# ---------------------------------------------------------------------------

def run_sp_benchmark() -> dict:
    """Download UniProt SP/non-SP sets, evaluate BEER signal peptide detector."""
    print("\n=== Signal Peptide Benchmark ===")
    sp_seqs = fetch_fasta_sequences(UNIPROT_SP_QUERY)
    nosp_seqs = fetch_fasta_sequences(UNIPROT_NOSP_QUERY)

    y_true_list, y_score_list = [], []
    for _, seq in sp_seqs[:300]:
        result = predict_signal_peptide(seq)
        y_true_list.append(1)
        # Use 1.0 if SP predicted, score proportional to h-region KD
        y_score_list.append(float(result.get("d_score", 0.0)))
    for _, seq in nosp_seqs[:300]:
        result = predict_signal_peptide(seq)
        y_true_list.append(0)
        y_score_list.append(float(result.get("d_score", 0.0)))

    y_true = np.array(y_true_list)
    y_score = np.array(y_score_list)
    fpr, tpr, auroc = compute_roc(y_true, y_score)
    _, lo, hi = bootstrap_auroc(y_true, y_score)
    print(f"  SP AUROC = {auroc:.3f} ({lo:.3f}–{hi:.3f})")
    return {
        "method": "BEER_sequence",
        "auroc": auroc,
        "ci_lower": lo,
        "ci_upper": hi,
        "n_pos": int(y_true.sum()),
        "n_neg": int((1 - y_true).sum()),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    }


# ---------------------------------------------------------------------------
# Disorder benchmark
# ---------------------------------------------------------------------------

def run_disorder_benchmark() -> list[dict]:
    """Download DisProt 2024, evaluate classical and ESM2 disorder proxies."""
    print("\n=== Disorder Benchmark ===")
    entries = fetch_disprot()
    seqs, labels = build_disorder_labels(entries)
    print(f"  {len(seqs)} proteins with mixed disorder/order labels.")

    # Pool all residues (cap at 200 proteins to stay manageable)
    seqs = seqs[:200]
    labels = labels[:200]

    y_true_pool = np.concatenate(labels)

    results = []

    # Classical KD-proxy
    print("  Running KD-hydrophilicity disorder proxy …")
    kd_scores = np.concatenate([iupred_proxy(s) for s in seqs])
    fpr_kd, tpr_kd, auroc_kd = compute_roc(y_true_pool, kd_scores)
    _, lo_kd, hi_kd = bootstrap_auroc(y_true_pool, kd_scores)
    print(f"  KD-proxy AUROC = {auroc_kd:.3f} ({lo_kd:.3f}–{hi_kd:.3f})")
    results.append({"method": "KD_proxy", "auroc": auroc_kd,
                    "ci_lower": lo_kd, "ci_upper": hi_kd,
                    "fpr": fpr_kd.tolist(), "tpr": tpr_kd.tolist()})

    # ESM2 disorder head
    print("  Running ESM2 disorder head …")
    esm_scores_list = esm2_disorder_scores(seqs)
    if esm_scores_list is not None:
        esm_scores = np.concatenate(esm_scores_list)
        fpr_esm, tpr_esm, auroc_e = compute_roc(y_true_pool, esm_scores)
        _, lo_e, hi_e = bootstrap_auroc(y_true_pool, esm_scores)
        print(f"  ESM2 AUROC = {auroc_e:.3f} ({lo_e:.3f}–{hi_e:.3f})")
        results.append({
            "method": "BEER_ESM2",
            "auroc": auroc_e,
            "ci_lower": lo_e,
            "ci_upper": hi_e,
            "fpr": fpr_esm.tolist(),
            "tpr": tpr_esm.tolist(),
        })
    else:
        print("  ESM2 unavailable — skipping.")

    for r in results:
        r["n_residues"] = int(len(y_true_pool))
        r["n_proteins"] = len(seqs)

    return results


# ---------------------------------------------------------------------------
# ROC figure
# ---------------------------------------------------------------------------

def plot_roc_figure(disorder_results: list[dict], sp_result: dict,
                    outpath: pathlib.Path) -> None:
    """Save a 2-panel ROC figure with actual curves."""
    COLORS = {
        "KD_proxy":      ("#f3722c", "--"),
        "BEER_ESM2":     ("#4361ee", "-"),
        "BEER_sequence": ("#43aa8b", "-"),
    }
    LABELS = {
        "KD_proxy":      "KD hydrophilicity proxy",
        "BEER_ESM2":     "BEER (ESM2 linear probe)",
        "BEER_sequence": "BEER (sequence heuristic)",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=150)

    for ax, title in zip(axes, ["Disorder (DisProt, 200 proteins)",
                                  "Signal Peptide (UniProt, 600 entries)"]):
        ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=1.0, linestyle="--",
                label="Random (AUROC=0.500)", zorder=1)
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    # Disorder panel
    for res in disorder_results:
        color, ls = COLORS.get(res["method"], ("#888888", "-"))
        label = LABELS.get(res["method"], res["method"])
        auroc, lo, hi = res["auroc"], res["ci_lower"], res["ci_upper"]
        fpr = np.array(res["fpr"])
        tpr = np.array(res["tpr"])
        axes[0].plot(fpr, tpr, color=color, lw=2.0, linestyle=ls, zorder=3,
                     label=f"{label}\nAUROC = {auroc:.3f} ({lo:.3f}–{hi:.3f})")

    axes[0].legend(fontsize=8, loc="lower right", framealpha=0.9,
                   edgecolor="#d0d4e0", handlelength=1.5)

    # Signal peptide panel
    color, ls = COLORS["BEER_sequence"]
    label = LABELS["BEER_sequence"]
    auroc = sp_result["auroc"]
    lo, hi = sp_result["ci_lower"], sp_result["ci_upper"]
    fpr = np.array(sp_result["fpr"])
    tpr = np.array(sp_result["tpr"])
    axes[1].plot(fpr, tpr, color=color, lw=2.0, linestyle=ls, zorder=3,
                 label=f"{label}\nAUROC = {auroc:.3f} ({lo:.3f}–{hi:.3f})")
    axes[1].legend(fontsize=8, loc="lower right", framealpha=0.9,
                   edgecolor="#d0d4e0", handlelength=1.5)

    fig.tight_layout(pad=2.0)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    print(f"  Saved ROC figure → {outpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    out_dir = ROOT / "benchmark_results"
    out_dir.mkdir(exist_ok=True)

    all_results: dict = {}

    # Disorder
    disorder_res = run_disorder_benchmark()
    all_results["disorder"] = disorder_res

    # Signal peptide
    sp_res = run_sp_benchmark()
    all_results["signal_peptide"] = sp_res

    # Save JSON
    json_path = out_dir / "benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {json_path}")

    # ROC figure
    plot_roc_figure(
        disorder_res, sp_res,
        out_dir / "fig_benchmark_roc.png",
    )

    # Print summary table
    print("\n=== Summary ===")
    print(f"{'Task':<20} {'Method':<25} {'AUROC':>7}  {'95% CI'}")
    print("-" * 65)
    for res in disorder_res:
        print(
            f"{'Disorder':<20} {res['method']:<25} "
            f"{res['auroc']:>7.3f}  [{res['ci_lower']:.3f}–{res['ci_upper']:.3f}]"
        )
    print(
        f"{'Signal Peptide':<20} {sp_res['method']:<25} "
        f"{sp_res['auroc']:>7.3f}  [{sp_res['ci_lower']:.3f}–{sp_res['ci_upper']:.3f}]"
    )


if __name__ == "__main__":
    main()
