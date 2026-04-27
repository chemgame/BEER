#!/usr/bin/env python3
"""Audit training datasets for all BEER BiLSTM heads before committing to training.

Run this BEFORE train_all_heads.py to identify heads with insufficient data,
extreme class imbalance, or too few clusters for a reliable split.

Output: a per-head table with pass/warn/fail status for each metric.

Usage
-----
    conda run -n beer python scripts/dataset_audit.py
    conda run -n beer python scripts/dataset_audit.py --tasks disorder transmembrane
    conda run -n beer python scripts/dataset_audit.py --json audit_results.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_all_heads import (
    TASKS, CACHE_DIR, fetch_uniprot, build_labels, MAX_SEQ_LEN, RNG_SEED
)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

MIN_PROTEINS        = 200     # fewer → FAIL (not enough to train)
WARN_PROTEINS       = 500     # fewer → WARN
MIN_POS_RESIDUES    = 5_000   # fewer positive residues → FAIL
WARN_POS_RESIDUES   = 15_000  # fewer → WARN
MIN_POS_FRACTION    = 0.002   # <0.2% positive fraction → FAIL
WARN_POS_FRACTION   = 0.005   # <0.5% → WARN
MIN_VAL_POS         = 300     # fewer positive residues in val → FAIL (unreliable metrics)
WARN_VAL_POS        = 800     # fewer → WARN
MIN_CLUSTERS        = 50      # fewer clusters after MMseqs2 → WARN
CLUSTER_ID          = 0.30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_fasta(seqs: list[str], path: str) -> None:
    with open(path, "w") as f:
        for i, s in enumerate(seqs):
            f.write(f">seq{i}\n{s}\n")


def count_clusters(seqs: list[str]) -> int | None:
    """Return number of clusters at CLUSTER_ID identity, or None if mmseqs unavailable."""
    try:
        subprocess.run(["mmseqs", "version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    with tempfile.TemporaryDirectory(prefix="beer_audit_") as tmpdir:
        fasta  = f"{tmpdir}/seqs.fasta"
        clust  = f"{tmpdir}/clusters"
        tsv    = f"{tmpdir}/clusters_cluster.tsv"
        _write_fasta(seqs, fasta)
        try:
            subprocess.run(
                ["mmseqs", "easy-cluster", fasta, clust, tmpdir,
                 "--min-seq-id", str(CLUSTER_ID), "--cov-mode", "0",
                 "-c", "0.8", "--cluster-mode", "0", "-v", "0"],
                check=True, capture_output=True, timeout=300,
            )
            reps = set()
            with open(tsv) as f:
                for line in f:
                    reps.add(line.split("\t")[0])
            return len(reps)
        except Exception:
            return None


def flag(value, warn_thr, fail_thr, higher_is_better: bool = True) -> str:
    if higher_is_better:
        if value < fail_thr:
            return "FAIL"
        if value < warn_thr:
            return "WARN"
        return "OK"
    else:
        if value > fail_thr:
            return "FAIL"
        if value > warn_thr:
            return "WARN"
        return "OK"


# ---------------------------------------------------------------------------
# Per-task audit
# ---------------------------------------------------------------------------

def audit_task(task_name: str, cfg: dict, skip_clustering: bool) -> dict:
    print(f"\n  [{task_name}]", flush=True)

    proteins = fetch_uniprot(task_name, cfg)
    seqs, labels = build_labels(proteins, cfg)

    n_prot  = len(seqs)
    n_res   = sum(len(l) for l in labels)
    n_pos   = int(sum(l.sum() for l in labels))
    n_neg   = n_res - n_pos
    pos_frac = n_pos / max(n_res, 1)

    # Estimate val-set positives with a 10% hold-out
    n_val_prot = max(1, int(0.10 * n_prot))
    rng = np.random.default_rng(RNG_SEED)
    val_idx = rng.choice(n_prot, n_val_prot, replace=False)
    val_pos = int(sum(labels[i].sum() for i in val_idx))

    # Clustering (may be slow — skip with --no-cluster)
    n_clusters: int | None = None
    if not skip_clustering and n_prot > 0:
        print(f"    Clustering {n_prot} sequences …", flush=True)
        n_clusters = count_clusters(seqs)

    # Flags
    f_prot    = flag(n_prot,   WARN_PROTEINS,     MIN_PROTEINS)
    f_posres  = flag(n_pos,    WARN_POS_RESIDUES,  MIN_POS_RESIDUES)
    f_posfrac = flag(pos_frac, WARN_POS_FRACTION,  MIN_POS_FRACTION)
    f_valpos  = flag(val_pos,  WARN_VAL_POS,       MIN_VAL_POS)
    f_clust   = flag(n_clusters or 9999, MIN_CLUSTERS, 10) if n_clusters is not None else "SKIP"

    overall = "OK"
    for f in (f_prot, f_posres, f_posfrac, f_valpos):
        if f == "FAIL":
            overall = "FAIL"; break
        if f == "WARN" and overall == "OK":
            overall = "WARN"

    print(f"    proteins={n_prot} [{f_prot}]  "
          f"pos_residues={n_pos} [{f_posres}]  "
          f"pos_frac={pos_frac:.4f} [{f_posfrac}]  "
          f"val_pos~{val_pos} [{f_valpos}]  "
          f"clusters={n_clusters if n_clusters is not None else 'N/A'} [{f_clust}]  "
          f"→ {overall}", flush=True)

    return {
        "task": task_name,
        "description": cfg["description"],
        "n_proteins": n_prot,
        "n_residues": n_res,
        "n_positive_residues": n_pos,
        "n_negative_residues": n_neg,
        "pos_fraction": round(pos_frac, 6),
        "val_pos_estimate": val_pos,
        "n_clusters_30pct": n_clusters,
        "flags": {
            "proteins": f_prot,
            "pos_residues": f_posres,
            "pos_fraction": f_posfrac,
            "val_positives": f_valpos,
            "clusters": f_clust,
        },
        "overall": overall,
        "recommendation": _recommend(overall, f_prot, f_posres, f_posfrac, f_valpos, n_clusters),
    }


def _recommend(overall, f_prot, f_posres, f_posfrac, f_valpos, n_clusters) -> str:
    if overall == "OK":
        return "Ready to train."
    parts = []
    if f_prot == "FAIL":
        parts.append("Too few proteins — collect more data before training.")
    if f_posres == "FAIL":
        parts.append("Too few positive residues — model will be unreliable.")
    if f_posfrac == "FAIL":
        parts.append(
            "Extreme class imbalance (<0.2%) — consider whether this head is trainable "
            "from UniProt annotations alone; a curated specialist database is needed."
        )
    if f_valpos == "FAIL":
        parts.append(
            "Validation set will have too few positives for reliable F1/AUROC — "
            "increase dataset size or reduce val fraction."
        )
    if n_clusters is not None and n_clusters < MIN_CLUSTERS:
        parts.append(
            f"Only {n_clusters} sequence clusters — split may be unreliable. "
            "Consider pooling with related tasks or using a larger database."
        )
    if f_prot == "WARN":
        parts.append("Borderline protein count — consider fetching more.")
    return " ".join(parts) if parts else "Minor warnings; proceed with caution."


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="*", default=list(TASKS.keys()),
                        choices=list(TASKS.keys()),
                        help="Tasks to audit (default: all)")
    parser.add_argument("--json", metavar="FILE", default=None,
                        help="Write full results to a JSON file")
    parser.add_argument("--no-cluster", action="store_true",
                        help="Skip MMseqs2 clustering (faster, omits cluster count)")
    args = parser.parse_args()

    print(f"\nBEER dataset audit — {len(args.tasks)} task(s)", flush=True)
    print(f"Thresholds: proteins≥{MIN_PROTEINS}, pos_residues≥{MIN_POS_RESIDUES}, "
          f"pos_frac≥{MIN_POS_FRACTION:.3f}, val_pos≥{MIN_VAL_POS}\n", flush=True)

    all_results = []
    for task_name in args.tasks:
        try:
            result = audit_task(task_name, TASKS[task_name], args.no_cluster)
            all_results.append(result)
        except Exception as exc:
            import traceback
            print(f"  ERROR auditing {task_name}: {exc}", flush=True)
            traceback.print_exc()
            all_results.append({"task": task_name, "overall": "ERROR", "error": str(exc)})

    # Summary table
    print(f"\n{'='*70}", flush=True)
    print(f"  {'TASK':<24} {'PROTEINS':>9} {'POS_RES':>9} {'POS%':>7} {'VAL_POS':>8} {'STATUS':>6}", flush=True)
    print(f"{'='*70}", flush=True)
    n_ok = n_warn = n_fail = 0
    for r in all_results:
        if "error" in r:
            print(f"  {r['task']:<24}  ERROR", flush=True)
            n_fail += 1
            continue
        status = r["overall"]
        if status == "OK":    n_ok   += 1
        elif status == "WARN": n_warn += 1
        else:                  n_fail += 1
        print(f"  {r['task']:<24} {r['n_proteins']:>9,} {r['n_positive_residues']:>9,} "
              f"{r['pos_fraction']*100:>6.2f}% {r['val_pos_estimate']:>8,}  {status}", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  OK={n_ok}  WARN={n_warn}  FAIL={n_fail}", flush=True)

    if n_fail > 0:
        print(f"\n  FAIL tasks require action before training:", flush=True)
        for r in all_results:
            if r.get("overall") == "FAIL":
                print(f"    • {r['task']}: {r.get('recommendation','')}", flush=True)

    if n_warn > 0:
        print(f"\n  WARN tasks — review before training:", flush=True)
        for r in all_results:
            if r.get("overall") == "WARN":
                print(f"    • {r['task']}: {r.get('recommendation','')}", flush=True)

    if args.json:
        out_path = pathlib.Path(args.json)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Full results → {out_path}", flush=True)

    print(flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
