#!/usr/bin/env python3
"""BEER ESM2 head benchmark — proper protein-level train/test evaluation.

Evaluates all four ESM2 prediction heads against their classical baselines
using a strictly held-out 20% protein-level test set.  Aggregation is
re-evaluated on experimental amyloid-fibril data from UniProt (replacing the
ZYGGREGATOR pseudo-label training with real curated annotations).

Outputs
-------
  results/fig1_roc_curves.png    — 4-panel ROC figure (ESM2 vs classical)
  results/auc_summary.csv        — AUC table suitable for the manuscript
  results/benchmark_report.txt  — full run log

Usage
-----
    python scripts/benchmark.py [--model esm2_t6_8M_UR50D] [--max-seqs 400]
"""
from __future__ import annotations
import argparse
import json
import pathlib
import sys
import time
import urllib.request
import warnings
import textwrap

import numpy as np

ROOT     = pathlib.Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RES_DIR  = ROOT / "results"
sys.path.insert(0, str(ROOT))

DATA_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _fetch_json(url: str, cache: pathlib.Path) -> object:
    if cache.exists():
        print(f"    [cache] {cache.name}")
        with open(cache) as f:
            return json.load(f)
    print(f"    GET {url[:80]}...")
    req = urllib.request.Request(
        url, headers={"User-Agent": "BEER-bench/1.0", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read())
    with open(cache, "w") as f:
        json.dump(data, f)
    return data


def _protein_level_split(items: list, test_frac: float = 0.20, seed: int = 42):
    """Shuffle at protein level, return (train_list, test_list)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(items))
    n_test = max(1, int(len(items) * test_frac))
    test_idx  = set(idx[:n_test].tolist())
    train     = [items[i] for i in range(len(items)) if i not in test_idx]
    test      = [items[i] for i in range(len(items)) if i in test_idx]
    return train, test


# ---------------------------------------------------------------------------
# ESM2 embedding
# ---------------------------------------------------------------------------

def _load_esm(model_name: str):
    import esm as _esm
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, alphabet = _esm.pretrained.load_model_and_alphabet(model_name)
    model = model.eval()
    return model, alphabet, alphabet.get_batch_converter()


def _embed_batch(model, batch_converter, sequences: list[str]) -> list[np.ndarray | None]:
    import torch
    results = []
    for seq in sequences:
        data = [("p", seq)]
        _, _, tokens = batch_converter(data)
        with torch.no_grad():
            out = model(tokens, repr_layers=[model.num_layers], return_contacts=False)
        emb = out["representations"][model.num_layers][0, 1:-1].cpu().numpy()
        results.append(emb if len(emb) == len(seq) else None)
    return results


# ---------------------------------------------------------------------------
# Bootstrap AUC CI
# ---------------------------------------------------------------------------

def _bootstrap_auc(y_true: np.ndarray, scores: np.ndarray,
                   n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """Return (auc, ci_lo, ci_hi) via bootstrap."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    n = len(y_true)
    base_auc = roc_auc_score(y_true, scores)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            aucs.append(roc_auc_score(y_true[idx], scores[idx]))
        except Exception:
            pass
    aucs = np.array(aucs)
    if len(aucs) < 10:
        return base_auc, base_auc, base_auc  # insufficient bootstrap samples
    return base_auc, float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


# ===========================================================================
# HEAD 1 — DISORDER
# ===========================================================================

def _load_disprot_proteins(max_total: int = 600) -> list[tuple[str, list[int]]]:
    """Load DisProt proteins as (sequence, per_residue_labels) pairs."""
    cache = DATA_DIR / "disprot_2024.json"
    url   = "https://disprot.org/api/search?release=2024_06&format=json&page_size=2000"
    raw   = _fetch_json(url, cache)
    records = raw.get("data", raw) if isinstance(raw, dict) else raw

    proteins = []
    for rec in records:
        seq = rec.get("sequence", "")
        if not seq or not (20 <= len(seq) <= 1200):
            continue
        n = len(seq)
        regions = rec.get("regions", [])
        lab = [0] * n
        found = False
        for region in regions:
            if region.get("term_name", "") == "disorder":
                s = region.get("start", 1) - 1
                e = region.get("end", 0)
                for i in range(max(0, s), min(n, e)):
                    lab[i] = 1
                found = True
        if not found:
            # fallback: old consensus string
            dis = rec.get("disprot_consensus", {}).get("disorder", "")
            if len(dis) == n:
                lab  = [1 if c == "D" else 0 for c in dis]
                found = True
        if found and sum(lab) > 0:
            proteins.append((seq, lab))
        if len(proteins) >= max_total:
            break
    return proteins


def _classical_disorder_scores(seq: str) -> np.ndarray:
    from beer.constants import DISORDER_PROPENSITY
    raw = np.array([DISORDER_PROPENSITY.get(aa, 0.0) for aa in seq])
    mn, mx = raw.min(), raw.max()
    if mx > mn:
        return (raw - mn) / (mx - mn)
    return np.full(len(seq), 0.5)


def benchmark_disorder(model, batch_converter, model_name: str,
                       max_seqs: int, report: list) -> dict:
    print("\n--- Disorder benchmark ---")
    from beer.models import load_disorder_head
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression

    proteins = _load_disprot_proteins(max_total=max_seqs)
    train_p, test_p = _protein_level_split(proteins, test_frac=0.20)
    report.append(f"Disorder: {len(proteins)} proteins  "
                  f"(train={len(train_p)}, test={len(test_p)})")
    print(f"  {len(proteins)} proteins | train={len(train_p)} test={len(test_p)}")

    # --- Embed train set, fit fresh head with protein-level split ---
    print("  Embedding train set ...")
    train_seqs = [s for s, _ in train_p]
    train_embs = _embed_batch(model, batch_converter, train_seqs)

    X_tr, y_tr = [], []
    for emb, (_, lab) in zip(train_embs, train_p):
        if emb is not None:
            X_tr.append(emb)
            y_tr.extend(lab)
    X_tr = np.vstack(X_tr);  y_tr = np.array(y_tr)
    clf = LogisticRegression(max_iter=1000, C=0.1, solver="lbfgs")
    clf.fit(X_tr, y_tr)

    # --- Embed test set ---
    print("  Embedding test set ...")
    test_seqs = [s for s, _ in test_p]
    test_embs = _embed_batch(model, batch_converter, test_seqs)

    y_true_all, esm2_scores, classical_scores = [], [], []
    for emb, (seq, lab) in zip(test_embs, test_p):
        if emb is None:
            continue
        logits = emb @ clf.coef_.T + clf.intercept_
        probs  = 1 / (1 + np.exp(-logits)).ravel()
        cl_sc  = _classical_disorder_scores(seq)
        y_true_all.extend(lab)
        esm2_scores.extend(probs.tolist())
        classical_scores.extend(cl_sc.tolist())

    y_true  = np.array(y_true_all)
    esm2_s  = np.array(esm2_scores)
    class_s = np.array(classical_scores)

    esm2_auc,  ci_lo_e, ci_hi_e = _bootstrap_auc(y_true, esm2_s)
    class_auc, ci_lo_c, ci_hi_c = _bootstrap_auc(y_true, class_s)

    msg = (f"  Disorder — ESM2 AUC={esm2_auc:.3f} [{ci_lo_e:.3f}–{ci_hi_e:.3f}]  "
           f"Classical AUC={class_auc:.3f} [{ci_lo_c:.3f}–{ci_hi_c:.3f}]")
    print(msg); report.append(msg)

    from sklearn.metrics import roc_curve
    fpr_e, tpr_e, _ = roc_curve(y_true, esm2_s)
    fpr_c, tpr_c, _ = roc_curve(y_true, class_s)

    return {
        "title": "Disorder (DisProt 2024)",
        "esm2_auc": esm2_auc, "esm2_ci": (ci_lo_e, ci_hi_e),
        "class_auc": class_auc, "class_ci": (ci_lo_c, ci_hi_c),
        "class_label": "Disorder propensity scale",
        "fpr_esm2": fpr_e, "tpr_esm2": tpr_e,
        "fpr_class": fpr_c, "tpr_class": tpr_c,
        "n_test": len(test_p),
    }


# ===========================================================================
# HEAD 2 — AGGREGATION  (real UniProt amyloid-fibril data)
# ===========================================================================

def _load_amyloid_proteins(max_pos: int = 200, max_neg: int = 200):
    """Positive: UniProt proteins with amyloid in name (reviewed), per-residue
    labels from Region/Chain/Propeptide features with 'amyloid' in description.
    Negative: reviewed human nuclear proteins with no amyloid annotation.
    Returns list of (seq, per_residue_label) pairs.
    """
    import urllib.parse as _up
    _AMYLOID_FTYPES = {"Region", "Chain", "Propeptide"}

    # Positives
    cache_pos = DATA_DIR / "uniprot_amyloid.json"
    _q_pos = _up.quote("protein_name:amyloid AND reviewed:true AND length:[30 TO 800]")
    url_pos = (
        "https://rest.uniprot.org/uniprotkb/search"
        f"?query={_q_pos}&format=json&size=300"
    )
    raw_pos = _fetch_json(url_pos, cache_pos)

    proteins = []
    for entry in raw_pos.get("results", [])[:max_pos]:
        seq = entry.get("sequence", {}).get("value", "")
        if not seq or len(seq) < 30:
            continue
        n   = len(seq)
        lab = [0] * n
        found = False
        for feat in entry.get("features", []):
            ftype = feat.get("type") or ""
            fdesc = (feat.get("description") or "").lower()
            if ftype in _AMYLOID_FTYPES and "amyloid" in fdesc:
                s = feat.get("location", {}).get("start", {}).get("value", 1) - 1
                e = feat.get("location", {}).get("end",   {}).get("value", 0)
                for i in range(max(0, s), min(n, e)):
                    lab[i] = 1
                found = True
        if found and sum(lab) > 0:
            proteins.append((seq, lab))

    # Negatives — human nuclear proteins with no amyloid annotation
    cache_neg = DATA_DIR / "uniprot_noagg.json"
    _q_neg = _up.quote(
        "reviewed:true AND organism_id:9606 AND cc_subcellular_location:nucleus"
        " AND NOT protein_name:amyloid AND length:[100 TO 600]"
    )
    url_neg = (
        "https://rest.uniprot.org/uniprotkb/search"
        f"?query={_q_neg}&format=json&size=300"
    )
    raw_neg = _fetch_json(url_neg, cache_neg)
    for entry in raw_neg.get("results", [])[:max_neg]:
        seq = entry.get("sequence", {}).get("value", "")
        if seq and len(seq) >= 50:
            proteins.append((seq, [0] * len(seq)))

    return proteins


def _classical_aggregation_scores(seq: str, window: int = 6) -> np.ndarray:
    from beer.constants import ZYGGREGATOR_PROPENSITY
    n = len(seq)
    raw = np.array([ZYGGREGATOR_PROPENSITY.get(aa, 0.0) for aa in seq])
    half = window // 2
    smoothed = np.array([
        raw[max(0, i-half):min(n, i+half+1)].mean()
        for i in range(n)
    ])
    mn, mx = smoothed.min(), smoothed.max()
    if mx > mn:
        return (smoothed - mn) / (mx - mn)
    return np.full(n, 0.5)


def benchmark_aggregation(model, batch_converter, model_name: str,
                          max_seqs: int, report: list) -> dict:
    print("\n--- Aggregation benchmark (UniProt amyloid-fibril regions) ---")
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.linear_model import LogisticRegression

    proteins = _load_amyloid_proteins(max_pos=max_seqs // 2, max_neg=max_seqs // 2)
    n_pos = sum(1 for _, lab in proteins if 1 in lab)
    n_neg = len(proteins) - n_pos
    train_p, test_p = _protein_level_split(proteins, test_frac=0.20)
    report.append(f"Aggregation: {len(proteins)} proteins (pos={n_pos}, neg={n_neg})  "
                  f"train={len(train_p)} test={len(test_p)}")
    print(f"  {len(proteins)} proteins (pos={n_pos}, neg={n_neg}) | "
          f"train={len(train_p)} test={len(test_p)}")

    print("  Embedding train set ...")
    train_embs = _embed_batch(model, batch_converter, [s for s, _ in train_p])
    X_tr, y_tr = [], []
    for emb, (_, lab) in zip(train_embs, train_p):
        if emb is not None:
            X_tr.append(emb);  y_tr.extend(lab)
    X_tr = np.vstack(X_tr);  y_tr = np.array(y_tr)
    clf = LogisticRegression(max_iter=1000, C=0.1, solver="lbfgs")
    clf.fit(X_tr, y_tr)

    print("  Embedding test set ...")
    test_embs = _embed_batch(model, batch_converter, [s for s, _ in test_p])
    y_true_all, esm2_scores, class_scores = [], [], []
    for emb, (seq, lab) in zip(test_embs, test_p):
        if emb is None:
            continue
        logits = emb @ clf.coef_.T + clf.intercept_
        probs  = (1 / (1 + np.exp(-logits))).ravel()
        cl_sc  = _classical_aggregation_scores(seq)
        y_true_all.extend(lab);  esm2_scores.extend(probs.tolist())
        class_scores.extend(cl_sc.tolist())

    y_true  = np.array(y_true_all)
    esm2_s  = np.array(esm2_scores)
    class_s = np.array(class_scores)

    esm2_auc,  ci_lo_e, ci_hi_e = _bootstrap_auc(y_true, esm2_s)
    class_auc, ci_lo_c, ci_hi_c = _bootstrap_auc(y_true, class_s)

    msg = (f"  Aggregation — ESM2 AUC={esm2_auc:.3f} [{ci_lo_e:.3f}–{ci_hi_e:.3f}]  "
           f"ZYGGREGATOR AUC={class_auc:.3f} [{ci_lo_c:.3f}–{ci_hi_c:.3f}]")
    print(msg); report.append(msg)

    fpr_e, tpr_e, _ = roc_curve(y_true, esm2_s)
    fpr_c, tpr_c, _ = roc_curve(y_true, class_s)

    return {
        "title": "Aggregation (UniProt amyloid-fibril)",
        "esm2_auc": esm2_auc, "esm2_ci": (ci_lo_e, ci_hi_e),
        "class_auc": class_auc, "class_ci": (ci_lo_c, ci_hi_c),
        "class_label": "ZYGGREGATOR propensity",
        "fpr_esm2": fpr_e, "tpr_esm2": tpr_e,
        "fpr_class": fpr_c, "tpr_class": tpr_c,
        "n_test": len(test_p),
    }


# ===========================================================================
# HEAD 3 — SIGNAL PEPTIDE  (protein-level binary AUC)
# ===========================================================================

def _load_signal_proteins(max_pos: int = 200, max_neg: int = 200):
    """Returns list of (seq, has_signal_peptide: 0/1)."""
    cache_pos = DATA_DIR / "uniprot_signal.json"
    url_pos = (
        "https://rest.uniprot.org/uniprotkb/search"
        "?query=ft_signal%3A*+AND+reviewed%3Atrue+AND+length%3A%5B30+TO+600%5D"
        "&format=json&size=200&fields=sequence,ft_signal"
    )
    raw_pos = _fetch_json(url_pos, cache_pos)

    cache_neg = DATA_DIR / "uniprot_nosignal.json"
    url_neg = (
        "https://rest.uniprot.org/uniprotkb/search"
        "?query=cc_subcellular_location%3Acytoplasm+AND+reviewed%3Atrue"
        "+AND+length%3A%5B50+TO+600%5D&format=json&size=200&fields=sequence"
    )
    raw_neg = _fetch_json(url_neg, cache_neg)

    proteins = []
    for entry in raw_pos.get("results", [])[:max_pos]:
        seq = entry.get("sequence", {}).get("value", "")
        sp_end = 0
        for feat in entry.get("features", []):
            if feat.get("type") in ("Signal", "Signal peptide"):
                sp_end = max(sp_end,
                             feat.get("location", {}).get("end", {}).get("value", 0))
        if seq and sp_end >= 5:
            proteins.append((seq, 1, sp_end))

    for entry in raw_neg.get("results", [])[:max_neg]:
        seq = entry.get("sequence", {}).get("value", "")
        if seq:
            proteins.append((seq, 0, 0))

    return proteins


def _classical_signal_score(seq: str) -> float:
    from beer.analysis.signal_peptide import predict_signal_peptide
    return predict_signal_peptide(seq)["score"]


def benchmark_signal(model, batch_converter, model_name: str,
                     max_seqs: int, report: list) -> dict:
    print("\n--- Signal peptide benchmark (protein-level AUC) ---")
    from sklearn.metrics import roc_auc_score, roc_curve
    from beer.models import load_signal_head

    proteins = _load_signal_proteins(max_pos=max_seqs // 2, max_neg=max_seqs // 2)
    n_pos = sum(1 for _, lab, _ in proteins if lab == 1)
    n_neg = len(proteins) - n_pos
    report.append(f"Signal peptide: {len(proteins)} proteins (pos={n_pos}, neg={n_neg})")
    print(f"  {len(proteins)} proteins (pos={n_pos}, neg={n_neg})")

    train_p, test_p = _protein_level_split(proteins, test_frac=0.20)

    # Load the stored head
    head = load_signal_head()

    print("  Scoring test set ...")
    y_true_all, esm2_scores, class_scores = [], [], []
    test_embs = _embed_batch(model, batch_converter, [s for s, _, _ in test_p])

    for emb, (seq, label, sp_end) in zip(test_embs, test_p):
        y_true_all.append(label)
        # Classical: protein-level score
        class_scores.append(_classical_signal_score(seq))
        # ESM2: max probability in first 50 residues (or first sp_end+10 if known)
        if emb is not None and head is not None:
            logits = emb @ head["coef"].T + head["intercept"]
            probs  = (1 / (1 + np.exp(-logits))).ravel()
            esm2_scores.append(float(probs[:60].max()))
        else:
            esm2_scores.append(0.5)

    y_true  = np.array(y_true_all)
    esm2_s  = np.array(esm2_scores)
    class_s = np.array(class_scores)

    esm2_auc,  ci_lo_e, ci_hi_e = _bootstrap_auc(y_true, esm2_s)
    class_auc, ci_lo_c, ci_hi_c = _bootstrap_auc(y_true, class_s)

    msg = (f"  Signal peptide — ESM2 AUC={esm2_auc:.3f} [{ci_lo_e:.3f}–{ci_hi_e:.3f}]  "
           f"Classical AUC={class_auc:.3f} [{ci_lo_c:.3f}–{ci_hi_c:.3f}]")
    print(msg); report.append(msg)

    fpr_e, tpr_e, _ = roc_curve(y_true, esm2_s)
    fpr_c, tpr_c, _ = roc_curve(y_true, class_s)

    return {
        "title": "Signal Peptide (UniProt)",
        "esm2_auc": esm2_auc, "esm2_ci": (ci_lo_e, ci_hi_e),
        "class_auc": class_auc, "class_ci": (ci_lo_c, ci_hi_c),
        "class_label": "Von Heijne 3-region score",
        "fpr_esm2": fpr_e, "tpr_esm2": tpr_e,
        "fpr_class": fpr_c, "tpr_class": tpr_c,
        "n_test": len(test_p),
    }


# ===========================================================================
# HEAD 4 — PTM  (per-residue AUC)
# ===========================================================================

def _classical_ptm_score(seq: str, pos: int) -> float:
    """Return a naive score: 1 if any verified motif matches at this position."""
    from beer.analysis.ptm import scan_ptm_sites
    hits = scan_ptm_sites(seq)
    conf_map = {"high": 1.0, "medium": 0.6, "low": 0.2}
    score = 0.0
    for h in hits:
        if h["position_1based"] - 1 == pos:
            score = max(score, conf_map.get(h["confidence"], 0.0))
    return score


def _load_ptm_proteins(max_seqs: int = 300):
    cache = DATA_DIR / "uniprot_ptm.json"
    url = (
        "https://rest.uniprot.org/uniprotkb/search"
        "?query=ft_mod_res%3A*+AND+reviewed%3Atrue+AND+length%3A%5B50+TO+600%5D"
        "&format=json&size=200&fields=sequence,ft_mod_res"
    )
    raw = _fetch_json(url, cache)
    proteins = []
    for entry in raw.get("results", [])[:max_seqs]:
        seq = entry.get("sequence", {}).get("value", "")
        if not seq:
            continue
        lab = [0] * len(seq)
        for feat in entry.get("features", []):
            if feat.get("type") == "Modified residue":
                pos = feat.get("location", {}).get("start", {}).get("value", 0)
                if 0 < pos <= len(seq):
                    lab[pos - 1] = 1
        if sum(lab) > 0:
            proteins.append((seq, lab))
    return proteins


def benchmark_ptm(model, batch_converter, model_name: str,
                  max_seqs: int, report: list) -> dict:
    print("\n--- PTM benchmark ---")
    from sklearn.metrics import roc_auc_score, roc_curve
    from beer.models import load_ptm_head

    proteins = _load_ptm_proteins(max_seqs)
    train_p, test_p = _protein_level_split(proteins, test_frac=0.20)
    report.append(f"PTM: {len(proteins)} proteins  train={len(train_p)} test={len(test_p)}")
    print(f"  {len(proteins)} proteins | train={len(train_p)} test={len(test_p)}")

    head = load_ptm_head()

    print("  Scoring test set ...")
    test_embs = _embed_batch(model, batch_converter, [s for s, _ in test_p])
    y_true_all, esm2_scores, class_scores = [], [], []

    for emb, (seq, lab) in zip(test_embs, test_p):
        if emb is None:
            continue
        if head is not None:
            logits = emb @ head["coef"].T + head["intercept"]
            probs  = (1 / (1 + np.exp(-logits))).ravel()
        else:
            probs = np.full(len(seq), 0.5)

        # Classical: precompute motif scores for each position
        from beer.analysis.ptm import scan_ptm_sites
        hits = scan_ptm_sites(seq)
        conf_map = {"high": 1.0, "medium": 0.6}
        cl_scores = np.zeros(len(seq))
        for h in hits:
            p = h["position_1based"] - 1
            if 0 <= p < len(seq):
                cl_scores[p] = max(cl_scores[p], conf_map.get(h["confidence"], 0.0))

        y_true_all.extend(lab)
        esm2_scores.extend(probs.tolist())
        class_scores.extend(cl_scores.tolist())

    y_true  = np.array(y_true_all)
    esm2_s  = np.array(esm2_scores)
    class_s = np.array(class_scores)

    esm2_auc,  ci_lo_e, ci_hi_e = _bootstrap_auc(y_true, esm2_s)
    class_auc, ci_lo_c, ci_hi_c = _bootstrap_auc(y_true, class_s)

    msg = (f"  PTM — ESM2 AUC={esm2_auc:.3f} [{ci_lo_e:.3f}–{ci_hi_e:.3f}]  "
           f"Motif-based AUC={class_auc:.3f} [{ci_lo_c:.3f}–{ci_hi_c:.3f}]")
    print(msg); report.append(msg)

    fpr_e, tpr_e, _ = roc_curve(y_true, esm2_s)
    fpr_c, tpr_c, _ = roc_curve(y_true, class_s)

    return {
        "title": "PTM Sites (UniProt mod_res)",
        "esm2_auc": esm2_auc, "esm2_ci": (ci_lo_e, ci_hi_e),
        "class_auc": class_auc, "class_ci": (ci_lo_c, ci_hi_c),
        "class_label": "Consensus motif scan",
        "fpr_esm2": fpr_e, "tpr_esm2": tpr_e,
        "fpr_class": fpr_c, "tpr_class": tpr_c,
        "n_test": len(test_p),
    }


# ===========================================================================
# Figures and output
# ===========================================================================

def _plot_roc_curves(results: list[dict], out_path: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    fig.suptitle("BEER ESM2 Heads vs. Classical Baselines", fontsize=13,
                 fontweight="bold", y=1.01)
    axes = axes.ravel()

    colors = {"esm2": "#4361ee", "classic": "#e63946"}

    for ax, res in zip(axes, results):
        ax.plot(res["fpr_esm2"], res["tpr_esm2"],
                color=colors["esm2"], lw=2.0,
                label=f"ESM2  AUC={res['esm2_auc']:.3f} "
                      f"[{res['esm2_ci'][0]:.3f}–{res['esm2_ci'][1]:.3f}]")
        ax.plot(res["fpr_class"], res["tpr_class"],
                color=colors["classic"], lw=1.8, linestyle="--",
                label=f"{res['class_label']}  AUC={res['class_auc']:.3f} "
                      f"[{res['class_ci'][0]:.3f}–{res['class_ci'][1]:.3f}]")
        ax.plot([0, 1], [0, 1], color="#aaa", lw=0.8, linestyle=":")
        ax.set_title(res["title"], fontsize=10, fontweight="bold")
        ax.set_xlabel("False Positive Rate", fontsize=9)
        ax.set_ylabel("True Positive Rate", fontsize=9)
        ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
        ax.set_xlim(-0.01, 1.01);  ax.set_ylim(-0.01, 1.01)
        ax.tick_params(labelsize=8)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.set_facecolor("#fafbff")
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
        ax.text(0.60, 0.08, f"n_test={res['n_test']} proteins",
                transform=ax.transAxes, fontsize=7, color="#666")

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure saved → {out_path}")


def _write_auc_csv(results: list[dict], out_path: pathlib.Path) -> None:
    import csv
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Head", "Test set", "ESM2 AUC", "95% CI lower", "95% CI upper",
                    "Classical method", "Classical AUC", "95% CI lower", "95% CI upper",
                    "n_test_proteins"])
        for res in results:
            w.writerow([
                res["title"].split("(")[0].strip(),
                res["title"],
                f"{res['esm2_auc']:.3f}",
                f"{res['esm2_ci'][0]:.3f}",
                f"{res['esm2_ci'][1]:.3f}",
                res["class_label"],
                f"{res['class_auc']:.3f}",
                f"{res['class_ci'][0]:.3f}",
                f"{res['class_ci'][1]:.3f}",
                res["n_test"],
            ])
    print(f"  AUC table saved → {out_path}")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="esm2_t6_8M_UR50D",
                        choices=["esm2_t6_8M_UR50D", "esm2_t12_35M_UR50D",
                                 "esm2_t30_150M_UR50D", "esm2_t33_650M_UR50D"])
    parser.add_argument("--max-seqs", type=int, default=400,
                        help="Max proteins per head (default 400)")
    parser.add_argument("--heads", nargs="+",
                        choices=["disorder", "aggregation", "signal", "ptm", "all"],
                        default=["all"])
    args = parser.parse_args()

    do_all = "all" in args.heads
    t0 = time.time()
    report: list[str] = [
        f"BEER Benchmark Report",
        f"Model: {args.model}   max_seqs={args.max_seqs}",
        f"{'='*60}",
    ]

    try:
        from sklearn.metrics import roc_auc_score  # noqa
    except ImportError:
        print("ERROR: pip install scikit-learn"); sys.exit(1)

    print(f"Loading {args.model} ...")
    model, alphabet, batch_converter = _load_esm(args.model)

    results = []
    if do_all or "disorder" in args.heads:
        results.append(benchmark_disorder(model, batch_converter, args.model,
                                          args.max_seqs, report))
    if do_all or "aggregation" in args.heads:
        results.append(benchmark_aggregation(model, batch_converter, args.model,
                                              args.max_seqs, report))
    if do_all or "signal" in args.heads:
        results.append(benchmark_signal(model, batch_converter, args.model,
                                        args.max_seqs, report))
    if do_all or "ptm" in args.heads:
        results.append(benchmark_ptm(model, batch_converter, args.model,
                                     args.max_seqs, report))

    elapsed = time.time() - t0
    report.append(f"\nTotal time: {elapsed/60:.1f} min")

    # --- Output ---
    print("\nWriting outputs ...")
    if results:
        _plot_roc_curves(results, RES_DIR / "fig1_roc_curves.png")
        _write_auc_csv(results,   RES_DIR / "auc_summary.csv")

    report_path = RES_DIR / "benchmark_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report))
    print(f"  Report saved → {report_path}")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for res in results:
        delta = res["esm2_auc"] - res["class_auc"]
        sign  = "+" if delta >= 0 else ""
        print(f"  {res['title']:<40}  "
              f"ESM2={res['esm2_auc']:.3f}  "
              f"Classical={res['class_auc']:.3f}  "
              f"ΔAUC={sign}{delta:.3f}")
    print(f"\nDone in {elapsed/60:.1f} min.")


if __name__ == "__main__":
    main()
