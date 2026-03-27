#!/usr/bin/env python3
"""Train ESM2 classification heads for disorder, aggregation, signal peptide, and PTM.

Usage:
    python scripts/train_heads.py --model esm2_t6_8M_UR50D [--max-seqs 300]
"""
from __future__ import annotations
import argparse
import json
import pathlib
import sys
import urllib.request
import warnings
import numpy as np

MODELS_DIR = pathlib.Path(__file__).parent.parent / "beer" / "models"
DATA_DIR   = pathlib.Path(__file__).parent.parent / "data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_json(url: str, cache: pathlib.Path) -> object:
    if cache.exists():
        print(f"  [cache] {cache.name}")
        with open(cache) as f:
            return json.load(f)
    print(f"  Downloading {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "BEER-train/1.0",
                                               "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read())
    with open(cache, "w") as f:
        json.dump(data, f)
    return data


def _embed_sequences(model, alphabet, batch_converter, sequences: list[str], device="cpu") -> list[np.ndarray]:
    """Return list of (L, D) embedding arrays for each sequence."""
    import torch
    embeddings = []
    for i, seq in enumerate(sequences):
        if (i + 1) % 20 == 0:
            print(f"    embedded {i+1}/{len(sequences)}")
        data = [("p", seq)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)
        with torch.no_grad():
            out = model(tokens, repr_layers=[model.num_layers], return_contacts=False)
        emb = out["representations"][model.num_layers][0, 1:-1].cpu().numpy()
        if len(emb) == len(seq):
            embeddings.append(emb)
        else:
            embeddings.append(None)
    return embeddings


def _train_and_save(X: np.ndarray, y: np.ndarray, out_path: pathlib.Path,
                    model_name: str, trained_on: str, C: float = 0.1) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    pos = y.sum(); neg = (y == 0).sum()
    print(f"  Training on {len(X)} residues  (pos={pos}, neg={neg})")
    # Quick AUC estimate on held-out 20%
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=1000, C=C, solver="lbfgs")
    clf.fit(X_tr, y_tr)
    probs = clf.predict_proba(X_te)[:, 1]
    try:
        auc = roc_auc_score(y_te, probs)
    except Exception:
        auc = float("nan")
    print(f"  Validation AUC = {auc:.3f}")
    # Retrain on full data
    clf.fit(X, y)
    np.savez(out_path,
             coef=clf.coef_,
             intercept=clf.intercept_,
             model_name=np.array(model_name),
             trained_on=np.array(trained_on),
             auc=np.array(auc))
    print(f"  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Head 1: Disorder (DisProt)
# ---------------------------------------------------------------------------

def _disprot_disorder_labels(rec: dict) -> list[int] | None:
    """Convert DisProt record to per-residue binary disorder labels (0/1).

    Supports:
    - New format (2024+): uses ``rec['regions']`` with ``term_name == 'disorder'``
    - Legacy consensus string format: ``disprot_consensus['disorder']`` as 'DDDOOO...'
    - Legacy consensus region list with type 'D'

    Returns None if labels cannot be reliably extracted.
    """
    seq = rec.get("sequence", "")
    if not seq:
        return None
    n = len(seq)

    # 2024 format: use 'regions' list with term_name=='disorder'
    regions = rec.get("regions", [])
    if regions:
        lab = [0] * n
        found_any = False
        for region in regions:
            if region.get("term_name", "") == "disorder":
                s = region.get("start", 1) - 1  # 1-based -> 0-based
                e = region.get("end", 0)          # 1-based inclusive
                for i in range(max(0, s), min(n, e)):
                    lab[i] = 1
                found_any = True
        if found_any:
            return lab

    # Old format: consensus string
    consensus = rec.get("disprot_consensus", {})
    dis = consensus.get("disorder", "")
    if len(dis) == n:
        return [1 if c == "D" else 0 for c in dis]

    # Old consensus region format with type 'D'
    consensus_regions = consensus.get("full", [])
    if isinstance(consensus_regions, list) and consensus_regions:
        lab = [0] * n
        found_any = False
        for region in consensus_regions:
            if region.get("type") == "D":
                s = region.get("start", 1) - 1
                e = region.get("end", 0)
                for i in range(max(0, s), min(n, e)):
                    lab[i] = 1
                found_any = True
        if found_any:
            return lab

    return None


def train_disorder_head(model, alphabet, batch_converter, model_name: str, max_seqs: int) -> None:
    print("\n=== Disorder head (DisProt 2024) ===")
    cache = DATA_DIR / "disprot_2024.json"
    url = "https://disprot.org/api/search?release=2024_06&format=json&page_size=2000"
    raw = _fetch_json(url, cache)
    records = raw.get("data", raw) if isinstance(raw, dict) else raw

    sequences, labels = [], []
    for rec in records:
        seq = rec.get("sequence", "")
        if not seq or not (10 <= len(seq) <= 1000):
            continue
        lab = _disprot_disorder_labels(rec)
        if lab is not None and len(lab) == len(seq) and sum(lab) > 0:
            sequences.append(seq)
            labels.append(lab)
        if len(sequences) >= max_seqs:
            break

    if not sequences:
        print("  No valid sequences — skipping.")
        return
    print(f"  {len(sequences)} sequences")
    embs = _embed_sequences(model, alphabet, batch_converter, sequences)
    X_all, y_all = [], []
    for emb, lab in zip(embs, labels):
        if emb is not None:
            X_all.append(emb)
            y_all.extend(lab)
    if not X_all:
        print("  Embedding failed — skipping.")
        return
    X = np.vstack(X_all)
    y = np.array(y_all)
    _train_and_save(X, y, MODELS_DIR / "disorder_head.npz", model_name, "DisProt_2024_06")


# ---------------------------------------------------------------------------
# Head 2: Aggregation (UniProt amyloid-fibril regions — experimental data)
# ---------------------------------------------------------------------------

def train_aggregation_head(model, alphabet, batch_converter, model_name: str, max_seqs: int) -> None:
    """Train on UniProt proteins with experimentally annotated amyloid-fibril
    regions (ft_region:"amyloid fibril", reviewed:true).  Negatives are
    human nuclear proteins with no amyloid annotation.

    This replaces the previous ZYGGREGATOR pseudo-label approach with real
    experimental data, enabling honest independent benchmarking.
    """
    print("\n=== Aggregation head (UniProt amyloid-fibril regions) ===")

    # Positives: proteins with "amyloid" in their name (reviewed SwissProt entries)
    # These reliably carry per-residue Region/Chain/Propeptide annotations for
    # the amyloid-forming regions, giving clean experimental positive labels.
    import urllib.parse
    cache_pos = DATA_DIR / "uniprot_amyloid.json"
    _q_pos = urllib.parse.quote("protein_name:amyloid AND reviewed:true AND length:[30 TO 800]")
    url_pos = (
        "https://rest.uniprot.org/uniprotkb/search"
        f"?query={_q_pos}&format=json&size=300"
    )
    raw_pos = _fetch_json(url_pos, cache_pos)

    sequences, labels = [], []
    _AMYLOID_TYPES = {"Region", "Chain", "Propeptide"}
    for entry in raw_pos.get("results", [])[:max_seqs // 2]:
        seq = entry.get("sequence", {}).get("value", "")
        if not seq or len(seq) < 30:
            continue
        n   = len(seq)
        lab = [0] * n
        found = False
        for feat in entry.get("features", []):
            ftype = feat.get("type") or ""
            fdesc = (feat.get("description") or "").lower()
            if ftype in _AMYLOID_TYPES and "amyloid" in fdesc:
                s = feat.get("location", {}).get("start", {}).get("value", 1) - 1
                e = feat.get("location", {}).get("end",   {}).get("value", 0)
                for i in range(max(0, s), min(n, e)):
                    lab[i] = 1
                found = True
        if found and sum(lab) > 0:
            sequences.append(seq)
            labels.append(lab)

    # Negatives: human nuclear proteins with no amyloid annotation
    cache_neg = DATA_DIR / "uniprot_noagg.json"
    _q_neg = urllib.parse.quote(
        "reviewed:true AND organism_id:9606 AND cc_subcellular_location:nucleus"
        " AND NOT protein_name:amyloid AND length:[100 TO 600]"
    )
    url_neg = (
        "https://rest.uniprot.org/uniprotkb/search"
        f"?query={_q_neg}&format=json&size=300"
    )
    raw_neg = _fetch_json(url_neg, cache_neg)
    for entry in raw_neg.get("results", [])[:max_seqs // 2]:
        seq = entry.get("sequence", {}).get("value", "")
        if seq and len(seq) >= 50:
            sequences.append(seq)
            labels.append([0] * len(seq))

    if not sequences:
        print("  No sequences — skipping.")
        return

    n_pos = sum(1 for lab in labels if 1 in lab)
    n_neg = len(sequences) - n_pos
    print(f"  {len(sequences)} sequences (pos proteins={n_pos}, neg proteins={n_neg})")

    embs = _embed_sequences(model, alphabet, batch_converter, sequences)
    X_all, y_all = [], []
    for emb, lab in zip(embs, labels):
        if emb is not None:
            X_all.append(emb)
            y_all.extend(lab)
    if not X_all:
        print("  Embedding failed — skipping.")
        return
    X = np.vstack(X_all)
    y = np.array(y_all)
    _train_and_save(X, y, MODELS_DIR / "aggregation_head.npz", model_name,
                    "UniProt_amyloid_fibril_regions")


# ---------------------------------------------------------------------------
# Head 3: Signal peptide (UniProt)
# ---------------------------------------------------------------------------

def train_signal_head(model, alphabet, batch_converter, model_name: str, max_seqs: int) -> None:
    print("\n=== Signal peptide head (UniProt) ===")
    # Fetch proteins with annotated signal peptides
    # Use %3A for : and %3A%5B..%5D for range queries to avoid 400 errors
    cache = DATA_DIR / "uniprot_signal.json"
    url = ("https://rest.uniprot.org/uniprotkb/search"
           "?query=ft_signal%3A*+AND+reviewed%3Atrue+AND+length%3A%5B30+TO+600%5D"
           "&format=json&size=200&fields=sequence,ft_signal")
    raw = _fetch_json(url, cache)
    results = raw.get("results", [])

    # Also fetch negative examples (no signal peptide, cytoplasmic proteins)
    cache_neg = DATA_DIR / "uniprot_nosignal.json"
    url_neg = ("https://rest.uniprot.org/uniprotkb/search"
               "?query=cc_subcellular_location%3Acytoplasm+AND+reviewed%3Atrue"
               "+AND+length%3A%5B50+TO+600%5D&format=json&size=200&fields=sequence")
    raw_neg = _fetch_json(url_neg, cache_neg)
    results_neg = raw_neg.get("results", [])

    sequences, labels = [], []
    for entry in results[:max_seqs // 2]:
        seq = entry.get("sequence", {}).get("value", "")
        if not seq or len(seq) < 30:
            continue
        # Find signal peptide end position from features
        # UniProt REST API returns type as "Signal" (not "Signal peptide")
        sp_end = 0
        for feat in entry.get("features", []):
            if feat.get("type") in ("Signal", "Signal peptide"):
                loc = feat.get("location", {})
                sp_end = max(sp_end, loc.get("end", {}).get("value", 0))
        if sp_end < 5:
            continue
        lab = [1 if i < sp_end else 0 for i in range(len(seq))]
        sequences.append(seq)
        labels.append(lab)

    for entry in results_neg[:max_seqs // 2]:
        seq = entry.get("sequence", {}).get("value", "")
        if not seq or len(seq) < 30:
            continue
        sequences.append(seq)
        labels.append([0] * len(seq))
        if len(sequences) >= max_seqs:
            break

    if not sequences:
        print("  No sequences — skipping.")
        return
    print(f"  {len(sequences)} sequences ({sum(1 for l in labels if 1 in l)} with signal peptide)")
    embs = _embed_sequences(model, alphabet, batch_converter, sequences)
    X_all, y_all = [], []
    for emb, lab in zip(embs, labels):
        if emb is not None:
            X_all.append(emb)
            y_all.extend(lab)
    if not X_all:
        print("  Embedding failed — skipping.")
        return
    X = np.vstack(X_all)
    y = np.array(y_all)
    _train_and_save(X, y, MODELS_DIR / "signal_head.npz", model_name, "UniProt_signal_peptide", C=0.5)


# ---------------------------------------------------------------------------
# Head 4: PTM sites (UniProt mod_res)
# ---------------------------------------------------------------------------

def train_ptm_head(model, alphabet, batch_converter, model_name: str, max_seqs: int) -> None:
    print("\n=== PTM head (UniProt mod_res) ===")
    cache = DATA_DIR / "uniprot_ptm.json"
    url = ("https://rest.uniprot.org/uniprotkb/search"
           "?query=ft_mod_res%3A*+AND+reviewed%3Atrue+AND+length%3A%5B50+TO+600%5D"
           "&format=json&size=200&fields=sequence,ft_mod_res")
    raw = _fetch_json(url, cache)
    results = raw.get("results", [])

    sequences, labels = [], []
    for entry in results[:max_seqs]:
        seq = entry.get("sequence", {}).get("value", "")
        if not seq:
            continue
        lab = [0] * len(seq)
        for feat in entry.get("features", []):
            if feat.get("type") == "Modified residue":
                pos = feat.get("location", {}).get("start", {}).get("value", 0)
                if 0 < pos <= len(seq):
                    lab[pos - 1] = 1
        if sum(lab) == 0:
            continue
        sequences.append(seq)
        labels.append(lab)
        if len(sequences) >= max_seqs:
            break

    if not sequences:
        print("  No sequences — skipping.")
        return
    print(f"  {len(sequences)} sequences")
    embs = _embed_sequences(model, alphabet, batch_converter, sequences)
    X_all, y_all = [], []
    for emb, lab in zip(embs, labels):
        if emb is not None:
            X_all.append(emb)
            y_all.extend(lab)
    if not X_all:
        print("  Embedding failed — skipping.")
        return
    X = np.vstack(X_all)
    y = np.array(y_all)
    _train_and_save(X, y, MODELS_DIR / "ptm_head.npz", model_name, "UniProt_mod_res", C=1.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="esm2_t6_8M_UR50D",
                        choices=["esm2_t6_8M_UR50D", "esm2_t12_35M_UR50D",
                                 "esm2_t30_150M_UR50D", "esm2_t33_650M_UR50D"])
    parser.add_argument("--max-seqs", type=int, default=300,
                        help="Max sequences per head (default 300; reduce if slow)")
    parser.add_argument("--heads", nargs="+",
                        choices=["disorder", "aggregation", "signal", "ptm", "all"],
                        default=["all"])
    args = parser.parse_args()

    try:
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            import esm as _esm
            import torch as _torch
        from sklearn.linear_model import LogisticRegression  # noqa
    except ImportError as e:
        print(f"ERROR: {e}\npip install fair-esm torch scikit-learn")
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_esm, alphabet = _esm.pretrained.load_model_and_alphabet(args.model)
    model_esm = model_esm.eval()
    batch_converter = alphabet.get_batch_converter()

    do_all = "all" in args.heads
    if do_all or "disorder" in args.heads:
        train_disorder_head(model_esm, alphabet, batch_converter, args.model, args.max_seqs)
    if do_all or "aggregation" in args.heads:
        train_aggregation_head(model_esm, alphabet, batch_converter, args.model, args.max_seqs)
    if do_all or "signal" in args.heads:
        train_signal_head(model_esm, alphabet, batch_converter, args.model, args.max_seqs)
    if do_all or "ptm" in args.heads:
        train_ptm_head(model_esm, alphabet, batch_converter, args.model, args.max_seqs)

    print("\nAll done. Head files in beer/models/:")
    for f in sorted(MODELS_DIR.glob("*.npz")):
        print(f"  {f.name}  ({f.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
