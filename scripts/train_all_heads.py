#!/usr/bin/env python3
"""Train ESM2 + BiLSTM heads for all per-residue predictions in BEER.

Architecture per task
---------------------
  ESM2 650M (1280-dim, on MPS/GPU/CPU)  →  per-residue embeddings
  2-layer Bidirectional LSTM (hidden=256) →  per-residue context vectors (512-dim)
  Linear(512→1)  →  per-residue logit  →  sigmoid  →  probability

Why BiLSTM over MLP+window
  • Reads the full protein sequence in both directions before predicting each residue
  • Naturally captures long-range dependencies (disorder regions, TM topology)
  • No explicit window context needed — simpler 1280-dim input
  • 50-100× faster to train with PyTorch vs pure numpy

Speedups enabled here
  • ESM2 on MPS (Apple Silicon GPU)  — ~15× over CPU for embedding
  • BiLSTM training on MPS            — ~50× over CPU numpy
  • PackedSequence batching            — no wasted compute on padding
  • Shared embedding cache per model  — embeddings reused across tasks

Tasks trained (Swiss-Prot reviewed proteins)
  signal_peptide   ft_signal      N-terminal secretion signal
  transmembrane    ft_transmem    TM helix (multi-pass proteins)
  coiled_coil      ft_coiled      Coiled-coil region
  dna_binding      ft_dna_bind    DNA-binding residues
  active_site      ft_act_site    Catalytic residues
  binding_site     ft_binding     Ligand-binding residues
  phosphorylation  ft_mod_res     Phospho-Ser/Thr/Tyr

Saved as  beer/models/{task}_head.npz  with numpy weights for BEER inference.

Usage
-----
    conda run -n beer python scripts/train_all_heads.py
    conda run -n beer python scripts/train_all_heads.py --tasks signal_peptide transmembrane
    conda run -n beer python scripts/train_all_heads.py --model esm2_t12_35M_UR50D
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

import subprocess
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()
print(f"Training device: {DEVICE}", flush=True)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

RNG_SEED       = 42
LSTM_HIDDEN    = 256
LSTM_LAYERS    = 2
LSTM_DROPOUT   = 0.3
LR             = 5e-4          # slightly lower for stability with AdamW
WEIGHT_DECAY   = 1e-4          # L2 regularisation — guards against overfitting
WARMUP_EPOCHS  = 3             # linear LR warmup
EPOCHS         = 50            # more epochs; early stopping will cut short
BATCH_PROTEINS = 16
PATIENCE       = 8             # a bit more patience with LR scheduling
DEFAULT_MODEL  = "esm2_t33_650M_UR50D"
FOCAL_GAMMA    = 2.0           # focal loss focusing parameter (Lin et al. 2017)
CLUSTER_ID     = 0.30          # MMseqs2 sequence identity threshold for split

CACHE_DIR  = ROOT / "scripts" / ".head_caches"
MODELS_DIR = ROOT / "beer" / "models"

random.seed(RNG_SEED)
np.random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)

# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

TASKS: dict[str, dict] = {
    "disorder": {
        "description": "Intrinsically disordered region",
        "uniprot_query": "ft_region:disordered AND reviewed:true",
        "uniprot_field": "ft_region",
        "feature_types": {"Region"},
        "description_filter": "disordered",
        "max_proteins": 8000,
        # DisProt experimental annotations override UniProt computed annotations.
        # Produced by: python scripts/fetch_curated_data.py --source disprot
        "cache_override": "disorder_disprot.json",
    },
    "signal_peptide": {
        "description": "Signal peptide (N-terminal secretion signal)",
        "uniprot_query": "ft_signal:* AND reviewed:true",
        "uniprot_field": "ft_signal",
        "feature_types": {"Signal"},
        "description_filter": None,
        "max_proteins": 8000,
    },
    "transmembrane": {
        "description": "Transmembrane helix",
        "uniprot_query": "ft_transmem:* AND reviewed:true",
        "uniprot_field": "ft_transmem",
        "feature_types": {"Transmembrane"},
        "description_filter": None,
        "max_proteins": 6000,
        "architecture": "bilstm2_crf",   # BiLSTM + CRF with Viterbi topology decoding
    },
    "coiled_coil": {
        "description": "Coiled-coil region",
        "uniprot_query": "ft_coiled:* AND reviewed:true",
        "uniprot_field": "ft_coiled",
        "feature_types": {"Coiled coil"},
        "description_filter": None,
        "max_proteins": 6000,
    },
    "dna_binding": {
        "description": "DNA-binding region",
        "uniprot_query": "ft_dna_bind:* AND reviewed:true",
        "uniprot_field": "ft_dna_bind",
        "feature_types": {"DNA binding"},
        "description_filter": None,
        "max_proteins": 6000,
        # BioLiP structure-derived protein-DNA contacts override UniProt.
        "cache_override": "dna_binding_biolip.json",
    },
    "active_site": {
        "description": "Catalytic active-site residue",
        "uniprot_query": "ft_act_site:* AND reviewed:true",
        "uniprot_field": "ft_act_site",
        "feature_types": {"Active site"},
        "description_filter": None,
        "max_proteins": 8000,
        # M-CSA mechanistically validated catalytic residues override UniProt active site.
        # Produced by: python scripts/fetch_curated_data.py --source mcsa
        "cache_override": "active_site_mcsa.json",
    },
    "binding_site": {
        "description": "Ligand-binding residue",
        "uniprot_query": "ft_binding:* AND reviewed:true",
        "uniprot_field": "ft_binding",
        "feature_types": {"Binding site"},
        "description_filter": None,
        "max_proteins": 8000,
        # BioLiP structure-derived binding residues override UniProt binding site.
        # Produced by: python scripts/fetch_curated_data.py --source biolip
        "cache_override": "binding_site_biolip.json",
    },
    "phosphorylation": {
        "description": "Phosphorylation site (Ser/Thr/Tyr)",
        "uniprot_query": "ft_mod_res:Phospho* AND reviewed:true",
        "uniprot_field": "ft_mod_res",
        "feature_types": {"Modified residue"},
        "description_filter": "phospho",
        "max_proteins": 8000,
        # PhosphoSitePlus LTP experimental sites override UniProt phosphorylation.
        # Produced by: python scripts/fetch_curated_data.py --source psp
        # (requires PhosphoSitePlus bulk download — see README for registration)
        "cache_override": "phosphorylation_psp.json",
    },
    "lcd": {
        "description": "Low-complexity / compositionally biased region",
        "uniprot_query": "ft_compbias:* AND reviewed:true",
        "uniprot_field": "ft_compbias",
        "feature_types": {"Compositional bias"},
        "description_filter": None,
        "max_proteins": 8000,
    },
    "zinc_finger": {
        "description": "Zinc finger domain",
        "uniprot_query": "ft_zn_fing:* AND reviewed:true",
        "uniprot_field": "ft_zn_fing",
        "feature_types": {"Zinc finger"},
        "description_filter": None,
        "max_proteins": 6000,
        # BioLiP structure-derived Zn-coordination residues override UniProt.
        "cache_override": "zinc_finger_biolip.json",
    },
    "glycosylation": {
        "description": "Glycosylation site (N- and O-linked)",
        "uniprot_query": "ft_carbohyd:* AND reviewed:true",
        "uniprot_field": "ft_carbohyd",
        "feature_types": {"Glycosylation"},
        "description_filter": None,
        "max_proteins": 8000,
        # GlyConnect experimentally validated glycosylation sites override UniProt.
        "cache_override": "glycosylation_glyconnect.json",
    },
    "ubiquitination": {
        "description": "Ubiquitination site (Lys)",
        "uniprot_query": "ft_mod_res:Ubiquitin AND reviewed:true",
        "uniprot_field": "ft_mod_res",
        "feature_types": {"Modified residue"},
        "description_filter": "ubiquitin",
        "max_proteins": 6000,
        "cache_override": "ubiquitination_psp.json",
    },
    "methylation": {
        "description": "Methylation site",
        "uniprot_query": "ft_mod_res:Methyl AND reviewed:true",
        "uniprot_field": "ft_mod_res",
        "feature_types": {"Modified residue"},
        "description_filter": "methyl",
        "max_proteins": 6000,
        "cache_override": "methylation_psp.json",
    },
    "acetylation": {
        "description": "Lysine / N-terminal acetylation",
        "uniprot_query": "ft_mod_res:Acetyl AND reviewed:true",
        "uniprot_field": "ft_mod_res",
        "feature_types": {"Modified residue"},
        "description_filter": "acetyl",
        "max_proteins": 8000,
        "cache_override": "acetylation_psp.json",
    },
    "lipidation": {
        "description": "Lipidation site (myristoyl, palmitoyl, GPI)",
        "uniprot_query": "ft_lipid:* AND reviewed:true",
        "uniprot_field": "ft_lipid",
        "feature_types": {"Lipidation"},
        "description_filter": None,
        "max_proteins": 6000,
    },
    "disulfide": {
        "description": "Disulfide bond (Cys-Cys)",
        "uniprot_query": "ft_disulfid:* AND reviewed:true",
        "uniprot_field": "ft_disulfid",
        "feature_types": {"Disulfide bond"},
        "description_filter": None,
        "label_mode": "endpoints",
        "max_proteins": 6000,
        # PDB SSBOND structure-derived disulfides override UniProt annotations.
        # Produced by: python scripts/fetch_curated_data.py --source pdb_ssbond
        "cache_override": "disulfide_pdb.json",
    },
    "intramembrane": {
        "description": "Intramembrane region",
        "uniprot_query": "ft_intramem:* AND reviewed:true",
        "uniprot_field": "ft_intramem",
        "feature_types": {"Intramembrane"},
        "description_filter": None,
        "max_proteins": 6000,
    },
    "motif": {
        "description": "Short linear motif (SLiM)",
        "uniprot_query": "ft_motif:* AND reviewed:true",
        "uniprot_field": "ft_motif",
        "feature_types": {"Motif"},
        "description_filter": None,
        "max_proteins": 6000,
    },
    "propeptide": {
        "description": "Propeptide region",
        "uniprot_query": "ft_propep:* AND reviewed:true",
        "uniprot_field": "ft_propep",
        "feature_types": {"Propeptide"},
        "description_filter": None,
        "max_proteins": 6000,
    },
    "repeat": {
        "description": "Tandem repeat region",
        "uniprot_query": "ft_repeat:* AND reviewed:true",
        "uniprot_field": "ft_repeat",
        "feature_types": {"Repeat"},
        "description_filter": None,
        "max_proteins": 6000,
    },
    "rna_binding": {
        "description": "RNA-binding region",
        "uniprot_query": "ft_region:RNA-binding AND reviewed:true",
        "uniprot_field": "ft_region",
        "feature_types": {"Region"},
        "description_filter": "rna-binding",
        "max_proteins": 6000,
        # BioLiP structure-derived protein-RNA contacts override UniProt.
        "cache_override": "rna_binding_biolip.json",
    },
    "nucleotide_binding": {
        "description": "Nucleotide-binding region (ATP/GTP/NAD etc.)",
        "uniprot_query": "ft_binding:* AND reviewed:true",
        "uniprot_field": "ft_binding",
        "feature_types": {"Binding site"},
        "description_filter": "nucleotide",
        "max_proteins": 6000,
        # BioLiP structure-derived nucleotide cofactor contacts override UniProt.
        "cache_override": "nucleotide_binding_biolip.json",
    },
    "transit_peptide": {
        "description": "Mitochondrial / chloroplast transit peptide",
        "uniprot_query": "ft_transit:* AND reviewed:true",
        "uniprot_field": "ft_transit",
        "feature_types": {"Transit peptide"},
        "description_filter": None,
        "max_proteins": 6000,
    },
    "aggregation": {
        "description": "Aggregation propensity (amyloid-forming stretch)",
        "uniprot_query": "reviewed:true AND keyword:KW-0043",
        "uniprot_field": "ft_region",
        "feature_types": {"Region"},
        "description_filter": "amyloid",
        "max_proteins": 5000,
        "architecture": "bilstm2_window",
        # WALTZ-DB + AmyLoad + AmyPro + PDB fibrils + soluble negatives.
        # Produced by: python scripts/fetch_curated_data.py --source aggregation
        "cache_override": "aggregation_curated.json",
    },
}

# ---------------------------------------------------------------------------
# UniProt fetcher
# ---------------------------------------------------------------------------

def _fetch_url(url: str, max_retries: int = 6) -> tuple[bytes, dict]:
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(url, timeout=90) as resp:
                return resp.read(), {k.lower(): v for k, v in resp.headers.items()}
        except (urllib.error.URLError, TimeoutError) as exc:
            wait = 15 * (2 ** attempt)
            print(f"    attempt {attempt+1}/{max_retries} failed ({exc}); "
                  f"retrying in {wait}s …", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch {url} after {max_retries} attempts")


def fetch_uniprot(task_name: str, cfg: dict) -> list[dict]:
    cache_path = CACHE_DIR / f"{task_name}_uniprot.json"
    if cache_path.exists():
        print(f"  Loading {task_name} from cache …", flush=True)
        with open(cache_path) as f:
            proteins = json.load(f)
        print(f"  {len(proteins)} proteins.", flush=True)
        return proteins

    query = cfg["uniprot_query"]
    field = cfg["uniprot_field"]
    max_n = cfg["max_proteins"]
    print(f"  Fetching UniProt: {query}  (max {max_n}) …", flush=True)

    url = (
        "https://rest.uniprot.org/uniprotkb/search"
        f"?query={urllib.parse.quote(query)}"
        f"&format=json&fields=accession,sequence,{field}&size=500"
    )
    proteins: list[dict] = []
    while url and len(proteins) < max_n:
        body, headers = _fetch_url(url)
        data = json.loads(body)
        proteins.extend(data.get("results", []))
        print(f"    {len(proteins)} proteins …", flush=True)
        # Use regex — the URL contains commas (fields param) so split(",") breaks it
        link_hdr = headers.get("link", "")
        url = ""
        m = re.search(r'<([^>]+)>;\s*rel="next"', link_hdr)
        if m:
            url = m.group(1)
        time.sleep(0.2)

    proteins = proteins[:max_n]
    CACHE_DIR.mkdir(exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(proteins, f)
    print(f"  Cached {len(proteins)} to {cache_path}", flush=True)
    return proteins

# ---------------------------------------------------------------------------
# Label builder
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 1024  # must match embedding truncation


def load_curated_override(override_path: pathlib.Path) -> tuple[list[str], list[np.ndarray]] | None:
    """Load pre-built binary labels from a curated-data override file.

    Override files are produced by scripts/fetch_curated_data.py and contain:
        [{"seq": "MAST...", "labels": [0, 1, 0, ...]}, ...]

    This bypasses the UniProt fetch + build_labels() pipeline entirely,
    replacing it with experimentally validated per-residue annotations.
    Returns (seqs, labels) or None if file missing/corrupt.
    """
    if not override_path.exists():
        return None
    try:
        with open(override_path) as f:
            data = json.load(f)
        seqs, labels = [], []
        for entry in data:
            seq = entry.get("seq", "")[:MAX_SEQ_LEN]
            if len(seq) < 20:
                continue
            lab = np.array(entry["labels"][:len(seq)], dtype=np.int8)
            if lab.sum() == 0 or lab.sum() == len(lab):
                continue
            seqs.append(seq)
            labels.append(lab)
        pos_rate = sum(l.sum() for l in labels) / max(sum(len(l) for l in labels), 1)
        print(f"  Loaded {len(seqs)} curated proteins from {override_path.name} "
              f"(pos rate: {pos_rate:.4f})", flush=True)
        return seqs, labels
    except Exception as e:
        print(f"  WARNING: could not load override {override_path}: {e}", flush=True)
        return None

def build_labels(proteins: list[dict], cfg: dict) -> tuple[list[str], list[np.ndarray]]:
    feat_types   = cfg["feature_types"]
    desc_filter  = cfg.get("description_filter")
    label_mode   = cfg.get("label_mode", "region")  # "region" or "endpoints"
    seqs, labels = [], []
    for prot in proteins:
        seq = prot.get("sequence", {}).get("value", "")
        seq = seq[:MAX_SEQ_LEN]  # truncate to match embedding
        if len(seq) < 20:
            continue
        lab = np.zeros(len(seq), dtype=np.int8)
        for feat in prot.get("features", []):
            if feat.get("type") not in feat_types:
                continue
            if desc_filter and desc_filter not in feat.get("description", "").lower():
                continue
            sv = feat["location"]["start"].get("value")
            ev = feat["location"]["end"].get("value")
            if sv is None or ev is None:
                continue   # skip features with unknown/fuzzy boundaries
            s = max(0, sv - 1)
            e = min(len(seq), ev)
            if s < e:
                if label_mode == "endpoints":
                    lab[s] = 1
                    lab[e - 1] = 1
                else:
                    lab[s:e] = 1
        if lab.sum() == 0 or lab.sum() == len(lab):
            continue
        seqs.append(seq)
        labels.append(lab)
    pos_rate = sum(l.sum() for l in labels) / max(sum(len(l) for l in labels), 1)
    print(f"  {len(seqs)} proteins with mixed labels  (pos rate: {pos_rate:.4f})", flush=True)
    return seqs, labels

# ---------------------------------------------------------------------------
# ESM2 embedding on MPS/GPU/CPU — batched for speed
# ---------------------------------------------------------------------------

def _embed_cache_dir(model_name: str) -> pathlib.Path:
    """Directory holding one .npy file per cached sequence."""
    d = CACHE_DIR / f".embed_cache_{model_name}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _seq_cache_path(cache_dir: pathlib.Path, seq: str) -> pathlib.Path:
    import hashlib
    key = hashlib.md5(seq.encode()).hexdigest()
    return cache_dir / f"{key}.npy"


def embed_all(seqs: list[str], model_name: str) -> list[np.ndarray]:
    """Embed all sequences with ESM2, using a per-protein file cache.

    Each sequence is saved as an individual .npy file keyed by md5(seq).
    Saves are O(1) and loads never pull the whole cache into memory.
    """
    import esm as esm_module

    cache_dir = _embed_cache_dir(model_name)

    # Identify missing sequences
    missing_idx = [i for i, s in enumerate(seqs)
                   if not _seq_cache_path(cache_dir, s).exists()]
    n_cached = len(seqs) - len(missing_idx)
    print(f"  {n_cached} cached  /  {len(missing_idx)} to embed  "
          f"(total {len(seqs)}) …", flush=True)

    if missing_idx:
        esm_model, alphabet = getattr(esm_module.pretrained, model_name)()
        esm_model = esm_model.to(DEVICE).eval()
        batch_converter = alphabet.get_batch_converter()
        repr_layer = esm_model.num_layers

        missing_idx.sort(key=lambda i: len(seqs[i]))
        batch_size = 4  # reduced from 8 to cut peak RAM during embedding

        for batch_start in range(0, len(missing_idx), batch_size):
            batch_idxs = missing_idx[batch_start:batch_start + batch_size]
            data = [(str(i), seqs[i][:MAX_SEQ_LEN]) for i in batch_idxs]
            _, _, toks = batch_converter(data)
            toks = toks.to(DEVICE)

            with torch.no_grad():
                out = esm_model(toks, repr_layers=[repr_layer],
                                return_contacts=False)
            reps = out["representations"][repr_layer].cpu().numpy()

            for j, orig_i in enumerate(batch_idxs):
                L = min(len(seqs[orig_i]), MAX_SEQ_LEN)
                emb = reps[j, 1:L+1, :]
                np.save(_seq_cache_path(cache_dir, seqs[orig_i]), emb)

            # explicitly free GPU/MPS memory after each batch
            del toks, out, reps
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

            if batch_start % 500 == 0:
                pct = 100 * batch_start / len(missing_idx)
                print(f"  Embedded {batch_start}/{len(missing_idx)} "
                      f"({pct:.0f}%) …", flush=True)

        # free ESM2 from RAM entirely before training starts
        del esm_model, alphabet, batch_converter
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        import gc; gc.collect()
        print(f"  Embedding complete. ESM2 unloaded from memory.", flush=True)

    return [np.load(_seq_cache_path(cache_dir, s)) for s in seqs]

# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class ProteinDataset(Dataset):
    """Lazy-loading dataset — loads each embedding from disk per batch to avoid OOM."""
    def __init__(self, seqs: list[str], labels: list[np.ndarray],
                 idxs: np.ndarray, cache_dir: pathlib.Path):
        self.seqs      = seqs
        self.labels    = labels
        self.idxs      = idxs
        self.cache_dir = cache_dir

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        i   = self.idxs[item]
        emb = np.load(_seq_cache_path(self.cache_dir, self.seqs[i]))
        return (torch.from_numpy(emb.astype(np.float32)),
                torch.from_numpy(self.labels[i].astype(np.float32)))


def collate_fn(batch):
    """Pad sequences to max length in batch, return mask."""
    embs, labs = zip(*batch)
    lengths = torch.tensor([e.shape[0] for e in embs], dtype=torch.long)
    embs_padded = pad_sequence(embs, batch_first=True)   # (B, L_max, D)
    labs_padded = pad_sequence(labs, batch_first=True)   # (B, L_max)
    return embs_padded, labs_padded, lengths

# ---------------------------------------------------------------------------
# BiLSTM model
# ---------------------------------------------------------------------------

class BiLSTMHead(nn.Module):
    """2-layer bidirectional LSTM → per-residue binary classifier."""

    def __init__(self, in_dim: int, hidden: int = LSTM_HIDDEN,
                 num_layers: int = LSTM_LAYERS, dropout: float = LSTM_DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop       = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden * 2, 1)

    def forward(self, x: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Plain padded tensor — pack_padded_sequence is 15× slower on MPS
        out, _ = self.lstm(x)
        out = self.drop(out)
        return self.classifier(out).squeeze(-1)   # (B, L)

# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Binary focal loss (Lin et al. 2017, RetinaNet).

    FL(p) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    alpha: per-head class weight for positives, computed as N_neg/(N_pos+N_neg).
           High alpha when positives are rare — directly corrects class imbalance.
    gamma: focusing parameter. gamma=2 down-weights easy negatives by (1-p)^2,
           forcing the model to learn hard examples. gamma=0 reduces to weighted BCE.

    Why not BCEWithLogitsLoss(pos_weight)?
    pos_weight scales the loss for positives uniformly. Focal loss additionally
    down-weights easy negatives (high-confidence background predictions), which
    is critical for features like active sites (<1% of residues) where the model
    can trivially minimise loss by predicting 'not active site' everywhere.
    """

    def __init__(self, gamma: float = FOCAL_GAMMA, alpha: float = 0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * (1.0 - p_t) ** self.gamma * bce
        return loss.mean()


# ---------------------------------------------------------------------------
# Clustered train/val/test split (MMseqs2 at CLUSTER_ID sequence identity)
# ---------------------------------------------------------------------------

def _write_fasta(seqs: list[str], path: str) -> None:
    with open(path, "w") as f:
        for i, s in enumerate(seqs):
            f.write(f">seq{i}\n{s}\n")


def cluster_split(
    seqs: list[str],
    rng_seed: int = RNG_SEED,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split protein indices into train/val/test ensuring no cluster bridges splits.

    Uses MMseqs2 easy-cluster at CLUSTER_ID (30%) sequence identity so no
    protein in train is >30% identical to any protein in test.  Without this,
    models memorise training proteins and benchmark metrics are inflated.

    Falls back to a random split (with a loud warning) if MMseqs2 is not on PATH.
    The fallback is provided only for development convenience — do NOT use it
    for final model training.
    """
    n = len(seqs)
    rng = np.random.default_rng(rng_seed)

    try:
        subprocess.run(["mmseqs", "version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print(
            "\n  *** WARNING: MMseqs2 not found — falling back to RANDOM split. ***\n"
            "  *** This is ONLY acceptable for development/debugging.          ***\n"
            "  *** Install MMseqs2 before final training to prevent data leak. ***\n",
            flush=True,
        )
        perm = rng.permutation(n)
        n_tr = int(train_frac * n)
        n_va = int(val_frac * n)
        return perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]

    with tempfile.TemporaryDirectory(prefix="beer_mmseqs_") as tmpdir:
        fasta_path = f"{tmpdir}/seqs.fasta"
        db_path    = f"{tmpdir}/seqdb"
        clust_path = f"{tmpdir}/clusters"
        tsv_path   = f"{tmpdir}/clusters_cluster.tsv"

        _write_fasta(seqs, fasta_path)

        print(f"  Running MMseqs2 clustering at {CLUSTER_ID*100:.0f}% identity …",
              flush=True)
        subprocess.run(
            ["mmseqs", "easy-cluster", fasta_path, clust_path, tmpdir,
             "--min-seq-id", str(CLUSTER_ID),
             "--cov-mode", "0",           # bidirectional coverage
             "-c",         "0.8",         # 80% coverage
             "--cluster-mode", "0",       # greedy set cover
             "-v", "0"],                  # suppress mmseqs stdout
            check=True, capture_output=True,
        )

        # Parse cluster representative → member TSV
        rep_to_members: dict[str, list[int]] = {}
        with open(tsv_path) as f:
            for line in f:
                rep, member = line.strip().split("\t")
                idx = int(member.replace("seq", ""))
                rep_to_members.setdefault(rep, []).append(idx)

    clusters = list(rep_to_members.values())
    rng.shuffle(clusters)
    print(f"  {len(clusters)} clusters from {n} proteins.", flush=True)

    # Greedily assign clusters to splits respecting size fractions
    train_ids, val_ids, test_ids = [], [], []
    n_train_target = int(train_frac * n)
    n_val_target   = int(val_frac * n)

    for cluster in clusters:
        if len(train_ids) < n_train_target:
            train_ids.extend(cluster)
        elif len(val_ids) < n_val_target:
            val_ids.extend(cluster)
        else:
            test_ids.extend(cluster)

    # Edge case: ensure no split is empty
    if not val_ids:
        val_ids   = train_ids[-max(1, len(train_ids)//10):]
        train_ids = train_ids[:-len(val_ids)]
    if not test_ids:
        test_ids  = val_ids[-max(1, len(val_ids)//2):]
        val_ids   = val_ids[:-len(test_ids)]

    return (np.array(train_ids), np.array(val_ids), np.array(test_ids))


# ---------------------------------------------------------------------------
# Metrics (all pure numpy — no sklearn dependency)
# ---------------------------------------------------------------------------

def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(-y_score)
    yt    = y_true[order]
    n_pos = yt.sum();  n_neg = len(yt) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tp  = np.cumsum(yt)
    fp  = np.cumsum(1 - yt)
    tpr = np.concatenate([[0], tp / n_pos])
    fpr = np.concatenate([[0], fp / n_neg])
    return float(np.trapz(tpr, fpr))


def roc_curve_np(y_true: np.ndarray, y_score: np.ndarray, n_points: int = 500):
    """Full ROC curve downsampled to n_points for storage."""
    order = np.argsort(-y_score)
    yt    = y_true[order]
    n_pos = yt.sum();  n_neg = len(yt) - n_pos
    if n_pos == 0 or n_neg == 0:
        return [0., 1.], [0., 1.], float("nan")
    tp  = np.cumsum(yt)
    fp  = np.cumsum(1 - yt)
    tpr = np.concatenate([[0.], tp / n_pos, [1.]])
    fpr = np.concatenate([[0.], fp / n_neg, [1.]])
    auc = float(np.trapz(tpr, fpr))
    # Downsample to n_points evenly spaced in FPR
    idx = np.unique(np.linspace(0, len(fpr) - 1, n_points).astype(int))
    return fpr[idx].tolist(), tpr[idx].tolist(), auc


def pr_curve_np(y_true: np.ndarray, y_score: np.ndarray, n_points: int = 500):
    """Precision-Recall curve + AUPRC (area under PR curve)."""
    order = np.argsort(-y_score)
    yt    = y_true[order]
    n_pos = yt.sum()
    if n_pos == 0:
        return [1., 0.], [0., 1.], float("nan")
    tp   = np.cumsum(yt)
    fp   = np.cumsum(1 - yt)
    prec = tp / (tp + fp)
    rec  = tp / n_pos
    # prepend perfect-precision point
    prec = np.concatenate([[1.], prec])
    rec  = np.concatenate([[0.], rec])
    auprc = float(np.trapz(prec, rec))
    idx   = np.unique(np.linspace(0, len(rec) - 1, n_points).astype(int))
    return prec[idx].tolist(), rec[idx].tolist(), auprc


def f1_max(y_true: np.ndarray, y_score: np.ndarray):
    """Maximum F1 score and the threshold at which it occurs."""
    order  = np.argsort(-y_score)
    thrs   = y_score[order]
    yt     = y_true[order]
    tp     = np.cumsum(yt)
    fp     = np.cumsum(1 - yt)
    fn     = yt.sum() - tp
    prec   = tp / np.maximum(tp + fp, 1)
    rec    = tp / np.maximum(tp + fn, 1)
    f1     = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    best   = int(np.argmax(f1))
    return float(f1[best]), float(thrs[best]), float(prec[best]), float(rec[best])


def calibration_curve_np(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15):
    """Calibration (reliability diagram) data."""
    bins   = np.linspace(0., 1., n_bins + 1)
    mean_p, frac_p, counts = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        counts.append(int(mask.sum()))
        if mask.sum() > 0:
            mean_p.append(float(y_prob[mask].mean()))
            frac_p.append(float(y_true[mask].mean()))
        else:
            mean_p.append(float((lo + hi) / 2))
            frac_p.append(float("nan"))
    return mean_p, frac_p, counts


def bootstrap_auroc_ci(y_true: np.ndarray, y_score: np.ndarray,
                       n_boot: int = 1000, ci: float = 0.95):
    """Residue-level bootstrap confidence interval for AUROC."""
    rng  = np.random.default_rng(RNG_SEED)
    n    = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt  = y_true[idx];  ys = y_score[idx]
        if yt.sum() == 0 or yt.sum() == n:
            continue
        aucs.append(auroc(yt, ys))
    aucs  = np.array(aucs)
    alpha = (1 - ci) / 2
    return float(np.percentile(aucs, 100 * alpha)), float(np.percentile(aucs, 100 * (1 - alpha)))


def per_protein_aurocs(seqs, labels, idxs, model, cache_dir) -> list[float]:
    """AUROC computed separately for each test protein."""
    model.eval()
    aucs = []
    with torch.no_grad():
        for i in idxs:
            lab = labels[i].astype(np.float32)
            if lab.sum() == 0 or lab.sum() == len(lab):
                continue
            emb = np.load(_seq_cache_path(cache_dir, seqs[i]))
            emb = torch.from_numpy(emb.astype(np.float32)).unsqueeze(0).to(DEVICE)
            logit  = model(emb)
            prob   = torch.sigmoid(logit).squeeze(0).cpu().numpy()
            aucs.append(auroc(lab, prob))
    return aucs

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def evaluate(model: BiLSTMHead, loader: DataLoader) -> float:
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for embs, labs, lengths in loader:
            embs    = embs.to(DEVICE)
            logits  = model(embs, lengths)             # (B, L_max)
            probs   = torch.sigmoid(logits).cpu()
            for b in range(len(lengths)):
                L = lengths[b].item()
                all_probs.append(probs[b, :L].numpy())
                all_labels.append(labs[b, :L].numpy())
    return auroc(np.concatenate(all_labels), np.concatenate(all_probs))


def train_model(seqs: list, labels: list,
                train_idx: np.ndarray, val_idx: np.ndarray,
                in_dim: int, alpha: float,
                cache_dir: pathlib.Path,
                tag: str = "") -> tuple[BiLSTMHead, list[dict]]:
    """Train a BiLSTM head with focal loss.

    alpha: positive-class weight in [0,1] = N_neg / (N_pos + N_neg).
           Passed directly to FocalLoss — high when positives are rare.
    """
    model = BiLSTMHead(in_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    def lr_lambda(ep):
        if ep < WARMUP_EPOCHS:
            return (ep + 1) / WARMUP_EPOCHS
        progress = (ep - WARMUP_EPOCHS) / max(EPOCHS - WARMUP_EPOCHS, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = FocalLoss(gamma=FOCAL_GAMMA, alpha=alpha)

    train_ds = ProteinDataset(seqs, labels, train_idx, cache_dir)
    val_ds   = ProteinDataset(seqs, labels, val_idx,   cache_dir)
    train_loader = DataLoader(train_ds, batch_size=BATCH_PROTEINS,
                              shuffle=True,  collate_fn=collate_fn,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_PROTEINS,
                              shuffle=False, collate_fn=collate_fn,
                              num_workers=0, pin_memory=False)

    best_val_auc   = 0.0
    best_state     = None
    patience_count = 0
    history: list[dict] = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, n_res = 0.0, 0
        t0 = time.time()

        for embs, labs, lengths in train_loader:
            embs = embs.to(DEVICE)
            labs = labs.to(DEVICE)
            optimizer.zero_grad()
            logits = model(embs, lengths)

            mask = torch.zeros_like(logits, dtype=torch.bool)
            for b, L in enumerate(lengths):
                mask[b, :L] = True

            loss = criterion(logits[mask], labs[mask])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * mask.sum().item()
            n_res += mask.sum().item()
            time.sleep(0.05)  # thermal throttle — ~10% duty-cycle relief

        val_auc  = evaluate(model, val_loader)
        avg_loss = total_loss / max(n_res, 1)
        cur_lr   = optimizer.param_groups[0]["lr"]
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": round(avg_loss, 6),
                         "val_auroc": round(val_auc, 6), "lr": cur_lr})
        elapsed = time.time() - t0
        print(f"  {tag}Epoch {epoch:3d}/{EPOCHS}  "
              f"loss={avg_loss:.4f}  val_AUROC={val_auc:.4f}  "
              f"lr={cur_lr:.2e}  t={elapsed:.0f}s", flush=True)

        if val_auc > best_val_auc + 1e-4:
            best_val_auc   = val_auc
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"  {tag}Early stop at epoch {epoch} "
                      f"(best val AUROC={best_val_auc:.4f})", flush=True)
                break

    model.load_state_dict(best_state)
    print(f"  {tag}Best val AUROC = {best_val_auc:.4f}", flush=True)
    return model, history

# ---------------------------------------------------------------------------
# Export BiLSTM weights to numpy for BEER inference
# ---------------------------------------------------------------------------

def export_to_numpy(model: BiLSTMHead, model_name: str, task_name: str,
                    task_desc: str, auc: float, save_path: pathlib.Path):
    """Save all BiLSTM + classifier weights as numpy arrays in .npz."""
    sd = {k: v.cpu().numpy() for k, v in model.state_dict().items()}

    np.savez_compressed(
        save_path,
        architecture  = np.array("bilstm2", dtype=object),
        # LSTM weights — PyTorch stores gates as [i, f, g, o] concatenated
        **{k: v for k, v in sd.items()},
        # Convenience aliases for the classifier
        W_out         = sd["classifier.weight"],   # (1, 2*hidden)
        b_out         = sd["classifier.bias"],     # (1,)
        lstm_hidden   = np.array(LSTM_HIDDEN),
        lstm_layers   = np.array(LSTM_LAYERS),
        model_name    = np.array(model_name,  dtype=object),
        trained_on    = np.array("UniProt_SwissProt", dtype=object),
        task          = np.array(task_name,   dtype=object),
        description   = np.array(task_desc,   dtype=object),
        auc           = np.array(auc),
        # backward-compat keys so existing BEER code doesn't break
        coef          = sd["classifier.weight"],
        intercept     = sd["classifier.bias"],
    )
    print(f"  Saved → {save_path}  (AUROC={auc:.4f})", flush=True)

# ---------------------------------------------------------------------------
# CRF + TM-head models and training  (architecture = "bilstm2_crf")
# ---------------------------------------------------------------------------

N_STATES   = 3
OUTSIDE    = 0
TM_HELIX   = 1
INSIDE     = 2
PDBTM_URL  = "https://pdbtm.enzim.hu/data/pdbtm_all.xml.gz"


class LinearCRF(nn.Module):
    def __init__(self, n_states: int = N_STATES):
        super().__init__()
        self.n_states    = n_states
        self.transitions = nn.Parameter(torch.zeros(n_states, n_states))
        with torch.no_grad():
            self.transitions[OUTSIDE, INSIDE]  = -5.0
            self.transitions[INSIDE,  OUTSIDE] = -5.0
            self.transitions[OUTSIDE,  TM_HELIX] = 1.0
            self.transitions[INSIDE,   TM_HELIX] = 1.0
            self.transitions[TM_HELIX, OUTSIDE]  = 1.0
            self.transitions[TM_HELIX, INSIDE]   = 1.0

    def _score_sentence(self, emissions, tags, mask):
        B, L, S = emissions.shape
        score = emissions[:, 0, :].gather(1, tags[:, 0:1]).squeeze(1) * mask[:, 0]
        for t in range(1, L):
            m_t = mask[:, t]
            score += (self.transitions[tags[:, t-1], tags[:, t]] +
                      emissions[:, t, :].gather(1, tags[:, t:t+1]).squeeze(1)) * m_t
        return score

    def _forward_alg(self, emissions, mask):
        B, L, S = emissions.shape
        alpha = emissions[:, 0, :]
        for t in range(1, L):
            m_t    = mask[:, t].unsqueeze(1)
            scores = alpha.unsqueeze(2) + self.transitions.unsqueeze(0) + \
                     emissions[:, t, :].unsqueeze(1)
            new_alpha = torch.logsumexp(scores, dim=1)
            alpha = torch.where(m_t.bool(), new_alpha, alpha)
        return torch.logsumexp(alpha, dim=1)

    def neg_log_likelihood(self, emissions, tags, mask):
        return (self._forward_alg(emissions, mask) -
                self._score_sentence(emissions, tags, mask)).mean()

    def viterbi_decode(self, emissions, mask):
        B, L, S = emissions.shape
        viterbi  = emissions[:, 0, :]
        backptr  = []
        for t in range(1, L):
            scores = viterbi.unsqueeze(2) + self.transitions.unsqueeze(0)
            best_scores, best_tags = scores.max(dim=1)
            backptr.append(best_tags)
            m_t     = mask[:, t].unsqueeze(1).bool()
            viterbi = torch.where(m_t, best_scores + emissions[:, t, :], viterbi)
        best_last = viterbi.argmax(dim=1)
        all_seqs  = []
        for b in range(B):
            seq = [best_last[b].item()]
            for ptr in reversed(backptr):
                seq.append(ptr[b, seq[-1]].item())
            seq.reverse()
            all_seqs.append(seq[:int(mask[b].sum().item())])
        return all_seqs


class BiLSTMCRF(nn.Module):
    def __init__(self, in_dim, hidden=LSTM_HIDDEN, num_layers=LSTM_LAYERS,
                 dropout=LSTM_DROPOUT, n_states=N_STATES):
        super().__init__()
        self.lstm     = nn.LSTM(in_dim, hidden, num_layers, bidirectional=True,
                                batch_first=True,
                                dropout=dropout if num_layers > 1 else 0.0)
        self.drop     = nn.Dropout(dropout)
        self.emission = nn.Linear(hidden * 2, n_states)
        self.crf      = LinearCRF(n_states)

    def forward(self, x, tags=None, mask=None):
        out, _ = self.lstm(x)
        emits  = self.emission(self.drop(out))
        if tags is not None:
            return self.crf.neg_log_likelihood(emits, tags, mask)
        return self.crf.viterbi_decode(emits, mask)

    def predict_probs(self, x, mask):
        out, _ = self.lstm(x)
        emits  = self.emission(self.drop(out))
        return torch.softmax(emits, dim=-1)


class TMDataset(Dataset):
    def __init__(self, seqs, labels, idxs, cache_dir):
        self.seqs = seqs; self.labels = labels
        self.idxs = idxs; self.cache_dir = cache_dir

    def __len__(self): return len(self.idxs)

    def __getitem__(self, item):
        i   = self.idxs[item]
        emb = np.load(_seq_cache_path(self.cache_dir, self.seqs[i]))
        return (torch.from_numpy(emb.astype(np.float32)),
                torch.from_numpy(np.array(self.labels[i], dtype=np.int64)))


def tm_collate(batch):
    embs, labs = zip(*batch)
    lengths = torch.tensor([e.shape[0] for e in embs], dtype=torch.long)
    return (pad_sequence(embs, batch_first=True),
            pad_sequence(labs, batch_first=True, padding_value=-1),
            lengths)


def _count_tm_runs(labels):
    in_helix = False; count = 0
    for v in labels:
        if v == TM_HELIX:
            if not in_helix: count += 1; in_helix = True
        else: in_helix = False
    return count


def fetch_pdbtm(cache_path: pathlib.Path) -> list[dict]:
    import gzip as _gz, xml.etree.ElementTree as ET
    if cache_path.exists():
        with open(cache_path) as f: return json.load(f)
    xml_cache = cache_path.with_suffix(".xml.gz")
    if not xml_cache.exists():
        print("  Downloading PDBTM …", flush=True)
        import urllib.request as _ur
        _ur.urlretrieve(PDBTM_URL, xml_cache)
    print("  Parsing PDBTM XML …", flush=True)
    opener = _gz.open if str(xml_cache).endswith(".gz") else open
    with opener(xml_cache, "rb") as fh:
        tree = ET.parse(fh)
    ns = {"p": "http://pdbtm.enzim.hu"}
    proteins = []; skipped = 0
    for entry in tree.getroot().findall(".//p:PDBTM", ns):
        for chain in entry.findall(".//p:CHAIN", ns):
            seq_el = chain.find("p:SEQ", ns)
            if seq_el is None or not seq_el.text: skipped += 1; continue
            seq = seq_el.text.replace("\n", "").strip()[:1024]
            if len(seq) < 30: skipped += 1; continue
            L   = len(seq)
            lab = np.zeros(L, dtype=np.int8)
            for reg in chain.findall("p:REGION", ns):
                t = reg.get("type", "")
                try: rs = max(0, int(reg.get("seq_beg", 0)) - 1); re_ = min(L, int(reg.get("seq_end", 0)))
                except (ValueError, TypeError): continue
                if rs >= re_: continue
                if t == "H": lab[rs:re_] = TM_HELIX
                elif t in ("I", "L"): lab[rs:re_] = INSIDE
            if int((lab == TM_HELIX).sum()) < 10: skipped += 1; continue
            proteins.append({"seq": seq, "labels": lab.tolist()})
    print(f"  Parsed {len(proteins)} PDBTM chains ({skipped} skipped).", flush=True)
    with open(cache_path, "w") as f: json.dump(proteins, f)
    return proteins


def evaluate_tm(model, loader):
    model.eval()
    all_probs, all_binary, topo_correct, topo_total = [], [], 0, 0
    with torch.no_grad():
        for embs, labs, lengths in loader:
            embs = embs.to(DEVICE)
            mask = torch.zeros(embs.shape[:2], dtype=torch.bool, device=DEVICE)
            for b, L in enumerate(lengths): mask[b, :L] = True
            preds = model(embs, mask=mask)
            probs = model.predict_probs(embs, mask).cpu().numpy()
            for b, L in enumerate(lengths):
                L = int(L.item()); lab = labs[b, :L].numpy(); valid = lab != -1
                all_probs.append(probs[b, :L, TM_HELIX][valid])
                all_binary.append((lab[valid] == TM_HELIX).astype(np.float32))
                if _count_tm_runs(lab[:L]) == _count_tm_runs(np.array(preds[b])):
                    topo_correct += 1
                topo_total += 1
    auc_ = auroc(np.concatenate(all_binary), np.concatenate(all_probs))
    return auc_, topo_correct / max(topo_total, 1)


def train_tm(seqs, labels, train_idx, val_idx, in_dim, cache_dir, tag=""):
    model     = BiLSTMCRF(in_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    def lr_lambda(ep):
        if ep < WARMUP_EPOCHS: return (ep + 1) / WARMUP_EPOCHS
        return 0.5 * (1.0 + math.cos(math.pi * (ep - WARMUP_EPOCHS) / max(EPOCHS - WARMUP_EPOCHS, 1)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    train_loader = DataLoader(TMDataset(seqs, labels, train_idx, cache_dir),
                              BATCH_PROTEINS, shuffle=True,  collate_fn=tm_collate, num_workers=0)
    val_loader   = DataLoader(TMDataset(seqs, labels, val_idx,   cache_dir),
                              BATCH_PROTEINS, shuffle=False, collate_fn=tm_collate, num_workers=0)
    best_auc, best_state, patience_count, history = 0.0, None, 0, []
    for epoch in range(1, EPOCHS + 1):
        model.train(); total_loss = 0.0; n_batch = 0; t0 = time.time()
        for embs, labs, lengths in train_loader:
            embs = embs.to(DEVICE); labs = labs.to(DEVICE)
            mask = torch.zeros(embs.shape[:2], dtype=torch.bool, device=DEVICE)
            for b, L in enumerate(lengths): mask[b, :L] = True
            labs_m = labs.clone(); labs_m[~mask] = 0
            optimizer.zero_grad()
            loss = model(embs, tags=labs_m, mask=mask)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); total_loss += loss.item(); n_batch += 1; time.sleep(0.05)
        val_auc, val_topo = evaluate_tm(model, val_loader)
        avg_loss = total_loss / max(n_batch, 1); cur_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": round(avg_loss, 6),
                        "val_auroc": round(val_auc, 6), "val_topo_acc": round(val_topo, 4)})
        print(f"  {tag}Epoch {epoch:3d}/{EPOCHS}  loss={avg_loss:.4f}  "
              f"val_AUROC={val_auc:.4f}  topo={val_topo:.3f}  "
              f"lr={cur_lr:.2e}  t={time.time()-t0:.0f}s", flush=True)
        if val_auc > best_auc + 1e-4:
            best_auc = val_auc; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"  {tag}Early stop at epoch {epoch} (best AUROC={best_auc:.4f})", flush=True); break
    model.load_state_dict(best_state)
    return model, history


KNOWN_MULTI_TM = {"P38264": 4, "P02699": 7, "P00533": 1, "P04637": 0}


def validate_known_tm(model, model_name):
    print("\n  Validating on known multi-TM proteins …", flush=True)
    cache_dir = _embed_cache_dir(model_name)
    results = {}
    for acc, expected in KNOWN_MULTI_TM.items():
        url = f"https://rest.uniprot.org/uniprotkb/{acc}.json?fields=sequence"
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                seq = json.loads(r.read())["sequence"]["value"][:1024]
        except Exception as e:
            print(f"    {acc}: could not fetch ({e})", flush=True)
            results[acc] = {"expected": expected, "predicted": None}; continue
        embed_all([seq], model_name)
        emb  = np.load(_seq_cache_path(cache_dir, seq))
        x    = torch.from_numpy(emb.astype(np.float32)).unsqueeze(0).to(DEVICE)
        mask = torch.ones(1, emb.shape[0], dtype=torch.bool, device=DEVICE)
        model.eval()
        with torch.no_grad(): pred_seq = model(x, mask=mask)[0]
        n_pred = _count_tm_runs(np.array(pred_seq))
        status = "✓" if n_pred == expected else "✗"
        print(f"    {status} {acc}: expected {expected} TM, predicted {n_pred}", flush=True)
        results[acc] = {"expected": expected, "predicted": n_pred, "correct": n_pred == expected}
    return results


def export_bilstm_crf(model, model_name, auc, save_path):
    sd = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
    np.savez_compressed(save_path,
        architecture = np.array("bilstm2_crf", dtype=object),
        **sd,
        W_out        = sd["emission.weight"],
        b_out        = sd["emission.bias"],
        transitions  = sd["crf.transitions"],
        n_states     = np.array(N_STATES),
        lstm_hidden  = np.array(LSTM_HIDDEN),
        lstm_layers  = np.array(LSTM_LAYERS),
        model_name   = np.array(model_name,       dtype=object),
        trained_on   = np.array("PDBTM",           dtype=object),
        task         = np.array("transmembrane",   dtype=object),
        description  = np.array("Transmembrane helix (BiLSTM+CRF topology-aware)", dtype=object),
        auc          = np.array(auc),
        coef         = sd["emission.weight"],
        intercept    = sd["emission.bias"],
    )
    print(f"  Saved CRF model → {save_path}  (AUROC={auc:.4f})", flush=True)


def _train_task_tm(task_name, cfg, model_name, use_cluster_split):
    """Full training pipeline for the transmembrane CRF head."""
    save_path = MODELS_DIR / f"{task_name}_head.npz"
    if save_path.exists():
        print(f"  SKIP: {save_path.name} already exists.", flush=True); return

    pdbtm_cache = CACHE_DIR / "pdbtm_parsed.json"
    proteins    = fetch_pdbtm(pdbtm_cache)
    seqs        = [p["seq"]    for p in proteins]
    labels      = [p["labels"] for p in proteins]
    print(f"  {len(seqs)} PDBTM chains loaded.", flush=True)

    if use_cluster_split:
        train_idx, val_idx, test_idx = cluster_split(seqs)
    else:
        rng = np.random.default_rng(RNG_SEED); n = len(seqs); p = rng.permutation(n)
        train_idx = p[:int(0.8*n)]; val_idx = p[int(0.8*n):int(0.9*n)]; test_idx = p[int(0.9*n):]

    cache_dir = _embed_cache_dir(model_name)
    embed_all(seqs, model_name)
    in_dim = np.load(_seq_cache_path(cache_dir, seqs[0])).shape[1]

    print("\n  Training BiLSTM-CRF …", flush=True)
    model, history = train_tm(seqs, labels, train_idx, val_idx, in_dim, cache_dir)
    test_auc, test_topo = evaluate_tm(model,
        DataLoader(TMDataset(seqs, labels, test_idx, cache_dir),
                   BATCH_PROTEINS, shuffle=False, collate_fn=tm_collate, num_workers=0))
    print(f"\n  Test AUROC={test_auc:.4f}  Topology acc={test_topo:.3f}", flush=True)

    print("\n  Retraining on train+val …", flush=True)
    tv_idx = np.concatenate([train_idx, val_idx])
    model_fin, history_retrain = train_tm(seqs, labels, tv_idx, test_idx,
                                          in_dim, cache_dir, tag="[retrain] ")
    test_loader = DataLoader(TMDataset(seqs, labels, test_idx, cache_dir),
                             BATCH_PROTEINS, shuffle=False, collate_fn=tm_collate, num_workers=0)
    final_auc, final_topo = evaluate_tm(model_fin, test_loader)
    print(f"  Final AUROC={final_auc:.4f}  Topology acc={final_topo:.3f}", flush=True)

    known = validate_known_tm(model_fin, model_name)
    results = {
        "task": task_name, "architecture": "bilstm2_crf",
        "split_method": "mmseqs2_cluster" if use_cluster_split else "random",
        "dataset": {"n_train": int(len(train_idx)), "n_val": int(len(val_idx)), "n_test": int(len(test_idx))},
        "test_metrics": {"auroc": round(final_auc, 5), "topology_acc": round(final_topo, 4)},
        "known_protein_validation": known,
        "training_history": history, "retrain_history": history_retrain,
    }
    with open(CACHE_DIR / f"{task_name}_results.json", "w") as f: json.dump(results, f, indent=2)
    export_bilstm_crf(model_fin, model_name, final_auc, save_path)


# ---------------------------------------------------------------------------
# Window-pooling aggregation head  (architecture = "bilstm2_window")
# ---------------------------------------------------------------------------

WINDOW_SIZE = 7
WALTZ_URL   = "https://waltz.switchlab.org/downloads/WALTZ_positive.fasta"


class AggregationHead(nn.Module):
    def __init__(self, in_dim, hidden=LSTM_HIDDEN, num_layers=LSTM_LAYERS,
                 dropout=LSTM_DROPOUT, window=WINDOW_SIZE):
        super().__init__()
        self.window = window
        self.lstm   = nn.LSTM(in_dim, hidden, num_layers, bidirectional=True,
                              batch_first=True,
                              dropout=dropout if num_layers > 1 else 0.0)
        self.drop       = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden * 2, 1)

    def _window_pool(self, x):
        B, L, D = x.shape
        x_t = x.permute(0, 2, 1)
        pad = self.window // 2
        out = F.avg_pool1d(F.pad(x_t, (pad, pad), mode="reflect"),
                           kernel_size=self.window, stride=1, padding=0)
        return out.permute(0, 2, 1)

    def forward(self, x, lengths=None):
        out, _ = self.lstm(x)
        return self.classifier(self._window_pool(self.drop(out))).squeeze(-1)


class AggDataset(Dataset):
    def __init__(self, seqs, labels, idxs, cache_dir):
        self.seqs = seqs; self.labels = labels
        self.idxs = idxs; self.cache_dir = cache_dir

    def __len__(self): return len(self.idxs)

    def __getitem__(self, item):
        i   = self.idxs[item]
        emb = np.load(_seq_cache_path(self.cache_dir, self.seqs[i]))
        return (torch.from_numpy(emb.astype(np.float32)),
                torch.from_numpy(np.array(self.labels[i], dtype=np.float32)))


def agg_collate(batch):
    embs, labs = zip(*batch)
    lengths = torch.tensor([e.shape[0] for e in embs], dtype=torch.long)
    return pad_sequence(embs, batch_first=True), pad_sequence(labs, batch_first=True), lengths


def fetch_waltz(cache_path: pathlib.Path) -> list[dict]:
    if cache_path.exists():
        with open(cache_path) as f: return json.load(f)
    print("  Fetching WALTZ amyloidogenic peptides …", flush=True)
    proteins = []
    try:
        with urllib.request.urlopen(WALTZ_URL, timeout=60) as r:
            content = r.read().decode()
        seq = ""
        for line in content.splitlines():
            if line.startswith(">"):
                if seq: proteins.append({"seq": seq, "labels": [1] * len(seq)})
                seq = ""
            else: seq += line.strip()
        if seq: proteins.append({"seq": seq, "labels": [1] * len(seq)})
    except Exception as e:
        print(f"  Warning: WALTZ unavailable ({e}). Using UniProt amyloid fallback.", flush=True)
        url = ("https://rest.uniprot.org/uniprotkb/search"
               "?query=reviewed:true+AND+keyword:KW-0043&format=json"
               "&fields=accession,sequence,ft_region&size=500")
        while url:
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    body = r.read(); headers = {k.lower(): v for k, v in r.headers.items()}
                data = json.loads(body)
                import re as _re
                for p in data.get("results", []):
                    seq = p.get("sequence", {}).get("value", "")[:1024]
                    if len(seq) < 20: continue
                    lab = np.zeros(len(seq), dtype=np.int8)
                    for feat in p.get("features", []):
                        if feat.get("type") != "Region": continue
                        if "amyloid" not in feat.get("description", "").lower(): continue
                        s = max(0, feat["location"]["start"].get("value", 0) - 1)
                        e = min(len(seq), feat["location"]["end"].get("value", 0))
                        if s < e: lab[s:e] = 1
                    if lab.sum() > 0: proteins.append({"seq": seq, "labels": lab.tolist()})
                m = _re.search(r'<([^>]+)>;\s*rel="next"', headers.get("link", ""))
                url = m.group(1) if m else ""
            except Exception: break
    with open(cache_path, "w") as f: json.dump(proteins, f)
    print(f"  {len(proteins)} WALTZ/amyloid peptides loaded.", flush=True)
    return proteins


def fetch_soluble_negatives(cache_path: pathlib.Path, n_max: int = 4000) -> list[dict]:
    if cache_path.exists():
        with open(cache_path) as f: return json.load(f)
    print("  Fetching soluble negatives from UniProt …", flush=True)
    import re as _re
    url = ("https://rest.uniprot.org/uniprotkb/search"
           "?query=reviewed:true+AND+annotation_score:[5+TO+*]"
           "+AND+NOT+(keyword:KW-0043)+AND+NOT+(keyword:KW-0472)"
           "+AND+ft_strand:*&format=json&fields=accession,sequence&size=500")
    proteins = []
    while url and len(proteins) < n_max:
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                body = r.read(); headers = {k.lower(): v for k, v in r.headers.items()}
            for p in json.loads(body).get("results", []):
                seq = p.get("sequence", {}).get("value", "")[:1024]
                if len(seq) >= 30: proteins.append({"seq": seq, "labels": [0] * len(seq)})
            m = _re.search(r'<([^>]+)>;\s*rel="next"', headers.get("link", ""))
            url = m.group(1) if m else ""; time.sleep(0.2)
        except Exception as e: print(f"  Warning: {e}", flush=True); break
    proteins = proteins[:n_max]
    with open(cache_path, "w") as f: json.dump(proteins, f)
    print(f"  {len(proteins)} soluble negatives loaded.", flush=True)
    return proteins


def evaluate_agg(model, loader) -> float:
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for embs, labs, lengths in loader:
            probs = torch.sigmoid(model(embs.to(DEVICE), lengths)).cpu()
            for b, L in enumerate(lengths):
                L = int(L.item())
                all_probs.append(probs[b, :L].numpy())
                all_labels.append(labs[b, :L].numpy())
    return auroc(np.concatenate(all_labels), np.concatenate(all_probs))


def train_agg(seqs, labels, train_idx, val_idx, in_dim, alpha, cache_dir, tag=""):
    model     = AggregationHead(in_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    def lr_lambda(ep):
        if ep < WARMUP_EPOCHS: return (ep + 1) / WARMUP_EPOCHS
        return 0.5 * (1.0 + math.cos(math.pi * (ep - WARMUP_EPOCHS) / max(EPOCHS - WARMUP_EPOCHS, 1)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = FocalLoss(gamma=FOCAL_GAMMA, alpha=alpha)
    train_loader = DataLoader(AggDataset(seqs, labels, train_idx, cache_dir),
                              BATCH_PROTEINS, shuffle=True,  collate_fn=agg_collate, num_workers=0)
    val_loader   = DataLoader(AggDataset(seqs, labels, val_idx,   cache_dir),
                              BATCH_PROTEINS, shuffle=False, collate_fn=agg_collate, num_workers=0)
    best_auc, best_state, patience_count, history = 0.0, None, 0, []
    for epoch in range(1, EPOCHS + 1):
        model.train(); total_loss = 0.0; n_res = 0; t0 = time.time()
        for embs, labs, lengths in train_loader:
            embs = embs.to(DEVICE); labs = labs.to(DEVICE)
            logits = model(embs, lengths)
            mask = torch.zeros_like(logits, dtype=torch.bool)
            for b, L in enumerate(lengths): mask[b, :L] = True
            optimizer.zero_grad()
            loss = criterion(logits[mask], labs[mask])
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); total_loss += loss.item() * mask.sum().item()
            n_res += mask.sum().item(); time.sleep(0.05)
        val_auc  = evaluate_agg(model, val_loader)
        avg_loss = total_loss / max(n_res, 1); cur_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": round(avg_loss, 6), "val_auroc": round(val_auc, 6)})
        print(f"  {tag}Epoch {epoch:3d}/{EPOCHS}  loss={avg_loss:.4f}  "
              f"val_AUROC={val_auc:.4f}  lr={cur_lr:.2e}  t={time.time()-t0:.0f}s", flush=True)
        if val_auc > best_auc + 1e-4:
            best_auc = val_auc; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"  {tag}Early stop (best AUROC={best_auc:.4f})", flush=True); break
    model.load_state_dict(best_state)
    return model, history


def export_aggregation(model, model_name, auc, save_path):
    sd = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
    np.savez_compressed(save_path,
        architecture = np.array("bilstm2_window", dtype=object),
        **sd,
        W_out        = sd["classifier.weight"],
        b_out        = sd["classifier.bias"],
        window_size  = np.array(WINDOW_SIZE),
        lstm_hidden  = np.array(LSTM_HIDDEN),
        lstm_layers  = np.array(LSTM_LAYERS),
        model_name   = np.array(model_name, dtype=object),
        trained_on   = np.array("WALTZ+soluble_negatives", dtype=object),
        task         = np.array("aggregation", dtype=object),
        description  = np.array("Aggregation propensity (BiLSTM+window-pool)", dtype=object),
        auc          = np.array(auc),
        coef         = sd["classifier.weight"],
        intercept    = sd["classifier.bias"],
    )
    print(f"  Saved aggregation model → {save_path}  (AUROC={auc:.4f})", flush=True)


def _train_task_agg(task_name, cfg, model_name, use_cluster_split):
    """Full training pipeline for the aggregation window head."""
    save_path = MODELS_DIR / f"{task_name}_head.npz"
    if save_path.exists():
        print(f"  SKIP: {save_path.name} already exists.", flush=True); return

    # Curated override (WALTZ + AmyLoad + AmyPro + PDB fibrils + soluble negatives)
    # takes priority over inline WALTZ-only fetch.
    override_name = cfg.get("cache_override")
    override_path = CACHE_DIR / override_name if override_name else None
    curated = load_curated_override(override_path) if override_path else None

    if curated:
        seqs, labels = curated
        print(f"  Using curated override: {override_name} "
              f"({len(seqs)} proteins)", flush=True)
    else:
        if override_path:
            print(f"  Override not found ({override_name}); "
                  "falling back to WALTZ + soluble negatives.", flush=True)
        waltz_proteins   = fetch_waltz(CACHE_DIR / "waltz_proteins.json")
        soluble_proteins = fetch_soluble_negatives(CACHE_DIR / "soluble_negatives.json")
        all_proteins     = waltz_proteins + soluble_proteins
        seqs   = [p["seq"]    for p in all_proteins]
        labels = [np.array(p["labels"], dtype=np.float32) for p in all_proteins]

    n_pos = sum(sum(l) for l in labels); n_neg = sum(len(l) - sum(l) for l in labels)
    n_tot = n_pos + n_neg; alpha = float(n_neg / max(n_tot, 1))
    print(f"\n  {len(seqs)} proteins  pos_frac={n_pos/max(n_tot,1):.4f}  "
          f"focal_alpha={alpha:.4f}", flush=True)
    if len(seqs) < 100:
        print("  ABORT: too few proteins.", flush=True); return

    if use_cluster_split:
        train_idx, val_idx, test_idx = cluster_split(seqs)
    else:
        rng = np.random.default_rng(RNG_SEED); n = len(seqs); p = rng.permutation(n)
        train_idx = p[:int(0.8*n)]; val_idx = p[int(0.8*n):int(0.9*n)]; test_idx = p[int(0.9*n):]

    cache_dir = _embed_cache_dir(model_name)
    embed_all(seqs, model_name)
    in_dim = np.load(_seq_cache_path(cache_dir, seqs[0])).shape[1]

    print("\n  Training AggregationHead (BiLSTM+window-pool) …", flush=True)
    model, history = train_agg(seqs, labels, train_idx, val_idx, in_dim, alpha, cache_dir)
    test_loader = DataLoader(AggDataset(seqs, labels, test_idx, cache_dir),
                             BATCH_PROTEINS, shuffle=False, collate_fn=agg_collate, num_workers=0)
    test_auc = evaluate_agg(model, test_loader)
    print(f"\n  Test AUROC = {test_auc:.4f}", flush=True)

    print("\n  Retraining on train+val …", flush=True)
    tv_idx = np.concatenate([train_idx, val_idx])
    tv_pos = sum(sum(labels[i]) for i in tv_idx); tv_tot = sum(len(labels[i]) for i in tv_idx)
    model_fin, history_retrain = train_agg(seqs, labels, tv_idx, test_idx,
                                           in_dim, float(tv_tot - tv_pos) / max(tv_tot, 1),
                                           cache_dir, tag="[retrain] ")
    final_auc = evaluate_agg(model_fin, test_loader)
    print(f"  Final AUROC = {final_auc:.4f}", flush=True)

    fpr, tpr, _ = roc_curve_np(np.concatenate([np.array(labels[i], dtype=np.float32) for i in test_idx]),
                                np.zeros(sum(len(labels[i]) for i in test_idx)))  # placeholder
    results = {
        "task": task_name, "architecture": "bilstm2_window", "window_size": WINDOW_SIZE,
        "split_method": "mmseqs2_cluster" if use_cluster_split else "random",
        "focal_alpha": round(alpha, 5), "focal_gamma": FOCAL_GAMMA,
        "dataset": {"n_waltz": len(waltz_proteins), "n_negatives": len(soluble_proteins),
                    "n_train": int(len(train_idx)), "n_val": int(len(val_idx)), "n_test": int(len(test_idx))},
        "test_metrics": {"auroc": round(final_auc, 5)},
        "training_history": history, "retrain_history": history_retrain,
    }
    with open(CACHE_DIR / f"{task_name}_results.json", "w") as f: json.dump(results, f, indent=2)
    export_aggregation(model_fin, model_name, final_auc, save_path)


# ---------------------------------------------------------------------------
# Per-task pipeline
# ---------------------------------------------------------------------------

def train_task(task_name: str, cfg: dict, model_name: str,
               use_cluster_split: bool = True):
    print(f"\n{'='*65}", flush=True)
    print(f"  TASK : {task_name}", flush=True)
    print(f"  DESC : {cfg['description']}", flush=True)
    print(f"  MODEL: {model_name}  on  {DEVICE}", flush=True)
    print(f"{'='*65}", flush=True)

    arch = cfg.get("architecture", "bilstm2")
    if arch == "bilstm2_crf":
        return _train_task_tm(task_name, cfg, model_name, use_cluster_split)
    if arch == "bilstm2_window":
        return _train_task_agg(task_name, cfg, model_name, use_cluster_split)

    save_path = MODELS_DIR / f"{task_name}_head.npz"

    if save_path.exists():
        print(f"  SKIP: {save_path.name} already exists — skipping.", flush=True)
        return

    # 1. Fetch and label data — curated override takes priority over UniProt
    override_name = cfg.get("cache_override")
    override_path = CACHE_DIR / override_name if override_name else None
    curated = load_curated_override(override_path) if override_path else None

    if curated:
        seqs, labels = curated
        print(f"  Using curated override: {override_name}", flush=True)
    else:
        if override_path:
            print(f"  No curated override found ({override_name}); "
                  f"falling back to UniProt.", flush=True)
        proteins = fetch_uniprot(task_name, cfg)
        seqs, labels = build_labels(proteins, cfg)

    if len(seqs) < 100:
        print(f"  SKIP: only {len(seqs)} usable proteins.", flush=True)
        return

    # 3. Clustered 80/10/10 split by sequence identity (MMseqs2)
    #    Prevents train/test leakage from homologous proteins.
    #    Falls back to random split with warning if MMseqs2 unavailable.
    n = len(seqs)
    if use_cluster_split:
        train_idx, val_idx, test_idx = cluster_split(seqs)
    else:
        rng  = np.random.default_rng(RNG_SEED)
        perm = rng.permutation(n)
        n_train = int(0.8 * n);  n_val = int(0.1 * n)
        train_idx = perm[:n_train]
        val_idx   = perm[n_train:n_train + n_val]
        test_idx  = perm[n_train + n_val:]
    print(f"  Split: {len(train_idx)} train / {len(val_idx)} val / "
          f"{len(test_idx)} test  proteins", flush=True)

    # 4. Embed (MPS-accelerated, shared cache) — embeddings stay on disk
    print(f"\n  Embedding {n} sequences …", flush=True)
    cache_dir = _embed_cache_dir(model_name)
    embed_all(seqs, model_name)
    in_dim = np.load(_seq_cache_path(cache_dir, seqs[0])).shape[1]
    print(f"  Embedding dim: {in_dim}", flush=True)

    # 5. Focal loss alpha — fraction of negatives in training set.
    #    High alpha when positives are rare (e.g. active sites <1%).
    n_pos = sum(labels[i].sum() for i in train_idx)
    n_neg = sum(len(labels[i]) - labels[i].sum() for i in train_idx)
    total = float(n_pos + n_neg)
    alpha = float(n_neg / total) if total > 0 else 0.75
    print(f"  Focal alpha = {alpha:.4f}  "
          f"(n_pos={int(n_pos)}, n_neg={int(n_neg)}, "
          f"pos_rate={n_pos/total:.4f})", flush=True)

    # 6. Train — loads embeddings lazily from disk per batch (no OOM)
    print(f"\n  Training BiLSTM "
          f"({in_dim} → BiLSTM({LSTM_HIDDEN}×{LSTM_LAYERS}) → 1) …", flush=True)
    model, history = train_model(seqs, labels, train_idx, val_idx,
                                 in_dim, alpha, cache_dir)

    # 7. Test evaluation
    test_ds     = ProteinDataset(seqs, labels, test_idx, cache_dir)
    test_loader = DataLoader(test_ds, batch_size=BATCH_PROTEINS,
                             shuffle=False, collate_fn=collate_fn,
                             num_workers=0, pin_memory=False)
    model.eval()
    all_probs_test, all_labels_test = [], []
    with torch.no_grad():
        for embs, labs, lengths in test_loader:
            logits = model(embs.to(DEVICE), lengths)
            probs  = torch.sigmoid(logits).cpu()
            for b, L in enumerate(lengths):
                all_probs_test.append(probs[b, :L].numpy())
                all_labels_test.append(labs[b, :L].numpy())
    probs_cat  = np.concatenate(all_probs_test)
    labels_cat = np.concatenate(all_labels_test)

    fpr_l, tpr_l, test_auc  = roc_curve_np(labels_cat, probs_cat)
    prec_l, rec_l, auprc    = pr_curve_np(labels_cat, probs_cat)
    f1_best, thr_best, p_best, r_best = f1_max(labels_cat, probs_cat)
    cal_mp, cal_fp, cal_cnt = calibration_curve_np(labels_cat, probs_cat)
    ci_lo, ci_hi            = bootstrap_auroc_ci(labels_cat, probs_cat)
    pp_aucs                 = per_protein_aurocs(seqs, labels, test_idx, model, cache_dir)

    print(f"\n  Test AUROC = {test_auc:.4f} [{ci_lo:.4f}–{ci_hi:.4f}]  "
          f"AUPRC = {auprc:.4f}  F1_max = {f1_best:.4f}  "
          f"({len(test_idx)} proteins)", flush=True)

    # 8. Retrain on train+val — recompute alpha over the larger set
    print("\n  Retraining on train+val for final model …", flush=True)
    tv_idx  = np.concatenate([train_idx, val_idx])
    tv_pos  = sum(labels[i].sum() for i in tv_idx)
    tv_neg  = sum(len(labels[i]) - labels[i].sum() for i in tv_idx)
    tv_tot  = float(tv_pos + tv_neg)
    alpha_tv = float(tv_neg / tv_tot) if tv_tot > 0 else alpha
    model_fin, history_retrain = train_model(
        seqs, labels, tv_idx, test_idx,
        in_dim, alpha_tv, cache_dir, tag="[retrain] ")

    # Final metrics on test with retrained model
    model_fin.eval()
    all_probs_fin = []
    with torch.no_grad():
        for embs, labs, lengths in DataLoader(test_ds, batch_size=BATCH_PROTEINS,
                                              shuffle=False, collate_fn=collate_fn,
                                              num_workers=0, pin_memory=False):
            logits = model_fin(embs.to(DEVICE), lengths)
            probs  = torch.sigmoid(logits).cpu()
            for b, L in enumerate(lengths):
                all_probs_fin.append(probs[b, :L].numpy())
    probs_fin      = np.concatenate(all_probs_fin)
    fpr_f, tpr_f, final_auc  = roc_curve_np(labels_cat, probs_fin)
    prec_f, rec_f, final_auprc = pr_curve_np(labels_cat, probs_fin)
    f1_f, thr_f, p_f, r_f     = f1_max(labels_cat, probs_fin)
    ci_lo_f, ci_hi_f           = bootstrap_auroc_ci(labels_cat, probs_fin)
    cal_mp_f, cal_fp_f, cal_cnt_f = calibration_curve_np(labels_cat, probs_fin)
    pp_aucs_fin = per_protein_aurocs(seqs, labels, test_idx, model_fin, cache_dir)

    print(f"  Final AUROC = {final_auc:.4f} [{ci_lo_f:.4f}–{ci_hi_f:.4f}]  "
          f"AUPRC = {final_auprc:.4f}  F1_max = {f1_f:.4f}", flush=True)

    # 9. Save results JSON (for paper figures)
    n_test_res = int(sum(len(labels[i]) for i in test_idx))
    pos_rate_test = float(labels_cat.mean())
    pos_rate_train = float(
        sum(labels[i].sum() for i in train_idx) /
        max(sum(len(labels[i]) for i in train_idx), 1))

    results = {
        "task":        task_name,
        "description": cfg["description"],
        "model_name":  model_name,
        "architecture": "bilstm2",
        "lstm_hidden":  LSTM_HIDDEN,
        "lstm_layers":  LSTM_LAYERS,
        "dataset": {
            "n_train_proteins": int(len(train_idx)),
            "n_val_proteins":   int(len(val_idx)),
            "n_test_proteins":  int(len(test_idx)),
            "n_test_residues":  n_test_res,
            "pos_rate_train":   round(pos_rate_train, 5),
            "pos_rate_test":    round(pos_rate_test, 5),
            "focal_alpha":      round(alpha, 5),
            "focal_gamma":      FOCAL_GAMMA,
            "split_method":     "mmseqs2_cluster" if use_cluster_split else "random",
        },
        "training_history": history,
        "retrain_history":  history_retrain,
        "test_metrics": {
            "auroc":        round(final_auc, 5),
            "auroc_ci_lo":  round(ci_lo_f, 5),
            "auroc_ci_hi":  round(ci_hi_f, 5),
            "auprc":        round(final_auprc, 5),
            "f1_max":       round(f1_f, 5),
            "threshold_at_f1max": round(thr_f, 5),
            "precision_at_f1max": round(p_f, 5),
            "recall_at_f1max":    round(r_f, 5),
        },
        "roc_curve":    {"fpr": fpr_f, "tpr": tpr_f},
        "pr_curve":     {"precision": prec_f, "recall": rec_f},
        "calibration":  {"mean_predicted": cal_mp_f,
                         "fraction_positive": cal_fp_f,
                         "counts": cal_cnt_f},
        "per_protein_aurocs": [round(a, 5) for a in pp_aucs_fin],
    }
    results_path = CACHE_DIR / f"{task_name}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results JSON → {results_path}", flush=True)

    # 10. Export model weights
    MODELS_DIR.mkdir(exist_ok=True)
    export_to_numpy(model_fin, model_name, task_name,
                    cfg["description"], final_auc, save_path)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--tasks", nargs="*", default=list(TASKS.keys()),
        choices=list(TASKS.keys()),
        help="Tasks to train (default: all). Examples:\n"
             "  --tasks transmembrane\n"
             "  --tasks aggregation\n"
             "  --tasks disorder active_site phosphorylation")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        choices=["esm2_t6_8M_UR50D", "esm2_t12_35M_UR50D",
                 "esm2_t30_150M_UR50D", "esm2_t33_650M_UR50D"],
        help="ESM2 model (default: 650M)")
    parser.add_argument(
        "--no-cluster", action="store_true",
        help="Use random split instead of MMseqs2 clustered split (dev only)")
    args = parser.parse_args()

    CACHE_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    use_cluster = not args.no_cluster
    print(f"\nBEER head training — {len(args.tasks)} task(s) on {DEVICE}", flush=True)
    print(f"Model  : {args.model}", flush=True)
    print(f"Split  : {'MMseqs2 clustered (30% identity)' if use_cluster else 'RANDOM (dev mode)'}", flush=True)
    print(f"Loss   : Focal Loss (gamma={FOCAL_GAMMA}, alpha=per-head)", flush=True)
    print(f"Tasks  : {args.tasks}\n", flush=True)

    results: dict[str, float] = {}
    for task_name in args.tasks:
        try:
            train_task(task_name, TASKS[task_name], args.model,
                       use_cluster_split=use_cluster)
            npz = MODELS_DIR / f"{task_name}_head.npz"
            if npz.exists():
                d = np.load(npz, allow_pickle=True)
                results[task_name] = float(d["auc"])
        except Exception as exc:
            import traceback
            print(f"\n  ERROR in {task_name}: {exc}", flush=True)
            traceback.print_exc()

    print(f"\n{'='*65}", flush=True)
    print("  SUMMARY", flush=True)
    print(f"{'='*65}", flush=True)
    for task, auc in results.items():
        status = "✓" if not math.isnan(auc) else "✗"
        print(f"  {status}  {task:<22}  AUROC = {auc:.4f}", flush=True)
    print(f"{'='*65}\n", flush=True)


if __name__ == "__main__":
    main()
