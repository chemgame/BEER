"""ESM2 masked log-likelihood variant effect scoring.

For each position i in the sequence, computes the log-likelihood ratio
  LLR(mut) = log P(mut | context) - log P(wt | context)
using ESM2's masked language model head.  A negative LLR means the
mutation is predicted to be destabilising; positive means neutral/gain.

This is the same approach used in ESM1v (Meier et al. 2021) and the
ESM2 variant effect predictor.
"""
from __future__ import annotations
import numpy as np
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from beer.embeddings.base import SequenceEmbedder

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")


def compute_single_mutant_llr(
    seq: str,
    embedder: "SequenceEmbedder",
    positions: Optional[list[int]] = None,
) -> Optional[np.ndarray]:
    """Return an (L x 20) array of LLR values.

    Each row i contains LLR for mutating position i to each of the 20
    canonical amino acids (in AA_ORDER order).  WT diagonal is 0.

    Parameters
    ----------
    seq:        Wild-type sequence.
    embedder:   An ESM2Embedder instance (must have _model, _alphabet, _torch).
    positions:  Optional subset of 0-based positions to score.
                Defaults to all positions (can be slow for long sequences).

    Returns None if ESM2 is not available or scoring fails.
    """
    try:
        # Access internal ESM2 model (only ESM2Embedder has these attributes)
        embedder._load_model()
        model     = embedder._model
        alphabet  = embedder._alphabet
        bc        = embedder._batch_converter
        torch     = embedder._torch
        if model is None or alphabet is None:
            return None

        L = len(seq)
        if positions is None:
            positions = list(range(L))

        # Token indices for the 20 amino acids
        aa_tokens = [alphabet.get_idx(aa) for aa in AA_ORDER]

        # Encode WT sequence once
        _, _, tokens = bc([("wt", seq)])
        tokens = tokens.to(embedder._device)

        llr_matrix = np.zeros((L, 20), dtype=np.float32)

        with torch.no_grad():
            # Get WT log-probs at all positions in one forward pass
            # tokens shape: (1, L+2)
            wt_tokens = tokens.clone()
            logits = model(wt_tokens, repr_layers=[], return_contacts=False)["logits"]
            # logits: (1, L+2, vocab_size)
            log_probs = torch.log_softmax(logits[0], dim=-1).cpu().numpy()
            # position offset: +1 for BOS token
            for pos in positions:
                wt_aa  = seq[pos]
                wt_idx = AA_ORDER.index(wt_aa) if wt_aa in AA_ORDER else -1
                tok_pos = pos + 1  # offset for BOS
                wt_log_p = log_probs[tok_pos, alphabet.get_idx(wt_aa)] if wt_aa in AA_ORDER else 0.0
                for j, mut_aa in enumerate(AA_ORDER):
                    mut_log_p = log_probs[tok_pos, aa_tokens[j]]
                    llr_matrix[pos, j] = float(mut_log_p - wt_log_p)
                # WT → 0 by definition
                if wt_idx >= 0:
                    llr_matrix[pos, wt_idx] = 0.0

        return llr_matrix

    except Exception:
        return None


def mean_effect_per_position(llr_matrix: np.ndarray) -> np.ndarray:
    """Return per-position mean LLR across all non-WT substitutions."""
    # Exclude WT (where LLR=0) by averaging the 19 non-zero columns per row
    means = np.zeros(len(llr_matrix))
    for i in range(len(llr_matrix)):
        vals = llr_matrix[i]
        nonzero = vals[vals != 0.0]
        means[i] = float(np.mean(nonzero)) if len(nonzero) else 0.0
    return means
