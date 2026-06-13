"""ESMC log-likelihood ratio variant effect scoring.

For each position i, computes the log-likelihood ratio
  LLR(mut) = log P(mut | WT context) - log P(wt | WT context)
using ESMC's sequence_logits from a single unmasked forward pass.
Negative LLR: mutation predicted destabilising; positive: neutral/gain.
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
    canonical amino acids (AA_ORDER).  WT diagonal is 0 by definition.

    Uses ESMC's sequence_logits from a single unmasked forward pass.
    Returns None if ESMC is unavailable or scoring fails.
    """
    try:
        embedder._load_model()
        model = embedder._model
        torch = embedder._torch
        if model is None or torch is None:
            return None

        L = len(seq)
        if positions is None:
            positions = list(range(L))

        # Map each AA to its ESMC token ID via the model's tokenizer
        tok = model.tokenizer
        aa_token_ids = [tok.convert_tokens_to_ids(aa) for aa in AA_ORDER]

        tokens = model._tokenize([seq])          # (1, L+2) — BOS + seq + EOS
        tokens = tokens.to(embedder._device)

        with torch.no_grad():
            output = model.forward(sequence_tokens=tokens)

        # sequence_logits: (1, L+2, vocab_size)
        log_probs = torch.log_softmax(
            output.sequence_logits[0].cpu(), dim=-1
        ).numpy()   # (L+2, vocab_size)

        llr_matrix = np.zeros((L, 20), dtype=np.float32)
        for pos in positions:
            wt_aa = seq[pos]
            if wt_aa not in AA_ORDER:
                continue                         # non-canonical: leave row zero
            wt_idx   = AA_ORDER.index(wt_aa)
            tok_pos  = pos + 1                   # +1 for BOS token
            wt_log_p = log_probs[tok_pos, aa_token_ids[wt_idx]]
            for j, mut_tok_id in enumerate(aa_token_ids):
                llr_matrix[pos, j] = float(log_probs[tok_pos, mut_tok_id] - wt_log_p)
            llr_matrix[pos, wt_idx] = 0.0        # WT → 0 by definition

        return llr_matrix

    except Exception:
        return None


def mean_effect_per_position(llr_matrix: np.ndarray) -> np.ndarray:
    """Return per-position mean LLR across all 20 substitutions."""
    return np.mean(llr_matrix, axis=1)
