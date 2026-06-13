"""Embedder factory and availability flag."""
from __future__ import annotations

try:
    from esm.models.esmc import ESMC as _ESMC  # noqa: F401
    ESMC_AVAILABLE = True
except Exception:  # ImportError, AttributeError, RuntimeError (NumPy ABI mismatch), etc.
    ESMC_AVAILABLE = False

from beer.embeddings.base import SequenceEmbedder
from beer.embeddings.fallback_embedder import FallbackEmbedder

if ESMC_AVAILABLE:
    from beer.embeddings.esmc_embedder import ESMCEmbedder


def get_embedder(model_name: str = "esmc_600m", device: str = "cpu") -> SequenceEmbedder:
    """Return an ESMCEmbedder if available, else a FallbackEmbedder."""
    if ESMC_AVAILABLE:
        return ESMCEmbedder(model_name=model_name, device=device)
    return FallbackEmbedder()


__all__ = ["ESMC_AVAILABLE", "SequenceEmbedder", "FallbackEmbedder", "get_embedder"]
