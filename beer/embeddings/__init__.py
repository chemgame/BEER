"""Embedder factory and availability flag."""
from __future__ import annotations
from beer._version import __version__  # noqa: F401

try:
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        import esm  # noqa: F401
    ESM2_AVAILABLE = True
except Exception:  # ImportError, AttributeError, RuntimeError (NumPy ABI mismatch), etc.
    ESM2_AVAILABLE = False

from beer.embeddings.base import SequenceEmbedder
from beer.embeddings.fallback_embedder import FallbackEmbedder

if ESM2_AVAILABLE:
    from beer.embeddings.esm2_embedder import ESM2Embedder


def get_embedder(model_name: str = "esm2_t33_650M_UR50D", device: str = "cpu") -> SequenceEmbedder:
    """Return an ESM2Embedder if available, else a FallbackEmbedder."""
    if ESM2_AVAILABLE:
        return ESM2Embedder(model_name=model_name, device=device)
    return FallbackEmbedder()


__all__ = ["ESM2_AVAILABLE", "SequenceEmbedder", "FallbackEmbedder", "get_embedder"]
