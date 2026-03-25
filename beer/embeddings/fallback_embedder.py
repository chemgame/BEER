"""Fallback embedder used when ESM2 is not installed."""
from __future__ import annotations
import numpy as np
from beer.embeddings.base import SequenceEmbedder


class FallbackEmbedder(SequenceEmbedder):
    """No-op embedder returned when fair-esm is not installed."""

    def is_available(self) -> bool:
        return False

    def embed(self, seq: str) -> np.ndarray | None:  # noqa: ARG002
        return None
