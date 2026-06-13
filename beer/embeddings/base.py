"""Abstract base class for sequence embedders."""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class SequenceEmbedder(ABC):
    """Abstract base for per-residue sequence embedders."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the embedder backend is ready."""
        ...

    @abstractmethod
    def embed(self, seq: str) -> np.ndarray | None:
        """Return per-residue embeddings of shape (L, D), or None on failure."""
        ...

    def clear_cache(self) -> None:
        """Clear any embedding cache. Override in subclasses if needed."""
        pass
