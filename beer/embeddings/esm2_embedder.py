"""ESM2-based per-residue sequence embedder with LRU cache."""
from __future__ import annotations
import hashlib
from collections import OrderedDict
from typing import Optional
import numpy as np
from beer.embeddings.base import SequenceEmbedder

_CACHE_SIZE = 32


class ESM2Embedder(SequenceEmbedder):
    """Load an ESM2 model and produce per-residue embeddings.

    Parameters
    ----------
    model_name:
        One of ``esm2_t6_8M_UR50D``, ``esm2_t12_35M_UR50D``,
        ``esm2_t30_150M_UR50D``, ``esm2_t33_650M_UR50D``.
    device:
        Torch device string (``'cpu'`` or ``'cuda'``).
    """

    def __init__(self, model_name: str = "esm2_t33_650M_UR50D", device: str = "cpu") -> None:
        self._model_name = model_name
        self._device = device
        self._model = None
        self._alphabet = None
        self._batch_converter = None
        self._torch = None
        self._loaded = False  # deferred: load on first embed() call
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load model weights on first use (avoids download at startup)."""
        if self._loaded:
            return
        self._loaded = True
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import esm  # type: ignore[import]
                import torch  # type: ignore[import]
            model, alphabet = esm.pretrained.load_model_and_alphabet(self._model_name)
            model = model.eval().to(self._device)
            self._model = model
            self._alphabet = alphabet
            self._batch_converter = alphabet.get_batch_converter()
            self._torch = torch
        except Exception:  # pylint: disable=broad-except
            self._model = None

    def _cache_key(self, seq: str) -> str:
        return hashlib.md5(f"{self._model_name}:{seq}".encode()).hexdigest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        self._load_model()
        return self._model is not None

    def embed(self, seq: str) -> Optional[np.ndarray]:
        self._load_model()
        if self._model is None:
            return None
        key = self._cache_key(seq)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        try:
            data = [("protein", seq)]
            _, _, tokens = self._batch_converter(data)
            tokens = tokens.to(self._device)
            with self._torch.no_grad():
                results = self._model(tokens, repr_layers=[self._model.num_layers], return_contacts=False)
            # shape: (1, L+2, D) — strip BOS/EOS tokens
            emb: np.ndarray = results["representations"][self._model.num_layers][0, 1:-1].cpu().numpy()
            if len(emb) != len(seq):
                return None
            if len(self._cache) >= _CACHE_SIZE:
                self._cache.popitem(last=False)
            self._cache[key] = emb
            return emb
        except Exception:  # pylint: disable=broad-except
            return None

    def clear_cache(self) -> None:
        self._cache.clear()

    @property
    def model_name(self) -> str:
        return self._model_name
