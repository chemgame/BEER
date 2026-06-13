"""ESMC 600M per-residue sequence embedder with LRU cache."""
from __future__ import annotations
import hashlib
import threading
from collections import OrderedDict
from typing import Optional
import numpy as np
from beer.embeddings.base import SequenceEmbedder

_CACHE_SIZE = 32


class ESMCEmbedder(SequenceEmbedder):
    """Load an ESMC model and produce per-residue embeddings (1152-dim for 600M).

    Thread-safe: a single instance may be shared across multiple QThread workers.
    The internal RLock guards model load and LRU cache. A separate inference lock
    serializes ``model.forward()`` across threads — a single PyTorch module is not
    safe for concurrent forward passes (it can corrupt internal state and return
    wrong results), so inference is run one call at a time. The fast cache lookup
    stays concurrent; only the (cache-missing) forward pass serializes.

    Parameters
    ----------
    model_name:
        ``'esmc_300m'`` or ``'esmc_600m'`` (default).
    device:
        Torch device string (``'cpu'`` or ``'cuda'``).
    """

    def __init__(self, model_name: str = "esmc_600m", device: str = "cpu") -> None:
        self._model_name = model_name
        self._device = device
        self._model = None
        self._torch = None
        self._loaded = False
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.RLock()
        self._infer_lock = threading.Lock()   # serializes model.forward() across threads

    def _load_model(self) -> None:
        with self._lock:
            if self._loaded:
                return
            try:
                import torch  # type: ignore[import]
                from esm.models.esmc import ESMC  # type: ignore[import]
                device = torch.device(self._device)
                model = ESMC.from_pretrained(self._model_name, device=device)
                model = model.eval()
                self._model = model
                self._torch = torch
            except Exception:  # pylint: disable=broad-except
                self._model = None
            finally:
                self._loaded = True  # set only after the attempt completes

    def _cache_key(self, seq: str) -> str:
        return hashlib.md5(f"{self._model_name}:{seq}".encode()).hexdigest()

    def is_available(self) -> bool:
        self._load_model()
        return self._model is not None

    def embed(self, seq: str) -> Optional[np.ndarray]:
        self._load_model()
        if self._model is None:
            return None
        key = self._cache_key(seq)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        # Serialize the forward pass: concurrent forward() on one torch module
        # across worker threads can corrupt state and yield wrong embeddings.
        with self._infer_lock:
            # Another thread may have computed this same sequence while we waited.
            with self._lock:
                if key in self._cache:
                    self._cache.move_to_end(key)
                    return self._cache[key]
            try:
                with self._torch.no_grad():
                    tokens = self._model._tokenize([seq])  # (1, L+2)
                    output = self._model.forward(sequence_tokens=tokens)
                # output.embeddings: (1, L+2, D) — strip BOS/EOS, cast to float32
                emb: np.ndarray = output.embeddings[0, 1:-1].cpu().float().numpy()
                if len(emb) != len(seq):
                    return None
            except Exception as _exc:  # pylint: disable=broad-except
                import logging as _log
                _log.getLogger("beer.embeddings").warning(
                    "ESMC embedding failed for sequence of length %d: %s",
                    len(seq), _exc, exc_info=True,
                )
                return None
            with self._lock:
                if key not in self._cache and len(self._cache) >= _CACHE_SIZE:
                    self._cache.popitem(last=False)
                self._cache[key] = emb
            return emb

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    def release(self) -> None:
        """Release the torch model and cache (called at app shutdown)."""
        with self._lock:
            self._model = None
            self._torch = None
            self._cache.clear()

    @property
    def model_name(self) -> str:
        return self._model_name
