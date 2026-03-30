"""Load pre-trained head weights from bundled .npz files."""
from __future__ import annotations
import pathlib
import numpy as np

_MODELS_DIR = pathlib.Path(__file__).parent


def _load_head(name: str) -> dict | None:
    """Load a bundled .npz head weight file, or return None if not found."""
    path = _MODELS_DIR / name
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return dict(data)


def load_disorder_head() -> dict | None:
    return _load_head("disorder_head.npz")


def load_aggregation_head() -> dict | None:
    return _load_head("aggregation_head.npz")


def load_signal_head() -> dict | None:
    return _load_head("signal_head.npz")


