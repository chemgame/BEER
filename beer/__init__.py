"""BEER - Biophysical Evaluation Engine for Residues."""
from beer._version import __version__

try:
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        import esm  # noqa: F401
    ESM2_AVAILABLE = True
except Exception:  # ImportError, AttributeError, RuntimeError (NumPy ABI mismatch), etc.
    ESM2_AVAILABLE = False

__all__ = ["__version__", "ESM2_AVAILABLE"]
