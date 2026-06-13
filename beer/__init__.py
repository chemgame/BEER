"""BEER - Biophysical Evaluation Engine for Residues."""
from beer._version import __version__

try:
    from esm.models.esmc import ESMC as _ESMC  # noqa: F401
    ESMC_AVAILABLE = True
except (ImportError, AttributeError, RuntimeError, OSError):
    ESMC_AVAILABLE = False

__all__ = ["__version__", "ESMC_AVAILABLE"]
