"""BEER - Biophysical Evaluation Engine for Residues."""
import logging as _logging
import os as _os

from beer._version import __version__

# ESM C logs two notices when it is imported without the optional CUDA kernels (Transformer
# Engine, xformers or flash-attn). They report differences of a few units in the last place
# after the final layer normalisation, they are identical for every installation without a
# graphics card, and nothing can be done about them on a machine that has none. Set
# BEER_LOG_ESM=1 to see them.
if not _os.environ.get("BEER_LOG_ESM"):
    _logging.getLogger("esm").setLevel(_logging.ERROR)

try:
    from esm.models.esmc import ESMC as _ESMC  # noqa: F401
    ESMC_AVAILABLE = True
except (ImportError, AttributeError, RuntimeError, OSError):
    ESMC_AVAILABLE = False

__all__ = ["__version__", "ESMC_AVAILABLE"]
