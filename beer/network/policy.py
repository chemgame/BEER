"""Which BEER actions contact an outside server, and whether they may.

Every analysis and every AI prediction runs on this computer. A few optional features reach a
server: retrieving annotation for an accession, fetching structures, BLAST, structure prediction
with ESMFold2, and the one-time download of model weights. Two settings govern them.

``offline_mode`` blocks every outbound connection. Weights that are already cached keep working.
``confirm_uploads`` asks before any action that sends the sequence itself to a server, which is
the case for BLAST and for ESMFold2 prediction. Every other request sends only an accession, an
identifier or a sequence length.
"""
from __future__ import annotations

import os

from beer import config as _config

#: Actions that send the full sequence to a third party, with the server it goes to.
SEQUENCE_UPLOADS: dict[str, str] = {
    "NCBI BLAST": "the NCBI BLAST service (blast.ncbi.nlm.nih.gov)",
    "ESMFold2 structure prediction": "the Biohub Forge service (biohub.ai)",
}


class OfflineError(RuntimeError):
    """Raised when an action needs the network and offline mode is on."""


def offline() -> bool:
    """True when the user has switched on offline mode."""
    return bool(_config.get("offline_mode"))


def confirm_uploads() -> bool:
    """True when BEER should ask before sending a sequence to a server."""
    value = _config.get("confirm_uploads")
    return True if value is None else bool(value)


def apply_to_environment() -> None:
    """Tell the model libraries not to reach Hugging Face when offline mode is on."""
    if offline():
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
