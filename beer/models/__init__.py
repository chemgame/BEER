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

def load_signal_peptide_head() -> dict | None:
    return _load_head("signal_peptide_head.npz")

def load_transmembrane_head() -> dict | None:
    return _load_head("transmembrane_head.npz")

def load_coiled_coil_head() -> dict | None:
    return _load_head("coiled_coil_head.npz")

def load_dna_binding_head() -> dict | None:
    return _load_head("dna_binding_head.npz")

def load_active_site_head() -> dict | None:
    return _load_head("active_site_head.npz")

def load_binding_site_head() -> dict | None:
    return _load_head("binding_site_head.npz")

def load_phosphorylation_head() -> dict | None:
    return _load_head("phosphorylation_head.npz")

def load_lcd_head() -> dict | None:
    return _load_head("lcd_head.npz")

def load_zinc_finger_head() -> dict | None:
    return _load_head("zinc_finger_head.npz")

def load_glycosylation_head() -> dict | None:
    return _load_head("glycosylation_head.npz")

def load_ubiquitination_head() -> dict | None:
    return _load_head("ubiquitination_head.npz")

def load_methylation_head() -> dict | None:
    return _load_head("methylation_head.npz")

def load_acetylation_head() -> dict | None:
    return _load_head("acetylation_head.npz")

def load_lipidation_head() -> dict | None:
    return _load_head("lipidation_head.npz")

def load_disulfide_head() -> dict | None:
    return _load_head("disulfide_head.npz")

def load_intramembrane_head() -> dict | None:
    return _load_head("intramembrane_head.npz")

def load_motif_head() -> dict | None:
    return _load_head("motif_head.npz")

def load_propeptide_head() -> dict | None:
    return _load_head("propeptide_head.npz")

def load_repeat_head() -> dict | None:
    return _load_head("repeat_head.npz")

def load_rna_binding_head() -> dict | None:
    return _load_head("rna_binding_head.npz")

def load_nucleotide_binding_head() -> dict | None:
    return _load_head("nucleotide_binding_head.npz")

def load_transit_peptide_head() -> dict | None:
    return _load_head("transit_peptide_head.npz")

def load_aggregation_head() -> dict | None:
    return _load_head("aggregation_head.npz")

