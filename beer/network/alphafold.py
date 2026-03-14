"""BEER network/alphafold.py — AlphaFold structure fetch worker.

Downloads an AlphaFold predicted structure for a UniProt accession,
extracts per-residue pLDDT scores, and computes the Cα distance matrix.

AlphaFold EBI API:
    GET https://alphafold.ebi.ac.uk/api/prediction/{accession}
    → list of prediction entries, each with a ``pdbUrl`` field.

PDB parsing is delegated to ``beer.io.pdb``.
"""

import json
import urllib.request
from PyQt5.QtCore import QThread, pyqtSignal


class AlphaFoldWorker(QThread):
    """Fetch AlphaFold predicted structure and derived data for an accession.

    Signals
    -------
    finished(dict):
        Emitted on success with keys:
        - pdb_str (str)        Raw PDB text
        - plddt (list[float])  Per-residue pLDDT scores
        - dist_matrix (list)   n×n Cα distance matrix (as nested list or np.ndarray)
        - accession (str)      Echo of the queried accession
    error(str):
        Emitted on failure with a human-readable message.
    progress(str):
        Emitted with status updates during the download.
    """

    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, accession: str):
        super().__init__()
        self.accession = accession.strip().upper()

    def run(self):
        try:
            self.progress.emit(f"Querying AlphaFold for {self.accession}…")
            meta_url = (
                f"https://alphafold.ebi.ac.uk/api/prediction/{self.accession}"
            )
            req = urllib.request.Request(
                meta_url, headers={"Accept": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=30) as r:
                meta = json.loads(r.read().decode())

            if not meta:
                self.error.emit(
                    f"No AlphaFold prediction found for {self.accession}."
                )
                return

            pdb_url = meta[0]["pdbUrl"]
            self.progress.emit("Downloading PDB structure…")
            with urllib.request.urlopen(pdb_url, timeout=60) as r:
                pdb_str = r.read().decode()

            self.progress.emit("Extracting pLDDT and distance matrix…")
            from beer.io.pdb import extract_plddt_from_pdb, compute_ca_distance_matrix

            plddt = extract_plddt_from_pdb(pdb_str)
            dist_matrix = compute_ca_distance_matrix(pdb_str)

            self.finished.emit(
                {
                    "pdb_str": pdb_str,
                    "plddt": plddt,
                    "dist_matrix": dist_matrix,
                    "accession": self.accession,
                }
            )
        except Exception as exc:
            self.error.emit(f"AlphaFold fetch failed: {exc}")
