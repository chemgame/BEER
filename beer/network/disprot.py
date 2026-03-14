"""BEER network/disprot.py — DisProt database query worker.

Queries the DisProt REST API to fetch disorder region annotations for a
given UniProt accession.

REST endpoint:
    GET https://disprot.org/api/{uniprot_id}

Representative JSON response (abbreviated):
    {
      "disprot_id": "DP00001",
      "acc": "P04637",
      "protein_name": "Cellular tumor antigen p53",
      "length": 393,
      "regions": [
        {
          "start": 1,
          "end": 42,
          "region_id": "DP00001r001",
          "type_id": "D002",
          "type_name": "disorder",
          "evidence": [...]
        },
        ...
      ]
    }
"""

import json
import urllib.request
import urllib.error
from PyQt5.QtCore import QThread, pyqtSignal

_DISPROT_API_BASE = "https://disprot.org/api"
_TIMEOUT_SECONDS = 30


class DisPRotWorker(QThread):
    """Query DisProt database for disorder annotations of a UniProt accession.

    REST API: GET https://disprot.org/api/{uniprot_id}

    Signals
    -------
    finished(dict):
        Emitted on success (including the case of a valid 404 / not found).
        Dict keys:
        - found (bool)              False if the protein is not in DisProt
        - disprot_id (str)
        - protein_name (str)
        - accession (str)           Echo of the queried accession
        - sequence_length (int)     Total protein length (0 if unknown)
        - regions (list)            List of dicts: {start, end, type}
        - n_disordered_aa (int)     Total disordered residues (may overlap)
        - fraction_disordered (float)  n_disordered_aa / sequence_length
    error(str):
        Emitted on network/parse errors with a human-readable message.
    """

    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, accession: str):
        """Initialise the worker.

        Parameters
        ----------
        accession:
            UniProt accession (e.g. ``"P04637"``).  Stripped and upper-cased.
        """
        super().__init__()
        self.accession = accession.strip().upper()

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Fetch DisProt disorder annotations for the stored accession.

        On success emits ``finished`` with a populated result dict.
        If the protein is not found in DisProt, emits ``finished`` with
        ``{"found": False}``.
        On network / parse errors emits ``error``.
        """
        if not self.accession:
            self.error.emit("DisProt query failed: no accession provided.")
            return

        url = f"{_DISPROT_API_BASE}/{self.accession}"

        try:
            req = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "BEER-biophysics/1.0 (scientific software)",
                },
            )
            with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as response:
                raw = response.read().decode("utf-8")

        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                # Protein not in DisProt — not an error, just not found
                self.finished.emit({"found": False, "accession": self.accession})
                return
            self.error.emit(
                f"DisProt HTTP error {exc.code} for '{self.accession}': {exc.reason}"
            )
            return
        except urllib.error.URLError as exc:
            self.error.emit(
                f"DisProt network error for '{self.accession}': {exc.reason}"
            )
            return
        except Exception as exc:
            self.error.emit(f"DisProt query failed for '{self.accession}': {exc}")
            return

        # Parse JSON
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            self.error.emit(
                f"DisProt returned invalid JSON for '{self.accession}': {exc}"
            )
            return

        if not data:
            self.finished.emit({"found": False, "accession": self.accession})
            return

        # Extract fields — DisProt API may use slightly different key names
        disprot_id = data.get("disprot_id", data.get("id", ""))
        protein_name = data.get("protein_name", data.get("name", ""))
        sequence_length = int(data.get("length", data.get("sequence_length", 0)))

        # Parse disorder regions
        raw_regions = data.get("regions", [])
        regions: list = []
        for reg in raw_regions:
            if not isinstance(reg, dict):
                continue
            start = _safe_int(reg.get("start", reg.get("Start", 0)))
            end = _safe_int(reg.get("end", reg.get("End", 0)))
            rtype = (
                reg.get("type_name", reg.get("type", reg.get("term", "disorder")))
            )
            if isinstance(rtype, dict):
                rtype = rtype.get("name", rtype.get("term", "disorder"))
            regions.append({"start": start, "end": end, "type": str(rtype)})

        # Count disordered residues using a coverage set to handle overlaps
        disordered_positions: set = set()
        for reg in regions:
            for pos in range(reg["start"], reg["end"] + 1):
                disordered_positions.add(pos)

        n_disordered_aa = len(disordered_positions)
        fraction_disordered = (
            n_disordered_aa / sequence_length if sequence_length > 0 else 0.0
        )

        result = {
            "found": True,
            "disprot_id": disprot_id,
            "protein_name": protein_name,
            "accession": self.accession,
            "sequence_length": sequence_length,
            "regions": regions,
            "n_disordered_aa": n_disordered_aa,
            "fraction_disordered": fraction_disordered,
        }

        self.finished.emit(result)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _safe_int(value, default: int = 0) -> int:
    """Convert *value* to int, returning *default* on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
