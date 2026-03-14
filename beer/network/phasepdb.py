"""BEER network/phasepdb.py — PhaSepDB LLPS database lookup worker.

Queries PhaSepDB (Phase Separation DataBase) for phase-separation data
associated with a UniProt accession.

PhaSepDB REST endpoint:
    GET https://phasepdb.org/api/protein/{uniprot_id}

Representative JSON response (abbreviated):
    {
      "uniprot_id": "P04637",
      "gene_name": "TP53",
      "protein_name": "Cellular tumor antigen p53",
      "category": "scaffold",
      "evidence_type": "in vivo",
      "organism": "Homo sapiens",
      "references": [
        {"pmid": "30971819", "title": "..."},
        ...
      ]
    }

If the protein is not in PhaSepDB a 404 is returned.
"""

import json
import urllib.request
import urllib.error
from PyQt5.QtCore import QThread, pyqtSignal

_PHASEPDB_BASE = "https://phasepdb.org/api/protein"
_TIMEOUT_SECONDS = 10


class PhaSepDBWorker(QThread):
    """Query PhaSepDB for phase separation data.

    Uses the PhaSepDB REST API.  If the queried accession is not in the
    database, emits ``finished({"found": False})`` rather than an error.

    Signals
    -------
    finished(dict):
        Emitted when the query completes (success or not-found).
        Keys on success:
        - found (bool)            True
        - source (str)            "PhaSepDB"
        - accession (str)         Echo of queried accession
        - gene_name (str)
        - protein_name (str)
        - category (str)          e.g. "scaffold", "client", "regulator"
        - evidence_type (str)     e.g. "in vitro", "in vivo", "prediction"
        - organism (str)
        - references (list)       List of dicts: {pmid, title}
        Keys when not found:
        - found (bool)            False
        - accession (str)
    error(str):
        Emitted on genuine network / parse failures.
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
        """Try PhaSepDB and emit results.

        Performs a single GET request to PhaSepDB.  Emits ``finished`` with
        ``found=True`` on success, ``found=False`` on HTTP 404 / empty
        response, or ``error`` on genuine failures (network errors, malformed
        JSON, etc.).
        """
        if not self.accession:
            self.error.emit("PhaSepDB query failed: no accession provided.")
            return

        url = f"{_PHASEPDB_BASE}/{self.accession}"

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
                # Protein not in PhaSepDB — this is a valid outcome
                self.finished.emit({"found": False, "accession": self.accession})
                return
            self.error.emit(
                f"PhaSepDB HTTP error {exc.code} for '{self.accession}': {exc.reason}"
            )
            return
        except urllib.error.URLError as exc:
            self.error.emit(
                f"PhaSepDB network error for '{self.accession}': {exc.reason}"
            )
            return
        except Exception as exc:
            self.error.emit(f"PhaSepDB query failed for '{self.accession}': {exc}")
            return

        # Parse JSON
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            self.error.emit(
                f"PhaSepDB returned invalid JSON for '{self.accession}': {exc}"
            )
            return

        # Handle empty or missing data
        if not data:
            self.finished.emit({"found": False, "accession": self.accession})
            return

        # PhaSepDB may return a list (multiple entries) or a single object
        if isinstance(data, list):
            if len(data) == 0:
                self.finished.emit({"found": False, "accession": self.accession})
                return
            # Take the first match (most relevant)
            entry = data[0]
        elif isinstance(data, dict):
            entry = data
        else:
            self.finished.emit({"found": False, "accession": self.accession})
            return

        # Extract fields robustly
        references = entry.get("references", entry.get("refs", []))
        if not isinstance(references, list):
            references = []

        parsed_refs: list = []
        for ref in references:
            if isinstance(ref, dict):
                parsed_refs.append(
                    {
                        "pmid": str(ref.get("pmid", ref.get("PubMed", ""))),
                        "title": ref.get("title", ref.get("Title", "")),
                    }
                )
            elif isinstance(ref, str):
                parsed_refs.append({"pmid": ref, "title": ""})

        result = {
            "found": True,
            "source": "PhaSepDB",
            "accession": self.accession,
            "gene_name": entry.get("gene_name", entry.get("gene", "")),
            "protein_name": entry.get(
                "protein_name", entry.get("name", entry.get("protein", ""))
            ),
            "category": entry.get("category", entry.get("Category", "")),
            "evidence_type": entry.get(
                "evidence_type", entry.get("evidence", entry.get("Evidence", ""))
            ),
            "organism": entry.get("organism", entry.get("Organism", "")),
            "references": parsed_refs,
        }

        self.finished.emit(result)
