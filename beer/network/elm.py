"""BEER network/elm.py — ELM (Eukaryotic Linear Motif) database query worker.

Queries the ELM REST API by UniProt accession to retrieve experimentally
validated linear motif instances.

REST endpoint:
    GET https://elm.eu.org/instances.json?q={accession}

Response shape (abbreviated):
    {
      "instances": [
        {
          "elm_identifier": "LIG_SH3_2",
          "start": 123,
          "end": 129,
          "logic": "positive",
          "toGo": "...",
          "primary_reference_pmed_id": "12345678",
          ...
        },
        ...
      ]
    }
"""

import json
import urllib.request
import urllib.error
from PyQt5.QtCore import QThread, pyqtSignal

_ELM_BASE_URL = "https://elm.eu.org/instances.json"
_TIMEOUT_SECONDS = 30


class ELMWorker(QThread):
    """Query ELM database for a sequence to find experimentally validated linear motifs.

    Uses ELM REST API: https://elm.eu.org/
    Endpoint: GET https://elm.eu.org/instances.json?q={uniprot_id}

    Signals
    -------
    finished(list):
        Emitted on success with a list of dicts, each containing:
        - elm_identifier (str)
        - start (int)
        - end (int)
        - logic (str)          "positive" | "negative" | "neutral"
        - toGo (str)           GO term(s)
        - primary_reference_pmed_id (str)
        - accession (str)      copy of the queried accession
    error(str):
        Emitted if the query fails, with a human-readable error message.
    progress(str):
        Emitted with status updates during the query.
    """

    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, accession: str, seq: str = ""):
        """Initialise the worker.

        Parameters
        ----------
        accession:
            UniProt accession (e.g. ``"P04637"``).  Will be stripped and
            upper-cased before use.
        seq:
            Protein sequence string (currently unused; reserved for future
            sequence-based search support).
        """
        super().__init__()
        self.accession = accession.strip().upper()
        self.seq = seq

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Fetch ELM instances for the stored accession.

        Builds the query URL, performs an HTTP GET request, parses the JSON
        response, and emits ``finished`` with the list of motif dicts.
        On any error (network, HTTP, JSON), emits ``error`` with a
        descriptive message.
        """
        if not self.accession:
            self.error.emit("ELM query failed: no accession provided.")
            return

        url = f"{_ELM_BASE_URL}?q={self.accession}"
        self.progress.emit(f"Querying ELM database for {self.accession}…")

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
            self.error.emit(
                f"ELM HTTP error {exc.code} for accession '{self.accession}': {exc.reason}"
            )
            return
        except urllib.error.URLError as exc:
            self.error.emit(
                f"ELM network error for accession '{self.accession}': {exc.reason}"
            )
            return
        except Exception as exc:
            self.error.emit(f"ELM query failed for '{self.accession}': {exc}")
            return

        # Parse JSON
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            self.error.emit(
                f"ELM returned invalid JSON for '{self.accession}': {exc}"
            )
            return

        # Extract instances list — ELM may return the list directly or nested
        if isinstance(data, list):
            raw_instances = data
        elif isinstance(data, dict):
            # Top-level key may be "instances" or the accession itself
            raw_instances = (
                data.get("instances")
                or data.get("Instances")
                or []
            )
            # Some ELM responses wrap results per-accession
            if not isinstance(raw_instances, list):
                raw_instances = []
        else:
            raw_instances = []

        if not raw_instances:
            self.progress.emit(
                f"No ELM instances found for accession '{self.accession}'."
            )
            self.finished.emit([])
            return

        instances: list = []
        for item in raw_instances:
            if not isinstance(item, dict):
                continue
            instances.append(
                {
                    "elm_identifier": item.get("elm_identifier", item.get("elm_type", "")),
                    "start": _safe_int(item.get("start", item.get("Start", 0))),
                    "end": _safe_int(item.get("end", item.get("End", 0))),
                    "logic": item.get("logic", item.get("Logic", "")),
                    "toGo": item.get("toGo", item.get("to_go", "")),
                    "primary_reference_pmed_id": item.get(
                        "primary_reference_pmed_id",
                        item.get("pmed_id", ""),
                    ),
                    "accession": self.accession,
                    # Preserve any extra fields returned by ELM
                    "raw": item,
                }
            )

        self.progress.emit(
            f"ELM: found {len(instances)} instance(s) for '{self.accession}'."
        )
        self.finished.emit(instances)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _safe_int(value, default: int = 0) -> int:
    """Convert *value* to int, returning *default* on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
