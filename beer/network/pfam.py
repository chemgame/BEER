"""BEER network/pfam.py — Pfam domain annotation fetch worker.

Queries the InterPro EBI API for Pfam domain annotations of a UniProt
accession.

InterPro API endpoint:
    GET https://www.ebi.ac.uk/interpro/api/entry/pfam/protein/uniprot/{accession}/
        ?page_size=100

Returns a paginated list of Pfam domain entries with protein location
fragments (start/end positions).
"""

import json
import urllib.request
from PyQt5.QtCore import QThread, pyqtSignal


class PfamWorker(QThread):
    """Fetch Pfam domain annotations for a UniProt accession via InterPro.

    Signals
    -------
    finished(list):
        Emitted on success with a list of domain dicts (sorted by start):
        - name (str)         Human-readable domain name
        - accession (str)    Pfam accession (e.g. "PF00001")
        - start (int)        Domain start position (1-based)
        - end (int)          Domain end position (1-based)
    error(str):
        Emitted on failure with a human-readable message.
    """

    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, accession: str):
        super().__init__()
        self.accession = accession.strip().upper()

    def run(self):
        try:
            url = (
                f"https://www.ebi.ac.uk/interpro/api/entry/pfam"
                f"/protein/uniprot/{self.accession}/?page_size=100"
            )
            req = urllib.request.Request(
                url, headers={"Accept": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read().decode())

            domains = []
            for result in data.get("results", []):
                meta = result.get("metadata", {})
                raw_name = meta.get("name", meta.get("accession", "Unknown"))
                name = (
                    raw_name.get("name", raw_name)
                    if isinstance(raw_name, dict)
                    else raw_name
                )
                acc = meta.get("accession", "")
                for prot in result.get("proteins", []):
                    for loc in prot.get("entry_protein_locations", []):
                        for frag in loc.get("fragments", []):
                            domains.append(
                                {
                                    "name": name,
                                    "accession": acc,
                                    "start": frag["start"],
                                    "end": frag["end"],
                                }
                            )

            domains.sort(key=lambda d: d["start"])
            self.finished.emit(domains)
        except Exception as exc:
            self.error.emit(f"Pfam fetch failed: {exc}")
