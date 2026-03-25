"""QThread workers for BEER network operations (PySide6)."""
from __future__ import annotations

import json
import urllib.request
import urllib.error

from PySide6.QtCore import QThread, Signal

from beer.network._http import (
    fetch_elm,
    fetch_disprot,
    fetch_phasepdb,
    fetch_alphafold_pdb,
    fetch_pfam,
)


def _safe_int(value, default: int = 0) -> int:
    """Convert *value* to int, returning *default* on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class ELMWorker(QThread):
    """Query ELM database for a sequence to find experimentally validated linear motifs.

    Uses ELM REST API: https://elm.eu.org/
    Endpoint: GET https://elm.eu.org/instances.json?q={uniprot_id}

    Signals
    -------
    finished(list):
        Emitted on success with a list of motif dicts.
    error(str):
        Emitted if the query fails.
    progress(str):
        Emitted with status updates.
    """

    finished = Signal(list)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, accession: str, seq: str = ""):
        super().__init__()
        self.accession = accession.strip().upper()
        self.seq = seq

    def run(self) -> None:
        if not self.accession:
            self.error.emit("ELM query failed: no accession provided.")
            return
        self.progress.emit(f"Querying ELM database for {self.accession}\u2026")
        try:
            instances = fetch_elm(self.accession)
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

        if not instances:
            self.progress.emit(
                f"No ELM instances found for accession '{self.accession}'."
            )
        else:
            self.progress.emit(
                f"ELM: found {len(instances)} instance(s) for '{self.accession}'."
            )
        self.finished.emit(instances)


class DisPRotWorker(QThread):
    """Query DisProt database for disorder annotations of a UniProt accession.

    Signals
    -------
    finished(dict):
        Emitted on success (including not-found cases).
    error(str):
        Emitted on network/parse errors.
    """

    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, accession: str):
        super().__init__()
        self.accession = accession.strip().upper()

    def run(self) -> None:
        if not self.accession:
            self.error.emit("DisProt query failed: no accession provided.")
            return
        try:
            result = fetch_disprot(self.accession)
        except urllib.error.HTTPError as exc:
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
        self.finished.emit(result)


class PhaSepDBWorker(QThread):
    """Query PhaSepDB for phase separation data.

    Signals
    -------
    finished(dict):
        Emitted when the query completes (success or not-found).
    error(str):
        Emitted on genuine network/parse failures.
    """

    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, accession: str):
        super().__init__()
        self.accession = accession.strip().upper()

    def run(self) -> None:
        if not self.accession:
            self.error.emit("PhaSepDB query failed: no accession provided.")
            return
        try:
            result = fetch_phasepdb(self.accession)
        except urllib.error.HTTPError as exc:
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
        self.finished.emit(result)


class AlphaFoldWorker(QThread):
    """Fetch AlphaFold predicted structure for a UniProt accession.

    Emits finished(dict) with keys: pdb_str, plddt, dist_matrix, accession.

    Signals
    -------
    finished(dict)
    error(str)
    progress(str)
    """

    finished = Signal(dict)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, accession: str):
        super().__init__()
        self.accession = accession.strip().upper()

    def run(self) -> None:
        try:
            self.progress.emit(f"Querying AlphaFold for {self.accession}\u2026")
            result = fetch_alphafold_pdb(self.accession)
            pdb_str = result["pdb_str"]
            self.progress.emit("Extracting pLDDT and distance matrix\u2026")
            # Import analysis helpers locally to avoid circular imports
            from beer.utils.pdb import extract_plddt_from_pdb, compute_ca_distance_matrix
            plddt = extract_plddt_from_pdb(pdb_str)
            dist_matrix = compute_ca_distance_matrix(pdb_str)
            self.finished.emit({
                "pdb_str": pdb_str,
                "plddt": plddt,
                "dist_matrix": dist_matrix,
                "accession": self.accession,
            })
        except ValueError as exc:
            self.error.emit(str(exc))
        except urllib.error.HTTPError as exc:
            self.error.emit(f"AlphaFold fetch failed: HTTP {exc.code} {exc.reason}")
        except urllib.error.URLError as exc:
            self.error.emit(f"AlphaFold fetch failed: network error {exc.reason}")
        except Exception as exc:
            self.error.emit(f"AlphaFold fetch failed: {exc}")


class PfamWorker(QThread):
    """Fetch Pfam domain annotations for a UniProt accession via InterPro REST API.

    Signals
    -------
    finished(list)
    error(str)
    """

    finished = Signal(list)
    error = Signal(str)

    def __init__(self, accession: str):
        super().__init__()
        self.accession = accession.strip().upper()

    def run(self) -> None:
        try:
            domains = fetch_pfam(self.accession)
            self.finished.emit(domains)
        except urllib.error.HTTPError as exc:
            self.error.emit(f"Pfam fetch failed: HTTP {exc.code} {exc.reason}")
        except urllib.error.URLError as exc:
            self.error.emit(f"Pfam fetch failed: network error {exc.reason}")
        except Exception as exc:
            self.error.emit(f"Pfam fetch failed: {exc}")


class BlastWorker(QThread):
    """Run NCBI blastp and return top hits. Can take 1-3 minutes.

    Signals
    -------
    finished(list)
    error(str)
    progress(str)
    """

    finished = Signal(list)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, seq: str, database: str = "nr", hitlist_size: int = 20):
        super().__init__()
        self.seq = seq
        self.database = database
        self.hitlist_size = hitlist_size

    def run(self) -> None:
        try:
            from Bio.Blast import NCBIWWW, NCBIXML
            self.progress.emit("Submitting BLAST search (this may take 1\u20133 min)\u2026")
            result_handle = NCBIWWW.qblast(
                "blastp", self.database, self.seq,
                hitlist_size=self.hitlist_size,
            )
            self.progress.emit("Parsing BLAST results\u2026")
            blast_record = NCBIXML.read(result_handle)
            hits = []
            for aln in blast_record.alignments[:self.hitlist_size]:
                hsp = aln.hsps[0]
                hits.append({
                    "accession": aln.accession,
                    "title": aln.title[:100],
                    "length": aln.length,
                    "score": hsp.score,
                    "e_value": hsp.expect,
                    "identity": hsp.identities / hsp.align_length * 100,
                    "subject": hsp.sbjct.replace("-", ""),
                })
            self.finished.emit(hits)
        except ImportError:
            self.error.emit("Bio.Blast not available \u2014 install biopython.")
        except Exception as exc:
            self.error.emit(f"BLAST failed: {exc}")


class AnalysisWorker(QThread):
    """Non-blocking analysis in a QThread. Emits finished(dict) or error(str).

    Signals
    -------
    finished(dict)
    error(str)
    """

    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, seq, pH, window_size, use_reducing, pka):
        super().__init__()
        self.seq = seq
        self.pH = pH
        self.window_size = window_size
        self.use_reducing = use_reducing
        self.pka = pka

    def run(self):
        try:
            # Import locally to avoid circular imports at module load time
            from beer.analysis.core import AnalysisTools
            data = AnalysisTools.analyze_sequence(
                self.seq, self.pH, self.window_size,
                self.use_reducing, self.pka,
            )
            self.finished.emit(data)
        except Exception as exc:
            self.error.emit(str(exc))
