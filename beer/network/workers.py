"""QThread workers for BEER network operations (PySide6)."""
from __future__ import annotations

import json
import re
import urllib.request
import urllib.error

from PySide6.QtCore import QThread, Signal

from beer.utils.sequence import valid_uniprot, valid_pdb
from beer.network._http import (
    _USER_AGENT,
    fetch_elm,
    fetch_disprot,
    fetch_phasepdb,
    fetch_alphafold_pdb,
    fetch_pfam,
    fetch_mobidb,
    fetch_uniprot_variants,
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
        if not valid_uniprot(self.accession):
            self.error.emit(f"'{self.accession}' is not a valid UniProt accession.")
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
        if not valid_uniprot(self.accession):
            self.error.emit(f"'{self.accession}' is not a valid UniProt accession.")
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
        if not valid_uniprot(self.accession):
            self.error.emit(f"'{self.accession}' is not a valid UniProt accession.")
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
        if not valid_uniprot(self.accession):
            self.error.emit(f"'{self.accession}' is not a valid UniProt accession.")
            return
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


class FetchAccessionWorker(QThread):
    """Fetch sequence (and optionally structure) for a UniProt or PDB accession.

    Signals
    -------
    fetched_sequence(str, str):
        Emitted with (raw_fasta_text, acc) on success.
    fetched_structure(str, bool):
        Emitted with (struct_str, is_cif) when a PDB structure was retrieved.
    error(str):
        Emitted if the sequence fetch fails.
    progress(str):
        Status-bar messages suitable for display.
    """

    fetched_sequence = Signal(str, str)
    fetched_structure = Signal(str, bool)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, acc: str, is_pdb: bool, bio_assembly: bool = False):
        super().__init__()
        self.acc = acc
        self.is_pdb = is_pdb
        self.bio_assembly = bio_assembly

    def run(self) -> None:
        acc = self.acc
        if self.is_pdb:
            if not valid_pdb(acc):
                self.error.emit(f"'{acc}' is not a valid PDB entry ID (expected format: 1ABC).")
                return
        else:
            if not valid_uniprot(acc):
                self.error.emit(f"'{acc}' is not a valid UniProt accession.")
                return
        import urllib.parse as _up
        acc_quoted = _up.quote(acc.upper(), safe="")
        try:
            if self.is_pdb:
                url = f"https://www.rcsb.org/fasta/entry/{acc_quoted}"
                req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    raw = resp.read().decode()
            else:
                url = f"https://rest.uniprot.org/uniprotkb/{acc_quoted}.fasta"
                with urllib.request.urlopen(url, timeout=15) as resp:
                    raw = resp.read().decode()
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                self.error.emit(f"Rate limit (HTTP 429) — wait a moment and retry.")
            else:
                self.error.emit(f"HTTP {exc.code}: {exc.reason}")
            return
        except Exception as exc:
            self.error.emit(str(exc))
            return

        self.fetched_sequence.emit(raw, acc)

        if not self.is_pdb:
            return

        self.progress.emit(f"Downloading structure for {acc.upper()}…")
        try:
            if self.bio_assembly:
                from beer.network._http import fetch_rcsb_assembly_cif
                try:
                    cif_str = fetch_rcsb_assembly_cif(acc.upper(), assembly=1)
                    self.fetched_structure.emit(cif_str, True)
                    return
                except Exception:
                    self.progress.emit(
                        f"Assembly CIF not found for {acc.upper()}; falling back to asymmetric unit."
                    )
            import urllib.parse as _up2
            url = f"https://files.rcsb.org/download/{_up2.quote(acc.upper(), safe='')}.pdb"
            req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as resp:
                self.fetched_structure.emit(resp.read().decode(), False)
        except urllib.error.HTTPError as exc:
            self.progress.emit(
                f"Structure download failed (HTTP {exc.code}): {exc.reason}. Sequence loaded."
            )
        except Exception as exc:
            self.progress.emit(f"Structure download failed: {exc}. Sequence loaded.")


class FetchPDBStructureWorker(QThread):
    """Download only the PDB file for a given 4-character PDB ID (no FASTA fetch).

    Used by the Fix PDB tab to fetch the experimental structure independently
    of the main analysis workflow.

    Signals
    -------
    fetched(str):
        PDB string on success.
    error(str):
        Human-readable error message on failure.
    progress(str):
        Status messages.
    """

    fetched  = Signal(str)
    error    = Signal(str)
    progress = Signal(str)

    def __init__(self, pdb_id: str) -> None:
        super().__init__()
        self.pdb_id = pdb_id.strip().upper()

    def run(self) -> None:
        if not valid_pdb(self.pdb_id):
            self.error.emit(
                f"'{self.pdb_id}' is not a valid PDB entry ID (expected format: 1ABC).")
            return
        self.progress.emit(f"Downloading {self.pdb_id} from RCSB…")
        import urllib.parse as _up
        try:
            url = f"https://files.rcsb.org/download/{_up.quote(self.pdb_id, safe='')}.pdb"
            req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as resp:
                self.fetched.emit(resp.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                self.error.emit(f"Rate limit (HTTP 429) — wait a moment and retry.")
            else:
                self.error.emit(f"Could not download {self.pdb_id}: HTTP {exc.code} {exc.reason}")
        except Exception as exc:
            self.error.emit(f"Could not download {self.pdb_id}: {exc}")


class ESMFold2Worker(QThread):
    """Call the EvolutionaryScale Forge API to fold a sequence with ESMFold2.

    Signals
    -------
    finished(str):
        PDB string of the predicted structure.
    error(str):
        Human-readable error message.
    """

    finished = Signal(str)
    error    = Signal(str)

    def __init__(self, sequence: str, api_token: str) -> None:
        super().__init__()
        self.sequence  = sequence
        self.api_token = api_token

    def run(self) -> None:
        try:
            from esm.sdk.forge import SequenceStructureForgeInferenceClient
            from esm.sdk.api import ESMProtein, ESMProteinError
        except ImportError:
            self.error.emit(
                "esm package not installed.\n"
                "Install it with:  pip install esm"
            )
            return

        if not self.api_token:
            self.error.emit(
                "No BioHub API token found.\n"
                "Add your token in Settings → BioHub API Key."
            )
            return

        try:
            import warnings
            warnings.filterwarnings(
                "ignore",
                message="Entity ID not found in metadata",
                category=UserWarning,
            )
            client = SequenceStructureForgeInferenceClient(
                token=self.api_token,
                url="https://biohub.ai",
                model="esm3-open-2024-03",
            )
            result = client.fold(self.sequence)
            if isinstance(result, ESMProteinError):
                self.error.emit(f"ESMFold2 API error: {result.error_msg}")
                return
            pdb_str = result.to_pdb_string()
            self.finished.emit(pdb_str)
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")


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


class MobiDBWorker(QThread):
    """Query MobiDB for consensus disorder annotations of a UniProt accession.

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
            self.error.emit("MobiDB query failed: no accession provided.")
            return
        if not valid_uniprot(self.accession):
            self.error.emit(f"'{self.accession}' is not a valid UniProt accession.")
            return
        try:
            from beer.network._http import fetch_mobidb
            result = fetch_mobidb(self.accession)
        except urllib.error.HTTPError as exc:
            self.error.emit(
                f"MobiDB HTTP error {exc.code} for '{self.accession}': {exc.reason}"
            )
            return
        except urllib.error.URLError as exc:
            self.error.emit(
                f"MobiDB network error for '{self.accession}': {exc.reason}"
            )
            return
        except Exception as exc:
            self.error.emit(f"MobiDB query failed for '{self.accession}': {exc}")
            return
        self.finished.emit(result)


class UniProtVariantsWorker(QThread):
    """Fetch natural variants and disease mutations from UniProt for a UniProt accession.

    Signals
    -------
    finished(list):
        Emitted on success with a list of variant dicts (may be empty).
    error(str):
        Emitted on network/parse errors.
    progress(str):
        Emitted with status updates.
    """

    finished = Signal(list)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, accession: str):
        super().__init__()
        self.accession = accession.strip().upper()

    def run(self) -> None:
        if not self.accession:
            self.error.emit("UniProt variants query failed: no accession provided.")
            return
        if not valid_uniprot(self.accession):
            self.error.emit(f"'{self.accession}' is not a valid UniProt accession.")
            return
        self.progress.emit(f"Querying UniProt for variants of {self.accession}\u2026")
        try:
            from beer.network._http import fetch_uniprot_variants
            variants = fetch_uniprot_variants(self.accession)
        except urllib.error.HTTPError as exc:
            self.error.emit(
                f"UniProt variants HTTP error {exc.code} for '{self.accession}': {exc.reason}"
            )
            return
        except urllib.error.URLError as exc:
            self.error.emit(
                f"UniProt variants network error for '{self.accession}': {exc.reason}"
            )
            return
        except Exception as exc:
            self.error.emit(f"UniProt variants query failed for '{self.accession}': {exc}")
            return

        if not variants:
            self.progress.emit(
                f"No variants found for accession '{self.accession}'."
            )
        else:
            self.progress.emit(
                f"UniProt variants: found {len(variants)} variant(s) for '{self.accession}'."
            )
        self.finished.emit(variants)


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
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation.  The blocking qblast() call cannot be interrupted
        mid-flight; the caller should follow up with terminate() if the thread
        does not finish within a grace period."""
        self._cancelled = True
        self.requestInterruption()

    def run(self) -> None:
        try:
            from Bio.Blast import NCBIWWW, NCBIXML
            self.progress.emit("Submitting BLAST search (this may take 1\u20133 min)\u2026")
            result_handle = NCBIWWW.qblast(
                "blastp", self.database, self.seq,
                hitlist_size=self.hitlist_size,
            )
            # If cancel() was called while qblast() was blocking, discard results.
            if self._cancelled or self.isInterruptionRequested():
                return
            self.progress.emit("Parsing BLAST results\u2026")
            blast_record = NCBIXML.read(result_handle)
            if self._cancelled or self.isInterruptionRequested():
                return
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
            if not self._cancelled:
                self.error.emit(f"BLAST failed: {exc}")


class IntActWorker(QThread):
    """Fetch molecular interactions from IntAct via PSICQUIC.

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
            self.error.emit("IntAct query failed: no accession provided.")
            return
        if not valid_uniprot(self.accession):
            self.error.emit(f"'{self.accession}' is not a valid UniProt accession.")
            return
        try:
            from beer.network._http import fetch_intact
            result = fetch_intact(self.accession)
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                self.error.emit(f"IntAct rate limit (HTTP 429) — wait a moment and retry.")
            else:
                self.error.emit(
                    f"IntAct HTTP error {exc.code} for '{self.accession}': {exc.reason}"
                )
        except urllib.error.URLError as exc:
            self.error.emit(
                f"IntAct network error for '{self.accession}': {exc.reason}"
            )
        except Exception as exc:
            self.error.emit(f"IntAct query failed for '{self.accession}': {exc}")
        else:
            self.finished.emit(result)


class BatchAnalysisWorker(QThread):
    """Analyse a list of (rec_id, seq) pairs off the main thread.

    Emits one ``chain_result`` signal per chain as results arrive, then
    ``finished`` with the list of skipped rec_ids when all chains are done.

    Signals
    -------
    chain_result(str, str, dict):
        (rec_id, seq, analysis_data) for each successfully analysed chain.
    finished(list):
        List of rec_ids that were skipped (invalid sequence, too short, error).
    progress(str):
        Human-readable status for the status bar.
    """

    chain_result = Signal(str, str, dict)
    finished = Signal(list)
    progress = Signal(str)

    def __init__(self, entries: list, pH: float, window_size: int,
                 use_reducing: bool, pka: dict | None,
                 hydro_scale: str = "Kyte-Doolittle", embedder=None):
        super().__init__()
        self.entries = entries
        self.pH = pH
        self.window_size = window_size
        self.use_reducing = use_reducing
        self.pka = pka
        self.hydro_scale = hydro_scale
        self.embedder = embedder

    def run(self) -> None:
        from beer.analysis.core import AnalysisTools
        from beer.utils.sequence import is_valid_protein
        skipped: list[str] = []
        total = len(self.entries)
        for idx, (rec_id, seq) in enumerate(self.entries, 1):
            if self.isInterruptionRequested():
                break
            if not is_valid_protein(seq) or len(seq) < 5:
                skipped.append(rec_id)
                continue
            self.progress.emit(f"Analysing chain {idx}/{total}: {rec_id} ({len(seq)} aa)…")
            try:
                data = AnalysisTools.analyze_sequence(
                    seq, self.pH, self.window_size,
                    self.use_reducing, self.pka,
                    embedder=self.embedder,
                    hydro_scale=self.hydro_scale,
                )
                self.chain_result.emit(rec_id, seq, data)
            except Exception as exc:
                self.progress.emit(f"Skipping {rec_id}: {exc}")
                skipped.append(rec_id)
        self.finished.emit(skipped)


class AnalysisWorker(QThread):
    """Non-blocking analysis in a QThread. Emits finished(dict) or error(str).

    Signals
    -------
    finished(dict)
    error(str)
    """

    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, seq, pH, window_size, use_reducing, pka,
                 hydro_scale="Kyte-Doolittle", embedder=None):
        super().__init__()
        self.seq = seq
        self.pH = pH
        self.window_size = window_size
        self.use_reducing = use_reducing
        self.pka = pka
        self.hydro_scale = hydro_scale
        self.embedder = embedder

    def run(self):
        try:
            # Import locally to avoid circular imports at module load time
            from beer.analysis.core import AnalysisTools
            data = AnalysisTools.analyze_sequence(
                self.seq, self.pH, self.window_size,
                self.use_reducing, self.pka,
                embedder=self.embedder,
                hydro_scale=self.hydro_scale,
            )
            self.finished.emit(data)
        except Exception as exc:
            self.error.emit(str(exc))


class AISectionWorker(QThread):
    """Compute a single BiLSTM head on demand (lazy AI section loading).

    The ESMCEmbedder caches embeddings by sequence hash, so after the first
    head is computed the embedding is reused for all subsequent heads.

    Signals
    -------
    result_ready(str, list):
        (section_key, scores) — section_key is "AI:<display_name>",
        scores is the per-residue probability list.
    error(str, str):
        (section_key, message)
    """

    result_ready = Signal(str, list)
    error        = Signal(str, str)

    def __init__(self, section_key: str, data_key: str, seq: str, embedder):
        super().__init__()
        self.section_key = section_key
        self.data_key    = data_key
        self.seq         = seq
        self.embedder    = embedder

    def run(self) -> None:
        if self.embedder is None or not self.embedder.is_available():
            self.error.emit(
                self.section_key,
                "ESMC model is not available on this system. "
                "Install esm (pip install esm) to enable AI predictions."
            )
            return
        try:
            from beer.analysis.core import compute_single_bilstm_head
            scores = compute_single_bilstm_head(self.data_key, self.seq, self.embedder)
            if scores:
                self.result_ready.emit(self.section_key, scores)
            else:
                self.error.emit(
                    self.section_key,
                    f"Model not available or embedder unavailable for '{self.section_key}'."
                )
        except Exception as exc:
            import traceback as _tb
            self.error.emit(
                self.section_key,
                f"{type(exc).__name__}: {exc}\n{_tb.format_exc()}"
            )


class OverlayWorker(QThread):
    """Compute all selected overlay profiles, lazy-loading missing BiLSTM heads.

    The ESMC embedding is cached by sequence hash, so after the first head is
    computed subsequent heads only pay the fast BiLSTM forward-pass cost.

    Signals
    -------
    progress(str):
        Short message suitable for a QProgressDialog label (e.g. "Computing Disorder…")
    finished(dict):
        {display_name: [float, …]} for all successfully computed profiles.
    error(str):
        Human-readable error message (non-fatal; worker continues).
    """

    progress = Signal(str)
    finished = Signal(dict, dict)  # (profiles, newly_computed_analysis_keys)
    error    = Signal(str)

    def __init__(
        self,
        selected: "list[tuple[str, str, list | None]]",
        seq: str,
        embedder,
        bilstm_lazy_keys: "frozenset[str]",
    ):
        super().__init__()
        # selected: list of (display_name, key, already_fetched_data_or_None)
        self._selected        = selected
        self._seq             = seq
        self._embedder        = embedder
        self._bilstm_lazy_keys = bilstm_lazy_keys

    def run(self) -> None:
        from beer.analysis.core import compute_single_bilstm_head
        profiles: dict[str, list] = {}
        computed: dict[str, list] = {}
        for display_name, key, data in self._selected:
            if data is None and key in self._bilstm_lazy_keys:
                self.progress.emit(f"Computing {display_name}…")
                try:
                    data = compute_single_bilstm_head(key, self._seq, self._embedder)
                    if data is not None:
                        computed[key] = data
                except Exception as exc:
                    self.error.emit(f"{display_name}: {exc}")
                    continue
            if data is not None:
                try:
                    if len(data) > 0:
                        profiles[display_name] = list(data)
                except Exception:
                    pass
        self.finished.emit(profiles, computed)


class MCDropoutWorker(QThread):
    """Run MC-Dropout (N stochastic forward passes) for a single BiLSTM head.

    Signals
    -------
    result_ready(str, list):
        (graph_title, uncertainty_list)
    error(str, str):
        (graph_title, message)
    """

    result_ready = Signal(str, list)
    error        = Signal(str, str)

    def __init__(self, title: str, feat: str, seq: str, embedder,
                 n_passes: int = 20) -> None:
        super().__init__()
        self.title    = title
        self.feat     = feat
        self.seq      = seq
        self.embedder = embedder
        self.n_passes = n_passes

    def run(self) -> None:
        try:
            from beer.utils.structure import bilstm_predict_mc
            import beer.models as _m
            _loaders = {
                "disorder":      "load_disorder_head",
                "signal_peptide":"load_signal_peptide_head",
                "transmembrane": "load_transmembrane_head",
                "intramembrane": "load_intramembrane_head",
                "coiled_coil":   "load_coiled_coil_head",
                "dna_binding":   "load_dna_binding_head",
                "active_site":   "load_active_site_head",
                "binding_site":  "load_binding_site_head",
                "phosphorylation":"load_phosphorylation_head",
                "lcd":           "load_lcd_head",
                "zinc_finger":   "load_zinc_finger_head",
                "glycosylation": "load_glycosylation_head",
                "ubiquitination":"load_ubiquitination_head",
                "methylation":   "load_methylation_head",
                "acetylation":   "load_acetylation_head",
                "lipidation":    "load_lipidation_head",
                "disulfide":     "load_disulfide_head",
                "motif":         "load_motif_head",
                "propeptide":    "load_propeptide_head",
                "repeat":              "load_repeat_head",
                "rna_binding":         "load_rna_binding_head",
                "nucleotide_binding":  "load_nucleotide_binding_head",
                "transit_peptide":     "load_transit_peptide_head",
            }
            loader_name = _loaders.get(self.feat)
            if not loader_name or not hasattr(_m, loader_name):
                self.error.emit(self.title, f"No loader for feature '{self.feat}'")
                return
            head = getattr(_m, loader_name)()
            if head is None or self.embedder is None:
                self.error.emit(self.title, "Model or embedder unavailable.")
                return
            result = bilstm_predict_mc(
                self.seq, self.embedder, head, n_passes=self.n_passes)
            if result is not None:
                _, uncertainty = result
                self.result_ready.emit(self.title, list(uncertainty))
            else:
                self.error.emit(self.title, "MC-Dropout returned no result.")
        except Exception as exc:
            import traceback as _tb
            self.error.emit(self.title,
                            f"{type(exc).__name__}: {exc}\n{_tb.format_exc()}")


class AlphaMissenseWorker(QThread):
    """Fetches AlphaMissense scores via EBI API (requires internet + UniProt ID)."""
    finished = Signal(dict)
    error    = Signal(str)

    def __init__(self, uniprot_id: str, parent=None):
        super().__init__(parent)
        self._uid = uniprot_id

    def run(self):
        if not valid_uniprot(self._uid):
            self.error.emit(f"'{self._uid}' is not a valid UniProt accession.")
            return
        try:
            from beer.analysis.alphafold_data import fetch_alphafold_missense_scores
            data = fetch_alphafold_missense_scores(self._uid)
            self.finished.emit(data)
        except Exception as e:
            self.error.emit(str(e))


class UniProtSequenceSearchWorker(QThread):
    """Look up a UniProt accession from a raw sequence.

    Strategy (fastest-first)
    ------------------------
    1. Parse a UniProt accession directly from the sequence_name hint if
       provided (instant — covers sequences pasted as FASTA from UniProt).
    2. Fetch all reviewed Swiss-Prot entries of the same length and compare
       sequences in memory (~2 s for rare lengths, may be slow for very common
       lengths; capped at 500 candidates).
    3. NCBI BLAST against Swiss-Prot (~1–3 min; used only when steps 1–2 fail).

    Signals
    -------
    finished(str): accession found, or ``""`` if not found.
    error(str):    unrecoverable error message.
    progress(str): status-bar updates.
    """

    finished = Signal(str)
    error    = Signal(str)
    progress = Signal(str)

    _ACC_RE = re.compile(
        r"(?:^|[|/\s])([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})"
    )

    def __init__(self, seq: str, name_hint: str = "", parent=None):
        super().__init__(parent)
        self._seq  = seq.upper().strip()
        self._hint = name_hint

    def run(self):
        try:
            acc = self._from_name_hint() or self._length_exact_match()
            self.finished.emit(acc or "")
        except Exception as exc:
            self.error.emit(str(exc))

    # ── stage 1: parse accession from FASTA header ────────────────────────────

    def _from_name_hint(self) -> str:
        """Extract UniProt accession from the sequence name if it looks like
        a UniProt FASTA id (e.g. ``sp|P69905|HBA_HUMAN`` or bare ``P69905``)."""
        if not self._hint:
            return ""
        m = self._ACC_RE.search(self._hint)
        if not m:
            return ""
        candidate = m.group(1)
        self.progress.emit(f"Checking candidate accession {candidate}…")
        try:
            url = f"https://rest.uniprot.org/uniprotkb/{candidate}.json"
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read())
            uniprot_seq = data.get("sequence", {}).get("value", "")
            if uniprot_seq.upper() == self._seq:
                self.progress.emit(f"Matched from header: {candidate}")
                return candidate
        except Exception:
            pass
        return ""

    # ── stage 2: fetch same-length Swiss-Prot entries and compare ─────────────

    def _length_exact_match(self) -> str:
        import urllib.parse
        n = len(self._seq)
        self.progress.emit(f"Searching UniProt for reviewed proteins of length {n}…")
        query = urllib.parse.quote(f"reviewed:true AND length:[{n} TO {n}]")
        url = (
            "https://rest.uniprot.org/uniprotkb/search"
            f"?query={query}"
            "&format=json&fields=accession,sequence&size=500"
        )
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                data = json.loads(resp.read())
            for entry in data.get("results", []):
                acc = entry.get("primaryAccession", "")
                uniprot_seq = entry.get("sequence", {}).get("value", "")
                if uniprot_seq.upper() == self._seq:
                    self.progress.emit(f"Exact length match: {acc}")
                    return acc
        except Exception:
            pass
        return ""



class ProteinSummaryWorker(QThread):
    """Fetch protein metadata from UniProt or RCSB for the Summary tab.

    Performs a single HTTP request off the main thread so the UI stays
    responsive while the card data is being retrieved.

    Signals
    -------
    result(dict):
        Emitted with the parsed card dict on success.
    error(str):
        Emitted when the request fails (summary is best-effort; callers
        may silently ignore this signal).
    """

    result = Signal(dict)
    error  = Signal(str)

    def __init__(self, acc: str, is_pdb: bool, parent=None):
        super().__init__(parent)
        self.acc    = acc
        self.is_pdb = is_pdb

    def run(self) -> None:
        acc = self.acc
        if self.is_pdb:
            if not valid_pdb(acc):
                self.error.emit(f"'{acc}' is not a valid PDB entry ID (expected format: 1ABC).")
                return
        else:
            if not valid_uniprot(acc):
                self.error.emit(f"'{acc}' is not a valid UniProt accession.")
                return
        import urllib.parse as _up
        try:
            if self.is_pdb:
                url = f"https://data.rcsb.org/rest/v1/core/entry/{_up.quote(acc.upper(), safe='')}"
                req = urllib.request.Request(url, headers={
                    "Accept": "application/json", "User-Agent": _USER_AGENT})
                with urllib.request.urlopen(req, timeout=8) as resp:
                    data = json.loads(resp.read().decode())
                card = {
                    "source":    "PDB",
                    "accession": acc.upper(),
                    "name":      data.get("struct", {}).get("title", ""),
                }
            else:
                url = f"https://rest.uniprot.org/uniprotkb/{_up.quote(acc, safe='')}.json"
                req = urllib.request.Request(url, headers={
                    "Accept": "application/json", "User-Agent": _USER_AGENT})
                with urllib.request.urlopen(req, timeout=8) as resp:
                    data = json.loads(resp.read().decode())
                pd_obj = data.get("proteinDescription", {})
                rec = pd_obj.get("recommendedName") or (pd_obj.get("submittedNames") or [{}])[0]
                prot_name = (rec.get("fullName") or {}).get("value", "")
                genes     = data.get("genes", [])
                gene      = (genes[0].get("geneName") or {}).get("value", "") if genes else ""
                organism  = data.get("organism", {}).get("scientificName", "")
                func_texts, subcel_texts, disease_texts, ptm_texts, caution_texts = [], [], [], [], []
                for c in data.get("comments", []):
                    ct = c.get("commentType", "")
                    if ct == "FUNCTION":
                        for t in c.get("texts", []):
                            v = t.get("value", "").strip()
                            if v:
                                func_texts.append(v)
                    elif ct == "SUBCELLULAR LOCATION":
                        for loc in c.get("subcellularLocations", []):
                            lv = (loc.get("location") or {}).get("value", "")
                            if lv:
                                subcel_texts.append(lv)
                    elif ct == "DISEASE":
                        d = c.get("disease", {})
                        dname = d.get("diseaseId", "") or d.get("description", "")
                        if dname:
                            disease_texts.append(dname)
                    elif ct == "PTM":
                        for t in c.get("texts", []):
                            v = t.get("value", "").strip()
                            if v:
                                ptm_texts.append(v)
                    elif ct == "CAUTION":
                        for t in c.get("texts", []):
                            v = t.get("value", "").strip()
                            if v:
                                caution_texts.append(v)
                keywords = [kw.get("name", "") for kw in data.get("keywords", []) if kw.get("name")]
                card = {
                    "source":     "UniProt",
                    "accession":  acc,
                    "name":       prot_name,
                    "gene":       gene,
                    "organism":   organism,
                    "function":   func_texts,
                    "subcellular": list(dict.fromkeys(subcel_texts)),
                    "diseases":   disease_texts,
                    "ptm":        ptm_texts,
                    "keywords":   keywords,
                }
            self.result.emit(card)
        except Exception as exc:
            self.error.emit(str(exc))


class UniProtFeaturesWorker(QThread):
    """Fetch UniProt feature annotations for a given accession.

    Returns per-feature region lists suitable for dual-track visualization.
    Each region dict has keys: ``feature``, ``start`` (1-based), ``end`` (1-based),
    ``description``.

    Signals
    -------
    finished(dict):
        Maps feature type string → list of region dicts.
    error(str)
    """

    # UniProt ft_* field names → BEER feature names
    _FIELD_MAP: dict[str, str] = {
        "Signal":                           "signal_peptide",
        "Transit peptide":                  "transit_peptide",
        "Transmembrane":                    "transmembrane",
        "Intramembrane":                    "intramembrane",
        "Coiled coil":                      "coiled_coil",
        "DNA binding":                      "dna_binding",
        "Active site":                      "active_site",
        "Binding site":                     "binding_site",

        "Propeptide":                       "propeptide",
        "Repeat":                           "repeat",
        "Motif":                            "motif",
        "Region":                           "region",
        "Disulfide bond":                   "disulfide",
        "Zinc finger":                      "zinc_finger",
        "Glycosylation":                    "glycosylation",
        "Lipid moiety-binding region":      "lipidation",

        "Helix":                            "secondary_structure_helix",
        "Beta strand":                      "secondary_structure_strand",
    }
    # ft_mod_res description → BEER feature name (substring match, first hit wins)
    _MOD_RES_MAP: list[tuple[str, str]] = [
        ("phospho",      "phosphorylation"),
        ("ubiquitin",    "ubiquitination"),
        ("methyl",       "methylation"),
        ("acetyl",       "acetylation"),
    ]

    finished = Signal(dict)
    error    = Signal(str)

    def __init__(self, accession: str, parent=None):
        super().__init__(parent)
        self._acc = accession

    def run(self):
        if not valid_uniprot(self._acc):
            self.error.emit(f"'{self._acc}' is not a valid UniProt accession.")
            return
        try:
            import urllib.parse as _up
            fields = (
                "ft_signal,ft_transit,ft_transmem,ft_intramem,ft_coiled,ft_dna_bind,"
                "ft_act_site,ft_binding,ft_propep,ft_repeat,ft_motif,"
                "ft_region,ft_compbias,ft_disulfid,ft_mod_res,ft_carbohyd,ft_lipid,ft_zn_fing,"
                "ft_helix,ft_strand"
            )
            url = (
                f"https://rest.uniprot.org/uniprotkb/{_up.quote(self._acc, safe='')}.json"
                f"?fields=accession,{fields}"
            )
            try:
                resp_cm = urllib.request.urlopen(url, timeout=20)
            except urllib.error.HTTPError as _he:
                if _he.code == 404:
                    self.finished.emit({})
                    return
                raise
            with resp_cm as resp:
                data = json.loads(resp.read())

            result: dict[str, list] = {}
            for feat in data.get("features", []):
                ftype = feat.get("type", "")
                desc  = feat.get("description", "")
                loc   = feat.get("location", {})
                start = loc.get("start", {}).get("value")
                end   = loc.get("end",   {}).get("value")
                if start is None or end is None:
                    continue

                entry = {
                    "start":        int(start),
                    "end":          int(end),
                    "description":  desc,
                    "feature_type": ftype,
                }

                if ftype == "Modified residue":
                    # Dispatch by description keyword
                    desc_lc = desc.lower()
                    beer_name = None
                    for kw, name in self._MOD_RES_MAP:
                        if kw in desc_lc:
                            beer_name = name
                            break
                    if beer_name is None:
                        beer_name = "modified_residue"
                elif ftype == "Compositionally biased":
                    # All compositionally biased regions are low-complexity proxies
                    beer_name = "lcd"
                elif ftype == "Region":
                    beer_name = "region"
                    # Also emit under rna_binding if the description mentions RNA
                    if "rna" in desc.lower():
                        result.setdefault("rna_binding", []).append(entry)
                else:
                    beer_name = self._FIELD_MAP.get(ftype, ftype.lower().replace(" ", "_"))

                result.setdefault(beer_name, []).append(entry)
            self.finished.emit(result)
        except Exception as exc:
            import traceback as _tb
            self.error.emit(f"UniProt features fetch failed: {type(exc).__name__}: {exc}\n{_tb.format_exc()}")




class VariantEffectWorker(QThread):
    """Compute ESMC single-mutant log-likelihood ratios off the main thread.

    Signals
    -------
    finished(object):
        Emitted with the LLR numpy array on success.
    error(str):
        Emitted on failure (ESMC unavailable or computation error).
    """

    finished = Signal(object)
    error    = Signal(str)

    def __init__(self, seq: str, embedder, parent=None):
        super().__init__(parent)
        self._seq = seq
        self._embedder = embedder

    def run(self) -> None:
        try:
            from beer.analysis.variant_scoring import compute_single_mutant_llr
            llr = compute_single_mutant_llr(self._seq, self._embedder)
            if llr is None:
                self.error.emit("ESMC embedder unavailable — cannot compute variant effect map.")
            else:
                self.finished.emit(llr)
        except Exception as exc:
            import traceback as _tb
            self.error.emit(
                f"Variant effect computation failed: {type(exc).__name__}: {exc}\n{_tb.format_exc()}"
            )


class TruncationWorker(QThread):
    """Run percentage-based N/C-terminal truncation series off the main thread.

    Emits one ``row_ready`` per truncation variant, then ``finished`` when done.

    Signals
    -------
    row_ready(str, int, int, dict):
        ("N-term"|"C-term", pct_removed, remaining_length, analysis_data_dict)
    finished():
        Emitted when all truncations have been computed.
    error(str):
        Emitted on unrecoverable failure.
    progress(str):
        Emitted with status updates.
    """

    row_ready = Signal(str, int, int, dict)
    finished  = Signal()
    error     = Signal(str)
    progress  = Signal(str)

    def __init__(self, seq: str, embedder, step: int = 10,
                 do_n: bool = True, do_c: bool = True,
                 ph: float = 7.4, window: int = 9,
                 reducing: bool = False, pka: dict = None, parent=None):
        super().__init__(parent)
        self._seq = seq
        self._embedder = embedder
        self._step = step
        self._do_n = do_n
        self._do_c = do_c
        self._ph = ph
        self._window = window
        self._reducing = reducing
        self._pka = pka
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True
        self.requestInterruption()

    def run(self) -> None:
        try:
            from beer.analysis.core import AnalysisTools
            from beer.utils.sequence import is_valid_protein
            seq = self._seq
            n = len(seq)
            for pct in list(range(self._step, 100, self._step)) + [100]:
                if self._cancelled or self.isInterruptionRequested():
                    return
                n_rem = max(5, int(n * (1 - pct / 100)))
                for ttype, trunc in (
                    ("N-term", seq[n - n_rem:] if self._do_n else None),
                    ("C-term", seq[:n_rem]      if self._do_c else None),
                ):
                    if trunc is None or not is_valid_protein(trunc) or len(trunc) < 5:
                        continue
                    if self._cancelled or self.isInterruptionRequested():
                        return
                    self.progress.emit(f"{ttype} {pct}% removed ({n_rem} aa remaining)…")
                    try:
                        data = AnalysisTools.analyze_sequence(
                            trunc, self._ph, self._window,
                            self._reducing, self._pka, embedder=self._embedder)
                    except Exception as _exc:
                        import logging as _log
                        _log.getLogger("beer.workers").warning(
                            "Truncation %s pct=%d failed: %s", ttype, pct, _exc)
                        continue
                    self.row_ready.emit(ttype, pct, len(trunc), data)
            self.finished.emit()
        except Exception as exc:
            import traceback as _tb
            self.error.emit(
                f"Truncation series failed: {type(exc).__name__}: {exc}\n{_tb.format_exc()}"
            )


class CompositeStructureWorker(QThread):
    """Fetch AlphaFold model and build a composite structure in a background thread.

    Signals
    -------
    finished(object)  :  CompositeResult
    error(str)
    progress(str)
    """

    finished = Signal(object)
    error    = Signal(str)
    progress = Signal(str)

    def __init__(self, exp_pdb: str, accession: str) -> None:
        super().__init__()
        self.exp_pdb    = exp_pdb
        self.accession  = accession

    def run(self) -> None:
        try:
            self.progress.emit("Fetching AlphaFold model…")
            af_data = fetch_alphafold_pdb(self.accession)
            af_pdb  = af_data["pdb_str"]

            self.progress.emit("Building composite structure…")
            from beer.analysis.composite_structure import build_composite
            result = build_composite(self.exp_pdb, af_pdb)
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class CompositeStructureESMFold2Worker(QThread):
    """Fold a full-length sequence with ESMFold2 and build a composite structure.

    Alternative to :class:`CompositeStructureWorker` for proteins without an
    AlphaFold model: the experimental gaps are filled from an ESMFold2
    prediction of the complete construct sequence instead of AlphaFold.

    Signals
    -------
    finished(object)  :  CompositeResult
    error(str)
    progress(str)
    """

    finished = Signal(object)
    error    = Signal(str)
    progress = Signal(str)

    def __init__(self, exp_pdb: str, full_seq: str, api_token: str) -> None:
        super().__init__()
        self.exp_pdb   = exp_pdb
        self.full_seq  = full_seq
        self.api_token = api_token

    def run(self) -> None:
        try:
            from esm.sdk.forge import SequenceStructureForgeInferenceClient
            from esm.sdk.api import ESMProteinError
        except ImportError:
            self.error.emit(
                "esm package not installed.\nInstall it with:  pip install esm")
            return

        if not self.api_token:
            self.error.emit(
                "No BioHub API token found.\n"
                "Add your token in Settings → BioHub API Key.")
            return

        try:
            self.progress.emit("Predicting full structure with ESMFold2…")
            import warnings
            warnings.filterwarnings(
                "ignore",
                message="Entity ID not found in metadata",
                category=UserWarning,
            )
            client = SequenceStructureForgeInferenceClient(
                token=self.api_token,
                url="https://biohub.ai",
                model="esm3-open-2024-03",
            )
            result = client.fold(self.full_seq)
            if isinstance(result, ESMProteinError):
                self.error.emit(f"ESMFold2 API error: {result.error_msg}")
                return
            esm_pdb = result.to_pdb_string()

            self.progress.emit("Building composite structure…")
            from beer.analysis.composite_structure import build_composite
            comp = build_composite(self.exp_pdb, esm_pdb,
                                   fill_source="ESMFold2")
            self.finished.emit(comp)
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")
