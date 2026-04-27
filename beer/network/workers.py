"""QThread workers for BEER network operations (PySide6)."""
from __future__ import annotations

import json
import re
import urllib.request
import urllib.error

from PySide6.QtCore import QThread, Signal

from beer.network._http import (
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
        try:
            from beer.network._http import fetch_intact
            result = fetch_intact(self.accession)
        except urllib.error.HTTPError as exc:
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

    The ESM2Embedder caches embeddings by sequence hash, so after the first
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
                "aggregation":         "load_aggregation_head",
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


class DeepTMHMMWorker(QThread):
    """Runs DeepTMHMM via pybiolib (requires BioLib authentication).

    Emits finished(list) on success or error(str) on failure.
    The caller is responsible for falling back to the local TMHMM 2.0 result
    if the user chooses not to proceed.
    """
    finished = Signal(list)   # list of TM helix dicts
    error    = Signal(str)

    def __init__(self, seq: str, parent=None):
        super().__init__(parent)
        self._seq = seq

    def _run_deeptmhmm(self) -> list:
        """Submit sequence to DeepTMHMM via pybiolib. Raises on any failure."""
        import biolib
        import tempfile
        import os

        deeptmhmm = biolib.load("DTU/DeepTMHMM")
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta_path = os.path.join(tmpdir, "query.fasta")
            with open(fasta_path, "w") as fh:
                fh.write(f">query\n{self._seq}\n")
            result = deeptmhmm.cli(args=["--fasta", fasta_path])
            out_dir = os.path.join(tmpdir, "out")
            os.makedirs(out_dir, exist_ok=True)
            result.save_files(out_dir)

            # Locate GFF3 — search recursively in case save_files uses subdirs
            gff3_path = None
            for root, _, files in os.walk(out_dir):
                for fname in files:
                    if fname.endswith(".gff3"):
                        gff3_path = os.path.join(root, fname)
                        break
                if gff3_path:
                    break

            if not gff3_path or not os.path.exists(gff3_path):
                saved = []
                for root, _, files in os.walk(out_dir):
                    saved.extend(os.path.relpath(os.path.join(root, f), out_dir) for f in files)
                raise RuntimeError(
                    "DeepTMHMM returned no GFF3 output.\n"
                    f"Files saved: {saved or ['(none)']}\n"
                    "This usually means BioLib authentication failed. "
                    "Run: python -m biolib login"
                )

            gff3_content = open(gff3_path).read()

        helices = []
        for line in gff3_content.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) >= 9 and "TMhelix" in parts[2]:
                start = int(parts[3]) - 1   # GFF3 is 1-based
                end   = int(parts[4]) - 1
                helices.append({
                    "start": start,
                    "end": end,
                    "score": 1.0,
                    "orientation": "unknown",
                    "source": "DeepTMHMM",
                })
        if not helices:
            raise RuntimeError(
                "DeepTMHMM ran successfully but predicted 0 TM helices.\n"
                "If this is unexpected, verify the GFF3 output manually."
            )
        return helices

    def run(self):
        try:
            self.finished.emit(self._run_deeptmhmm())
        except ImportError:
            self.error.emit(
                "pybiolib is not installed.\n"
                "Install it with:  pip install pybiolib\n"
                "Then authenticate: python -m biolib login"
            )
        except Exception as exc:
            self.error.emit(str(exc))


class AlphaMissenseWorker(QThread):
    """Fetches AlphaMissense scores via EBI API (requires internet + UniProt ID)."""
    finished = Signal(dict)
    error    = Signal(str)

    def __init__(self, uniprot_id: str, parent=None):
        super().__init__(parent)
        self._uid = uniprot_id

    def run(self):
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
        try:
            fields = (
                "ft_signal,ft_transit,ft_transmem,ft_intramem,ft_coiled,ft_dna_bind,"
                "ft_act_site,ft_binding,ft_propep,ft_repeat,ft_motif,"
                "ft_region,ft_compbias,ft_disulfid,ft_mod_res,ft_carbohyd,ft_lipid,ft_zn_fing"
            )
            url = (
                f"https://rest.uniprot.org/uniprotkb/{self._acc}.json"
                f"?fields=accession,{fields}"
            )
            with urllib.request.urlopen(url, timeout=20) as resp:
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


class SignalP6Worker(QThread):
    """Runs SignalP 6.0 via pybiolib (requires BioLib authentication).

    Emits finished(dict) on success or error(str) on failure.
    The dict contains: cleavage_site (int, 1-based), probability (float),
    signal_type (str), source (str).
    """
    finished = Signal(dict)
    error    = Signal(str)

    def __init__(self, seq: str, organism: str = "eukarya", parent=None):
        super().__init__(parent)
        self._seq = seq
        self._organism = organism

    def _run_signalp6(self) -> dict:
        """Submit sequence to SignalP 6.0 via pybiolib. Raises on any failure."""
        import biolib
        import tempfile
        import os

        signalp = biolib.load("DTU/SignalP_6")
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta_path = os.path.join(tmpdir, "query.fasta")
            with open(fasta_path, "w") as fh:
                fh.write(f">query\n{self._seq}\n")
            result = signalp.cli(args=[
                "--fastafile", fasta_path,
                "--organism", self._organism,
                "--output_dir", os.path.join(tmpdir, "out"),
                "--format", "txt",
            ])
            out_dir = os.path.join(tmpdir, "out")
            os.makedirs(out_dir, exist_ok=True)
            result.save_files(out_dir)

            # Locate summary output file (prediction_results.txt or similar)
            txt_path = None
            for root, _, files in os.walk(out_dir):
                for fname in files:
                    if fname.endswith(".txt") or fname.endswith("_summary.signalp5"):
                        txt_path = os.path.join(root, fname)
                        break
                if txt_path:
                    break

            if not txt_path or not os.path.exists(txt_path):
                saved = []
                for root, _, files in os.walk(out_dir):
                    saved.extend(os.path.relpath(os.path.join(root, f), out_dir) for f in files)
                raise RuntimeError(
                    "SignalP 6.0 returned no output file.\n"
                    f"Files saved: {saved or ['(none)']}\n"
                    "This usually means BioLib authentication failed. "
                    "Run: python -m biolib login"
                )

            content = open(txt_path).read()

        # Parse SignalP 6.0 output
        # Format: # ID  Prediction  SP(Sec/SPI)  TAT(Tat/SPI)  LIPO(Sec/SPII)  OTHER  CS Position
        result_dict = {"source": "SignalP 6.0", "cleavage_site": -1,
                       "probability": 0.0, "signal_type": "OTHER"}
        for line in content.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 7:
                prediction = parts[1]
                result_dict["signal_type"] = prediction
                # CS Position column: "CS pos: X-Y, prob: Z"
                cs_text = " ".join(parts[6:])
                import re
                m = re.search(r"(\d+)-\d+", cs_text)
                if m:
                    result_dict["cleavage_site"] = int(m.group(1))
                mp = re.search(r"prob:\s*([\d.]+)", cs_text)
                if mp:
                    result_dict["probability"] = float(mp.group(1))
                # SP probability is in column index 2
                if prediction not in ("OTHER",) and len(parts) > 2:
                    try:
                        result_dict["probability"] = float(parts[2])
                    except ValueError:
                        pass
                break

        return result_dict

    def run(self):
        try:
            self.finished.emit(self._run_signalp6())
        except ImportError:
            self.error.emit(
                "pybiolib is not installed.\n"
                "Install it with:  pip install pybiolib\n"
                "Then authenticate: python -m biolib login"
            )
        except Exception as exc:
            self.error.emit(str(exc))
