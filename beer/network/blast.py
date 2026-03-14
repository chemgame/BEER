"""BEER network/blast.py — NCBI BLAST search worker.

Runs a remote NCBI blastp search for a protein sequence and returns the
top hits.  Requires Biopython (``pip install biopython``).
"""

import json
from PyQt5.QtCore import QThread, pyqtSignal


class BlastWorker(QThread):
    """Run NCBI blastp and return top hits."""

    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, seq: str, database: str = "nr", hitlist_size: int = 20):
        super().__init__()
        self.seq = seq
        self.database = database
        self.hitlist_size = hitlist_size

    def run(self):
        try:
            from Bio.Blast import NCBIWWW, NCBIXML

            self.progress.emit("Submitting BLAST search (this may take 1–3 min)…")
            result_handle = NCBIWWW.qblast(
                "blastp",
                self.database,
                self.seq,
                hitlist_size=self.hitlist_size,
            )
            self.progress.emit("Parsing BLAST results…")
            blast_record = NCBIXML.read(result_handle)
            hits = []
            for aln in blast_record.alignments[: self.hitlist_size]:
                hsp = aln.hsps[0]
                hits.append(
                    {
                        "accession": aln.accession,
                        "title": aln.title[:100],
                        "length": aln.length,
                        "score": hsp.score,
                        "e_value": hsp.expect,
                        "identity": hsp.identities / hsp.align_length * 100,
                        "subject": hsp.sbjct.replace("-", ""),
                    }
                )
            self.finished.emit(hits)
        except ImportError:
            self.error.emit("Bio.Blast not available — install biopython.")
        except Exception as exc:
            self.error.emit(f"BLAST failed: {exc}")
