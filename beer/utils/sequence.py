"""Sequence utility functions."""
from __future__ import annotations
import re
from beer.constants import VALID_AMINO_ACIDS

# UniProt accession: 6–10 alphanumeric chars, optionally followed by isoform "-N"
_UNIPROT_RE = re.compile(r"^[A-Z0-9]{6,10}(-\d+)?$", re.IGNORECASE)
# PDB entry ID: digit followed by exactly 3 alphanumeric chars (classic 4-char IDs)
_PDB_RE = re.compile(r"^[0-9][A-Z0-9]{3}$", re.IGNORECASE)


def valid_uniprot(acc: str) -> bool:
    """Return True if *acc* matches the UniProt accession format."""
    return bool(_UNIPROT_RE.match(acc.strip()))


def valid_pdb(acc: str) -> bool:
    """Return True if *acc* matches the classic 4-character PDB entry ID format."""
    return bool(_PDB_RE.match(acc.strip()))


def clean_sequence(seq: str) -> str:
    return seq.strip().replace(" ", "").upper()


def is_valid_protein(seq: str) -> bool:
    return all(aa in VALID_AMINO_ACIDS for aa in seq)


def format_sequence_block(seq: str, name: str = "", width: int = 60, group: int = 10) -> str:
    """Format sequence in UniProt style: groups of 10 residues, position numbers every 10."""
    lines = []
    if name:
        lines.append(f">{name}")
    for i in range(0, len(seq), width):
        chunk = seq[i:i + width]
        groups = "  ".join(chunk[j:j + group] for j in range(0, len(chunk), group))
        pos = str(i + 1).rjust(6)
        lines.append(f"{pos}  {groups}")
    return "\n".join(lines)
