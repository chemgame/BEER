"""Sequence utility functions."""
from __future__ import annotations
from beer.constants import VALID_AMINO_ACIDS


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
