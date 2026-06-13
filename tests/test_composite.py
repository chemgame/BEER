"""Fix-PDB engine: strip heteroatoms, fold-sequence reconstruction, SEQRES."""
from beer.analysis.composite_structure import (
    strip_to_protein, build_fold_sequence, parse_seqres,
)


def _atom(serial, name, res, ch, resseq, x, y, z, rec="ATOM  "):
    return (f'{rec}{serial:>5} {name:^4}{" "}{res:>3} {ch}{resseq:>4}    '
            f'{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00')


_PDB = "\n".join([
    "SEQRES   1 A    5  MET ALA GLY VAL LYS",
    _atom(1, "CA", "MET", "A", 1, 0, 0, 0),
    _atom(2, "CA", "ALA", "A", 2, 1, 0, 0),
    # gap at residue 3
    _atom(3, "CA", "VAL", "A", 4, 3, 0, 0),
    _atom(4, "CA", "MSE", "A", 5, 4, 0, 0, rec="HETATM"),   # modified residue: keep
    _atom(5, "O", "HOH", "A", 99, 9, 9, 9, rec="HETATM"),   # water: drop
    _atom(6, "CA", "CA", "A", 200, 8, 8, 8, rec="HETATM"),  # ion: drop
])


def test_strip_removes_water_and_ion_keeps_protein():
    out = strip_to_protein(_PDB)
    assert "HOH" not in out
    assert "CA  A 200" not in out        # calcium ion removed
    assert "MSE" in out                  # modified residue kept
    assert "SEQRES" in out               # non-coordinate lines preserved


def test_build_fold_sequence_spans_gaps():
    seq = build_fold_sequence(strip_to_protein(_PDB))
    # Residue range 1..5 → 5 chars, with a placeholder at the unresolved residue 3.
    assert len(seq) == 5
    assert seq[0] == "M" and seq[1] == "A"     # resolved
    assert seq[3] == "V"                        # resolved (resseq 4)


def test_parse_seqres():
    assert parse_seqres(_PDB, "A") == "MAGVK"
