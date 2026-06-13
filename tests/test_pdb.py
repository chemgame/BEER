"""Canonical PDB parsers: insertion codes, calcium-ion guard, HELIX/SHEET."""
from beer.utils.pdb import parse_ca_atoms, parse_helix_sheet_records


def _atom(serial, name, res, ch, resseq, icode, x, y, z, occ=1.0, b=0.0, rec="ATOM  "):
    return (f'{rec}{serial:>5} {name:^4}{" "}{res:>3} {ch}{resseq:>4}{icode}   '
            f'{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{b:6.2f}')


def _ss_rec(kind, ch, start, end):
    """Build a HELIX/SHEET record at the exact columns the parser reads."""
    b = [" "] * 40
    b[0:6] = list(f"{kind:<6}")
    if kind == "HELIX":
        b[19] = ch
        b[21:25] = list(f"{start:>4}")
    else:  # SHEET
        b[21] = ch
        b[22:26] = list(f"{start:>4}")
    b[33:37] = list(f"{end:>4}")
    return "".join(b)


_PDB = "\n".join([
    _ss_rec("HELIX", "A", 2, 4),
    _ss_rec("SHEET", "A", 6, 7),
    _atom(1, "CA", "MET", "A", 1, " ", 0, 0, 0, b=11.0),
    _atom(2, "CA", "ALA", "A", 52, " ", 1, 0, 0, b=22.0),
    _atom(3, "CA", "GLY", "A", 52, "A", 2, 0, 0, b=33.0),
    _atom(4, "CA", "MSE", "A", 3, " ", 3, 0, 0, b=44.0, rec="HETATM"),
    _atom(5, "CA", "CA", "A", 200, " ", 9, 9, 9, b=99.0, rec="HETATM"),
])


def test_insertion_codes_kept_distinct():
    recs = parse_ca_atoms(_PDB)
    keys = {(r["resseq"], r["icode"]) for r in recs}
    assert (52, " ") in keys and (52, "A") in keys


def test_calcium_ion_excluded():
    recs = parse_ca_atoms(_PDB)
    assert not any(r["resname"] == "CA" for r in recs)


def test_modified_residue_mse_kept():
    recs = parse_ca_atoms(_PDB)
    assert any(r["resname"] == "MSE" for r in recs)


def test_chain_filter_and_bfactor():
    recs = parse_ca_atoms(_PDB, chain="A")
    assert all(r["chain"] == "A" for r in recs)
    met = next(r for r in recs if r["resname"] == "MET")
    assert abs(met["bfac"] - 11.0) < 1e-6


def test_helix_sheet_records():
    helix, sheet = parse_helix_sheet_records(_PDB)
    assert ("A", 2) in helix and ("A", 4) in helix
    assert ("A", 6) in sheet and ("A", 7) in sheet
