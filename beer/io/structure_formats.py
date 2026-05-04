"""Structure format converters: PDB string → mmCIF / GRO / XYZ."""
from __future__ import annotations
from io import StringIO


def _parse_pdb(pdb_str: str, quiet: bool = True):
    """Parse a PDB string and return a BioPython Structure, raising ValueError on failure."""
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=quiet)
    struct = parser.get_structure("mol", StringIO(pdb_str))
    if struct is None or len(list(struct.get_atoms())) == 0:
        raise ValueError(
            "PDB parsing produced an empty structure — the input may be malformed or incomplete."
        )
    return struct


def pdb_to_mmcif(pdb_str: str) -> str:
    """Convert a PDB string to mmCIF format using BioPython MMCIFIO."""
    try:
        from Bio.PDB import MMCIFIO
        struct = _parse_pdb(pdb_str)
        io = MMCIFIO()
        io.set_structure(struct)
        buf = StringIO()
        io.save(buf)
        return buf.getvalue()
    except Exception as exc:
        raise ValueError(f"PDB → mmCIF conversion failed: {exc}") from exc


def pdb_to_gro(pdb_str: str) -> str:
    """Convert a PDB string to GROMACS GRO format.

    Coordinates are converted from Å to nm.  No velocities are written.
    A default cubic box of 10 × 10 × 10 nm is appended.
    """
    try:
        struct = _parse_pdb(pdb_str)

        atoms = []
        for model in struct:
            for chain in model:
                for res in chain:
                    res_name = res.get_resname().strip()
                    res_seq  = res.get_id()[1]
                    for atom in res:
                        coord = atom.get_vector()
                        atoms.append({
                            'res_seq':   res_seq,
                            'res_name':  res_name,
                            'atom_name': atom.get_name().strip(),
                            'x': coord[0] / 10.0,
                            'y': coord[1] / 10.0,
                            'z': coord[2] / 10.0,
                        })
            break  # first model only

        lines = ["BEER structure export — GRO format", f"{len(atoms)}"]
        for i, a in enumerate(atoms, 1):
            lines.append(
                f"{a['res_seq']:5d}{a['res_name']:<5s}{a['atom_name']:>5s}{i:5d}"
                f"{a['x']:8.3f}{a['y']:8.3f}{a['z']:8.3f}"
            )
        lines.append("  10.00000  10.00000  10.00000")
        return "\n".join(lines) + "\n"
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"PDB → GRO conversion failed: {exc}") from exc


def pdb_to_xyz(pdb_str: str) -> str:
    """Convert a PDB string to XYZ format (Ångström coordinates).

    Element symbols are taken from BioPython's atom.element attribute;
    if absent, they are inferred from the first non-numeric character of
    the atom name.
    """
    try:
        struct = _parse_pdb(pdb_str)

        atoms = []
        for model in struct:
            for chain in model:
                for res in chain:
                    for atom in res:
                        element = (atom.element or "").strip().capitalize()
                        if not element:
                            name = atom.get_name().strip().lstrip("0123456789")
                            element = name[0].upper() if name else "X"
                        coord = atom.get_vector()
                        atoms.append((element, coord[0], coord[1], coord[2]))
            break  # first model only

        lines = [
            str(len(atoms)),
            "BEER structure export — XYZ format (Angstrom)",
        ]
        for elem, x, y, z in atoms:
            lines.append(f"{elem:<4s}  {x:12.6f}  {y:12.6f}  {z:12.6f}")
        return "\n".join(lines) + "\n"
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"PDB → XYZ conversion failed: {exc}") from exc
