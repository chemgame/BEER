"""PDB file import and structural analysis utilities."""
import math
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1


def import_pdb_sequence(file_name: str) -> dict:
    """Extract one-letter sequences for each chain in a PDB file.

    Returns dict mapping chain_id -> sequence string.
    """
    try:
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("pdb", file_name)
        chains = {}
        model = next(struct.get_models())
        for chain in model:
            seq = ""
            for res in chain:
                if is_aa(res, standard=True):
                    try:
                        seq += seq1(res.get_resname())
                    except KeyError:
                        continue
            if seq:
                chains[chain.id] = seq
        return chains
    except Exception as e:
        raise RuntimeError(f"PDB parse error: {e}") from e


def compute_ca_distance_matrix(pdb_str: str) -> np.ndarray:
    """Compute Cα pairwise distance matrix from PDB string.

    Returns square numpy array of distances in Angstroms.
    """
    import io
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import is_aa
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("x", io.StringIO(pdb_str))
    ca_coords = []
    for model in struct:
        for chain in model:
            for res in chain:
                if is_aa(res, standard=True) and "CA" in res:
                    ca_coords.append(res["CA"].get_vector().get_array())
        break  # first model only
    if not ca_coords:
        return np.array([[]])
    coords = np.array(ca_coords)
    n = len(coords)
    diff = coords[:, None, :] - coords[None, :, :]  # n x n x 3
    return np.sqrt((diff ** 2).sum(axis=-1))


def extract_plddt_from_pdb(pdb_str: str) -> list:
    """Extract per-residue pLDDT scores from AlphaFold PDB B-factor column.

    Returns list of floats.
    """
    import io
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import is_aa
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("x", io.StringIO(pdb_str))
    plddt = []
    for model in struct:
        for chain in model:
            for res in chain:
                if is_aa(res, standard=True):
                    for atom in res.get_atoms():
                        plddt.append(atom.get_bfactor())
                        break  # one atom per residue is enough
        break
    return plddt


def extract_phi_psi(pdb_str: str) -> list:
    """Extract φ/ψ dihedral angles for each residue from a PDB string.

    Returns list of dicts: {phi, psi, resname, resnum, chain_id, ss}
    ss: 'H' (helix) or 'E' (sheet) or 'C' (coil) - estimated from phi/psi values.
    Uses Biopython's PPBuilder.
    """
    import io
    from Bio.PDB import PDBParser, PPBuilder
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("x", io.StringIO(pdb_str))
    ppb = PPBuilder()
    result = []
    for model in struct:
        for pp in ppb.build_peptides(model):
            phi_psi = pp.get_phi_psi_list()
            for i, (residue, (phi, psi)) in enumerate(zip(pp, phi_psi)):
                if phi is None or psi is None:
                    continue
                phi_deg = math.degrees(phi)
                psi_deg = math.degrees(psi)
                # Estimate secondary structure from dihedral angles
                if -160 <= phi_deg <= -45 and -60 <= psi_deg <= 50:
                    ss = 'H'
                elif phi_deg <= -90 and (psi_deg >= 90 or psi_deg <= -150):
                    ss = 'E'
                else:
                    ss = 'C'
                result.append({
                    "phi": phi_deg,
                    "psi": psi_deg,
                    "resname": residue.get_resname(),
                    "resnum": residue.get_id()[1],
                    "chain_id": residue.get_parent().get_id(),
                    "ss": ss,
                })
        break  # first model only
    return result
