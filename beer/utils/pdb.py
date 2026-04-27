"""PDB/structural file utilities."""
from __future__ import annotations
import math
import os
from io import StringIO

import numpy as np
from Bio.PDB import PDBParser, MMCIFParser, PDBIO, Select as PDBSelect, PPBuilder
from Bio.PDB.Polypeptide import is_aa


def extract_phi_psi(pdb_str: str) -> list:
    """Extract phi/psi dihedral angles for each residue from a PDB string.

    Returns list of dicts: {phi, psi, resname, resnum, chain_id, ss}
    ss: 'H' (helix) or 'E' (sheet) or 'C' (coil) - estimated from phi/psi values.
    Uses Biopython's PPBuilder.
    """
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("x", StringIO(pdb_str))
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


# Alias used by legacy beer.py code
_extract_phi_psi = extract_phi_psi


def compute_ca_distance_matrix(pdb_str: str) -> np.ndarray:
    """Return symmetric Ca pairwise distance matrix (Angstrom) from a PDB string.

    Only the first chain of the first model is used so that the matrix length
    matches the single-chain pLDDT array returned by extract_plddt_from_pdb().
    """
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("af", StringIO(pdb_str))
    coords = []
    for model in struct:
        first_chain = next(iter(model), None)
        if first_chain is None:
            break
        for res in first_chain:
            if is_aa(res, standard=True) and res.has_id("CA"):
                coords.append(res["CA"].get_vector().get_array())
        break
    if not coords:
        return np.array([])
    ca = np.array(coords, dtype=float)
    diff = ca[:, np.newaxis, :] - ca[np.newaxis, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def extract_plddt_from_pdb(pdb_str: str) -> list:
    """Extract per-residue pLDDT / B-factor confidence scores from a PDB string.

    Only the first chain of the first model is read.  This matches the single-chain
    output produced by AlphaFold and avoids silently concatenating scores from
    multiple chains when a multi-chain PDB is passed.
    """
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("af", StringIO(pdb_str))
    scores = []
    for model in struct:
        first_chain = next(iter(model), None)
        if first_chain is None:
            break
        for res in first_chain:
            if is_aa(res, standard=True):
                for atom in res:
                    if atom.get_name() == "CA":
                        scores.append(atom.get_bfactor())
                        break
        break
    return scores


def import_pdb_sequence(pdb_path_or_str: str) -> dict:
    """Extract single-letter amino acid sequences from a PDB file or string.

    Accepts either a file path or a raw PDB string.  Returns a dict mapping
    chain IDs to their single-letter amino acid sequences, e.g.
    ``{'A': 'MKTAY...', 'B': 'GSHM...'}``.  Only standard amino acids are
    included.  Chains that yield an empty sequence are omitted.
    """
    from Bio.SeqUtils import seq1
    # Decide whether we got a file path or a raw PDB string.
    import os
    if os.path.isfile(pdb_path_or_str):
        with open(pdb_path_or_str, "r") as fh:
            pdb_str = fh.read()
    else:
        pdb_str = pdb_path_or_str
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("x", StringIO(pdb_str))
    chains = {}
    for model in struct:
        for chain in model:
            seq_parts = []
            for res in chain:
                if is_aa(res, standard=True):
                    seq_parts.append(seq1(res.get_resname()))
            if seq_parts:
                chains[chain.id] = "".join(seq_parts)
        break  # first model only
    return chains


def extract_chain_structures(pdb_str: str) -> dict:
    """Return per-chain structure dicts keyed by chain ID.

    Each value is ``{"pdb_str": str, "plddt": list, "dist_matrix": ndarray}``.
    Uses PDBIO to write each chain individually so that pLDDT extraction and
    distance-matrix computation always operate on a single, correctly-sized chain.
    """
    class _ChainSelect(PDBSelect):
        def __init__(self, cid):
            self._cid = cid

        def accept_chain(self, chain):
            return chain.id == self._cid

    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("x", StringIO(pdb_str))
    result = {}
    model = next(struct.get_models())
    io = PDBIO()
    io.set_structure(struct)
    for chain in model:
        buf = StringIO()
        io.save(buf, _ChainSelect(chain.id))
        chain_pdb = buf.getvalue()
        if not chain_pdb.strip():
            continue
        plddt = extract_plddt_from_pdb(chain_pdb)
        dm = compute_ca_distance_matrix(chain_pdb)
        result[chain.id] = {"pdb_str": chain_pdb, "plddt": plddt, "dist_matrix": dm}
    return result


def import_mmcif_sequence(path_or_str: str) -> dict:
    """Extract chain sequences from an mmCIF file path or string.

    Returns ``{chain_id: sequence}`` for all chains with standard amino acids,
    identical in shape to :func:`import_pdb_sequence`.
    """
    from Bio.SeqUtils import seq1
    if os.path.isfile(path_or_str):
        parser = MMCIFParser(QUIET=True)
        struct = parser.get_structure("x", path_or_str)
    else:
        import tempfile, uuid
        tmp = os.path.join(tempfile.gettempdir(), f"beer_cif_{uuid.uuid4().hex}.cif")
        try:
            with open(tmp, "w") as fh:
                fh.write(path_or_str)
            parser = MMCIFParser(QUIET=True)
            struct = parser.get_structure("x", tmp)
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    chains = {}
    for model in struct:
        for chain in model:
            seq_parts = []
            for res in chain:
                if is_aa(res, standard=True):
                    seq_parts.append(seq1(res.get_resname()))
            if seq_parts:
                chains[chain.id] = "".join(seq_parts)
        break  # first model only
    return chains


def extract_chain_structures_mmcif(mmcif_str: str) -> dict:
    """Return per-chain structure dicts from an mmCIF string.

    Parses via :class:`Bio.PDB.MMCIFParser`, then writes each chain as a
    temporary PDB string so the existing pLDDT/distance-matrix helpers work
    without modification.  Returns ``{chain_id: {"pdb_str", "plddt", "dist_matrix"}}``.
    """
    import tempfile, uuid

    class _ChainSelect(PDBSelect):
        def __init__(self, cid):
            self._cid = cid
        def accept_chain(self, chain):
            return chain.id == self._cid

    tmp = os.path.join(tempfile.gettempdir(), f"beer_cif_{uuid.uuid4().hex}.cif")
    try:
        with open(tmp, "w") as fh:
            fh.write(mmcif_str)
        parser = MMCIFParser(QUIET=True)
        struct = parser.get_structure("x", tmp)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass

    result = {}
    model = next(struct.get_models())
    io = PDBIO()
    io.set_structure(struct)
    for chain in model:
        buf = StringIO()
        io.save(buf, _ChainSelect(chain.id))
        chain_pdb = buf.getvalue()
        if not chain_pdb.strip():
            continue
        plddt = extract_plddt_from_pdb(chain_pdb)
        dm = compute_ca_distance_matrix(chain_pdb)
        result[chain.id] = {"pdb_str": chain_pdb, "plddt": plddt, "dist_matrix": dm}
    return result
