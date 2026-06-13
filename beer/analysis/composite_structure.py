"""
Composite structure builder: experimental PDB + AlphaFold gap-filling.

Algorithm
---------
1. Extract Cα traces from both structures (primary chain only).
2. Needleman-Wunsch sequence alignment → residue correspondence.
3. Kabsch SVD superimposition: AF → experimental coordinate frame.
4. Transplant all AF atoms for gap positions (transformed coordinates).
5. Merge PDB lines; renumber atoms and residues sequentially.
6. Check Cα–Cα distances at every gap junction.

Composite occupancy convention
-------------------------------
  1.00  →  experimental (ground-truth coordinates)
  0.00  →  AlphaFold gap-fill (predicted)

This lets downstream tools (and our own JS colorfunc) distinguish sources
without an extra data channel.
"""
from __future__ import annotations

import math as _math
from dataclasses import dataclass

import numpy as _np

_AA3TO1: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "C", "PYL": "K", "HSD": "H", "HSE": "H",
    "HSP": "H", "HIE": "H", "HID": "H", "HIP": "H", "CYX": "C",
}

_JUNCTION_WARN_DIST = 4.5   # Å — Cα–Cα > this triggers a warning (ideal ~3.8 Å)
_JUNCTION_CRIT_DIST = 6.0   # Å — > this is a critical warning

# pLDDT thresholds for AF gap-fill quality
PLDDT_RELIABLE  = 70.0  # above this: confident prediction → show as "predicted"
# at/below 70: low confidence → included but flagged as "low_confidence"
# (planned exclusion of very-low-confidence gap residues is part of the deferred
#  Fix-PDB junction-connectivity work, not yet implemented)


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class ResidueInfo:
    seq_pos: int            # 0-based canonical index
    resi_composite: int     # 1-based residue number in composite PDB
    resname: str
    source: str             # 'experimental' | 'predicted' | 'low_confidence'
    plddt: float = 0.0      # AF pLDDT (0 for experimental residues)


@dataclass
class CompositeResult:
    pdb_str: str
    residues: list           # list[ResidueInfo]
    n_experimental: int
    n_predicted: int         # AF fill with pLDDT ≥ 70
    n_low_confidence: int    # AF fill with pLDDT < 70 (included but flagged)
    n_gaps: int              # number of separate gap regions
    junction_warnings: list  # list[str]
    rmsd_fit: float          # Cα RMSD of the AF→exp Kabsch fit


# ── PDB atom parsing ─────────────────────────────────────────────────────────

def _parse_atoms(pdb_str: str) -> list[dict]:
    atoms: list[dict] = []
    for line in pdb_str.splitlines():
        if not (line.startswith("ATOM  ") or line.startswith("HETATM")):
            continue
        if len(line) < 54:
            continue
        try:
            atoms.append({
                "line":    line,
                "record":  line[:6].rstrip(),
                "serial":  int(line[6:11]),
                "name":    line[12:16],
                "altloc":  line[16],
                "resname": line[17:20].strip(),
                "chain":   line[21],
                "resseq":  int(line[22:26]),
                "icode":   line[26],
                "x":       float(line[30:38]),
                "y":       float(line[38:46]),
                "z":       float(line[46:54]),
                "occ":     float(line[54:60]) if len(line) > 59 else 1.0,
                "bfac":    float(line[60:66]) if len(line) > 65 else 0.0,
            })
        except (ValueError, IndexError):
            continue
    return atoms


def parse_seqres(pdb_str: str, chain: str | None = None) -> str:
    """Full construct sequence from SEQRES records (preferred over ATOM records).

    SEQRES lists the complete polymer sequence including residues that are
    unresolved (missing) in the ATOM records, which is exactly what gap-filling
    needs to fold from. Returns the one-letter sequence for *chain* (or the
    SEQRES chain with the most residues when *chain* is None). Returns "" when
    no SEQRES records are present.
    """
    by_chain: dict[str, list[str]] = {}
    for line in pdb_str.splitlines():
        if not line.startswith("SEQRES"):
            continue
        # Columns: 12 = chain ID, 20+ = residue names in groups of 4 chars.
        cid = line[11] if len(line) > 11 else " "
        names = line[19:].split()
        by_chain.setdefault(cid, []).extend(names)
    if not by_chain:
        return ""
    if chain is None or chain not in by_chain:
        chain = max(by_chain, key=lambda c: len(by_chain[c]))
    return "".join(_AA3TO1.get(r, "X") for r in by_chain[chain])


def primary_seqres_chain(pdb_str: str) -> str | None:
    """Chain ID of the SEQRES chain with the most residues (None if no SEQRES)."""
    by_chain: dict[str, int] = {}
    for line in pdb_str.splitlines():
        if not line.startswith("SEQRES"):
            continue
        cid = line[11] if len(line) > 11 else " "
        names = line[19:].split()
        by_chain[cid] = by_chain.get(cid, 0) + len(names)
    return max(by_chain, key=by_chain.__getitem__) if by_chain else None


def strip_to_protein(pdb_str: str) -> str:
    """Remove water, ions, and non-polymer ligands, keeping only protein atoms.

    Keeps ATOM records and HETATM/ANISOU records whose residue name is a known
    (possibly modified) amino acid — e.g. selenomethionine ``MSE``, which is a
    genuine backbone residue and must NOT be dropped (stripping it would punch an
    artificial gap into the chain). Drops water (``HOH``/``DOD``), monatomic ions
    — notably the Ca²⁺ ion, whose residue/atom name ``CA`` would otherwise be
    mis-read as a backbone Cα and corrupt the gap alignment — and small-molecule
    ligands. All non-coordinate lines (SEQRES, CRYST1, headers, TER) are kept so
    the ESMFold2 full-sequence path still has the complete construct sequence.
    """
    kept: list[str] = []
    for line in pdb_str.splitlines():
        if line.startswith("ATOM"):
            kept.append(line)
        elif line.startswith(("HETATM", "ANISOU")):
            resname = line[17:20].strip() if len(line) >= 20 else ""
            if resname in _AA3TO1:
                kept.append(line)
            # else: water / ion / ligand → drop
        else:
            kept.append(line)
    return "\n".join(kept) + "\n"


def build_fold_sequence(exp_pdb: str, gap_placeholder: str = "A") -> str:
    """Sequence for folding a gap-fill model, derived from the experimental chain.

    Spans the resolved residue-NUMBER range of the primary chain: each resolved
    residue contributes its own identity, and every missing (unresolved) position
    in that range contributes a placeholder residue (default alanine). This
    guarantees the predicted model contains a residue at every gap, so
    :func:`build_composite` has coordinates to transplant for each missing
    position — matching what an AlphaFold full-length model provides.

    Why not SEQRES/FASTA? Those collapse modified residues to a single ``X``
    (e.g. the GFP chromophore ``CRO`` spans residue numbers 65–67 but appears as
    one ``X`` in SEQRES), so folding them leaves nothing to fill a multi-residue
    gap. Gap residues are modelled as placeholders, so only their backbone
    geometry is meaningful — for exact gap identities use the AlphaFold path.

    Returns ``""`` when the chain has no resolved Cα atoms.
    """
    ph = gap_placeholder if gap_placeholder in _AA3TO1.values() else "A"
    atoms = _parse_atoms(exp_pdb)
    if not atoms:
        return ""
    chain = _primary_chain(atoms)
    cas = _ca_list(atoms, chain)
    if not cas:
        return ""
    resolved = {a["resseq"]: _AA3TO1.get(a["resname"], ph) for a in cas}
    lo, hi = min(resolved), max(resolved)
    return "".join(resolved.get(n, ph) for n in range(lo, hi + 1))


def _primary_chain(atoms: list[dict]) -> str:
    """Chain ID of the chain with the most polymer (ATOM or polymer-HETATM) records."""
    counts: dict[str, int] = {}
    for a in atoms:
        # Include standard ATOM records and HETATM records that are known polymer AAs
        is_polymer = a["record"] == "ATOM" or (
            a["record"] == "HETATM" and a["resname"] in _AA3TO1
        )
        if is_polymer:
            counts[a["chain"]] = counts.get(a["chain"], 0) + 1
    return max(counts, key=counts.__getitem__) if counts else "A"


def _ca_list(atoms: list[dict], chain: str) -> list[dict]:
    """One CA entry per residue on *chain*, sorted by (resseq, icode)."""
    seen: dict[tuple, dict] = {}
    for a in atoms:
        if a["chain"] != chain or a["name"].strip() != "CA":
            continue
        # Ignore non-polymer atoms named "CA" (a Ca²⁺ ion is HETATM with residue
        # name "CA"), which would otherwise be mis-counted as a backbone Cα and
        # inject a spurious residue into the alignment.
        if a["record"] != "ATOM" and a["resname"] not in _AA3TO1:
            continue
        key = (a["resseq"], a["icode"].strip())
        if key not in seen:
            seen[key] = a
    return sorted(seen.values(), key=lambda a: (a["resseq"], a["icode"]))


# ── Needleman-Wunsch (affine gap) ────────────────────────────────────────────

def _nw(seq_a: str, seq_b: str) -> list[tuple[int, int]]:
    """Semi-global alignment: seq_a (experimental fragment) fully anchored;
    seq_b (AF full-length) has free end-gaps so a fragment aligns without
    paying terminal-gap penalties for the AF N/C-terminal extensions."""
    na, nb = len(seq_a), len(seq_b)
    MATCH, MISMATCH, GOPEN, GEXT = 2, -1, -4, -1
    NEG = float("-inf")

    M = [[NEG] * (nb + 1) for _ in range(na + 1)]
    X = [[NEG] * (nb + 1) for _ in range(na + 1)]
    Y = [[NEG] * (nb + 1) for _ in range(na + 1)]
    M[0][0] = 0.0
    for i in range(1, na + 1): X[i][0] = GOPEN + (i - 1) * GEXT
    # Free leading gaps in seq_b (AF N-terminal extension before exp region)
    for j in range(1, nb + 1): Y[0][j] = 0.0

    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            s = MATCH if seq_a[i-1] == seq_b[j-1] else MISMATCH
            M[i][j] = s + max(M[i-1][j-1], X[i-1][j-1], Y[i-1][j-1])
            X[i][j] = max(M[i-1][j] + GOPEN, X[i-1][j] + GEXT,
                          Y[i-1][j] + GOPEN)
            Y[i][j] = max(M[i][j-1] + GOPEN, X[i][j-1] + GOPEN,
                          Y[i][j-1] + GEXT)

    # Free trailing gaps in seq_b: find the best ending column in last row
    best_j = max(range(nb + 1),
                 key=lambda jj: max(M[na][jj], X[na][jj], Y[na][jj]))
    i, j = na, best_j
    best = max(("M", M[i][j]), ("X", X[i][j]), ("Y", Y[i][j]),
               key=lambda t: t[1])
    state = best[0]

    pairs: list[tuple[int, int]] = []
    while i > 0 or j > 0:
        if state == "M":
            pairs.append((i - 1, j - 1))
            scores = [M[i-1][j-1], X[i-1][j-1], Y[i-1][j-1]]
            state = ["M", "X", "Y"][scores.index(max(scores))]
            i -= 1; j -= 1
        elif state == "X":
            best = max(M[i-1][j] + GOPEN, X[i-1][j] + GEXT, Y[i-1][j] + GOPEN)
            if M[i-1][j] + GOPEN == best:   state = "M"
            elif Y[i-1][j] + GOPEN == best: state = "Y"
            else:                            state = "X"
            i -= 1
        else:  # state == "Y"
            best = max(M[i][j-1] + GOPEN, X[i][j-1] + GOPEN, Y[i][j-1] + GEXT)
            if M[i][j-1] + GOPEN == best:   state = "M"
            elif X[i][j-1] + GOPEN == best: state = "X"
            else:                            state = "Y"
            j -= 1

    pairs.reverse()
    return pairs


# ── Kabsch superimposition ────────────────────────────────────────────────────

def _kabsch(mob: "_np.ndarray", ref: "_np.ndarray") -> tuple:
    """Return (R, t) such that mob @ R.T + t ≈ ref (least-squares SVD)."""
    mc, rc = mob.mean(0), ref.mean(0)
    mob_c = mob - mc
    ref_c = ref - rc
    if _np.linalg.matrix_rank(mob_c) < 3 or _np.linalg.matrix_rank(ref_c) < 3:
        raise ValueError("Degenerate atom set: cannot compute Kabsch rotation (collinear/identical points).")
    H = mob_c.T @ ref_c
    U, _, Vt = _np.linalg.svd(H)
    d = _np.sign(_np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ _np.diag([1.0, 1.0, d]) @ U.T
    t = rc - mc @ R.T
    return R, t


# ── PDB line rewriting ────────────────────────────────────────────────────────

def _rewrite_line(line: str, serial: int, chain: str,
                  resseq: int, occ: float, bfac: float,
                  x: float | None = None,
                  y: float | None = None,
                  z: float | None = None) -> str:
    """Rewrite serial, chain, resseq, icode(cleared), occ, bfac and optionally xyz."""
    if len(line) < 66:
        line = line.ljust(80)
    # Columns 30-53 hold xyz; update only when new coords supplied
    if x is not None:
        xyz_part = f"{x:8.3f}{y:8.3f}{z:8.3f}"
    else:
        xyz_part = line[30:54]
    return (
        f"{line[:6]}{serial:5d}"   # cols 1-11
        f"{line[11:21]}"           # space + atom name + altloc + resname + space
        f"{chain}{resseq:4d} "     # chain + resseq + icode (cleared to space)
        f"{line[27:30]}"           # 3 padding spaces
        f"{xyz_part}"              # x y z (24 chars)
        f"{occ:6.2f}{bfac:6.2f}"  # occ + bfac
        f"{line[66:]}"             # element, charge, etc.
    )


# ── Main entry point ──────────────────────────────────────────────────────────

def build_composite(exp_pdb: str, af_pdb: str,
                    fill_source: str = "AlphaFold") -> CompositeResult:
    """Build a composite structure: experimental coordinates + predicted gap-filling.

    Parameters
    ----------
    exp_pdb
        PDB string from the experimental structure (RCSB or similar).
        May have missing residues.
    af_pdb
        PDB string of the full-length predicted model for the same protein
        (AlphaFold or ESMFold2) — used to fill the missing-residue gaps.
    fill_source
        Human-readable label of the gap-fill source ("AlphaFold" or
        "ESMFold2"). Only affects provenance REMARKs in the output header.

    Returns
    -------
    CompositeResult
    """
    exp_atoms = _parse_atoms(exp_pdb)
    af_atoms  = _parse_atoms(af_pdb)

    if not exp_atoms:
        raise ValueError("No ATOM records found in the experimental PDB.")
    if not af_atoms:
        raise ValueError("No ATOM records found in the AlphaFold PDB.")

    exp_chain = _primary_chain(exp_atoms)
    af_chain  = _primary_chain(af_atoms)

    exp_ca = _ca_list(exp_atoms, exp_chain)
    af_ca  = _ca_list(af_atoms,  af_chain)

    if len(exp_ca) < 4:
        raise ValueError(
            f"Too few Cα atoms in experimental structure ({len(exp_ca)}).")
    if len(af_ca) < 4:
        raise ValueError(
            f"Too few Cα atoms in AlphaFold structure ({len(af_ca)}).")

    exp_seq = "".join(_AA3TO1.get(a["resname"], "X") for a in exp_ca)
    af_seq  = "".join(_AA3TO1.get(a["resname"], "X") for a in af_ca)

    pairs = _nw(exp_seq, af_seq)
    if len(pairs) < 4:
        raise ValueError(
            f"Sequences have too few aligned positions ({len(pairs)}) "
            "to build a reliable composite. Check that both structures "
            "are for the same protein.")

    # ── Kabsch: superimpose AF Cα onto experimental ───────────────────────────
    fit_exp = _np.array([[exp_ca[i]["x"], exp_ca[i]["y"], exp_ca[i]["z"]]
                          for i, _ in pairs])
    fit_af  = _np.array([[af_ca[j]["x"],  af_ca[j]["y"],  af_ca[j]["z"]]
                          for _, j in pairs])
    R, t = _kabsch(fit_af, fit_exp)

    rmsd = float(_np.sqrt(((fit_af @ R.T + t - fit_exp) ** 2).sum(1).mean()))

    # ── Identify gap positions ────────────────────────────────────────────────
    exp_covered_af = {j for _, j in pairs}
    gap_af_indices = set(range(len(af_ca))) - exp_covered_af

    # Map: AF Cα index → exp Cα index (None for gaps)
    af_to_exp: dict[int, int] = {j: i for i, j in pairs}

    # ── Pre-transform all AF atom coordinates ─────────────────────────────────
    af_xyz = _np.array([[a["x"], a["y"], a["z"]] for a in af_atoms])
    af_xyz_t = af_xyz @ R.T + t

    # Group AF atoms by (resseq, icode) for the primary chain, ATOM only
    af_res_to_atoms: dict[tuple, list] = {}
    for idx, a in enumerate(af_atoms):
        if a["chain"] != af_chain or a["record"] != "ATOM":
            continue
        key = (a["resseq"], a["icode"].strip())
        af_res_to_atoms.setdefault(key, []).append((idx, a))

    # Keys of CA-bearing residues on the experimental primary chain (includes polymer
    # HETATM like CRO chromophore — they appear in exp_ca and must be written inline).
    exp_ca_keys: set[tuple] = {
        (a["resseq"], a["icode"].strip())
        for a in exp_ca if a["chain"] == exp_chain
    }

    # Group experimental atoms by (resseq, icode) for the primary chain.
    # Include HETATM for CA-bearing residues (modified polymer residues like CRO).
    exp_res_to_atoms: dict[tuple, list] = {}
    for a in exp_atoms:
        if a["chain"] != exp_chain:
            continue
        key = (a["resseq"], a["icode"].strip())
        if a["record"] != "ATOM" and key not in exp_ca_keys:
            continue  # skip non-polymer HETATM (ligands, water)
        exp_res_to_atoms.setdefault(key, []).append(a)

    # HETATM passthrough: ligands/cofactors not already handled inline.
    # Exclude primary-chain CA-bearing residues (already written as polymer atoms).
    exp_hetatm = [
        a for a in exp_atoms
        if a["record"] == "HETATM"
        and (a["chain"] != exp_chain
             or (a["resseq"], a["icode"].strip()) not in exp_ca_keys)
    ]

    # ── Build composite ───────────────────────────────────────────────────────
    residues: list[ResidueInfo] = []
    composite_lines: list[str] = []
    serial = 1

    # Cα xyz for junction checking — populated as we go
    ca_xyz: dict[int, tuple] = {}   # canonical_resi → (x, y, z)

    for af_idx, af_ca_atom in enumerate(af_ca):
        canonical_resi = af_idx + 1
        af_key = (af_ca_atom["resseq"], af_ca_atom["icode"].strip())

        if af_idx in gap_af_indices:
            # ── AlphaFold gap residue ─────────────────────────────────────────
            plddt = af_ca_atom["bfac"]
            src = "predicted" if plddt >= PLDDT_RELIABLE else "low_confidence"
            entries = af_res_to_atoms.get(af_key, [])
            for atom_idx, a in entries:
                tx, ty, tz = af_xyz_t[atom_idx]
                newline = _rewrite_line(a["line"], serial, "A", canonical_resi,
                                        occ=0.00, bfac=plddt,
                                        x=tx, y=ty, z=tz)
                composite_lines.append(newline)
                serial += 1
                if a["name"].strip() == "CA":
                    ca_xyz[canonical_resi] = (tx, ty, tz)

            residues.append(ResidueInfo(
                seq_pos=af_idx, resi_composite=canonical_resi,
                resname=af_ca_atom["resname"], source=src,
                plddt=plddt,
            ))

        else:
            # ── Experimental residue ──────────────────────────────────────────
            exp_idx = af_to_exp[af_idx]
            exp_ca_atom = exp_ca[exp_idx]
            exp_key = (exp_ca_atom["resseq"], exp_ca_atom["icode"].strip())
            entries = exp_res_to_atoms.get(exp_key, [])
            for a in entries:
                try:
                    orig_bfac = float(a["line"][60:66])
                except (ValueError, IndexError):
                    orig_bfac = 0.0
                # Force HETATM polymer residues (CRO etc.) to ATOM in composite
                line = a["line"]
                if line.startswith("HETATM"):
                    line = "ATOM  " + line[6:]
                newline = _rewrite_line(line, serial, "A", canonical_resi,
                                        occ=1.00, bfac=orig_bfac)
                composite_lines.append(newline)
                serial += 1
                if a["name"].strip() == "CA":
                    ca_xyz[canonical_resi] = (a["x"], a["y"], a["z"])

            residues.append(ResidueInfo(
                seq_pos=af_idx, resi_composite=canonical_resi,
                resname=exp_ca_atom["resname"], source="experimental",
                plddt=0.0,
            ))

    # ── HETATM passthrough (renumber serials only) ────────────────────────────
    hetatm_lines: list[str] = []
    for a in exp_hetatm:
        line = a["line"]
        if len(line) < 66:
            line = line.ljust(80)
        hetatm_lines.append(f"{line[:6]}{serial:5d}{line[11:]}")
        serial += 1

    # ── Junction checks ───────────────────────────────────────────────────────
    junction_warnings: list[str] = []
    n_gaps = 0
    in_gap = False
    _gap_sources = {"predicted", "low_confidence"}

    for res in residues:
        resi = res.resi_composite
        if res.source in _gap_sources and not in_gap:
            in_gap = True
            n_gaps += 1
            _check_junction(resi - 1, resi, "left", ca_xyz, junction_warnings)
        elif res.source == "experimental" and in_gap:
            in_gap = False
            _check_junction(resi - 1, resi, "right", ca_xyz, junction_warnings)

    # ── Assemble final PDB ────────────────────────────────────────────────────
    n_exp     = sum(1 for r in residues if r.source == "experimental")
    n_pred    = sum(1 for r in residues if r.source == "predicted")
    n_lowconf = sum(1 for r in residues if r.source == "low_confidence")

    header = _build_header(exp_pdb, n_exp, n_pred + n_lowconf, len(pairs), rmsd,
                           fill_source)

    pdb_out = "\n".join(
        header
        + composite_lines
        + ["TER"]
        + hetatm_lines
        + ["END"]
    )

    return CompositeResult(
        pdb_str=pdb_out,
        residues=residues,
        n_experimental=n_exp,
        n_predicted=n_pred,
        n_low_confidence=n_lowconf,
        n_gaps=n_gaps,
        junction_warnings=junction_warnings,
        rmsd_fit=rmsd,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_junction(r1: int, r2: int, side: str,
                    ca_xyz: dict, warnings: list) -> None:
    if r1 not in ca_xyz or r2 not in ca_xyz:
        return
    p1, p2 = _np.array(ca_xyz[r1]), _np.array(ca_xyz[r2])
    dist = float(_np.linalg.norm(p2 - p1))
    if dist > _JUNCTION_WARN_DIST:
        sev = "CRITICAL" if dist > _JUNCTION_CRIT_DIST else "Warning"
        warnings.append(
            f"{sev}: {side} junction at residue {r2} — "
            f"Cα–Cα = {dist:.1f} Å (ideal ~3.8 Å)"
        )


def _build_header(exp_pdb: str, n_exp: int, n_pred: int,
                  n_pairs: int, rmsd: float,
                  fill_source: str = "AlphaFold") -> list[str]:
    """Carry over safe header records from experimental PDB + BEER REMARKs."""
    _KEEP = {"HEADER", "TITLE", "COMPND", "SOURCE", "KEYWDS",
              "EXPDTA", "AUTHOR", "CRYST1",
              "ORIGX1", "ORIGX2", "ORIGX3",
              "SCALE1", "SCALE2", "SCALE3"}
    exp_header = [l for l in exp_pdb.splitlines()
                  if l[:6].strip() in _KEEP]
    remarks = [
        "REMARK   0 COMPOSITE STRUCTURE generated by BEER v3",
        "REMARK   0 Experimental residues (occupancy 1.00): "
        f"{n_exp}",
        f"REMARK   0 {fill_source} gap-fill   (occupancy 0.00): "
        f"{n_pred}",
        f"REMARK   0 Kabsch fit: {n_pairs} Cα pairs, RMSD = {rmsd:.2f} Å",
        "REMARK   0 WARNING: junction geometry is rigid-body stitched,",
        "REMARK   0   not refined. Check Cα-Cα distances at gap boundaries.",
    ]
    return exp_header + remarks
