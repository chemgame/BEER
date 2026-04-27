#!/usr/bin/env python3
"""Fetch curated training data for BEER BiLSTM heads from specialist databases.

This script replaces UniProt annotations with experimentally validated,
higher-quality data for the heads where UniProt is known to be insufficient.

Each source writes a JSON override file to CACHE_DIR:
    disorder_disprot.json       ← DisProt experimental disorder
    active_site_mcsa.json       ← M-CSA catalytic residues
    binding_site_biolip.json    ← BioLiP structure-derived ligand contacts
    disulfide_pdb.json          ← PDB SSBOND structure-derived disulfides
    phosphorylation_psp.json    ← dbPTM phosphorylation (PSP-compatible format)
    ubiquitination_psp.json     ← dbPTM ubiquitination
    methylation_psp.json        ← dbPTM methylation
    acetylation_psp.json        ← dbPTM acetylation

train_all_heads.py automatically detects these files and uses them.

Usage
-----
    # Run all automated sources (no registration needed):
    conda run -n beer python scripts/fetch_curated_data.py

    # Run specific sources:
    conda run -n beer python scripts/fetch_curated_data.py --source disprot mcsa pdb_ssbond dbptm

    # BioLiP — specify the file directly (already downloaded):
    conda run -n beer python scripts/fetch_curated_data.py --source biolip --biolip-file /path/to/BioLiP.txt

    # After downloading PhosphoSitePlus files (if access restored):
    conda run -n beer python scripts/fetch_curated_data.py --source psp --psp-dir scripts/data/psp

WHAT YOU MUST DO MANUALLY (see bottom of this file for detailed instructions):
    1. BioLiP — already downloaded; pass --biolip-file or place at repo root as BioLiP.txt
    2. SignalP 6.0 — DTU Biosustain form required (optional, UniProt SP is good)
"""
from __future__ import annotations

import argparse
import gzip
import json
import pathlib
import re
import sys
import time
import urllib.request
import urllib.error
from collections import defaultdict

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_all_heads import CACHE_DIR, MAX_SEQ_LEN

OUTPUT_FORMAT = "[{\"seq\": \"...\", \"labels\": [0, 1, ...]}]"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _fetch(url: str, timeout: int = 120, retries: int = 5,
           headers: dict | None = None) -> bytes:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers or {
                "User-Agent": "BEER-data-fetcher/2.0 (academic research)"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError) as e:
            wait = 15 * (2 ** attempt)
            print(f"    attempt {attempt+1}/{retries} failed ({e}); "
                  f"retrying in {wait}s …", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts")


def _fetch_uniprot_sequences(accessions: list[str],
                              batch_size: int = 100) -> dict[str, str]:
    """Fetch canonical sequences for a list of UniProt accessions."""
    seq_map = {}
    for i in range(0, len(accessions), batch_size):
        batch = accessions[i:i + batch_size]
        query = " OR ".join(f"accession:{a}" for a in batch)
        url   = (f"https://rest.uniprot.org/uniprotkb/search"
                 f"?query={urllib.request.quote(query)}"
                 f"&format=json&fields=accession,sequence&size={batch_size}")
        try:
            body = _fetch(url, timeout=60)
            data = json.loads(body)
            for entry in data.get("results", []):
                acc = entry["primaryAccession"]
                seq = entry.get("sequence", {}).get("value", "")
                if seq:
                    seq_map[acc] = seq[:MAX_SEQ_LEN]
        except Exception as e:
            print(f"    Warning: batch {i//batch_size} failed ({e})", flush=True)
        time.sleep(0.3)
    return seq_map


def _write_override(data: list[dict], path: pathlib.Path):
    CACHE_DIR.mkdir(exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
    pos_res  = sum(sum(e["labels"]) for e in data)
    tot_res  = sum(len(e["labels"]) for e in data)
    pos_frac = pos_res / max(tot_res, 1)
    print(f"  Wrote {len(data)} proteins ({pos_res}/{tot_res} positive residues, "
          f"{pos_frac:.4f} pos rate) → {path}", flush=True)


# ---------------------------------------------------------------------------
# DisProt — experimental disorder annotations
# ---------------------------------------------------------------------------

def fetch_disprot(force: bool = False):
    """Fetch DisProt bulk data and produce disorder_disprot.json."""
    out_path = CACHE_DIR / "disorder_disprot.json"
    if out_path.exists() and not force:
        print(f"  [DisProt] {out_path.name} already exists — skipping.", flush=True)
        return

    print("\n  [DisProt] Downloading bulk disorder annotations …", flush=True)

    # DisProt REST API — paginated
    entries = []
    page    = 1
    while True:
        url = (f"https://disprot.org/api/search"
               f"?release=current&show_ambiguous=false&show_obsolete=false"
               f"&format=json&page_size=200&page={page}")
        try:
            body = _fetch(url, timeout=90)
            data = json.loads(body)
        except Exception as e:
            print(f"  [DisProt] ERROR fetching page {page}: {e}", flush=True)
            break

        results = data.get("results", [])
        if not results:
            break
        entries.extend(results)
        total = data.get("count", 0)
        print(f"  [DisProt] page {page}: {len(entries)}/{total} entries …",
              flush=True)
        if len(entries) >= total:
            break
        page += 1
        time.sleep(0.3)

    print(f"  [DisProt] {len(entries)} entries fetched.", flush=True)

    # Build acc → disordered regions
    acc_to_regions: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for entry in entries:
        acc = entry.get("acc")
        if not acc:
            continue
        for reg in entry.get("disprot_consensus", {}).get("full", []):
            s = reg.get("start")
            e = reg.get("end")
            if s and e and s <= e:
                acc_to_regions[acc].append((int(s), int(e)))
        # Fallback: use individual regions if consensus empty
        if not acc_to_regions[acc]:
            for reg in entry.get("regions", []):
                s = reg.get("start")
                e = reg.get("end")
                if s and e and s <= e:
                    acc_to_regions[acc].append((int(s), int(e)))

    print(f"  [DisProt] {len(acc_to_regions)} proteins with disorder regions.",
          flush=True)

    # Fetch sequences for all accessions
    all_accs = list(acc_to_regions.keys())
    print(f"  [DisProt] Fetching {len(all_accs)} sequences from UniProt …",
          flush=True)
    seq_map = _fetch_uniprot_sequences(all_accs)
    print(f"  [DisProt] Got sequences for {len(seq_map)} accessions.", flush=True)

    # Build binary labels
    proteins = []
    for acc, regions in acc_to_regions.items():
        seq = seq_map.get(acc, "")
        if len(seq) < 20:
            continue
        L   = len(seq)
        lab = [0] * L
        for s, e in regions:
            s0 = max(0, s - 1)    # 1-based → 0-based
            e0 = min(L, e)
            for i in range(s0, e0):
                lab[i] = 1
        if sum(lab) == 0 or sum(lab) == L:
            continue
        proteins.append({"seq": seq, "labels": lab})

    _write_override(proteins, out_path)


# ---------------------------------------------------------------------------
# M-CSA — catalytic residue annotations (active site)
# ---------------------------------------------------------------------------

def fetch_mcsa(force: bool = False):
    """Fetch M-CSA entries and produce active_site_mcsa.json."""
    out_path = CACHE_DIR / "active_site_mcsa.json"
    if out_path.exists() and not force:
        print(f"  [M-CSA] {out_path.name} already exists — skipping.", flush=True)
        return

    print("\n  [M-CSA] Downloading catalytic residue annotations …", flush=True)

    entries = []
    url     = ("https://www.ebi.ac.uk/thornton-srv/m-csa/api/entries/"
               "?format=json&page_size=500")
    while url:
        try:
            body = _fetch(url, timeout=60)
            data = json.loads(body)
        except Exception as e:
            print(f"  [M-CSA] ERROR: {e}", flush=True)
            break
        entries.extend(data.get("results", []))
        url = data.get("next")
        print(f"  [M-CSA] {len(entries)} entries …", flush=True)
        time.sleep(0.3)

    print(f"  [M-CSA] {len(entries)} entries fetched.", flush=True)

    # Build acc → catalytic residue positions (1-based)
    acc_to_residues: dict[str, list[int]] = defaultdict(list)
    for entry in entries:
        for residue in entry.get("residues", []):
            for chain_res in residue.get("residuechain_set", []):
                if not chain_res.get("is_reference_chain", True):
                    continue
                uniprot_data = chain_res.get("uniprot_residue_id")
                if uniprot_data:
                    # Some M-CSA entries have UniProt position directly
                    try:
                        pos = int(uniprot_data)
                        acc = entry.get("uniprot_id", "")
                        if acc:
                            acc_to_residues[acc].append(pos)
                        continue
                    except (ValueError, TypeError):
                        pass
                # Fallback: use PDB residue number and map via accession
                pdb_pos = chain_res.get("pdb_residue_id")
                acc     = entry.get("uniprot_id", "")
                if acc and pdb_pos:
                    try:
                        acc_to_residues[acc].append(int(pdb_pos))
                    except (ValueError, TypeError):
                        pass

    print(f"  [M-CSA] {len(acc_to_residues)} UniProt accessions with catalytic residues.",
          flush=True)

    all_accs = list(acc_to_residues.keys())
    seq_map  = _fetch_uniprot_sequences(all_accs)

    proteins = []
    for acc, positions in acc_to_residues.items():
        seq = seq_map.get(acc, "")
        if len(seq) < 20:
            continue
        L   = len(seq)
        lab = [0] * L
        for pos in positions:
            idx = pos - 1   # 1-based → 0-based
            if 0 <= idx < L:
                lab[idx] = 1
        if sum(lab) == 0:
            continue
        proteins.append({"seq": seq, "labels": lab})

    _write_override(proteins, out_path)


# ---------------------------------------------------------------------------
# PDB SSBOND — structure-derived disulfide bonds via RCSB API
# ---------------------------------------------------------------------------

def fetch_pdb_ssbond(force: bool = False, max_entries: int = 8000):
    """Fetch PDB disulfide bond annotations via RCSB search and produce disulfide_pdb.json."""
    out_path = CACHE_DIR / "disulfide_pdb.json"
    if out_path.exists() and not force:
        print(f"  [PDB-SSBOND] {out_path.name} already exists — skipping.", flush=True)
        return

    print("\n  [PDB-SSBOND] Searching RCSB for structures with disulfide bonds …",
          flush=True)

    # RCSB search: structures with SS bonds, resolution < 2.5 Å, X-ray only
    search_query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": "X-RAY DIFFRACTION"
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.disulfide_bond_count",
                        "operator": "greater",
                        "value": 0
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": 2.5
                    }
                }
            ]
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": max_entries},
            "results_verbosity": "minimal"
        }
    }

    try:
        body = _fetch(
            "https://search.rcsb.org/rcsbsearch/v2/query",
            headers={
                "Content-Type": "application/json",
                "User-Agent": "BEER-data-fetcher/2.0"
            }
        )
        # Actually need to POST — use urllib directly
        import urllib.request as _ur
        req = _ur.Request(
            "https://search.rcsb.org/rcsbsearch/v2/query",
            data=json.dumps(search_query).encode(),
            headers={"Content-Type": "application/json",
                     "User-Agent": "BEER-data-fetcher/2.0"},
            method="POST"
        )
        with _ur.urlopen(req, timeout=60) as r:
            result = json.loads(r.read())
        pdb_ids = [hit["identifier"] for hit in result.get("result_set", [])]
    except Exception as e:
        print(f"  [PDB-SSBOND] RCSB search failed ({e}). "
              "Try running from your machine.", flush=True)
        return

    print(f"  [PDB-SSBOND] {len(pdb_ids)} PDB entries found.", flush=True)

    # For each PDB entry, get disulfide bond residue info via GraphQL
    # We use UniProt accession mapping to get canonical positions
    # Process in batches via RCSB data API
    acc_to_ssbond_positions: dict[str, set[int]] = defaultdict(set)

    for i in range(0, min(len(pdb_ids), max_entries), 50):
        batch = pdb_ids[i:i + 50]
        ids_str = '["' + '", "'.join(batch) + '"]'
        gql = f"""
        {{
          entries(entry_ids: {ids_str}) {{
            rcsb_id
            struct_conn {{
              conn_type_id
              pdbx_dist_value
              ptnr1_auth_seq_id
              ptnr2_auth_seq_id
              ptnr1_label_comp_id
              ptnr2_label_comp_id
            }}
            polymer_entities {{
              rcsb_polymer_entity_container_identifiers {{
                uniprot_ids
              }}
              entity_poly {{
                pdbx_seq_one_letter_code_can
              }}
            }}
          }}
        }}
        """
        try:
            req = urllib.request.Request(
                "https://data.rcsb.org/graphql",
                data=json.dumps({"query": gql}).encode(),
                headers={"Content-Type": "application/json",
                         "User-Agent": "BEER-data-fetcher/2.0"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                gql_data = json.loads(r.read())

            for entry in (gql_data.get("data", {}).get("entries") or []):
                struct_conns = entry.get("struct_conn") or []
                ss_positions = set()
                for conn in struct_conns:
                    if conn.get("conn_type_id") != "disulf":
                        continue
                    for pos_key in ("ptnr1_auth_seq_id", "ptnr2_auth_seq_id"):
                        try:
                            ss_positions.add(int(conn[pos_key]))
                        except (TypeError, KeyError, ValueError):
                            pass

                if not ss_positions:
                    continue

                for entity in (entry.get("polymer_entities") or []):
                    uniprot_ids = (entity.get(
                        "rcsb_polymer_entity_container_identifiers", {}
                    ) or {}).get("uniprot_ids") or []
                    for acc in uniprot_ids:
                        for pos in ss_positions:
                            acc_to_ssbond_positions[acc].add(pos)

        except Exception as e:
            print(f"  [PDB-SSBOND] GraphQL batch {i//50} failed: {e}", flush=True)
        time.sleep(0.5)

    print(f"  [PDB-SSBOND] {len(acc_to_ssbond_positions)} UniProt accessions "
          "with SSBOND data.", flush=True)

    all_accs = list(acc_to_ssbond_positions.keys())
    seq_map  = _fetch_uniprot_sequences(all_accs)

    proteins = []
    for acc, positions in acc_to_ssbond_positions.items():
        seq = seq_map.get(acc, "")
        if len(seq) < 20:
            continue
        L   = len(seq)
        lab = [0] * L
        for pos in positions:
            idx = pos - 1
            if 0 <= idx < L:
                lab[idx] = 1
        if sum(lab) == 0:
            continue
        proteins.append({"seq": seq, "labels": lab})

    _write_override(proteins, out_path)


# ---------------------------------------------------------------------------
# BioLiP — single-pass multi-head extractor
# One parse of BioLiP.txt produces 5 override files:
#   binding_site_biolip.json    small-molecule ligand contacts
#   dna_binding_biolip.json     protein-DNA contacts
#   rna_binding_biolip.json     protein-RNA contacts
#   nucleotide_binding_biolip.json  ATP/NAD/FAD/CoA/GTP etc.
#   zinc_finger_biolip.json     Zn-coordination residues
# ---------------------------------------------------------------------------

# BioLiP.txt column indices (0-based, tab-separated):
#   0  PDB ID    4  Ligand name   8  Binding residues renumbered (1-based into col 18)
#   1  Chain     5  Lig chain    15  UniProt accession
#   2  Res       6  Lig serial   18  Receptor sequence
_BL_COL_LIGAND  = 4
_BL_COL_BINDING = 8
_BL_COL_UNIPROT = 15
_BL_COL_SEQ     = 18
_BL_MIN_COLS    = 19

# Nucleotide cofactor ligand IDs → nucleotide_binding head
_BL_NUCL_COFACTORS = frozenset({
    "ATP", "ADP", "AMP", "GTP", "GDP", "GMP", "CTP", "CDP", "CMP",
    "UTP", "UDP", "UMP", "NAD", "NADH", "NAP", "NDP", "FAD", "FMN",
    "COA", "SAM", "SAH", "ANP", "AGS", "TPP", "H4B", "PLP", "PLR", "PMP",
})

# Ligands excluded from the generic binding_site head
# (metals, solvents, ions — each has its own head or is noise)
_BL_SKIP_BINDING_SITE = frozenset({
    "HOH", "DOD", "EDO", "PEG", "PO4", "SO4", "GOL", "BME", "MPD",
    "FMT", "ACT", "ACE", "NH2", "NHE", "CL", "MG", "CA", "NA", "MN",
    "FE", "FE2", "K", "SE", "BR", "I", "CU", "CO", "NI",
    "dna", "rna", "peptide",
    "ZN",   # → zinc_finger head
}) | _BL_NUCL_COFACTORS  # → nucleotide_binding head


def _resolve_biolip_file(biolip_file: pathlib.Path | None,
                          biolip_dir:  pathlib.Path | None) -> pathlib.Path | None:
    if biolip_file and pathlib.Path(biolip_file).exists():
        return pathlib.Path(biolip_file)
    root_candidate = ROOT / "BioLiP.txt"
    if root_candidate.exists():
        return root_candidate
    if biolip_dir:
        candidates = sorted(pathlib.Path(biolip_dir).glob("BioLiP*.txt"))
        if candidates:
            return candidates[0]
    return None


def fetch_biolip(biolip_file: pathlib.Path | None = None,
                 biolip_dir:  pathlib.Path | None = None,
                 force: bool = False):
    """Parse BioLiP.txt once and write 5 override files.

    File is auto-detected at repo root, or pass --biolip-file / --biolip-dir.
    """
    out_paths = {
        "binding":    CACHE_DIR / "binding_site_biolip.json",
        "dna":        CACHE_DIR / "dna_binding_biolip.json",
        "rna":        CACHE_DIR / "rna_binding_biolip.json",
        "nucleotide": CACHE_DIR / "nucleotide_binding_biolip.json",
        "zinc":       CACHE_DIR / "zinc_finger_biolip.json",
    }

    needed = {k for k, p in out_paths.items() if not p.exists() or force}
    if not needed:
        print("  [BioLiP] All output files already exist — skipping.", flush=True)
        return

    txt_file = _resolve_biolip_file(biolip_file, biolip_dir)
    if not txt_file:
        print("  [BioLiP] SKIPPED: BioLiP.txt not found.\n"
              "  Place at repo root or pass --biolip-file /path/to/BioLiP.txt",
              flush=True)
        return

    print(f"\n  [BioLiP] Parsing {txt_file} for heads: {sorted(needed)} …",
          flush=True)

    # Per-head accumulators: acc → [seq, [positions]]
    stores: dict[str, dict[str, list]] = {k: {} for k in needed}
    skipped_cols = 0

    def _acc(store, acc, seq, positions):
        if acc not in store:
            store[acc] = [seq, []]
        store[acc][1].extend(positions)

    with open(txt_file) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < _BL_MIN_COLS:
                skipped_cols += 1
                continue

            ligand  = parts[_BL_COL_LIGAND].strip()
            uniprot = parts[_BL_COL_UNIPROT].strip()
            seq     = parts[_BL_COL_SEQ].strip()
            bstr    = parts[_BL_COL_BINDING].strip()

            if not uniprot or uniprot in ("-", "") or len(seq) < 10 or not bstr:
                continue

            positions = []
            for token in bstr.split():
                try:
                    positions.append(int(re.sub(r"[^0-9]", "", token)))
                except ValueError:
                    pass
            if not positions:
                continue

            seq_t = seq[:MAX_SEQ_LEN]

            if ligand == "dna" and "dna" in needed:
                _acc(stores["dna"], uniprot, seq_t, positions)
            elif ligand == "rna" and "rna" in needed:
                _acc(stores["rna"], uniprot, seq_t, positions)
            elif ligand == "ZN" and "zinc" in needed:
                _acc(stores["zinc"], uniprot, seq_t, positions)
            elif ligand in _BL_NUCL_COFACTORS and "nucleotide" in needed:
                _acc(stores["nucleotide"], uniprot, seq_t, positions)
            elif (ligand not in _BL_SKIP_BINDING_SITE
                  and len(ligand) > 1 and "binding" in needed):
                _acc(stores["binding"], uniprot, seq_t, positions)

    if skipped_cols:
        print(f"  [BioLiP] {skipped_cols} short lines skipped.", flush=True)

    for head_key in needed:
        store = stores[head_key]
        print(f"  [BioLiP:{head_key}] {len(store)} accessions accumulated.",
              flush=True)
        proteins = []
        for acc, (seq, positions) in store.items():
            L   = len(seq)
            lab = [0] * L
            for pos in set(positions):
                idx = pos - 1
                if 0 <= idx < L:
                    lab[idx] = 1
            if sum(lab) == 0:
                continue
            proteins.append({"seq": seq, "labels": lab})
        _write_override(proteins, out_paths[head_key])


# ---------------------------------------------------------------------------
# PhosphoSitePlus — PTM experimental annotations
# Requires free academic registration at phosphositesplus.org
# ---------------------------------------------------------------------------

_PSP_MODS = {
    "phosphorylation": ("Phosphorylation_site_dataset.gz",
                        "phosphorylation_psp.json",
                        {"Phosphoserine", "Phosphothreonine", "Phosphotyrosine"}),
    "ubiquitination":  ("Ubiquitination_site_dataset.gz",
                        "ubiquitination_psp.json", None),
    "methylation":     ("Methylation_site_dataset.gz",
                        "methylation_psp.json", None),
    "acetylation":     ("Acetylation_site_dataset.gz",
                        "acetylation_psp.json", None),
}


def parse_phosphosite_plus(psp_dir: pathlib.Path, force: bool = False):
    """Parse PhosphoSitePlus bulk download files.

    Files required in psp_dir (download from phosphositesplus.org after registration):
        Phosphorylation_site_dataset.gz
        Ubiquitination_site_dataset.gz
        Methylation_site_dataset.gz
        Acetylation_site_dataset.gz

    These files are tab-separated. Columns include:
        GENE, PROTEIN, ACC_ID, HU_CHR_LOC, MOD_RSD, SITE_GRP_ID,
        ORGANISM, MW_kD, DOMAIN, SITE_+/-7_AA, LT_LIT, MS_LIT, MS_CST, ...

    We keep only sites with LT_LIT > 0 (low-throughput literature evidence).
    """
    psp_dir = pathlib.Path(psp_dir)

    for mod_name, (filename, out_name, mod_filter) in _PSP_MODS.items():
        out_path = CACHE_DIR / out_name
        if out_path.exists() and not force:
            print(f"  [PSP] {out_name} already exists — skipping.", flush=True)
            continue

        fpath = psp_dir / filename
        if not fpath.exists():
            print(f"  [PSP] {filename} not found in {psp_dir}. "
                  f"Download from phosphositesplus.org and re-run.", flush=True)
            continue

        print(f"\n  [PSP] Parsing {filename} …", flush=True)
        opener = gzip.open if str(fpath).endswith(".gz") else open

        acc_to_sites: dict[str, list[int]] = defaultdict(list)

        with opener(fpath, "rt", encoding="utf-8", errors="replace") as f:
            # Skip header lines starting with #
            header = None
            for line in f:
                if line.startswith("#"):
                    continue
                if header is None:
                    header = line.strip().split("\t")
                    # Normalize column names
                    header = [h.strip().upper().replace(" ", "_") for h in header]
                    break
            if header is None:
                print(f"  [PSP] Could not parse header in {filename}.", flush=True)
                continue

            def _col(name):
                for h in header:
                    if name in h:
                        return header.index(h)
                return -1

            acc_col    = _col("ACC_ID")
            mod_col    = _col("MOD_RSD")
            lt_col     = _col("LT_LIT")
            org_col    = _col("ORGANISM")
            modtype_col= _col("MODIFICATION")

            if acc_col < 0 or mod_col < 0:
                print(f"  [PSP] Cannot find ACC_ID or MOD_RSD columns. "
                      f"Headers: {header[:10]}", flush=True)
                continue

            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.strip().split("\t")
                if len(parts) <= max(acc_col, mod_col):
                    continue

                # Filter: human or mouse only
                if org_col >= 0 and len(parts) > org_col:
                    org = parts[org_col].lower()
                    if "human" not in org and "mouse" not in org:
                        continue

                # Filter: LTP experimental evidence only
                if lt_col >= 0 and len(parts) > lt_col:
                    try:
                        if int(parts[lt_col]) == 0:
                            continue
                    except ValueError:
                        pass

                acc    = parts[acc_col].strip().split("-")[0]  # strip isoform
                mod_rsd = parts[mod_col].strip()  # e.g. "S127-p" or "K48-ub"
                try:
                    pos = int(re.sub(r"[^0-9]", "", mod_rsd.split("-")[0]))
                except ValueError:
                    continue

                acc_to_sites[acc].append(pos)

        print(f"  [PSP] {len(acc_to_sites)} accessions with {mod_name} sites.",
              flush=True)

        all_accs = list(acc_to_sites.keys())
        seq_map  = _fetch_uniprot_sequences(all_accs)

        proteins = []
        for acc, positions in acc_to_sites.items():
            seq = seq_map.get(acc, "")
            if len(seq) < 20:
                continue
            L   = len(seq)
            lab = [0] * L
            for pos in set(positions):
                idx = pos - 1
                if 0 <= idx < L:
                    lab[idx] = 1
            if sum(lab) == 0:
                continue
            proteins.append({"seq": seq, "labels": lab})

        _write_override(proteins, out_path)


# ---------------------------------------------------------------------------
# GlyConnect — experimentally validated glycosylation sites (ExPASy)
# ---------------------------------------------------------------------------

def fetch_glyconnect(force: bool = False):
    """Fetch glycosylation sites from GlyConnect and produce glycosylation_glyconnect.json.

    GlyConnect aggregates N-linked and O-linked glycosylation from UniProt,
    PDB, and literature, all with experimental evidence.
    """
    out_path = CACHE_DIR / "glycosylation_glyconnect.json"
    if out_path.exists() and not force:
        print(f"  [GlyConnect] {out_path.name} already exists — skipping.", flush=True)
        return

    print("\n  [GlyConnect] Fetching glycosylation sites …", flush=True)

    acc_to_sites: dict[str, list[int]] = defaultdict(list)
    url: str | None = "https://glyconnect.expasy.org/api/proteins/?format=json&page_size=200"
    page = 0

    while url:
        try:
            body = _fetch(url, timeout=60)
            data = json.loads(body)
        except Exception as e:
            print(f"  [GlyConnect] ERROR on page {page}: {e}", flush=True)
            break

        results = data.get("results", [])
        if not results and page == 0:
            # API may use different structure — print top-level keys to diagnose
            print(f"  [GlyConnect] Unexpected response keys: {list(data.keys())}",
                  flush=True)
            break

        for protein in results:
            acc = (protein.get("uniprot_ac")
                   or protein.get("uniprot_id")
                   or protein.get("accession")
                   or protein.get("uniprotAccession", "")).strip()
            if not acc:
                continue
            acc = acc.split("-")[0].split(";")[0]

            # Sites may be nested under 'sites', 'glycosylation_sites', or inline
            sites = (protein.get("sites")
                     or protein.get("glycosylation_sites")
                     or [])
            for site in sites:
                pos = (site.get("position")
                       or site.get("site_position")
                       or site.get("pos"))
                try:
                    acc_to_sites[acc].append(int(pos))
                except (TypeError, ValueError):
                    pass

        url = data.get("next")
        page += 1
        total = data.get("count", "?")
        n_acc = len(acc_to_sites)
        print(f"  [GlyConnect] page {page}: {n_acc} accessions so far "
              f"(total in db: {total}) …", flush=True)
        time.sleep(0.3)

    print(f"  [GlyConnect] {len(acc_to_sites)} accessions with glycosylation sites.",
          flush=True)

    if not acc_to_sites:
        print("  [GlyConnect] No data retrieved — check API availability.",
              flush=True)
        return

    seq_map = _fetch_uniprot_sequences(list(acc_to_sites.keys()))

    proteins = []
    for acc, positions in acc_to_sites.items():
        seq = seq_map.get(acc, "")
        if len(seq) < 20:
            continue
        L   = len(seq)
        lab = [0] * L
        for pos in set(positions):
            idx = pos - 1
            if 0 <= idx < L:
                lab[idx] = 1
        if sum(lab) == 0:
            continue
        proteins.append({"seq": seq, "labels": lab})

    _write_override(proteins, out_path)


# ---------------------------------------------------------------------------
# dbPTM — PTM experimental annotations (replaces PhosphoSitePlus)
# Freely downloadable, aggregates PSP + PhosphoELM + HPRD + Swiss-Prot + others
# ---------------------------------------------------------------------------

_DBPTM_MODS = {
    "phosphorylation": ("Phosphorylation.gz", "phosphorylation_psp.json"),
    "ubiquitination":  ("Ubiquitination.gz",  "ubiquitination_psp.json"),
    "methylation":     ("Methylation.gz",      "methylation_psp.json"),
    "acetylation":     ("Acetylation.gz",      "acetylation_psp.json"),
}
DBPTM_BASE = "https://dbptm.bioinformatics.tw/download/experiment/"


def fetch_dbptm(force: bool = False):
    """Download PTM data from dbPTM and produce *_psp.json override files.

    dbPTM aggregates PhosphoSitePlus, PhosphoELM, HPRD, Swiss-Prot, and others.
    No registration needed — fully open access.

    Output files (same names as PSP to maintain compatibility with train_all_heads.py):
        phosphorylation_psp.json
        ubiquitination_psp.json
        methylation_psp.json
        acetylation_psp.json
    """
    for mod_name, (filename, out_name) in _DBPTM_MODS.items():
        out_path = CACHE_DIR / out_name
        if out_path.exists() and not force:
            print(f"  [dbPTM] {out_name} already exists — skipping.", flush=True)
            continue

        url = DBPTM_BASE + filename
        print(f"\n  [dbPTM] Downloading {filename} …", flush=True)
        try:
            raw = _fetch(url, timeout=300, retries=5)
        except RuntimeError as e:
            print(f"  [dbPTM] Download failed: {e}\n"
                  f"  Manual fallback: download {filename} from {DBPTM_BASE}\n"
                  f"  and place it in CACHE_DIR ({CACHE_DIR}), then re-run.",
                  flush=True)
            continue

        print(f"  [dbPTM] Parsing {mod_name} …", flush=True)
        acc_to_sites: dict[str, list[int]] = defaultdict(list)
        lines_parsed = 0

        opener = gzip.open
        import io
        with opener(io.BytesIO(raw), "rt", encoding="utf-8", errors="replace") as f:
            header: list[str] | None = None
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                # Header detection: first non-comment line, or lines starting with '#'
                if line.startswith("##"):
                    continue
                if header is None:
                    # Strip leading '#' if present
                    header = line.lstrip("#").split("\t")
                    header = [h.strip().upper() for h in header]
                    continue

                parts = line.split("\t")

                # Column detection by header name (robust across dbPTM versions)
                def _ci(keywords: list[str]) -> int:
                    for kw in keywords:
                        for i, h in enumerate(header):
                            if kw in h:
                                return i
                    return -1

                if lines_parsed == 0:
                    # Detect column indices once on first data line
                    acc_ci  = _ci(["UNIPROT", "ACC", "ACCESSION"])
                    pos_ci  = _ci(["POSITION", "POS", "SITE"])
                    org_ci  = _ci(["ORGANISM", "SPECIES", "ORG"])
                    src_ci  = _ci(["SOURCE", "REF", "EVIDENCE", "DATABASE"])
                    if acc_ci < 0 or pos_ci < 0:
                        print(f"  [dbPTM] Cannot identify required columns.\n"
                              f"  Header: {header[:12]}", flush=True)
                        break

                lines_parsed += 1
                if len(parts) <= max(acc_ci, pos_ci):
                    continue

                # Organism filter: human and mouse only
                if org_ci >= 0 and len(parts) > org_ci:
                    org = parts[org_ci].lower()
                    if "sapiens" not in org and "musculus" not in org \
                            and "human" not in org and "mouse" not in org:
                        continue

                acc = parts[acc_ci].strip().split("-")[0].split(";")[0]
                if not acc or len(acc) > 10:
                    continue

                pos_str = parts[pos_ci].strip()
                try:
                    pos = int(re.sub(r"[^0-9]", "", pos_str))
                except ValueError:
                    continue
                if pos <= 0:
                    continue

                acc_to_sites[acc].append(pos)

        print(f"  [dbPTM] {len(acc_to_sites)} accessions with {mod_name} sites "
              f"({lines_parsed} lines processed).", flush=True)

        if not acc_to_sites:
            print(f"  [dbPTM] No data parsed for {mod_name} — skipping write.",
                  flush=True)
            continue

        all_accs = list(acc_to_sites.keys())
        seq_map  = _fetch_uniprot_sequences(all_accs)

        proteins = []
        for acc, positions in acc_to_sites.items():
            seq = seq_map.get(acc, "")
            if len(seq) < 20:
                continue
            L   = len(seq)
            lab = [0] * L
            for pos in set(positions):
                idx = pos - 1
                if 0 <= idx < L:
                    lab[idx] = 1
            if sum(lab) == 0:
                continue
            proteins.append({"seq": seq, "labels": lab})

        _write_override(proteins, out_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Aggregation — WALTZ-DB 2.0 + AmyLoad + AmyPro + PDB amyloid fibrils
# ---------------------------------------------------------------------------

_WALTZ_URLS = [
    "https://waltz.switchlab.org/downloads/WALTZ_positive.fasta",
    "https://waltz.switchlab.org/downloads/waltz_positive.fasta",
]

_AMYLOAD_URLS = [
    "https://comprec-lin.iiar.pwr.edu.pl/amyload/downloads/amyload_positive.fasta",
    "https://comprec-lin.iiar.pwr.edu.pl/amyload/downloads/AmyLoad_positive.fasta",
]

_AMYPRO_API = "https://amypro.net/api/entries/?format=json&page_size=200"


def _fasta_to_proteins(raw: bytes, label: int = 1) -> list[dict]:
    """Parse FASTA bytes → list of {seq, labels} with uniform label."""
    proteins = []
    seq = ""
    for line in raw.decode(errors="replace").splitlines():
        if line.startswith(">"):
            if seq and len(seq) >= 4:
                proteins.append({"seq": seq, "labels": [label] * len(seq)})
            seq = ""
        else:
            seq += line.strip().upper()
    if seq and len(seq) >= 4:
        proteins.append({"seq": seq, "labels": [label] * len(seq)})
    return proteins


def _fetch_pdb_amyloid_fibrils(max_entries: int = 3000) -> list[dict]:
    """Fetch short amyloid fibril chain sequences from RCSB. All residues = positive."""
    print("  [Aggregation/PDB] Searching RCSB for amyloid fibril structures …",
          flush=True)
    search_query = {
        "query": {
            "type": "group",
            "logical_operator": "or",
            "nodes": [
                {"type": "terminal", "service": "text",
                 "parameters": {"attribute": "struct.title",
                                "operator": "contains_phrase", "value": "amyloid fibril"}},
                {"type": "terminal", "service": "text",
                 "parameters": {"attribute": "struct.title",
                                "operator": "contains_phrase", "value": "amyloid fiber"}},
                {"type": "terminal", "service": "text",
                 "parameters": {"attribute": "struct.pdbx_descriptor",
                                "operator": "contains_words", "value": "amyloid"}},
            ],
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": max_entries},
            "results_verbosity": "minimal",
        },
    }
    try:
        req = urllib.request.Request(
            "https://search.rcsb.org/rcsbsearch/v2/query",
            data=json.dumps(search_query).encode(),
            headers={"Content-Type": "application/json",
                     "User-Agent": "BEER-data-fetcher/2.0"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            result = json.loads(r.read())
        pdb_ids = [h["identifier"] for h in result.get("result_set", [])]
    except Exception as e:
        print(f"  [Aggregation/PDB] RCSB search failed: {e}", flush=True)
        return []

    print(f"  [Aggregation/PDB] {len(pdb_ids)} fibril PDB entries found.", flush=True)

    proteins = []
    for i in range(0, len(pdb_ids), 50):
        batch   = pdb_ids[i:i + 50]
        ids_str = '["' + '", "'.join(batch) + '"]'
        gql = f"""
        {{
          entries(entry_ids: {ids_str}) {{
            polymer_entities {{
              entity_poly {{
                pdbx_seq_one_letter_code_can
                type
              }}
            }}
          }}
        }}
        """
        try:
            req = urllib.request.Request(
                "https://data.rcsb.org/graphql",
                data=json.dumps({"query": gql}).encode(),
                headers={"Content-Type": "application/json",
                         "User-Agent": "BEER-data-fetcher/2.0"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                gql_data = json.loads(r.read())
            for entry in (gql_data.get("data", {}).get("entries") or []):
                for entity in (entry.get("polymer_entities") or []):
                    poly = entity.get("entity_poly") or {}
                    ptype = poly.get("type", "")
                    if "polypeptide" not in ptype.lower():
                        continue
                    seq = (poly.get("pdbx_seq_one_letter_code_can") or "").replace("\n", "").strip()
                    # Keep only realistic fibril cores (4–150 residues)
                    if 4 <= len(seq) <= 150:
                        proteins.append({"seq": seq, "labels": [1] * len(seq)})
        except Exception as e:
            print(f"  [Aggregation/PDB] GraphQL batch {i//50} failed: {e}", flush=True)
        time.sleep(0.4)

    print(f"  [Aggregation/PDB] {len(proteins)} fibril sequences extracted.", flush=True)
    return proteins


def _fetch_amypro() -> list[dict]:
    """Fetch AmyPro entries (experimentally validated amyloid regions in proteins)."""
    print("  [Aggregation/AmyPro] Fetching AmyPro entries …", flush=True)
    acc_to_regions: dict[str, list[tuple[int, int]]] = defaultdict(list)
    url: str | None = _AMYPRO_API
    page = 0
    while url:
        try:
            body = _fetch(url, timeout=60)
            data = json.loads(body)
        except Exception as e:
            print(f"  [Aggregation/AmyPro] ERROR page {page}: {e}", flush=True)
            break
        results = data.get("results", [])
        if page == 0 and not results:
            # Try alternate API structure
            if isinstance(data, list):
                results = data
            else:
                print(f"  [Aggregation/AmyPro] Unexpected response: {list(data.keys())}",
                      flush=True)
                break
        for entry in results:
            acc = (entry.get("uniprot_acc") or entry.get("accession")
                   or entry.get("uniprot_id") or "").strip().split("-")[0]
            if not acc:
                continue
            for region in (entry.get("amyloid_regions") or entry.get("regions") or []):
                s = region.get("start") or region.get("begin")
                e = region.get("end")
                try:
                    acc_to_regions[acc].append((int(s), int(e)))
                except (TypeError, ValueError):
                    pass
        url = data.get("next") if isinstance(data, dict) else None
        page += 1
        time.sleep(0.3)

    if not acc_to_regions:
        print("  [Aggregation/AmyPro] No data retrieved.", flush=True)
        return []

    print(f"  [Aggregation/AmyPro] {len(acc_to_regions)} accessions. "
          "Fetching sequences …", flush=True)
    seq_map = _fetch_uniprot_sequences(list(acc_to_regions.keys()))
    proteins = []
    for acc, regions in acc_to_regions.items():
        seq = seq_map.get(acc, "")
        if len(seq) < 20:
            continue
        L   = len(seq)
        lab = [0] * L
        for s, e in regions:
            for idx in range(max(0, s - 1), min(L, e)):
                lab[idx] = 1
        if sum(lab) == 0:
            continue
        proteins.append({"seq": seq, "labels": lab})
    print(f"  [Aggregation/AmyPro] {len(proteins)} proteins with amyloid regions.",
          flush=True)
    return proteins


def fetch_aggregation(force: bool = False):
    """Fetch aggregation training data from 4 sources and produce aggregation_curated.json.

    Sources (all positives = experimentally validated):
      1. WALTZ-DB 2.0   — amyloidogenic hexapeptides (Switch Lab)
      2. AmyLoad        — ~10K experimentally tested amyloidogenic peptides
      3. AmyPro         — ~3,500 proteins with residue-level amyloid regions
      4. PDB fibrils    — amyloid fibril structures from RCSB (~800 entries)
      5. Soluble negatives — folded monomeric proteins from UniProt (all-zero labels)

    Recent papers (AggAlpha 2023, AggPredict2 2024) use subsets of these same sources.
    """
    out_path = CACHE_DIR / "aggregation_curated.json"
    if out_path.exists() and not force:
        print(f"  [Aggregation] {out_path.name} already exists — skipping.", flush=True)
        return

    print("\n  [Aggregation] Collecting training data from all sources …", flush=True)
    positives: list[dict] = []

    # 1. WALTZ-DB
    for url in _WALTZ_URLS:
        try:
            raw = _fetch(url, timeout=60, retries=3)
            waltz = _fasta_to_proteins(raw, label=1)
            print(f"  [Aggregation/WALTZ] {len(waltz)} peptides.", flush=True)
            positives.extend(waltz)
            break
        except Exception as e:
            print(f"  [Aggregation/WALTZ] {url} failed: {e}", flush=True)

    # 2. AmyLoad
    for url in _AMYLOAD_URLS:
        try:
            raw = _fetch(url, timeout=60, retries=3)
            amyload = _fasta_to_proteins(raw, label=1)
            print(f"  [Aggregation/AmyLoad] {len(amyload)} peptides.", flush=True)
            positives.extend(amyload)
            break
        except Exception as e:
            print(f"  [Aggregation/AmyLoad] {url} failed: {e}", flush=True)

    # 3. AmyPro
    amypro = _fetch_amypro()
    positives.extend(amypro)

    # 4. PDB amyloid fibrils
    fibrils = _fetch_pdb_amyloid_fibrils()
    positives.extend(fibrils)

    # De-duplicate positives by sequence
    seen: set[str] = set()
    positives_dedup = []
    for p in positives:
        if p["seq"] not in seen:
            seen.add(p["seq"])
            positives_dedup.append(p)
    print(f"  [Aggregation] {len(positives_dedup)} unique positive sequences total.",
          flush=True)

    if len(positives_dedup) < 50:
        print("  [Aggregation] WARNING: very few positive sequences — "
              "check network access to WALTZ/AmyLoad.", flush=True)

    # 5. Soluble negatives — target ~3× positive residue count
    n_pos_res = sum(sum(p["labels"]) for p in positives_dedup)
    target_neg_res = n_pos_res * 3

    print(f"  [Aggregation] Fetching soluble negatives "
          f"(target ~{target_neg_res // 300} proteins) …", flush=True)

    neg_url = (
        "https://rest.uniprot.org/uniprotkb/search"
        "?query=reviewed:true+AND+annotation_score:[5+TO+*]"
        "+AND+NOT+(keyword:KW-0043)+AND+NOT+(keyword:KW-0472)"
        "+AND+ft_strand:*&format=json&fields=accession,sequence&size=500"
    )
    import re as _re
    negatives: list[dict] = []
    neg_res_count = 0
    while neg_url and neg_res_count < target_neg_res:
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(neg_url,
                        headers={"User-Agent": "BEER-data-fetcher/2.0"}),
                    timeout=60) as r:
                body = r.read()
                headers = {k.lower(): v for k, v in r.headers.items()}
            for p in json.loads(body).get("results", []):
                seq = p.get("sequence", {}).get("value", "")[:MAX_SEQ_LEN]
                if len(seq) >= 30 and seq not in seen:
                    negatives.append({"seq": seq, "labels": [0] * len(seq)})
                    seen.add(seq)
                    neg_res_count += len(seq)
            m = _re.search(r'<([^>]+)>;\s*rel="next"', headers.get("link", ""))
            neg_url = m.group(1) if m else ""
            time.sleep(0.2)
        except Exception as e:
            print(f"  [Aggregation] Negatives fetch error: {e}", flush=True)
            break

    print(f"  [Aggregation] {len(negatives)} soluble negatives.", flush=True)

    all_proteins = positives_dedup + negatives
    _write_override(all_proteins, out_path)


ALL_AUTO_SOURCES = ["disprot", "mcsa", "pdb_ssbond", "dbptm", "glyconnect",
                    "biolip", "aggregation"]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source", nargs="*",
        default=ALL_AUTO_SOURCES,
        choices=ALL_AUTO_SOURCES + ["psp"],
        help=f"Data sources to fetch (default: all automated = {ALL_AUTO_SOURCES})")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if cache file exists")
    parser.add_argument("--biolip-file", metavar="FILE", default=None,
                        help="Path to BioLiP.txt directly (highest priority)")
    parser.add_argument("--biolip-dir", metavar="DIR", default=None,
                        help="Directory containing a BioLiP*.txt file")
    parser.add_argument("--psp-dir", metavar="DIR", default=None,
                        help="Directory containing PhosphoSitePlus .gz files "
                             "(only if PSP access is ever restored)")
    args = parser.parse_args()

    sources = set(args.source)
    print(f"\nBEER curated data fetcher — sources: {sorted(sources)}\n", flush=True)

    if "disprot" in sources:
        fetch_disprot(force=args.force)

    if "mcsa" in sources:
        fetch_mcsa(force=args.force)

    if "pdb_ssbond" in sources:
        fetch_pdb_ssbond(force=args.force)

    if "dbptm" in sources:
        fetch_dbptm(force=args.force)

    if "glyconnect" in sources:
        fetch_glyconnect(force=args.force)

    if "biolip" in sources:
        fetch_biolip(
            biolip_file=args.biolip_file,
            biolip_dir=args.biolip_dir,
            force=args.force,
        )

    if "aggregation" in sources:
        fetch_aggregation(force=args.force)

    if "psp" in sources:
        if not args.psp_dir:
            print(
                "\n  [PSP] PhosphoSitePlus access is currently suspended.\n"
                "  dbPTM (--source dbptm) provides equivalent data without registration.\n"
                "  If PSP access is restored, download the .gz files and re-run with\n"
                "  --source psp --psp-dir <download-dir>", flush=True)
        else:
            parse_phosphosite_plus(pathlib.Path(args.psp_dir), force=args.force)

    print("\nDone. Run `python scripts/dataset_audit.py` to verify all datasets "
          "before training.\n", flush=True)


# ---------------------------------------------------------------------------
# Manual download instructions (printed here for reference)
# ---------------------------------------------------------------------------

MANUAL_INSTRUCTIONS = """
╔══════════════════════════════════════════════════════════════════════════╗
║            WHAT YOU MUST DOWNLOAD MANUALLY                             ║
╚══════════════════════════════════════════════════════════════════════════╝

1. BioLiP (binding site) — ALREADY DOWNLOADED
   ─────────────────────────────────────────────────────────────────────────
   • File already placed at: <repo_root>/BioLiP.txt
   • Run:
       python scripts/fetch_curated_data.py --source biolip
   • (Auto-detected at repo root; no extra flags needed)

2. SignalP 6.0 training benchmark (signal peptide) — OPTIONAL
   ─────────────────────────────────────────────────────────────────────────
   • URL:  https://services.healthtech.dtu.dk/services/SignalP-6.0/
   • UniProt Swiss-Prot signal peptide annotations are already high quality —
     this is optional, not critical for a first-version model.

3. eCLIP data for RNA-binding — OPTIONAL, large dataset
   ─────────────────────────────────────────────────────────────────────────
   • URL:  https://www.encodeproject.org/ (search eCLIP)
   • Per-experiment files totalling hundreds of GB.
   • The RNA-binding head trained on RBPDB + UniProt is serviceable without this.

4. CC+ (coiled-coil) — OPTIONAL
   ─────────────────────────────────────────────────────────────────────────
   • URL:  https://coiledcoils.chm.bris.ac.uk/ccplus/search/
   • Requires contacting the Bristol group.
   • UniProt ft_coiled annotations are reasonable for an initial model.

AUTOMATED (no action needed — fetch_curated_data.py handles these):
   ✓ DisProt (disorder)
   ✓ M-CSA (active site)
   ✓ PDB SSBOND via RCSB (disulfide)
   ✓ dbPTM (phospho/ubiq/methyl/acetyl — replaces PhosphoSitePlus, no registration)
   ✓ PDBTM (transmembrane — via train_tm_head.py)
   ✓ WALTZ (aggregation — via train_aggregation_head.py)

NOTE ON PhosphoSitePlus:
   PSP has suspended academic licensing. dbPTM is used instead — it aggregates
   data from PSP, PhosphoELM, HPRD, Swiss-Prot, and ~20 other sources.
   If PSP access is restored: python scripts/fetch_curated_data.py --source psp --psp-dir <dir>
"""


if __name__ == "__main__":
    main()
