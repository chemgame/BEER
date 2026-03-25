"""Pure-Python HTTP helpers for BEER network queries (no Qt dependencies)."""
from __future__ import annotations

import json
import urllib.request
import urllib.error

_TIMEOUT_SECONDS = 30
_USER_AGENT = "BEER-biophysics/1.0 (scientific software)"

_ELM_BASE_URL = "https://elm.eu.org/instances.json"
_DISPROT_API_BASE = "https://disprot.org/api"
_PHASEPDB_BASE = "https://phasepdb.org/api/protein"


def _safe_int(value, default: int = 0) -> int:
    """Convert *value* to int, returning *default* on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _get_json(url: str, timeout: int = _TIMEOUT_SECONDS) -> object:
    """Perform a GET request and return decoded JSON.

    Raises urllib.error.HTTPError / urllib.error.URLError on failure.
    Raises json.JSONDecodeError on bad JSON.
    """
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": _USER_AGENT,
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw)


def fetch_elm(uniprot_accession: str) -> list:
    """Query ELM database for linear motif instances for a UniProt accession.

    Parameters
    ----------
    uniprot_accession:
        UniProt accession (e.g. ``"P04637"``).

    Returns
    -------
    list of dicts with keys: elm_identifier, start, end, logic, toGo,
    primary_reference_pmed_id, accession, raw.

    Raises
    ------
    urllib.error.HTTPError / urllib.error.URLError:
        On network/HTTP failures.
    json.JSONDecodeError:
        On malformed responses.
    ValueError:
        If accession is empty.
    """
    accession = uniprot_accession.strip().upper()
    if not accession:
        raise ValueError("ELM query: no accession provided.")

    url = f"{_ELM_BASE_URL}?q={accession}"
    data = _get_json(url)

    if isinstance(data, list):
        raw_instances = data
    elif isinstance(data, dict):
        raw_instances = data.get("instances") or data.get("Instances") or []
        if not isinstance(raw_instances, list):
            raw_instances = []
    else:
        raw_instances = []

    instances: list = []
    for item in raw_instances:
        if not isinstance(item, dict):
            continue
        instances.append({
            "elm_identifier": item.get("elm_identifier", item.get("elm_type", "")),
            "start": _safe_int(item.get("start", item.get("Start", 0))),
            "end": _safe_int(item.get("end", item.get("End", 0))),
            "logic": item.get("logic", item.get("Logic", "")),
            "toGo": item.get("toGo", item.get("to_go", "")),
            "primary_reference_pmed_id": item.get(
                "primary_reference_pmed_id", item.get("pmed_id", "")
            ),
            "accession": accession,
            "raw": item,
        })
    return instances


def fetch_disprot(uniprot_accession: str) -> dict:
    """Fetch DisProt disorder annotations for a UniProt accession.

    Parameters
    ----------
    uniprot_accession:
        UniProt accession.

    Returns
    -------
    dict with keys: found (bool), and on success also disprot_id,
    protein_name, accession, sequence_length, regions, n_disordered_aa,
    fraction_disordered.

    Raises
    ------
    urllib.error.HTTPError (non-404) / urllib.error.URLError on failure.
    ValueError:
        If accession is empty.
    """
    accession = uniprot_accession.strip().upper()
    if not accession:
        raise ValueError("DisProt query: no accession provided.")

    url = f"{_DISPROT_API_BASE}/{accession}"
    try:
        data = _get_json(url)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return {"found": False, "accession": accession}
        raise

    if not data:
        return {"found": False, "accession": accession}

    disprot_id = data.get("disprot_id", data.get("id", ""))
    protein_name = data.get("protein_name", data.get("name", ""))
    sequence_length = int(data.get("length", data.get("sequence_length", 0)))

    raw_regions = data.get("regions", [])
    regions: list = []
    for reg in raw_regions:
        if not isinstance(reg, dict):
            continue
        start = _safe_int(reg.get("start", reg.get("Start", 0)))
        end = _safe_int(reg.get("end", reg.get("End", 0)))
        rtype = reg.get("type_name", reg.get("type", reg.get("term", "disorder")))
        if isinstance(rtype, dict):
            rtype = rtype.get("name", rtype.get("term", "disorder"))
        regions.append({"start": start, "end": end, "type": str(rtype)})

    disordered_positions: set = set()
    for reg in regions:
        for pos in range(reg["start"], reg["end"] + 1):
            disordered_positions.add(pos)

    n_disordered_aa = len(disordered_positions)
    fraction_disordered = (
        n_disordered_aa / sequence_length if sequence_length > 0 else 0.0
    )

    return {
        "found": True,
        "disprot_id": disprot_id,
        "protein_name": protein_name,
        "accession": accession,
        "sequence_length": sequence_length,
        "regions": regions,
        "n_disordered_aa": n_disordered_aa,
        "fraction_disordered": fraction_disordered,
    }


def fetch_phasepdb(uniprot_accession: str) -> dict:
    """Fetch PhaSepDB phase separation data for a UniProt accession.

    Parameters
    ----------
    uniprot_accession:
        UniProt accession.

    Returns
    -------
    dict with keys: found (bool), and on success also source, accession,
    gene_name, protein_name, category, evidence_type, organism, references.

    Raises
    ------
    urllib.error.HTTPError (non-404) / urllib.error.URLError on failure.
    ValueError:
        If accession is empty.
    """
    accession = uniprot_accession.strip().upper()
    if not accession:
        raise ValueError("PhaSepDB query: no accession provided.")

    url = f"{_PHASEPDB_BASE}/{accession}"
    try:
        data = _get_json(url, timeout=10)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return {"found": False, "accession": accession}
        raise

    if not data:
        return {"found": False, "accession": accession}

    if isinstance(data, list):
        if len(data) == 0:
            return {"found": False, "accession": accession}
        entry = data[0]
    elif isinstance(data, dict):
        entry = data
    else:
        return {"found": False, "accession": accession}

    references = entry.get("references", entry.get("refs", []))
    if not isinstance(references, list):
        references = []

    parsed_refs: list = []
    for ref in references:
        if isinstance(ref, dict):
            parsed_refs.append({
                "pmid": str(ref.get("pmid", ref.get("PubMed", ""))),
                "title": ref.get("title", ref.get("Title", "")),
            })
        elif isinstance(ref, str):
            parsed_refs.append({"pmid": ref, "title": ""})

    return {
        "found": True,
        "source": "PhaSepDB",
        "accession": accession,
        "gene_name": entry.get("gene_name", entry.get("gene", "")),
        "protein_name": entry.get("protein_name", entry.get("name", entry.get("protein", ""))),
        "category": entry.get("category", entry.get("Category", "")),
        "evidence_type": entry.get("evidence_type", entry.get("evidence", entry.get("Evidence", ""))),
        "organism": entry.get("organism", entry.get("Organism", "")),
        "references": parsed_refs,
    }


def fetch_alphafold_pdb(uniprot_accession: str) -> dict:
    """Fetch AlphaFold predicted structure for a UniProt accession.

    Parameters
    ----------
    uniprot_accession:
        UniProt accession.

    Returns
    -------
    dict with keys: pdb_str, accession.

    Raises
    ------
    urllib.error.HTTPError / urllib.error.URLError on failure.
    ValueError:
        If accession is empty or no prediction found.
    """
    accession = uniprot_accession.strip().upper()
    if not accession:
        raise ValueError("AlphaFold query: no accession provided.")

    meta_url = f"https://alphafold.ebi.ac.uk/api/prediction/{accession}"
    meta = _get_json(meta_url)
    if not meta:
        raise ValueError(f"No AlphaFold prediction found for {accession}.")

    pdb_url = meta[0]["pdbUrl"]
    req = urllib.request.Request(
        pdb_url,
        headers={"User-Agent": _USER_AGENT},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        pdb_str = r.read().decode()

    return {"pdb_str": pdb_str, "accession": accession}


def fetch_pfam(uniprot_accession: str) -> list:
    """Fetch Pfam domain annotations for a UniProt accession via InterPro REST API.

    Parameters
    ----------
    uniprot_accession:
        UniProt accession.

    Returns
    -------
    Sorted list of dicts: name, accession, start, end.

    Raises
    ------
    urllib.error.HTTPError / urllib.error.URLError on failure.
    ValueError:
        If accession is empty.
    """
    accession = uniprot_accession.strip().upper()
    if not accession:
        raise ValueError("Pfam query: no accession provided.")

    url = (
        f"https://www.ebi.ac.uk/interpro/api/entry/pfam"
        f"/protein/uniprot/{accession}/?page_size=100"
    )
    data = _get_json(url)
    domains = []
    for result in data.get("results", []):
        meta = result.get("metadata", {})
        raw_name = meta.get("name", meta.get("accession", "Unknown"))
        name = raw_name.get("name", "") if isinstance(raw_name, dict) else str(raw_name)
        acc = meta.get("accession", "")
        for prot in result.get("proteins", []):
            for loc in prot.get("entry_protein_locations", []):
                for frag in loc.get("fragments", []):
                    domains.append({
                        "name": name,
                        "accession": acc,
                        "start": frag["start"],
                        "end": frag["end"],
                    })
    domains.sort(key=lambda d: d["start"])
    return domains


def fetch_uniprot_fasta(query: str) -> str:
    """Fetch FASTA sequence from UniProt for an accession or search query.

    Parameters
    ----------
    query:
        UniProt accession or free-text search term.

    Returns
    -------
    FASTA-format string.

    Raises
    ------
    urllib.error.HTTPError / urllib.error.URLError on failure.
    """
    # Try direct accession fetch first
    accession = query.strip().upper()
    url = f"https://www.uniprot.org/uniprot/{accession}.fasta"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": _USER_AGENT},
    )
    with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as r:
        return r.read().decode("utf-8")


def fetch_rcsb_pdb(pdb_id: str) -> str:
    """Fetch a PDB file from RCSB.

    Parameters
    ----------
    pdb_id:
        4-character PDB identifier (case-insensitive).

    Returns
    -------
    PDB file contents as a string.

    Raises
    ------
    urllib.error.HTTPError / urllib.error.URLError on failure.
    ValueError:
        If pdb_id is empty.
    """
    pdb_id = pdb_id.strip().upper()
    if not pdb_id:
        raise ValueError("RCSB fetch: no PDB ID provided.")
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": _USER_AGENT},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8")
