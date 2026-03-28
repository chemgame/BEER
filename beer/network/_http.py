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


_MOBIDB_API_BASE = "https://mobidb.org/api/entry"
_UNIPROT_REST_BASE = "https://rest.uniprot.org/uniprotkb"


def fetch_mobidb(uniprot_accession: str) -> dict:
    """Fetch MobiDB consensus disorder annotations for a UniProt accession.

    Parameters
    ----------
    uniprot_accession:
        UniProt accession (e.g. ``"P04637"``).

    Returns
    -------
    dict with keys: found (bool), accession (str), disorder_regions (list of
    dicts with start/end/length), fraction_disorder (float), n_predictors (int),
    source (str).

    Raises
    ------
    urllib.error.HTTPError (non-404) / urllib.error.URLError on failure.
    ValueError:
        If accession is empty.
    """
    accession = uniprot_accession.strip().upper()
    if not accession:
        raise ValueError("MobiDB query: no accession provided.")

    url = f"{_MOBIDB_API_BASE}/{accession}"
    try:
        data = _get_json(url)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return {"found": False, "accession": accession}
        raise

    if not data:
        return {"found": False, "accession": accession}

    # MobiDB v4 format: top-level predictions dict
    predictions = data.get("predictions", {})

    # MobiDB v3 fallback: data.{acc}.mobidb_lite
    if not predictions:
        nested = data.get("data", {})
        if isinstance(nested, dict):
            inner = nested.get(accession, nested.get(accession.lower(), {}))
            if isinstance(inner, dict):
                mobidb_lite_v3 = inner.get("mobidb_lite")
                if mobidb_lite_v3:
                    predictions = {"mobidb_lite": mobidb_lite_v3}

    n_predictors = len(predictions) if isinstance(predictions, dict) else 0

    mobidb_lite = predictions.get("mobidb_lite", {}) if isinstance(predictions, dict) else {}
    if not isinstance(mobidb_lite, dict):
        mobidb_lite = {}

    disorder_block = mobidb_lite.get("disorder", {})
    if not isinstance(disorder_block, dict):
        disorder_block = {}

    raw_regions = disorder_block.get("regions", [])
    if not isinstance(raw_regions, list):
        raw_regions = []

    disorder_regions: list = []
    for item in raw_regions:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            start = _safe_int(item[0])
            end = _safe_int(item[1])
            disorder_regions.append({"start": start, "end": end, "length": max(0, end - start + 1)})
        elif isinstance(item, dict):
            start = _safe_int(item.get("start", item.get("s", 0)))
            end = _safe_int(item.get("end", item.get("e", 0)))
            disorder_regions.append({"start": start, "end": end, "length": max(0, end - start + 1)})

    fraction_disorder = float(disorder_block.get("content_fraction", 0.0) or 0.0)

    return {
        "found": True,
        "accession": accession,
        "disorder_regions": disorder_regions,
        "fraction_disorder": fraction_disorder,
        "n_predictors": n_predictors,
        "source": "mobidb_lite",
    }


def _parse_variant_amino_acids(description: str) -> tuple[str, str]:
    """Best-effort extraction of original and variant amino acids from a description string.

    Handles formats like "R → H", "R > H", "Val -> Ala", and dbSNP entries.
    Returns a tuple (original, variant); both are empty string if not parseable.
    """
    import re
    for pattern in (
        r"([A-Za-z]+)\s*(?:\u2192|->|>)\s*([A-Za-z]+)",
    ):
        m = re.search(pattern, description)
        if m:
            return m.group(1), m.group(2)
    return "", ""


def _parse_disease_from_description(description: str) -> str:
    """Extract a disease name from a UniProt variant description if present.

    Looks for patterns like "in DISEASE_NAME;" or "associated with DISEASE_NAME".
    Returns the disease string or empty string.
    """
    import re
    # UniProt style: "in CANCER_TYPE;" or "in disease XYZ;"
    m = re.search(r"\bin\s+([A-Z][A-Z0-9_ ]{2,60}?)(?:\s*;|\s*\()", description)
    if m:
        candidate = m.group(1).strip()
        # Skip very generic tokens
        if candidate.lower() not in {"vitro", "vivo", "cancer", "disease", "humans", "mice"}:
            return candidate
    return ""


def fetch_uniprot_variants(uniprot_accession: str) -> list:
    """Fetch natural variants and disease mutations from UniProt for a given accession.

    Uses the UniProt REST API endpoint: GET
    ``https://rest.uniprot.org/uniprotkb/{accession}.json``

    Parameters
    ----------
    uniprot_accession:
        UniProt accession.

    Returns
    -------
    List of dicts with keys: position (int), original (str), variant (str),
    description (str), type (str), disease (str).  Returns an empty list on
    404 or if no relevant features exist.

    Raises
    ------
    urllib.error.HTTPError (non-404) / urllib.error.URLError on failure.
    ValueError:
        If accession is empty.
    """
    accession = uniprot_accession.strip().upper()
    if not accession:
        raise ValueError("UniProt variants query: no accession provided.")

    url = f"{_UNIPROT_REST_BASE}/{accession}.json"
    try:
        data = _get_json(url)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return []
        raise

    if not data:
        return []

    features = data.get("features", [])
    if not isinstance(features, list):
        return []

    _type_map = {
        "Natural variant": "natural_variant",
        "Mutagenesis": "mutagenesis",
    }

    variants: list = []
    for feature in features:
        if not isinstance(feature, dict):
            continue
        ftype = feature.get("type", "")
        if ftype not in _type_map:
            continue

        location = feature.get("location", {})
        start_block = location.get("start", {})
        end_block = location.get("end", {})
        position = _safe_int(start_block.get("value", 0)) if isinstance(start_block, dict) else 0
        end_position = _safe_int(end_block.get("value", 0)) if isinstance(end_block, dict) else 0

        description = feature.get("description", "") or ""
        description_truncated = description[:200]

        evidences = feature.get("evidences", [])
        source_db = ""
        if isinstance(evidences, list) and evidences:
            first_ev = evidences[0]
            if isinstance(first_ev, dict):
                src = first_ev.get("source", {})
                if isinstance(src, dict):
                    source_db = src.get("name", "")

        original, variant = _parse_variant_amino_acids(description)
        disease = _parse_disease_from_description(description)

        variants.append({
            "position": position,
            "original": original,
            "variant": variant,
            "description": description_truncated,
            "type": _type_map[ftype],
            "disease": disease,
        })

    return variants


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


# ---------------------------------------------------------------------------
# IntAct molecular interactions (PSICQUIC / MITAB 2.5)
# ---------------------------------------------------------------------------

_INTACT_PSICQUIC = (
    "https://www.ebi.ac.uk/Tools/webservices/psicquic/intact"
    "/webservices/current/search/query"
)


def _parse_mitab_id(field: str) -> str:
    """Extract primary accession from a MITAB identifier field.

    e.g. ``'uniprotkb:P04637'``  →  ``'P04637'``
    """
    if not field or field == "-":
        return "-"
    first = field.split("|")[0]
    return first.split(":", 1)[1] if ":" in first else first


def _parse_mitab_alias(field: str) -> str:
    """Return the preferred gene-name alias from a MITAB alias field.

    Prefers entries annotated as ``(gene name)``; falls back to the first
    token after the namespace prefix.
    """
    if not field or field == "-":
        return "-"
    for part in field.split("|"):
        if "gene name" in part.lower() and ":" in part:
            name = part.split(":", 1)[1]
            if "(" in name:
                name = name[: name.index("(")]
            return name.strip()
    first = field.split("|")[0]
    if ":" in first:
        name = first.split(":", 1)[1]
        if "(" in name:
            name = name[: name.index("(")]
        return name.strip()
    return first


def _parse_mitab_cv(field: str) -> str:
    """Extract the human-readable label from a PSI-MI CV term.

    e.g. ``'psi-mi:"MI:0018"(two hybrid)'``  →  ``'two hybrid'``
    """
    if not field or field == "-":
        return "-"
    first = field.split("|")[0]
    if "(" in first and ")" in first:
        return first[first.index("(") + 1 : first.rindex(")")]
    return first


def _parse_mitab_score(field: str) -> float | None:
    """Parse the IntAct MI-score from a confidence column.

    e.g. ``'intact-miscore:0.35'``  →  ``0.35``
    """
    if not field or field == "-":
        return None
    for part in field.split("|"):
        if ":" in part:
            try:
                return float(part.split(":", 1)[1])
            except ValueError:
                pass
    return None


def fetch_intact(uniprot_accession: str, max_results: int = 100) -> dict:
    """Fetch binary interactions from IntAct via the PSICQUIC REST service.

    Queries IntAct using the ``identifier:{accession}`` PSICQUIC syntax and
    parses the MITAB 2.5 tab-delimited response.

    Parameters
    ----------
    uniprot_accession:
        UniProt accession (e.g. ``'P04637'``).
    max_results:
        Maximum number of interactions to retrieve (default 100).

    Returns
    -------
    dict with keys:

    ``found`` (bool)
        ``True`` if at least one interaction was returned.
    ``accession`` (str)
        The queried accession.
    ``interactions`` (list[dict])
        Each dict: ``partner_id``, ``partner_name``, ``detection_method``,
        ``interaction_type``, ``pmid``, ``score`` (float or None).
    ``n_total`` (int)
        Number of interactions parsed.

    Raises
    ------
    urllib.error.HTTPError (non-404) / urllib.error.URLError on network failure.
    ValueError: if accession is empty.
    """
    accession = uniprot_accession.strip().upper()
    if not accession:
        raise ValueError("IntAct query: no accession provided.")

    url = (
        f"{_INTACT_PSICQUIC}/identifier:{accession}"
        f"?firstResult=0&maxResults={max_results}"
    )
    req = urllib.request.Request(
        url,
        headers={"Accept": "text/plain", "User-Agent": _USER_AGENT},
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return {"found": False, "accession": accession,
                    "interactions": [], "n_total": 0}
        raise

    interactions: list[dict] = []
    for line in raw.strip().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 12:
            continue

        id_a = _parse_mitab_id(parts[0])
        id_b = _parse_mitab_id(parts[1])
        alias_a = _parse_mitab_alias(parts[4]) if len(parts) > 4 else "-"
        alias_b = _parse_mitab_alias(parts[5]) if len(parts) > 5 else "-"

        # Identify the interaction partner (the molecule that is NOT the query)
        if accession in id_a.upper():
            partner_id, partner_name = id_b, alias_b
        else:
            partner_id, partner_name = id_a, alias_a

        detection = _parse_mitab_cv(parts[6]) if len(parts) > 6 else "-"
        interaction_type = _parse_mitab_cv(parts[11]) if len(parts) > 11 else "-"
        pmid = _parse_mitab_id(parts[8]) if len(parts) > 8 else "-"
        score = _parse_mitab_score(parts[14]) if len(parts) > 14 else None

        interactions.append({
            "partner_id":        partner_id,
            "partner_name":      partner_name,
            "detection_method":  detection,
            "interaction_type":  interaction_type,
            "pmid":              pmid,
            "score":             score,
        })

    return {
        "found":        len(interactions) > 0,
        "accession":    accession,
        "interactions": interactions,
        "n_total":      len(interactions),
    }
