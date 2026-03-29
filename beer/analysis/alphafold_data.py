"""AlphaMissense score lookup via EBI AlphaFold API."""
from __future__ import annotations
import json
import urllib.request
import urllib.error
import csv
import io


def fetch_alphafold_entry(uniprot_id: str) -> dict:
    """Fetch AlphaFold entry metadata for a UniProt accession.
    Returns dict with keys including 'amAnnotationsUrl' for AlphaMissense scores.
    Raises ValueError on failure.
    """
    url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id.upper()}"
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if isinstance(data, list) and data:
            return data[0]
        raise ValueError(f"No AlphaFold entry for {uniprot_id}")
    except urllib.error.HTTPError as e:
        raise ValueError(f"EBI API HTTP {e.code} for {uniprot_id}") from e
    except Exception as e:
        raise ValueError(str(e)) from e


def fetch_alphafold_missense_scores(uniprot_id: str) -> dict:
    """Download AlphaMissense per-variant scores for a UniProt accession.

    Returns dict:
        {
          "uniprot_id": str,
          "scores": dict[int, dict[str, float]],  # pos (1-based) -> {mut_aa: score}
          "mean_per_position": list[float],        # mean pathogenicity per position (1-based ordered)
          "seq_length": int,
        }

    Scores are AlphaMissense pathogenicity (0=benign, 1=pathogenic).
    Raises ValueError if data unavailable.
    """
    entry = fetch_alphafold_entry(uniprot_id)
    am_url = entry.get("amAnnotationsUrl")
    if not am_url:
        raise ValueError(f"No AlphaMissense data available for {uniprot_id}")

    try:
        with urllib.request.urlopen(am_url, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
    except Exception as e:
        raise ValueError(f"Failed to download AlphaMissense scores: {e}") from e

    # Parse CSV: protein_variant, am_pathogenicity, am_class
    # protein_variant format: A123G (wt_aa + 1based_pos + mut_aa)
    scores: dict[int, dict[str, float]] = {}
    reader = csv.DictReader(io.StringIO(raw))
    for row in reader:
        variant = row.get("protein_variant", "")
        if len(variant) < 3:
            continue
        try:
            pos = int(variant[1:-1])
            mut_aa = variant[-1]
            path = float(row.get("am_pathogenicity", 0.5))
            if pos not in scores:
                scores[pos] = {}
            scores[pos][mut_aa] = path
        except (ValueError, IndexError):
            continue

    if not scores:
        raise ValueError(f"AlphaMissense CSV parsed but no data found for {uniprot_id}")

    max_pos = max(scores.keys())
    mean_per_pos = []
    for p in range(1, max_pos + 1):
        vals = list(scores.get(p, {}).values())
        mean_per_pos.append(sum(vals) / len(vals) if vals else 0.5)

    return {
        "uniprot_id": uniprot_id.upper(),
        "scores": scores,
        "mean_per_position": mean_per_pos,
        "seq_length": max_pos,
    }
