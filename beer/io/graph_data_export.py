"""Graph data export — extract the underlying data for each BEER graph.

Each public function returns ``(suggested_filename, content_str, extension)``
where *extension* is ``"csv"`` or ``"json"``.  The caller is responsible for
writing the file.

CSV files use comma separators and include a header row.
JSON files are pretty-printed with 2-space indent.
"""
from __future__ import annotations

import json
import math
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _csv(header: list[str], rows: list[list]) -> str:
    lines = [",".join(str(h) for h in header)]
    for row in rows:
        lines.append(",".join("" if v is None else str(v) for v in row))
    return "\n".join(lines) + "\n"


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str)


def _residue_profile_csv(
    values: list[float],
    col_name: str,
    offset: int = 0,
) -> str:
    """Single per-residue (or per-window) column → CSV."""
    header = ["position", col_name]
    rows = [[i + 1 + offset, round(float(v), 6)] for i, v in enumerate(values)]
    return _csv(header, rows)


def _two_profile_csv(
    xs: list[float],
    ys: list[float],
    xcol: str,
    ycol: str,
) -> str:
    header = [xcol, ycol]
    rows = [[round(float(x), 4), round(float(y), 6)] for x, y in zip(xs, ys)]
    return _csv(header, rows)


# ---------------------------------------------------------------------------
# Per-graph extractors
# Returns (filename_stem, content, extension)
# ---------------------------------------------------------------------------

def _amino_acid_composition(ad: dict, _extra: dict) -> tuple[str, str, str]:
    counts = ad.get("aa_counts", {})
    freqs  = ad.get("aa_freq",   {})
    header = ["amino_acid", "count", "frequency"]
    rows   = sorted(
        [[aa, counts.get(aa, 0), round(float(freqs.get(aa, 0.0)), 6)]
         for aa in set(list(counts) + list(freqs))],
        key=lambda r: r[0],
    )
    return "aa_composition", _csv(header, rows), "csv"


def _hydrophobicity_profile(ad: dict, _extra: dict) -> tuple[str, str, str]:
    prof   = ad.get("hydro_profile", [])
    seq    = ad.get("seq", "")
    win    = ad.get("window_size", 9)
    offset = (len(seq) - len(prof)) // 2 if seq else 0
    xs = [i + 1 + offset for i in range(len(prof))]
    content = _two_profile_csv(xs, prof, "window_center", "kd_hydrophobicity")
    return "hydrophobicity_profile", content, "csv"


def _local_charge_profile(ad: dict, _extra: dict) -> tuple[str, str, str]:
    prof = ad.get("ncpr_profile", [])
    seq  = ad.get("seq", "")
    win  = ad.get("window_size", 9)
    offset = (len(seq) - len(prof)) // 2 if seq else 0
    xs = [i + 1 + offset for i in range(len(prof))]
    content = _two_profile_csv(xs, prof, "window_center", "ncpr")
    return "local_charge_profile", content, "csv"


def _local_complexity(ad: dict, _extra: dict) -> tuple[str, str, str]:
    prof = ad.get("entropy_profile", [])
    seq  = ad.get("seq", "")
    offset = (len(seq) - len(prof)) // 2 if seq else 0
    xs = [i + 1 + offset for i in range(len(prof))]
    content = _two_profile_csv(xs, prof, "window_center", "shannon_entropy_bits")
    return "local_complexity", content, "csv"


def _disorder_profile(ad: dict, _extra: dict) -> tuple[str, str, str]:
    scores = ad.get("disorder_scores", [])
    content = _residue_profile_csv(scores, "disorder_score")
    return "disorder_profile", content, "csv"


def _coiled_coil(ad: dict, _extra: dict) -> tuple[str, str, str]:
    prof = ad.get("cc_profile", [])
    content = _residue_profile_csv(prof, "p_coiled_coil")
    return "coiled_coil_profile", content, "csv"


def _scd_profile(ad: dict, _extra: dict) -> tuple[str, str, str]:
    prof = ad.get("scd_profile", [])
    seq  = ad.get("seq", "")
    win  = 20
    offset = (len(seq) - len(prof)) // 2 if seq else 0
    xs = [i + 1 + offset for i in range(len(prof))]
    content = _two_profile_csv(xs, prof, "window_center", "scd")
    return "scd_profile", content, "csv"


def _tm_topology(ad: dict, _extra: dict) -> tuple[str, str, str]:
    helices = ad.get("tm_helices", [])
    header  = ["helix_number", "start_1based", "end_1based", "length", "score", "orientation", "source"]
    rows    = [
        [i + 1, h["start"] + 1, h["end"] + 1,
         h["end"] - h["start"] + 1,
         round(h.get("score", 0.0), 4),
         h.get("orientation", ""),
         h.get("source", "")]
        for i, h in enumerate(helices)
    ]
    return "tm_helices", _csv(header, rows), "csv"


def _aggregation_profile(ad: dict, _extra: dict) -> tuple[str, str, str]:
    prof_zygg = ad.get("aggr_profile", [])
    prof_esm2 = ad.get("aggr_profile_esm2")
    if prof_esm2 and len(prof_esm2) == len(prof_zygg):
        header = ["position", "zyggregator_score", "esm2_score"]
        rows   = [[i + 1, round(float(z), 6), round(float(e), 6)]
                  for i, (z, e) in enumerate(zip(prof_zygg, prof_esm2))]
    else:
        header = ["position", "zyggregator_score"]
        rows   = [[i + 1, round(float(v), 6)] for i, v in enumerate(prof_zygg)]
    return "aggregation_profile", _csv(header, rows), "csv"


def _solubility_profile(ad: dict, _extra: dict) -> tuple[str, str, str]:
    seq = ad.get("seq", "")
    try:
        from beer.analysis.aggregation import calc_camsolmt_score
        scores = calc_camsolmt_score(seq)
    except Exception:
        scores = []
    content = _residue_profile_csv(scores, "camsolmt_score")
    return "solubility_profile", content, "csv"


def _plaac_profile(ad: dict, _extra: dict) -> tuple[str, str, str]:
    plaac = ad.get("plaac", {})
    prof  = plaac.get("profile", []) if isinstance(plaac, dict) else []
    content = _residue_profile_csv(prof, "plaac_log_odds")
    return "plaac_profile", content, "csv"


def _rbp_profile(ad: dict, _extra: dict) -> tuple[str, str, str]:
    prof = ad.get("rbp_profile", [])
    seq  = ad.get("seq", "")
    win  = 11
    offset = (len(seq) - len(prof)) // 2 if seq else 0
    xs = [i + 1 + offset for i in range(len(prof))]
    content = _two_profile_csv(xs, prof, "window_center", "catrapid_score")
    return "rna_binding_profile", content, "csv"


def _hydrophobic_moment(ad: dict, _extra: dict) -> tuple[str, str, str]:
    alpha = ad.get("moment_alpha", [])
    beta  = ad.get("moment_beta",  [])
    seq   = ad.get("seq", "")
    n     = max(len(alpha), len(beta))
    header = ["position", "moment_alpha_uH", "moment_beta_uH"]
    rows   = [
        [i + 1,
         round(float(alpha[i]), 6) if i < len(alpha) else "",
         round(float(beta[i]),  6) if i < len(beta)  else ""]
        for i in range(n)
    ]
    return "hydrophobic_moment", _csv(header, rows), "csv"


def _annotation_track(ad: dict, _extra: dict) -> tuple[str, str, str]:
    seq      = ad.get("seq", "")
    dis      = ad.get("disorder_scores", [])
    hydro    = ad.get("hydro_profile",   [])
    aggr     = ad.get("aggr_profile",    [])
    n        = len(seq)
    win      = ad.get("window_size", 9)
    h_offset = (n - len(hydro)) // 2 if seq else 0

    header = ["position", "residue", "disorder_score",
              "hydrophobicity", "aggregation_score"]
    rows = []
    for i, aa in enumerate(seq):
        dis_v  = round(float(dis[i]),  6) if i < len(dis)  else ""
        aggr_v = round(float(aggr[i]), 6) if i < len(aggr) else ""
        j_hyd  = i - h_offset
        hyd_v  = round(float(hydro[j_hyd]), 6) if 0 <= j_hyd < len(hydro) else ""
        rows.append([i + 1, aa, dis_v, hyd_v, aggr_v])
    return "annotation_track", _csv(header, rows), "csv"


def _linear_sequence_map(ad: dict, _extra: dict) -> tuple[str, str, str]:
    return _annotation_track(ad, _extra)[0], _annotation_track(ad, _extra)[1], "csv"


def _domain_architecture(ad: dict, extra: dict) -> tuple[str, str, str]:
    pfam    = extra.get("pfam_domains", [])
    tm      = ad.get("tm_helices", [])
    dis     = ad.get("disorder_scores", [])
    seq     = ad.get("seq", "")
    n       = len(seq)
    obj: dict[str, Any] = {
        "sequence_length": n,
        "pfam_domains": [
            {"name": d.get("name", ""), "accession": d.get("accession", ""),
             "start": d.get("start", 0), "end": d.get("end", 0),
             "evalue": d.get("evalue", None), "description": d.get("description", "")}
            for d in pfam
        ],
        "tm_helices": [
            {"start_1based": h["start"] + 1, "end_1based": h["end"] + 1,
             "orientation": h.get("orientation", ""), "source": h.get("source", "")}
            for h in tm
        ],
        "disordered_regions": _disorder_runs(dis),
    }
    return "domain_architecture", _json(obj), "json"


def _disorder_runs(scores: list[float], threshold: float = 0.5) -> list[dict]:
    runs, in_r, start = [], False, 0
    for i, v in enumerate(scores):
        if v > threshold and not in_r:
            in_r, start = True, i + 1
        elif v <= threshold and in_r:
            runs.append({"start": start, "end": i})
            in_r = False
    if in_r:
        runs.append({"start": start, "end": len(scores)})
    return runs


def _cleavage_map(ad: dict, _extra: dict) -> tuple[str, str, str]:
    sites  = ad.get("prot_sites", {})
    header = ["enzyme", "cut_position_1based"]
    rows   = []
    for enzyme, cuts in sorted(sites.items()):
        for pos in cuts:
            rows.append([enzyme, pos])
    return "cleavage_map", _csv(header, rows), "csv"


def _linear_motifs(ad: dict, _extra: dict) -> tuple[str, str, str]:
    # Motifs are derived from sequence, re-scan on export
    seq = ad.get("seq", "")
    try:
        from beer.utils.structure import scan_linear_motifs
        motifs = scan_linear_motifs(seq)
    except Exception:
        motifs = []
    header = ["name", "start_1based", "end_1based", "match", "description"]
    rows   = [[m["name"], m["start"] + 1, m["end"] + 1,
               m["match"], m.get("description", "")] for m in motifs]
    return "linear_motifs", _csv(header, rows), "csv"


def _isoelectric_focus(ad: dict, _extra: dict) -> tuple[str, str, str]:
    seq = ad.get("seq", "")
    try:
        from beer.utils.biophysics import calc_net_charge
        pka = ad.get("custom_pka", None)
        ph_vals  = [round(ph * 0.1, 1) for ph in range(0, 141)]
        charges  = [round(float(calc_net_charge(seq, ph, pka)), 4) for ph in ph_vals]
    except Exception:
        ph_vals, charges = [], []
    content = _two_profile_csv(ph_vals, charges, "pH", "net_charge")
    return "isoelectric_focus", content, "csv"


def _charge_decoration(ad: dict, _extra: dict) -> tuple[str, str, str]:
    fcr  = ad.get("fcr",  0.0)
    ncpr = ad.get("ncpr", 0.0)
    obj  = {"FCR": fcr, "NCPR": ncpr,
            "description": "Das-Pappu phase diagram coordinates"}
    return "charge_decoration", _json(obj), "json"


def _uversky_phase_plot(ad: dict, _extra: dict) -> tuple[str, str, str]:
    seq = ad.get("seq", "")
    try:
        from beer.utils.biophysics import calc_net_charge
        from beer.constants import KYTE_DOOLITTLE
        fcr  = ad.get("fcr",  0.0)
        ncpr = ad.get("ncpr", 0.0)
        mean_kd = sum(KYTE_DOOLITTLE.get(aa, 0.0) for aa in seq) / max(len(seq), 1)
        dis  = ad.get("disorder_scores", [])
        mean_dis = sum(dis) / len(dis) if dis else 0.0
        obj  = {"FCR": fcr, "NCPR": ncpr,
                "mean_KD_hydrophobicity": round(mean_kd, 4),
                "mean_disorder_score": round(mean_dis, 4),
                "seq_length": len(seq)}
    except Exception as exc:
        obj = {"error": str(exc)}
    return "uversky_phase_plot", _json(obj), "json"


def _larks(ad: dict, _extra: dict) -> tuple[str, str, str]:
    larks = ad.get("larks", [])
    header = ["start_1based", "end_1based", "sequence",
              "n_aromatic", "lc_fraction", "entropy_bits"]
    rows = [[l["start"] + 1, l["end"] + 1, l["seq"],
             l["n_arom"], l["lc_frac"], l["entropy"]] for l in larks]
    return "larks", _csv(header, rows), "csv"


def _variant_effect_map(ad: dict, extra: dict) -> tuple[str, str, str]:
    llr = extra.get("variant_llr", {})
    seq = ad.get("seq", "")
    if not llr:
        return "variant_effect_map", _json({}), "json"
    aas = sorted(next(iter(llr.values())).keys()) if llr else []
    header = ["position", "wild_type"] + aas
    rows = []
    for i, wt in enumerate(seq):
        pos_key = str(i + 1)
        if pos_key in llr:
            row = [i + 1, wt] + [round(float(llr[pos_key].get(aa, 0.0)), 4) for aa in aas]
        else:
            row = [i + 1, wt] + [""] * len(aas)
        rows.append(row)
    return "variant_effect_map", _csv(header, rows), "csv"


def _plddt_profile(ad: dict, extra: dict) -> tuple[str, str, str]:
    plddt = extra.get("plddt", [])
    content = _residue_profile_csv(plddt, "plddt_score")
    return "plddt_profile", content, "csv"


def _sticker_map(ad: dict, _extra: dict) -> tuple[str, str, str]:
    seq = ad.get("seq", "")
    sticker_set = set("FWYKRDE")
    header = ["position", "residue", "is_sticker",
              "is_aromatic_sticker", "is_electrostatic_sticker"]
    rows = [
        [i + 1, aa,
         int(aa in sticker_set),
         int(aa in "FWY"),
         int(aa in "KRDE")]
        for i, aa in enumerate(seq)
    ]
    return "sticker_map", _csv(header, rows), "csv"


def _cation_pi_map(ad: dict, _extra: dict) -> tuple[str, str, str]:
    seq    = ad.get("seq", "")
    window = 8
    arom   = set("FWY")
    basic  = set("KR")
    pairs  = []
    for i, aa_i in enumerate(seq):
        for j in range(max(0, i - window), min(len(seq), i + window + 1)):
            if j == i:
                continue
            aa_j = seq[j]
            if (aa_i in basic and aa_j in arom) or (aa_i in arom and aa_j in basic):
                pairs.append([i + 1, aa_i, j + 1, aa_j, abs(i - j)])
    header = ["pos_a", "res_a", "pos_b", "res_b", "sequence_separation"]
    return "cation_pi_pairs", _csv(header, pairs), "csv"


def _alphafold_missense(ad: dict, extra: dict) -> tuple[str, str, str]:
    am = extra.get("alphafold_missense", {})
    seq = ad.get("seq", "")
    if not am:
        return "alphafold_missense", _json({}), "json"
    aas = sorted(am.get("amino_acids", []))
    scores_dict = am.get("scores", {})
    header = ["position", "wild_type"] + aas
    rows = []
    for i, wt in enumerate(seq):
        row_scores = scores_dict.get(str(i + 1), {})
        rows.append([i + 1, wt] + [round(float(row_scores.get(aa, 0.0)), 4) for aa in aas])
    return "alphafold_missense", _csv(header, rows), "csv"


def _tandem_repeats(ad: dict, _extra: dict) -> tuple[str, str, str]:
    stats = ad.get("tandem_stats", {})
    return "tandem_repeats", _json(stats), "json"


def _proteolytic_map(ad: dict, _extra: dict) -> tuple[str, str, str]:
    return _cleavage_map(ad, _extra)


def _phosphorylation(ad: dict, _extra: dict) -> tuple[str, str, str]:
    sites = ad.get("phospho_sites", {})
    header = ["kinase", "position_1based", "residue", "pwm_score"]
    rows = []
    for kinase, hits in sorted(sites.items()):
        for h in hits:
            rows.append([kinase, h["position"], h["residue"], h["score"]])
    return "phosphorylation_sites", _csv(header, rows), "csv"


def _msa_conservation(ad: dict, extra: dict) -> tuple[str, str, str]:
    cons = extra.get("msa_conservation", [])
    if not cons:
        seqs = extra.get("msa_sequences", [])
        if seqs:
            import collections as _col
            n_seq, aln_len = len(seqs), len(seqs[0])
            for col in range(aln_len):
                chars = [seqs[s][col] for s in range(n_seq) if col < len(seqs[s])]
                distinct = set(chars)
                if len(distinct) > 1:
                    ctr = _col.Counter(chars)
                    n = len(chars)
                    h = -sum((c / n) * math.log2(c / n) for c in ctr.values() if c > 0)
                    cons.append(max(0.0, 1.0 - h / math.log2(len(distinct))))
                else:
                    cons.append(1.0)
    content = _residue_profile_csv(cons, "conservation_score")
    return "msa_conservation", content, "csv"


def _truncation_series(ad: dict, extra: dict) -> tuple[str, str, str]:
    data = extra.get("truncation_data", {})
    return "truncation_series", _json(data), "json"


def _complex_mass(ad: dict, extra: dict) -> tuple[str, str, str]:
    obj = extra.get("complex_mass", {})
    return "complex_mass", _json(obj), "json"


def _single_residue_perturbation(ad: dict, _extra: dict) -> tuple[str, str, str]:
    return _variant_effect_map(ad, _extra)


# ---------------------------------------------------------------------------
# Dispatch table: graph title → extractor
# ---------------------------------------------------------------------------

_EXTRACTORS: dict[str, Any] = {
    "Amino Acid Composition (Bar)":      _amino_acid_composition,
    "Amino Acid Composition (Pie)":      _amino_acid_composition,
    "Hydrophobicity Profile":            _hydrophobicity_profile,
    "Local Charge Profile":              _local_charge_profile,
    "Local Complexity":                  _local_complexity,
    "Disorder Profile":                  _disorder_profile,
    "Coiled-Coil Profile":               _coiled_coil,
    "SCD Profile":                       _scd_profile,
    "TM Topology":                       _tm_topology,
    "\u03b2-Aggregation Profile":        _aggregation_profile,
    "Solubility Profile":                _solubility_profile,
    "PLAAC Profile":                     _plaac_profile,
    "RNA-Binding Profile":               _rbp_profile,
    "Hydrophobic Moment":                _hydrophobic_moment,
    "Annotation Track":                  _annotation_track,
    "Linear Sequence Map":               _linear_sequence_map,
    "Domain Architecture":               _domain_architecture,
    "Cleavage Map":                      _cleavage_map,
    "Isoelectric Focus":                 _isoelectric_focus,
    "Charge Decoration":                 _charge_decoration,
    "Uversky Phase Plot":                _uversky_phase_plot,
    "LARKS":                             _larks,
    "Sticker Map":                       _sticker_map,
    "Cation\u2013\u03c0 Map":            _cation_pi_map,
    "Linear Motifs":                     _linear_motifs,
    "Variant Effect Map":                _variant_effect_map,
    "AlphaMissense":                     _alphafold_missense,
    "Single-Residue Perturbation Map":   _single_residue_perturbation,
    "pLDDT Profile":                     _plddt_profile,
    "MSA Conservation":                  _msa_conservation,
    "Truncation Series":                 _truncation_series,
    "Complex Mass":                      _complex_mass,
    "Proteolytic Map":                   _proteolytic_map,
    "Phosphorylation":                   _phosphorylation,
}


def get_graph_data(
    title: str,
    analysis_data: dict,
    extra: dict | None = None,
) -> tuple[str, str, str] | None:
    """Return ``(filename_stem, content, extension)`` for *title*, or ``None``.

    Parameters
    ----------
    title:
        The graph title as used in GRAPH_TITLES / the tree browser.
    analysis_data:
        The dict returned by ``BEERAnalysis.analyze()``.
    extra:
        Optional dict with additional data not stored in analysis_data:
        ``pfam_domains``, ``plddt``, ``variant_llr``, ``alphafold_missense``,
        ``msa_conservation``, ``truncation_data``, ``complex_mass``.
    """
    fn = _EXTRACTORS.get(title)
    if fn is None:
        return None
    return fn(analysis_data, extra or {})
