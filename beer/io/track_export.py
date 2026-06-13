"""Interoperability exports: per-residue feature tracks (GFF3) and structure
colouring scripts (PyMOL .pml / ChimeraX .cxc).

These let an expert take BEER's predictions into a genome/feature browser or
reproduce a per-residue colouring in PyMOL / UCSF ChimeraX without re-deriving
anything by hand.
"""
from __future__ import annotations


# data_key → (GFF3 feature type, default threshold). Order = output order.
GFF3_FEATURES: list[tuple[str, str, float]] = [
    ("disorder_scores",         "disordered_region",      0.50),
    ("sp_bilstm_profile",       "signal_peptide",         0.50),
    ("tm_bilstm_profile",       "transmembrane_region",   0.50),
    ("intramem_bilstm_profile", "intramembrane_region",   0.50),
    ("cc_bilstm_profile",       "coiled_coil",            0.50),
    ("dna_bilstm_profile",      "DNA_binding_region",     0.50),
    ("rnabind_bilstm_profile",  "RNA_binding_region",     0.50),
    ("act_bilstm_profile",      "active_site",            0.50),
    ("bnd_bilstm_profile",      "binding_site",           0.50),
    ("phos_bilstm_profile",     "phosphorylation_site",   0.50),
    ("lcd_bilstm_profile",      "low_complexity_region",  0.50),
    ("znf_bilstm_profile",      "zinc_finger",            0.50),
    ("glyc_bilstm_profile",     "glycosylation_site",     0.50),
    ("ubiq_bilstm_profile",     "ubiquitination_site",    0.50),
    ("meth_bilstm_profile",     "methylation_site",       0.50),
    ("acet_bilstm_profile",     "acetylation_site",       0.50),
    ("lipid_bilstm_profile",    "lipidation_site",        0.50),
    ("disulf_bilstm_profile",   "disulfide_region",       0.50),
    ("motif_bilstm_profile",    "functional_motif",       0.50),
    ("prop_bilstm_profile",     "propeptide",             0.50),
    ("rep_bilstm_profile",      "repeat_region",          0.50),
    ("nucbind_bilstm_profile",  "nucleotide_binding",     0.50),
    ("transit_bilstm_profile",  "transit_peptide",        0.50),
]

# Per-residue tracks that make sense to colour a structure by.
COLOUR_TRACKS: list[tuple[str, str]] = [
    ("disorder_scores",        "Disorder"),
    ("tm_bilstm_profile",      "Transmembrane"),
    ("act_bilstm_profile",     "Active Site"),
    ("bnd_bilstm_profile",     "Binding Site"),
    ("dna_bilstm_profile",     "DNA-Binding"),
    ("rnabind_bilstm_profile", "RNA-Binding"),
    ("phos_bilstm_profile",    "Phosphorylation"),
    ("cc_bilstm_profile",      "Coiled-Coil"),
    ("ss3_h_profile",          "SS3 Helix"),
    ("ss3_e_profile",          "SS3 Strand"),
]


def _regions(scores: list, threshold: float) -> list[tuple[int, int, float]]:
    """Contiguous 1-based [start, end] runs where score >= threshold (+ peak)."""
    out: list[tuple[int, int, float]] = []
    n = len(scores)
    i = 0
    while i < n:
        try:
            above = float(scores[i]) >= threshold
        except (TypeError, ValueError):
            above = False
        if above:
            j = i
            while j + 1 < n and float(scores[j + 1]) >= threshold:
                j += 1
            peak = max(float(scores[k]) for k in range(i, j + 1))
            out.append((i + 1, j + 1, peak))
            i = j + 1
        else:
            i += 1
    return out


def features_to_gff3(data: dict, seq_name: str = "protein",
                     thresholds: "dict | None" = None) -> str:
    """Predicted per-residue features as a GFF3 document.

    thresholds: optional {data_key: float} overrides for the default 0.5 cut.
    """
    from beer.io.provenance import text_header
    seq = data.get("seq", "") or ""
    L = len(seq)
    name = (seq_name or "protein").replace(" ", "_") or "protein"
    lines = ["##gff-version 3", f"##sequence-region {name} 1 {max(L, 1)}"]
    lines.extend(text_header("# ").rstrip("\n").splitlines())
    thresholds = thresholds or {}
    for key, ftype, default_thr in GFF3_FEATURES:
        scores = data.get(key)
        if not scores:
            continue
        thr = float(thresholds.get(key, default_thr))
        for start, end, peak in _regions(scores, thr):
            lines.append(
                f"{name}\tBEER\t{ftype}\t{start}\t{end}\t{peak:.3f}\t.\t.\t"
                f"Note={ftype};peak_score={peak:.3f}"
            )
    return "\n".join(lines) + "\n"


def coloring_to_pymol(data: dict, track_key: str, track_label: str = "",
                      chain: str = "A") -> "str | None":
    """PyMOL .pml that loads per-residue scores into the B-factor column and
    spectrum-colours the chain (blue→white→red). Returns None if no scores."""
    scores = data.get(track_key)
    if not scores:
        return None
    from beer.io.provenance import text_header
    label = track_label or track_key
    pairs = ", ".join(f"{i}: {float(v):.4f}" for i, v in enumerate(scores, 1))
    return (
        text_header("# ", title=f"BEER per-residue colouring — {label}")
        + "# Load your structure object first, then:  run this_file.pml\n"
        f"python\n"
        f"from pymol import cmd\n"
        f"beer_scores = {{{pairs}}}\n"
        f"for _resi, _b in beer_scores.items():\n"
        f"    cmd.alter(f'chain {chain} and resi {{_resi}}', f'b={{_b}}')\n"
        f"cmd.rebuild()\n"
        f"cmd.spectrum('b', 'blue_white_red', 'chain {chain}')\n"
        f"cmd.show_as('cartoon', 'chain {chain}')\n"
        f"python end\n"
    )


def coloring_to_chimerax(data: dict, track_key: str, track_label: str = "",
                         chain: str = "A", model: str = "#1") -> "str | None":
    """ChimeraX .cxc that sets a per-residue attribute and colours by it.
    Returns None if no scores."""
    scores = data.get(track_key)
    if not scores:
        return None
    from beer.io.provenance import text_header
    label = track_label or track_key
    lines = text_header("# ", title=f"BEER per-residue colouring — {label}").rstrip("\n").splitlines()
    lines.append("# Open your structure first, then:  open this_file.cxc")
    for i, v in enumerate(scores, 1):
        lines.append(f"setattr {model}/{chain}:{i} residues beerScore {float(v):.4f}")
    lines.append(
        f"color byattribute r:beerScore {model}/{chain} "
        f"palette blue:white:red"
    )
    lines.append(f"cartoon {model}/{chain}")
    return "\n".join(lines) + "\n"


def coloring_to_bfactor_pdb(pdb_str: str, scores: list,
                            scale: float = 100.0) -> "str | None":
    """Rewrite a PDB's B-factor column with a per-residue track (× scale), so any
    viewer can reproduce the colouring via 'spectrum b' / 'color byattribute'.

    Residue i (1-based by PDB residue number) gets ``scores[i-1]``; this matches
    AlphaFold/single-chain numbering. Returns None if there is no structure/scores.
    """
    if not pdb_str or not scores:
        return None
    from beer.io.provenance import beer_version
    import datetime
    out = [f"REMARK   1 B-factor column set to a per-residue track by BEER "
           f"v{beer_version()} ({datetime.date.today().isoformat()})"]
    for line in pdb_str.splitlines():
        if line.startswith(("ATOM", "HETATM")) and len(line) >= 66:
            try:
                resi = int(line[22:26])
            except ValueError:
                out.append(line)
                continue
            if 1 <= resi <= len(scores):
                try:
                    b = round(float(scores[resi - 1]) * scale, 2)
                except (TypeError, ValueError):
                    out.append(line)
                    continue
                line = line[:60] + f"{b:6.2f}" + line[66:]
        out.append(line)
    return "\n".join(out) + "\n"


def available_colour_tracks(data: dict) -> list[tuple[str, str]]:
    """(data_key, label) for colour tracks present in *data* (computed)."""
    return [(k, lbl) for k, lbl in COLOUR_TRACKS if data.get(k)]
