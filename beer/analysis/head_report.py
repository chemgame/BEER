"""What one prediction head found in one protein, as HTML for the report and the GUI.

Both the exported report and the GUI's AI Predictions sections use this, so a head's
threshold, its residue- or region-level findings and its protein-level call read the same
everywhere.
"""
from __future__ import annotations

from beer.analysis.calls import SITE_HEADS, protein_call, rule_text, runs_above, tm_segments
from beer.models.metrics import f1_threshold, protein_auroc

#: analysis_data key → head task name
DATA_KEY_TO_TASK: dict[str, str] = {
    "disorder_scores":         "disorder",
    "sp_bilstm_profile":       "signal_peptide",
    "tm_bilstm_profile":       "transmembrane",
    "intramem_bilstm_profile": "intramembrane",
    "cc_bilstm_profile":       "coiled_coil",
    "dna_bilstm_profile":      "dna_binding",
    "act_bilstm_profile":      "active_site",
    "bnd_bilstm_profile":      "binding_site",
    "phos_bilstm_profile":     "phosphorylation",
    "lcd_bilstm_profile":      "lcd",
    "znf_bilstm_profile":      "zinc_finger",
    "glyc_bilstm_profile":     "glycosylation",
    "ubiq_bilstm_profile":     "ubiquitination",
    "meth_bilstm_profile":     "methylation",
    "acet_bilstm_profile":     "acetylation",
    "lipid_bilstm_profile":    "lipidation",
    "disulf_bilstm_profile":   "disulfide",
    "motif_bilstm_profile":    "motif",
    "prop_bilstm_profile":     "propeptide",
    "rep_bilstm_profile":      "repeat",
    "rnabind_bilstm_profile":  "rna_binding",
    "nucbind_bilstm_profile":  "nucleotide_binding",
    "transit_bilstm_profile":  "transit_peptide",
    "ss3_h_profile":           "secondary_structure_helix",
    "ss3_e_profile":           "secondary_structure_strand",
    "ss3_c_profile":           "secondary_structure_coil",
}

MAX_SITES_LISTED = 25

_AA3 = {"A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys", "Q": "Gln", "E": "Glu",
        "G": "Gly", "H": "His", "I": "Ile", "L": "Leu", "K": "Lys", "M": "Met", "F": "Phe",
        "P": "Pro", "S": "Ser", "T": "Thr", "W": "Trp", "Y": "Tyr", "V": "Val"}


def _residue(seq: str, pos: int) -> str:
    aa = seq[pos - 1] if 0 < pos <= len(seq) else ""
    return f"{_AA3.get(aa, aa)}-{pos}" if aa else str(pos)


def _span_rows(seq: str, spans: list[tuple[int, int]]) -> str:
    return "".join(f"<tr><td>{k}</td><td>{_residue(seq, s)} – {_residue(seq, e)}</td>"
                   f"<td>{e - s + 1}</td></tr>" for k, (s, e) in enumerate(spans, 1))


def _protein_line(task: str, called: bool | None) -> str:
    if called is None:
        auc = protein_auroc(task)
        return ("<p><b>Protein-level call:</b> not made. On the v3.1 test proteins this head "
                "separates proteins with and without the feature only weakly (protein-level "
                f"AUROC {auc:.2f}); read the residue scores below instead.</p>")
    verdict = "predicted" if called else "not predicted"
    return (f"<p><b>Protein-level call: {verdict}.</b> "
            f"<span class='note'>Rule: {rule_text(task)}.</span></p>")


def head_findings_html(task: str, scores: "list[float]", seq: str = "",
                       tm_topology: "list[int] | None" = None) -> str:
    """Threshold, protein-level call and the residues or regions a head calls in *seq*."""
    if not scores:
        return "<p class='note'>No prediction available.</p>"
    n = len(scores)
    mean = sum(scores) / n

    if task == "transmembrane":
        segments = tm_segments(tm_topology) if tm_topology else []
        rows = _span_rows(seq, segments)
        body = (f"<p><b>Protein-level call: {'transmembrane protein' if segments else 'no transmembrane segment'}"
                f".</b></p>"
                + (f"<table><tr><th>#</th><th>Segment</th><th>Length (aa)</th></tr>{rows}</table>"
                   if segments else ""))
        return (f"<table><tr><th>Property</th><th>Value</th></tr>"
                f"<tr><td>Mean membrane probability</td><td>{mean:.3f}</td></tr>"
                f"<tr><td>Membrane segments (Viterbi)</td><td>{len(segments)}</td></tr></table>{body}")

    thr = f1_threshold(task)
    above = [i + 1 for i, v in enumerate(scores) if v > thr]
    table = (f"<table><tr><th>Property</th><th>Value</th></tr>"
             f"<tr><td>Mean score</td><td>{mean:.3f}</td></tr>"
             f"<tr><td>Decision threshold</td><td>{thr:.2f}</td></tr>"
             f"<tr><td>Residues above threshold</td><td>{len(above)} of {n} "
             f"({len(above) / n * 100:.1f}%)</td></tr></table>")

    call = protein_call(task, scores)
    protein = _protein_line(task, call.called) if (call.called is not None or protein_auroc(task)) else ""

    if task in SITE_HEADS:
        top = sorted(above, key=lambda p: -scores[p - 1])[:MAX_SITES_LISTED]
        if top:
            rows = "".join(f"<tr><td>{_residue(seq, p)}</td><td>{scores[p - 1]:.2f}</td></tr>"
                           for p in sorted(top))
            more = (f" Showing the {MAX_SITES_LISTED} highest-scoring."
                    if len(above) > MAX_SITES_LISTED else "")
            detail = (f"<h3 style='margin:10px 0 4px;font-size:12px'>Predicted sites</h3>"
                      f"<table><tr><th>Residue</th><th>Score</th></tr>{rows}</table>"
                      f"<p class='note'>{len(above)} residue(s) above threshold {thr:.2f}.{more}</p>")
        else:
            detail = f"<p class='note'>No residue above threshold {thr:.2f}.</p>"
    else:
        # The regions behind the call; every above-threshold run when the head makes no call.
        spans = runs_above(scores, thr) if call.called is None else call.spans
        if spans:
            detail = (f"<h3 style='margin:10px 0 4px;font-size:12px'>Predicted regions</h3>"
                      f"<table><tr><th>#</th><th>Residues</th><th>Length (aa)</th></tr>"
                      f"{_span_rows(seq, spans)}</table>")
        else:
            detail = f"<p class='note'>No region above threshold {thr:.2f}.</p>"
    return table + protein + detail
