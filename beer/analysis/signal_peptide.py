"""Signal peptide and GPI anchor prediction."""
from __future__ import annotations

import math

from beer.constants import KYTE_DOOLITTLE
from beer.reports.css import make_style_tag, method_badge


def calc_signal_peptide_profile(
    seq: str,
    embedder=None,
    head: dict | None = None,
) -> "list[float] | None":
    """Per-residue BiLSTM signal-peptide probability (0–1), or None if unavailable."""
    from beer.utils.structure import bilstm_predict
    return bilstm_predict(seq, embedder, head)

# Small neutral residues eligible for the GPI ω (cleavage) site
_OMEGA_AA = frozenset('ASTDNGC')
# ω+2 residue must be G, A, S, or T (Eisenhaber et al. 1999)
_OMEGA_PLUS2_AA = frozenset('GAST')


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _kd_mean(seq_sub: str) -> float:
    """Mean Kyte-Doolittle score for a subsequence."""
    if not seq_sub:
        return 0.0
    return sum(KYTE_DOOLITTLE.get(aa, 0.0) for aa in seq_sub) / len(seq_sub)


# ---------------------------------------------------------------------------
# GPI anchor prediction
# ---------------------------------------------------------------------------

def predict_gpi_anchor(seq: str) -> dict:
    """Predict GPI anchor signal using the Eisenhaber et al. 1999 model.

    GPI signal structure (C-terminal precursor region):
    ``[protein]--[ω site]--[GPI signal: linker + hydrophobic domain]--COOH``

    The ω site is the GPI attachment/cleavage residue; residues C-terminal to ω
    form the GPI signal region (typically 8-20 aa combining a short linker and a
    hydrophobic domain for ER membrane insertion).

    Three Eisenhaber 1999 criteria are checked:

    * **ω site**: small neutral residue (A,C,D,G,N,S,T) at position -8 to -20
      from the C-terminus (leaving 8-20 residues downstream for the GPI signal).
    * **ω+2 residue**: must be G, A, S, or T.
    * **Charge**: at most 3 charged residues (K,R,D,E) in the GPI signal region.

    The mean KD of the GPI signal region is reported as a continuous value;
    no hard hydrophobicity threshold is applied.
    For high-confidence GPI anchor prediction use GPI-SOM or PredGPI.

    Parameters
    ----------
    seq:
        Full protein sequence in single-letter uppercase code.

    Returns
    -------
    dict
        Keys: ``omega_found``, ``omega_position``, ``omega_aa``,
        ``omega_plus2_ok``, ``gpi_signal_seq``, ``gpi_signal_len``,
        ``gpi_kd_mean``, ``charge_ok``, ``has_gpi``.

    References
    ----------
    Eisenhaber, B., Bork, P. & Eisenhaber, F. (1999) J. Mol. Biol.
    292(3):741-758.
    """
    n = len(seq)
    tail_region = seq[max(0, n - 40):]
    offset = max(0, n - 40)
    tn = len(tail_region)

    # Search C→N for the most C-terminal qualifying ω site that leaves 8-20
    # residues downstream (Eisenhaber 1999; GPI-SOM/PredGPI convention).
    omega_pos_local = -1
    omega_aa = ''
    for rel in range(tn - 8, max(-1, tn - 21), -1):
        if tail_region[rel] in _OMEGA_AA:
            omega_pos_local = rel
            omega_aa = tail_region[rel]
            break

    omega_found = omega_pos_local >= 0

    # ω+2 residue must be G, A, S, or T (Eisenhaber 1999 criterion 2)
    omega_plus2_ok = (
        omega_found
        and (omega_pos_local + 2) < tn
        and tail_region[omega_pos_local + 2] in _OMEGA_PLUS2_AA
    )

    # GPI signal region: all residues from ω+1 to C-terminus
    if omega_found:
        gpi_signal_seq = tail_region[omega_pos_local + 1:]
        gpi_signal_len = len(gpi_signal_seq)
        gpi_kd_mean = _kd_mean(gpi_signal_seq)
        n_charged = sum(1 for aa in gpi_signal_seq if aa in 'KRDE')
        charge_ok = n_charged <= 3
        length_ok = 8 <= gpi_signal_len <= 20
        has_gpi = omega_plus2_ok and charge_ok and length_ok
    else:
        gpi_signal_seq = ''
        gpi_signal_len = 0
        gpi_kd_mean = 0.0
        n_charged = 0
        charge_ok = False
        has_gpi = False

    omega_position = (offset + omega_pos_local + 1) if omega_found else -1

    return {
        'omega_found': omega_found,
        'omega_position': omega_position,
        'omega_aa': omega_aa,
        'omega_plus2_ok': omega_plus2_ok,
        'gpi_signal_seq': gpi_signal_seq,
        'gpi_signal_len': gpi_signal_len,
        'gpi_kd_mean': round(gpi_kd_mean, 4),
        'charge_ok': charge_ok,
        'has_gpi': has_gpi,
    }


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def format_signal_report(
    seq: str,
    style_tag: str,
    bilstm_scores: "list[float] | None" = None,
) -> str:
    """Generate HTML section for signal peptide and GPI anchor predictions.

    Parameters
    ----------
    seq:
        Protein sequence in single-letter uppercase code.
    style_tag:
        Accent colour hex string (e.g. ``"#4361ee"``).
    bilstm_scores:
        Optional per-residue BiLSTM signal-peptide probabilities.
    """
    accent = style_tag if style_tag else "#4361ee"
    _s = make_style_tag(accent)

    gpi = predict_gpi_anchor(seq)

    # ---- Signal peptide table ----
    if bilstm_scores is not None and len(bilstm_scores) >= 10:
        n70 = min(70, len(bilstm_scores))
        bilstm_sp_score = sum(bilstm_scores[:n70]) / n70
        bl_label = "likely SP" if bilstm_sp_score >= 0.5 else "unlikely SP"
        sp_rows = (
            f"<tr><td><b>BiLSTM SP score (mean, pos 1&ndash;{n70})</b></td>"
            f"<td><b>{bilstm_sp_score:.3f}</b> &mdash; {bl_label}"
            f" &nbsp;<em>(ESMC 600M, AUROC 0.9999)</em></td></tr>"
        )
        _sp_note = (
            "BiLSTM: ESMC 600M BiLSTM head trained on UniProt Swiss-Prot signal-peptide annotations "
            "(AUROC 0.9999 on held-out test set); score shown is mean per-residue probability over "
            "N-terminal 70 residues. Threshold: score &ge; 0.50."
        )
        _sp_badge = method_badge("ESMC BiLSTM", "bilstm")
        sp_html = (
            f"<h2>Signal Peptide {_sp_badge}</h2>"
            "<table>"
            "<tr><th>Parameter</th><th>Value</th></tr>"
            f"{sp_rows}"
            "</table>"
            f"<p class='note'>{_sp_note}</p>"
        )
    else:
        sp_html = (
            "<h2>Signal Peptide</h2>"
            "<p class='note'>Signal peptide AI head not available.</p>"
        )

    # ---- GPI anchor table ----
    if gpi['omega_found']:
        omega_str = f"Position {gpi['omega_position']} ({gpi['omega_aa']})"
        plus2_str = "&#10003; yes" if gpi['omega_plus2_ok'] else "&#10007; no"
        sig_len = gpi['gpi_signal_len']
        sig_len_str = (
            f"{sig_len} aa "
            f"({'&#10003; 8&ndash;20' if 8 <= sig_len <= 20 else '&#10007; outside 8&ndash;20'})"
        )
        n_charged = sum(1 for aa in gpi['gpi_signal_seq'] if aa in 'KRDE')
        charge_str = f"{n_charged} ({'&#10003; &le;3' if gpi['charge_ok'] else '&#10007; &gt;3'})"
        pred_str = (
            "<b style='color:#16a34a'>GPI anchor signal predicted</b>"
            if gpi['has_gpi'] else "Criteria not met"
        )
        tail_row = (
            f"<tr><td>GPI signal region (&omega;+1 to C-term)</td>"
            f"<td><code>{gpi['gpi_signal_seq']}</code></td></tr>"
        )
    else:
        omega_str = "Not found in C-terminal 40 residues"
        plus2_str = sig_len_str = charge_str = "N/A"
        pred_str = "No &omega;-site found"
        tail_row = ""

    gpi_rows = (
        f"<tr><td>&omega;-site (A/C/D/G/N/S/T, position &minus;8 to &minus;20 from C-term)</td>"
        f"<td>{omega_str}</td></tr>"
        f"<tr><td>&omega;+2 amino acid (must be G/A/S/T)</td><td>{plus2_str}</td></tr>"
        + tail_row +
        f"<tr><td>GPI signal region length (8&ndash;20 aa)</td><td>{sig_len_str}</td></tr>"
        f"<tr><td>Mean KD hydrophobicity of GPI signal</td><td>{gpi['gpi_kd_mean']:.3f}</td></tr>"
        f"<tr><td>Charged residues in GPI signal (&le;3)</td><td>{charge_str}</td></tr>"
        f"<tr><td><b>Prediction</b></td><td>{pred_str}</td></tr>"
    )

    gpi_html = (
        "<h2>GPI Anchor Features (Eisenhaber et al. 1999)</h2>"
        "<table>"
        "<tr><th>Parameter</th><th>Value</th></tr>"
        f"{gpi_rows}"
        "</table>"
        "<p class='note'>"
        "Eisenhaber, B., Bork, P. &amp; Eisenhaber, F. (1999) J. Mol. Biol. 292:741. "
        "Criteria: small neutral &omega;-site leaving 8&ndash;20 C-terminal residues; "
        "&omega;+2 is G/A/S/T; &le;3 charged residues in the GPI signal region. "
        "KD hydrophobicity reported as continuous value; no hard threshold applied. "
        "For high-confidence GPI anchor prediction use GPI-SOM or PredGPI."
        "</p>"
    )

    return _s + sp_html + gpi_html
