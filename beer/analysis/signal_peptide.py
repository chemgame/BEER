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

# Small neutral residues allowed at signal-peptide cleavage (-3, -1) positions
_SMALL_NEUTRAL = frozenset('AGSTC')

# Small neutral residues for GPI omega site
_OMEGA_AA = frozenset('ASTDNGC')


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _kd_mean(seq_sub: str) -> float:
    """Mean Kyte-Doolittle score for a subsequence."""
    if not seq_sub:
        return 0.0
    return sum(KYTE_DOOLITTLE.get(aa, 0.0) for aa in seq_sub) / len(seq_sub)


def _find_best_hydrophobic_window(
    seq: str,
    start: int,
    end: int,
    min_win: int = 10,
    max_win: int = 15,
) -> tuple[int, int, float]:
    """Return (h_start, h_end, mean_kd) for the best hydrophobic window.

    Searches all windows of length *min_win* to *max_win* within seq[start:end]
    for the one with the highest mean KD score.  The mean KD is returned as a
    continuous value; no hard threshold is applied here.

    von Heijne (1986) defined the h-region as typically 6–12 residues with
    clearly positive mean hydrophobicity (KD > 0).  No specific numerical
    threshold for mean KD was given in that paper.

    Returns
    -------
    tuple (h_start, h_end, mean_kd)
        All indices are into the *original* seq (not the sub-slice).
    """
    best_score = -999.0
    best_start = start
    best_end = start + min_win

    for w in range(min_win, max_win + 1):
        for i in range(start, end - w + 1):
            score = _kd_mean(seq[i:i + w])
            if score > best_score:
                best_score = score
                best_start = i
                best_end = i + w

    return best_start, best_end, best_score


# ---------------------------------------------------------------------------
# Signal peptide prediction
# ---------------------------------------------------------------------------

def predict_signal_peptide(seq: str) -> dict:
    """Predict signal peptide using the von Heijne three-region (n, h, c) model.

    Only the first 70 residues are analysed; signal peptides are N-terminal
    features.  The algorithm follows the classic three-region description:

    * **n-region** (positions 1-5): basic residues (K, R) provide a positive
      charge that targets the ribosome-translocon.
    * **h-region** (within the first 30 aa): the most hydrophobic stretch of
      7–15 residues.  von Heijne (1986) described the h-region as having
      clearly positive mean hydrophobicity (KD > 0); no hard numerical cutoff
      is applied here.  The mean KD of the best window is reported as a
      continuous value.
    * **c-region** (3-7 aa after h-region): ends with AXA or SXA (-3/-1 rule).
    * **Cleavage site**: immediately C-terminal to the c-region.

    Parameters
    ----------
    seq:
        Full protein sequence in single-letter uppercase code.

    Returns
    -------
    dict
        Keys: ``n_end``, ``h_start``, ``h_end``, ``h_length``, ``c_start``,
        ``c_has_axa``, ``cleavage_site``, ``h_region_seq``, ``h_region_score``,
        ``n_score``.

    References
    ----------
    von Heijne, G. (1986) Nucleic Acids Res. 14(11):4683-4690.
    """
    region = seq[:70]
    n = len(region)

    # ---- n-region: count K/R in first 5 positions ----
    n_end = min(5, n)
    n_score_raw = sum(1 for aa in region[:n_end] if aa in 'KR')

    # ---- h-region: best hydrophobic window within first 30 aa ----
    # Mean KD is reported as a continuous value; no hard threshold applied.
    h_search_end = min(30, n)
    h_start, h_end, h_kd = _find_best_hydrophobic_window(
        region, 0, h_search_end, min_win=7, max_win=15
    )
    h_length = h_end - h_start
    h_region_seq = region[h_start:h_end]

    # ---- c-region: 3-7 aa after h-region, look for AXA motif ----
    c_start = h_end
    c_end_max = min(c_start + 7, n)
    # Search for best cleavage position: -3 and -1 must be small neutral
    cleavage_site = -1
    c_has_axa = False
    for cs in range(c_start + 2, c_end_max + 1):
        # positions -3 and -1 relative to cs (0-based)
        pos_m3 = cs - 3
        pos_m1 = cs - 1
        if 0 <= pos_m3 < n and 0 <= pos_m1 < n:
            if region[pos_m3] in _SMALL_NEUTRAL and region[pos_m1] in _SMALL_NEUTRAL:
                c_has_axa = True
                cleavage_site = cs  # cleavage after position cs (1-based: cs+1)
                break

    # If no AXA found, default cleavage site estimate is end of c-region
    if cleavage_site == -1:
        cleavage_site = min(c_start + 4, n)

    d_score = _d_score(h_kd, n_score_raw, c_has_axa)

    return {
        'n_end': n_end,
        'h_start': h_start,
        'h_end': h_end,
        'h_length': h_length,
        'c_start': c_start,
        'c_has_axa': c_has_axa,
        'cleavage_site': cleavage_site,
        'h_region_seq': h_region_seq,
        'h_region_score': round(h_kd, 4),
        'n_score': n_score_raw,
        'd_score': d_score,
    }


def _d_score(h_kd: float, n_basic: int, has_axa: bool) -> float:
    """SignalP-inspired discriminant score P(SP) in [0, 1].

    Combines h-region hydrophobicity (dominant term), n-region basicity, and
    presence of the AXA cleavage motif into a single confidence value via a
    calibrated logistic function.

    Calibration anchors:
      * Strong SP (h_kd=3.0, n_basic=2, AXA): P ≈ 0.91
      * Typical SP (h_kd=2.3, n_basic=1, AXA): P ≈ 0.79
      * Non-SP  (h_kd=0.3, n_basic=0, no AXA): P ≈ 0.10

    References: Nielsen et al. 1997 (SignalP 1.0); Bendtsen et al. 2004
    (SignalP 2.0 D-score weighting scheme).
    """
    h_norm = min(max(h_kd, 0.0), 3.5) / 3.5
    logit = 4.0 * h_norm + 0.3 * n_basic + 0.8 * int(has_axa) - 2.5
    return round(1.0 / (1.0 + math.exp(-logit)), 3)


# ---------------------------------------------------------------------------
# GPI anchor prediction
# ---------------------------------------------------------------------------

def predict_gpi_anchor(seq: str) -> dict:
    """Predict GPI anchor signal using the Eisenhaber et al. 1999 model.

    Analyses only the last 50 residues.  A GPI anchor requires three elements:

    * **omega site**: small neutral amino acid at C-terminal -8 to -11.
    * **Spacer**: 5-10 aa after omega.
    * **Hydrophobic tail**: last 8-15 aa (Eisenhaber et al. 1999 defined this
      as the last 11-20 residues; positional logic is used here).
      The mean KD of the C-terminal region is reported as a continuous value;
      no hard KD threshold is applied.

    Parameters
    ----------
    seq:
        Full protein sequence in single-letter uppercase code.

    Returns
    -------
    dict
        Keys: ``omega_found``, ``omega_position``, ``omega_aa``,
        ``spacer_length``, ``spacer_ok``, ``tail_start``, ``tail_seq``,
        ``tail_kd_mean``.

    References
    ----------
    Eisenhaber, B., Bork, P. & Eisenhaber, F. (1999) J. Mol. Biol.
    292(3):741-758.
    """
    n = len(seq)
    tail_region = seq[max(0, n - 50):]
    offset = max(0, n - 50)  # offset into original seq
    tn = len(tail_region)

    # ---- Hydrophobic tail: last 8-15 aa ----
    # Mean KD reported as continuous value; no threshold applied.
    tail_len = min(15, tn)
    tail_start_local = tn - tail_len
    tail_seq = tail_region[tail_start_local:]
    tail_kd = _kd_mean(tail_seq)

    # ---- omega-site: small neutral at positions -8 to -11 from the C-terminus ----
    # In tail_region coordinates: positions tn-11 to tn-8
    omega_pos_local = -1
    omega_aa = ''
    for rel in range(tn - 11, tn - 7):  # -11, -10, -9, -8
        if 0 <= rel < tn and tail_region[rel] in _OMEGA_AA:
            omega_pos_local = rel
            omega_aa = tail_region[rel]
            break  # take the most N-terminal (most conservative)

    omega_found = omega_pos_local >= 0

    # ---- Spacer: residues between omega and the hydrophobic tail ----
    if omega_found:
        spacer_len = tail_start_local - omega_pos_local - 1
    else:
        spacer_len = 0
    spacer_ok = 5 <= spacer_len <= 10

    omega_position = (offset + omega_pos_local + 1) if omega_found else -1
    tail_start_global = offset + tail_start_local + 1  # 1-based

    return {
        'omega_found': omega_found,
        'omega_position': omega_position,
        'omega_aa': omega_aa,
        'spacer_length': spacer_len,
        'spacer_ok': spacer_ok,
        'tail_start': tail_start_global,
        'tail_seq': tail_seq,
        'tail_kd_mean': round(tail_kd, 4),
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
            f" &nbsp;<em>(ESM2 650M, AUROC 0.9999)</em></td></tr>"
        )
        _sp_note = (
            "BiLSTM: ESM2 650M BiLSTM head trained on UniProt Swiss-Prot signal-peptide annotations "
            "(AUROC 0.9999 on held-out test set); score shown is mean per-residue probability over "
            "N-terminal 70 residues. Threshold: score &ge; 0.50."
        )
    else:
        sp = predict_signal_peptide(seq)
        if sp['cleavage_site'] > 0 and sp['cleavage_site'] <= len(seq):
            cs = sp['cleavage_site']
            cs_context = seq[max(0, cs - 5):cs] + " | " + seq[cs:min(len(seq), cs + 5)]
        else:
            cs_context = "N/A"
        d = sp['d_score']
        d_label = "likely SP" if d >= 0.5 else "unlikely SP"
        sp_rows = (
            f"<tr><td><b>D-score P(SP)</b></td>"
            f"<td><b>{d:.3f}</b> &mdash; {d_label}</td></tr>"
            f"<tr><td>n-region basic residues (K,R in pos 1&ndash;5)</td><td>{sp['n_score']}</td></tr>"
            f"<tr><td>h-region position</td>"
            f"<td>{sp['h_start']+1}&ndash;{sp['h_end']} ({sp['h_length']} aa)</td></tr>"
            f"<tr><td>h-region sequence</td><td><code>{sp['h_region_seq']}</code></td></tr>"
            f"<tr><td>h-region mean KD hydrophobicity</td><td>{sp['h_region_score']:.3f}</td></tr>"
            f"<tr><td>c-region AXA cleavage motif found</td><td>{'Yes' if sp['c_has_axa'] else 'No'}</td></tr>"
            f"<tr><td>Predicted cleavage site (after pos)</td>"
            f"<td>{sp['cleavage_site']} &nbsp;[{cs_context}]</td></tr>"
        )
        _sp_note = (
            "D-score: logistic discriminant combining h-region KD, n-region basicity, and AXA motif "
            "(von Heijne 1986; Nielsen et al. 1997 SignalP; Bendtsen et al. 2004). "
            "Threshold: D &ge; 0.50."
        )

    _sp_badge = method_badge("ESM2 BiLSTM", "bilstm") if bilstm_scores is not None else method_badge("classical", "classical")
    sp_html = (
        f"<h2>Signal Peptide {_sp_badge}</h2>"
        "<table>"
        "<tr><th>Parameter</th><th>Value</th></tr>"
        f"{sp_rows}"
        "</table>"
        f"<p class='note'>{_sp_note}</p>"
    )

    # ---- GPI anchor table ----
    omega_str = (
        f"Position {gpi['omega_position']} ({gpi['omega_aa']})"
        if gpi['omega_found'] else "Not found"
    )
    spacer_ok_str = f"{gpi['spacer_length']} aa ({'within' if gpi['spacer_ok'] else 'outside'} 5&ndash;10 aa range)"

    gpi_rows = (
        f"<tr><td>&omega;-site (small neutral at &minus;8 to &minus;11)</td><td>{omega_str}</td></tr>"
        f"<tr><td>Spacer length</td><td>{spacer_ok_str}</td></tr>"
        f"<tr><td>Hydrophobic tail start</td><td>{gpi['tail_start']}</td></tr>"
        f"<tr><td>Hydrophobic tail sequence</td><td><code>{gpi['tail_seq']}</code></td></tr>"
        f"<tr><td>Tail mean KD hydrophobicity</td><td>{gpi['tail_kd_mean']:.3f}</td></tr>"
    )

    gpi_html = (
        "<h2>GPI Anchor Features (Eisenhaber et al. 1999)</h2>"
        "<table>"
        "<tr><th>Parameter</th><th>Value</th></tr>"
        f"{gpi_rows}"
        "</table>"
        "<p class='note'>"
        "Eisenhaber, B., Bork, P. &amp; Eisenhaber, F. (1999) J. Mol. Biol. 292:741. "
        "Reports structural features only (&omega;-site, spacer, hydrophobic tail mean KD). "
        "The mean KD of the C-terminal region is reported as a continuous value; "
        "no binary hydrophobicity threshold is applied. "
        "For high-confidence GPI anchor prediction use GPI-SOM or PredGPI."
        "</p>"
    )

    return _s + sp_html + gpi_html
