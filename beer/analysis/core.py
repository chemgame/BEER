"""Analysis core orchestrator — wraps all BEER analysis modules."""
from __future__ import annotations
import math
from typing import Any

from Bio.SeqUtils.ProtParam import ProteinAnalysis as BPProteinAnalysis

from beer.constants import (
    KYTE_DOOLITTLE,
    DEFAULT_PKA,
    VALID_AMINO_ACIDS,
    STICKER_AROMATIC,
    STICKER_ALL,
    PRION_LIKE,
    DISORDER_PROMOTING,
    ORDER_PROMOTING,
    LINEAR_MOTIFS,
    HYDROPHOBICITY_SCALES,
)
from beer.reports.css import make_style_tag
from beer.utils.biophysics import (
    calc_net_charge,
    calc_kappa,
    calc_omega,
    fraction_low_complexity,
    sliding_window_hydrophobicity,
    sliding_window_ncpr,
    sliding_window_entropy,
    calc_shannon_entropy,
    count_pairs,
    sticker_spacing_stats,
)
from beer.utils.structure import (
    calc_disorder_profile,
    predict_tm_helices,
    detect_larks,
    predict_coiled_coil,
    scan_linear_motifs,
)
from beer.analysis.aggregation import (
    calc_aggregation_profile,
    calc_aggregation_profile_esm2,
    calc_solubility_stats,
    format_aggregation_report,
)
from beer.models import (
    load_disorder_head,
    load_aggregation_head,
)
from beer.analysis.signal_peptide import (
    predict_signal_peptide,
    predict_gpi_anchor,
    format_signal_report,
)
from beer.analysis.amphipathic import (
    calc_hydrophobic_moment_profile,
    predict_amphipathic_helices,
    format_amphipathic_report,
)
from beer.analysis.scd import (
    calc_scd,
    calc_scd_profile,
    calc_pos_neg_block_lengths,
    format_scd_report,
)
from beer.analysis.rnabinding import (
    calc_rbp_score,
    calc_rbp_profile,
    format_rbp_report,
)
from beer.analysis.tandem_repeats import (
    calc_repeat_stats,
    format_tandem_repeats_report,
)
from beer.analysis.proteolysis import (
    calc_proteolytic_sites,
    format_proteolysis_report,
)
from beer.utils.biophysics import calc_polyx_stretches, calc_plaac_score
from beer.embeddings.base import SequenceEmbedder


class AnalysisTools:
    @staticmethod
    def analyze_sequence(
        seq: str,
        pH_value: float = 7.0,
        window_size: int = 9,
        use_reducing: bool = False,
        pka: dict = None,
        embedder: SequenceEmbedder | None = None,
        hydro_scale: str = "Kyte-Doolittle",
        use_esm2_aggregation: bool = False,
    ) -> dict[str, Any]:
        pa          = BPProteinAnalysis(seq)
        aa_counts   = pa.count_amino_acids()
        seq_length  = len(seq)
        aa_freq     = {aa: count / seq_length * 100 for aa, count in aa_counts.items()}
        mol_weight  = pa.molecular_weight()
        iso_point   = pa.isoelectric_point()
        gravy       = pa.gravy()
        aromaticity = pa.aromaticity()
        net_charge_7  = calc_net_charge(seq, 7.0, pka)
        net_charge_pH = calc_net_charge(seq, pH_value, pka)

        n_cystine  = 0 if use_reducing else seq.count("C") // 2
        extinction = 5500 * seq.count("W") + 1490 * seq.count("Y") + 125 * n_cystine

        # --- Charge features ---
        pos_n   = sum(aa_counts.get(k, 0) for k in "KR")
        neg_n   = sum(aa_counts.get(k, 0) for k in "DE")
        fcr     = (pos_n + neg_n) / seq_length
        ncpr    = (pos_n - neg_n) / seq_length
        kappa   = calc_kappa(seq)
        ch_asym = (pos_n / neg_n) if neg_n > 0 else float('inf')


        # --- Aromatic & π features ---
        n_tyr  = aa_counts.get("Y", 0)
        n_phe  = aa_counts.get("F", 0)
        n_trp  = aa_counts.get("W", 0)
        arom_n = n_tyr + n_phe + n_trp
        arom_f = arom_n / seq_length
        cation_pi_n = count_pairs(seq, set("KR"), STICKER_AROMATIC, window=4)
        pi_pi_n     = count_pairs(seq, STICKER_AROMATIC, STICKER_AROMATIC, window=4)

        # --- Low complexity features ---
        entropy      = calc_shannon_entropy(seq)
        entropy_norm = entropy / math.log2(20)
        unique_aa    = sum(1 for v in aa_counts.values() if v > 0)
        # Q/N-rich fraction: fraction of residues in {N,Q,S,G,Y} — amino acids
        # enriched in yeast prion domains (Alberti et al. 2009, Cell 137:146-158).
        # This is a compositional proxy, not the PLAAC score.
        qn_rich_fraction = sum(aa_counts.get(k, 0) for k in PRION_LIKE) / seq_length
        lc_frac      = fraction_low_complexity(seq, window_size=12, threshold=2.0)

        # --- Disorder features ---
        disorder_f   = sum(aa_counts.get(k, 0) for k in DISORDER_PROMOTING) / seq_length
        order_f      = sum(aa_counts.get(k, 0) for k in ORDER_PROMOTING) / seq_length
        aliphatic_idx = (
            aa_counts.get("A", 0)
            + 2.9 * aa_counts.get("V", 0)
            + 3.9 * (aa_counts.get("I", 0) + aa_counts.get("L", 0))
        ) / seq_length * 100
        omega = calc_omega(seq)

        # --- Repeat motifs ---
        rgg_n  = seq.count("RGG")
        fg_n   = seq.count("FG")
        yg_n   = seq.count("YG") + seq.count("GY")
        sr_n   = seq.count("SR") + seq.count("RS")
        qn_n   = seq.count("QN") + seq.count("NQ")

        # --- Sticker & spacer ---
        sticker_arom_n  = arom_n
        sticker_elec_n  = sum(aa_counts.get(k, 0) for k in "KRDE")
        sticker_total_n = sum(1 for aa in seq if aa in STICKER_ALL)
        sticker_frac    = sticker_total_n / seq_length
        spacing         = sticker_spacing_stats(seq)
        _fmt_spacing    = lambda v: f"{v:.1f}" if v is not None else "N/A"

        # --- Hydrophobicity features ---
        hydro_vals    = [KYTE_DOOLITTLE[aa] for aa in seq]
        avg_kd        = sum(hydro_vals) / seq_length
        n_hydrophobic = sum(1 for v in hydro_vals if v > 0)
        n_hydrophilic = sum(1 for v in hydro_vals if v < 0)
        n_neutral_kd  = seq_length - n_hydrophobic - n_hydrophilic
        pct_hydro     = n_hydrophobic / seq_length * 100
        pct_hydrophil = n_hydrophilic / seq_length * 100

        # --- HTML sections (styled) ---
        _style = make_style_tag()
        extra_charge = (
            f"<tr><td>Net Charge (pH {pH_value:.1f})</td><td>{net_charge_pH:.2f}</td></tr>"
            if abs(pH_value - 7.0) >= 1e-6 else ""
        )

        sorted_aas = sorted(aa_counts, key=lambda aa: aa_freq[aa], reverse=True)
        comp_html = _style + (
            "<h2>Composition</h2>"
            "<table>"
            "<tr><th>Amino Acid</th><th>Count</th><th>Frequency (%)</th></tr>"
            + "".join(
                f"<tr><td>{aa}</td><td>{aa_counts[aa]}</td><td>{aa_freq[aa]:.2f}%</td></tr>"
                for aa in sorted_aas
            )
            + "</table>"
        )

        bio_html = _style + f"""
        <h2>Properties</h2>
        <table>
          <tr><th>Property</th><th>Value</th></tr>
          <tr><td>Sequence Length</td><td>{seq_length} aa</td></tr>
          <tr><td>Molecular Weight</td><td>{mol_weight:.2f} Da</td></tr>
          <tr><td>Isoelectric Point (pI)</td><td>{iso_point:.2f}</td></tr>
          <tr><td>Net Charge (pH 7.0)</td><td>{net_charge_7:.2f}</td></tr>
          {extra_charge}
          <tr><td>Extinction Coeff. (280 nm)</td><td>{extinction} M&#8315;&#185;cm&#8315;&#185;</td></tr>
          <tr><td>GRAVY Score</td><td>{gravy:.3f}</td></tr>
          <tr><td>Aromaticity</td><td>{aromaticity:.3f}</td></tr>
          <tr><td>Aliphatic Index</td><td>{aliphatic_idx:.1f}</td></tr>
        </table>
        <p class="note">Aliphatic index: relative volume of aliphatic side chains (A, V, I, L); higher values correlate with thermostability (Ikai 1980).</p>
        """

        hydro_html = _style + f"""
        <h2>Hydrophobicity</h2>
        <table>
          <tr><th>Property</th><th>Value</th></tr>
          <tr><td>GRAVY Score (Kyte-Doolittle)</td><td>{gravy:.4f}</td></tr>
          <tr><td>Average hydrophobicity per residue</td><td>{avg_kd:.4f}</td></tr>
          <tr><td>Hydrophobic residues (KD &gt; 0)</td><td>{n_hydrophobic} ({pct_hydro:.1f}%)</td></tr>
          <tr><td>Hydrophilic residues (KD &lt; 0)</td><td>{n_hydrophilic} ({pct_hydrophil:.1f}%)</td></tr>
          <tr><td>Neutral residues (KD = 0)</td><td>{n_neutral_kd} ({n_neutral_kd/seq_length*100:.1f}%)</td></tr>
        </table>
        <h2>Kyte-Doolittle Values by Residue</h2>
        <table>
          <tr><th>Amino Acid</th><th>Count</th><th>KD Score</th><th>Contribution</th></tr>
          {"".join(
            f"<tr><td>{aa}</td><td>{aa_counts.get(aa,0)}</td>"
            f"<td>{KYTE_DOOLITTLE[aa]:+.1f}</td>"
            f"<td>{KYTE_DOOLITTLE[aa]*aa_counts.get(aa,0)/seq_length:+.4f}</td></tr>"
            for aa in sorted(KYTE_DOOLITTLE, key=lambda x: KYTE_DOOLITTLE[x], reverse=True)
          )}
        </table>
        <p class="note">GRAVY (Grand Average of Hydropathicity): positive = hydrophobic, negative = hydrophilic (Kyte &amp; Doolittle 1982)</p>
        """

        charge_html = _style + f"""
        <h2>Charge</h2>
        <table>
          <tr><th>Property</th><th>Value</th></tr>
          <tr><td>Positive residues (K, R)</td><td>{pos_n}</td></tr>
          <tr><td>Negative residues (D, E)</td><td>{neg_n}</td></tr>
          <tr><td>FCR (fraction charged)</td><td>{fcr:.3f}</td></tr>
          <tr><td>NCPR (net charge/residue)</td><td>{ncpr:+.3f}</td></tr>
          <tr><td>K+R / D+E ratio</td><td>{"%.2f" % ch_asym if neg_n > 0 else "&#8734; (no D or E)"}</td></tr>
          <tr><td>Kappa (&kappa;)</td><td>{kappa:.3f}</td></tr>
        </table>
        <p class="note">&kappa;: 0 = well-mixed, 1 = fully segregated (Das &amp; Pappu 2013)</p>
        """

        aromatic_html = _style + f"""
        <h2>Aromatic &amp; &pi;-Interactions</h2>
        <table>
          <tr><th>Property</th><th>Value</th></tr>
          <tr><td>Aromatic fraction (F+W+Y)</td><td>{arom_f:.3f} ({arom_n} residues)</td></tr>
          <tr><td>Tyr (Y)</td><td>{n_tyr} ({n_tyr/seq_length*100:.1f}%)</td></tr>
          <tr><td>Phe (F)</td><td>{n_phe} ({n_phe/seq_length*100:.1f}%)</td></tr>
          <tr><td>Trp (W)</td><td>{n_trp} ({n_trp/seq_length*100:.1f}%)</td></tr>
          <tr><td>Cation&ndash;&pi; pairs (K/R &harr; F/W/Y, &plusmn;4)</td><td>{cation_pi_n}</td></tr>
          <tr><td>&pi;&ndash;&pi; pairs (F/W/Y &harr; F/W/Y, &plusmn;4)</td><td>{pi_pi_n}</td></tr>
        </table>
        """

        # --- PLAAC score ---
        _plaac = calc_plaac_score(seq)
        _plaac_max  = _plaac["max_score"]

        lc_html = _style + f"""
        <h2>Low Complexity</h2>
        <table>
          <tr><th>Property</th><th>Value</th></tr>
          <tr><td>Shannon entropy</td><td>{entropy:.3f} bits (max 4.32)</td></tr>
          <tr><td>Normalized entropy</td><td>{entropy_norm:.3f}</td></tr>
          <tr><td>Unique amino acids</td><td>{unique_aa} / 20</td></tr>
          <tr><td>Q/N-rich fraction (N,Q,S,G,Y)</td><td>{qn_rich_fraction:.3f}</td></tr>
          <tr><td>LC fraction (w=12, H&lt;2.0 bits)</td><td>{lc_frac:.3f}</td></tr>
          <tr><td>PLAAC max log-odds score</td><td>{_plaac_max:.3f}</td></tr>
        </table>
        <p class="note">Q/N-rich fraction: fraction of {{N,Q,S,G,Y}}, amino acids enriched in yeast prion domains (Alberti et al. 2009, Cell 137:146). This is a compositional proxy, not the PLAAC score. PLAAC: log-odds of yeast prion-like composition vs. SwissProt background, smoothed w=41 (Lancaster et al. 2014 Cell Reports). Regions with consistently positive PLAAC scores are candidate prion-like domains; refer to Lancaster et al. (2014) for interpretation.</p>
        """

        # --- Disorder profile (ESM2-aware) ---
        _disorder_head = load_disorder_head()
        disorder_scores = calc_disorder_profile(seq, window=window_size, embedder=embedder, head=_disorder_head)
        mean_disorder   = sum(disorder_scores) / seq_length

        # Determine which disorder method was actually used (for report transparency)
        _disorder_method: str
        if embedder is not None and embedder.is_available() and _disorder_head is not None:
            _disorder_method = "ESM2 logistic probe (AUC 0.874, DisProt 2024)"
            _is_calibrated_prob = True
        else:
            try:
                import metapredict  # noqa: F401
                _disorder_method = "metapredict (Emenecker et al. 2021, Cell Syst.)"
                _is_calibrated_prob = True
            except ImportError:
                _disorder_method = "classical sliding-window propensity scale"
                _is_calibrated_prob = False

        # 0.5 threshold is valid only for calibrated probability outputs (ESM2 probe,
        # metapredict). The classical propensity scale has a different distribution and
        # range and does not support a 0.5 decision boundary.
        # Threshold of 0.5 on calibrated disorder probability, as used in DisProt
        # benchmark (Necci et al. 2021, Nucleic Acids Res.).
        if _is_calibrated_prob:
            disordered_frac = sum(1 for v in disorder_scores if v > 0.5) / seq_length
            _disordered_frac_html = (
                f"<tr><td>Disordered fraction (score &gt; 0.5)</td>"
                f"<td>{disordered_frac:.3f} ({disordered_frac*100:.1f}%)</td></tr>"
            )
            _disorder_note_thresh = (
                "Threshold of 0.5 on calibrated disorder probability, as used in DisProt "
                "benchmark (Necci et al. 2021, Nucleic Acids Res.)."
            )
        else:
            disordered_frac = None
            _disordered_frac_html = ""
            _disorder_note_thresh = (
                "Classical propensity scale does not produce calibrated probabilities; "
                "no 0.5 decision boundary is applied. The per-residue profile is reported "
                "as a continuous propensity value only."
            )

        disorder_html = _style + f"""
        <h2>Disorder &amp; Flexibility</h2>
        <table>
          <tr><th>Property</th><th>Value</th></tr>
          <tr><td>Disorder-promoting fraction (A,E,G,K,P,Q,R,S)</td><td>{disorder_f:.3f}</td></tr>
          <tr><td>Order-promoting fraction (C,F,H,I,L,M,V,W,Y)</td><td>{order_f:.3f}</td></tr>
          <tr><td>Omega (&Omega;)</td><td>{omega:.3f}</td></tr>
          <tr><td>Mean per-residue disorder score</td><td>{mean_disorder:.3f}</td></tr>
          {_disordered_frac_html}
        </table>
        <p class="note">Disorder/order fractions: Uversky 2003. &Omega;: sticker patterning, 0 = evenly distributed, 1 = clustered (Das et al. 2015). Per-residue disorder method used: <em>{_disorder_method}</em>. {_disorder_note_thresh}</p>
        """

        repeats_html = _style + f"""
        <h2>Repeat Motifs</h2>
        <table>
          <tr><th>Motif</th><th>Count</th></tr>
          <tr><td>RGG</td><td>{rgg_n}</td></tr>
          <tr><td>FG</td><td>{fg_n}</td></tr>
          <tr><td>YG + GY</td><td>{yg_n}</td></tr>
          <tr><td>SR + RS</td><td>{sr_n}</td></tr>
          <tr><td>QN + NQ</td><td>{qn_n}</td></tr>
        </table>
        """

        sticker_html = _style + f"""
        <h2>Sticker &amp; Spacer</h2>
        <table>
          <tr><th>Property</th><th>Value</th></tr>
          <tr><td>Total stickers (F,W,Y,K,R,D,E)</td><td>{sticker_total_n} ({sticker_frac*100:.1f}%)</td></tr>
          <tr><td>Aromatic stickers (F,W,Y)</td><td>{sticker_arom_n}</td></tr>
          <tr><td>Electrostatic stickers (K,R,D,E)</td><td>{sticker_elec_n}</td></tr>
          <tr><td>Mean sticker spacing</td><td>{_fmt_spacing(spacing["mean"])} residues</td></tr>
          <tr><td>Min sticker spacing</td><td>{_fmt_spacing(spacing["min"])} residues</td></tr>
          <tr><td>Max sticker spacing</td><td>{_fmt_spacing(spacing["max"])} residues</td></tr>
        </table>
        <p class="note">Sticker-and-spacer model: Mittag &amp; Pappu</p>
        """

        # --- Transmembrane helix prediction ---
        tm_helices = predict_tm_helices(seq)
        n_tm       = len(tm_helices)
        tm_rows = "".join(
            f"<tr><td>{i}</td><td>{h['start']+1}</td><td>{h['end']+1}</td>"
            f"<td>{h['end']-h['start']+1}</td><td>{h['score']:.3f}</td>"
            f"<td>{h['orientation']}</td></tr>"
            for i, h in enumerate(tm_helices, 1)
        )
        tm_body = (
            tm_rows
            if tm_helices
            else "<tr><td colspan='6'><em>No TM helices predicted</em></td></tr>"
        )
        tm_html = _style + f"""
        <h2>Transmembrane Helices</h2>
        <table>
          <tr><th>#</th><th>Start</th><th>End</th>
              <th>Length</th><th>Avg KD Score</th><th>Orientation</th></tr>
          {tm_body}
        </table>
        <p class="note">Kyte-Doolittle sliding window (w=19, threshold=1.6).
        Orientation by inside-positive rule (von Heijne): out&rarr;in = N-term extracellular.</p>
        """

        # --- LARKS ---
        larks = detect_larks(seq)
        larks_rows = "".join(
            f"<tr><td>{h['start']+1}–{h['end']+1}</td><td>{h['seq']}</td>"
            f"<td>{h['n_arom']}</td><td>{h['lc_frac']:.2f}</td><td>{h['entropy']:.2f}</td></tr>"
            for h in larks
        ) or "<tr><td colspan='5'><em>No LARKS detected</em></td></tr>"
        larks_html = _style + f"""
        <h2>LARKS (Low-complexity Aromatic-Rich Kinked Segments)</h2>
        <table>
          <tr><th>Position</th><th>Sequence</th><th>Aromatics</th><th>LC Frac</th><th>Entropy (bits)</th></tr>
          {larks_rows}
        </table>
        <p class="note">LARKS: 7-residue windows with &ge;1 aromatic (F/W/Y), &ge;50% low-complexity residues (G/A/S/T/N/Q),
        and Shannon entropy &lt;1.8 bits. Criteria from Hughes et al. (2018) eLife.
        LARKS form cross-&beta; spines in amyloid-like fibrils of low-complexity domains.</p>
        """

        # --- Coiled-coil prediction ---
        cc_profile  = predict_coiled_coil(seq)
        # Report only the continuous per-residue coiled-coil propensity score.
        # No hard threshold is applied: the normalisation and any fixed cutoff
        # are not from Lupas et al. (1991) or Berger et al. (1995).
        # Higher scores indicate stronger coiled-coil character; no universal
        # threshold applies (Lupas et al. 1991; Berger et al. 1995).

        # --- Linear motif scan ---
        motifs = scan_linear_motifs(seq)
        motif_rows = "".join(
            f"<tr><td>{m['name']}</td><td>{m['start']+1}–{m['end']+1}</td>"
            f"<td><tt>{m['match']}</tt></td><td>{m['description']}</td></tr>"
            for m in motifs
        ) or "<tr><td colspan='4'><em>No motifs found</em></td></tr>"
        motifs_html = _style + f"""
        <h2>Linear Motif Scan</h2>
        <p>Scanned against {len(LINEAR_MOTIFS)} built-in motif patterns.</p>
        <table>
          <tr><th>Motif</th><th>Position</th><th>Match</th><th>Description</th></tr>
          {motif_rows}
        </table>
        <p class="note">Pattern library includes NLS, NES, PxxP, 14-3-3, RGG, FG, KFERQ, KDEL,
        N-glycosylation, SUMOylation, CK2 phosphorylation, caspase cleavage, WW-domain, SxIP, PKA sites.
        Matches are regex-based and require experimental validation.</p>
        """

        # --- Aggregation & solubility ---
        # Pass the accent colour (hex string) expected by each format_* function,
        # not the full <style>…</style> HTML tag stored in _style.
        _accent = "#4361ee"
        aggr_html   = format_aggregation_report(seq, _accent)
        solub_stats = calc_solubility_stats(seq)
        # Primary aggregation profile: always ZYGGREGATOR (Tartaglia & Vendruscolo 2008)
        aggr_profile_zygg = calc_aggregation_profile(seq)
        # Optional ESM2-only aggregation profile (settings-controlled, no blend)
        aggr_profile_esm2 = None
        if use_esm2_aggregation and embedder is not None and embedder.is_available():
            _aggr_head = load_aggregation_head()
            if _aggr_head is not None:
                aggr_profile_esm2 = calc_aggregation_profile_esm2(
                    seq, embedder=embedder, head=_aggr_head
                )

        # --- Signal peptide & GPI ---
        signal_html = format_signal_report(seq, _accent)
        sp_result   = predict_signal_peptide(seq)
        gpi_result  = predict_gpi_anchor(seq)

        # --- Amphipathic helices ---
        amph_html    = format_amphipathic_report(seq, _accent)
        moment_alpha = calc_hydrophobic_moment_profile(seq, angle_deg=100.0)
        moment_beta  = calc_hydrophobic_moment_profile(seq, angle_deg=160.0)
        amph_regions = predict_amphipathic_helices(seq)

        # --- SCD ---
        scd_val          = calc_scd(seq)
        scd_profile_data = calc_scd_profile(seq, window=20)
        scd_blocks       = calc_pos_neg_block_lengths(seq)
        scd_html         = format_scd_report(seq, _accent)

        # --- RNA binding ---
        rbp_result       = calc_rbp_score(seq)
        rbp_profile_data = calc_rbp_profile(seq)
        rbp_html         = format_rbp_report(seq, _accent)

        # --- Tandem repeats + PolyX ---
        polyx_stretches = calc_polyx_stretches(seq, min_length=4)
        tandem_stats    = calc_repeat_stats(seq)

        # Append PolyX table to the tandem repeats HTML
        tandem_html_base = format_tandem_repeats_report(seq, _accent)
        if polyx_stretches:
            _px_rows = "".join(
                f"<tr><td>poly{r['aa']}</td><td>{r['aa'] * min(r['length'], 6)}{'…' if r['length'] > 6 else ''}</td>"
                f"<td>{r['start_1based']}–{r['end_1based']}</td><td>{r['length']}</td></tr>"
                for r in polyx_stretches
            )
            _px_html = (
                "<h2>Homopolymeric Runs (PolyX)</h2>"
                "<table><tr><th>Type</th><th>Sequence</th><th>Position</th><th>Length</th></tr>"
                f"{_px_rows}</table>"
                "<p class='note'>Homopolymeric stretches &ge;4 consecutive identical residues. "
                "Biologically relevant: polyQ (neurodegeneration), polyA (PABPN1 myopathy), polyE, polyS.</p>"
            )
            tandem_html = tandem_html_base + _px_html
        else:
            tandem_html = tandem_html_base

        # --- Proteolytic cleavage ---
        prot_html   = format_proteolysis_report(seq, _accent)
        prot_sites  = calc_proteolytic_sites(seq)

        return {
            "report_sections": {
                "Composition":             comp_html,
                "Properties":              bio_html,
                "Hydrophobicity":          hydro_html,
                "Charge":                  charge_html,
                "Aromatic & \u03c0":       aromatic_html,
                "Low Complexity":          lc_html,
                "Disorder":                disorder_html,
                "Repeat Motifs":           repeats_html,
                "Sticker & Spacer":        sticker_html,
                "TM Helices":              tm_html,
                "LARKS":                   larks_html,
                "Linear Motifs":           motifs_html,
                "\u03b2-Aggregation & Solubility": aggr_html,
                "Signal Peptide & GPI":    signal_html,
                "Amphipathic Helices":     amph_html,
                "Charge Decoration (SCD)": scd_html,
                "RNA Binding":             rbp_html,
                "Tandem Repeats":          tandem_html,
                "Proteolytic Map":         prot_html,
            },
            "tm_helices":      tm_helices,
            "aa_counts":       aa_counts,
            "aa_freq":         aa_freq,
            "hydro_profile":   sliding_window_hydrophobicity(seq, window_size, HYDROPHOBICITY_SCALES.get(hydro_scale, HYDROPHOBICITY_SCALES["Kyte-Doolittle"])["values"]),
            "hydro_scale":     hydro_scale,
            "ncpr_profile":    sliding_window_ncpr(seq, window_size),
            "entropy_profile": sliding_window_entropy(seq, window_size),
            "disorder_scores":  disorder_scores,
            "disorder_method":  _disorder_method,
            "window_size":      window_size,
            "seq":             seq,
            "mol_weight":      mol_weight,
            "iso_point":       iso_point,
            "net_charge_7":    net_charge_7,
            "extinction":      extinction,
            "gravy":           gravy,
            "aromaticity":     aromaticity,
            "fcr":             fcr,
            "ncpr":            ncpr,
            "arom_f":          arom_f,
            "qn_rich_fraction": qn_rich_fraction,
            "kappa":           kappa,
            "omega":           omega,
            "disorder_f":      disorder_f,
            "larks":           larks,
            "cc_profile":      cc_profile,
            "motifs":          motifs,
            "solub_stats":     solub_stats,
            "aggr_profile":    aggr_profile_zygg,
            "aggr_profile_esm2": aggr_profile_esm2,
            "sp_result":       sp_result,
            "gpi_result":      gpi_result,
            "moment_alpha":    moment_alpha,
            "moment_beta":     moment_beta,
            "amph_regions":    amph_regions,
            "scd":             scd_val,
            "scd_profile":     scd_profile_data,
            "scd_blocks":      scd_blocks,
            "rbp":             rbp_result,
            "rbp_profile":     rbp_profile_data,
            "tandem_stats":    tandem_stats,
            "polyx_stretches": polyx_stretches,
            "prot_sites":      prot_sites,
            "plaac":           _plaac,
        }
