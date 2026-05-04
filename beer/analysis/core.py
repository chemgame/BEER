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
from beer.reports.css import make_style_tag, method_badge
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
    detect_larks,
    predict_coiled_coil,
    scan_linear_motifs,
)
from beer.analysis.aggregation import (
    calc_aggregation_profile,
    calc_solubility_stats,
    format_aggregation_report,
)
from beer.models import (
    load_disorder_head,
    load_signal_peptide_head,
    load_transmembrane_head,
    load_coiled_coil_head,
    load_dna_binding_head,
    load_active_site_head,
    load_binding_site_head,
    load_phosphorylation_head,
    load_lcd_head,
    load_zinc_finger_head,
    load_glycosylation_head,
    load_ubiquitination_head,
    load_methylation_head,
    load_acetylation_head,
    load_lipidation_head,
    load_disulfide_head,
    load_intramembrane_head,
    load_motif_head,
    load_propeptide_head,
    load_repeat_head,
    load_rna_binding_head,
    load_nucleotide_binding_head,
    load_transit_peptide_head,
    load_aggregation_head,
)
from beer.analysis.signal_peptide import (
    predict_signal_peptide,
    predict_gpi_anchor,
    calc_signal_peptide_profile,
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
    calc_shd_profile,
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
from beer.analysis.phosphorylation import (
    predict_phosphorylation,
    format_phospho_report,
)
from beer.utils.biophysics import calc_polyx_stretches, calc_plaac_score
from beer.embeddings.base import SequenceEmbedder

# Maps analysis_data key → model-loader callable for lazy per-section AI computation.
_BILSTM_HEAD_LOADERS: dict[str, "callable"] = {
    "disorder_scores":       load_disorder_head,
    "sp_bilstm_profile":     load_signal_peptide_head,
    "tm_bilstm_profile":     load_transmembrane_head,
    "intramem_bilstm_profile": load_intramembrane_head,
    "cc_bilstm_profile":     load_coiled_coil_head,
    "dna_bilstm_profile":    load_dna_binding_head,
    "act_bilstm_profile":    load_active_site_head,
    "bnd_bilstm_profile":    load_binding_site_head,
    "phos_bilstm_profile":   load_phosphorylation_head,
    "lcd_bilstm_profile":    load_lcd_head,
    "znf_bilstm_profile":    load_zinc_finger_head,
    "glyc_bilstm_profile":   load_glycosylation_head,
    "ubiq_bilstm_profile":   load_ubiquitination_head,
    "meth_bilstm_profile":   load_methylation_head,
    "acet_bilstm_profile":   load_acetylation_head,
    "lipid_bilstm_profile":  load_lipidation_head,
    "disulf_bilstm_profile": load_disulfide_head,
    "motif_bilstm_profile":  load_motif_head,
    "prop_bilstm_profile":   load_propeptide_head,
    "rep_bilstm_profile":    load_repeat_head,
    "rnabind_bilstm_profile":  load_rna_binding_head,
    "nucbind_bilstm_profile":  load_nucleotide_binding_head,
    "transit_bilstm_profile":  load_transit_peptide_head,
    "agg_bilstm_profile":      load_aggregation_head,
}


def compute_single_bilstm_head(
    data_key: str,
    seq: str,
    embedder: "SequenceEmbedder | None",
) -> "list[float] | None":
    """Run the BiLSTM head for *data_key* on *seq* using *embedder*.

    The ESM2Embedder caches embeddings by sequence hash, so calling this for
    multiple heads on the same sequence only computes the forward pass once.

    Returns per-residue probabilities (list of float, same length as seq),
    or None if the head model file is missing or the embedder is unavailable.
    """
    from beer.utils.structure import bilstm_predict
    loader = _BILSTM_HEAD_LOADERS.get(data_key)
    if loader is None:
        return None
    head = loader()
    return bilstm_predict(seq, embedder, head)


def _make_summary_bullets(
    seq, seq_length, mw, pI, disorder_f, disorder_scores,
    sp_bilstm_profile, tm_bilstm_profile, cc_bilstm_profile,
    aggr_profile, catgranule, rbp, fcr, ncpr, kappa, larks,
    gpi_result, phospho_sites, prot_sites,
) -> list[str]:
    """Return a list of concise bullet-point strings summarising key findings."""
    bullets: list[str] = []

    # --- Basic identity ---
    bullets.append(
        f"{seq_length} residues · MW {mw/1000:.1f} kDa · pI {pI:.2f}"
    )

    # --- Disorder ---
    if disorder_scores and seq_length > 0:
        d_frac = sum(1 for v in disorder_scores if v > 0.5) / seq_length
        if d_frac >= 0.5:
            bullets.append(
                f"Highly disordered — {d_frac*100:.0f}% of residues predicted disordered (BiLSTM)"
            )
        elif d_frac >= 0.25:
            bullets.append(
                f"Partially disordered — {d_frac*100:.0f}% disordered regions (BiLSTM)"
            )
        else:
            bullets.append(
                f"Predominantly ordered — only {d_frac*100:.0f}% disordered (BiLSTM)"
            )

    # --- Signal peptide ---
    if sp_bilstm_profile:
        sp_max = max(sp_bilstm_profile[:35]) if len(sp_bilstm_profile) >= 10 else 0
        if sp_max > 0.7:
            bullets.append(
                f"Strong N-terminal signal peptide detected (BiLSTM score {sp_max:.2f})"
            )

    # --- GPI anchor ---
    if gpi_result and gpi_result.get("has_gpi"):
        bullets.append("GPI anchor signal present at C-terminus")

    # --- Transmembrane ---
    if tm_bilstm_profile:
        n_tm = sum(1 for v in tm_bilstm_profile if v > 0.5)
        if n_tm >= 15:
            bullets.append(
                f"Transmembrane protein — ~{n_tm} residues predicted in TM helices (BiLSTM)"
            )

    # --- Coiled coil ---
    if cc_bilstm_profile:
        n_cc = sum(1 for v in cc_bilstm_profile if v > 0.5)
        if n_cc >= 10:
            bullets.append(f"Coiled-coil region(s) predicted ({n_cc} residues, BiLSTM)")

    # --- Aggregation ---
    if aggr_profile:
        n_hot = sum(1 for v in aggr_profile if v >= 1.0)
        if n_hot >= 4:
            bullets.append(
                f"{n_hot} residues in β-aggregation hotspots (ZYGGREGATOR Z ≥ 1.0)"
            )

    # --- Phase separation / condensate ---
    if catgranule is not None and catgranule > 0:
        bullets.append(
            f"Condensate-forming potential: catGRANULE score {catgranule:+.2f} (positive = prone)"
        )

    # --- RNA binding ---
    if rbp and rbp.get("composite_score", 0) > 0:
        bullets.append(
            f"RNA-binding propensity: catRAPID composite ω̄ = {rbp['composite_score']:.2f}"
        )

    # --- Charge character ---
    if fcr >= 0.35:
        bullets.append(
            f"Strong polyelectrolyte character — FCR {fcr:.2f}, NCPR {ncpr:+.2f} (Das-Pappu)"
        )
    elif fcr >= 0.25:
        bullets.append(f"Moderately charged — FCR {fcr:.2f}, NCPR {ncpr:+.2f}")

    # --- LARKS ---
    if larks:
        bullets.append(f"{len(larks)} LARKS segment(s) — potential amyloid-like interaction core(s)")

    # --- Phosphorylation ---
    if phospho_sites:
        n_p = sum(len(v) for v in phospho_sites.values())
        if n_p:
            bullets.append(f"{n_p} predicted phosphorylation site(s) (PWM scan: PKA/PKC/CK2/Tyr)")

    return bullets


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
    ) -> dict[str, Any]:
        if not seq:
            raise ValueError("Cannot analyse an empty sequence.")
        pa          = BPProteinAnalysis(seq)
        aa_counts   = pa.count_amino_acids()
        seq_length  = len(seq)
        aa_freq     = {aa: count / seq_length * 100 for aa, count in aa_counts.items()}
        mol_weight  = pa.molecular_weight()  # average masses (Biopython; NIST average atomic weights)
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

        # --- Hydrophobicity features (selected scale) ---
        _scale_data   = HYDROPHOBICITY_SCALES.get(hydro_scale, HYDROPHOBICITY_SCALES["Kyte-Doolittle"])
        _scale_vals   = _scale_data["values"]
        hydro_vals    = [_scale_vals.get(aa, 0.0) for aa in seq]
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

        _scale_ref  = _scale_data.get("ref", hydro_scale)
        hydro_html = _style + f"""
        <h2>Hydrophobicity ({hydro_scale})</h2>
        <table>
          <tr><th>Property</th><th>Value</th></tr>
          <tr><td>GRAVY Score (Kyte-Doolittle, BioPython)</td><td>{gravy:.4f}</td></tr>
          <tr><td>Mean hydrophobicity per residue ({hydro_scale})</td><td>{avg_kd:.4f}</td></tr>
          <tr><td>Hydrophobic residues (score &gt; 0)</td><td>{n_hydrophobic} ({pct_hydro:.1f}%)</td></tr>
          <tr><td>Hydrophilic residues (score &lt; 0)</td><td>{n_hydrophilic} ({pct_hydrophil:.1f}%)</td></tr>
          <tr><td>Neutral residues (score = 0)</td><td>{n_neutral_kd} ({n_neutral_kd/seq_length*100:.1f}%)</td></tr>
        </table>
        <h2>{hydro_scale} Values by Residue</h2>
        <table>
          <tr><th>Amino Acid</th><th>Count</th><th>Score</th><th>Contribution</th></tr>
          {"".join(
            f"<tr><td>{aa}</td><td>{aa_counts.get(aa,0)}</td>"
            f"<td>{_scale_vals.get(aa,0.0):+.2f}</td>"
            f"<td>{_scale_vals.get(aa,0.0)*aa_counts.get(aa,0)/seq_length:+.4f}</td></tr>"
            for aa in sorted(_scale_vals, key=lambda x: _scale_vals[x], reverse=True)
          )}
        </table>
        <p class="note">GRAVY (Grand Average of Hydropathicity, Kyte &amp; Doolittle 1982): positive = hydrophobic, negative = hydrophilic. Scale shown: {_scale_ref}.</p>
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

        # --- Disorder profile (ESM2 BiLSTM only — no classical/metapredict fallback) ---
        # When embedder is None (classical Analyze), disorder_scores is empty.
        # The user must click "AI Predictions → Disorder" for lazy computation.
        from beer.utils.structure import bilstm_predict_mc as _bilstm_mc
        _disorder_head = load_disorder_head()
        disorder_scores: list = []
        disorder_uncertainty: list | None = None

        if embedder is not None and embedder.is_available() and _disorder_head is not None:
            disorder_scores = calc_disorder_profile(
                seq, window=window_size, embedder=embedder, head=_disorder_head)
            _disorder_arch = (_disorder_head.get("architecture", "") if _disorder_head else "")
            if hasattr(_disorder_arch, "item"):
                _disorder_arch = _disorder_arch.item()
            if _disorder_arch == "bilstm2":
                _mc = _bilstm_mc(seq, embedder, _disorder_head, n_passes=20)
                if _mc is not None:
                    disorder_scores, disorder_uncertainty = _mc

        mean_disorder = sum(disorder_scores) / seq_length if (disorder_scores and seq_length > 0) else 0.0

        # Disorder HTML — placeholder when scores not yet computed
        def _contiguous_regions(scores, threshold=0.5):
            regions = []
            in_region = False
            start = 0
            for i, v in enumerate(scores):
                if v > threshold and not in_region:
                    in_region = True
                    start = i
                elif v <= threshold and in_region:
                    in_region = False
                    regions.append((start + 1, i))
            if in_region:
                regions.append((start + 1, len(scores)))
            return regions

        if disorder_scores and seq_length > 0:
            disordered_frac = sum(1 for v in disorder_scores if v > 0.5) / seq_length
            _disordered_frac_html = (
                f"<tr><td>Disordered fraction (score &gt; 0.5)</td>"
                f"<td>{disordered_frac:.3f} ({disordered_frac*100:.1f}%)</td></tr>"
            )
            _idr_regions = _contiguous_regions(disorder_scores, 0.5)
            if _idr_regions:
                _idr_rows = "".join(
                    f"<tr><td>IDR {k}</td><td>{s}–{e}</td><td>{e-s+1}</td></tr>"
                    for k, (s, e) in enumerate(_idr_regions, 1)
                )
                _idr_table = (
                    "<h3 style='margin:10px 0 4px'>Predicted Intrinsically Disordered Regions</h3>"
                    "<table><tr><th>#</th><th>Residues</th><th>Length</th></tr>"
                    f"{_idr_rows}</table>"
                    f"<p class='note'>{len(_idr_regions)} IDR(s) predicted at threshold 0.5.</p>"
                )
            else:
                _idr_table = "<p class='note'>No contiguous disordered regions predicted (threshold 0.5).</p>"
            _disorder_note = (
                "ESM2 650M BiLSTM disorder head. "
                "Threshold of 0.5 on calibrated disorder probability (DisProt benchmark, "
                "Necci et al. 2021, Nucleic Acids Res.)."
            )
            disorder_html = _style + f"""
            <h2>Disorder &amp; Flexibility {method_badge("ESM2 BiLSTM", "bilstm")}</h2>
            <table>
              <tr><th>Property</th><th>Value</th></tr>
              <tr><td>Disorder-promoting fraction (A,E,G,K,P,Q,R,S)</td><td>{disorder_f:.3f}</td></tr>
              <tr><td>Order-promoting fraction (C,F,H,I,L,M,V,W,Y)</td><td>{order_f:.3f}</td></tr>
              <tr><td>Omega (&Omega;)</td><td>{omega:.3f}</td></tr>
              <tr><td>Mean per-residue disorder score</td><td>{mean_disorder:.3f}</td></tr>
              {_disordered_frac_html}
            </table>
            {_idr_table}
            <p class="note">Disorder/order fractions: Uversky 2003. &Omega;: sticker patterning (Das et al. 2015). {_disorder_note}</p>
            """
        else:
            disorder_html = _style + """
            <h2>Disorder &amp; Flexibility</h2>
            <table>
              <tr><th>Property</th><th>Value</th></tr>
              <tr><td>Disorder-promoting fraction (A,E,G,K,P,Q,R,S)</td><td>{disorder_f:.3f}</td></tr>
              <tr><td>Order-promoting fraction (C,F,H,I,L,M,V,W,Y)</td><td>{order_f:.3f}</td></tr>
              <tr><td>Omega (&Omega;)</td><td>{omega:.3f}</td></tr>
            </table>
            <div style='background:#f0f4ff;border-left:4px solid #4361ee;padding:12px 16px;
                        border-radius:4px;margin:12px 0'>
              <b>Per-residue disorder scores not yet computed.</b>
              <p style='margin:6px 0 0'>Click <b>AI Predictions &rarr; Disorder</b> in the
              sidebar to run the ESM2 BiLSTM disorder head.</p>
            </div>
            <p class="note">Disorder/order fractions: Uversky 2003. &Omega;: sticker patterning (Das et al. 2015).</p>
            """.format(disorder_f=disorder_f, order_f=order_f, omega=omega)

        _disorder_method     = "ESM2 650M BiLSTM" if disorder_scores else ""
        _disorder_has_bilstm = bool(disorder_scores)

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

        # --- catGRANULE score (Bolognesi et al. 2016 Cell Reports 14:2535) ---
        # Global score: 1.325·catRAPID + 0.647·disorder - 1.490·KD - 0.256
        # catRAPID per-residue composite (Bellucci 2011):
        #   ω(i) = 0.0169·SP + 0.0117·HP + 0.0283·vdW
        # Bias −0.256 calibrated so average non-granule protein ≈ 0.
        from beer.constants import CHOU_FASMAN_HELIX, VDW_VOLUME
        _cg_per = [
            0.0169 * CHOU_FASMAN_HELIX.get(aa, 1.0)
            + 0.0117 * (-KYTE_DOOLITTLE.get(aa, 0.0) / 4.5)
            + 0.0283 * VDW_VOLUME.get(aa, 0.5)
            for aa in seq
        ]
        _catrapid_mean = sum(_cg_per) / seq_length
        catgranule_score = round(
            1.325 * _catrapid_mean + 0.647 * mean_disorder - 1.490 * gravy - 0.256,
            3,
        )
        # Sliding-window catGRANULE profile (window=10) — per-residue contribution
        _cg_win = 10
        catgranule_profile = [
            round(
                (1.325 * sum(_cg_per[j] for j in range(i, i + _cg_win)) / _cg_win
                 + 0.647 * (sum(disorder_scores[j] for j in range(i, i + _cg_win)) / _cg_win
                            if disorder_scores and len(disorder_scores) == seq_length else 0.0)
                 - 1.490 * sum(KYTE_DOOLITTLE.get(seq[j], 0.0) for j in range(i, i + _cg_win)) / _cg_win
                 - 0.256),
                4,
            )
            for i in range(seq_length - _cg_win + 1)
        ] if seq_length >= _cg_win else []
        _cg_label = "high" if catgranule_score > 0 else "low"

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
          <tr><td><b>catGRANULE score</b></td><td><b>{catgranule_score:.3f}</b> &mdash; {_cg_label} phase-separation propensity</td></tr>
        </table>
        <p class="note">Sticker-and-spacer model: Mittag &amp; Pappu (2021) Nat Rev Mol Cell Biol. &nbsp;
        catGRANULE score (Bolognesi et al. 2016 Cell Reports 14:2535): linear combination of RNA-binding
        propensity, disorder propensity, and inverse hydrophobicity. Score &gt; 0 predicts granule/condensate
        formation. Per-residue scales: Jeong et al. 2012 (RBP); BEER disorder propensity; Kyte-Doolittle.</p>
        """

        # --- Transmembrane helices: classical KD method removed in v2.0 ---
        # tm_helices is populated by TMHMM (dedicated button) or AI Analysis.
        tm_helices: list = []
        tm_html = _style + """
        <h2>Transmembrane Helices</h2>
        <div style='background:#fff8e1;border-left:4px solid #f59e0b;
                    padding:12px 16px;border-radius:4px;margin:12px 0'>
          <b style='color:#b45309'>Classical prediction removed in BEER v2.0</b>
          <p style='margin:6px 0 0'>Kyte-Doolittle sliding-window TM prediction has been
          retired. Use <b>AI Predictions &rarr; Transmembrane</b> for ESM2 BiLSTM
          per-residue scores, or run <b>TMHMM</b> via the dedicated button for
          full topology predictions.</p>
        </div>
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
        and Shannon entropy &lt;1.8 bits. Criteria from Hughes et al. (2018) eLife 7:e41464.
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
        <p class="note">Pattern sources: NLS — Kalderon et al. (1984) Cell 39:499;
        NES — Fornerod et al. (1997) Cell 90:1051;
        PxxP/SH3 — Ren et al. (1993) Science 259:1157;
        14-3-3 — Yaffe et al. (1997) Cell 91:961;
        RGG — Thandapani et al. (2013) Mol. Cell 50:613;
        FG repeats — Rout et al. (2000) J. Cell Biol. 148:635;
        KFERQ — Dice (1990) Trends Biochem. Sci. 15:305;
        KDEL — Munro &amp; Pelham (1987) Cell 48:899;
        PKA (RxxS/T) — Kemp et al. (1977) Biochem. J. 165:163;
        SxIP/EB1 — Honnappa et al. (2009) Cell 138:366;
        PPxY/WW — Sudol (1994) Oncogene 9:2145;
        Caspase-3 — Nicholson et al. (1995) Nature 376:37;
        N-glycosylation — Marshall (1974) Biochem. Soc. Symp. 40:17;
        SUMOylation (&Psi;KxE) — Sampson et al. (2001) J. Cell Biol. 154:341;
        CK2 — Meggio &amp; Pinna (2003) FASEB J. 17:349.
        All matches are regex-based and require experimental validation.</p>
        """

        # --- Aggregation & solubility ---
        # Pass the accent colour (hex string) expected by each format_* function,
        # not the full <style>…</style> HTML tag stored in _style.
        _accent = "#4361ee"
        aggr_html   = format_aggregation_report(seq, _accent)
        solub_stats = calc_solubility_stats(seq)
        aggr_profile_zygg = calc_aggregation_profile(seq)

        # --- Signal peptide & GPI ---
        _sp_head = load_signal_peptide_head()
        sp_bilstm_profile = calc_signal_peptide_profile(seq, embedder, _sp_head)
        signal_html = format_signal_report(seq, _accent, bilstm_scores=sp_bilstm_profile)
        gpi_result  = predict_gpi_anchor(seq)

        # Derive sp_result from BiLSTM when available; classical fallback otherwise.
        # sp_result is used by: structure annotation (sp_end), composite map (h_start, h_end,
        # cleavage_site), and summary table (h_region_score).
        if sp_bilstm_profile is not None:
            _n70 = sp_bilstm_profile[:70]
            _sp_max = max(_n70) if _n70 else 0.0
            _sp_det = _sp_max > 0.5
            _sp_end = 0
            if _sp_det:
                _peak_i = _n70.index(max(_n70))
                _sp_end = len(_n70)
                for _si in range(_peak_i, len(_n70)):
                    if _n70[_si] < 0.5:
                        _sp_end = _si
                        break
            sp_result: dict = {
                "has_signal_peptide": _sp_det,
                "signal_peptide_prob": round(_sp_max, 4),
                "sp_end": _sp_end,
                "h_start": 1 if _sp_det else 0,
                "h_end": _sp_end if _sp_det else 0,
                "cleavage_site": _sp_end,
                "h_region_score": 0.0,
            }
        else:
            _sp_cls = predict_signal_peptide(seq)
            sp_result = {
                **_sp_cls,
                "has_signal_peptide": _sp_cls["d_score"] >= 0.5,
                "signal_peptide_prob": round(_sp_cls["d_score"], 4),
                "sp_end": _sp_cls["cleavage_site"],
            }

        # --- Additional BiLSTM heads ---
        from beer.utils.structure import bilstm_predict as _bp
        def _run_head(loader):
            _h = loader()
            return _bp(seq, embedder, _h)

        tm_bilstm_profile    = _run_head(load_transmembrane_head)
        intramem_bilstm_profile = _run_head(load_intramembrane_head)
        cc_bilstm_profile    = _run_head(load_coiled_coil_head)
        dna_bilstm_profile   = _run_head(load_dna_binding_head)
        act_bilstm_profile   = _run_head(load_active_site_head)
        bnd_bilstm_profile   = _run_head(load_binding_site_head)
        phos_bilstm_profile  = _run_head(load_phosphorylation_head)
        lcd_bilstm_profile   = _run_head(load_lcd_head)
        znf_bilstm_profile   = _run_head(load_zinc_finger_head)
        glyc_bilstm_profile  = _run_head(load_glycosylation_head)
        ubiq_bilstm_profile  = _run_head(load_ubiquitination_head)
        meth_bilstm_profile  = _run_head(load_methylation_head)
        acet_bilstm_profile  = _run_head(load_acetylation_head)
        lipid_bilstm_profile = _run_head(load_lipidation_head)
        disulf_bilstm_profile = _run_head(load_disulfide_head)
        motif_bilstm_profile  = _run_head(load_motif_head)
        prop_bilstm_profile   = _run_head(load_propeptide_head)
        rep_bilstm_profile    = _run_head(load_repeat_head)
        rnabind_bilstm_profile  = _run_head(load_rna_binding_head)
        nucbind_bilstm_profile  = _run_head(load_nucleotide_binding_head)
        transit_bilstm_profile  = _run_head(load_transit_peptide_head)
        agg_bilstm_profile      = _run_head(load_aggregation_head)

        def _bilstm_head_html(
            section_title: str,
            feature_label: str,
            scores: "list[float] | None",
            train_labels: str,
            classical_note: str,
            ref_note: str,
        ) -> str:
            """Shared HTML template for every BiLSTM head analysis section."""
            _b = method_badge("ESM2 BiLSTM", "bilstm")
            if scores is None:
                return _style + (
                    f"<h2>{section_title} {_b}</h2>"
                    f"<div style='background:#fff8e1;border-left:4px solid #f59e0b;"
                    f"padding:12px 16px;border-radius:4px;margin:12px 0'>"
                    f"<b style='color:#b45309'>⏳ Model training in progress</b><br>"
                    f"<p style='margin:6px 0 0'>The <b>{feature_label}</b> BiLSTM head "
                    f"is currently being trained on UniProt Swiss-Prot annotations. "
                    f"This section will populate automatically once training completes "
                    f"and the model file is available. No action required.</p>"
                    f"</div>"
                )
            _scores_np = scores
            _n = seq_length
            _mean = sum(_scores_np) / _n
            _n_pos = sum(1 for v in _scores_np if v > 0.5)
            _frac  = _n_pos / _n
            # Predicted regions
            _regs, _in, _st = [], False, 0
            for _i, _v in enumerate(_scores_np):
                if _v > 0.5 and not _in:
                    _in, _st = True, _i + 1
                elif _v <= 0.5 and _in:
                    _regs.append((_st, _i))
                    _in = False
            if _in:
                _regs.append((_st, _n))
            _reg_str = (
                ", ".join(f"{rs}–{re}" for rs, re in _regs[:8])
                + ("…" if len(_regs) > 8 else "")
            ) if _regs else "None"
            return _style + f"""
            <h2>{section_title} {_b}</h2>
            <table>
              <tr><th>Property</th><th>Value</th></tr>
              <tr><td>Mean per-residue probability</td><td>{_mean:.3f}</td></tr>
              <tr><td>Predicted positive fraction (score &gt; 0.5)</td>
                  <td>{_frac:.3f} &mdash; {_n_pos} / {_n} residues ({_frac*100:.1f}%)</td></tr>
              <tr><td>Predicted regions (1-based)</td><td>{_reg_str}</td></tr>
              <tr><td>Number of predicted regions</td><td>{len(_regs)}</td></tr>
            </table>
            <p class="note">
              BiLSTM: ESM2 650M (1280-dim, frozen) → 2-layer Bidirectional LSTM
              (hidden = 256) → Linear(512 → 1) → Sigmoid; trained on {train_labels}.
              Threshold 0.5. {classical_note}
              {ref_note}
            </p>
            """

        _TM_NOTE = (
            "For full topology predictions run TMHMM via the dedicated button."
        )
        _CC_NOTE = (
            "Classical comparison: heptad-weighted Chou-Fasman propensity — see Coiled-Coil Profile tab."
        )

        tm_html_bilstm = _bilstm_head_html(
            "Transmembrane Helices — BiLSTM", "transmembrane",
            tm_bilstm_profile,
            "UniProt Swiss-Prot 'Transmembrane' (ft_transmem) annotations",
            _TM_NOTE,
            "Predicted residues form TM helix segments (topology inferred from inside-positive rule). "
            "For orthogonal validation use DeepTMHMM or TMHMM.",
        )
        cc_html_bilstm = _bilstm_head_html(
            "Coiled-Coil Regions — BiLSTM", "coiled_coil",
            cc_bilstm_profile,
            "UniProt Swiss-Prot 'Coiled coil' (ft_coiled) annotations",
            _CC_NOTE,
            "Coiled coils are α-helical assemblies with heptad repeats (a-b-c-d-e-f-g)ₙ; "
            "positions a/d are typically hydrophobic (Lupas et al. 1991 Science 252:1162).",
        )
        dna_html_bilstm = _bilstm_head_html(
            "DNA-Binding Regions — BiLSTM", "dna_binding",
            dna_bilstm_profile,
            "UniProt Swiss-Prot 'DNA binding' (ft_dna_bind) annotations",
            "",
            "Predicted residues contact DNA in the native structure. "
            "Common motifs: helix-turn-helix, zinc finger, leucine zipper.",
        )
        act_html_bilstm = _bilstm_head_html(
            "Active Site Residues — BiLSTM", "active_site",
            act_bilstm_profile,
            "UniProt Swiss-Prot 'Active site' (ft_act_site) annotations",
            "",
            "Catalytic residues directly participate in the enzymatic reaction. "
            "Single-residue annotations in UniProt; BiLSTM smooths to per-residue probability.",
        )
        bnd_html_bilstm = _bilstm_head_html(
            "Binding Site Residues — BiLSTM", "binding_site",
            bnd_bilstm_profile,
            "UniProt Swiss-Prot 'Binding site' (ft_binding) annotations",
            "",
            "Residues that contact small-molecule ligands, metal ions, or cofactors "
            "in the native structure.",
        )
        phos_html_bilstm = _bilstm_head_html(
            "Phosphorylation Sites — BiLSTM", "phosphorylation",
            phos_bilstm_profile,
            "UniProt Swiss-Prot phospho-Ser/Thr/Tyr (ft_mod_res) annotations",
            "Classical comparison: kinase consensus motifs (PKA, CK2, CDK) — see Phosphorylation section.",
            "Phosphorylation regulates activity, localisation, and interactions. "
            "Predicted S/T/Y residues with high scores are candidate phosphosites.",
        )
        lcd_html_bilstm = _bilstm_head_html(
            "Low-Complexity / Compositionally Biased Regions — BiLSTM", "lcd",
            lcd_bilstm_profile,
            "UniProt Swiss-Prot 'Compositionally biased' (ft_compbias) annotations",
            "Classical comparison: sliding-window Shannon entropy (see Local Complexity tab).",
            "Low-complexity regions are enriched in a small subset of amino acids and are "
            "typically disordered; they drive liquid-liquid phase separation and stress-granule assembly.",
        )
        intramem_html_bilstm = _bilstm_head_html(
            "Intramembrane Regions — BiLSTM", "intramembrane",
            intramem_bilstm_profile,
            "UniProt Swiss-Prot 'Intramembrane' (ft_intramem) annotations",
            "",
            "Intramembrane regions penetrate the lipid bilayer but do not fully span it; "
            "common in re-entrant loops of ion channels and transporters.",
        )
        znf_html_bilstm = _bilstm_head_html(
            "Zinc Finger Regions — BiLSTM", "zinc_finger",
            znf_bilstm_profile,
            "UniProt Swiss-Prot 'Zinc finger' (ft_zn_fing) annotations",
            "",
            "Zinc fingers are small structural motifs coordinating Zn²⁺ via Cys/His residues; "
            "involved in DNA/RNA binding, protein–protein interactions, and ubiquitination (RING domains).",
        )
        glyc_html_bilstm = _bilstm_head_html(
            "Glycosylation Sites — BiLSTM", "glycosylation",
            glyc_bilstm_profile,
            "UniProt Swiss-Prot N/O-linked glycan (ft_carbohyd) annotations",
            "",
            "N-glycosylation occurs on Asn (NxS/T sequon); O-glycosylation on Ser/Thr. "
            "Glycans influence folding, stability, trafficking, and immune recognition.",
        )
        ubiq_html_bilstm = _bilstm_head_html(
            "Ubiquitination Sites — BiLSTM", "ubiquitination",
            ubiq_bilstm_profile,
            "UniProt Swiss-Prot 'Ubiquitin' ft_mod_res annotations",
            "",
            "Ubiquitination on Lys regulates proteasomal degradation, DNA repair, "
            "endocytosis, and NF-κB signalling. Poly-Ub chains via K48 target degradation; "
            "K63 chains serve non-degradative signalling roles.",
        )
        meth_html_bilstm = _bilstm_head_html(
            "Methylation Sites — BiLSTM", "methylation",
            meth_bilstm_profile,
            "UniProt Swiss-Prot 'Methyl' ft_mod_res annotations",
            "",
            "Methylation of Lys/Arg is a key epigenetic mark on histones and regulates "
            "gene expression, RNA processing, and protein–protein interactions.",
        )
        acet_html_bilstm = _bilstm_head_html(
            "Acetylation Sites — BiLSTM", "acetylation",
            acet_bilstm_profile,
            "UniProt Swiss-Prot 'Acetyl' ft_mod_res annotations",
            "",
            "N-terminal acetylation occurs co-translationally on ~80% of human proteins. "
            "Lys acetylation (e.g., histones) competes with ubiquitination and regulates "
            "chromatin accessibility and enzyme activity.",
        )
        lipid_html_bilstm = _bilstm_head_html(
            "Lipidation Sites — BiLSTM", "lipidation",
            lipid_bilstm_profile,
            "UniProt Swiss-Prot GPI/myristoyl/palmitoyl (ft_lipid) annotations",
            "",
            "Lipid modifications anchor proteins to membranes. Myristoylation (Gly-1), "
            "palmitoylation (Cys), GPI anchors (C-terminal), and prenylation (CAAX Cys) "
            "each target different membrane compartments.",
        )
        disulf_html_bilstm = _bilstm_head_html(
            "Disulfide Bonds — BiLSTM", "disulfide",
            disulf_bilstm_profile,
            "UniProt Swiss-Prot 'Disulfide bond' (ft_disulfid) annotations",
            "",
            "Disulfide bonds (Cys–Cys) stabilise extracellular and secreted proteins. "
            "The BiLSTM predicts which Cys residues participate; bond pairing is not predicted.",
        )
        motif_html_bilstm = _bilstm_head_html(
            "Functional Sequence Motifs — BiLSTM", "motif",
            motif_bilstm_profile,
            "UniProt Swiss-Prot 'Motif' (ft_motif) annotations",
            "Classical comparison: regex-based linear motif scan — see Linear Motifs section.",
            "Short functional motifs mediate post-translational modifications, localisation "
            "signals, and protein–protein interactions (e.g., NLS, NES, KDEL, SH3-binding).",
        )
        prop_html_bilstm = _bilstm_head_html(
            "Propeptide Regions — BiLSTM", "propeptide",
            prop_bilstm_profile,
            "UniProt Swiss-Prot 'Propeptide' (ft_propep) annotations",
            "",
            "Propeptides are cleavage products of precursor proteins that are removed "
            "during maturation (zymogen activation, signal peptide removal). "
            "They often act as intramolecular chaperones.",
        )
        rep_html_bilstm = _bilstm_head_html(
            "Repeat Regions — BiLSTM", "repeat",
            rep_bilstm_profile,
            "UniProt Swiss-Prot 'Repeat' (ft_repeat) annotations",
            "Classical comparison: tandem repeat detection — see Tandem Repeats section.",
            "Repeat regions include ankyrin, armadillo, WD40, HEAT, and leucine-rich repeats; "
            "they mediate protein–protein interactions and structural scaffolding.",
        )
        rnabind_html_bilstm = _bilstm_head_html(
            "RNA-Binding Regions — BiLSTM", "rna_binding",
            rnabind_bilstm_profile,
            "UniProt Swiss-Prot 'RNA binding' region (ft_region, description_filter='RNA-binding') annotations",
            "Classical comparison: catRAPID sliding-window score — see RNA Binding section.",
            "RNA-binding regions include RRM, KH, DEAD-box helicase, and intrinsically disordered "
            "low-complexity regions that engage RNA via non-canonical contacts.",
        )
        nucbind_html_bilstm = _bilstm_head_html(
            "Nucleotide-Binding Sites — BiLSTM", "nucleotide_binding",
            nucbind_bilstm_profile,
            "UniProt Swiss-Prot 'Nucleotide binding' site (ft_binding, nucleotide-filtered) annotations",
            "",
            "Nucleotide-binding sites coordinate ATP, GTP, NAD⁺, FAD, and related cofactors. "
            "Walker A/B (P-loop NTPases), Rossmann fold, and GHKL ATPase motifs are the most common architectures.",
        )
        transit_html_bilstm = _bilstm_head_html(
            "Transit Peptides — BiLSTM", "transit_peptide",
            transit_bilstm_profile,
            "UniProt Swiss-Prot 'Transit peptide' (ft_transit) annotations",
            "Classical comparison: TargetP / MitoFates N-terminal scoring.",
            "Mitochondrial and chloroplast transit peptides are N-terminal targeting sequences "
            "cleaved upon import. They are typically rich in basic and hydroxylated residues "
            "and lack acidic residues (von Heijne 1986 EMBO J 5:1335).",
        )

        # --- Amphipathic helices ---
        amph_html    = format_amphipathic_report(seq, _accent)
        moment_alpha = calc_hydrophobic_moment_profile(seq, angle_deg=100.0)
        moment_beta  = calc_hydrophobic_moment_profile(seq, angle_deg=160.0)
        amph_regions = predict_amphipathic_helices(seq)

        # --- SCD + SHD ---
        scd_val          = calc_scd(seq)
        scd_profile_data = calc_scd_profile(seq, window=20)
        scd_blocks       = calc_pos_neg_block_lengths(seq)
        scd_html         = format_scd_report(seq, _accent)
        shd_profile_data = calc_shd_profile(seq, _scale_vals, window=20)

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

        # --- Phosphorylation ---
        phospho_html  = format_phospho_report(seq, _accent,
                                              disorder_scores=disorder_scores)
        phospho_sites = predict_phosphorylation(seq)

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
                "TM Helices (BiLSTM)":         tm_html_bilstm,
                "Intramembrane (BiLSTM)":      intramem_html_bilstm,
                "Coiled-Coil (BiLSTM)":        cc_html_bilstm,
                "DNA-Binding (BiLSTM)":        dna_html_bilstm,
                "Active Site (BiLSTM)":        act_html_bilstm,
                "Binding Site (BiLSTM)":       bnd_html_bilstm,
                "Phosphorylation (BiLSTM)":    phos_html_bilstm,
                "Low-Complexity (BiLSTM)":     lcd_html_bilstm,
                "Zinc Finger (BiLSTM)":        znf_html_bilstm,
                "Glycosylation (BiLSTM)":      glyc_html_bilstm,
                "Ubiquitination (BiLSTM)":     ubiq_html_bilstm,
                "Methylation (BiLSTM)":        meth_html_bilstm,
                "Acetylation (BiLSTM)":        acet_html_bilstm,
                "Lipidation (BiLSTM)":         lipid_html_bilstm,
                "Disulfide Bonds (BiLSTM)":    disulf_html_bilstm,
                "Functional Motifs (BiLSTM)":  motif_html_bilstm,
                "Propeptide (BiLSTM)":         prop_html_bilstm,
                "Repeat Regions (BiLSTM)":          rep_html_bilstm,
                "RNA-Binding (BiLSTM)":             rnabind_html_bilstm,
                "Nucleotide-Binding (BiLSTM)":      nucbind_html_bilstm,
                "Transit Peptide (BiLSTM)":         transit_html_bilstm,
                "LARKS":                   larks_html,
                "Linear Motifs":           motifs_html,
                "\u03b2-Aggregation & Solubility": aggr_html,
                "Signal Peptide & GPI":    signal_html,
                "Amphipathic Helices":     amph_html,
                "Charge Decoration (SCD)": scd_html,
                "RNA Binding":             rbp_html,
                "Tandem Repeats":          tandem_html,
                "Proteolytic Map":         prot_html,
                "Phosphorylation":         phospho_html,
            },
            "tm_helices":      tm_helices,
            "aa_counts":       aa_counts,
            "aa_freq":         aa_freq,
            "hydro_profile":   sliding_window_hydrophobicity(seq, window_size, HYDROPHOBICITY_SCALES.get(hydro_scale, HYDROPHOBICITY_SCALES["Kyte-Doolittle"])["values"]),
            "hydro_scale":     hydro_scale,
            "ncpr_profile":    sliding_window_ncpr(seq, window_size),
            "entropy_profile": sliding_window_entropy(seq, window_size),
            "disorder_scores":      disorder_scores,
            "disorder_uncertainty": disorder_uncertainty,
            "disorder_method":      _disorder_method,
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
            "disorder_has_bilstm":      _disorder_has_bilstm,
            "sp_result":          sp_result,
            "sp_bilstm_profile":  sp_bilstm_profile,
            "gpi_result":         gpi_result,
            "tm_bilstm_profile":      tm_bilstm_profile,
            "intramem_bilstm_profile": intramem_bilstm_profile,
            "cc_bilstm_profile":      cc_bilstm_profile,
            "dna_bilstm_profile":     dna_bilstm_profile,
            "act_bilstm_profile":     act_bilstm_profile,
            "bnd_bilstm_profile":     bnd_bilstm_profile,
            "phos_bilstm_profile":    phos_bilstm_profile,
            "lcd_bilstm_profile":     lcd_bilstm_profile,
            "znf_bilstm_profile":     znf_bilstm_profile,
            "glyc_bilstm_profile":    glyc_bilstm_profile,
            "ubiq_bilstm_profile":    ubiq_bilstm_profile,
            "meth_bilstm_profile":    meth_bilstm_profile,
            "acet_bilstm_profile":    acet_bilstm_profile,
            "lipid_bilstm_profile":   lipid_bilstm_profile,
            "disulf_bilstm_profile":  disulf_bilstm_profile,
            "motif_bilstm_profile":   motif_bilstm_profile,
            "prop_bilstm_profile":    prop_bilstm_profile,
            "rep_bilstm_profile":       rep_bilstm_profile,
            "rnabind_bilstm_profile":   rnabind_bilstm_profile,
            "nucbind_bilstm_profile":   nucbind_bilstm_profile,
            "transit_bilstm_profile":   transit_bilstm_profile,
            "agg_bilstm_profile":       agg_bilstm_profile,
            "moment_alpha":    moment_alpha,
            "moment_beta":     moment_beta,
            "amph_regions":    amph_regions,
            "scd":             scd_val,
            "scd_profile":     scd_profile_data,
            "scd_blocks":      scd_blocks,
            "shd_profile":     shd_profile_data,
            "hydro_scale":     hydro_scale,
            "catgranule":      catgranule_score,
            "catgranule_profile": catgranule_profile,
            "rbp":             rbp_result,
            "rbp_profile":     rbp_profile_data,
            "tandem_stats":    tandem_stats,
            "polyx_stretches": polyx_stretches,
            "prot_sites":      prot_sites,
            "phospho_sites":   phospho_sites,
            "plaac":           _plaac,
            "summary_bullets": _make_summary_bullets(
                seq=seq,
                seq_length=seq_length,
                mw=mol_weight,
                pI=iso_point,
                disorder_f=disorder_f,
                disorder_scores=disorder_scores,
                sp_bilstm_profile=sp_bilstm_profile,
                tm_bilstm_profile=tm_bilstm_profile,
                cc_bilstm_profile=cc_bilstm_profile,
                aggr_profile=aggr_profile_zygg,
                catgranule=catgranule_score,
                rbp=rbp_result,
                fcr=fcr,
                ncpr=ncpr,
                kappa=kappa,
                larks=larks,
                gpi_result=gpi_result,
                phospho_sites=phospho_sites,
                prot_sites=prot_sites,
            ),
        }
