"""BEER analysis modules — re-exports of all public analysis functions."""

from beer.analysis.aggregation import (
    calc_aggregation_profile,
    predict_aggregation_hotspots,
    calc_camsolmt_score,
    calc_solubility_stats,
    format_aggregation_report,
)
from beer.analysis.signal_peptide import (
    predict_signal_peptide,
    predict_gpi_anchor,
    format_signal_report,
)
from beer.analysis.amphipathic import (
    calc_hydrophobic_moment,
    calc_hydrophobic_moment_profile,
    predict_amphipathic_helices,
    format_amphipathic_report,
)
from beer.analysis.scd import (
    calc_scd,
    calc_scd_profile,
    calc_charge_segregation_score,
    calc_mean_block_length,
    calc_pos_neg_block_lengths,
    format_scd_report,
)
from beer.analysis.rnabinding import (
    calc_rbp_score,
    calc_rbp_profile,
    format_rbp_report,
)
from beer.analysis.tandem_repeats import (
    find_tandem_repeats,
    find_direct_repeats,
    find_compositional_repeats,
    calc_repeat_stats,
    format_repeats_report,
    format_tandem_repeats_report,
)

__all__ = [
    # aggregation
    "calc_aggregation_profile",
    "predict_aggregation_hotspots",
    "calc_camsolmt_score",
    "calc_solubility_stats",
    "format_aggregation_report",
    # signal_peptide
    "predict_signal_peptide",
    "predict_gpi_anchor",
    "format_signal_report",
    # amphipathic
    "calc_hydrophobic_moment",
    "calc_hydrophobic_moment_profile",
    "predict_amphipathic_helices",
    "format_amphipathic_report",
    # scd
    "calc_scd",
    "calc_scd_profile",
    "calc_charge_segregation_score",
    "calc_mean_block_length",
    "calc_pos_neg_block_lengths",
    "format_scd_report",
    # rnabinding
    "calc_rbp_score",
    "calc_rbp_profile",
    "format_rbp_report",
    # tandem_repeats
    "find_tandem_repeats",
    "find_direct_repeats",
    "find_compositional_repeats",
    "calc_repeat_stats",
    "format_repeats_report",
    "format_tandem_repeats_report",
]
