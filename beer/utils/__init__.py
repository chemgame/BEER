"""BEER utility functions — re-exports of public API."""
from beer.utils.sequence import clean_sequence, is_valid_protein, format_sequence_block
from beer.utils.biophysics import (
    calc_net_charge,
    sliding_window_hydrophobicity,
    calc_shannon_entropy,
    sliding_window_ncpr,
    sliding_window_entropy,
    calc_kappa,
    calc_omega,
    count_pairs,
    fraction_low_complexity,
    sticker_spacing_stats,
)
from beer.utils.structure import (
    calc_disorder_profile,
    predict_tm_helices,
    detect_larks,
    predict_coiled_coil,
    scan_linear_motifs,
)
from beer.utils.pdb import (
    extract_phi_psi,
    compute_ca_distance_matrix,
    extract_plddt_from_pdb,
    import_pdb_sequence,
    extract_chain_structures,
)

__all__ = [
    "clean_sequence",
    "is_valid_protein",
    "format_sequence_block",
    "calc_net_charge",
    "sliding_window_hydrophobicity",
    "calc_shannon_entropy",
    "sliding_window_ncpr",
    "sliding_window_entropy",
    "calc_kappa",
    "calc_omega",
    "count_pairs",
    "fraction_low_complexity",
    "sticker_spacing_stats",
    "calc_disorder_profile",
    "predict_tm_helices",
    "detect_larks",
    "predict_coiled_coil",
    "scan_linear_motifs",
    "extract_phi_psi",
    "compute_ca_distance_matrix",
    "extract_plddt_from_pdb",
    "import_pdb_sequence",
    "extract_chain_structures",
]
