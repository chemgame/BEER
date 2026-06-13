"""PSSM from an aligned MSA: log-odds, conservation, CSV serialisation."""
from beer.analysis.msa_pssm import (
    AA_ORDER, compute_pssm, pssm_to_csv, consensus_sequence,
)


def test_conserved_column_scores_high_for_present_residue():
    # Column 0 is all 'A' → A's log-odds should be the column maximum.
    aln = ["ACD", "AGD", "AHD"]
    rows, conservation, coverage = compute_pssm(aln)
    assert len(rows) == 3 and len(conservation) == 3
    assert coverage[0] == 3
    col0 = rows[0]
    assert col0["A"] == max(col0.values())
    # A fully conserved column carries more information than a variable one.
    assert conservation[0] > conservation[1]


def test_gaps_excluded_from_coverage():
    aln = ["A-C", "A-C", "AGC"]
    _, _, coverage = compute_pssm(aln)
    assert coverage[1] == 1   # only one non-gap residue in the gappy column


def test_unequal_length_raises():
    import pytest
    with pytest.raises(ValueError):
        compute_pssm(["ACD", "AC"])


def test_csv_roundtrip_shape():
    aln = ["ACD", "AGD", "AHD"]
    rows, cons, cov = compute_pssm(aln)
    csv = pssm_to_csv(rows, cons, cov, consensus_sequence(aln))
    lines = csv.strip().splitlines()
    assert lines[0].startswith("position,consensus,coverage,conservation_bits,")
    assert lines[0].endswith(",".join(AA_ORDER))
    assert len(lines) == 1 + 3   # header + one row per column


def test_consensus_picks_majority_residue():
    aln = ["AAA", "AAC", "ATA"]
    assert consensus_sequence(aln)[0] == "A"
