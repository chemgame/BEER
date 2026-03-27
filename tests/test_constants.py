"""Tests for beer.constants — verify scales cover all 20 standard amino acids."""
from __future__ import annotations
import pytest
from beer.constants import (
    ZYGGREGATOR_PROPENSITY,
    PASTA_ENERGY,
    CAMSOLMT_SCALE,
    KYTE_DOOLITTLE,
    EISENBERG_SCALE,
    COILED_COIL_PROPENSITY,
    CHOU_FASMAN_HELIX,
    CHOU_FASMAN_SHEET,
    DISORDER_PROPENSITY,
    RBP_RESIDUE_PROPENSITY,
    VALID_AMINO_ACIDS,
)

ALL_20 = set("ACDEFGHIKLMNPQRSTVWY")


@pytest.mark.parametrize("scale,name", [
    (ZYGGREGATOR_PROPENSITY, "ZYGGREGATOR_PROPENSITY"),
    (PASTA_ENERGY,           "PASTA_ENERGY"),
    (CAMSOLMT_SCALE,         "CAMSOLMT_SCALE"),
    (KYTE_DOOLITTLE,         "KYTE_DOOLITTLE"),
    (EISENBERG_SCALE,        "EISENBERG_SCALE"),
    (COILED_COIL_PROPENSITY, "COILED_COIL_PROPENSITY"),
    (CHOU_FASMAN_HELIX,      "CHOU_FASMAN_HELIX"),
    (CHOU_FASMAN_SHEET,      "CHOU_FASMAN_SHEET"),
    (DISORDER_PROPENSITY,    "DISORDER_PROPENSITY"),
    (RBP_RESIDUE_PROPENSITY, "RBP_RESIDUE_PROPENSITY"),
])
def test_scale_has_all_20_amino_acids(scale, name):
    """Every amino acid scale must contain exactly the 20 standard amino acids."""
    assert set(scale.keys()) == ALL_20, (
        f"{name} is missing amino acids: {ALL_20 - set(scale.keys())}"
    )


def test_valid_amino_acids_set():
    """VALID_AMINO_ACIDS must be the standard 20-letter set."""
    assert VALID_AMINO_ACIDS == ALL_20


def test_scale_values_are_numeric():
    """All scale values must be real numbers (int or float)."""
    for aa, val in KYTE_DOOLITTLE.items():
        assert isinstance(val, (int, float)), f"KD[{aa}] = {val!r} is not numeric"
