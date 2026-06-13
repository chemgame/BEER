"""Deterministic numerics: net charge, isoelectric point, pKa sets."""
from beer.utils.biophysics import calc_net_charge, calc_isoelectric_point
from beer.constants import PKA_SETS


def test_charged_residue_signs():
    # A bare Ala carries only the termini, which roughly cancel near pH 7.
    assert abs(calc_net_charge("A", 7.0)) < 0.2
    # Lys is basic (+), Asp is acidic (-).
    assert calc_net_charge("KKKK", 7.0) > 2.0
    assert calc_net_charge("DDDD", 7.0) < -2.0


def test_net_charge_decreases_with_pH():
    seq = "ACDEFGHIKLMNPQRSTVWY"
    assert calc_net_charge(seq, 3.0) > calc_net_charge(seq, 7.0) > calc_net_charge(seq, 11.0)


def test_isoelectric_point_in_range_and_neutral():
    seq = "ACDEFGHIKLMNPQRSTVWYKKRRDDEE"
    pI = calc_isoelectric_point(seq)
    assert 0.0 < pI < 14.0
    # Net charge is ~0 at the isoelectric point.
    assert abs(calc_net_charge(seq, pI)) < 0.15


def test_pka_sets_shift_pi():
    seq = "ACDEFGHIKLMNPQRSTVWYKKRRDDEE"
    pis = {
        name: calc_isoelectric_point(seq, None if name == "BEER default" else tbl)
        for name, tbl in PKA_SETS.items()
    }
    # Different published sets must give measurably different pI.
    assert len({round(v, 2) for v in pis.values()}) >= 3
    # Bjellqvist (ProtParam) must differ from the BEER default.
    assert abs(pis["Bjellqvist (ProtParam)"] - pis["BEER default"]) > 0.05
