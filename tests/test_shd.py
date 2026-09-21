"""SHD follows the published definition of Zheng et al. (J. Phys. Chem. Lett. 2020, 11, 3408)."""
import itertools

import pytest

from beer.analysis.scd import HPS_KR_LAMBDA, calc_shd, calc_shd_profile


def _reference(seq):
    n = len(seq)
    lam = [HPS_KR_LAMBDA.get(a, 0.0) for a in seq]
    return sum((lam[i] + lam[j]) / (j - i) for i, j in itertools.combinations(range(n), 2)) / n


@pytest.mark.parametrize("seq", ["AA", "RRRR", "LKLKLKLK", "MDEKLLIVPFGW", "GSGSGSYGSG"])
def test_matches_the_published_formula(seq):
    assert calc_shd(seq) == pytest.approx(_reference(seq))


def test_two_residue_value():
    # (lambda_A + lambda_A) / 1, divided by N = 2
    assert calc_shd("AA") == pytest.approx(HPS_KR_LAMBDA["A"])


def test_never_negative_and_zero_for_arginine():
    assert calc_shd("RRRRRRRR") == 0.0
    assert calc_shd("MDEKLLIVPFGWRRQ") > 0.0


def test_hydrophobic_residues_together_score_higher():
    # Same composition; hydrophobic residues adjacent versus spread apart
    together = "LLLLRRRR"
    apart = "LRLRLRLR"
    assert calc_shd(together) > calc_shd(apart)


def test_uses_its_own_scale_not_a_user_scale():
    assert calc_shd("MDEKLLIV") == calc_shd("MDEKLLIV", None)


def test_profile_window():
    seq = "MDEKLLIVPFGWRRQSTNAYCH" * 2
    prof = calc_shd_profile(seq, window=20)
    assert len(prof) == len(seq) - 20 + 1
    assert prof[0] == pytest.approx(calc_shd(seq[:20]))
