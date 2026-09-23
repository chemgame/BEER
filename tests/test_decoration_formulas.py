"""SCD and SHD must equal the published double sums, computed directly."""
import math
import random

import pytest

from beer.analysis.scd import HPS_KR_LAMBDA, calc_scd, calc_shd

CHARGE = {"K": 1.0, "R": 1.0, "D": -1.0, "E": -1.0}


def _scd_direct(seq):
    s = [CHARGE.get(a, 0.0) for a in seq]
    return sum(s[i] * s[j] * math.sqrt(j - i)
               for i in range(len(seq)) for j in range(i + 1, len(seq))) / len(seq)


def _shd_direct(seq):
    lam = [HPS_KR_LAMBDA.get(a, 0.0) for a in seq]
    return sum((lam[i] + lam[j]) / (j - i)
               for i in range(len(seq)) for j in range(i + 1, len(seq))) / len(seq)


def _sequences():
    rng = random.Random(0)
    aa = "ACDEFGHIKLMNPQRSTVWY"
    return ["EKEKEKEKEKEKEKEKEKEK", "EEEEEEEEEEKKKKKKKKKK", "GSGSGSGSGSGSGSGS",
            *["".join(rng.choice(aa) for _ in range(rng.randint(30, 120))) for _ in range(8)]]


def test_scd_matches_the_direct_sum():
    for seq in _sequences():
        assert calc_scd(seq) == pytest.approx(_scd_direct(seq), abs=1e-9)


def test_shd_matches_the_direct_sum():
    for seq in _sequences():
        assert calc_shd(seq) == pytest.approx(_shd_direct(seq), abs=1e-9)


def test_charge_decoration_signs():
    """Blocky opposite charges give a negative value, one dominant sign a positive one."""
    assert calc_scd("EEEEEEEEEEKKKKKKKKKK") < calc_scd("EKEKEKEKEKEKEKEKEKEK")
    assert calc_scd("KKKKKKKKKKKKKKKKKKKK") > 0
