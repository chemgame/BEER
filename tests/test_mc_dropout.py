"""MC-dropout must work for every head architecture, not only plain BiLSTM.

Uses the shipped weights with a fixed random embedding (no ESM C needed); skips when the
weight files are not present locally.
"""
from __future__ import annotations

import functools
import pathlib

import numpy as np
import pytest

from beer import models
from beer.models.metrics import HEAD_METRICS
from beer.utils import structure as st

MODELS_DIR = pathlib.Path(models.__file__).parent
L = 24
EMB = np.random.default_rng(0).normal(size=(L, 1152)).astype(np.float32)
SEQ = "A" * L
BINARY = sorted(t for t in HEAD_METRICS if t != "secondary_structure")


class _FixedEmbedder:
    def is_available(self):
        return True

    def embed(self, seq):
        return EMB


@functools.lru_cache(maxsize=None)
def _head(task):
    if not (MODELS_DIR / f"{task}_head.npz").exists():
        pytest.skip(f"{task}_head.npz not present locally")
    return getattr(models, f"load_{task}_head")()


@pytest.mark.parametrize("task", BINARY)
def test_without_dropout_mc_equals_the_deterministic_profile(task):
    head = _head(task)
    det = np.asarray(st.bilstm_predict(SEQ, _FixedEmbedder(), head))
    mean, sd = st.bilstm_predict_mc(SEQ, _FixedEmbedder(), head, n_passes=2, dropout_p=0.0)
    np.testing.assert_allclose(mean, det, atol=1e-5)
    assert np.allclose(sd, 0.0)


@pytest.mark.parametrize("task", BINARY)
def test_dropout_gives_a_spread_of_valid_probabilities(task):
    mean, sd = st.bilstm_predict_mc(SEQ, _FixedEmbedder(), _head(task),
                                    n_passes=8, dropout_p=0.3, seed=1)
    mean, sd = np.asarray(mean), np.asarray(sd)
    assert mean.shape == sd.shape == (L,)
    assert ((mean >= 0) & (mean <= 1)).all()
    assert (sd >= 0).all() and sd.max() > 0


@pytest.mark.parametrize("cls, slot", [(1, 0), (2, 1), (0, 2)])   # helix, strand, coil
def test_secondary_structure_classes(cls, slot):
    head = _head("secondary_structure")
    det = st.bilstm_predict_ss3(SEQ, _FixedEmbedder(), head)[slot]
    mean, sd = st.bilstm_predict_mc(SEQ, _FixedEmbedder(), head, n_passes=2,
                                    dropout_p=0.0, ss3_class=cls)
    np.testing.assert_allclose(mean, det, atol=1e-5)
    _, sd = st.bilstm_predict_mc(SEQ, _FixedEmbedder(), head, n_passes=6,
                                 dropout_p=0.3, ss3_class=cls, seed=2)
    assert max(sd) > 0


def test_secondary_structure_needs_a_class():
    assert st.bilstm_predict_mc(SEQ, _FixedEmbedder(), _head("secondary_structure")) is None


def test_seed_makes_passes_reproducible():
    head = _head("disorder")
    a = st.bilstm_predict_mc(SEQ, _FixedEmbedder(), head, n_passes=4, seed=7)
    b = st.bilstm_predict_mc(SEQ, _FixedEmbedder(), head, n_passes=4, seed=7)
    assert a == b
