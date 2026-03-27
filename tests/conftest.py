"""Shared fixtures for BEER test suite."""
from __future__ import annotations
import pytest
import numpy as np

FUS_LC = "MASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSGYGQSSYSSYGQSQ"
SHORT_SEQ = "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG"
HYDROPHOBIC_SEQ = "MIIIIIIIIIIIVVVVVVVLLLLLL"
CHARGED_SEQ = "MEKEKEKEKEKRKRKRKRKR"


class MockEmbedder:
    def __init__(self, dim: int = 320, available: bool = True):
        self._dim = dim
        self._available = available

    def is_available(self) -> bool:
        return self._available

    def embed(self, seq: str):
        if not self._available:
            return None
        return np.zeros((len(seq), self._dim), dtype=np.float32)

    def clear_cache(self) -> None:
        pass


class UnavailableEmbedder(MockEmbedder):
    def __init__(self):
        super().__init__(available=False)


@pytest.fixture
def fus_lc():
    return FUS_LC


@pytest.fixture
def short_seq():
    return SHORT_SEQ


@pytest.fixture
def hydrophobic_seq():
    return HYDROPHOBIC_SEQ


@pytest.fixture
def charged_seq():
    return CHARGED_SEQ


@pytest.fixture
def mock_embedder():
    return MockEmbedder(dim=320)


@pytest.fixture
def unavailable_embedder():
    return UnavailableEmbedder()


@pytest.fixture
def disorder_head_weights():
    return {
        "coef": np.zeros((1, 320), dtype=np.float32),
        "intercept": np.array([0.0], dtype=np.float32),
        "model_name": "test",
        "trained_on": "test",
    }
