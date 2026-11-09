"""Offline mode blocks every outbound request; sequence uploads ask first."""
import os

import pytest

from beer import config
from beer.network import policy


def _settings(monkeypatch, **values):
    real = config.get
    monkeypatch.setattr(config, "get", lambda k: values[k] if k in values else real(k))


def test_defaults_allow_network_and_ask_before_uploads(monkeypatch):
    _settings(monkeypatch, offline_mode=False, confirm_uploads=None)
    assert not policy.offline()
    assert policy.confirm_uploads()


def test_offline_sets_hugging_face_offline(monkeypatch):
    _settings(monkeypatch, offline_mode=True)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    policy.apply_to_environment()
    assert os.environ.get("HF_HUB_OFFLINE") == "1"


def test_only_blast_and_esmfold2_upload_the_sequence():
    assert set(policy.SEQUENCE_UPLOADS) == {"NCBI BLAST", "ESMFold2 structure prediction"}


def test_weights_are_not_downloaded_offline(monkeypatch, tmp_path):
    import beer.models as m
    _settings(monkeypatch, offline_mode=True)
    monkeypatch.setattr(m, "_user_cache_dir", lambda: tmp_path)
    called = []
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: called.append(a))
    assert m._download("x_head.npz", {"url": "https://example.org/x_head.npz", "sha256": "0"}) is None
    assert not called


def test_gui_gate(monkeypatch, qtbot=None):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QMessageBox
    QApplication.instance() or QApplication([])
    from beer.gui.main_window import ProteinAnalyzerGUI

    class _Stub:
        _network_permitted = ProteinAnalyzerGUI._network_permitted

    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.append(a))
    _settings(monkeypatch, offline_mode=True)
    assert _Stub()._network_permitted("DisProt") is False
    assert shown, "offline mode should explain why the request was blocked"
    shown.clear()
    assert _Stub()._network_permitted("UniProt summary", quiet=True) is False
    assert not shown
    _settings(monkeypatch, offline_mode=False, confirm_uploads=True)
    assert _Stub()._network_permitted("DisProt") is True


def test_structure_prediction_requests_an_esmfold2_model():
    from beer.network.workers import ESMFOLD2_MODEL
    assert ESMFOLD2_MODEL.startswith("esmfold2")
