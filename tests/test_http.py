"""Tests for beer.network._http — fetch functions with mocked urllib."""
from __future__ import annotations
import json
import urllib.error
from unittest.mock import patch, MagicMock
import pytest

from beer.network._http import (
    fetch_elm,
    fetch_disprot,
    fetch_phasepdb,
    fetch_uniprot_fasta,
    _safe_int,
    _get_json,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(payload):
    """Return a mock context-manager response that yields JSON bytes."""
    raw = json.dumps(payload).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = raw
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _make_text_response(text: str):
    """Return a mock context-manager response that yields plain text bytes."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = text.encode("utf-8")
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


# ---------------------------------------------------------------------------
# _safe_int
# ---------------------------------------------------------------------------

def test_safe_int_converts_string():
    from beer.network._http import _safe_int
    assert _safe_int("42") == 42


def test_safe_int_default_on_none():
    from beer.network._http import _safe_int
    assert _safe_int(None, default=-1) == -1


def test_safe_int_default_on_bad_value():
    from beer.network._http import _safe_int
    assert _safe_int("not-a-number", default=0) == 0


# ---------------------------------------------------------------------------
# _get_json
# ---------------------------------------------------------------------------

def test_get_json_parses_json():
    payload = {"key": "value", "n": 42}
    with patch("urllib.request.urlopen", return_value=_make_response(payload)):
        result = _get_json("https://example.com/api")
    assert result == payload


def test_get_json_raises_on_network_error():
    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.URLError("network down"),
    ):
        with pytest.raises(urllib.error.URLError):
            _get_json("https://example.com/api")


# ---------------------------------------------------------------------------
# fetch_elm
# ---------------------------------------------------------------------------

_ELM_RESPONSE = [
    {
        "elm_identifier": "LIG_FHA_1",
        "start": 10,
        "end": 20,
        "logic": "true positive",
        "toGo": "",
        "primary_reference_pmed_id": "1234567",
        "accession": "ELM000001",
    }
]


def test_fetch_elm_returns_list():
    with patch("urllib.request.urlopen", return_value=_make_response(_ELM_RESPONSE)):
        result = fetch_elm("P04637")
    assert isinstance(result, list)


def test_fetch_elm_list_items_have_required_keys():
    with patch("urllib.request.urlopen", return_value=_make_response(_ELM_RESPONSE)):
        result = fetch_elm("P04637")
    if result:
        required = {"elm_identifier", "start", "end"}
        for item in result:
            assert required <= set(item.keys()), (
                f"ELM item missing keys: {required - set(item.keys())}"
            )


def test_fetch_elm_empty_response():
    with patch("urllib.request.urlopen", return_value=_make_response([])):
        result = fetch_elm("P00000")
    assert result == []


def test_fetch_elm_raises_on_http_error():
    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.HTTPError(
            url="", code=404, msg="Not Found", hdrs={}, fp=None
        ),
    ):
        with pytest.raises(urllib.error.HTTPError):
            fetch_elm("INVALID")


# ---------------------------------------------------------------------------
# fetch_disprot
# ---------------------------------------------------------------------------

_DISPROT_RESPONSE = {
    "disprot_id": "DP00086",
    "acc": "P04637",
    "regions": [
        {"start": 1, "end": 67, "term_name": "disordered region"},
    ],
}


def test_fetch_disprot_returns_dict():
    with patch("urllib.request.urlopen", return_value=_make_response(_DISPROT_RESPONSE)):
        result = fetch_disprot("P04637")
    assert isinstance(result, dict)


def test_fetch_disprot_regions_is_list():
    with patch("urllib.request.urlopen", return_value=_make_response(_DISPROT_RESPONSE)):
        result = fetch_disprot("P04637")
    assert "regions" in result
    assert isinstance(result["regions"], list)


def test_fetch_disprot_raises_on_network_error():
    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.URLError("offline"),
    ):
        with pytest.raises(urllib.error.URLError):
            fetch_disprot("P04637")


# ---------------------------------------------------------------------------
# fetch_phasepdb
# ---------------------------------------------------------------------------

_PHASEPDB_RESPONSE = {
    "accession": "P04637",
    "condensates": [{"name": "stress granule", "evidence": "in vivo"}],
}


def test_fetch_phasepdb_returns_dict():
    with patch("urllib.request.urlopen", return_value=_make_response(_PHASEPDB_RESPONSE)):
        result = fetch_phasepdb("P04637")
    assert isinstance(result, dict)


def test_fetch_phasepdb_raises_on_http_error():
    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.HTTPError(
            url="", code=500, msg="Server Error", hdrs={}, fp=None
        ),
    ):
        with pytest.raises(urllib.error.HTTPError):
            fetch_phasepdb("P04637")


# ---------------------------------------------------------------------------
# fetch_uniprot_fasta
# ---------------------------------------------------------------------------

_FASTA = ">sp|P04637|P53_HUMAN Cellular tumor antigen p53\nMESQSDASVEPPPQHLIRV\n"


def test_fetch_uniprot_fasta_returns_string():
    with patch("urllib.request.urlopen", return_value=_make_text_response(_FASTA)):
        result = fetch_uniprot_fasta("P04637")
    assert isinstance(result, str)


def test_fetch_uniprot_fasta_contains_fasta_content():
    with patch("urllib.request.urlopen", return_value=_make_text_response(_FASTA)):
        result = fetch_uniprot_fasta("P04637")
    assert ">" in result or len(result) > 0
