"""Download-on-first-use head loader: manifest, resolution, verification."""
import json
import pathlib

import beer.models as m


def test_manifest_lists_all_heads_with_required_fields():
    man = m._load_manifest()
    assert len(man) == 24
    for name, meta in man.items():
        assert name.endswith(".npz")
        assert meta["sha256"] and isinstance(meta["bytes"], int)
        assert meta["url"].startswith("https://github.com/chemgame/BEER/releases/")


def test_manifest_file_is_valid_json():
    data = json.loads(m._MANIFEST_PATH.read_text(encoding="utf-8"))
    assert data["release_tag"] == "v3.0.0" and "files" in data


def test_verify_rejects_wrong_checksum(tmp_path):
    f = tmp_path / "x.npz"
    f.write_bytes(b"hello")
    assert m._verify(f, {"sha256": "0" * 64}) is False
    assert m._verify(f, {}) is True            # no checksum to check → accept
    assert m._verify(tmp_path / "missing.npz", {"sha256": "x"}) is False


def test_verify_rejects_wrong_size(tmp_path):
    f = tmp_path / "x.npz"
    f.write_bytes(b"hello")
    assert m._verify(f, {"bytes": 999}) is False
    assert m._verify(f, {"bytes": 5}) is True


def test_resolve_prefers_existing_verified_file(monkeypatch):
    # A bogus download must never be reached when a valid local copy exists.
    monkeypatch.setattr(m, "_download", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("download should not be called when a valid copy exists")))
    assert m._resolve("disorder_head.npz") is not None


def test_env_override_takes_priority(monkeypatch, tmp_path):
    monkeypatch.setenv("BEER_MODELS_DIR", str(tmp_path))
    paths = m._candidate_paths("disorder_head.npz")
    assert paths[0] == tmp_path / "disorder_head.npz"
