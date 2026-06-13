"""Load pre-trained head weights, downloading them on first use.

The 24 BiLSTM head ``.npz`` files (~490 MB) are **not** bundled in the package or
git. They are hosted as assets on the BEER GitHub Release and fetched on first use
into a per-user cache. ``beer/models/weights_manifest.json`` (committed) lists the
sha256, size, and download URL for each file so downloads are verifiable.

Resolution order for a head file:
  1. ``$BEER_MODELS_DIR`` (override for air-gapped / custom installs)
  2. the package directory itself (developer checkout with weights present)
  3. the per-user cache (``~/.cache/beer/models`` or OS equivalent)
  4. download from the manifest URL into the cache, verifying sha256 + size
"""
from __future__ import annotations
import hashlib
import json
import logging
import os
import pathlib
import threading
import numpy as np

_log = logging.getLogger("beer.models")

_MODELS_DIR = pathlib.Path(__file__).parent
_MANIFEST_PATH = _MODELS_DIR / "weights_manifest.json"
_DOWNLOAD_LOCK = threading.Lock()   # heads load from QThread workers; serialise fetches


def _user_cache_dir() -> pathlib.Path:
    """Per-user cache dir for downloaded weights (XDG / macOS / Windows aware)."""
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or (pathlib.Path.home() / "AppData" / "Local")
    else:
        base = os.environ.get("XDG_CACHE_HOME") or (pathlib.Path.home() / ".cache")
    return pathlib.Path(base) / "beer" / "models"


def _load_manifest() -> dict:
    try:
        return json.loads(_MANIFEST_PATH.read_text(encoding="utf-8")).get("files", {})
    except (OSError, ValueError):
        return {}


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify(path: pathlib.Path, meta: dict) -> bool:
    """True if *path* matches the manifest sha256 (and size, if present)."""
    if not path.exists():
        return False
    if "bytes" in meta and path.stat().st_size != meta["bytes"]:
        return False
    want = meta.get("sha256")
    return True if not want else _sha256(path) == want


def _candidate_paths(name: str) -> list[pathlib.Path]:
    paths = []
    override = os.environ.get("BEER_MODELS_DIR")
    if override:
        paths.append(pathlib.Path(override) / name)
    paths.append(_MODELS_DIR / name)          # dev checkout with weights present
    paths.append(_user_cache_dir() / name)    # previously downloaded
    return paths


def _download(name: str, meta: dict) -> pathlib.Path | None:
    """Download a head into the user cache, verify it, return its path or None."""
    url = meta.get("url")
    if not url:
        return None
    import urllib.request
    cache_dir = _user_cache_dir()
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _log.error("Cannot create weights cache %s: %s", cache_dir, exc)
        return None
    dest = cache_dir / name
    tmp = dest.with_suffix(dest.suffix + ".part")
    mb = meta.get("bytes", 0) / 1048576
    _log.info("Downloading %s (%.0f MB) from %s", name, mb, url)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "BEER/3.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        if not _verify(tmp, meta):
            _log.error("Checksum/size mismatch for downloaded %s — discarding.", name)
            tmp.unlink(missing_ok=True)
            return None
        tmp.replace(dest)
        return dest
    except Exception as exc:  # network / HTTP / disk
        _log.error("Failed to download %s: %s", name, exc)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return None


def _resolve(name: str) -> pathlib.Path | None:
    """Find a verified local copy of *name*, downloading it if necessary."""
    manifest = _load_manifest()
    meta = manifest.get(name, {})
    # 1-3: any existing candidate that passes verification (or has no manifest entry).
    for path in _candidate_paths(name):
        if path.exists() and (_verify(path, meta) if meta else True):
            return path
    # 4: download (serialised — workers may race on first use).
    if not meta:
        return None
    with _DOWNLOAD_LOCK:
        cached = _user_cache_dir() / name
        if _verify(cached, meta):     # another thread finished while we waited
            return cached
        return _download(name, meta)


def _load_head(name: str) -> dict | None:
    """Load a head weight file, fetching it on first use. None if unavailable."""
    path = _resolve(name)
    if path is None:
        return None
    data = np.load(path, allow_pickle=False)
    return dict(data)


def load_disorder_head() -> dict | None:
    return _load_head("disorder_head.npz")

def load_signal_peptide_head() -> dict | None:
    return _load_head("signal_peptide_head.npz")

def load_transmembrane_head() -> dict | None:
    return _load_head("transmembrane_head.npz")

def load_coiled_coil_head() -> dict | None:
    return _load_head("coiled_coil_head.npz")

def load_dna_binding_head() -> dict | None:
    return _load_head("dna_binding_head.npz")

def load_active_site_head() -> dict | None:
    return _load_head("active_site_head.npz")

def load_binding_site_head() -> dict | None:
    return _load_head("binding_site_head.npz")

def load_phosphorylation_head() -> dict | None:
    return _load_head("phosphorylation_head.npz")

def load_lcd_head() -> dict | None:
    return _load_head("lcd_head.npz")

def load_zinc_finger_head() -> dict | None:
    return _load_head("zinc_finger_head.npz")

def load_glycosylation_head() -> dict | None:
    return _load_head("glycosylation_head.npz")

def load_ubiquitination_head() -> dict | None:
    return _load_head("ubiquitination_head.npz")

def load_methylation_head() -> dict | None:
    return _load_head("methylation_head.npz")

def load_acetylation_head() -> dict | None:
    return _load_head("acetylation_head.npz")

def load_lipidation_head() -> dict | None:
    return _load_head("lipidation_head.npz")

def load_disulfide_head() -> dict | None:
    return _load_head("disulfide_head.npz")

def load_intramembrane_head() -> dict | None:
    return _load_head("intramembrane_head.npz")

def load_motif_head() -> dict | None:
    return _load_head("motif_head.npz")

def load_propeptide_head() -> dict | None:
    return _load_head("propeptide_head.npz")

def load_repeat_head() -> dict | None:
    return _load_head("repeat_head.npz")

def load_rna_binding_head() -> dict | None:
    return _load_head("rna_binding_head.npz")

def load_nucleotide_binding_head() -> dict | None:
    return _load_head("nucleotide_binding_head.npz")

def load_transit_peptide_head() -> dict | None:
    return _load_head("transit_peptide_head.npz")

def load_secondary_structure_head() -> dict | None:
    return _load_head("secondary_structure_head.npz")

