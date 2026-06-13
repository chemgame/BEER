"""BEER session save/load helpers."""
from __future__ import annotations

import json
import os
import pathlib
import tempfile
from typing import Any

SESSION_VERSION = "3.0"


def save_session(data: dict[str, Any], path: str | pathlib.Path) -> None:
    """Write session atomically: write to a temp file then rename."""
    path = pathlib.Path(path)
    payload = {"beer_session_version": SESSION_VERSION, **data}
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_session(path: str | pathlib.Path) -> dict[str, Any]:
    """Load a session file, raising informative errors on failure."""
    path = pathlib.Path(path)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Session file '{path.name}' is corrupted or not a valid BEER session: {exc}"
        ) from exc
    stored_ver = data.get("beer_session_version", "unknown")
    if stored_ver != SESSION_VERSION:
        import warnings
        warnings.warn(
            f"Session was saved with BEER version '{stored_ver}'; "
            f"current version is '{SESSION_VERSION}'. "
            "Some data may not restore correctly.",
            UserWarning,
            stacklevel=2,
        )
    # Always fill missing keys so partial writes or hand-edited files don't
    # raise KeyError downstream (defaults are cheap and idempotent).
    _apply_session_defaults(data)
    return data


# Keys introduced in v3.0 that may be missing from older session files.
_V3_DEFAULTS: dict[str, Any] = {
    "analysis_data":     None,
    "alphafold_data":    None,
    "esmfold2_data":     None,
    "uniprot_features":  {},
    "charge_pH":         7.4,
    "roi":               None,
    "seq_name":          "",
    "seq":               "",
}


def _apply_session_defaults(data: dict[str, Any]) -> None:
    """Populate missing v3 keys so old sessions don't crash on key access."""
    for key, default in _V3_DEFAULTS.items():
        data.setdefault(key, default)


__all__ = ["SESSION_VERSION", "save_session", "load_session"]
