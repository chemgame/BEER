"""Persistent application settings stored in ~/.beer/config.json."""
from __future__ import annotations
import json
import os
from pathlib import Path

_CONFIG_DIR  = Path.home() / ".beer"
_CONFIG_FILE = _CONFIG_DIR / "config.json"

_DEFAULTS: dict = {
    "theme_dark":          False,
    "window_size":         9,
    "ph":                  7.0,
    "use_reducing":        False,
    "custom_pka":          None,
    "colormap":            "coolwarm",
    "graph_color":         "Royal Blue",
    "label_font_size":     14,
    "tick_font_size":      12,
    "marker_size":         10,
    "show_bead_labels":    True,
    "transparent_bg":      True,
    "show_heading":        True,
    "show_grid":           True,
    "graph_format":        "PNG",
    "app_font_size":       12,
    "enable_tooltips":     True,
    "colorblind_safe":     False,
    "esm2_model":          "esm2_t33_650M_UR50D",
    "recent_sequences":    [],
}

def load() -> dict:
    """Load settings from disk, falling back to defaults for missing keys."""
    try:
        with open(_CONFIG_FILE) as f:
            stored = json.load(f)
        data = dict(_DEFAULTS)
        data.update(stored)
        return data
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return dict(_DEFAULTS)

def save(settings: dict) -> None:
    """Persist settings to disk."""
    try:
        _CONFIG_DIR.mkdir(exist_ok=True)
        with open(_CONFIG_FILE, "w") as f:
            json.dump(settings, f, indent=2)
    except OSError:
        pass

def get(key: str):
    """Read a single setting value."""
    return load().get(key, _DEFAULTS.get(key))

def set_value(key: str, value) -> None:
    """Update a single setting and persist."""
    cfg = load()
    cfg[key] = value
    save(cfg)
