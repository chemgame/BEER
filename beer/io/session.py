"""BEER session save/load helpers."""
from __future__ import annotations

import json
import pathlib
from typing import Any

SESSION_VERSION = "3.0"


def save_session(data: dict[str, Any], path: str | pathlib.Path) -> None:
    payload = {"beer_session_version": SESSION_VERSION, **data}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_session(path: str | pathlib.Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


__all__ = ["SESSION_VERSION", "save_session", "load_session"]
