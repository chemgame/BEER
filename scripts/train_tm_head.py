#!/usr/bin/env python3
"""Convenience wrapper — trains only the transmembrane head.

Equivalent to:
    python scripts/train_all_heads.py --tasks transmembrane [--model ...] [--no-cluster]

The full implementation lives in train_all_heads.py.
"""
import subprocess, sys, pathlib

args = sys.argv[1:]  # pass through all flags unchanged
subprocess.run(
    [sys.executable, str(pathlib.Path(__file__).parent / "train_all_heads.py"),
     "--tasks", "transmembrane", *args],
    check=True,
)
