#!/usr/bin/env python3
"""Case study analysis of 10 well-characterised proteins using BEER.

Downloads sequences from UniProt, runs full BEER analysis, and generates
a publication-quality multi-panel figure showing that BEER captures known
biophysical properties of each protein class.

Output
------
  results/fig2_case_studies.png

Usage
-----
    python scripts/case_studies.py
"""
from __future__ import annotations
import json
import pathlib
import sys
import urllib.request
import warnings

import numpy as np

ROOT     = pathlib.Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RES_DIR  = ROOT / "results"
sys.path.insert(0, str(ROOT))
DATA_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Protein panel: (UniProt ID, short name, class, expected highlights)
# ---------------------------------------------------------------------------
PROTEINS = [
    ("P35637", "FUS",          "Phase-sep. IDP",    "High κ, Ω, LARKS, RGG"),
    ("Q13148", "TDP-43",       "ALS IDP",           "Disordered C-term, aggregation"),
    ("P09651", "hnRNPA1",      "Phase-sep. IDP",    "Prion-like domain, low-complexity"),
    ("P37840", "α-Synuclein",  "Aggregation",       "Amphipathic helix, amyloid"),
    ("P42858", "Huntingtin",   "Polyglutamine",     "High prion-like score, repeats"),
    ("P04637", "p53",          "Partial order",     "Disordered N/C, ordered core"),
    ("P0CG47", "Ubiquitin",    "Globular (ctrl)",   "Low disorder, low κ"),
    ("P02945", "Bacteriorh.",  "7-TM protein",      "7 TM helices"),
    ("P00533", "EGFR",         "Signal+TM",         "Signal peptide + TM domain"),
    ("P04156", "Prion (PrP)",  "GPI+aggregation",   "GPI anchor, amyloid region"),
]


def _fetch_sequence(uniprot_id: str) -> str:
    cache = DATA_DIR / f"seq_{uniprot_id}.fasta"
    if cache.exists() and cache.stat().st_size > 0:
        lines = cache.read_text().splitlines()
        seq = "".join(l for l in lines if not l.startswith(">"))
        if seq:
            return seq
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    req = urllib.request.Request(url, headers={"User-Agent": "BEER-case/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        text = r.read().decode()
    lines = text.splitlines()
    seq = "".join(l for l in lines if not l.startswith(">"))
    if seq:
        cache.write_text(text)
    return seq


def _run_beer(seq: str) -> dict:
    from beer.analysis.core import AnalysisTools
    return AnalysisTools.analyze_sequence(seq)


def _make_figure(records: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    n = len(records)
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle("BEER Case Studies: Biophysical Profiles of Representative Proteins",
                 fontsize=13, fontweight="bold", y=1.005)

    outer = gridspec.GridSpec(n, 1, figure=fig, hspace=0.55)

    accent   = "#4361ee"
    aggr_col = "#e63946"
    dis_col  = "#f3722c"

    for row_i, rec in enumerate(records):
        uid, name, pclass, note = rec["meta"]
        data = rec["data"]
        seq  = data["seq"]
        L    = len(seq)
        xs   = list(range(1, L + 1))

        inner = gridspec.GridSpecFromSubplotSpec(
            1, 4, subplot_spec=outer[row_i], wspace=0.35)

        # ── panel A: disorder profile ─────────────────────────────────────
        ax0 = fig.add_subplot(inner[0])
        dis = data.get("disorder_scores", [0.5] * L)
        ax0.fill_between(xs, dis, 0.5,
                         where=[v > 0.5 for v in dis],
                         alpha=0.45, color=dis_col, interpolate=True)
        ax0.plot(xs, dis, color=dis_col, lw=1.0)
        ax0.axhline(0.5, color="#aaa", lw=0.7, ls="--")
        ax0.set_ylim(0, 1.05);  ax0.set_xlim(1, L)
        ax0.set_ylabel("Disorder", fontsize=7, color=dis_col)
        ax0.tick_params(labelsize=6)
        ax0.set_facecolor("#fafbff")
        for sp in ["top", "right"]: ax0.spines[sp].set_visible(False)

        # ── panel B: hydrophobicity profile ───────────────────────────────
        ax1 = fig.add_subplot(inner[1])
        hydro = data.get("hydro_profile", [0.0] * L)
        xs_h  = list(range(1, len(hydro) + 1))
        ax1.fill_between(xs_h, hydro, 0,
                         where=[v >= 0 for v in hydro],
                         alpha=0.35, color=accent, interpolate=True)
        ax1.fill_between(xs_h, hydro, 0,
                         where=[v < 0 for v in hydro],
                         alpha=0.35, color="#f72585", interpolate=True)
        ax1.plot(xs_h, hydro, color=accent, lw=0.9)
        ax1.axhline(0, color="#aaa", lw=0.7, ls="--")
        ax1.set_xlim(1, L)
        ax1.set_ylabel("Hydrophobicity", fontsize=7, color=accent)
        ax1.tick_params(labelsize=6)
        ax1.set_facecolor("#fafbff")
        for sp in ["top", "right"]: ax1.spines[sp].set_visible(False)

        # ── panel C: aggregation profile ─────────────────────────────────
        ax2 = fig.add_subplot(inner[2])
        aggr = data.get("aggr_profile_esm2", [0.5] * L)
        ax2.fill_between(xs, aggr, 0.5,
                         where=[v > 0.5 for v in aggr],
                         alpha=0.45, color=aggr_col, interpolate=True)
        ax2.plot(xs, aggr, color=aggr_col, lw=1.0)
        ax2.axhline(0.5, color="#aaa", lw=0.7, ls="--")
        ax2.set_ylim(0, 1.05);  ax2.set_xlim(1, L)
        ax2.set_ylabel("Aggregation", fontsize=7, color=aggr_col)
        ax2.tick_params(labelsize=6)
        ax2.set_facecolor("#fafbff")
        for sp in ["top", "right"]: ax2.spines[sp].set_visible(False)

        # ── panel D: key scalar metrics ───────────────────────────────────
        ax3 = fig.add_subplot(inner[3])
        ax3.axis("off")
        metrics = [
            ("Length",       f"{L} aa"),
            ("GRAVY",        f"{data.get('gravy', 0):.3f}"),
            ("FCR",          f"{data.get('fcr', 0):.3f}"),
            ("NCPR",         f"{data.get('ncpr', 0):+.3f}"),
            ("κ",            f"{data.get('omega', 0):.3f}"),
            ("Ω",            f"{data.get('omega', 0):.3f}"),
            ("SCD",          f"{data.get('scd', 0):.2f}"),
            ("Prion-like",   f"{data.get('prion_score', 0):.3f}"),
            ("LARKS",        f"{len(data.get('larks', []))}"),
            ("TM helices",   f"{len(data.get('tm_helices', []))}"),
        ]
        y0 = 0.97
        for mname, mval in metrics:
            ax3.text(0.0, y0, mname, fontsize=7, color="#555",
                     transform=ax3.transAxes, va="top")
            ax3.text(0.55, y0, mval, fontsize=7, fontweight="600",
                     transform=ax3.transAxes, va="top")
            y0 -= 0.095

        # Row label on the left of the row
        fig.text(0.005, (n - row_i - 0.5) / n,
                 f"{name}\n({pclass})",
                 fontsize=8, fontweight="bold", color="#1a1a2e",
                 va="center", ha="left",
                 transform=fig.transFigure)

    # Column headers above the first row
    for i, title in enumerate(["Disorder Profile",
                                "Hydrophobicity Profile",
                                "Aggregation Profile",
                                "Key Metrics"]):
        fig.text(0.12 + i * 0.215, 1.005, title,
                 fontsize=9, fontweight="bold", ha="center",
                 transform=fig.transFigure)

    out = RES_DIR / "fig2_case_studies.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  Figure saved → {out}")


def main() -> None:
    print("=== BEER Case Studies ===\n")
    records = []
    for uid, name, pclass, note in PROTEINS:
        print(f"  {name} ({uid}) ...", end=" ", flush=True)
        try:
            seq = _fetch_sequence(uid)
            if not seq:
                print("FAILED: empty sequence"); continue
            # Truncate very long sequences for speed (keep first 600 aa)
            if len(seq) > 600:
                seq = seq[:600]
                print(f"[truncated to 600 aa]", end=" ")
            data = _run_beer(seq)
            records.append({"meta": (uid, name, pclass, note), "data": data})
            print(f"OK  L={len(seq)}")
        except Exception as exc:
            print(f"FAILED: {exc}")

    if not records:
        print("No results — aborting."); sys.exit(1)

    print("\nGenerating figure ...")
    _make_figure(records)

    # Print summary table
    print("\n" + "="*70)
    print(f"{'Protein':<14} {'L':>5} {'GRAVY':>7} {'FCR':>6} "
          f"{'NCPR':>7} {'Ω':>6} {'LARKS':>6} {'TM':>4}")
    print("-"*70)
    for rec in records:
        _, name, _, _ = rec["meta"]
        d = rec["data"]
        print(f"{name:<14} {len(d['seq']):>5} {d.get('gravy',0):>7.3f} "
              f"{d.get('fcr',0):>6.3f} {d.get('ncpr',0):>+7.3f} "
              f"{d.get('omega',0):>6.3f} {len(d.get('larks',[])):>6} "
              f"{len(d.get('tm_helices',[])):>4}")
    print("="*70)


if __name__ == "__main__":
    main()
