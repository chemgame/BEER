# BEER Project — Claude Code Context

## What this project is
BEER (**B**iophysical and **E**volutionary **E**valuation of **R**esidues) is a cross-platform
Python desktop application (PySide6 GUI) for integrated biophysical analysis of protein sequences.
It is being developed for publication in **Journal of Chemical Information and Modeling (JCIM)**
as a software article by **Saumyak Mukherjee, MPI Biophysics, Frankfurt**.

## Current state (as of 2026-03-21)
- Full package refactored from monolith `beer.py` into `beer/` package (Phases 1–6 complete)
- ESM2 linear probe heads trained and bundled (`beer/models/*.npz`)
- All benchmarking scripts written and executed
- Full JCIM manuscript drafted in LaTeX (`manuscript/main.tex` + `manuscript/references.bib`)
- Comprehensive GUI improvements implemented (see UI features below)

## GUI features added (2026-03-21)
- **Persistent settings** — `beer/config.py`; all settings saved to `~/.beer/config.json`
- **Drag-and-drop FASTA** — drop `.fasta` files onto the window
- **Progress dialog** — cancellable modal dialog during analysis
- **Export CSV** button on Analysis tab — exports all 17 scalar metrics
- **Comparison profile overlays** — Compare tab now shows disorder/hydrophobicity/aggregation side-by-side
- **Colourblind-safe palette** — Paul Tol palette checkbox in Settings
- **Keyboard shortcut overlay** — `Ctrl+/` shows all shortcuts in a dialog
- **Right-click figure menu** — copy to clipboard or save from any graph
- **Cite BEER button** — copies BibTeX citation to clipboard (Help tab)
- **Methods Generator** — auto-generates a methods paragraph for manuscripts (Help tab)
- **Help text cleaned** — removed Chou-Fasman and LLPS composite score descriptions
- **Persistent history** — recent sequences survive app restart

## Package structure
```
beer/
  analysis/      — computation modules (aggregation, ptm, signal_peptide, scd, rnabinding, etc.)
  utils/         — biophysics primitives (biophysics.py, structure.py, pdb.py, sequence.py)
  graphs/        — 25 Matplotlib figure factories across 8 sub-modules
  gui/           — PySide6 application, tabs, dialogs (main_window.py is the main GUI class)
  embeddings/    — ESM2Embedder with LRU cache + FallbackEmbedder
  models/        — pre-trained .npz head weights (disorder, aggregation, signal, ptm)
  reports/       — HTML report generation (css.py, sections.py)
  io/            — session save/load (.beer JSON) and PDF export
  network/       — QThread workers + pure HTTP helpers
scripts/
  train_heads.py      — downloads data, trains ESM2 linear probes, saves .npz
  benchmark.py        — full benchmarking pipeline with protein-level splits
  case_studies.py     — 10-protein case study figure
  proteome_analysis.py — IDP vs globular proteome comparison figures
manuscript/
  main.tex            — complete JCIM article (achemso template)
  references.bib      — 40 BibTeX entries
results/
  fig1_roc_curves.png       — ROC curves (ESM2 vs classical vs IUPred3)
  fig2_case_studies.png     — 10-protein biophysical profiles
  fig3_proteome_combined.png — 4-panel proteome analysis
  auc_summary.csv           — machine-readable AUC table
  benchmark_report.txt      — full benchmark text report
```

## Key benchmarking results (all real, from held-out test sets)

| Task | Test set | ESM2 AUC | Classical AUC | External tool AUC |
|------|----------|----------|---------------|-------------------|
| Disorder | DisProt 2024 (n=324) | **0.831** | 0.552 (propensity scale) | 0.743 (IUPred3) |
| Aggregation | UniProt amyloid (n=59) | **0.972** | 0.411 (ZYGGREGATOR) | pending |
| Signal peptide | UniProt ft_signal (n=80) | **1.000** | 0.927 (von Heijne) | pending (SignalP 6.0) |
| PTM sites | UniProt mod_res (n=40) | **0.927** | 0.672 (motif scan) | pending (NetPhos 3.1) |

All benchmarks use **protein-level 80/20 splits** (seed=42) to prevent data leakage.
IUPred3 was run via public REST API on all 324 DisProt test proteins (324/324 OK).

## What remains to complete the manuscript

### Author must do (cannot be automated):
1. **GitHub repo** — create public repo, add URL to manuscript (search `[GITHUB URL]` in main.tex)
2. **GUI screenshot** — take a screenshot of the running BEER window, save as `manuscript/figS1_screenshot.png`
3. **Acknowledgements** — add funding source to `\begin{acknowledgement}` block in main.tex
4. **SignalP 6.0** — register at DTU Health Tech, download standalone, run on `results/test_signal_*.fasta`, insert AUC into Table 1 row 3
5. **NetPhos 3.1** — same DTU registration, run on `results/test_ptm.fasta`, insert AUC into Table 1 row 4
6. **AGGRESCAN3D** — optional; requires AlphaFold PDB structures; can mark as "n/a" in Table 1

### Next coding tasks (if continuing):
- Add FASTA export to `benchmark.py` for SignalP/NetPhos input files
- Fix κ/Ω note: `core.py` now returns `"kappa"` key (was missing before — fixed this session)
- Possible: integrate IUPred3 comparison into `fig1_roc_curves.png` as a third curve
- Install LaTeX (`brew install --cask mactex`) and compile: `cd manuscript && pdflatex main && bibtex main && pdflatex main && pdflatex main`

## Metrics removed (scientifically unsound — do not re-add)
These were removed after peer-reviewer-level assessment:
- **Instability Index** (Guruprasad 1990) — unreliable, removed from display
- **LLPS composite score** — arbitrary weights, unvalidated
- **Chou-Fasman secondary structure** (1974, Q3~56%) — superseded
- **RBP composite score/verdict** (0.6/0.4 weights arbitrary)
- **3 PTM types**: Phosphotyrosine EGFR-like (FPR>90%), Lys acetylation KxxK/GKxx, Palmitoylation DHHC (FPR~80%)

## Analysis modules (19 report sections)
Properties, Disorder (ESM2), Charge, Hydrophobicity, IEF/Gel, Composition,
Aggregation (ESM2), Solubility, PTM Sites (ESM2), Signal Peptide (ESM2),
RNA Binding, Amphipathic Helices, SCD/κ/Ω, LARKS, Tandem Repeats,
TM Topology, Coiled Coil, Linear Motifs (ELM), Comparison

## Important conventions
- Always update `USER_MANUAL.md` after any user-visible code change
- ESM2 model: `esm2_t6_8M_UR50D` (default); user-selectable in Settings tab
- Head weights stored as `.npz` in `beer/models/`; retrain with `scripts/train_heads.py`
- Session files use `.beer` JSON format
- GUI toolkit: PySide6 (migrated from PyQt5); signals use `Signal`, slots use `Slot`
- Tests: `pytest tests/` — 125 passing as of last run

## Author details
- Name: Saumyak Mukherjee
- Affiliation: Department of Theoretical Biophysics, Max Planck Institute of Biophysics, Frankfurt am Main, Germany
- Email: mukherjee.saumyak50@gmail.com
- Target journal: Journal of Chemical Information and Modeling (JCIM)
- LaTeX template: achemso, journal=jcisd8, manuscript=article
