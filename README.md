# BEER — Biophysical and Evolutionary Evaluation of Residues

A cross-platform desktop GUI for integrated biophysical analysis of protein sequences.
**BEER** accepts FASTA, PDB, or plain-text input (single or multi-sequence / multi-chain),
fetches sequences from **UniProt** or **RCSB PDB** by accession, and produces comprehensive
biochemical profiles with interactive, publication-quality visualisations.

## Reference

If you use BEER in your research, please cite:

> Mukherjee, S. *arXiv*:2504.20561.
> DOI: [https://doi.org/10.48550/arXiv.2504.20561](https://arxiv.org/abs/2504.20561)

---

## Requirements

| Requirement | Version |
|-------------|---------|
| Python | ≥ 3.10 |
| OS | Windows, macOS, Linux (X11/Wayland) |
| Disk space | ~200 MB (base install) |

---

## Installation

### Recommended (conda environment)

```bash
conda create -n beer python=3.12 -y
conda activate beer
git clone https://github.com/chemgame/BEER.git
cd BEER
pip install .
```

### With ESM2 neural network features

ESM2 improves disorder, aggregation, signal peptide, and PTM predictions using a protein language model. Bundled head weights are included — you only need the runtime:

```bash
# CPU-only (recommended for most users):
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install fair-esm

# Or install everything at once:
pip install ".[esm2]"
```

BEER detects ESM2 automatically at startup. Without it, all analyses still run using classical algorithms — no functionality is lost except the neural augmentation.

### Optional: 3D structure viewer

```bash
pip install PySide6-WebEngine
```

Without it, BEER still works fully — the 3D viewer in the Structure tab is replaced by a message. PDB files can still be saved and opened in PyMOL or ChimeraX.

### From PyPI (once published)

```bash
pip install beer-biophys
```

---

## Launching BEER

```bash
conda activate beer
beer
```

The GUI window opens. No internet connection is required for local analysis. Internet is only needed for UniProt / AlphaFold / Pfam / ELM / DisProt / PhaSepDB fetches.

> **Linux note:** If BEER fails with a Qt platform error, install: `sudo apt-get install libxcb-cursor0`

---

## Quick Start

1. **Paste** an amino-acid sequence into the sequence box.
2. Click **Analyze** or press `Ctrl+Enter`.
3. Browse the 19 report sections in the left panel of the Analysis tab.
4. Switch to the **Graphs** tab and click any graph name.
5. Click **Export PDF** to save the full report.

> **Drag & Drop:** Drag a `.fasta` file directly onto the BEER window to load it instantly.

---

## Input Methods

| Method | How |
|--------|-----|
| **Paste sequence** | Type or paste a bare amino-acid string or FASTA block and click **Analyze** |
| **Import FASTA** | Click **Import FASTA** → select a `.fa` / `.fasta` file (single or multi-sequence) |
| **Import PDB** | Click **Import PDB** → select a `.pdb` file; all chains are extracted |
| **Fetch UniProt** | Enter a UniProt accession (e.g. `P04637`) → click **Fetch** |
| **Fetch PDB ID** | Enter a 4-character RCSB code (e.g. `1UBQ`) → click **Fetch** |

---

## Analysis Modules (19 sections)

| Section | Contents |
|---------|----------|
| **Properties** | MW, pI, GRAVY, aromaticity, extinction coefficient |
| **Composition** | AA counts and frequencies with sort controls |
| **Hydrophobicity** | Kyte-Doolittle statistics; hydrophobic/hydrophilic fractions |
| **Charge** | FCR, NCPR, κ, Ω, net charge, charge asymmetry |
| **Aromatic & π** | Aromatic fraction, cation–π and π–π pair counts |
| **Low Complexity** | Shannon entropy, prion-like score, LC fraction |
| **Disorder** | ESM2 logistic probe (DisProt 2024, AUC 0.831); classical propensity scale fallback |
| **Aggregation** | ESM2 probe (UniProt amyloid, AUC 0.972); ZYGGREGATOR hotspots; CamSol solubility |
| **PTM Sites** | ESM2 probe (UniProt mod_res, AUC 0.927); CK2, PKA, ubiquitination, SUMOylation, glycosylation, methylation |
| **Signal Peptide** | ESM2 probe (UniProt ft_signal, AUC 1.000); n/h/c-region annotation; GPI signal |
| **RNA Binding** | Per-residue propensity; RGG, RRM, KH, SR, DEAD-box, Zinc finger motif hits |
| **Amphipathic Helices** | Detected helices with hydrophobic moment |
| **SCD / κ / Ω** | Sequence charge decoration profile |
| **LARKS** | Low-complexity Aromatic-Rich Kinked Segments (Hughes et al. 2018) |
| **Tandem Repeats** | Direct, tandem, and compositional repeats |
| **TM Topology** | KD sliding-window TM helix prediction; inside-positive topology |
| **Coiled Coil** | Heptad-periodicity score profile |
| **Linear Motifs** | Regex scan: NLS, NES, PxxP, 14-3-3, KFERQ, KDEL, SxIP, NxS/T, … |
| **Comparison** | Side-by-side disorder / hydrophobicity / aggregation overlays |

---

## Graphs (25 total)

Navigate via the **category tree** on the left of the Graphs tab.

| Category | Graphs |
|----------|--------|
| Composition | AA Composition (Bar), AA Composition (Pie) |
| Profiles | Hydrophobicity, Local Charge, Local Complexity, Disorder, Linear Sequence Map, Coiled-Coil |
| Charge & π | Isoelectric Focus, Charge Decoration, Cation–π Map |
| Structure & Folding | Bead Model (Hydrophobicity), Bead Model (Charge), Sticker Map, Helical Wheel, TM Topology |
| Aggregation | β-Aggregation Profile, Solubility Profile, Hydrophobic Moment |
| IDP / Phase Sep | Uversky Phase Plot, Saturation Mutagenesis |
| New Features | PTM Map, RNA-Binding Profile, SCD Profile, pI/MW Map |
| AlphaFold / Structural | pLDDT Profile, Distance Map, Domain Architecture, Ramachandran Plot, Contact Network |

All graphs are rendered at 120 dpi on-screen and 200 dpi on export. Right-click any graph to copy to clipboard or save.

---

## ESM2 Neural Features

BEER uses Meta's ESM2 protein language model (pre-trained linear probe heads bundled in `beer/models/`). No training is required.

| Prediction | AUC (test set) | Test set |
|------------|---------------|----------|
| Disorder | 0.831 | DisProt 2024 (n=324) |
| Aggregation | 0.972 | UniProt amyloid (n=59) |
| Signal peptide | 1.000 | UniProt ft_signal (n=80) |
| PTM sites | 0.927 | UniProt mod_res (n=40) |

All benchmarks use protein-level 80/20 splits (seed=42) to prevent data leakage.

### Model sizes

| Model | Parameters | Speed | Download |
|-------|-----------|-------|----------|
| `esm2_t6_8M_UR50D` *(default)* | 8 M | Fastest | ~30 MB |
| `esm2_t12_35M_UR50D` | 35 M | Fast | ~140 MB |
| `esm2_t30_150M_UR50D` | 150 M | Moderate | ~580 MB |
| `esm2_t33_650M_UR50D` | 650 M | Slow | ~2.6 GB |

Weights are downloaded once on the first `Analyze` call and cached in `~/.cache/torch/hub/`.

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Enter` | Run analysis |
| `Ctrl+E` | Export PDF report |
| `Ctrl+G` | Jump to Graphs tab |
| `Ctrl+S` | Save session |
| `Ctrl+O` | Load session |
| `Ctrl+F` | Focus motif search box |
| `Ctrl+/` | Show all shortcuts |

---

## Package Structure

```
beer/
  analysis/      — computation modules
  utils/         — biophysics primitives
  graphs/        — 25 Matplotlib figure factories
  gui/           — PySide6 application and tabs
  embeddings/    — ESM2Embedder with LRU cache
  models/        — pre-trained .npz head weights
  reports/       — HTML report generation
  io/            — session save/load and PDF export
  network/       — QThread workers
scripts/
  train_heads.py      — retrain ESM2 linear probe heads
  benchmark.py        — full benchmarking pipeline
  case_studies.py     — 10-protein case study figures
  proteome_analysis.py — IDP vs globular proteome figures
manuscript/           — JCIM article (LaTeX)
results/              — benchmark figures and reports
tests/                — pytest test suite (125 tests)
```

---

## License

Released under the GNU General Public License v2. See `LICENSE` for full details.

---

## Author & Contact

Developed by **Saumyak Mukherjee**
Department of Theoretical Biophysics, Max Planck Institute of Biophysics, Frankfurt am Main, Germany
Email: mukherjee.saumyak50@gmail.com
