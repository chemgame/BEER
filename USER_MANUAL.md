# BEER User Manual
**Biochemical Estimator & Explorer of Residues — v3.0**

---

## Table of Contents

1. [Installation](#1-installation)
2. [Launching BEER](#2-launching-beer)
3. [Quick Start](#3-quick-start)
4. [Input Methods](#4-input-methods)
5. [Analysis Tab](#5-analysis-tab)
6. [Graphs Tab](#6-graphs-tab)
7. [Structure Tab](#7-structure-tab)
8. [BLAST Tab](#8-blast-tab)
9. [Multichain Analysis Tab](#9-multichain-analysis-tab)
10. [Compare Tab](#10-compare-tab)
11. [Truncation Series Tab](#11-truncation-series-tab)
12. [MSA Tab](#12-msa-tab)
13. [Complex Mass Tab](#13-complex-mass-tab)
14. [Settings Tab](#14-settings-tab)
15. [Help Tab](#15-help-tab)
16. [Keyboard Shortcuts](#16-keyboard-shortcuts)
17. [Session Save & Load](#17-session-save--load)
18. [Export PDF Report](#18-export-pdf-report)
19. [ESM2 Embeddings](#19-esm2-embeddings)
20. [Metrics Reference](#20-metrics-reference)

---

## 1. Installation

### Requirements

- Python ≥ 3.10
- A desktop environment (macOS, Windows, Linux with X11/Wayland)
- ~200 MB disk space for the base install

### From GitHub (recommended)

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/BEER.git
cd BEER

# 2. Create a dedicated conda environment
conda create -n beer python=3.12 -y
conda activate beer

# 3. Install in editable mode (includes all dependencies)
pip install -e .

# 4. Verify
beer --help   # should print usage or open the GUI
```

> **Note:** The `beer` command launches the GUI. If it fails with a Qt platform error on Linux, install `libxcb-cursor0`:
> ```bash
> sudo apt-get install libxcb-cursor0
> ```

### From PyPI (once published)

```bash
pip install beer-biophys
```

### Optional: 3D structure viewer

The Structure tab uses an embedded web view. Install the WebEngine addon for it to work:

```bash
pip install PySide6-WebEngine
```

Without it, BEER still works fully — only the 3D viewer is replaced by a message. You can still save PDB files and open them in PyMOL or ChimeraX.

### Optional: ESM2 neural embeddings

ESM2 improves four predictions (disorder, aggregation, signal peptide, PTM) using a protein language model. Bundled head weights are already included in `beer/models/`. You only need to install the runtime:

```bash
pip install fair-esm torch
# For CPU-only (smaller download, no GPU required):
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install fair-esm
```

> BEER detects ESM2 automatically at startup. If not installed, all analyses still run using classical algorithms — no functionality is lost except the neural augmentation.

> The first `Analyze` call downloads ESM2 model weights (~30 MB for the default 8M model) and caches them in `~/.cache/torch/hub/`. Subsequent calls are instant.

---

## 2. Launching BEER

```bash
conda activate beer
beer
```

The GUI window opens. No internet connection is required for local analysis. Internet is only needed for UniProt/AlphaFold/Pfam/ELM/DisProt/PhaSepDB fetches.

---

## 3. Quick Start

1. **Paste** a protein sequence (single-letter uppercase code) into the sequence box.
2. Click **Analyze** or press `Ctrl+Enter`.
3. Browse the 19 report sections in the left panel of the Analysis tab.
4. Switch to the **Graphs** tab and click any graph name in the tree on the left.
5. Click **Export PDF** to save the full report.

---

## 4. Input Methods

| Method | How |
|--------|-----|
| **Paste sequence** | Type or paste a bare amino-acid string (`MAEGEITT…`) or a FASTA block into the sequence box and click **Analyze**. |
| **Import FASTA** | Click **Import FASTA** → select a `.fa` / `.fasta` file. Multi-sequence files load all chains into the Multichain tab and set the first as the active sequence. |
| **Import PDB** | Click **Import PDB** → select a `.pdb` file. All chains are extracted and available in the **Chain** dropdown. Structure graphs (Ramachandran, distance map, pLDDT) become available immediately. |
| **Fetch UniProt** | Type a UniProt accession (e.g. `P04637`) in the **Fetch** box → click **Fetch**. Sequence is downloaded, loaded, and analysed automatically. Unlocks **Fetch AlphaFold**, **Fetch Pfam**, **DisProt**, and **PhaSepDB** buttons. |
| **Fetch PDB ID** | Type a 4-character RCSB PDB code (e.g. `1UBQ`) in the Fetch box → click **Fetch**. All chains and the coordinate file are downloaded; structural graphs appear immediately. |

---

## 5. Analysis Tab

The main results tab. After running analysis the left panel shows 19 report sections; click any section name to display it on the right.

### Toolbar buttons

| Button | Action |
|--------|--------|
| **Import FASTA** | Load sequence(s) from a FASTA file |
| **Import PDB** | Load sequence(s) from a PDB file |
| **Analyze** | Run analysis on the current sequence (`Ctrl+Enter`) |
| **Export PDF** | Generate and save the full HTML/PDF report |
| **Export CSV** | Export all scalar metrics for the current sequence to a CSV file |
| **Mutate…** | Open the point-mutation dialog — pick position and new amino acid |
| **Save Session** | Save the current state to a `.beer` JSON file |
| **Load Session** | Restore a previously saved session |
| **Figure Composer** | Assemble a custom multi-panel publication figure |
| **Fetch** | Download sequence from UniProt or RCSB PDB |
| **Fetch AlphaFold** | Download predicted structure from EBI (requires UniProt accession) |
| **Fetch Pfam** | Download domain annotations from InterPro (requires UniProt accession) |
| **Fetch ELM** | Fetch experimentally validated linear motifs from ELM database |
| **DisProt** | Fetch disorder annotations from DisProt |
| **PhaSepDB** | Check whether the protein is in the phase-separation database |

> **Drag & Drop:** You can drag a FASTA file directly onto the BEER window to load it without using the Import button.

### Sequence viewer

Displays the sequence with colour-coded residues (UniProt style):

| Colour | Residue class |
|--------|--------------|
| Orange | Hydrophobic (A C I L M V) |
| Dark amber | Aromatic (F W Y) |
| Blue | Positive (K R H) |
| Red | Negative (D E) |
| Teal | Polar uncharged (N Q S T) |
| Purple | Special (G P) |

Use the **Search** box and **Highlight** button to find and colour-highlight any motif or regex pattern in the sequence.

### Report sections

| Section | Contents |
|---------|----------|
| **Composition** | Amino acid counts and percentage frequencies with sort controls (A–Z, by frequency, hydrophobicity ↑/↓) |
| **Properties** | MW, pI, GRAVY, aromaticity, extinction coefficient |
| **Hydrophobicity** | Kyte-Doolittle profile statistics, hydrophobic/hydrophilic fractions |
| **Charge** | FCR, NCPR, κ, net charge at default pH, charge asymmetry |
| **Aromatic & π** | Aromatic fraction, cation–π pairs, π–π pairs |
| **Low Complexity** | Shannon entropy, prion-like score, LC fraction, repeat motif counts |
| **Disorder** | Disorder/order-promoting fractions, aliphatic index, Ω, mean per-residue disorder score and disordered fraction |
| **Repeat Motifs** | RGG, FG, SR/RS, QN/NQ counts and positions |
| **Sticker & Spacer** | Sticker residue mapping, mean/min/max spacer lengths |
| **TM Helices** | Predicted transmembrane helices with topology (inside-positive rule) |
| **LARKS** | Low-complexity Aromatic-Rich Kinked Segments: position, sequence, aromatic count, LC fraction, entropy |
| **Linear Motifs** | Regex hits for 15 functional SLiM patterns (NLS, NES, PxxP, …) |
| **β-Aggregation & Solubility** | ZYGGREGATOR hotspots, CamSol intrinsic solubility |
| **PTM Sites** | Phosphorylation (CK2, PKA), ubiquitination (ΨKxE), SUMOylation (ΨKxE), N-linked glycosylation (NxS/T), N-terminal acetylation, arginine methylation |
| **Signal Peptide & GPI** | Signal peptide prediction (n/h/c-region), GPI anchor signal |
| **Amphipathic Helices** | Detected amphipathic helices with hydrophobic moment |
| **Charge Decoration (SCD)** | Sequence charge decoration score and profile |
| **RNA Binding** | Mean per-residue RBP propensity, K/R/Y/F/W composition, RNA-binding motif hits (RGG, RRM, KH, SR, Zinc finger, DEAD-box) |
| **Tandem Repeats** | Direct, tandem, and compositional repeats |

---

## 6. Graphs Tab

Navigate using the **category tree** on the left. Click any graph name to display it. Use the matplotlib toolbar (zoom, pan, home) above each figure.

**Save Graph** — saves the current figure in the format chosen in Settings (PNG / SVG / PDF).
**Save All Graphs** — exports every generated graph to a chosen directory.

### Graph categories

#### Composition
| Graph | Description |
|-------|-------------|
| Amino Acid Composition (Bar) | Count/frequency bar chart; sort order matches the report buttons |
| Amino Acid Composition (Pie) | Pie chart of residue type groups |

#### Profiles
| Graph | Description |
|-------|-------------|
| Hydrophobicity Profile | Kyte-Doolittle sliding-window average |
| Local Charge Profile | Sliding-window NCPR |
| Local Complexity | Sliding-window Shannon entropy; dashed line = 2.0 bit threshold |
| Disorder Profile | Per-residue disorder score 0–1; orange fill = disordered (>0.5) |
| Linear Sequence Map | Three-track overview: hydrophobicity, NCPR, disorder |
| Coiled-Coil Profile | Heptad-periodicity score; fill above 0.50 = predicted coiled coil |

#### Charge & π-Interactions
| Graph | Description |
|-------|-------------|
| Isoelectric Focus | Henderson-Hasselbalch charge curve 0–14; pI annotated with arrow (2 decimal places), pH 7.4 physiological charge marked |
| Charge Decoration | Das-Pappu FCR vs \|NCPR\| phase diagram |
| Cation–π Map | Proximity heat map for K/R ↔ F/W/Y pairs within ±8 residues |

#### Structure & Folding
| Graph | Description |
|-------|-------------|
| Bead Model (Hydrophobicity) | Per-residue KD score; colourmap selectable in Settings |
| Bead Model (Charge) | K/R blue, D/E red, H cyan, neutral grey |
| Sticker Map | Aromatic (amber), basic (blue), acidic (pink), spacer (grey) |
| Helical Wheel | Cartesian projection of first 18 residues at 100°/step |
| TM Topology | Snake-plot of predicted TM helices with inside-outside annotation |

#### Phase Separation / IDP
| Graph | Description |
|-------|-------------|
| Uversky Phase Plot | Mean \|charge\| vs normalised hydrophobicity; Uversky boundary line |
| Saturation Mutagenesis | 20×n heatmap of \|ΔGRAVY\| + \|ΔNCPR\| for all single substitutions |

#### Aggregation & Solubility
| Graph | Description |
|-------|-------------|
| β-Aggregation Profile | Per-residue aggregation propensity; hotspots shaded red |
| Solubility Profile | CamSol intrinsic score; smoothed 7-residue window |
| Hydrophobic Moment | μH per window; amphipathic helices annotated |

#### New Features
| Graph | Description |
|-------|-------------|
| PTM Map | Per-residue PTM probability heatmap |
| RNA-Binding Profile | Per-window RBP score |
| SCD Profile | Sequence charge decoration per window |
| pI / MW Map | Protein vs background gel-like pI–MW scatter |
| Truncation Series | MW, GRAVY, pI across progressive N/C truncations (run from Truncation tab) |
| MSA Conservation | Per-column conservation from a loaded MSA (run from MSA tab) |
| Complex Mass | Bar chart of complex subunit masses (run from Complex tab) |

#### AlphaFold / Structural *(requires a loaded structure)*
| Graph | Description |
|-------|-------------|
| pLDDT Profile | Per-residue AlphaFold confidence 0–100 with four colour bands |
| Distance Map | Cα pairwise distances; pink contour = 8 Å contact threshold |
| Domain Architecture | Multi-track: Pfam domains, disorder, LC regions, TM helices |
| Ramachandran Plot | φ/ψ dihedral angles coloured by secondary structure |
| Residue Contact Network | Graph of residues within 8 Å |

---

## 7. Structure Tab

Displays an interactive 3D structure viewer powered by [3Dmol.js](https://3dmol.csb.pitt.edu).

**Requires:** `pip install PySide6-WebEngine`
If not installed, the tab shows a message; PDB files can still be saved locally and opened in PyMOL or ChimeraX.

### Colour modes
| Button | Meaning |
|--------|---------|
| Color: pLDDT | Red (low confidence) → white → blue (high confidence) |
| Color: Residue Type | Standard amino-acid colour scheme |
| Color: Chain | Each chain a different colour |
| Cartoon / Sphere | Toggle between cartoon and sphere representation |

**Save PDB** — saves the currently loaded structure as a `.pdb` file.

---

## 8. BLAST Tab

Submits the current sequence to NCBI blastp via Biopython. Requires an internet connection; typically takes 1–3 minutes.

| Control | Description |
|---------|-------------|
| Database | nr, swissprot, pdb, refseq_protein |
| Max hits | Number of alignments to retrieve (5–100) |
| BLAST Current Sequence | Submit the last analysed sequence |

Results table columns: Accession · Description · Length · Score · E-value · % Identity · **Load** (loads the hit sequence and immediately re-runs analysis).

---

## 9. Multichain Analysis Tab

Populated automatically when a multi-FASTA file or multi-chain PDB is imported, or when a PDB ID is fetched. Columns: ID · Length · MW · Net Charge · % Hydrophobic · % Hydrophilic · % +Charged · % −Charged · % Neutral.

**Double-click** any row to load that chain into the Analysis tab.
**Export CSV / Export JSON** — save the full table.

---

## 10. Compare Tab

Paste two sequences (or FASTA entries) side by side and click **Compare Sequences** to get a side-by-side property table: length, MW, pI, GRAVY, FCR, NCPR, net charge, aromaticity, extinction coefficient. Overlay profile graphs (disorder, hydrophobicity, aggregation) are automatically generated in the Graphs tab after comparison.

---

## 11. Truncation Series Tab

With a sequence already analysed, set the truncation step (%) and choose N-terminal, C-terminal, or both, then click **Run Truncation Series**. BEER computes properties at each truncation length and generates the **Truncation Series** graph in the Graphs tab.

---

## 12. MSA Tab

Paste a multi-FASTA block (pre-aligned or unaligned). Click **Run MSA**. BEER performs a simple progressive pairwise alignment if not already aligned, then generates the **MSA Conservation** graph showing per-column sequence conservation.

---

## 13. Complex Mass Tab

Paste chain sequences in FASTA format (one per chain, header = chain ID). Enter stoichiometry as a string like `A2B1` (2 copies of chain A, 1 of chain B). Click **Calculate Complex** to get per-chain and total MW, extinction coefficients, and the **Complex Mass** bar chart.

---

## 14. Settings Tab

### Analysis Parameters
| Setting | Default | Description |
|---------|---------|-------------|
| Default pH | 7.0 | pH for net-charge and charge-curve calculations |
| Sliding Window Size | 9 | Window width for hydrophobicity, NCPR, entropy profiles |
| Override pKa | — | Nine comma-separated values: N-term, C-term, D, E, C, Y, H, K, R |
| Reducing conditions | Off | If on, Cys is not counted in disulphide pairs for extinction coefficient |

### Graph Appearance
| Setting | Description |
|---------|-------------|
| Label / Tick Font Size | Point size of axis titles and tick labels |
| Marker Size | Size of data markers in scatter/line graphs |
| Default Graph Format | PNG / SVG / PDF when saving graphs |
| Bead Colormap | Colour map for the bead hydrophobicity model |
| Graph Accent Colour | Main accent colour for bars, lines, and fills |
| Show Graph Titles | Toggle axis titles on/off |
| Show Grid | Toggle grid lines on/off |
| Transparent background | PNG/SVG exports use a transparent background |

### Interface
| Setting | Description |
|---------|-------------|
| UI Font Size | Global application font size in points |
| Dark Theme | Toggle light/dark colour scheme |
| Enable Tooltips | Show/hide hover tooltips |
| Colourblind-safe palette | Apply Paul Tol colourblind-safe accent colours to all graphs |

### ESM2 Embeddings
| Setting | Description |
|---------|-------------|
| ESM2 model | Choose model size: 8M (fast, default), 35M, 150M, 650M (most accurate) |

Requires `fair-esm` and `torch` to be installed. Changing the model and clicking **Apply Settings** reinitialises the embedder and clears the embedding cache.

Click **Apply Settings** to apply all changes and automatically save them to `~/.beer/config.json`. Settings persist across restarts. **Reset to Defaults** restores all values to factory defaults.

---

## 15. Help Tab

Built-in reference covering: Getting Started · Sequence Analysis · TM Helices · AlphaFold & 3D Structure · Pfam Domains · BLAST · Graphs Reference · Phase Separation · Linear Motifs · Multichain & Compare · Settings & Session.

Two additional buttons appear at the bottom of the Help tab:

- **Copy Citation (BibTeX)** — copies the BibTeX entry for BEER to the clipboard for use in papers.
- **Generate Methods Paragraph** — auto-generates a methods paragraph based on the current sequence and settings, ready to paste into your manuscript.

---

## 16. Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Enter` | Run analysis |
| `Ctrl+E` | Export PDF report |
| `Ctrl+G` | Jump to Graphs tab |
| `Ctrl+S` | Save session |
| `Ctrl+O` | Load session |
| `Ctrl+F` | Focus motif search box |
| `Ctrl+/` | Show keyboard shortcut reference overlay |

> **Tip:** Press `Ctrl+/` at any time to bring up a quick-reference overlay of all keyboard shortcuts.

---

## 17. Session Save & Load

Sessions are saved as `.beer` JSON files that capture:
- The current sequence and sequence name
- All settings (pH, window size, colours, etc.)

**Save Session** → choose a file path → saves immediately.
**Load Session** → choose a `.beer` file → restores state and re-runs analysis.

---

## 18. Export PDF Report

Click **Export PDF** (or `Ctrl+E`). BEER generates a self-contained HTML report with:
- All 19 report sections
- All graphs embedded as base64 PNG images
- Consistent BEER styling

The file is saved as `.html` (openable in any browser and printable to PDF from there) or `.pdf` if a Qt print driver is available.

---

## 19. ESM2 Embeddings

BEER uses Meta's ESM2 protein language model to augment four predictions. **Pre-trained head weights are bundled** in `beer/models/` — no training step is required for end users.

### What ESM2 improves

| Prediction | Without ESM2 | With ESM2 |
|------------|-------------|-----------|
| **Disorder profile** | IUPred-inspired sliding-window propensity scale | Logistic regression probe on ESM2 per-residue embeddings, trained on DisProt 2024 (AUC = 0.87) |
| **Aggregation profile** | ZYGGREGATOR propensity scale | 40% ESM2 logistic probe + 60% ZYGGREGATOR blend (AUC = 0.97) |
| **Signal peptide** | Hydrophobicity heuristic | ESM2 context-aware probe trained on UniProt signal annotations (AUC = 1.00) |
| **PTM sites** | Consensus motif scanning | ESM2 per-position probe trained on UniProt modified-residue annotations (AUC = 0.92) |

### Installation

```bash
# CPU-only (recommended for most users):
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install fair-esm scikit-learn

# With GPU support (faster for 150M / 650M models):
pip install torch fair-esm scikit-learn
```

### Usage

1. Install the packages above, then restart BEER — it detects ESM2 automatically.
2. ESM2 augmentation is **active by default** using the 8M model. No settings change is needed.
3. To use a larger, more accurate model: go to **Settings → ESM2 model** → select a larger model → click **Apply Settings**.
4. Re-run analysis — ESM2-augmented predictions are used automatically.

### Model sizes

| Model | Parameters | Embedding dim | Speed | Download size |
|-------|-----------|--------------|-------|---------------|
| `esm2_t6_8M_UR50D` | 8 M | 320 | Fastest | ~30 MB |
| `esm2_t12_35M_UR50D` | 35 M | 480 | Fast | ~140 MB |
| `esm2_t30_150M_UR50D` | 150 M | 640 | Moderate | ~580 MB |
| `esm2_t33_650M_UR50D` | 650 M | 1280 | Slow (GPU recommended) | ~2.6 GB |

Model weights are downloaded once on the first `Analyze` call and cached in `~/.cache/torch/hub/`. Up to 32 sequences are cached in memory (LRU) per session.

### Retraining heads on newer data

The bundled heads were trained with `esm2_t6_8M_UR50D`. To retrain on updated DisProt/UniProt data or a different model:

```bash
conda activate beer
pip install scikit-learn  # if not already installed
python scripts/train_heads.py --model esm2_t6_8M_UR50D --max-seqs 300
# Or train only specific heads:
python scripts/train_heads.py --heads disorder aggregation
```

This downloads training data to `data/` (cached for future runs), trains all four heads, and overwrites the `.npz` files in `beer/models/`. Retraining takes ~10–30 minutes on a CPU depending on model size and sequence count.

> **Note:** The `data/` directory is listed in `.gitignore` — training data is not committed to the repository. Only the resulting `.npz` weight files are tracked.

---

## 20. Metrics Reference

### Sequence properties

| Metric | Formula / Definition |
|--------|---------------------|
| **Molecular Weight** | Sum of residue masses + water (Da) |
| **pI** | pH where net charge = 0 (Henderson-Hasselbalch) |
| **GRAVY** | Mean Kyte-Doolittle hydropathicity |
| **Aromaticity** | (F + W + Y) / length |
| **Extinction coefficient** | W×5500 + Y×1490 + (C–C)×125 at 280 nm |

### Charge metrics

| Metric | Definition |
|--------|-----------|
| **FCR** | Fraction of charged residues = (K + R + D + E) / length |
| **NCPR** | Net charge per residue = (pos − neg) / length |
| **κ (kappa)** | Charge patterning: 0 = well-mixed, 1 = fully segregated (Das & Pappu 2013) |
| **Ω (omega)** | Sticker patterning (same scale as κ) |

### Phase separation / IDP

| Metric | Definition |
|--------|-----------|
| **LARKS** | Low-complexity Aromatic-Rich Kinked Segments: 7-residue windows with ≥1 aromatic (F/W/Y), ≥50% LC residues (G/A/S/T/N/Q), Shannon entropy < 1.8 bits (Hughes et al. 2018 eLife) |
| **SCD** | Sequence charge decoration — pairwise charge product weighted by sequence separation (Sawle & Ghosh 2015) |
| **Prion-like score** | Fraction of N, Q, S, G, Y residues; used as a compositional screen for prion-like domains |

### Aggregation & solubility

| Metric | Source |
|--------|--------|
| **ZYGGREGATOR** | β-aggregation propensity per residue (Tartaglia & Vendruscolo 2008) |
| **PASTA energy** | Amyloid pairing energy (Trovato et al. 2007) |
| **CamSol** | Intrinsic solubility scale (Sormanni et al. 2015) |

### Structure predictions

| Prediction | Method |
|------------|--------|
| **TM helices** | KD sliding window ≥1.6, segments 17–25 aa, inside-positive topology rule |
| **Coiled coil** | 28-residue heptad periodicity score ≥0.50 (Lupas 1991) |
| **Disorder profile** | ESM2 logistic probe (DisProt 2024, AUC 0.87) when available; IUPred-inspired propensity scale otherwise |

### PTM predictions

Only validated motif rules are included. Low-confidence or highly nonspecific rules have been excluded:

| PTM type | Method | Confidence |
|----------|--------|-----------|
| N-linked glycosylation | N[^P][ST] sequon | High |
| O-linked glycosylation | ≥3 S/T in 5-aa window | Medium |
| Phosphoserine/Thr (CK2) | [ST]xx[DE] | Medium |
| Phosphoserine/Thr (PKA) | R[^P][^P][ST] | Medium |
| Ubiquitination (ΨKxE) | [LVIMF]K.[DE] | Medium |
| SUMOylation (ΨKxE) | [VILMF]K.E | Medium |
| N-terminal acetylation (NatA) | Met removal + small aa, or N-term S/A/T/G/C | Medium |
| Arginine methylation | RGG, RG, GR motifs | Medium |

> Phosphotyrosine (EGFR-like), lysine acetylation (KxxK/GKxx), and palmitoylation (DHHC) predictions were removed — these rules have false positive rates >80% and are not suitable for general sequence screening.

### RNA binding

No composite score is reported. Instead:
- **Mean per-residue propensity** (Jeong et al. 2012)
- **K/R/Y/F/W fraction** (residues known to make backbone and stacking contacts with RNA)
- **Motif hits**: RGG box, RRM RNP1, KH GXXG, SR dipeptides, YGG/GGY, DEAD-box, Zinc finger CCHH

A numeric RBP score would require validation against experimental RBP-binding data, which is not available for the current scale.

---

*BEER v3.0 — Mukherjee Lab*
