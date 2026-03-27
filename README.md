# BEER — Biophysical and Evolutionary Evaluation of Residues

**BEER** is a desktop application for integrated biophysical analysis of protein sequences. It accepts a sequence (pasted, imported as FASTA/PDB, or fetched from UniProt/RCSB), runs 19 analysis modules in one click, and gives you interactive publication-quality graphs, a 3D structure viewer, and exportable reports — all from a single GUI.

I built BEER because I wanted a single tool that handles everything from basic physicochemical properties to disorder prediction, aggregation hotspots, PTM sites, RNA-binding propensity, and phase separation metrics, without having to jump between half a dozen web servers.

> **If you use BEER in your research, please cite:**
>
> Mukherjee, S. *arXiv*:2504.20561. DOI: [https://doi.org/10.48550/arXiv.2504.20561](https://arxiv.org/abs/2504.20561)

---

## What's new in v2.0

Version 1.0 was a single monolithic script (`beer.py`) with a basic PySide6 GUI. v2.0 is a full rewrite:

- **Restructured as a proper Python package** (`beer/`) — modular, importable, installable via `pip`
- **ESM2 neural network predictions** for disorder, aggregation, signal peptide, and PTM sites — pre-trained heads bundled, no training needed
- **VMD-style 3D structure viewer** with color schemes, color bar, coordinate axes, spin, snapshot, and background color controls
- **25 graphs** across 8 categories (up from ~12 in v1.0), including Ramachandran, contact network, pLDDT profile, and domain architecture
- **New analysis modules**: RNA binding, SCD/κ/Ω, LARKS, tandem repeats, TM topology, coiled coil, ELM linear motifs
- **New utility tabs**: BLAST, Multichain, Compare, Truncation Series, MSA Conservation, Complex Mass
- **Persistent settings**, drag-and-drop FASTA, progress dialog, session save/load, colourblind-safe palette, keyboard shortcuts overlay, right-click figure menu
- **Copy Sequence** and **Clear All** buttons in the Analysis tab
- **Structure export** in PDB, mmCIF, GRO, XYZ, and FASTA formats
- Removed scientifically unreliable metrics (Instability Index, LLPS composite score, Chou-Fasman, unvalidated PTM rules)

---

## Requirements

| | |
|---|---|
| Python | ≥ 3.10 |
| OS | macOS, Windows, Linux (X11 or Wayland) |
| Disk space | ~200 MB for base install |

---

## Installation

I recommend using a dedicated conda environment to keep things clean:

```bash
conda create -n beer python=3.12 -y
conda activate beer
git clone https://github.com/chemgame/BEER.git
cd BEER
pip install .
```

That's it. All required dependencies (PySide6, matplotlib, BioPython, numpy, mplcursors) are pulled in automatically.

> **Note for Linux users:** If BEER fails on startup with a Qt platform plugin error, you may need a system library:
> ```bash
> sudo apt-get install libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-xinerama0
> ```
> On Fedora/RHEL: `sudo dnf install xcb-util-cursor xcb-util-image xcb-util-keysyms xcb-util-renderutil`

### 3D structure viewer

The 3D viewer in the Structure tab uses Qt's WebEngine component, which **is already bundled inside PySide6** — no extra install is needed. It should work out of the box after `pip install .`.

If the viewer shows a blank page on Linux, your system may be missing the chromium sandbox libraries:
```bash
sudo apt-get install libnss3 libatk-bridge2.0-0 libdrm2 libxcomposite1 libxdamage1 libxrandr2 libgbm1 libxkbcommon0
```

### Optional: ESM2 neural predictions

ESM2 improves four predictions (disorder, aggregation, signal peptide, PTM). The pre-trained heads are bundled — you only need the torch runtime. **Important:** use `numpy<2` (already enforced by `pip install .`) and install the CPU torch wheel directly to avoid version conflicts:

```bash
# CPU-only (recommended — works everywhere, no GPU required):
pip install "torch>=2.0" --index-url https://download.pytorch.org/whl/cpu
pip install fair-esm scipy

# Or install everything at once with the esm2 extra:
pip install ".[esm2]" --extra-index-url https://download.pytorch.org/whl/cpu
```

> If you see a warning about NumPy 1.x/2.x compatibility when importing torch, run:
> `pip install "numpy>=1.24,<2"` — this is already constrained in BEER's requirements but if you have a pre-existing numpy 2.x in your environment, you'll need to downgrade it.

BEER auto-detects ESM2 at startup. If it's not installed, all 19 analyses still run using classical algorithms.

### From PyPI

```bash
pip install beer-biophys
```

---

## Launching BEER

```bash
conda activate beer
beer
```

> **Linux:** If you get a Qt platform error, run `sudo apt-get install libxcb-cursor0` first.

Internet is only needed for external fetches (UniProt, AlphaFold, Pfam, ELM, DisProt, PhaSepDB, BLAST). All local analysis runs fully offline.

---

## Quick Start

1. Paste an amino-acid sequence into the sequence box (or drag-and-drop a `.fasta` file onto the window)
2. Click **Analyze** or press `Ctrl+Enter`
3. Browse the 19 report sections in the left panel of the Analysis tab
4. Go to the **Graphs** tab and click any graph name in the tree on the left
5. Click **Export Analysis** to save the full report (CSV, JSON, PDF, or DAT)

---

## Input Methods

| Method | How |
|--------|-----|
| **Paste sequence** | Type or paste a bare amino-acid string or FASTA block and click **Analyze** |
| **Import FASTA** | Click **Import FASTA** → select a `.fa` / `.fasta` file — multi-sequence files load all chains into the Multichain tab |
| **Import PDB** | Click **Import PDB** → select a `.pdb` file — all chains are extracted and available in the Chain dropdown |
| **Fetch UniProt** | Type a UniProt accession (e.g. `P04637`) → click **Fetch** — downloads sequence and unlocks AlphaFold, Pfam, DisProt, PhaSepDB buttons |
| **Fetch PDB ID** | Type a 4-character RCSB code (e.g. `1UBQ`) → click **Fetch** — downloads sequence and coordinates |
| **Drag & Drop** | Drag a `.fasta` file directly onto the BEER window |

---

## Analysis Tab

After running analysis, the left panel lists 19 report sections. Click any section name to display it.

### Toolbar

| Button | Action |
|--------|--------|
| **Analyze** | Run analysis on the current sequence (`Ctrl+Enter`) |
| **Export Analysis** | Save the report — opens a format dialog (CSV / JSON / PDF / DAT) (`Ctrl+E`) |
| **Mutate…** | Point-mutation dialog — pick position and replacement amino acid |
| **Save / Load Session** | Save or restore a `.beer` JSON session file |
| **Figure Composer** | Assemble a custom multi-panel publication figure |
| **Fetch** | Download sequence from UniProt or RCSB PDB |
| **Fetch AlphaFold** | Download predicted structure from EBI AlphaFold (needs UniProt accession) |
| **Fetch Pfam** | Domain annotations from InterPro |
| **Fetch ELM** | Experimentally validated linear motifs |
| **DisProt / PhaSepDB** | Disorder and phase-separation database annotations |

### Sequence viewer

Residues are colour-coded by type (hydrophobic, aromatic, positive, negative, polar, special). Use the **Search** box and **Highlight** button to find and highlight any motif or regex pattern.

Below the viewer:

| Button | Action |
|--------|--------|
| **Copy Sequence** | *Copy whole sequence* or *Copy range…* (start/end residue numbers) — result goes to clipboard |
| **Clear All** | Clears the loaded protein, all analysis, all graphs, and the structure viewer |

### Report sections (19 total)

| Section | Contents |
|---------|----------|
| **Properties** | MW, pI, GRAVY, aromaticity, extinction coefficient |
| **Composition** | AA counts and frequencies, sortable by name / frequency / hydrophobicity |
| **Hydrophobicity** | Kyte-Doolittle statistics, hydrophobic and hydrophilic fractions |
| **Charge** | FCR, NCPR, κ, Ω, net charge, charge asymmetry |
| **Aromatic & π** | Aromatic fraction, cation–π and π–π pair counts |
| **Low Complexity** | Shannon entropy, prion-like score, LC fraction |
| **Disorder** | ESM2 logistic probe (DisProt 2024, AUC 0.83); classical propensity fallback |
| **Aggregation** | ESM2 probe (AUC 0.97); ZYGGREGATOR hotspots; CamSol solubility |
| **PTM Sites** | ESM2 probe (AUC 0.93); CK2, PKA, ubiquitination, SUMOylation, glycosylation, methylation |
| **Signal Peptide** | ESM2 probe (AUC 1.00); n/h/c-region annotation; GPI signal |
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

## Graphs Tab

Navigate using the **category tree** on the left — click any graph name to display it. The matplotlib toolbar (zoom, pan, home, save) appears above each figure. Right-click any graph to copy it to clipboard or save it.

**Save Graph** saves the current figure. **Save All Graphs** exports every generated graph to a chosen directory.

### Graph categories (25 graphs total)

**Composition**
- Amino Acid Composition (Bar), Amino Acid Composition (Pie)

**Profiles**
- Hydrophobicity Profile (Kyte-Doolittle sliding window)
- Local Charge Profile (NCPR sliding window)
- Local Complexity (Shannon entropy, dashed threshold at 2.0 bit)
- Disorder Profile (per-residue 0–1, orange fill = disordered > 0.5)
- Linear Sequence Map (three-track overview: hydrophobicity / NCPR / disorder)
- Coiled-Coil Profile (heptad-periodicity score, fill above 0.50)

**Charge & π-Interactions**
- Isoelectric Focus (Henderson-Hasselbalch charge curve, pI and pH 7.4 annotated)
- Charge Decoration (Das-Pappu FCR vs |NCPR| phase diagram)
- Cation–π Map (proximity heat map for K/R ↔ F/W/Y pairs)

**Structure & Folding**
- Bead Model (Hydrophobicity), Bead Model (Charge)
- Sticker Map, Helical Wheel, TM Topology

**Phase Separation / IDP**
- Uversky Phase Plot (mean |charge| vs normalised hydrophobicity)
- Saturation Mutagenesis (20×n heatmap of single-substitution effects on GRAVY + NCPR)

**Aggregation & Solubility**
- β-Aggregation Profile, Solubility Profile, Hydrophobic Moment

**New Features**
- PTM Map, RNA-Binding Profile, SCD Profile, pI/MW Map
- Truncation Series (from Truncation tab), MSA Conservation (from MSA tab), Complex Mass (from Complex tab)

**AlphaFold / Structural** *(requires a loaded structure)*
- pLDDT Profile (four confidence bands: very high / high / low / very low)
- Distance Map (Cα pairwise distances, 8 Å contact contour)
- Domain Architecture (multi-track: Pfam domains, disorder, LC, TM)
- Ramachandran Plot (φ/ψ coloured by secondary structure)
- Residue Contact Network (graph of residues within 8 Å)

---

## Structure Tab

Interactive 3D viewer powered by [3Dmol.js](https://3dmol.csb.pitt.edu). Requires `pip install PySide6-WebEngine`.

The tab is split into a **left control panel** and a **3D canvas**.

### Representation

| Option | Description |
|--------|-------------|
| Cartoon | Ribbon/helix cartoon (default) |
| Stick | Licorice stick bonds |
| Sphere | Space-filling VDW spheres |
| Line | Wireframe |
| Surface | Molecular surface (SAS) with semi-transparent cartoon underneath |
| Opacity slider | Adjusts transparency (10–100 %) |

### Color modes

| Mode | Schemes |
|------|---------|
| pLDDT / B-factor | Red-White-Blue · Blue-White-Red · Rainbow · Sinebow |
| Residue type | Amino Acid (UniProt style) · Shapely |
| Chain | Chain colors |
| Charge | Blue (positive) / Red (negative) / Grey (neutral) |
| Hydrophobicity | Cyan-White-Orange · Blue-White-Red · Green-White-Red |
| Mass | Blue-Red · Rainbow |

### Legend & view controls

- **Show color bar** — gradient legend (bottom-right) for continuous modes, or categorical legend for Charge/Residue/Chain
- **Show XYZ axes** — VMD-style coordinate axis indicator (bottom-left): X = red, Y = green, Z = blue; updates in real time as you rotate
- **Background presets**: Black, White, Grey — or **Custom…** opens a color picker
- **Reset View** — zoom to fit the whole structure
- **Spin** — continuous auto-rotation (choose X / Y / Z axis)
- **Snapshot PNG** — renders the current view and saves as PNG

### Export Structure / Sequence

Opens a format chooser:

| Format | Description |
|--------|-------------|
| PDB | Standard Protein Data Bank format |
| mmCIF | PDBx/mmCIF format |
| GRO | GROMACS coordinate format (coordinates in nm) |
| XYZ | Element + Cartesian coordinates (Ångström) |
| FASTA | Amino-acid sequence only |

---

## BLAST Tab

Submits the current sequence to NCBI blastp via Biopython. Requires internet; typically takes 1–3 minutes.

Choose database (nr, swissprot, pdb, refseq_protein) and max hits (5–100), then click **BLAST Current Sequence**. Results show Accession, Description, Length, Score, E-value, % Identity. Click **Load** in any row to immediately load that sequence and re-run analysis.

---

## Multichain Analysis Tab

Populated automatically when a multi-FASTA file or multi-chain PDB is imported. Columns: ID · Length · MW · Net Charge · % Hydrophobic · % Hydrophilic · % +Charged · % −Charged · % Neutral. Double-click any row to load that chain. Export as CSV or JSON.

---

## Compare Tab

Paste two sequences (or FASTA entries) side by side and click **Compare Sequences** to get a property table (length, MW, pI, GRAVY, FCR, NCPR, net charge, aromaticity, extinction coefficient) plus side-by-side disorder, hydrophobicity, and aggregation profile overlays in the Graphs tab.

---

## Truncation Series Tab

With a sequence analysed, set the truncation step (%) and choose N-terminal, C-terminal, or both, then click **Run Truncation Series**. BEER computes properties at each truncation length and generates the Truncation Series graph.

---

## MSA Tab

Paste a multi-FASTA alignment. Click **Run MSA** to compute per-column sequence conservation and generate the MSA Conservation graph.

---

## Complex Mass Tab

Paste chain sequences in FASTA format (header = chain ID). Enter stoichiometry as a string like `A2B1`. Click **Calculate Complex** to get per-chain and total MW, extinction coefficients, and the Complex Mass bar chart.

---

## Settings Tab

### Analysis parameters
| Setting | Default | Description |
|---------|---------|-------------|
| Default pH | 7.0 | pH for net-charge and charge-curve calculations |
| Sliding Window Size | 9 | Window width for hydrophobicity / NCPR / entropy profiles |
| Override pKa | — | Nine comma-separated values: N-term, C-term, D, E, C, Y, H, K, R |
| Reducing conditions | Off | If on, Cys is not paired in disulphide bonds for extinction coefficient |

### Graph appearance
| Setting | Description |
|---------|-------------|
| Label / Tick Font Size | Axis title and tick label size |
| Marker Size | Data marker size in scatter/line graphs |
| Default Graph Format | PNG / SVG / PDF |
| Bead Colormap | Colormap for bead hydrophobicity model |
| Graph Accent Colour | Main accent colour for bars, lines, fills |
| Show Graph Titles | Toggle axis titles |
| Show Grid | Toggle grid lines |
| Transparent background | Transparent PNG/SVG exports |

### Interface
| Setting | Description |
|---------|-------------|
| UI Font Size | Global application font size |
| Dark Theme | Light/dark colour scheme |
| Enable Tooltips | Hover tooltips |
| Colourblind-safe palette | Paul Tol colourblind-safe colours |

### ESM2
Choose model size (8M default, 35M, 150M, 650M). Changing model and clicking **Apply Settings** reinitialises the embedder and clears the cache.

All settings save automatically to `~/.beer/config.json` and persist across restarts. **Reset to Defaults** restores factory values.

---

## Help Tab

Built-in reference covering all major features. Two extra buttons at the bottom:

- **Copy Citation (BibTeX)** — copies the BibTeX entry for BEER to the clipboard
- **Generate Methods Paragraph** — auto-generates a methods paragraph based on your current sequence and settings, ready to paste into a manuscript

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Enter` | Run analysis |
| `Ctrl+E` | Export analysis |
| `Ctrl+G` | Jump to Graphs tab |
| `Ctrl+S` | Save session |
| `Ctrl+O` | Load session |
| `Ctrl+F` | Focus motif search box |
| `Ctrl+/` | Show all shortcuts overlay |

---

## Session Save & Load

Sessions are saved as `.beer` JSON files that capture the current sequence, name, and all settings. **Load Session** restores the state and re-runs analysis automatically.

---

## ESM2 Neural Predictions

BEER uses Meta's ESM2 protein language model with pre-trained linear probe heads (bundled in `beer/models/`). No training step is required.

| Prediction | Without ESM2 | With ESM2 | Test set |
|------------|-------------|-----------|----------|
| Disorder | Propensity scale | ESM2 logistic probe | DisProt 2024, AUC 0.83 |
| Aggregation | ZYGGREGATOR scale | ESM2 probe | UniProt amyloid, AUC 0.97 |
| Signal peptide | Hydrophobicity heuristic | ESM2 context-aware probe | UniProt ft_signal, AUC 1.00 |
| PTM sites | Consensus motif scan | ESM2 per-position probe | UniProt mod_res, AUC 0.93 |

All benchmarks use protein-level 80/20 splits (seed=42).

### Model sizes

| Model | Parameters | Speed | Download |
|-------|-----------|-------|----------|
| `esm2_t6_8M_UR50D` *(default)* | 8 M | Fastest | ~30 MB |
| `esm2_t12_35M_UR50D` | 35 M | Fast | ~140 MB |
| `esm2_t30_150M_UR50D` | 150 M | Moderate | ~580 MB |
| `esm2_t33_650M_UR50D` | 650 M | Slow (GPU helps) | ~2.6 GB |

Weights download once on the first Analyze call and cache in `~/.cache/torch/hub/`. Up to 32 sequences are cached in memory per session.

---

## Metrics Reference

### Sequence properties

| Metric | Definition |
|--------|-----------|
| MW | Sum of residue masses + water (Da) |
| pI | pH where net charge = 0 (Henderson-Hasselbalch) |
| GRAVY | Mean Kyte-Doolittle hydropathicity |
| Aromaticity | (F + W + Y) / length |
| Extinction coefficient | W×5500 + Y×1490 + (C–C)×125 at 280 nm |

### Charge metrics

| Metric | Definition |
|--------|-----------|
| FCR | Fraction of charged residues = (K + R + D + E) / length |
| NCPR | Net charge per residue = (positive − negative) / length |
| κ (kappa) | Charge patterning: 0 = well-mixed, 1 = fully segregated (Das & Pappu 2013) |
| Ω (omega) | Sticker patterning, same scale as κ |

### Phase separation / IDP metrics

| Metric | Definition |
|--------|-----------|
| LARKS | 7-residue windows with ≥1 aromatic (F/W/Y), ≥50% LC residues, Shannon entropy < 1.8 bit (Hughes et al. 2018) |
| SCD | Sequence charge decoration — pairwise charge product weighted by sequence separation (Sawle & Ghosh 2015) |
| Prion-like score | Fraction of N, Q, S, G, Y residues |

### Aggregation & solubility

| Metric | Source |
|--------|--------|
| ZYGGREGATOR | β-aggregation propensity per residue (Tartaglia & Vendruscolo 2008) |
| CamSol | Intrinsic solubility scale (Sormanni et al. 2015) |

### PTM predictions

Only well-validated rules are included. Rules with false positive rates > 50% have been removed.

| PTM | Method |
|-----|--------|
| N-linked glycosylation | N[^P][ST] sequon |
| Phosphoserine/Thr (CK2) | [ST]xx[DE] |
| Phosphoserine/Thr (PKA) | R[^P][^P][ST] |
| Ubiquitination | [LVIMF]K.[DE] |
| SUMOylation | [VILMF]K.E |
| N-terminal acetylation | NatA substrate rules |
| Arginine methylation | RGG, RG, GR motifs |

### RNA binding

No composite score is reported (any weighting would be arbitrary without validation data). Instead, BEER reports mean per-residue propensity, K/R/Y/F/W fraction, and motif hits (RGG, RRM, KH, SR, DEAD-box, Zinc finger).

---

## License

GNU General Public License v2. See `LICENSE`.

---

## Author

Saumyak Mukherjee
Department of Theoretical Biophysics, Max Planck Institute of Biophysics, Frankfurt am Main, Germany
mukherjee.saumyak50@gmail.com
