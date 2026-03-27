# BEER — Biophysical and Evolutionary Evaluation of Residues

**BEER** is a desktop application for integrated biophysical analysis of protein sequences. It accepts a sequence (pasted, imported as FASTA/PDB, or fetched from UniProt/RCSB), runs 19 analysis modules in one click, and gives you interactive publication-quality graphs, a 3D structure viewer, and exportable reports — all from a single GUI.

I built BEER because I wanted a single tool that handles everything from basic physicochemical properties to disorder prediction, aggregation hotspots, PTM sites, RNA-binding propensity, and phase separation metrics, without jumping between half a dozen web servers.

> **If you use BEER in your research, please cite:**
> Mukherjee, S. *arXiv*:2504.20561. DOI: [https://doi.org/10.48550/arXiv.2504.20561](https://arxiv.org/abs/2504.20561)

---

## What's new in v2.0

Version 1.0 was a single monolithic script with a basic GUI. v2.0 is a full rewrite:

- **Proper Python package** (`beer/`) — modular, installable via `pip`
- **ESM2 neural predictions** for disorder, aggregation, signal peptide, and PTM — pre-trained heads bundled, no training needed
- **3D structure viewer** with multiple representations, colour modes, colour bar, spin, and snapshot export
- **25 graphs** across 8 categories (up from ~12), including Ramachandran, contact network, pLDDT profile, domain architecture
- **New analysis modules**: RNA binding, SCD/κ/Ω, LARKS, tandem repeats, TM topology, coiled coil, ELM linear motifs
- **New utility tabs**: BLAST, Multichain, Compare, Truncation Series, MSA Conservation, Complex Mass
- **Protein summary bar**: fetches name, gene, organism, and function from UniProt or RCSB automatically after a fetch
- **Session-only history**: the last 10 analysed sequences are available in a dropdown during the session and cleared when you close the app
- Persistent settings, drag-and-drop FASTA, session save/load, keyboard shortcuts overlay, right-click figure menu
- Structure export in PDB, mmCIF, GRO, XYZ, and FASTA formats
- Removed unreliable metrics (Instability Index, LLPS composite score, Chou-Fasman, unvalidated PTM rules)

---

## Installation

**Requirements:** Python 3.12 · macOS, Windows, or Linux · ~200 MB disk space

```bash
conda create -n beer python=3.12 -y
conda activate beer
git clone https://github.com/chemgame/BEER.git
cd BEER
pip install .
```

Install PyTorch and ESM2 neural prediction models (CPU build):

```bash
pip install "torch>=2.0" --index-url https://download.pytorch.org/whl/cpu
pip install fair-esm scipy
```

**Linux only** — install Qt platform libraries and set the library path:

```bash
conda install -n beer -c conda-forge xcb-util-cursor xcb-util-image xcb-util-keysyms xcb-util-renderutil xcb-util-wm libxkbcommon libnss libdrm libxcomposite libxdamage libxrandr libgbm -y
```

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/activate.d/beer_xcb.sh
```

---

## Quick Start

```bash
conda activate beer
beer
```

1. Paste an amino-acid sequence (or drag-and-drop a `.fasta` file onto the window)
2. Click **Analyze** or press `Ctrl+Enter`
3. Browse the 19 report sections in the left panel of the Analysis tab
4. Go to the **Graphs** tab and click any graph name
5. Click **Export Analysis** to save the full report (CSV, JSON, PDF, or DAT)

Internet is only needed for external fetches (UniProt, AlphaFold, Pfam, ELM, DisProt, PhaSepDB, BLAST). All local analysis runs offline.

---

## Input Methods

| Method | How |
|--------|-----|
| **Paste sequence** | Type or paste a bare amino-acid string or FASTA block and click **Analyze** |
| **Import FASTA** | Click **Import FASTA** → select a `.fa` / `.fasta` file; multi-sequence files load all chains into the Multichain tab |
| **Import PDB** | Click **Import PDB** → select a `.pdb` file; all chains available in the Chain dropdown |
| **Fetch UniProt** | Type a UniProt accession (e.g. `P04637`) → click **Fetch**; unlocks AlphaFold, Pfam, DisProt, PhaSepDB buttons and shows a protein summary |
| **Fetch PDB ID** | Type a 4-character RCSB code (e.g. `1UBQ`) → click **Fetch**; loads structure and all chains automatically |
| **Drag & Drop** | Drag a `.fasta` file directly onto the BEER window |
| **History** | A dropdown next to the toolbar lists the last 10 sequences analysed in the current session; selecting one re-runs the full analysis immediately. History is cleared when you close the app. |

---

## Analysis Tab

After running analysis, the left panel lists 19 report sections. Click any section name to display it.

When you fetch a protein from UniProt or RCSB, a compact **protein info bar** appears above the report panel, showing the protein name, gene, organism, and a one-line functional description.

### Toolbar

| Button | Action |
|--------|--------|
| **Analyze** | Run analysis (`Ctrl+Enter`) |
| **Export Analysis** | CSV / JSON / PDF / DAT (`Ctrl+E`) |
| **Mutate…** | Point-mutation dialog |
| **Save / Load Session** | Save or restore a `.beer` JSON session file |
| **Figure Composer** | Assemble a custom multi-panel publication figure |
| **Fetch** | Download sequence from UniProt or RCSB PDB |
| **Fetch AlphaFold** | Download predicted structure from EBI AlphaFold |
| **Fetch Pfam / ELM** | Domain and linear motif annotations |
| **DisProt / PhaSepDB** | Disorder and phase-separation database annotations |
| **MobiDB** | Consensus disorder annotations from MobiDB (fraction disordered, predictor count, disordered regions) |
| **Variants** | Natural variants and mutagenesis data from UniProt |

Residues in the sequence viewer are colour-coded by type. Use the **Search** / **Highlight** box to find motifs or regex patterns. Below the viewer: **Copy Sequence** (whole or range) and **Clear All** (resets everything).

### Report sections

| Section | Contents |
|---------|----------|
| **Properties** | MW, pI, GRAVY, aromaticity, aliphatic index, extinction coefficient |
| **Composition** | AA counts and frequencies, sortable by name / frequency / hydrophobicity |
| **Hydrophobicity** | Kyte-Doolittle statistics, hydrophobic and hydrophilic fractions |
| **Charge** | FCR, NCPR, κ, Ω, net charge, charge asymmetry |
| **Aromatic & π** | Aromatic fraction, cation–π and π–π pair counts |
| **Low Complexity** | Shannon entropy, prion-like score, LC fraction, PLAAC score (Lancaster et al. 2014), PolyX stretches |
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
| **Proteolytic Map** | Predicted cleavage sites for 9 enzymes (Trypsin, Chymotrypsin, Lys-C, Asp-N, Glu-C, CNBr, Arg-C); peptide masses in Da |
| **Comparison** | Side-by-side disorder / hydrophobicity / aggregation overlays |

---

## Graphs Tab

Navigate using the **category tree** on the left. The matplotlib toolbar (zoom, pan, home) appears above each figure. Right-click any graph to copy to clipboard or save. **Save All Graphs** exports every generated graph to a directory.

| Category | Graphs |
|----------|--------|
| Composition | AA Composition (Bar), AA Composition (Pie) |
| Profiles | Hydrophobicity, Local Charge, Local Complexity, Disorder, Linear Sequence Map, Coiled-Coil |
| Charge & π | Isoelectric Focus, Charge Decoration (Das-Pappu), Cation–π Map |
| Structure & Folding | Bead Model (Hydrophobicity), Bead Model (Charge), Sticker Map, Helical Wheel, TM Topology |
| Phase Sep / IDP | Uversky Phase Plot, Saturation Mutagenesis |
| Aggregation | β-Aggregation Profile, Solubility Profile, Hydrophobic Moment |
| Sequence Analysis | Annotation Track, Cleavage Map, PLAAC Profile |
| New Features | PTM Map, RNA-Binding Profile, SCD Profile, pI/MW Map, Truncation Series, MSA Conservation, Complex Mass |
| AlphaFold / Structural* | pLDDT Profile, Distance Map, Domain Architecture, Ramachandran Plot, Residue Contact Network |

*Structural graphs require a loaded structure (from AlphaFold fetch or PDB import).

**Sequence Analysis graphs:**

| Graph | Description |
|-------|-------------|
| **Annotation Track** | Unified five-track view: disorder, hydrophobicity, aggregation, feature annotations (TM helices, signal peptide, LARKS, PTM sites), and a sequence ruler — all aligned on the same x-axis |
| **Cleavage Map** | Predicted proteolytic cut sites for 9 enzymes displayed as coloured ticks on horizontal tracks; includes a trypsin peptide mass summary |
| **PLAAC Profile** | Per-residue prion-like amino acid composition score (Lancaster et al. 2014), with prion-like regions highlighted |

---

## Structure Tab

Interactive 3D viewer powered by [3Dmol.js](https://3dmol.csb.pitt.edu), embedded via Qt WebEngine (bundled with PySide6). The tab has a **left control panel** and a **3D canvas**. The default background is white.

| Control | Options |
|---------|---------|
| **Representation** | Cartoon (default), Stick, Sphere, Line, Cross, Trace, Surface |
| **Color mode** | pLDDT/B-factor, Residue type, Chain, Charge, Hydrophobicity, Mass, Secondary Structure, Spectrum (N→C) |
| **Color scheme** | Mode-dependent: Red-White-Blue, Rainbow, Shapely, Cyan-White-Orange, JMol, PyMOL, Spectrum, etc. |
| **Color bar** | Toggleable gradient/categorical legend overlay (bottom-right) |
| **Background** | White (default), Black, Grey presets or custom color picker |
| **Spin** | Continuous auto-rotation on X / Y / Z axis |
| **Reset View** | Restores default representation, colour mode, white background, and camera position |
| **Snapshot PNG** | Saves current view as PNG |

**Export Structure / Sequence** saves in PDB, mmCIF, GRO, XYZ, or FASTA format.

---

## Other Tabs

| Tab | What it does |
|-----|-------------|
| **BLAST** | Submits current sequence to NCBI blastp (1–3 min); click **Load** on any hit to re-run analysis on that sequence |
| **Multichain** | Auto-populated from multi-FASTA or multi-chain PDB; shows MW, charge, composition per chain; double-click a row to load it |
| **Compare** | Side-by-side property table and profile overlays for two sequences |
| **Truncation Series** | Computes properties across progressive N/C truncations and generates the Truncation Series graph |
| **MSA** | Paste a multi-FASTA alignment → per-column conservation graph |
| **Complex Mass** | Paste chains + stoichiometry (e.g. `A2B1`) → total MW, extinction coefficients, bar chart |
| **Help** | Built-in reference; **Copy Citation (BibTeX)** and **Generate Methods Paragraph** buttons |

---

## Settings Tab

| Group | Setting | Default |
|-------|---------|---------|
| Analysis | Default pH | 7.0 |
| Analysis | Sliding Window Size | 9 |
| Analysis | Override pKa | — (nine comma-separated values) |
| Analysis | Reducing conditions | Off |
| Graphs | Label / Tick font size, Marker size, Format (PNG/SVG/PDF) | — |
| Graphs | Bead colormap, Accent colour, Titles, Grid, Transparent BG | — |
| Interface | Dark theme, Tooltips | — |
| ESM2 | Model size (8M / 35M / 150M / 650M) | 8M |

Click **Apply Settings** to save to `~/.beer/config.json`. **Reset to Defaults** restores factory values.

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

## ESM2 Neural Predictions

BEER uses Meta's ESM2 protein language model with pre-trained linear probe heads bundled in `beer/models/`. No training is required. Weights download once on the first Analyze call and cache in `~/.cache/torch/hub/`.

| Model | Parameters | Speed | Download |
|-------|-----------|-------|----------|
| `esm2_t6_8M_UR50D` *(default)* | 8 M | Fastest | ~30 MB |
| `esm2_t12_35M_UR50D` | 35 M | Fast | ~140 MB |
| `esm2_t30_150M_UR50D` | 150 M | Moderate | ~580 MB |
| `esm2_t33_650M_UR50D` | 650 M | Slow (GPU recommended) | ~2.6 GB |

Change the model in **Settings → ESM2 model** and click **Apply Settings**.

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
| FCR | (K + R + D + E) / length |
| NCPR | (positive − negative) / length |
| κ (kappa) | Charge patterning: 0 = well-mixed, 1 = fully segregated (Das & Pappu 2013) |
| Ω (omega) | Sticker patterning, same scale as κ |

### IDP / Phase separation

| Metric | Definition |
|--------|-----------|
| Aliphatic index | 100 × (A + 2.9V + 3.9(I+L)) / length; higher values indicate greater thermostability (Ikai 1980) |
| LARKS | 7-residue windows: ≥1 aromatic, ≥50% LC residues, entropy < 1.8 bit (Hughes et al. 2018) |
| SCD | Pairwise charge product weighted by sequence separation (Sawle & Ghosh 2015) |
| PLAAC score | Per-residue log-odds of yeast prion-like FG vs SwissProt background, window = 41 (Lancaster et al. 2014) |
| PolyX stretch | Run of ≥4 identical consecutive residues |
| Prion-like score | Fraction of N, Q, S, G, Y residues |
| ZYGGREGATOR | β-aggregation propensity per residue (Tartaglia & Vendruscolo 2008) |
| CamSol | Intrinsic solubility scale (Sormanni et al. 2015) |

### PTM predictions

Only well-validated motif rules are included (false positive rate < 50%).

| PTM | Motif |
|-----|-------|
| N-linked glycosylation | N[^P][ST] |
| Phosphoserine/Thr (CK2) | [ST]xx[DE] |
| Phosphoserine/Thr (PKA) | R[^P][^P][ST] |
| Ubiquitination | [LVIMF]K.[DE] |
| SUMOylation | [VILMF]K.E |
| N-terminal acetylation | NatA substrate rules |
| Arginine methylation | RGG, RG, GR motifs |

---

## License

GNU General Public License v2. See `LICENSE`.

---

## Author

Saumyak Mukherjee
Department of Theoretical Biophysics, Max Planck Institute of Biophysics, Frankfurt am Main, Germany
