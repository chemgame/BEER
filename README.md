# BEER — Biophysical Evaluation Engine for Residues

**Website:** [https://www.saumyakmukherjee.com/beer](https://www.saumyakmukherjee.com/beer)

**BEER** is a desktop application for integrated biophysical analysis of protein sequences. It accepts a sequence (pasted, imported as FASTA/PDB, or fetched from UniProt/RCSB), runs 19 analysis modules and up to 24 ESM2 BiLSTM neural prediction heads on demand, and gives you interactive publication-quality graphs, a 3D structure viewer, and exportable per-section reports — all from a single GUI.

I built BEER because I wanted a single tool that handles everything from basic physicochemical properties to disorder prediction, aggregation hotspots, RNA-binding propensity, and phase separation metrics, without jumping between half a dozen web servers.

> **If you use BEER in your research, please cite:**
> Mukherjee, S. *arXiv*:2504.20561. DOI: [https://doi.org/10.48550/arXiv.2504.20561](https://arxiv.org/abs/2504.20561)

---

## What's new in v2.0

Version 1.0 was a single monolithic script with a basic GUI. v2.0 is a full rewrite:

- **Proper Python package** (`beer/`) — modular, installable via `pip`
- **24 ESM2 BiLSTM neural prediction heads** — per-residue prediction of: disorder, signal peptide, transmembrane helices, coiled coils, DNA binding, active site, binding site, phosphorylation, low complexity, zinc finger, glycosylation, ubiquitination, methylation, acetylation, lipidation, disulfide bonds, intramembrane regions, functional motifs, propeptide, tandem repeats, RNA binding, nucleotide binding, transit peptide, and **aggregation propensity** (new in v2.0). Training uses focal loss, MMseqs2 clustered train/val/test splits, and curated structural databases (BioLiP, PDBTM, M-CSA, DisProt, dbPTM, GlyConnect, AmyLoad, AmyPro, WALTZ-DB 2.0) where available — not raw UniProt annotation alone
- **On-demand AI section loading** — AI Predictions sections are computed lazily (click-to-compute, VMD/PyMOL style); each head shares the cached ESM2 embedding so subsequent heads are fast
- **Unified BiLSTM overlay design** — each head shows a single figure; UniProt annotations (when fetched) are overlaid on the same axes as semi-transparent spans for direct visual comparison
- **3D structure viewer** with multiple representations, colour modes, colour bar, spin, and snapshot export
- **53 graphs** across 12 categories (up from ~12), including all 20 BiLSTM head profiles, Ramachandran, contact network, pLDDT profile, domain architecture
- **New analysis modules**: RNA binding (catRAPID), SCD/κ/Ω, LARKS, tandem repeats, TM topology (TMHMM 2.0 local), coiled coil (COILS), ELM linear motifs, phosphorylation PWMs (NetPhos-style), catGRANULE phase-separation, SignalP D-score
- **New utility tabs**: BLAST, Multichain, Compare, Truncation Series, MSA Conservation, Complex Mass
- **Protein summary bar**: fetches name, gene, organism, and function from UniProt or RCSB automatically after a fetch
- **Session-only history**: the last 10 analysed sequences are available in a dropdown during the session and cleared when you close the app
- **Official logo** and About dialog — logo displayed in the taskbar/Dock and accessible via the Help tab
- **3D viewer residue click** — click any residue in the structure viewer to highlight it in gold (PyMOL-style) and open a popup with all per-residue prediction scores (disorder, signal peptide, TM, BiLSTM heads, pLDDT)
- **Alanine scan sub-tab** — inside the Analysis tab, systematically mutate every position in a chosen range to Ala and see ΔGRAVY, ΔMW, ΔCharge, and ΔDisorder fraction in a table + bar chart; export as CSV
- **Phosphorylation context filter** — predicted phospho sites in disordered regions (BiLSTM > 0.5) or low-confidence structure (pLDDT < 70) are flagged as higher-confidence predictions
- **MC-Dropout uncertainty** — each BiLSTM profile tab has a "Show Uncertainty" button that runs Monte Carlo dropout and adds ±1σ bands to the profile
- **Smart Summary tab** — dedicated bulleted summary of all key predictions; shown after analysis
- **Headless CLI** — `beer analyze` subcommand with full flags for batch use (see below)
- Persistent settings, drag-and-drop FASTA, session save/load, keyboard shortcuts overlay, right-click figure menu (copy, save PNG/SVG/PDF, export underlying data as CSV/JSON)
- Structure export in PDB, mmCIF, GRO, XYZ, and FASTA formats
- Removed unreliable metrics (Instability Index, LLPS composite score, Chou-Fasman)
- Removed bead models and composition pie chart (redundant with bar plot and 3D viewer)

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
3. Browse the report sections in the left panel of the Analysis tab — click any section to display it
4. Click any section under **AI Predictions** to compute that BiLSTM head on demand
5. Go to the **Graphs** tab and click any graph name; right-click a graph to save it as PNG/SVG/PDF or export its data

Internet is only needed for external fetches (UniProt, AlphaFold, Pfam, ELM, DisProt, PhaSepDB, BLAST). All local analysis runs offline.

### CLI (headless) mode

BEER also runs without a display for batch use:

```bash
# Analyse a sequence, output JSON to stdout
beer analyze --sequence MADVFGKDMVNQ... --format json

# Analyse from a FASTA file, write HTML report
beer analyze --fasta my_protein.fasta --format html --output report.html

# Fetch from UniProt and export a TSV summary (skip ESM2 for speed)
beer analyze --accession P04637 --format tsv --no-esm2 --output p53.tsv

# Print version
beer version
```

Full flag list:

| Flag | Short | Description |
|------|-------|-------------|
| `--sequence` | `-s` | Protein sequence (single-letter code) |
| `--fasta` | `-f` | FASTA file path (first record used) |
| `--accession` | `-a` | UniProt or PDB accession to fetch |
| `--output` | `-o` | Output file (default: stdout) |
| `--format` | | `json` (default), `csv`, `tsv`, `html` |
| `--window` | `-w` | Sliding window size (default: 9) |
| `--hydro-scale` | | Hydrophobicity scale name |
| `--no-esm2` | | Skip BiLSTM predictions |
| `--sections` | | Report only specified sections |
| `--accent` | | Accent colour hex for HTML output |

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

After running analysis, the left panel lists 19 report sections. Click any section name to display it. The report panel has two tabs — **Report** (all sections) and **Alanine Scan** (systematic single-site Ala mutation scan with ΔGRAVY, ΔMW, ΔCharge, and ΔDisorder fraction).

When you fetch a protein from UniProt or RCSB, a compact **protein info bar** appears above the report panel, showing the protein name, gene, organism, and a one-line functional description.

### Toolbar

| Button | Action |
|--------|--------|
| **Analyze** | Run fast (classical) analysis (`Ctrl+Enter`) |
| **AI Analysis** | Run full ESM2 BiLSTM analysis (all 24 heads at once) |
| **Mutate…** | Point-mutation dialog |
| **Save / Load Session** | Save or restore a `.beer` JSON session file |
| **Figure Composer** | Assemble a custom multi-panel publication figure |
| **Fetch** | Download sequence from UniProt or RCSB PDB |
| **Fetch AlphaFold** | Download predicted structure from EBI AlphaFold |
| **Fetch Pfam / ELM** | Domain and linear motif annotations |
| **DisProt / PhaSepDB** | Disorder and phase-separation database annotations |
| **MobiDB** | Consensus disorder annotations from MobiDB (fraction disordered, predictor count, disordered regions) |
| **Variants** | Natural variants and mutagenesis data from UniProt |
| **IntAct** | Curated binary interactions from the IntAct molecular interaction database (EBI); shows partner, detection method, MI-score, and PubMed link |

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
| **Disorder** | ESM2 BiLSTM head (2-layer BiLSTM, trained on UniProt Swiss-Prot, AUROC 0.991); falls back to metapredict (Emenecker et al. 2021) or classical propensity scale; includes mean probability, predicted fraction, and predicted region list |
| **Aggregation** | ZYGGREGATOR hotspots (Tartaglia & Vendruscolo 2008); CamSol solubility; optional ESM2 probe (Settings) |
| **Signal Peptide** | ESM2 BiLSTM head (AUROC 0.9999); classical Von Heijne (1986) n/h/c model and D-score fallback; AXA cleavage motif; GPI anchor (Eisenhaber et al. 1999). Optional deep-learning upgrade via **SignalP 6.0** button (BioLib, requires pybiolib + login) |
| **RNA Binding** | catRAPID-style composite score ω̄ (Bellucci et al. 2011 Nat Methods); per-residue catRAPID profile; RGG/RRM/KH/SR/DEAD-box motif scan |
| **Amphipathic Helices** | Regions with μH ≥ 0.35 (Eisenberg 1984); hydrophobic moment profile for α-helix (δ=100°) and β-strand (δ=160°) |
| **SCD / κ / Ω** | Sequence charge decoration profile |
| **LARKS** | Low-complexity Aromatic-Rich Kinked Segments (Hughes et al. 2018) |
| **Tandem Repeats** | Direct, tandem, and compositional repeats |
| **TM Topology** | TMHMM 2.0 (Krogh et al. 2001) bundled locally — 395-state profile HMM, NumPy Viterbi, no internet needed. Classical Kyte-Doolittle sliding-window TM prediction has been removed in v2.0; use **AI Predictions → Transmembrane** (ESM2 BiLSTM) or the **TMHMM** button for topology. Optional cloud upgrade via **DeepTMHMM** button (BioLib) |
| **Coiled Coil** | Full COILS algorithm (Lupas et al. 1991 Science 252:1162): MTIDK 20×7 position-weight matrix, all 7 heptad registers swept, log-odds converted to P(CC) via calibrated sigmoid |
| **Linear Motifs** | Regex scan: NLS, NES, PxxP, 14-3-3, KFERQ, KDEL, SxIP, NxS/T, … |
| **Proteolytic Map** | Predicted cleavage sites for 9 enzymes (Trypsin, Chymotrypsin, Lys-C, Asp-N, Glu-C, CNBr, Arg-C); peptide masses in Da |
| **Phosphorylation** | NetPhos-style PWM scan (Blom et al. 1999) for PKA (R[R/K]x[S/T]), PKC ([S/T]x[R/K]), CK2 ([S/T]xxE/D), and Src/Tyr kinase (YxxΦ) sites |
| **Sticker & Spacer** | Sticker count/spacing, **catGRANULE score** (Bolognesi et al. 2016 Cell Reports 14:2535): linear combination of catRAPID, disorder, and inverse hydrophobicity; score > 0 predicts condensate formation |
| **Comparison** | Side-by-side disorder / hydrophobicity / aggregation overlays |

---

## Graphs Tab

Navigate using the **category tree** on the left. The matplotlib toolbar (zoom, pan, home) appears above each figure. Click the **ⓘ** button (bottom-right of each graph) for a detailed description, equations, and references. **Right-click any graph** for three options: copy to clipboard, save figure (PNG/SVG/PDF), or **Export Graph Data…** — writes the underlying data (residue scores, domain lists, site tables, etc.) to a CSV or JSON file so you can re-plot or analyse it with external tools. Individual graphs and reports are exported per-section; there is no bulk "export all" function.

| Category | Graphs |
|----------|--------|
| Composition | AA Composition (Bar), AA Composition (Pie) |
| Sequence Profiles | Hydrophobicity, Local Charge, Local Complexity, Disorder Profile, Linear Sequence Map, Coiled-Coil Profile |
| Charge & π | Isoelectric Focus, Charge Decoration (Das-Pappu), Cation–π Map |
| Structure & Folding | Bead Model (Hydrophobicity), Bead Model (Charge), Sticker Map, Helical Wheel, TM Topology |
| Phase Sep / IDP | Uversky Phase Plot, Single-Residue Perturbation Map, SCD Profile |
| Aggregation | β-Aggregation Profile, Solubility Profile, Hydrophobic Moment |
| Sequence Analysis | Annotation Track, Cleavage Map, PLAAC Profile |
| Post-Translational & Binding | RNA-Binding Profile |
| Evolutionary & Comparative | Truncation Series, MSA Conservation, MSA Covariance, Complex Mass |
| AlphaFold / Structural* | pLDDT Profile, Distance Map, Domain Architecture, Ramachandran Plot, Residue Contact Network |
| Variant Effects | Variant Effect Map, AlphaMissense Pathogenicity |
| BiLSTM Head Profiles | Signal Peptide Profile, Transmembrane Profile, Coiled-Coil Profile, DNA-Binding Profile, Active Site Profile, Binding Site Profile, Phosphorylation Profile, Low-Complexity Profile, Intramembrane Profile, Zinc Finger Profile, Glycosylation Profile, Ubiquitination Profile, Methylation Profile, Acetylation Profile, Lipidation Profile, Disulfide Bond Profile, Functional Motif Profile, Propeptide Profile, Repeat Region Profile |

*Structural graphs require a loaded structure (from AlphaFold fetch or PDB import).

**Sequence Analysis graphs:**

| Graph | Description |
|-------|-------------|
| **Annotation Track** | Unified five-track view: disorder, hydrophobicity, aggregation, feature annotations (TM helices, signal peptide, LARKS), and a sequence ruler — all aligned on the same x-axis |
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
| **MSA** | Paste a multi-FASTA alignment → per-column conservation graph + residue covariance heatmap (MI with APC; requires ≥4 sequences, ≤500 columns) |
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
| Graphs | Label / Tick font size (default 11/9), Marker size, Format (PNG/SVG/PDF) | — |
| Graphs | Bead colormap, Heatmap colormap, Accent colour, Titles, Grid, Transparent BG | — |
| Interface | Dark theme, Tooltips | — |
| ESM2 | Model size (8M / 35M / 150M / 650M) | 650M |

Click **Apply Settings** to save to `~/.beer/config.json`. **Reset to Defaults** restores factory values.

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Enter` | Run analysis |
| `Ctrl+G` | Jump to Graphs tab |
| `Ctrl+S` | Save session |
| `Ctrl+O` | Load session |
| `Ctrl+F` | Focus motif search box |
| `Ctrl+/` | Show all shortcuts overlay |

---

## ESM2 Neural Predictions

BEER uses Meta's ESM2 650M protein language model with 24 pre-trained BiLSTM head weights bundled in `beer/models/`. The ESM2 backbone (~2.6 GB) downloads once on the first AI Predictions call and caches in `~/.cache/torch/hub/`.

**On-demand computation**: clicking any section under **AI Predictions** in the sidebar triggers computation of that head only. The ESM2 embedding is cached after the first head, so all subsequent heads reuse it and are fast. Running **AI Analysis** computes all 24 heads at once.

### Head architectures

| Architecture | Used for | Details |
|---|---|---|
| **BiLSTM** (standard) | 21 heads | 2-layer bidirectional LSTM, hidden=256, focal loss, MMseqs2-clustered split |
| **BiLSTM-CRF** | Transmembrane | CRF decoder enforces valid TM topology (outside→helix→inside transitions) |
| **BiLSTM-Window** | Aggregation | Window-average pooling over 9-residue context before sigmoid output |

### Training data sources

| Head | Primary curated source | Fallback |
|------|----------------------|---------|
| Disorder | DisProt (experimental) | UniProt ft_region:disordered |
| Signal Peptide | UniProt ft_signal (Swiss-Prot) | — |
| Transmembrane | PDBTM (structural, per-helix) | UniProt ft_transmem |
| Active Site | M-CSA (catalytic mechanisms) | UniProt ft_act_site |
| Binding Site | BioLiP (PDB-derived, small-molecule) | UniProt ft_binding |
| DNA Binding | BioLiP (PDB-derived) | UniProt ft_dna_bind |
| RNA Binding | BioLiP (PDB-derived) | UniProt ft_region:RNA-binding |
| Nucleotide Binding | BioLiP (ATP/ADP/NAD/FAD/CoA/…) | UniProt ft_np_bind |
| Zinc Finger | BioLiP (Zn-coordinating residues) | UniProt ft_zn_fing |
| Phosphorylation | dbPTM (PSP + PhosphoELM + HPRD aggregate) | UniProt ft_mod_res |
| Ubiquitination | dbPTM | UniProt ft_mod_res |
| Methylation | dbPTM | UniProt ft_mod_res |
| Acetylation | dbPTM | UniProt ft_mod_res |
| Glycosylation | GlyConnect (site-resolved glycoproteomics) | UniProt ft_carbohyd |
| Aggregation | WALTZ-DB 2.0 + AmyLoad + AmyPro + PDB fibrils | — |
| Coiled Coil | UniProt ft_coiled (COILS/Marcoil) | — |
| Lipidation | UniProt ft_lipid | — |
| Disulfide Bond | UniProt ft_disulfid | — |
| Intramembrane | UniProt ft_intramem | — |
| Functional Motif | UniProt ft_motif | — |
| Propeptide | UniProt ft_propep | — |
| Repeat Region | UniProt ft_repeat | — |
| Low Complexity | UniProt ft_compbias | — |
| Transit Peptide | UniProt ft_transit | — |

### AUROC

| Head | AUROC |
|------|-------|
| Disorder | 0.991 |
| Signal Peptide | 0.9999 |
| Transmembrane | 0.992 |
| All others | — *(training in progress; values will be updated)* |

If ESM2 is not installed, BEER falls back automatically: disorder uses **metapredict** (Emenecker et al. 2021, *Cell Syst.*) if available, or a classical sliding-window propensity scale otherwise. All other analysis runs fully offline without ESM2. The BiLSTM head profiles are silently skipped if the head file is not present.

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
| ZYGGREGATOR | Per-residue β-aggregation Z-score (Z_agg^i): 7-residue window average of intrinsic propensity p_agg^i with 21-residue gatekeeper charge correction, normalised to a SwissProt random-sequence baseline. Hotspots where Z_agg ≥ 1.0 over ≥ 4 consecutive residues. (Tartaglia et al. 2008 J. Mol. Biol.; Tartaglia & Vendruscolo 2008 Chem. Soc. Rev.) |
| CamSol | Intrinsic solubility scale (Sormanni et al. 2015) |

---

## License

GNU General Public License v2. See `LICENSE`.

---

## Author

Saumyak Mukherjee
Theoretical Biophysics, Max Planck Institute of Biophysics, Frankfurt am Main, Germany
