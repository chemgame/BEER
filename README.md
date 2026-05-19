# BEER — Biophysical Evaluation Engine for Residues

**Website:** [https://chemgame.github.io/BEER](https://chemgame.github.io/BEER)

**BEER** is a desktop application for integrated biophysical analysis of protein sequences. It accepts a sequence (pasted, imported as FASTA/PDB, or fetched from UniProt/RCSB), runs 14 classical analysis sections and 12 ESM2 BiLSTM neural AI prediction heads on demand, and gives you interactive publication-quality graphs, a 3D structure viewer, and exportable per-section reports — all from a single GUI.

I built BEER because I wanted a single tool that handles everything from basic physicochemical properties to disorder prediction, aggregation hotspots, RNA-binding propensity, and phase separation metrics, without jumping between half a dozen web servers.

> **If you use BEER in your research, please cite:**
> Mukherjee, S. *arXiv*:2504.20561. DOI: [https://doi.org/10.48550/arXiv.2504.20561](https://arxiv.org/abs/2504.20561)

<details>
<summary>BibTeX</summary>

```bibtex
@misc{mukherjee2025beer,
  author    = {Mukherjee, Saumyak},
  title     = {{BEER}: Biophysical Evaluation Engine for Residues},
  year      = {2025},
  eprint    = {2504.20561},
  archivePrefix = {arXiv},
  primaryClass  = {q-bio.QM},
  doi       = {10.48550/arXiv.2504.20561},
  url       = {https://arxiv.org/abs/2504.20561},
}
```

</details>

---

## Installation

**Requirements:** Python 3.10 or later · macOS, Windows, or Linux · ~200 MB disk space

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

For GPU acceleration (CUDA 12.1):

```bash
pip install "torch>=2.0" --index-url https://download.pytorch.org/whl/cu121
pip install fair-esm scipy
```

**Development install** (editable, changes to source take effect immediately):

```bash
git clone https://github.com/chemgame/BEER.git && cd BEER
pip install -e ".[dev]"
```

**Linux only** — Qt requires xcb platform libraries. Choose the method that fits your system:

**With root access — Ubuntu / Debian:**
```bash
sudo apt-get install -y \
    libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
    libxcb-render-util0 libxcb-xkb1 libxkbcommon-x11-0 \
    libnss3 libxcomposite1 libxrandr2 libxdamage1 libdrm2 libgbm1
```

**With root access — Fedora / RHEL:**
```bash
sudo dnf install -y \
    xcb-util-cursor xcb-util-icccm xcb-util-image xcb-util-keysyms \
    xcb-util-renderutil libxkbcommon-x11 \
    nss libXcomposite libXrandr libXdamage libdrm mesa-libgbm
```

**Without root access** (HPC clusters, shared systems) — install into the conda environment instead:
```bash
conda install -n beer -c conda-forge \
    xcb-util-cursor xcb-util-image xcb-util-keysyms xcb-util-renderutil \
    xcb-util-wm libxkbcommon xorg-libxrandr xorg-libxcomposite \
    xorg-libxdamage nss libdrm -y

mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' \
    > $CONDA_PREFIX/etc/conda/activate.d/beer_xcb.sh
conda deactivate && conda activate beer
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

After running analysis, the left panel lists 14 classical analysis sections and a separate **AI Predictions** group. Click any section name to display it. The report panel has two tabs — **Report** (all sections) and **Alanine Scan** (systematic single-site Ala mutation scan with ΔGRAVY, ΔMW, ΔCharge, and ΔDisorder fraction).

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

### Classical analysis sections

These 14 sections are always computed when you click **Analyze** (no ESM2 required):

| Section | Contents |
|---------|----------|
| **Composition** | AA counts and frequencies, sortable by name / frequency / hydrophobicity |
| **Properties** | MW, pI, GRAVY, aromaticity, aliphatic index, extinction coefficient |
| **Hydrophobicity** | Sliding-window hydrophobicity profile; scale selectable in Settings (Kyte-Doolittle, Eisenberg, Wimley-White, Hessa, Moon-Fleming, GES, Hopp-Woods, Fauchère-Pliska, Urry); y-axis label, units, and the ⓘ tooltip all update to reflect the chosen scale |
| **Charge** | FCR, NCPR, κ, Ω, net charge, charge asymmetry |
| **Aromatic & π** | Aromatic fraction, cation–π and π–π pair counts |
| **Repeat Motifs** | Shannon entropy, prion-like score, LC fraction, PLAAC score (Lancaster et al. 2014), PolyX stretches |
| **Sticker & Spacer** | Sticker count/spacing, **catGRANULE score** (Bolognesi et al. 2016 Cell Reports 14:2535): linear combination of catRAPID, disorder, and inverse hydrophobicity; score > 0 predicts condensate formation |
| **LARKS** | Low-complexity Aromatic-Rich Kinked Segments (Hughes et al. 2018) |
| **Linear Motifs** | Regex scan: NLS, NES, PxxP, 14-3-3, KFERQ, KDEL, SxIP, NxS/T, … |
| **β-Aggregation & Solubility** | ZYGGREGATOR hotspots (Tartaglia & Vendruscolo 2008); CamSol solubility (Sormanni et al. 2015) |
| **Amphipathic Helices** | Regions with μH ≥ 0.35 (Eisenberg 1984); hydrophobic moment profile for α-helix (δ=100°) and β-strand (δ=160°) |
| **Charge Decoration (SCD)** | Sequence charge decoration profile (Sawle & Ghosh 2015) |
| **Tandem Repeats** | Direct, tandem, and compositional repeats |
| **Proteolytic Map** | Predicted cleavage sites for 9 enzymes (Trypsin, Chymotrypsin, Lys-C, Asp-N, Glu-C, CNBr, Arg-C, Pepsin, Thermolysin); peptide masses in Da |

### AI Predictions sections

Computed on-demand when you click a section under **AI Predictions** in the sidebar (or click **AI Analysis** to run all 24 at once). Every profile tab has a **Show Uncertainty (MC-Dropout)** checkbox that adds ±1σ confidence bands.

| Head | Notes |
|------|-------|
| **Disorder** | AUROC 0.991; falls back to metapredict (Emenecker et al. 2021) or classical scale if ESM2 unavailable |
| **Signal Peptide** | AUROC 0.9999; classical Von Heijne D-score and AXA motif shown alongside; GPI anchor detection |
| **Transmembrane** | BiLSTM-CRF with Viterbi topology decoding (outside→helix→inside); AUROC 0.992 |
| **Intramembrane** | Re-entrant membrane loops |
| **Coiled Coil** | |
| **DNA Binding** | Trained on BioLiP structure-derived protein-DNA contacts |
| **RNA Binding** | Trained on BioLiP protein-RNA contacts |
| **Active Site** | Trained on M-CSA mechanistically validated catalytic residues |
| **Binding Site** | Trained on BioLiP small-molecule binding residues |
| **Phosphorylation** | Trained on dbPTM experimental sites |
| **Ubiquitination** | Trained on dbPTM |
| **Methylation** | Trained on dbPTM |
| **Acetylation** | Trained on dbPTM |
| **Glycosylation** | Trained on GlyConnect site-resolved glycoproteomics data |
| **Lipidation** | |
| **Disulfide Bond** | |
| **Zinc Finger** | Trained on BioLiP Zn-coordinating residues |
| **Nucleotide Binding** | Trained on BioLiP (ATP/ADP/NAD/FAD/CoA/…) |
| **Low Complexity** | |
| **Functional Motif** | |
| **Propeptide** | |
| **Repeat Region** | |
| **Transit Peptide** | |
| **Aggregation Propensity** | BiLSTM-Window architecture (9-residue window-average pool); trained on WALTZ-DB 2.0 + AmyLoad + AmyPro + PDB amyloid fibrils |

---

## Graphs Tab

Navigate using the **category tree** on the left. The matplotlib toolbar (zoom, pan, home) appears above each figure. Click the **ⓘ** button (bottom-right of each graph) for a detailed description, equations, and references. **Right-click any graph** for three options: copy to clipboard, save figure (PNG/SVG/PDF), or **Export Graph Data…** — writes the underlying data (residue scores, domain lists, site tables, etc.) to a CSV or JSON file so you can re-plot or analyse it with external tools.

Each graph panel has two inline controls in its toolbar:

- **Format dropdown** (PNG / SVG / PDF) — sets the save format for that specific graph; overrides the global default.
- **Colormap dropdown** (heatmap-type graphs only: Distance Map, Residue Contact Network, Cation–π Map, MSA Covariance, Single-Residue Perturbation Map, Variant Effect Map, AlphaMissense, Helical Wheel) — lets you switch the colour scheme per graph without touching Settings.

Individual graphs and reports are exported per-section; there is no bulk "export all" function.

| Category | Graphs |
|----------|--------|
| **Composition** | Amino Acid Composition (Bar) |
| **AI Predictions** | Disorder Profile, Signal Peptide Profile, Transmembrane Profile, Intramembrane Profile, Coiled-Coil Profile, DNA-Binding Profile, RNA Binding Profile, Active Site Profile, Binding Site Profile, Phosphorylation Profile, Low-Complexity Profile, Zinc Finger Profile, Glycosylation Profile, Ubiquitination Profile, Methylation Profile, Acetylation Profile, Lipidation Profile, Disulfide Bond Profile, Functional Motif Profile, Propeptide Profile, Repeat Region Profile, Nucleotide-Binding Profile, Transit Peptide Profile, Aggregation Propensity Profile |
| **Other Sequence Profiles** | Hydrophobicity Profile, Local Charge Profile, SCD Profile, SHD Profile |
| **Charge & π-Interactions** | Isoelectric Focus, Charge Decoration, Cation–π Map |
| **Membrane & Amphipathicity** | TM Topology, Hydrophobic Moment, Helical Wheel |
| **Aggregation & Solubility** | β-Aggregation Profile, Solubility Profile |
| **Phase Separation & IDP** | Uversky Phase Plot, Single-Residue Perturbation Map, Sticker Map, PLAAC Profile |
| **Sequence Maps & Annotation** | Linear Sequence Map, Annotation Track, Domain Architecture, Cleavage Map |
| **AlphaFold & Structure**† | pLDDT / B-Factor Profile, SASA Profile, SS Bead Model, Distance Map, Residue Contact Network, Ramachandran Plot |
| **Variant Effects**† | Variant Effect Map, AlphaMissense |
| **Evolutionary & Comparative** | MSA Conservation, MSA Covariance, Truncation Series, Complex Mass |

†Requires a loaded structure (AlphaFold fetch or PDB import) or UniProt fetch for variant data.

**pLDDT / B-Factor Profile**: when the structure was loaded from the AlphaFold EBI database the graph is titled "pLDDT Profile" with confidence-zone colouring (Very High ≥ 90, Confident 70–90, Low 50–70, Very Low < 50). When loaded from RCSB PDB or a local file the graph switches to "B-Factor Profile" with crystallographic B-factor zones (< 20 / 20–40 / 40–60 / > 60 Å²).

**Domain Architecture**: shows Pfam domains (UniProt fetch required), BEER AI disorder regions, and disordered linker spans. TM helices are intentionally omitted here — use the dedicated **TM Topology** graph (TMHMM 2.0 snake-plot with inside/outside orientation) for transmembrane topology.

---

## Structure Tab

Interactive 3D viewer powered by [3Dmol.js](https://3dmol.csb.pitt.edu) (bundled locally — fully offline, no CDN), embedded via Qt WebEngine. The tab has a **View** control panel, an **Interact** panel, and a **3D canvas**. The default background is white.

**View tab controls**

| Control | Options |
|---------|---------|
| **Representation** | Cartoon (default), Stick, Sphere, Line, Cross, Trace, Surface |
| **Color mode** | pLDDT/B-factor, Residue type, Chain, Charge, Hydrophobicity, Mass, Secondary Structure, Residue Number, Solvent Accessibility, AI Features, Aggregation |
| **Color scheme** | Mode-dependent: Red-White-Blue, Rainbow, Shapely, Cyan-White-Orange, JMol, PyMOL, Plasma, etc. |
| **Color bar** | Toggleable gradient/categorical legend overlay (bottom-right) |
| **H-bonds** | Toggle backbone N–O hydrogen bonds (< 3.5 Å); choose cylinder colour and radius |
| **Contacts (8 Å)** | Toggle Cα–Cα contact lines; choose line colour and opacity |
| **Background** | White (default), Black, Grey presets or custom color picker |
| **Spin** | Continuous auto-rotation on X / Y / Z axis |
| **Reset View** | Restores default representation (Cartoon), colour mode (pLDDT), clears selections, and resets camera |
| **Snapshot PNG** | Saves current view as PNG |
| **Chains** | Toggle visibility of individual chains |

**Interact tab** — residue selection (by number, range, name, or chain), distance / angle / dihedral measurements between clicked atoms.

**Export Structure / Sequence** saves in PDB, mmCIF, GRO, XYZ, or FASTA format.

---

## Other Tabs

| Tab | What it does |
|-----|-------------|
| **BLAST** | Submits current sequence to NCBI blastp (1–3 min); click **Load** on any hit to re-run analysis on that sequence |
| **Multichain** | Auto-populated from multi-FASTA or multi-chain PDB; shows MW, charge, composition per chain; double-click a row to load it |
| **Compare** | Side-by-side property table and profile overlays for two sequences |
| **Truncation** | Computes properties across progressive N/C truncations and generates the Truncation Series graph |
| **MSA** | Paste a multi-FASTA alignment → per-column conservation graph + residue covariance heatmap (MI with APC; requires ≥4 sequences, ≤500 columns) |
| **Complex** | Paste chains + stoichiometry (e.g. `A2B1`) → total MW, extinction coefficients, bar chart |
| **Help** | Built-in reference; **Copy Citation (BibTeX)** and **Generate Methods Paragraph** buttons |

---

## Settings Tab

| Group | Setting | Default |
|-------|---------|---------|
| Analysis | Default pH | 7.0 |
| Analysis | Sliding Window Size | 9 |
| Analysis | Hydrophobicity Scale | Kyte-Doolittle |
| Analysis | Override pKa | — (nine comma-separated values) |
| Analysis | Reducing conditions | Off |
| Sequence Display | Sequence Name | — (uses FASTA/PDB name automatically) |
| Graphs | Label Font Size | 14 |
| Graphs | Tick Font Size | 12 |
| Graphs | Marker Size | 10 |
| Graphs | Graph Accent Colour | Royal Blue |
| Graphs | Show Graph Titles | On |
| Graphs | Show Grid | On |
| Graphs | Show residue labels on bead models (≤60 aa) | On |
| Graphs | Transparent background on PNG/SVG export | On |
| Interface | Dark Theme | Off |
| Interface | Enable Tooltips | On |

Click **Apply Settings** to save to `~/.beer/config.json`. **Reset to Defaults** restores factory values.

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Enter` | Run analysis |
| `Ctrl+G` | Jump to Graphs tab |
| `Ctrl+2` | Switch to Structure tab |
| `Ctrl+3` | Switch to BLAST tab |
| `Ctrl+7` | Switch to MSA tab |
| `Ctrl+S` | Save session |
| `Ctrl+O` | Load session |
| `Ctrl+F` | Focus motif search box |
| `Ctrl+Z` | Undo last mutation |
| `Ctrl+Right` | Next graph |
| `Ctrl+Left` | Previous graph |
| `Ctrl+/` | Show all shortcuts overlay |

---

## ESM2 Neural Predictions

The AI Predictions heads use Meta's ESM2 650M protein language model. The ESM2 backbone (~2.6 GB) downloads automatically on the first AI Predictions call and caches in `~/.cache/torch/hub/` — this only happens once.

**On-demand computation**: each head is computed only when you click it; the embedding is cached after the first head, so all subsequent heads are fast. Running **AI Analysis** computes all available heads at once.

If ESM2 is not installed, disorder falls back to **metapredict** (Emenecker et al. 2021, *Cell Syst.*) if available, or a classical sliding-window scale otherwise. All classical analysis runs fully offline without ESM2.

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
| ZYGGREGATOR | Per-residue β-aggregation propensity; hotspots flagged where score ≥ 1.0 over ≥ 4 consecutive residues (Tartaglia & Vendruscolo 2008) |
| CamSol | Intrinsic solubility scale (Sormanni et al. 2015) |

---

## License

GNU General Public License v3. See `LICENSE`.

---

## Author

Saumyak Mukherjee
Theoretical Biophysics, Max Planck Institute of Biophysics, Frankfurt am Main, Germany
