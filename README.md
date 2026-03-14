# BEER – Biochemical Estimator & Explorer of Residues

A cross-platform desktop GUI for rapid physicochemical analysis of protein sequences.
**BEER** accepts FASTA, PDB, or plain-text input (single or multi-sequence / multi-chain),
fetches sequences from **UniProt** or **RCSB PDB** by accession, and produces comprehensive
biochemical profiles with interactive, publication-quality visualisations.

This repository contains the main application `beer.py`, an example structure file
`1GP2.pdb`, `README.md`, and `LICENSE`.

## Reference

If you use BEER in your research, please cite:

> Mukherjee, S. *arXiv*:2504.20561.
> DOI: [https://doi.org/10.48550/arXiv.2504.20561](https://arxiv.org/abs/2504.20561)

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Operating system | Windows, macOS, Linux (tested on CentOS Stream 8, macOS Sequoia 15.3.2, Windows 11) |
| Python | 3.11 – 3.12 |

## Dependencies

```bash
conda create -n beer python=3.12
conda activate beer
pip install biopython matplotlib pyqt5 mplcursors
# Optional: 3D structure viewer
pip install PyQtWebEngine
```

## Getting Started

```bash
git clone https://github.com/chemgame/beer.git
cd beer
conda activate beer
python beer.py
```

**Optional — make the script executable system-wide:**

```bash
chmod +x beer.py
mkdir -p ~/bin && cp beer.py ~/bin/
export PATH="$HOME/bin:$PATH"
```

---

## Application Overview

BEER is organised into eight sections, accessible via the **left sidebar**:

| Section | Purpose |
|---------|---------|
| **Analysis** | Sequence input, chain selector, sequence viewer, 13 report sections |
| **Graphs** | 23 interactive, publication-quality plots |
| **Structure** | Interactive 3D viewer (requires PyQtWebEngine) + AlphaFold fetch |
| **BLAST** | NCBI blastp against nr, SwissProt, PDB, or RefSeq |
| **Compare** | Side-by-side physicochemical comparison of two sequences |
| **Multichain Analysis** | Batch summary table; CSV / JSON export |
| **Settings** | All configurable parameters |
| **Help** | Built-in method and graph reference |

---

## Sequence Input

1. **Paste / type** a plain amino-acid string or FASTA block.
2. **Import FASTA** — single or multi-record `.fa` / `.fasta` file.
3. **Import PDB** — all protein chains extracted; named `<PDB>_<chain>`.
4. **Fetch by accession:**
   - **UniProt ID** (e.g. `P04637`) — fetches from UniProt REST API; also enables
     **Fetch AlphaFold** and **Fetch Pfam** buttons.
   - **PDB ID** (e.g. `1ABC`) — fetches the FASTA sequence from RCSB PDB.
     After fetching a PDB ID, clicking **Fetch AlphaFold** or **Fetch Pfam** will
     prompt you to enter a UniProt accession.

The **Sequence Viewer** displays the active sequence in UniProt-style formatting
(groups of 10, ruler every 10 residues). A built-in regex motif search and highlight
tool is provided.

---

## Analysis Modules

Each module is shown in a dedicated section within the Analysis tab.

### Composition
Amino acid count and frequency (%) for all 20 standard residues.
Sort buttons: A–Z, by frequency, hydrophobicity ascending/descending.

### Properties
Molecular weight · pI · Net charge (pH 7.0 and custom) · Extinction coefficient (280 nm,
reducing/non-reducing) · GRAVY · Instability index · Aromaticity.

### Hydrophobicity
Kyte-Doolittle per-residue values, GRAVY, and fraction of hydrophobic / hydrophilic / neutral.

### Charge
FCR · NCPR · Charge asymmetry · κ (Das & Pappu 2013).

### Aromatic & π-Interactions
Aromatic fraction (F+W+Y), cation–π pairs (K/R ↔ F/W/Y, ±4 residues), π–π pairs.

### Low Complexity
Shannon entropy · Normalised entropy · Unique AA count · Prion-like score (N,Q,S,G,Y) ·
LC fraction (windows with entropy < 2.0 bits, w=12).

### Disorder
Disorder-promoting fraction (Uversky) · Order-promoting fraction · Aliphatic index ·
Ω (Das et al. 2015).

### Secondary Structure (Chou-Fasman)
Mean Pα and Pβ propensities, count of helix- and sheet-promoting residues, mean IUPred-inspired disorder score.

### Repeat Motifs
RGG · FG · YG/GY · SR/RS · QN/NQ.

### Sticker & Spacer
Sticker counts (aromatic + electrostatic), mean/min/max inter-sticker spacing (Mittag & Pappu).

### TM Helices
Kyte-Doolittle sliding-window (w=19, threshold=1.6) transmembrane helix prediction with
inside-positive topology. See algorithm details below.

### Phase Separation
Composite LLPS propensity score (0–1) weighted from aromatic fraction, prion-like score,
disorder fraction, FCR, Omega, LARKS density, and |NCPR| penalty. LARKS detection (7-residue
windows: ≥1 aromatic, ≥50% LC residues, entropy < 1.8 bits; Hughes et al. 2018 *Science*).

### Linear Motifs
Regex scan against 15 built-in SLiM patterns: NLS, NES, PxxP, 14-3-3, RGG, FG, KFERQ,
KDEL, PKA (RxxS/T), SxIP, WW ligand, caspase-3, N-glycosylation, SUMOylation, CK2 sites.

---

## Transmembrane Helix Prediction

1. Full 19-residue windows (no partial edges) are scored by KD average.
2. Residues covered by any window ≥ 1.6 are marked as TM candidates.
3. Contiguous segments of 17–25 aa are retained as helices; over-long segments are trimmed to the single best window.
4. Topology assigned by **inside-positive rule (von Heijne)**: the flanking side with more K/R is cytoplasmic.

---

## Graphs (23 total)

All graphs are rendered at 120 dpi on-screen and 200 dpi on export with interactive
cursors (`mplcursors`).

### Composition
| Graph | Description |
|-------|-------------|
| AA Composition (Bar) | Residue counts with frequency annotations |
| AA Composition (Pie) | Residue frequencies (zero-count residues hidden) |

### Profiles
| Graph | Description |
|-------|-------------|
| Hydrophobicity Profile | Sliding-window KD average (window from Settings) |
| Local Charge Profile | Sliding-window NCPR |
| Local Complexity | Sliding-window Shannon entropy; LC threshold line |
| Disorder Profile | IUPred-inspired per-residue score; orange fill = disordered |
| Coiled-Coil Profile | Per-residue heptad-periodicity score; fill above 0.50 = predicted coiled-coil |
| Linear Sequence Map | Four-track: hydrophobicity, NCPR, disorder, helix Pα |
| Secondary Structure | **Two-panel**: helix Pα (top) and sheet Pβ (bottom); fill above 1.0 |

### Charge & π-Interactions
| Graph | Description |
|-------|-------------|
| Net Charge vs pH | Henderson-Hasselbalch curve 0–14; pI annotated |
| Isoelectric Focus | Enhanced curve with pH 7.4 annotation |
| Charge Decoration | Das-Pappu FCR vs \|NCPR\| phase diagram |
| Cation–π Map | K/R ↔ F/W/Y proximity heat map (score = 1/distance, ±8 residues) |

### Structure & Folding
| Graph | Description |
|-------|-------------|
| Bead Model (Hydrophobicity) | Per-residue KD scatter; 30+ colourmap choices |
| Bead Model (Charge) | K/R blue, D/E red, H cyan, neutral grey |
| Sticker Map | Aromatic amber, basic blue, acidic pink, spacer grey |
| Helical Wheel | Cartesian 18-residue projection; connecting lines; luminance-contrast labels |
| TM Topology | Snake-plot of predicted TM helices |

### AlphaFold / Structural
| Graph | Description |
|-------|-------------|
| pLDDT Profile | Per-residue confidence (0–100); requires Fetch AlphaFold |
| Cα Distance Map | Pairwise distance heatmap; 8 Å contact contour; requires Fetch AlphaFold |
| Domain Architecture | Multi-track: Pfam domains + Disorder + Low Complexity + TM Helices |

### Phase Separation / IDP
| Graph | Description |
|-------|-------------|
| Uversky Phase Plot | Mean \|net charge\| vs mean hydrophobicity; Uversky 2000 boundary line; IDP vs ordered classification |
| Saturation Mutagenesis | 20×n heatmap of \|ΔGRAVY\| + \|ΔNCPR\| for all single substitutions; white dot = wild type (≤500 aa) |

---

## AlphaFold Integration

After fetching a UniProt accession, click **Fetch AlphaFold** to download the
AlphaFold2 predicted structure from EBI. Provides:
- Per-residue **pLDDT** confidence (stored in B-factor column)
- **Cα distance matrix** for the contact map
- Interactive **3D viewer** in the Structure section (requires PyQtWebEngine)

---

## Pfam Domain Annotations

After fetching a UniProt accession, click **Fetch Pfam** to query the
EMBL-EBI InterPro REST API for Pfam domain positions. The **Domain Architecture**
graph adds a Pfam track on top of the always-available Disorder, Low Complexity,
and TM Helix tracks.

---

## BLAST

Submits the current sequence to NCBI via `Bio.Blast.NCBIWWW` (blastp).
Databases: nr, swissprot, pdb, refseq_protein. Returns top hits with accession,
description, length, score, E-value, % identity. Clicking **Load** re-analyses
a hit sequence directly in BEER.

---

## Export

| Action | Output |
|--------|--------|
| **Export PDF** | Formatted PDF report (text/tables; graphs saved separately) |
| **Save Graph** | Single graph, 200 dpi, PNG / SVG / PDF |
| **Save All Graphs** | All graphs to a directory; respects transparent-background setting |
| **Export CSV / JSON** | Multichain summary table |

---

## Settings

All parameters take effect when **Apply Settings** is clicked.
**Reset to Defaults** restores everything to factory values immediately.

| Setting | Default | Description |
|---------|---------|-------------|
| Default pH | 7.0 | pH for net-charge calculations |
| Sliding Window Size | 9 | Window for hydrophobicity, charge, and entropy profiles |
| Override pKa | — | N-term, C-term, D, E, C, Y, H, K, R (comma-separated) |
| Reducing conditions | off | Free Cys for extinction coefficient |
| Label / Tick Font Size | 14 / 12 | Graph axis and tick label sizes (pt) |
| Default Graph Format | PNG | PNG / SVG / PDF |
| Bead Colormap | coolwarm | 30+ matplotlib colourmap choices |
| Graph Accent Colour | Royal Blue | 24 named colour choices; applied immediately on update |
| Show Graph Titles | on | Toggle titles above graphs |
| Show Grid | on | Toggle gridlines |
| Show bead labels | on | Residue letters on bead models (≤ 60 aa) |
| Transparent background | **on** | Transparent PNG/SVG export |
| UI Font Size | 12 pt | Global font size (8–24 pt) |
| Dark Theme | off | Light ↔ dark toggle |
| Enable Tooltips | **on** | Hover tooltips on Settings widgets |

---

## License

Released under the GNU General Public License v2. See the `LICENSE` file for full details.

---

## Author & Contact

Developed by Saumyak Mukherjee with help from LLMs
Email: saumyak.mukherjee@biophys.mpg.de
