# BEER – Biochemical Estimator & Explorer of Residues

A cross-platform desktop GUI for rapid physicochemical analysis of protein sequences.
**BEER** accepts FASTA or PDB inputs (including multi-chain files), multi-sequence
paste input, or manual sequence entry, and produces comprehensive biochemical profiles
with interactive, publication-quality visualisations.

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

Install all required packages into a dedicated Conda environment:

```bash
conda create -n beer python=3.12
conda activate beer
pip install biopython matplotlib pyqt5 mplcursors
```

## Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/chemgame/beer.git
cd beer

# 2. Activate the environment
conda activate beer

# 3. Run the application
python beer.py
```

**Optional – make the script executable system-wide:**

```bash
chmod +x beer.py
mkdir -p ~/bin
cp beer.py ~/bin/
export PATH="$HOME/bin:$PATH"
```

---

## Application Overview

BEER is organised into five main tabs:

| Tab | Purpose |
|-----|---------|
| **Analysis** | Sequence input, chain selector, sequence viewer, and per-section report tabs |
| **Graphs** | 11 interactive, publication-quality plots |
| **Multichain Analysis** | Summary table for batch comparisons; CSV / JSON export |
| **Settings** | All configurable parameters |
| **Help** | Built-in definitions and method references |

---

## Sequence Input

BEER accepts sequences in three ways:

1. **Type or paste a single sequence** – plain amino acid string (standard one-letter codes).
2. **Paste multiple sequences** – either in FASTA format (`>name` header lines) or as bare
   sequences, one per line. Multiple sequences are automatically loaded into the
   Multichain Analysis tab; the first sequence is analysed immediately.
3. **Import FASTA file** – single or multi-record; record IDs are used as sequence names.
4. **Import PDB file** – all protein chains are extracted; the PDB filename and chain
   letter are used as the sequence name (e.g. `1GP2_A`).

The **Sequence Viewer** panel (left side of the Analysis tab) displays the active sequence
in UniProt-style formatting: groups of 10 residues, position numbers every 10 residues.

---

## Analysis Modules

Each module is shown in a dedicated sub-tab within the Analysis tab.

### Composition
- Amino acid count and frequency (%) for all 20 standard residues
- Sort buttons: alphabetical, by frequency, hydrophobicity ascending/descending

### Properties
- Sequence length
- Molecular weight (Da)
- Isoelectric point (pI)
- Net charge at pH 7.0 and at the user-defined pH
- Extinction coefficient at 280 nm (M⁻¹cm⁻¹); supports reducing / non-reducing conditions
- GRAVY score (Kyte-Doolittle grand average)
- Instability index
- Aromaticity (fraction of F, W, Y)

### Hydrophobicity
- GRAVY score and average Kyte-Doolittle value per residue
- Count and percentage of hydrophobic (KD > 0), hydrophilic (KD < 0), and neutral residues
- Per-residue KD value table sorted from most hydrophobic to most hydrophilic, with
  weighted contribution of each residue to the overall GRAVY score

### Charge
- Positive (K, R) and negative (D, E) residue counts
- FCR – fraction of charged residues
- NCPR – net charge per residue
- Charge asymmetry (positive / negative ratio)
- κ (kappa) – charge patterning parameter (Das & Pappu 2013):
  0 = well-mixed charges, 1 = fully segregated

### Aromatic & π-Interactions
- Aromatic fraction (F + W + Y) and per-type counts
- Cation–π pairs: K/R within ±4 positions of F/W/Y
- π–π pairs: F/W/Y within ±4 positions of another F/W/Y

### Low Complexity
- Shannon compositional entropy (bits; max = log₂20 ≈ 4.32)
- Normalised entropy
- Unique amino acid count
- Prion-like score (fraction of N, Q, S, G, Y; Lancaster & Bhatt)
- LC fraction: fraction of residues covered by windows with entropy < 2.0 bits (window = 12)

### Disorder
- Disorder-promoting fraction (A, E, G, K, P, Q, R, S; Uversky)
- Order-promoting fraction (C, F, H, I, L, M, V, W, Y)
- Aliphatic index (Ikai 1980)
- Ω (omega) – sticker patterning: 0 = evenly distributed, 1 = clustered (Das et al. 2015)

### Repeat Motifs
Counts of biologically relevant dipeptide / tripeptide repeat motifs:

| Motif | Biological context |
|-------|--------------------|
| RGG | RNA-binding; FUS, hnRNP family |
| FG | Nucleoporin IDRs |
| YG / GY | Tyr-Gly variants |
| SR / RS | Splicing factor signature |
| QN / NQ | Yeast prion signature |

### Sticker & Spacer
Based on the sticker-and-spacer model (Mittag & Pappu):

- Total, aromatic (F, W, Y), and electrostatic (K, R, D, E) sticker counts
- Mean, minimum, and maximum spacing between consecutive sticker residues

---

## Graphs

All 11 graphs are rendered at publication quality (120 dpi on screen, 200 dpi on export)
with clean spines, professional colour palettes, and interactive cursors (`mplcursors`).

| Graph | Description |
|-------|-------------|
| **Amino Acid Composition (Bar)** | Bar chart of residue counts, annotated with frequency percentages |
| **Amino Acid Composition (Pie)** | Pie chart of residue frequencies (zero-count residues hidden) |
| **Hydrophobicity Profile** | Sliding-window Kyte-Doolittle average; filled area above/below zero |
| **Net Charge vs pH** | Charge curve pH 0–14 with pI annotation and filled positive/negative regions |
| **Bead Model (Hydrophobicity)** | Per-residue scatter coloured by KD value (configurable colormap); ruler every 10 positions |
| **Bead Model (Charge)** | Per-residue scatter coloured by charge state (K/R blue, D/E pink, H cyan, neutral grey) |
| **Properties Radar Chart** | Normalised pentagon radar of MW, pI, GRAVY, instability, and aromaticity |
| **Sticker Map** | Per-residue scatter: aromatic (amber), basic (blue), acidic (pink), spacer (grey) |
| **Local Charge Profile** | Sliding-window NCPR with filled positive/negative regions |
| **Local Complexity** | Sliding-window Shannon entropy; LC threshold at 2.0 bits highlighted |
| **Cation–π Map** | 2-D heat map of K/R ↔ F/W/Y proximity (score = 1 / sequence distance) |

Each graph tab includes a **Save Graph** button (PNG / SVG / PDF at 200 dpi).
A **Save All Graphs** button batch-exports all generated graphs to a chosen directory.

---

## Multichain Analysis

When a multi-record FASTA or multi-chain PDB is imported (or multiple sequences are
pasted), a summary table is populated with:

| Column | Metric |
|--------|--------|
| ID | Sequence / chain identifier |
| Length | Number of residues |
| MW (Da) | Molecular weight |
| Net Charge | At pH 7.0 |
| % Hydro | % hydrophobic residues (KD > 0) |
| % Hydrophil | % hydrophilic residues |
| % +Charged | % positive residues (K, R, H) |
| % -Charged | % negative residues (D, E) |
| % Neutral | % remaining residues |

Double-clicking any row loads that sequence into the Analysis and Graphs tabs.
Export the full table as **CSV** or **JSON**.

---

## Export

| Action | Output |
|--------|--------|
| **Export PDF** | Formatted PDF report containing the sequence (UniProt style) and all analysis tables. Graphs are **not** embedded — save them separately from the Graphs tab. |
| **Save Graph** | Single graph at 200 dpi in PNG, SVG, or PDF format |
| **Save All Graphs** | All generated graphs exported to a chosen directory |
| **Export CSV / JSON** | Multichain summary table |

---

## Settings

All parameters take effect when **Apply Settings** is clicked.

| Setting | Default | Description |
|---------|---------|-------------|
| Default pH | 7.0 | pH used for net-charge calculations |
| Sliding Window Size | 9 | Window length for hydrophobicity and charge profiles |
| Override pKa list | — | Nine comma-separated values (N-term, C-term, D, E, C, Y, H, K, R) |
| Reducing conditions | off | Counts Cys as free thiol for the 280 nm extinction coefficient |
| Sequence Name | — | Override auto-detected name from FASTA/PDB |
| Label Font Size | 14 | Axis label font size in graphs (pt) |
| Tick Font Size | 12 | Tick label font size in graphs (pt) |
| Marker Size | 10 | Data marker size in line graphs |
| Default Graph Format | PNG | File format for Save Graph (PNG / SVG / PDF) |
| Bead Colormap | coolwarm | Colormap for the bead hydrophobicity model |
| Graph Accent Colour | #4361ee | Accent colour for line and bar graphs |
| Show Graph Titles | on | Toggle titles above graphs |
| Show Grid | on | Toggle gridlines on plots |
| Show bead labels | on | Show residue letters on bead models (sequences ≤ 60 aa) |
| UI Font Size (pt) | 12 | Global application font size (8–24 pt) |
| Dark Theme | off | Switch between light and dark UI themes |
| Enable Tooltips | off | Show tooltip hints on widgets |

**Reset to Defaults** restores all settings to the values in the table above.

---

## License

Released under the GNU General Public License v2. See the `LICENSE` file for full details.

---

## Author & Contact

Developed by Saumyak Mukherjee with help from LLMs
Email: saumyak.mukherjee@biophys.mpg.de
