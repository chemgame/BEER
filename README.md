# PRISM – Protein Residue Informatics & Sequence Metrics

A cross-platform desktop GUI for rapid physicochemical analysis of protein sequences. PRISM accepts FASTA or PDB inputs (including multichain files) or manual sequence entry and produces comprehensive biochemical profiles with interactive visualizations.

## Prerequisites

- **Operating Systems:** Windows, macOS, Linux (tested on CentOS Stream 8, macOS Sequoia 15.3.2., and Windows 11)
- **Python:** 3.11–3.12

## Dependencies

Install the required packages in a dedicated Conda environment:

```bash
conda create -n prism python=3.12
conda activate prism
pip install biopython matplotlib pyqt5 mplcursors
```

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/chemgame/PRISM.git
   cd PRISM
   ```
2. Ensure your `prism` environment is active (see Dependencies).
3. Run the application:
   ```bash
   python prism.py
   ```

To make the script executable and available system-wide:

```bash
chmod +x prism.py
mkdir -p ~/bin
mv prism.py ~/bin/prism
export PATH="$HOME/bin:$PATH"
```

## Features

- **Sequence Import:** FASTA or PDB (multi-chain)
- **Biochemical Analysis:** Amino acid composition, hydrophobicity profiles, net charge vs. pH (with pI), solubility prediction, molecular weight, extinction coefficient, GRAVY, instability index, aromaticity
- **Interactive Graphs:** Bar & pie charts, sliding-window plots, bead models, radar charts (powered by `mplcursors`)
- **Multichain Comparison:** Side-by-side CSV/JSON export of summary metrics
- **Reporting:** Save text reports, export full PDF reports, and batch-export graphs in PNG/SVG/PDF
- **Customizable Settings:** Adjustable pH/window size, theme toggle (light/dark), font sizes, colormaps, and override defaults (e.g., custom pKa lists)

## License

Released under the GNU General Public License v2. See the `LICENSE` file for details.

## Author & Contact

Developed by Saumyak Mukherjee  
Email: saumyak.mukherjee@biophys.mpg.de
