# BEER – Biochemical Estimator & Explorer of Residues

A cross-platform desktop GUI for rapid physicochemical analysis of protein sequences. ```BEER``` accepts FASTA or PDB inputs (including multichain files) or manual sequence entry and produces comprehensive biochemical profiles with interactive visualizations. This repository contains the main application `beer.py` and an example pdb file, `1GP2.pdb` along with the `README.md` and `LICENSE` files.

## Reference
Please cite the following paper if you use this application for your research:
Mukherjee, S. arXiv:2504.20561. DOI: [https://doi.org/10.48550/arXiv.2504.20561](https://arxiv.org/abs/2504.20561)

## Prerequisites

- **Operating Systems:** Windows, macOS, Linux (tested on CentOS Stream 8, macOS Sequoia 15.3.2., and Windows 11)
- **Python:** 3.11–3.12

## Dependencies

Install the required packages in a dedicated Conda environment:

```bash
conda create -n beer python=3.12
conda activate beer
pip install biopython matplotlib pyqt5 mplcursors
```
Alternatively, you may also use a conda virtual environment.

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/chemgame/beer.git
   cd beer
   ```
2. Ensure your `beer` environment is active (see Dependencies).
3. Run the application:
   ```bash
   python beer.py
   ```
To make the script executable and available system-wide:

```bash
chmod +x beer.py
mkdir -p ~/bin
mv beer.py ~/bin
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
