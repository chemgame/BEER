# PROBE - PROtein analyzer and Bioinformatics Estimator

A GUI-based protein sequence analyzer that integrates multiple analysis methods and interactive visualizations for protein sequence analyses. The application supports importing sequences via FASTA or PDB files, performing detailed biochemical analyses, and exporting comprehensive reports.

## Features
- **Sequence Import:**  
  Import protein sequences from FASTA and PDB files. Supports multichain files.

- **Comprehensive Analysis:**  
  Provides detailed biochemical properties including amino acid composition, molecular weight, isoelectric point, extinction coefficient, GRAVY score, instability index, aromaticity, net charge (at customizable pH values), and predictions for solubility.

- **Interactive Graphs:**  
  Visualizations include bar and pie charts for amino acid composition, line graphs for hydrophobicity profiles and net charge vs. pH, bead models for hydrophobicity and charge, and a radar chart comparing key properties. Graphs are enhanced with interactive tooltips using mplcursors.

- **Multichain Analysis:**  
  Information on multiple protein chains simultaneously with CSV and JSON export options for the resulting summary data.

- **Reporting and Exporting:**  
  - Save reports as formatted text files.
  - Export full analysis reports along with graphs to PDF.
  - Save individual graph images and export all graphs in one operation.

- **Customizable Settings:**  
  Adjust parameters and toggle between light and dark themes.

## Dependencies
Ensure you have the following Python packages installed:

- **Biopython**
- **Matplotlib**
- **PyQt5**
- **mplcursors**
- **Requests**

Install them using pip or conda:

```bash
pip install biopython matplotlib PyQt5 mplcursors requests
```
```bash
conda install biopython matplotlib PyQt5 mplcursors requests
```
It is recommended to work in a separate conda environment.
```bash
conda create -n probe python=3.12
```
```bash
conda activate probe
```
## Installation
1. Clone or download the repository containing the probe.py script.
2. Make sure your Python environment meets the above dependencies.

## Usage
To run the application, simply execute the following command in your terminal:
```bash
python probe.py
```
Upon launching, the GUI will appear, offering tabs for individual sequence analysis, graphs, batch processing, settings, and help. Import your protein sequences via FASTA or PDB files (or copy-pasting the sequence) and click Analyze to see the results.
  
## License
This project is released under the GNU General Public License. For more details, please refer to the license file accompanying this project.

## Author and Contact
- Developed by Saumyak Mukherjee
- Contact: saumyak.mukherjee@biophys.mpg.de
