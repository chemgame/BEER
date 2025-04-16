# PROBE
PROBE – Protein Review &amp; Bioinformatics Evaluator

A professional-grade protein sequence analyzer with a graphical user interface (GUI) built using PyQt5 and Matplotlib. The software computes a wide range of protein properties, presents detailed reports in HTML tables, and displays interactive graphs. It also supports batch analysis of multi-FASTA files, theme customization, and dynamic settings updates.

## Features

- **Sequence Analysis:**
  - Calculate amino acid composition, molecular weight, isoelectric point (pI), and molar extinction coefficient.
  - Compute GRAVY (hydropathy) score, instability index, aromaticity, and secondary structure predictions.
  - Predict net charge at pH 7.0 and a user-specified pH.
  - Generate a hydrophobicity profile using a sliding-window algorithm (Kyte–Doolittle scale).

- **Reporting:**
  - Detailed HTML-formatted reports presented as tables.
  - A dedicated Annotation section (placeholder for future integration with external annotation services).

- **Visualization:**
  - Embedded graphs for:
    - Amino acid composition (bar chart)
    - Hydrophobicity profile (line chart)
    - Net charge versus pH
    - Bead models (colored by hydrophobicity and by charge)
  - Adjustable label and tick fonts, plus configurable colormap.
  - Navigation toolbars for zooming and panning on graphs.

- **Batch Analysis:**
  - Import and analyze multiple protein sequences from a multi-FASTA file.
  - Summary table view with key properties for each sequence.
  - Double-click table rows to view additional details.

- **Customization & Export:**
  - Settings tab to adjust sliding window size, font sizes, colormap, theme (light/dark), and bead label display.
  - Dynamic update of reports and graphs when settings are applied.
  - Export reports as text and export a combined PDF report (including graphs) or view a print preview.

## Installation

### Prerequisites

- Python 3.x  
- [Biopython](https://biopython.org/)
- [Matplotlib](https://matplotlib.org/)
- [PyQt5](https://pypi.org/project/PyQt5/)

### Install Dependencies

You can install the required packages using pip:

```bash
pip install biopython matplotlib PyQt5
