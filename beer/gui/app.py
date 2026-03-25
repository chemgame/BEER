"""BEER application entry point (PySide6)."""
from __future__ import annotations

import sys
import warnings


def main() -> None:
    warnings.filterwarnings("ignore", message="Setting the 'color' property will override")
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

    import matplotlib
    matplotlib.use("QtAgg")

    from PySide6.QtWidgets import QApplication
    from beer.embeddings import ESM2_AVAILABLE, get_embedder
    from beer.gui.main_window import ProteinAnalyzerGUI

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("BEER")

    # Only initialise the embedder if ESM2 is importable; skip the model
    # download at startup — the model loads lazily on first embed() call.
    embedder = get_embedder("esm2_t6_8M_UR50D") if ESM2_AVAILABLE else None

    window = ProteinAnalyzerGUI(embedder=embedder)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
