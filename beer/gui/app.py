"""BEER application entry point (PySide6)."""
from __future__ import annotations

import logging
import sys
import traceback
import warnings
from pathlib import Path


def _setup_logging() -> None:
    """Configure file + console logging for the BEER session."""
    log_dir = Path.home() / ".beer"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "beer.log"

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stderr),
        ],
    )
    # Quieten noisy third-party loggers
    for noisy in ("matplotlib", "PIL", "urllib3", "torch", "esm"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _install_exception_hook(app) -> None:
    """Show a dialog for unhandled exceptions instead of silently crashing."""
    import logging
    _log = logging.getLogger("beer.excepthook")

    def _handler(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        _log.error("Unhandled exception:\n%s", tb_str)
        try:
            from PySide6.QtWidgets import QMessageBox, QApplication
            if QApplication.instance():
                msg = QMessageBox()
                msg.setWindowTitle("BEER — Unexpected Error")
                msg.setIcon(QMessageBox.Icon.Critical)
                msg.setText(
                    "<b>An unexpected error occurred.</b><br>"
                    "The error has been logged to <tt>~/.beer/beer.log</tt>.<br><br>"
                    "You can continue using BEER, but some functions may not work correctly."
                )
                msg.setDetailedText(tb_str)
                msg.setStandardButtons(QMessageBox.StandardButton.Ok)
                msg.exec()
        except Exception:
            pass

    sys.excepthook = _handler


def main() -> None:
    _setup_logging()
    _log = logging.getLogger("beer.app")
    _log.info("BEER starting up")

    warnings.filterwarnings("ignore", message="Setting the 'color' property will override")
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    warnings.filterwarnings("ignore", message=".*tight_layout.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*constrained_layout.*", category=UserWarning)

    import matplotlib
    matplotlib.use("QtAgg")

    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QIcon
    import importlib.resources
    from beer.embeddings import ESM2_AVAILABLE, get_embedder
    from beer.gui.main_window import ProteinAnalyzerGUI
    import beer

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("BEER")
    app.setApplicationVersion(beer.__version__)

    _logo_path = str(importlib.resources.files("beer").joinpath("beer.png"))
    if Path(_logo_path).exists():
        app.setWindowIcon(QIcon(_logo_path))

    _install_exception_hook(app)

    if ESM2_AVAILABLE:
        _log.info("ESM2 available — initialising embedder (lazy load)")
    else:
        _log.warning("ESM2 not available (fair-esm / torch not installed)")

    embedder = get_embedder("esm2_t6_8M_UR50D") if ESM2_AVAILABLE else None

    window = ProteinAnalyzerGUI(embedder=embedder)
    window.show()
    _log.info("BEER v%s window shown", beer.__version__)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
