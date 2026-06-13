"""BEER application entry point (PySide6)."""
from __future__ import annotations

import logging
import sys
import traceback
import warnings
from pathlib import Path


def _harden_native_runtime() -> None:
    """Best-effort guards against native (C-level) crashes before torch is imported.

    Two macOS-specific hazards have been observed when the ESMC backbone is first
    loaded from a Qt worker thread (e.g. MC-Dropout):
      * Duplicate OpenMP runtime — numpy/MKL and torch each ship a libomp; loading
        both can hard-segfault. ``KMP_DUPLICATE_LIB_OK=TRUE`` permits coexistence.
      * HuggingFace *tokenizers* fork-based parallelism leaks semaphores and can
        crash when driven from a background thread; disable it.
    Must run before any ``import torch`` / ``esm`` so the env vars take effect.
    Also enables ``faulthandler`` so a future hard crash prints a Python/C stack
    instead of a bare "segmentation fault".
    """
    import os
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        import faulthandler
        faulthandler.enable()
    except Exception:
        pass


def _setup_logging() -> None:
    """Configure file + console logging for the BEER session."""
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    try:
        log_dir = Path.home() / ".beer"
        log_dir.mkdir(exist_ok=True)
        handlers.insert(0, logging.FileHandler(log_dir / "beer.log", encoding="utf-8"))
    except OSError:
        pass  # log to stderr only if home dir is not writable
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)
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


# ---------------------------------------------------------------------------
# Headless CLI
# ---------------------------------------------------------------------------

def _run_cli(argv: list[str]) -> None:
    """Handle `beer analyze ...` and other CLI subcommands."""
    import argparse, json, csv as _csv

    parser = argparse.ArgumentParser(
        prog="beer",
        description="BEER — Biophysical Evaluation Engine for Residues",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ── analyze ──────────────────────────────────────────────────────────────
    p_ana = sub.add_parser(
        "analyze",
        help="Run headless analysis on a protein sequence and write results.",
    )
    src = p_ana.add_mutually_exclusive_group(required=True)
    src.add_argument("--sequence", "-s", metavar="SEQ",
                     help="Protein sequence in single-letter code.")
    src.add_argument("--fasta", "-f", metavar="FILE",
                     help="FASTA file (first record is used).")
    src.add_argument("--accession", "-a", metavar="ID",
                     help="UniProt or PDB accession to fetch and analyse.")

    p_ana.add_argument("--output", "-o", metavar="FILE",
                       help="Output file path (default: stdout for JSON/CSV, auto-named for HTML/PDF).")
    p_ana.add_argument("--format", metavar="FMT",
                       choices=["json", "csv", "html", "tsv"],
                       default="json",
                       help="Output format: json (default), csv, html, tsv.")
    p_ana.add_argument("--hydro-scale", metavar="SCALE",
                       default="Kyte-Doolittle",
                       help="Hydrophobicity scale name (default: Kyte-Doolittle).")
    p_ana.add_argument("--window", "-w", metavar="N", type=int, default=9,
                       help="Sliding window size for profile calculations (default: 9).")
    p_ana.add_argument("--no-esmc", action="store_true",
                       help="Skip ESMC BiLSTM predictions (faster; no GPU/ESMC required).")
    p_ana.add_argument("--sections", metavar="SEC", nargs="+",
                       help="Report only these named sections (e.g. Disorder 'Signal Peptide & GPI').")
    p_ana.add_argument("--accent", metavar="HEX", default="#4361ee",
                       help="Accent colour for HTML output (default: #4361ee).")

    # ── version ──────────────────────────────────────────────────────────────
    import beer as _beer_pkg
    sub.add_parser("version", help="Print the BEER version and exit.")

    args = parser.parse_args(argv)

    if args.command == "version" or args.command is None:
        print(f"BEER v{_beer_pkg.__version__}")
        return

    if args.command == "analyze":
        _cli_analyze(args)


def _cli_analyze(args) -> None:
    import json, csv as _csv, sys
    from pathlib import Path

    # --- Resolve sequence ---
    seq = ""
    if args.sequence:
        seq = args.sequence.strip().upper()
    elif args.fasta:
        from Bio import SeqIO
        rec = next(SeqIO.parse(args.fasta, "fasta"))
        seq = str(rec.seq).upper()
    elif args.accession:
        import urllib.request
        import urllib.parse as _urlparse
        from beer.utils.sequence import valid_uniprot, valid_pdb
        acc = args.accession.strip()
        if not valid_uniprot(acc) and not valid_pdb(acc):
            sys.exit(f"[beer] '{acc}' is not a valid UniProt accession or PDB entry ID.")
        url = f"https://rest.uniprot.org/uniprotkb/{_urlparse.quote(acc, safe='')}.fasta"
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                from io import StringIO
                from Bio import SeqIO
                rec = next(SeqIO.parse(StringIO(r.read().decode()), "fasta"))
                seq = str(rec.seq).upper()
        except Exception as exc:
            sys.exit(f"[beer] Failed to fetch {acc}: {exc}")

    seq = "".join(c for c in seq if c.isalpha())
    if not seq:
        sys.exit("[beer] Empty sequence — nothing to analyse.")

    print(f"[beer] Analysing {len(seq)}-residue sequence …", file=sys.stderr)

    # --- Load embedder ---
    embedder = None
    if not args.no_esmc:
        try:
            from beer.embeddings import ESMC_AVAILABLE, get_embedder
            if ESMC_AVAILABLE:
                embedder = get_embedder("esmc_600m")
        except Exception:
            pass

    # --- Run analysis ---
    from beer.analysis.core import analyze_sequence
    data = analyze_sequence(
        seq,
        embedder=embedder,
        window_size=args.window,
        hydro_scale=args.hydro_scale,
    )

    # --- Filter sections if requested ---
    sections = data.get("report_sections", {})
    if args.sections:
        sections = {k: v for k, v in sections.items() if k in args.sections}

    # --- Emit output ---
    fmt = args.format
    out_path = args.output

    if fmt == "json":
        numeric_keys = [k for k, v in data.items() if isinstance(v, (int, float, list, dict))
                        and k != "report_sections"]
        payload = {k: data[k] for k in numeric_keys}
        payload["section_names"] = list(sections.keys())
        text = json.dumps(payload, default=str, indent=2)
        _write_or_print(text, out_path)

    elif fmt in ("csv", "tsv"):
        delim = "\t" if fmt == "tsv" else ","
        import io
        buf = io.StringIO()
        w = _csv.writer(buf, delimiter=delim)
        w.writerow(["metric", "value"])
        scalar_keys = [
            "mol_weight", "iso_point", "net_charge_7", "gravy",
            "aromaticity", "fcr", "ncpr", "kappa", "omega",
            "disorder_f", "scd",
        ]
        for k in scalar_keys:
            v = data.get(k)
            if v is not None:
                w.writerow([k, v])
        text = buf.getvalue()
        _write_or_print(text, out_path)

    elif fmt == "html":
        parts = [
            "<html><head><meta charset='utf-8'>"
            "<style>body{font-family:sans-serif;max-width:960px;margin:auto;padding:20px}"
            "h2{color:#4361ee}table{border-collapse:collapse;width:100%}"
            "th,td{border:1px solid #e2e8f0;padding:6px 10px}</style></head><body>",
            f"<h1>BEER Analysis — {len(seq)} aa</h1>",
        ]
        for sec_name, sec_html in sections.items():
            parts.append(f"<section id='{sec_name.replace(' ','_')}'>{sec_html}</section>")
        parts.append("</body></html>")
        text = "".join(parts)
        target = out_path or "beer_report.html"
        Path(target).write_text(text, encoding="utf-8")
        print(f"[beer] HTML report written → {target}", file=sys.stderr)
        return

    print(f"[beer] Done.", file=sys.stderr)


def _write_or_print(text: str, path: str | None) -> None:
    import sys
    if path:
        from pathlib import Path
        Path(path).write_text(text, encoding="utf-8")
        print(f"[beer] Output written → {path}", file=sys.stderr)
    else:
        sys.stdout.write(text)


# ---------------------------------------------------------------------------
# GUI entry point
# ---------------------------------------------------------------------------

def _force_qt_file_dialogs() -> None:
    """Force Qt's own file dialogs instead of the OS-native ones.

    On macOS with Qt 6.11 the native Save/Open panels can become non-functional
    (folder navigation and the file-format selector stop responding). Wrap the
    static QFileDialog helpers so every call uses the reliable non-native Qt
    dialog (one call already did this individually). Idempotent.
    """
    from PySide6.QtWidgets import QFileDialog
    if getattr(QFileDialog, "_beer_nonnative", False):
        return
    _opt = QFileDialog.Option.DontUseNativeDialog
    for _name in ("getSaveFileName", "getOpenFileName",
                  "getOpenFileNames", "getExistingDirectory"):
        _orig = getattr(QFileDialog, _name)

        def _make(orig):
            def _wrapped(*args, **kwargs):
                kwargs.setdefault("options", _opt)
                return orig(*args, **kwargs)
            return _wrapped

        setattr(QFileDialog, _name, staticmethod(_make(_orig)))
    QFileDialog._beer_nonnative = True


def main() -> None:
    _harden_native_runtime()   # must precede any torch/esm import
    _setup_logging()
    _log = logging.getLogger("beer.app")

    # Dispatch to CLI if any arguments are provided
    raw_args = sys.argv[1:]
    if raw_args and raw_args[0] in ("analyze", "version", "--version", "-V"):
        if raw_args[0] in ("--version", "-V"):
            import beer as _bp; print(f"BEER v{_bp.__version__}"); return
        _run_cli(raw_args)
        return

    _log.info("BEER starting up")

    # Fail early with a clear message if there is no display.
    # Importing main_window sets matplotlib.use("QtAgg"); importing it on a
    # headless system raises ImportError, which we surface here with a clear message.
    try:
        import matplotlib
        matplotlib.use("QtAgg")
    except ImportError as _e:
        sys.exit(
            f"BEER requires a display to run the GUI.\n"
            f"On a headless server use X11 forwarding (ssh -X) or VNC.\n"
            f"Detail: {_e}"
        )

    warnings.filterwarnings("ignore", message="Setting the 'color' property will override")
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    warnings.filterwarnings("ignore", message=".*tight_layout.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*constrained_layout.*", category=UserWarning)
    # torch registers an internal POSIX semaphore at import time; Python's resource
    # tracker sees it as leaked because torch's atexit cleanup runs after the tracker.
    # Suppress the noise — the semaphore is reclaimed by the OS at process exit.
    warnings.filterwarnings(
        "ignore",
        message=".*resource_tracker.*leaked semaphore.*",
        category=UserWarning,
    )

    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QIcon
    from PySide6.QtCore import QThread, Signal as _Signal
    import importlib.resources
    from beer.gui.main_window import ProteinAnalyzerGUI
    import beer

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("BEER")
    app.setApplicationVersion(beer.__version__)
    _force_qt_file_dialogs()

    _logo_path = str(importlib.resources.files("beer").joinpath("beer.png"))
    if Path(_logo_path).exists():
        app.setWindowIcon(QIcon(_logo_path))

    _install_exception_hook(app)

    # Show window immediately; load esm/torch in background to avoid blocking
    # the GUI thread with the torch initialisation (can take 2-5 s on first run).
    window = ProteinAnalyzerGUI(embedder=None)
    window.show()
    _log.info("BEER v%s window shown", beer.__version__)

    class _EmbedderLoader(QThread):
        ready = _Signal(object)

        def run(self):
            try:
                from beer.embeddings import ESMC_AVAILABLE, get_embedder
                if ESMC_AVAILABLE:
                    embedder = get_embedder("esmc_600m")
                    self.ready.emit(embedder)
                else:
                    _log.warning("ESMC not available (esm / torch not installed)")
                    self.ready.emit(None)
            except Exception as _e:
                _log.warning("Embedder load failed: %s", _e)
                self.ready.emit(None)

    _loader = _EmbedderLoader()
    _loader.ready.connect(window.set_embedder)
    _loader.start()

    _ret = app.exec()
    # Release torch model and stop loader thread before resource_tracker runs.
    try:
        _loader.quit()
        if not _loader.wait(1500):
            _loader.terminate()
        # Release ESMC model to unregister its POSIX semaphore cleanly.
        if hasattr(window, "_embedder") and window._embedder is not None:
            try:
                emb = window._embedder
                window._embedder = None
                if hasattr(emb, "release"):
                    emb.release()
                del emb
            except Exception:
                pass
        del _loader
        del window
        import gc
        gc.collect()
    except Exception:
        pass
    sys.exit(_ret)


if __name__ == "__main__":
    main()
