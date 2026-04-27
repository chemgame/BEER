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
    p_ana.add_argument("--no-esm2", action="store_true",
                       help="Skip ESM2 BiLSTM predictions (faster; no GPU/ESM2 required).")
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
        acc = args.accession.strip()
        url = f"https://www.uniprot.org/uniprot/{acc}.fasta"
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
    if not args.no_esm2:
        try:
            from beer.embeddings import ESM2_AVAILABLE, get_embedder
            if ESM2_AVAILABLE:
                embedder = get_embedder("esm2_t33_650M_UR50D")
        except Exception:
            pass

    # --- Run analysis ---
    from beer.analysis.core import analyze_sequence
    data = analyze_sequence(
        seq,
        embedder=embedder,
        accent_color=args.accent,
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

def main() -> None:
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

    embedder = get_embedder("esm2_t33_650M_UR50D") if ESM2_AVAILABLE else None

    window = ProteinAnalyzerGUI(embedder=embedder)
    window.show()
    _log.info("BEER v%s window shown", beer.__version__)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
