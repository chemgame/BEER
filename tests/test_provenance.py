"""Provenance stamps: format-safe figure metadata + text/JSON headers."""
import matplotlib
matplotlib.use("Agg")
from io import BytesIO
from matplotlib.figure import Figure

from beer.io.provenance import figure_metadata, text_header, provenance_record


def test_figure_metadata_safe_for_all_formats():
    # savefig with the per-format metadata must never raise.
    for ext in ("png", "pdf", "svg"):
        fig = Figure()
        fig.add_subplot(111).plot([1, 2, 3])
        buf = BytesIO()
        fig.savefig(buf, format=ext, metadata=figure_metadata(ext))
        assert buf.getvalue()


def test_text_header_and_record():
    h = text_header("# ", title="Disorder track")
    assert h.startswith("# Disorder track")
    assert "BEER v" in h and "github.com/chemgame/BEER" in h
    rec = provenance_record()
    assert rec["software"].startswith("BEER v") and "generated" in rec
