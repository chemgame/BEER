"""HTML report section generators for all BEER analysis modules.

Each ``format_*_report`` function produces a self-contained HTML fragment
(with embedded ``<style>`` block) suitable for direct insertion into a Qt
WebView or any HTML renderer.

All functions accept a ``seq`` (protein sequence) and an optional
``accent`` colour string (default ``"#4361ee"``).

Functions are imported from their canonical analysis sub-modules so that
this module serves as a single import surface for callers that need all
report formatters at once.
"""
from __future__ import annotations

from beer.reports.css import make_style_tag  # noqa: F401 — re-exported for convenience
from beer.analysis.aggregation import format_aggregation_report
from beer.analysis.ptm import format_ptm_report
from beer.analysis.signal_peptide import format_signal_report
from beer.analysis.amphipathic import format_amphipathic_report
from beer.analysis.scd import format_scd_report
from beer.analysis.rnabinding import format_rbp_report
from beer.analysis.tandem_repeats import format_repeats_report, format_tandem_repeats_report

__all__ = [
    "make_style_tag",
    "format_aggregation_report",
    "format_ptm_report",
    "format_signal_report",
    "format_amphipathic_report",
    "format_scd_report",
    "format_rbp_report",
    "format_repeats_report",
    "format_tandem_repeats_report",
]
