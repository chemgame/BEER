"""BEER export utilities — PDF report generation (PySide6)."""
from __future__ import annotations

from beer.reports.css import REPORT_CSS
from beer.constants import REPORT_SECTIONS
from beer.utils import format_sequence_block


class ExportTools:
    @staticmethod
    def _generate_full_html(analysis_data, seq_name=""):
        """Generate HTML for the PDF report (text/tables only)."""
        if not analysis_data or "report_sections" not in analysis_data:
            return "<p>No analysis data available.</p>"

        import re as _re

        from html import escape as _he
        header_name = _he(seq_name) if seq_name else "Protein Sequence"
        seq = analysis_data.get("seq", "")
        seq_block = format_sequence_block(seq, name=seq_name)
        seq_block_html = seq_block.replace("&", "&amp;").replace("<", "&lt;")

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
{REPORT_CSS}
@page {{ margin: 18mm 20mm; }}
.page-break {{ page-break-after: always; }}
.seq-block {{
    font-family: 'Courier New', monospace;
    font-size: 9.5pt;
    background: #f8f9fd;
    border: 1px solid #e8eaf0;
    border-radius: 4px;
    padding: 10px 14px;
    line-height: 2.0;
    color: #1a1a2e;
    white-space: pre;
    margin-bottom: 14px;
    overflow-x: auto;
}}
</style>
</head><body>
<h1>BEER Analysis Report</h1>
<p style="color:#718096;font-size:10pt;">
  Sequence: <strong>{header_name}</strong>
  &nbsp;&bull;&nbsp; Length: <strong>{len(seq)} aa</strong>
</p>
<h2 style="margin-top:16px;">Sequence</h2>
<div class="seq-block">{seq_block_html}</div>
<div class="page-break"></div>
"""
        for sec in REPORT_SECTIONS:
            content = analysis_data["report_sections"].get(sec, "")
            content = _re.sub(r"<style>[^<]*</style>", "", content, flags=_re.DOTALL)
            html += content + "\n"

        html += "</body></html>"
        return html

    @staticmethod
    def export_pdf(analysis_data, file_name, parent, seq_name=""):
        from PySide6.QtPrintSupport import QPrinter
        from PySide6.QtWidgets import QTextBrowser, QMessageBox

        try:
            printer = QPrinter(QPrinter.PrinterMode.HighResolution)
            printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
            printer.setOutputFileName(file_name)
            browser = QTextBrowser()
            browser.setHtml(ExportTools._generate_full_html(analysis_data, seq_name))
            browser.document().print_(printer)
            QMessageBox.information(parent, "Success", f"PDF exported to {file_name}")
        except Exception as e:
            QMessageBox.warning(parent, "Export Failed", f"PDF export error: {e}")


__all__ = ["ExportTools"]
