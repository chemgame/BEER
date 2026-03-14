"""BEER export utilities: PDF, CSV, JSON."""
import csv, json, os
from PyQt5.QtWidgets import QMessageBox, QTextBrowser
from PyQt5.QtPrintSupport import QPrinter


class ExportTools:
    @staticmethod
    def _generate_full_html(analysis_data, seq_name="", report_css=""):
        if not analysis_data or "report_sections" not in analysis_data:
            return "<p>No analysis data available.</p>"
        from beer.io.format_helpers import format_sequence_block
        header_name = seq_name or "Protein Sequence"
        seq = analysis_data.get("seq", "")
        seq_block = format_sequence_block(seq, name=seq_name)
        seq_block_html = seq_block.replace("&", "&amp;").replace("<", "&lt;")
        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
{report_css}
@page {{ margin: 18mm 20mm; }}
.page-break {{ page-break-after: always; }}
.seq-block {{
    font-family: 'Courier New', monospace; font-size: 9.5pt;
    background: #f8f9fd; border: 1px solid #e8eaf0; border-radius: 4px;
    padding: 10px 14px; line-height: 2.0; color: #1a1a2e; white-space: pre;
    margin-bottom: 14px; overflow-x: auto;
}}
</style></head><body>
<h1>BEER Analysis Report</h1>
<p style="color:#718096;font-size:10pt;">
  Sequence: <strong>{header_name}</strong>
  &nbsp;&bull;&nbsp; Length: <strong>{len(seq)} aa</strong>
</p>
<h2 style="margin-top:16px;">Sequence</h2>
<div class="seq-block">{seq_block_html}</div>
<div class="page-break"></div>
"""
        import re as _re
        for sec, content in analysis_data["report_sections"].items():
            content = _re.sub(r"<style>[^<]*</style>", "", content, flags=_re.DOTALL)
            html += content + "\n"
        html += "</body></html>"
        return html

    @staticmethod
    def export_pdf(analysis_data, file_name, parent, seq_name="", report_css=""):
        try:
            printer = QPrinter(QPrinter.HighResolution)
            printer.setOutputFormat(QPrinter.PdfFormat)
            printer.setOutputFileName(file_name)
            browser = QTextBrowser()
            browser.setHtml(ExportTools._generate_full_html(analysis_data, seq_name, report_css))
            browser.document().print_(printer)
            QMessageBox.information(parent, "Success", f"PDF exported to {file_name}")
        except Exception as e:
            QMessageBox.warning(parent, "Export Failed", f"PDF export error: {e}")
