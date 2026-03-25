"""Shared HTML/CSS styling for all BEER HTML reports."""

# Comprehensive merged CSS (union of _REPORT_CSS and REPORT_CSS from beer.py,
# preferring the more complete REPORT_CSS version which includes h1, pre.sequence, etc.)
REPORT_CSS: str = """
body {
    font-family: Arial, 'Helvetica Neue';
    font-size: 11pt;
    color: #1a1a2e;
    margin: 0;
    padding: 0;
    line-height: 1.6;
}
h1 { font-size: 18pt; color: #1a1a2e; border-bottom: 2px solid #4361ee; padding-bottom: 6px; margin-top: 20px; }
h2 { font-size: 13pt; color: #4361ee; margin-top: 18px; margin-bottom: 8px; font-weight: 600; }
h3 { font-size: 11pt; color: #4361ee; margin-top: 14px; margin-bottom: 4px; font-weight: 600; }
table {
    border-collapse: collapse;
    width: 100%;
    margin: 10px 0 16px 0;
    font-size: 10pt;
}
th {
    background-color: #4361ee;
    color: #ffffff;
    padding: 7px 12px;
    text-align: left;
    font-weight: 600;
}
td {
    padding: 6px 12px;
    border-bottom: 1px solid #e8eaf0;
    color: #2d3748;
}
tr:nth-child(even) td { background-color: #f8f9fd; }
tr:hover td { background-color: #eef0f8; }
p.note {
    font-size: 9pt;
    color: #718096;
    font-style: italic;
    margin: 4px 0 12px 0;
}
pre.sequence {
    font-family: 'Courier New', Courier, monospace;
    font-size: 10pt;
    background: #f8f9fd;
    border: 1px solid #e8eaf0;
    border-radius: 4px;
    padding: 10px 14px;
    line-height: 1.8;
    color: #1a1a2e;
    white-space: pre;
}
"""

# Alias used by legacy analysis modules that reference _REPORT_CSS
_REPORT_CSS: str = REPORT_CSS


def make_style_tag(accent: str = "#4361ee") -> str:
    """Return a ``<style>`` HTML tag with the report CSS, accent colour substituted."""
    css = REPORT_CSS.replace("#4361ee", accent)
    return f"<style>{css}</style>"
