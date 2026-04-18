"""Shared HTML/CSS styling for all BEER HTML reports."""

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

REPORT_CSS_DARK: str = """
body {
    font-family: Arial, 'Helvetica Neue';
    font-size: 11pt;
    color: #e2e8f0;
    margin: 0;
    padding: 0;
    line-height: 1.6;
    background-color: #16213e;
}
h1 { font-size: 18pt; color: #e2e8f0; border-bottom: 2px solid #4cc9f0; padding-bottom: 6px; margin-top: 20px; }
h2 { font-size: 13pt; color: #4cc9f0; margin-top: 18px; margin-bottom: 8px; font-weight: 600; }
h3 { font-size: 11pt; color: #4cc9f0; margin-top: 14px; margin-bottom: 4px; font-weight: 600; }
table {
    border-collapse: collapse;
    width: 100%;
    margin: 10px 0 16px 0;
    font-size: 10pt;
}
th {
    background-color: #0f3460;
    color: #4cc9f0;
    padding: 7px 12px;
    text-align: left;
    font-weight: 600;
}
td {
    padding: 6px 12px;
    border-bottom: 1px solid #2d3561;
    color: #e2e8f0;
}
tr:nth-child(even) td { background-color: #1e2a4a; }
tr:hover td { background-color: #1a3a5c; }
p.note {
    font-size: 9pt;
    color: #94a3b8;
    font-style: italic;
    margin: 4px 0 12px 0;
}
pre.sequence {
    font-family: 'Courier New', Courier, monospace;
    font-size: 10pt;
    background: #16213e;
    border: 1px solid #2d3561;
    border-radius: 4px;
    padding: 10px 14px;
    line-height: 1.8;
    color: #e2e8f0;
    white-space: pre;
}
"""

# Alias used by legacy analysis modules that reference _REPORT_CSS
_REPORT_CSS: str = REPORT_CSS


def get_report_css(dark: bool = False) -> str:
    """Return the appropriate report CSS for the current theme."""
    return REPORT_CSS_DARK if dark else REPORT_CSS


def make_style_tag(accent: str = "#4361ee", dark: bool = False) -> str:
    """Return a ``<style>`` HTML tag with the report CSS, accent colour substituted."""
    css = get_report_css(dark).replace("#4361ee" if not dark else "#4cc9f0", accent)
    return f"<style>{css}</style>"
