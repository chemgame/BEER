"""BEER GUI themes — CSS stylesheets for light and dark modes."""
from __future__ import annotations

from beer.constants import NAMED_COLORS  # re-export
from beer.constants import NAMED_COLORMAPS  # re-export

LIGHT_THEME_CSS = """
 QWidget {
     background-color: #f5f6fa;
     color: #1a1a2e;
     font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
     font-size: 13px;
 }
 QMainWindow { background-color: #f5f6fa; }
 QLineEdit, QTextEdit, QTextBrowser {
     background-color: #ffffff;
     color: #1a1a2e;
     border: 1px solid #d0d4e0;
     border-radius: 4px;
     padding: 4px 6px;
     selection-background-color: #4361ee;
 }
 QTextBrowser > QWidget, QTextEdit > QWidget {
     background-color: #ffffff;
     color: #1a1a2e;
 }
 QPushButton {
     background-color: #4361ee;
     color: #ffffff;
     border: none;
     border-radius: 4px;
     padding: 6px 14px;
     font-weight: 600;
     letter-spacing: 0.3px;
 }
 QPushButton:hover { background-color: #3451d1; }
 QPushButton:pressed { background-color: #2940b8; }
 QPushButton:disabled { background-color: #b0b8cc; color: #f0f0f0; }
 QTabWidget::pane { border: 1px solid #d0d4e0; border-radius: 4px; background: #ffffff; }
 QTabBar::tab {
     background: #e8eaf0;
     color: #4a5568;
     padding: 8px 16px;
     border-top-left-radius: 4px;
     border-top-right-radius: 4px;
     margin-right: 2px;
     font-weight: 500;
 }
 QTabBar::tab:selected { background: #4361ee; color: #ffffff; }
 QTabBar::tab:hover:!selected { background: #d0d4e8; }
 QTableWidget {
     background-color: #ffffff;
     gridline-color: #e8eaf0;
     border: 1px solid #d0d4e0;
     border-radius: 4px;
     alternate-background-color: #f8f9fd;
 }
 QHeaderView::section {
     background-color: #4361ee;
     color: #ffffff;
     padding: 6px 10px;
     border: none;
     font-weight: 600;
 }
 QComboBox {
     background-color: #ffffff;
     border: 1px solid #d0d4e0;
     border-radius: 4px;
     padding: 4px 8px;
 }
 QComboBox::drop-down { border: none; }
 QLabel { color: #2d3748; font-weight: 500; }
 QCheckBox { color: #2d3748; spacing: 6px; }
 QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #d0d4e0; border-radius: 4px; }
 QCheckBox::indicator:checked { background-color: #4361ee; border-color: #4361ee; }
 QScrollBar:vertical { background: #f0f0f5; width: 10px; border-radius: 5px; }
 QScrollBar::handle:vertical { background: #c0c4d0; border-radius: 5px; min-height: 30px; }
 QStatusBar { background-color: #4361ee; color: #ffffff; font-size: 12px; }
 QToolBar { background-color: #eef0f8; border: 1px solid #d0d4e0; border-radius: 4px; spacing: 4px; padding: 4px; }
 QToolBar QToolButton { background-color: #ffffff; border: 1px solid #d0d4e0; border-radius: 4px; padding: 3px; color: #2d3748; }
 QToolBar QToolButton:hover { background-color: #e0e4f4; border-color: #4361ee; }
 QToolBar QToolButton:pressed { background-color: #c8d0ec; }
 /* --- Tooltips --- */
 QToolTip {
     background-color: #1e2640;
     color: #f0f4ff;
     border: 1px solid #4361ee;
     border-radius: 4px;
     padding: 5px 10px;
     font-size: 10pt;
     font-weight: normal;
 }
 /* --- Left navigation sidebar --- */
 QListWidget#nav_bar {
     background-color: #e4e8f4;
     border: none;
     border-right: 1px solid #c8cede;
     padding: 8px 0;
     font-size: 12px;
     font-weight: 500;
     outline: 0;
 }
 QListWidget#nav_bar::item {
     padding: 12px 12px;
     color: #4a5568;
     border-left: 3px solid transparent;
 }
 QListWidget#nav_bar::item:selected {
     background-color: #dce3f8;
     color: #4361ee;
     border-left: 3px solid #4361ee;
     font-weight: 700;
 }
 QListWidget#nav_bar::item:hover:!selected { background-color: #d4d9ec; }
 QFrame#nav_sep { color: #c8cede; max-width: 1px; }
 /* --- Graph tree & report nav --- */
 QTreeWidget#graph_tree, QTreeWidget#report_nav {
     background-color: #f0f2fa;
     border: none;
     border-right: 1px solid #d0d4e0;
     font-size: 12px;
     outline: 0;
 }
 QTreeWidget#graph_tree::item { padding: 6px 8px; color: #4a5568; }
 QTreeWidget#graph_tree::item:selected { background-color: #4361ee; color: #ffffff; border-radius: 4px; }
 QTreeWidget#graph_tree::branch { background-color: #f0f2fa; }
 QTreeWidget#report_nav::item { padding: 6px 8px; color: #4a5568; background-color: #f0f2fa; }

 QTreeWidget#report_nav::item:selected { background-color: #4361ee; color: #ffffff; border-radius: 4px; }
 QTreeWidget#report_nav::item:hover:!selected { background-color: #dce3f8; }
 QTreeWidget#report_nav::branch { background-color: #f0f2fa; }
 /* --- Fetch / chip buttons --- */
 QPushButton#chip_btn {
     background: transparent;
     color: #4361ee;
     border: 1px solid #b0bae8;
     border-radius: 99px;
     padding: 4px 12px;
     font-size: 12px;
     min-height: 28px;
     font-weight: 600;
 }
 QPushButton#chip_btn:hover:!disabled { background: #e8eeff; border-color: #4361ee; }
 QPushButton#chip_btn:pressed:!disabled { background: #d0d8f8; }
 QPushButton#chip_btn:disabled { color: #b8bdd4; border-color: #dcdee8; background: transparent; }
 QPushButton#chip_btn[chip_state="fetched"] {
     background: #e6f9f0;
     color: #1a7a4a;
     border: 1px solid #43aa8b;
     font-weight: 700;
 }
 QPushButton#chip_btn[chip_state="fetched"]:hover { background: #cef2e3; border-color: #2e8b57; }
 QPushButton#chip_btn[chip_state="fetched"]:disabled { background: #d4f3e8; color: #3a8a5a; }
 QPushButton#chip_btn[chip_state="loading"] {
     background: #fff8e6; color: #b45309;
     border: 1px solid #f59e0b; font-weight: 700;
 }
 QPushButton#chip_btn[chip_state="loading"]:hover { background: #fef3c7; border-color: #d97706; }
 QPushButton#chip_btn[chip_state="error"] {
     background: #fff0f0; color: #c0392b;
     border: 1px solid #e74c3c; font-weight: 700;
 }
 QPushButton#chip_btn[chip_state="error"]:hover { background: #fde8e8; border-color: #c0392b; }
 /* --- Section group labels (fetch bar headers) --- */
 QLabel#group_lbl { color: #8892b0; font-size: 11px; font-weight: 600; }
 /* --- Disabled nav items (gated tabs) --- */
 QListWidget#nav_bar::item:disabled { color: #a0a8bc; font-style: italic; }
 /* --- Settings section headers --- */
 QLabel#section_header {
     font-size: 11pt;
     font-weight: 700;
     color: #4361ee;
     border-bottom: 1px solid #d0d4e0;
     padding-bottom: 4px;
     margin-top: 8px;
 }
 /* --- Welcome banner --- */
 QFrame#welcome_banner { background: #eef1fc; border: 1px solid #c8d0ec; border-radius: 4px; padding: 4px; }
 QLabel#welcome_lbl { color: #2d3748; }
 QLabel#welcome_lbl a { color: #4361ee; }
 /* --- Accent labels (sequence panel headings) --- */
 QLabel#accent_lbl { color: #4361ee; font-weight: 600; }
 /* --- Status / hint labels --- */
 QLabel#status_lbl { color: #718096; font-style: italic; }
 QLabel#status_lbl[status_state="idle"]    { color: #718096; font-style: italic; font-weight: normal; }
 QLabel#status_lbl[status_state="success"] { color: #1a7a4a; font-style: normal; font-weight: 600; }
 QLabel#status_lbl[status_state="error"]   { color: #c0392b; font-style: normal; font-weight: normal; }
 /* --- Disorder-method status indicator --- */
 QLabel#esm2_lbl { font-size: 11px; font-weight: 600; }
 QLabel#esm2_lbl[esm2_state="ready"]        { color: #4361ee; }
 QLabel#esm2_lbl[esm2_state="active"]       { color: #1a7a4a; }
 QLabel#esm2_lbl[esm2_state="metapredict"]  { color: #b07d00; }
 QLabel#esm2_lbl[esm2_state="classical"]    { color: #8b4513; }
 QLabel#esm2_lbl[esm2_state="missing"]      { color: #718096; }
 /* --- Protein info bar --- */
 QTextBrowser#info_bar {
     background: #f0f4ff;
     border: 1px solid #c8d0ec;
     border-radius: 4px;
     padding: 4px 8px;
     font-size: 10pt;
     color: #2d3748;
     font-weight: normal;
 }
 /* --- Info dialog text browser --- */
 QTextBrowser#info_dialog {
     background: #f8f9ff;
     border: none;
     padding: 8px;
     font-family: 'Menlo', 'Consolas', monospace;
     font-size: 10pt;
     color: #1a1a2e;
 }
 /* --- Info (i) button --- */
 QToolButton#info_btn {
     font-size: 11pt;
     border: none;
     background: transparent;
     color: #8899b0;
 }
 QToolButton#info_btn:hover { color: #4361ee; background: transparent; }
 /* --- Delete / danger buttons --- */
 QPushButton#delete_btn { background-color: #e63946; color: #ffffff; }
 QPushButton#delete_btn:hover { background-color: #c1121f; }
 QPushButton#delete_btn:pressed { background-color: #a00e17; }
 QPushButton#danger_btn {
     background: transparent;
     color: #c0392b;
     border: 1px solid #c0392b;
     border-radius: 4px;
     padding: 6px 14px;
     font-weight: 600;
 }
 QPushButton#danger_btn:hover { background: #fde8e8; }
 /* --- Vertical separator in fetch bar --- */
 QFrame#v_sep { color: #d0d4e0; }
 /* --- Muted placeholder / info labels --- */
 QLabel#placeholder_lbl { color: #718096; font-style: italic; }
 /* --- Help button --- */
 QToolButton#help_btn { font-weight: bold; border-radius: 99px; }
 /* --- PDB xref section label --- */
 QLabel#pdb_xref_lbl { color: #4a5568; font-size: 9pt; font-weight: 600; }
 /* --- Chain selector label --- */
 QLabel#chain_lbl { font-weight: 600; }
"""

DARK_THEME_CSS = """
 QWidget {
     background-color: #1a1a2e;
     color: #e2e8f0;
     font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
     font-size: 13px;
 }
 QMainWindow { background-color: #1a1a2e; }
 QLineEdit, QTextEdit, QTextBrowser {
     background-color: #16213e;
     color: #e2e8f0;
     border: 1px solid #2d3561;
     border-radius: 4px;
     padding: 4px 6px;
     selection-background-color: #7b9cff;
 }
 QTextBrowser > QWidget, QTextEdit > QWidget {
     background-color: #16213e;
     color: #e2e8f0;
 }
 QPushButton {
     background-color: #7b9cff;
     color: #1a1a2e;
     border: none;
     border-radius: 4px;
     padding: 6px 14px;
     font-weight: 600;
     letter-spacing: 0.3px;
 }
 QPushButton:hover { background-color: #6b8eff; }
 QPushButton:pressed { background-color: #5575ee; }
 QPushButton:disabled { background-color: #2d3561; color: #6b7280; }
 QTabWidget::pane { border: 1px solid #2d3561; border-radius: 4px; background: #16213e; }
 QTabBar::tab {
     background: #0f3460;
     color: #94a3b8;
     padding: 8px 16px;
     border-top-left-radius: 4px;
     border-top-right-radius: 4px;
     margin-right: 2px;
     font-weight: 500;
 }
 QTabBar::tab:selected { background: #7b9cff; color: #1a1a2e; }
 QTabBar::tab:hover:!selected { background: #1a3a5c; }
 QTableWidget {
     background-color: #16213e;
     gridline-color: #2d3561;
     border: 1px solid #2d3561;
     border-radius: 4px;
     alternate-background-color: #1e2a4a;
 }
 QHeaderView::section {
     background-color: #0f3460;
     color: #7b9cff;
     padding: 6px 10px;
     border: none;
     font-weight: 600;
 }
 QComboBox {
     background-color: #16213e;
     color: #e2e8f0;
     border: 1px solid #2d3561;
     border-radius: 4px;
     padding: 4px 8px;
 }
 QComboBox::drop-down { border: none; }
 QLabel { color: #94a3b8; font-weight: 500; }
 QCheckBox { color: #94a3b8; spacing: 6px; }
 QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #2d3561; border-radius: 4px; }
 QCheckBox::indicator:checked { background-color: #7b9cff; border-color: #7b9cff; }
 QScrollBar:vertical { background: #16213e; width: 10px; border-radius: 5px; }
 QScrollBar::handle:vertical { background: #2d3561; border-radius: 5px; min-height: 30px; }
 QStatusBar { background-color: #0f3460; color: #7b9cff; font-size: 12px; }
 /* --- Tooltips --- */
 QToolTip {
     background-color: #0f3460;
     color: #e2e8f0;
     border: 1px solid #7b9cff;
     border-radius: 4px;
     padding: 5px 10px;
     font-size: 10pt;
     font-weight: normal;
 }
 /* --- Left navigation sidebar --- */
 QListWidget#nav_bar {
     background-color: #0f3460;
     border: none;
     border-right: 1px solid #1a3a5c;
     padding: 8px 0;
     font-size: 12px;
     font-weight: 500;
     outline: 0;
 }
 QListWidget#nav_bar::item {
     padding: 12px 12px;
     color: #94a3b8;
     border-left: 3px solid transparent;
 }
 QListWidget#nav_bar::item:selected {
     background-color: #1a3a5c;
     color: #7b9cff;
     border-left: 3px solid #7b9cff;
     font-weight: 700;
 }
 QListWidget#nav_bar::item:hover:!selected { background-color: #1a3a5c; color: #e2e8f0; }
 QFrame#nav_sep { color: #1a3a5c; max-width: 1px; }
 /* --- Graph tree & report nav --- */
 QTreeWidget#graph_tree, QTreeWidget#report_nav {
     background-color: #16213e;
     border: none;
     border-right: 1px solid #2d3561;
     font-size: 12px;
     outline: 0;
 }
 QTreeWidget#graph_tree::item { padding: 6px 8px; color: #94a3b8; }
 QTreeWidget#graph_tree::item:selected { background-color: #7b9cff; color: #1a1a2e; border-radius: 4px; }
 QTreeWidget#graph_tree::branch { background-color: #16213e; }
 QTreeWidget#report_nav::item { padding: 6px 8px; color: #94a3b8; background-color: #16213e; }
 QTreeWidget#report_nav::item:selected { background-color: #7b9cff; color: #1a1a2e; border-radius: 4px; }
 QTreeWidget#report_nav::item:hover:!selected { background-color: #1a3a5c; }
 QTreeWidget#report_nav::branch { background-color: #16213e; }
 /* --- Fetch / chip buttons --- */
 QPushButton#chip_btn {
     background: transparent;
     color: #7b9cff;
     border: 1px solid #2d4a6e;
     border-radius: 99px;
     padding: 4px 12px;
     font-size: 12px;
     min-height: 28px;
     font-weight: 600;
 }
 QPushButton#chip_btn:hover:!disabled { background: #1a3a5c; border-color: #7b9cff; }
 QPushButton#chip_btn:pressed:!disabled { background: #0f3460; }
 QPushButton#chip_btn:disabled { color: #2d3a5a; border-color: #1a2540; background: transparent; }
 QPushButton#chip_btn[chip_state="fetched"] {
     background: #0d3325;
     color: #43aa8b;
     border: 1px solid #2e6651;
     font-weight: 700;
 }
 QPushButton#chip_btn[chip_state="fetched"]:hover { background: #0f3d2c; border-color: #43aa8b; }
 QPushButton#chip_btn[chip_state="fetched"]:disabled { background: #0d2a1e; color: #2e7a5a; }
 QPushButton#chip_btn[chip_state="loading"] {
     background: #2a1f00; color: #fbbf24;
     border: 1px solid #d97706; font-weight: 700;
 }
 QPushButton#chip_btn[chip_state="loading"]:hover { background: #3a2a00; border-color: #fbbf24; }
 QPushButton#chip_btn[chip_state="error"] {
     background: #2a0d0d; color: #f87171;
     border: 1px solid #ef4444; font-weight: 700;
 }
 QPushButton#chip_btn[chip_state="error"]:hover { background: #3a1010; border-color: #f87171; }
 /* --- Section group labels (fetch bar headers) --- */
 QLabel#group_lbl { color: #4a5a7a; font-size: 11px; font-weight: 600; }
 /* --- Disabled nav items (gated tabs) --- */
 QListWidget#nav_bar::item:disabled { color: #3a4a6a; font-style: italic; }
 /* --- Settings section headers --- */
 QLabel#section_header {
     font-size: 11pt;
     font-weight: 700;
     color: #7b9cff;
     border-bottom: 1px solid #2d3561;
     padding-bottom: 4px;
     margin-top: 8px;
 }
 /* --- Welcome banner --- */
 QFrame#welcome_banner { background: #1a2240; border: 1px solid #2d3a6a; border-radius: 4px; padding: 4px; }
 QLabel#welcome_lbl { color: #c8d0ec; }
 QLabel#welcome_lbl a { color: #7b9cff; }
 /* --- Accent labels --- */
 QLabel#accent_lbl { color: #7b9cff; font-weight: 600; }
 /* --- Status / hint labels --- */
 QLabel#status_lbl { color: #5a6787; font-style: italic; }
 QLabel#status_lbl[status_state="idle"]    { color: #5a6787; font-style: italic; font-weight: normal; }
 QLabel#status_lbl[status_state="success"] { color: #43aa8b; font-style: normal; font-weight: 600; }
 QLabel#status_lbl[status_state="error"]   { color: #f87171; font-style: normal; font-weight: normal; }
 /* --- Disorder-method status indicator --- */
 QLabel#esm2_lbl { font-size: 11px; font-weight: 600; }
 QLabel#esm2_lbl[esm2_state="ready"]        { color: #7b9cff; }
 QLabel#esm2_lbl[esm2_state="active"]       { color: #43aa8b; }
 QLabel#esm2_lbl[esm2_state="metapredict"]  { color: #f4a261; }
 QLabel#esm2_lbl[esm2_state="classical"]    { color: #e9c46a; }
 QLabel#esm2_lbl[esm2_state="missing"]      { color: #5a6787; }
 /* --- Protein info bar --- */
 QTextBrowser#info_bar {
     background: #16213e;
     border: 1px solid #2d3561;
     border-radius: 4px;
     padding: 4px 8px;
     font-size: 10pt;
     color: #94a3b8;
     font-weight: normal;
 }
 /* --- Info dialog text browser --- */
 QTextBrowser#info_dialog {
     background: #16213e;
     border: none;
     padding: 8px;
     font-family: 'Menlo', 'Consolas', monospace;
     font-size: 10pt;
     color: #e2e8f0;
 }
 /* --- Info (i) button --- */
 QToolButton#info_btn {
     font-size: 11pt;
     border: none;
     background: transparent;
     color: #4a5a7a;
 }
 QToolButton#info_btn:hover { color: #7b9cff; background: transparent; }
 /* --- Delete / danger buttons --- */
 QPushButton#delete_btn { background-color: #c1121f; color: #ffffff; }
 QPushButton#delete_btn:hover { background-color: #a50f1a; }
 QPushButton#delete_btn:pressed { background-color: #8a0c15; }
 QPushButton#danger_btn {
     background: transparent;
     color: #f87171;
     border: 1px solid #f87171;
     border-radius: 4px;
     padding: 6px 14px;
     font-weight: 600;
 }
 QPushButton#danger_btn:hover { background: #2a0d0d; }
 /* --- Vertical separator in fetch bar --- */
 QFrame#v_sep { color: #2d3561; }
 /* --- Muted placeholder / info labels --- */
 QLabel#placeholder_lbl { color: #6a7a9a; font-style: italic; }
 /* --- Help button --- */
 QToolButton#help_btn { font-weight: bold; border-radius: 99px; }
 /* --- PDB xref section label --- */
 QLabel#pdb_xref_lbl { color: #94a3b8; font-size: 9pt; font-weight: 600; }
 /* --- Chain selector label --- */
 QLabel#chain_lbl { font-weight: 600; }
"""

# ── Structure viewer control panel CSS ───────────────────────────────────────
# Applied to struct_ctrl_scroll (QTabWidget) so selectors scope to its children.
# Kept separate from the main QSS because macOS native style overrides QTabBar
# colours when set via the application stylesheet; a widget-level setStyleSheet
# on the tab bar itself is the only reliable cross-platform workaround.

STRUCT_PANEL_CSS_LIGHT = """
    QScrollArea { border: none; background: transparent; }
    QWidget#structCtrl { background: transparent; }
    QGroupBox {
        font-weight: 700; font-size: 9pt; color: #3b4fc8;
        border: 1px solid #e2e6f5; border-radius: 4px;
        margin-top: 8px; padding: 12px 8px 8px 8px; background: white;
    }
    QGroupBox::title {
        subcontrol-origin: margin; subcontrol-position: top left;
        left: 8px; padding: 0 4px; color: #3b4fc8; background: white;
    }
    QPushButton {
        border: 1px solid #c8d0ec; border-radius: 4px;
        padding: 4px 8px; background: white; color: #2d3748;
        font-size: 9pt; min-height: 26px;
    }
    QPushButton:hover  { background: #eef1fc; border-color: #4361ee; color: #4361ee; }
    QPushButton:pressed { background: #dce2fb; }
    QPushButton:checked { background: #4361ee; color: white; border-color: #3451c5; font-weight: 600; }
    QComboBox {
        border: 1px solid #c8d0ec; border-radius: 4px;
        padding: 3px 6px; background: white; color: #2d3748;
        font-size: 9pt; min-height: 24px;
    }
    QComboBox:hover { border-color: #4361ee; }
    QLabel    { font-size: 9pt; color: #5a6787; background: transparent; }
    QLabel#struct_hint  { color: #8892b0; font-size: 8pt; }
    QLabel#struct_count { color: #6b78cc; font-size: 8pt; padding-top: 1px; }
    QCheckBox { font-size: 9pt; color: #2d3748; spacing: 6px; background: transparent; }
    QCheckBox::indicator {
        width: 14px; height: 14px;
        border: 1px solid #c8d0ec; border-radius: 4px; background: white;
    }
    QCheckBox::indicator:checked { background: #4361ee; border-color: #3451c5; }
    QTabWidget::pane {
        border: 1px solid #d1d9f0; border-radius: 0 4px 4px 4px;
        background: #f4f6fd;
    }
"""

STRUCT_TABBAR_CSS_LIGHT = """
    QTabBar::tab {
        padding: 5px 8px; min-width: 52px;
        font-size: 9pt; font-weight: 600;
        background: #e8eaf4; color: #2d3748;
        border: 1px solid #d1d9f0; border-bottom: none;
        border-radius: 4px 4px 0 0; margin-right: 2px;
    }
    QTabBar::tab:selected {
        background: #f4f6fd; color: #3b4fc8;
        border-bottom: 1px solid #f4f6fd;
    }
    QTabBar::tab:hover:!selected { background: #dde0f0; color: #3b4fc8; }
"""

STRUCT_PANEL_CSS_DARK = """
    QScrollArea { border: none; background: transparent; }
    QWidget#structCtrl { background: transparent; }
    QGroupBox {
        font-weight: 700; font-size: 9pt; color: #7b9cff;
        border: 1px solid #2d3561; border-radius: 4px;
        margin-top: 8px; padding: 12px 8px 8px 8px; background: #16213e;
    }
    QGroupBox::title {
        subcontrol-origin: margin; subcontrol-position: top left;
        left: 8px; padding: 0 4px; color: #7b9cff; background: #16213e;
    }
    QPushButton {
        border: 1px solid #2d3561; border-radius: 4px;
        padding: 4px 8px; background: #16213e; color: #e2e8f0;
        font-size: 9pt; min-height: 26px;
    }
    QPushButton:hover  { background: #1a3a5c; border-color: #7b9cff; color: #7b9cff; }
    QPushButton:pressed { background: #0f3460; }
    QPushButton:checked { background: #7b9cff; color: #1a1a2e; border-color: #6b8eff; font-weight: 600; }
    QComboBox {
        border: 1px solid #2d3561; border-radius: 4px;
        padding: 3px 6px; background: #16213e; color: #e2e8f0;
        font-size: 9pt; min-height: 24px;
    }
    QComboBox:hover { border-color: #7b9cff; }
    QLabel    { font-size: 9pt; color: #94a3b8; background: transparent; }
    QLabel#struct_hint  { color: #6b78a8; font-size: 8pt; }
    QLabel#struct_count { color: #7b9cff; font-size: 8pt; padding-top: 1px; }
    QCheckBox { font-size: 9pt; color: #e2e8f0; spacing: 6px; background: transparent; }
    QCheckBox::indicator {
        width: 14px; height: 14px;
        border: 1px solid #2d3561; border-radius: 4px; background: #16213e;
    }
    QCheckBox::indicator:checked { background: #7b9cff; border-color: #6b8eff; }
    QTabWidget::pane {
        border: 1px solid #1a3a5c; border-radius: 0 4px 4px 4px;
        background: #0f3460;
    }
"""

STRUCT_TABBAR_CSS_DARK = """
    QTabBar::tab {
        padding: 5px 8px; min-width: 52px;
        font-size: 9pt; font-weight: 600;
        background: #16213e; color: #c8d8e8;
        border: 1px solid #1a3a5c; border-bottom: none;
        border-radius: 4px 4px 0 0; margin-right: 2px;
    }
    QTabBar::tab:selected {
        background: #0f3460; color: #7b9cff;
        border-bottom: 1px solid #0f3460;
    }
    QTabBar::tab:hover:!selected { background: #1a3a5c; color: #7b9cff; }
"""

__all__ = [
    "LIGHT_THEME_CSS",
    "DARK_THEME_CSS",
    "STRUCT_PANEL_CSS_LIGHT",
    "STRUCT_TABBAR_CSS_LIGHT",
    "STRUCT_PANEL_CSS_DARK",
    "STRUCT_TABBAR_CSS_DARK",
    "NAMED_COLORS",
    "NAMED_COLORMAPS",
]
