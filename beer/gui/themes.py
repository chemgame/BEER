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
 QPushButton:disabled { background-color: #b0b8cc; color: #2d3748; }
 QTabWidget::pane { border: 1px solid #d0d4e0; border-radius: 4px; background: #ffffff; }
 QTabBar { qproperty-drawBase: 0; }
 QTabBar::tab {
     background: #e8eaf0;
     color: #4a5568;
     padding: 6px 16px;
     border: 1px solid #c8d0ec;
     border-radius: 4px;
     margin-right: 4px;
     margin-bottom: 4px;
     font-weight: 500;
 }
 QTabBar::tab:selected { background: #4361ee; color: #ffffff; border-color: #3451c5; }
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
     padding: 4px 28px 4px 8px;
 }
 QComboBox::drop-down {
     subcontrol-origin: padding; subcontrol-position: top right;
     width: 22px; border-left: 1px solid #d0d4e0;
     background: #eef0f8;
     border-top-right-radius: 3px; border-bottom-right-radius: 3px;
 }
 QComboBox::down-arrow { width: 10px; height: 10px; }
 QComboBox:hover { border-color: #4361ee; background: #f0f4ff; }
 QComboBox:hover::drop-down { background: #dce3f8; border-left-color: #4361ee; }
 QComboBox:focus { border-color: #4361ee; background: #f8f9ff; }
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
 QFrame#nav_sep   { color: #c8cede; max-width: 1px; }
 QFrame#nav_sep_h { color: #c0c8de; max-height: 1px; background: #c0c8de; border: none; }
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
 QTreeWidget#graph_tree::item:hover:!selected { background-color: #dce3f8; }
 QTreeWidget#graph_tree::branch { background-color: #f0f2fa; image: none; }
 QTreeWidget#report_nav::item { padding: 6px 8px; color: #4a5568; background-color: #f0f2fa; }
 QTreeWidget#report_nav::item:selected { background-color: #4361ee; color: #ffffff; border-radius: 4px; }
 QTreeWidget#report_nav::item:hover:!selected { background-color: #dce3f8; }
 QTreeWidget#report_nav::branch { background-color: #f0f2fa; image: none; }
 /* --- Fetch / chip buttons --- */
 QPushButton#chip_btn {
     background: transparent;
     color: #4361ee;
     border: 1px solid #b0bae8;
     border-radius: 10px;
     padding: 4px 12px;
     font-size: 12px;
     min-height: 28px;
     font-weight: 600;
 }
 QPushButton#chip_btn:hover:!disabled { background: #e8eeff; border-color: #4361ee; }
 QPushButton#chip_btn:pressed:!disabled { background: #d0d8f8; }
 QPushButton#chip_btn:disabled { color: #7a85a0; border-color: #d0d4e8; background: transparent; border-radius: 10px; }
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
 QListWidget#nav_bar::item:disabled { color: #6b78a0; font-style: italic; }
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
 /* --- Secondary (outline) buttons --- */
 QPushButton#secondary_btn {
     background: transparent;
     color: #4361ee;
     border: 1px solid #c8d0ec;
     border-radius: 4px;
     padding: 6px 14px;
     font-weight: 500;
 }
 QPushButton#secondary_btn:hover { background: #eef1fc; border-color: #4361ee; }
 QPushButton#secondary_btn:pressed { background: #dce3f8; }
 QPushButton#secondary_btn:disabled { color: #718096; border-color: #d0d4e8; }
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
 QToolButton#help_btn { font-weight: bold; border-radius: 10px; }
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
 QTabBar { qproperty-drawBase: 0; }
 QTabBar::tab {
     background: #0f3460;
     color: #94a3b8;
     padding: 6px 16px;
     border: 1px solid #1a3a5c;
     border-radius: 4px;
     margin-right: 4px;
     margin-bottom: 4px;
     font-weight: 500;
 }
 QTabBar::tab:selected { background: #7b9cff; color: #1a1a2e; border-color: #6b8eff; }
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
     padding: 4px 28px 4px 8px;
 }
 QComboBox::drop-down {
     subcontrol-origin: padding; subcontrol-position: top right;
     width: 22px; border-left: 1px solid #2d3561;
     background: #1a2a50;
     border-top-right-radius: 3px; border-bottom-right-radius: 3px;
 }
 QComboBox::down-arrow { width: 10px; height: 10px; }
 QComboBox:hover { border-color: #7b9cff; background: #1e2448; }
 QComboBox:hover::drop-down { background: #1e3060; border-left-color: #7b9cff; }
 QComboBox:focus { border-color: #7b9cff; background: #1a2240; }
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
 QFrame#nav_sep   { color: #1a3a5c; max-width: 1px; }
 QFrame#nav_sep_h { color: #1a3a5c; max-height: 1px; background: #1a3a5c; border: none; }
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
 QTreeWidget#graph_tree::item:hover:!selected { background-color: #1a3a5c; }
 QTreeWidget#graph_tree::branch { background-color: #16213e; image: none; }
 QTreeWidget#report_nav::item { padding: 6px 8px; color: #94a3b8; background-color: #16213e; }
 QTreeWidget#report_nav::item:selected { background-color: #7b9cff; color: #1a1a2e; border-radius: 4px; }
 QTreeWidget#report_nav::item:hover:!selected { background-color: #1a3a5c; }
 QTreeWidget#report_nav::branch { background-color: #16213e; image: none; }
 /* --- Fetch / chip buttons --- */
 QPushButton#chip_btn {
     background: transparent;
     color: #7b9cff;
     border: 1px solid #2d4a6e;
     border-radius: 10px;
     padding: 4px 12px;
     font-size: 12px;
     min-height: 28px;
     font-weight: 600;
 }
 QPushButton#chip_btn:hover:!disabled { background: #1a3a5c; border-color: #7b9cff; }
 QPushButton#chip_btn:pressed:!disabled { background: #0f3460; }
 QPushButton#chip_btn:disabled { color: #6a7896; border-color: #2d3a5a; background: transparent; border-radius: 10px; }
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
 QListWidget#nav_bar::item:disabled { color: #6b7ba5; font-style: italic; }
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
 /* --- Secondary (outline) buttons --- */
 QPushButton#secondary_btn {
     background: transparent;
     color: #7b9cff;
     border: 1px solid #2d3561;
     border-radius: 4px;
     padding: 6px 14px;
     font-weight: 500;
 }
 QPushButton#secondary_btn:hover { background: #1a2a50; border-color: #7b9cff; }
 QPushButton#secondary_btn:pressed { background: #0f3460; }
 QPushButton#secondary_btn:disabled { color: #6a7896; border-color: #2d3a5a; }
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
 QToolButton#help_btn { font-weight: bold; border-radius: 10px; }
 /* --- PDB xref section label --- */
 QLabel#pdb_xref_lbl { color: #94a3b8; font-size: 9pt; font-weight: 600; }
 /* --- Chain selector label --- */
 QLabel#chain_lbl { font-weight: 600; }
 /* --- Toolbar (dark) --- */
 QToolBar { background-color: #0f3460; border: 1px solid #2d3561; border-radius: 4px; spacing: 4px; padding: 4px; }
 QToolBar QToolButton { background-color: #16213e; border: 1px solid #2d3561; border-radius: 4px; padding: 3px; color: #94a3b8; }
 QToolBar QToolButton:hover { background-color: #1a3a5c; border-color: #7b9cff; color: #e2e8f0; }
 QToolBar QToolButton:pressed { background-color: #0f3460; }
 /* --- Horizontal scrollbar (dark) --- */
 QScrollBar:horizontal { background: #16213e; height: 10px; border-radius: 5px; }
 QScrollBar::handle:horizontal { background: #2d3561; border-radius: 5px; min-width: 30px; }
 QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }
 QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: none; }
 /* --- SpinBox (dark) --- */
 QSpinBox, QDoubleSpinBox {
     background-color: #16213e; border: 1px solid #2d3561; border-radius: 4px;
     color: #e2e8f0; padding: 2px 4px; min-height: 24px;
 }
 QSpinBox:focus, QDoubleSpinBox:focus { border-color: #7b9cff; }
 QSpinBox::up-button, QDoubleSpinBox::up-button,
 QSpinBox::down-button, QDoubleSpinBox::down-button {
     background-color: #1a2a50; border: none; width: 16px;
 }
 QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
 QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover { background-color: #2d3a6a; }
 /* --- Splitter (dark) --- */
 QSplitter::handle { background-color: #2d3561; }
 QSplitter::handle:horizontal { width: 4px; }
 QSplitter::handle:vertical { height: 4px; }
 QSplitter::handle:hover { background-color: #7b9cff; }
 /* --- RadioButton (dark) --- */
 QRadioButton { color: #94a3b8; spacing: 6px; }
 QRadioButton::indicator { width: 14px; height: 14px; border: 1px solid #2d3561; border-radius: 7px; background: #16213e; }
 QRadioButton::indicator:checked { background-color: #7b9cff; border-color: #7b9cff; }
 QRadioButton::indicator:hover { border-color: #7b9cff; }
 /* --- ProgressDialog / ProgressBar (dark) --- */
 QProgressDialog { background-color: #1a1a2e; color: #e2e8f0; }
 QProgressDialog QLabel { color: #e2e8f0; }
 QProgressDialog QPushButton { background-color: #16213e; color: #94a3b8; border: 1px solid #2d3561; border-radius: 4px; padding: 4px 12px; }
 QProgressDialog QPushButton:hover { background-color: #1a3a5c; border-color: #7b9cff; color: #e2e8f0; }
 QProgressBar { background-color: #16213e; border: 1px solid #2d3561; border-radius: 4px; color: #e2e8f0; text-align: center; }
 QProgressBar::chunk { background-color: #7b9cff; border-radius: 3px; }
"""

# ── Structure viewer control panel CSS ───────────────────────────────────────
# Applied to struct_ctrl_scroll (QTabWidget) so selectors scope to its children.
# Kept separate from the main QSS because macOS native style overrides QTabBar
# colours when set via the application stylesheet; a widget-level setStyleSheet
# on the tab bar itself is the only reliable cross-platform workaround.

STRUCT_PANEL_CSS_LIGHT = """
    /* ── scroll area shell ───────────────────────────────────────────────── */
    QScrollArea {
        border: none;
        background: #f4f6fd;
    }
    QScrollBar:vertical {
        background: #eceef8;
        width: 6px;
        border-radius: 3px;
        margin: 0;
    }
    QScrollBar::handle:vertical {
        background: #b8c0d8;
        border-radius: 3px;
        min-height: 24px;
    }
    QScrollBar::handle:vertical:hover { background: #8892c8; }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
    QScrollBar::add-page:vertical,  QScrollBar::sub-page:vertical  { background: none; }

    /* ── panel root ──────────────────────────────────────────────────────── */
    QWidget#structCtrl {
        background: #f4f6fd;
    }

    /* ── card sections ───────────────────────────────────────────────────── */
    QWidget.card {
        background: #ffffff;
        border: 1px solid #e2e6f5;
        border-radius: 8px;
    }

    /* ── section header labels (uppercase, accent-left-border) ──────────── */
    QLabel.section_header {
        font-size: 8pt;
        font-weight: 700;
        letter-spacing: 0.08em;
        color: #3b4fc8;
        background: transparent;
        padding: 0px 0px 0px 8px;
        border-left: 3px solid #4361ee;
    }

    /* ── collapsible section toggle button ───────────────────────────────── */
    QPushButton.section_toggle {
        text-align: left;
        font-size: 8pt;
        font-weight: 700;
        letter-spacing: 0.06em;
        color: #3b4fc8;
        background: #ffffff;
        border: 1px solid #e2e6f5;
        border-radius: 8px;
        padding: 7px 10px;
        min-height: 30px;
    }
    QPushButton.section_toggle:hover {
        background: #eef1fc;
        border-color: #b8c4e8;
    }
    QPushButton.section_toggle:pressed {
        background: #dce3f8;
    }

    /* ── reset / persistent view button ─────────────────────────────────── */
    QPushButton#struct_reset_btn {
        background: #ffffff;
        color: #3b4fc8;
        border: 1px solid #c8d0ec;
        border-radius: 6px;
        padding: 6px 10px;
        font-size: 9pt;
        font-weight: 600;
        min-height: 28px;
        text-align: center;
    }
    QPushButton#struct_reset_btn:hover {
        background: #eef1fc;
        border-color: #4361ee;
        color: #4361ee;
    }
    QPushButton#struct_reset_btn:pressed { background: #dce3f8; }

    /* ── generic buttons inside panel ───────────────────────────────────── */
    QPushButton {
        background: #ffffff;
        color: #2d3748;
        border: 1px solid #c8d0ec;
        border-radius: 6px;
        padding: 4px 8px;
        font-size: 9pt;
        min-height: 26px;
    }
    QPushButton:hover  { background: #eef1fc; border-color: #4361ee; color: #4361ee; }
    QPushButton:pressed { background: #dce2fb; }
    QPushButton:checked {
        background: #4361ee;
        color: #ffffff;
        border-color: #3451c5;
        font-weight: 600;
    }
    QPushButton:checked:hover { background: #3451d1; }
    QPushButton:disabled { background: #f0f2f8; color: #a0a8c0; border-color: #dde2f0; }

    /* ── measurement mode pill buttons ──────────────────────────────────── */
    QPushButton.mode_btn {
        border-radius: 0px;
        border: 1px solid #c8d0ec;
        min-height: 26px;
        font-size: 8.5pt;
        padding: 4px 6px;
    }
    QPushButton.mode_btn:first-child { border-radius: 6px 0 0 6px; border-right: none; }
    QPushButton.mode_btn:last-child  { border-radius: 0 6px 6px 0; border-left:  none; }
    QPushButton.mode_btn:checked {
        background: #4361ee;
        color: #ffffff;
        border-color: #3451c5;
        font-weight: 600;
    }
    QPushButton.mode_btn:hover:!checked { background: #eef1fc; border-color: #4361ee; }

    /* ── dropdowns ───────────────────────────────────────────────────────── */
    QComboBox {
        background: #ffffff;
        color: #2d3748;
        border: 1px solid #c8d0ec;
        border-radius: 6px;
        padding: 3px 26px 3px 7px;
        font-size: 9pt;
        min-height: 26px;
        selection-background-color: #4361ee;
    }
    QComboBox::drop-down {
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 20px;
        border-left: 1px solid #c8d0ec;
        background: #eef1fc;
        border-top-right-radius: 5px;
        border-bottom-right-radius: 5px;
    }
    QComboBox::down-arrow { width: 8px; height: 8px; }
    QComboBox:hover { border-color: #4361ee; background: #f0f4ff; }
    QComboBox:hover::drop-down { background: #dce3f8; border-left-color: #4361ee; }
    QComboBox:focus { border-color: #4361ee; }
    QComboBox QAbstractItemView {
        background: #ffffff;
        border: 1px solid #c8d0ec;
        border-radius: 4px;
        selection-background-color: #4361ee;
        selection-color: #ffffff;
        font-size: 9pt;
    }

    /* ── checkboxes ──────────────────────────────────────────────────────── */
    QCheckBox {
        font-size: 9pt;
        color: #2d3748;
        spacing: 6px;
        background: transparent;
    }
    QCheckBox::indicator {
        width: 14px;
        height: 14px;
        border: 1.5px solid #b8c0d8;
        border-radius: 4px;
        background: #ffffff;
    }
    QCheckBox::indicator:hover  { border-color: #4361ee; }
    QCheckBox::indicator:checked {
        background: #4361ee;
        border-color: #3451c5;
        image: none;
    }

    /* ── labels ──────────────────────────────────────────────────────────── */
    QLabel {
        font-size: 9pt;
        color: #4a5568;
        background: transparent;
    }
    QLabel#struct_hint {
        color: #8892b0;
        font-size: 8pt;
        font-style: italic;
    }
    QLabel#struct_count {
        color: #4361ee;
        font-size: 8pt;
        font-weight: 600;
        padding-top: 1px;
    }

    /* ── opacity slider ──────────────────────────────────────────────────── */
    QSlider::groove:horizontal {
        height: 4px;
        background: #dde2f0;
        border-radius: 2px;
    }
    QSlider::handle:horizontal {
        background: #4361ee;
        border: 2px solid #ffffff;
        width: 14px;
        height: 14px;
        margin: -5px 0;
        border-radius: 7px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
    }
    QSlider::handle:horizontal:hover { background: #3451d1; }
    QSlider::sub-page:horizontal {
        background: #4361ee;
        border-radius: 2px;
    }

    /* ── line edit (selection box) ───────────────────────────────────────── */
    QLineEdit {
        background: #ffffff;
        color: #2d3748;
        border: 1px solid #c8d0ec;
        border-radius: 6px;
        padding: 4px 7px;
        font-size: 9pt;
        min-height: 26px;
        selection-background-color: #4361ee;
    }
    QLineEdit:focus { border-color: #4361ee; }
"""

STRUCT_TABBAR_CSS_LIGHT = """
    QTabBar { qproperty-drawBase: 0; }
    QTabBar::tab {
        padding: 4px 10px; min-width: 52px;
        font-size: 9pt; font-weight: 600;
        background: #e8eaf4; color: #4a5568;
        border: 1px solid #c8d0ec;
        border-radius: 4px;
        margin-right: 3px; margin-bottom: 4px;
    }
    QTabBar::tab:selected { background: #4361ee; color: #ffffff; border-color: #3451c5; }
    QTabBar::tab:hover:!selected { background: #dde0f0; color: #3b4fc8; }
"""

STRUCT_PANEL_CSS_DARK = """
    /* ── scroll area shell ───────────────────────────────────────────────── */
    QScrollArea {
        border: none;
        background: #111827;
    }
    QScrollBar:vertical {
        background: #1a2240;
        width: 6px;
        border-radius: 3px;
        margin: 0;
    }
    QScrollBar::handle:vertical {
        background: #2d3a5c;
        border-radius: 3px;
        min-height: 24px;
    }
    QScrollBar::handle:vertical:hover { background: #3d4f7a; }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
    QScrollBar::add-page:vertical,  QScrollBar::sub-page:vertical  { background: none; }

    /* ── panel root ──────────────────────────────────────────────────────── */
    QWidget#structCtrl {
        background: #111827;
    }

    /* ── card sections ───────────────────────────────────────────────────── */
    QWidget.card {
        background: #16213e;
        border: 1px solid #2a3d6a;
        border-radius: 8px;
    }

    /* ── section header labels ───────────────────────────────────────────── */
    QLabel.section_header {
        font-size: 8pt;
        font-weight: 700;
        letter-spacing: 0.08em;
        color: #7b9cff;
        background: transparent;
        padding: 0px 0px 0px 8px;
        border-left: 3px solid #7b9cff;
    }

    /* ── collapsible section toggle button ───────────────────────────────── */
    QPushButton.section_toggle {
        text-align: left;
        font-size: 8pt;
        font-weight: 700;
        letter-spacing: 0.06em;
        color: #7b9cff;
        background: #16213e;
        border: 1px solid #1e2d50;
        border-radius: 8px;
        padding: 7px 10px;
        min-height: 30px;
    }
    QPushButton.section_toggle:hover {
        background: #1a2a50;
        border-color: #2d3a6a;
    }
    QPushButton.section_toggle:pressed {
        background: #0f3460;
    }

    /* ── reset / persistent view button ─────────────────────────────────── */
    QPushButton#struct_reset_btn {
        background: #16213e;
        color: #7b9cff;
        border: 1px solid #2d3561;
        border-radius: 6px;
        padding: 6px 10px;
        font-size: 9pt;
        font-weight: 600;
        min-height: 28px;
        text-align: center;
    }
    QPushButton#struct_reset_btn:hover {
        background: #1a2a50;
        border-color: #7b9cff;
    }
    QPushButton#struct_reset_btn:pressed { background: #0f3460; }

    /* ── generic buttons inside panel ───────────────────────────────────── */
    QPushButton {
        background: #16213e;
        color: #e2e8f0;
        border: 1px solid #2d3561;
        border-radius: 6px;
        padding: 4px 8px;
        font-size: 9pt;
        min-height: 26px;
    }
    QPushButton:hover  { background: #1a3a5c; border-color: #7b9cff; color: #7b9cff; }
    QPushButton:pressed { background: #0f3460; }
    QPushButton:checked {
        background: #7b9cff;
        color: #1a1a2e;
        border-color: #6b8eff;
        font-weight: 600;
    }
    QPushButton:checked:hover { background: #6b8eff; }
    QPushButton:disabled { background: #111827; color: #3d4f6a; border-color: #1e2d50; }

    /* ── measurement mode pill buttons ──────────────────────────────────── */
    QPushButton.mode_btn {
        border-radius: 0px;
        border: 1px solid #2d3561;
        min-height: 26px;
        font-size: 8.5pt;
        padding: 4px 6px;
    }
    QPushButton.mode_btn:first-child { border-radius: 6px 0 0 6px; border-right: none; }
    QPushButton.mode_btn:last-child  { border-radius: 0 6px 6px 0; border-left:  none; }
    QPushButton.mode_btn:checked {
        background: #7b9cff;
        color: #1a1a2e;
        border-color: #6b8eff;
        font-weight: 600;
    }
    QPushButton.mode_btn:hover:!checked { background: #1a2a50; border-color: #7b9cff; }

    /* ── dropdowns ───────────────────────────────────────────────────────── */
    QComboBox {
        background: #16213e;
        color: #e2e8f0;
        border: 1px solid #2d3561;
        border-radius: 6px;
        padding: 3px 26px 3px 7px;
        font-size: 9pt;
        min-height: 26px;
        selection-background-color: #7b9cff;
    }
    QComboBox::drop-down {
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 20px;
        border-left: 1px solid #2d3561;
        background: #1a2a50;
        border-top-right-radius: 5px;
        border-bottom-right-radius: 5px;
    }
    QComboBox::down-arrow { width: 8px; height: 8px; }
    QComboBox:hover { border-color: #7b9cff; background: #1a2448; }
    QComboBox:hover::drop-down { background: #1e3060; border-left-color: #7b9cff; }
    QComboBox:focus { border-color: #7b9cff; }
    QComboBox QAbstractItemView {
        background: #16213e;
        border: 1px solid #2d3561;
        border-radius: 4px;
        selection-background-color: #7b9cff;
        selection-color: #1a1a2e;
        color: #e2e8f0;
        font-size: 9pt;
    }

    /* ── checkboxes ──────────────────────────────────────────────────────── */
    QCheckBox {
        font-size: 9pt;
        color: #e2e8f0;
        spacing: 6px;
        background: transparent;
    }
    QCheckBox::indicator {
        width: 14px;
        height: 14px;
        border: 1.5px solid #2d3a5c;
        border-radius: 4px;
        background: #16213e;
    }
    QCheckBox::indicator:hover  { border-color: #7b9cff; }
    QCheckBox::indicator:checked {
        background: #7b9cff;
        border-color: #6b8eff;
    }

    /* ── labels ──────────────────────────────────────────────────────────── */
    QLabel {
        font-size: 9pt;
        color: #94a3b8;
        background: transparent;
    }
    QLabel#struct_hint {
        color: #7a8fa8;
        font-size: 8pt;
        font-style: italic;
    }
    QLabel#struct_count {
        color: #7b9cff;
        font-size: 8pt;
        font-weight: 600;
        padding-top: 1px;
    }

    /* ── opacity slider ──────────────────────────────────────────────────── */
    QSlider::groove:horizontal {
        height: 4px;
        background: #2d3561;
        border-radius: 2px;
    }
    QSlider::handle:horizontal {
        background: #7b9cff;
        border: 2px solid #16213e;
        width: 14px;
        height: 14px;
        margin: -5px 0;
        border-radius: 7px;
    }
    QSlider::handle:horizontal:hover { background: #6b8eff; }
    QSlider::sub-page:horizontal {
        background: #7b9cff;
        border-radius: 2px;
    }

    /* ── line edit (selection box) ───────────────────────────────────────── */
    QLineEdit {
        background: #16213e;
        color: #e2e8f0;
        border: 1px solid #2d3561;
        border-radius: 6px;
        padding: 4px 7px;
        font-size: 9pt;
        min-height: 26px;
        selection-background-color: #7b9cff;
    }
    QLineEdit:focus { border-color: #7b9cff; }
"""

STRUCT_TABBAR_CSS_DARK = """
    QTabBar { qproperty-drawBase: 0; }
    QTabBar::tab {
        padding: 4px 10px; min-width: 52px;
        font-size: 9pt; font-weight: 600;
        background: #16213e; color: #94a3b8;
        border: 1px solid #1a3a5c;
        border-radius: 4px;
        margin-right: 3px; margin-bottom: 4px;
    }
    QTabBar::tab:selected { background: #7b9cff; color: #1a1a2e; border-color: #6b8eff; }
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
