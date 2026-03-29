"""BEER GUI themes — CSS stylesheets for light and dark modes."""
from __future__ import annotations

from beer.constants import NAMED_COLORS  # re-export
from beer.constants import NAMED_COLORMAPS  # re-export

LIGHT_THEME_CSS = """
 QWidget {
     background-color: #f5f6fa;
     color: #1a1a2e;
     font-family: Arial, 'Helvetica Neue';
     font-size: 12px;
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
 QPushButton {
     background-color: #4361ee;
     color: #ffffff;
     border: none;
     border-radius: 5px;
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
     border-top-left-radius: 5px;
     border-top-right-radius: 5px;
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
 QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #d0d4e0; border-radius: 3px; }
 QCheckBox::indicator:checked { background-color: #4361ee; border-color: #4361ee; }
 QScrollBar:vertical { background: #f0f0f5; width: 10px; border-radius: 5px; }
 QScrollBar::handle:vertical { background: #c0c4d0; border-radius: 5px; min-height: 30px; }
 QStatusBar { background-color: #4361ee; color: #ffffff; font-size: 11px; }
 QToolBar { background-color: #eef0f8; border: 1px solid #d0d4e0; border-radius: 4px; spacing: 2px; padding: 2px; }
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
     font-size: 11px;
     font-weight: 500;
     outline: 0;
 }
 QListWidget#nav_bar::item {
     padding: 11px 10px;
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
     font-size: 11px;
     outline: 0;
 }
 QTreeWidget#graph_tree::item { padding: 5px 6px; color: #4a5568; }
 QTreeWidget#graph_tree::item:selected { background-color: #4361ee; color: #ffffff; border-radius: 3px; }
 QTreeWidget#graph_tree::branch { background-color: #f0f2fa; }
 QTreeWidget#report_nav::item { padding: 6px 8px; color: #4a5568; }
 QTreeWidget#report_nav::item:selected { background-color: #4361ee; color: #ffffff; border-radius: 3px; }
 QTreeWidget#report_nav::item:hover:!selected { background-color: #dce3f8; }
 QTreeWidget#report_nav::branch { background-color: #f0f2fa; }
 /* --- Fetch / chip buttons --- */
 QPushButton#chip_btn {
     background: transparent;
     color: #4361ee;
     border: 1px solid #b0bae8;
     border-radius: 10px;
     padding: 2px 9px;
     font-size: 10px;
     min-height: 24px;
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
 /* --- Section group labels (fetch bar headers) --- */
 QLabel#group_lbl { color: #8892b0; font-size: 9px; font-weight: 600; }
 /* --- Settings section headers --- */
 QLabel#section_header {
     font-size: 11pt;
     font-weight: 700;
     color: #4361ee;
     border-bottom: 1px solid #d0d4e0;
     padding-bottom: 4px;
     margin-top: 8px;
 }
 /* --- Accent labels (sequence panel headings) --- */
 QLabel#accent_lbl { color: #4361ee; font-weight: 600; }
 QLabel#seq_info_lbl { color: #4361ee; font-size: 9pt; font-weight: 600; padding: 2px 4px; }
 /* --- Status / hint labels --- */
 QLabel#status_lbl { color: #718096; font-style: italic; }
 QLabel#status_lbl[status_state="idle"]    { color: #718096; font-style: italic; font-weight: normal; }
 QLabel#status_lbl[status_state="success"] { color: #1a7a4a; font-style: normal; font-weight: 600; }
 QLabel#status_lbl[status_state="error"]   { color: #c0392b; font-style: normal; font-weight: normal; }
 /* --- ESM2 status indicator --- */
 QLabel#esm2_lbl { font-size: 10px; font-weight: 600; }
 QLabel#esm2_lbl[esm2_state="ready"]   { color: #4361ee; }
 QLabel#esm2_lbl[esm2_state="active"]  { color: #f72585; }
 QLabel#esm2_lbl[esm2_state="missing"] { color: #718096; }
 /* --- Protein info bar --- */
 QTextBrowser#info_bar {
     background: #f0f4ff;
     border: 1px solid #c8d0ec;
     border-radius: 6px;
     padding: 4px 8px;
     font-size: 9pt;
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
     border-radius: 5px;
     padding: 6px 14px;
     font-weight: 600;
 }
 QPushButton#danger_btn:hover { background: #fde8e8; }
 /* --- Vertical separator in fetch bar --- */
 QFrame#v_sep { color: #d0d4e0; }
 /* --- Muted placeholder / info labels --- */
 QLabel#placeholder_lbl { color: #718096; font-style: italic; }
"""

DARK_THEME_CSS = """
 QWidget {
     background-color: #1a1a2e;
     color: #e2e8f0;
     font-family: Arial, 'Helvetica Neue';
     font-size: 12px;
 }
 QMainWindow { background-color: #1a1a2e; }
 QLineEdit, QTextEdit, QTextBrowser {
     background-color: #16213e;
     color: #e2e8f0;
     border: 1px solid #2d3561;
     border-radius: 4px;
     padding: 4px 6px;
     selection-background-color: #4cc9f0;
 }
 QPushButton {
     background-color: #4cc9f0;
     color: #1a1a2e;
     border: none;
     border-radius: 5px;
     padding: 6px 14px;
     font-weight: 600;
     letter-spacing: 0.3px;
 }
 QPushButton:hover { background-color: #3ab7dd; }
 QPushButton:pressed { background-color: #28a4c9; }
 QPushButton:disabled { background-color: #2d3561; color: #6b7280; }
 QTabWidget::pane { border: 1px solid #2d3561; border-radius: 4px; background: #16213e; }
 QTabBar::tab {
     background: #0f3460;
     color: #94a3b8;
     padding: 8px 16px;
     border-top-left-radius: 5px;
     border-top-right-radius: 5px;
     margin-right: 2px;
     font-weight: 500;
 }
 QTabBar::tab:selected { background: #4cc9f0; color: #1a1a2e; }
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
     color: #4cc9f0;
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
 QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #2d3561; border-radius: 3px; }
 QCheckBox::indicator:checked { background-color: #4cc9f0; border-color: #4cc9f0; }
 QScrollBar:vertical { background: #16213e; width: 10px; border-radius: 5px; }
 QScrollBar::handle:vertical { background: #2d3561; border-radius: 5px; min-height: 30px; }
 QStatusBar { background-color: #0f3460; color: #4cc9f0; font-size: 11px; }
 /* --- Tooltips --- */
 QToolTip {
     background-color: #0f3460;
     color: #e2e8f0;
     border: 1px solid #4cc9f0;
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
     font-size: 11px;
     font-weight: 500;
     outline: 0;
 }
 QListWidget#nav_bar::item {
     padding: 11px 10px;
     color: #94a3b8;
     border-left: 3px solid transparent;
 }
 QListWidget#nav_bar::item:selected {
     background-color: #1a3a5c;
     color: #4cc9f0;
     border-left: 3px solid #4cc9f0;
     font-weight: 700;
 }
 QListWidget#nav_bar::item:hover:!selected { background-color: #1a3a5c; color: #e2e8f0; }
 QFrame#nav_sep { color: #1a3a5c; max-width: 1px; }
 /* --- Graph tree & report nav --- */
 QTreeWidget#graph_tree, QTreeWidget#report_nav {
     background-color: #16213e;
     border: none;
     border-right: 1px solid #2d3561;
     font-size: 11px;
     outline: 0;
 }
 QTreeWidget#graph_tree::item { padding: 5px 6px; color: #94a3b8; }
 QTreeWidget#graph_tree::item:selected { background-color: #4cc9f0; color: #1a1a2e; border-radius: 3px; }
 QTreeWidget#graph_tree::branch { background-color: #16213e; }
 QTreeWidget#report_nav::item { padding: 6px 8px; color: #94a3b8; }
 QTreeWidget#report_nav::item:selected { background-color: #4cc9f0; color: #1a1a2e; border-radius: 3px; }
 QTreeWidget#report_nav::item:hover:!selected { background-color: #1a3a5c; }
 QTreeWidget#report_nav::branch { background-color: #16213e; }
 /* --- Fetch / chip buttons --- */
 QPushButton#chip_btn {
     background: transparent;
     color: #4cc9f0;
     border: 1px solid #2d4a6e;
     border-radius: 10px;
     padding: 2px 9px;
     font-size: 10px;
     min-height: 24px;
     font-weight: 600;
 }
 QPushButton#chip_btn:hover:!disabled { background: #1a3a5c; border-color: #4cc9f0; }
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
 /* --- Section group labels (fetch bar headers) --- */
 QLabel#group_lbl { color: #4a5a7a; font-size: 9px; font-weight: 600; }
 /* --- Settings section headers --- */
 QLabel#section_header {
     font-size: 11pt;
     font-weight: 700;
     color: #4cc9f0;
     border-bottom: 1px solid #2d3561;
     padding-bottom: 4px;
     margin-top: 8px;
 }
 /* --- Accent labels --- */
 QLabel#accent_lbl { color: #4cc9f0; font-weight: 600; }
 QLabel#seq_info_lbl { color: #4cc9f0; font-size: 9pt; font-weight: 600; padding: 2px 4px; }
 /* --- Status / hint labels --- */
 QLabel#status_lbl { color: #5a6787; font-style: italic; }
 QLabel#status_lbl[status_state="idle"]    { color: #5a6787; font-style: italic; font-weight: normal; }
 QLabel#status_lbl[status_state="success"] { color: #43aa8b; font-style: normal; font-weight: 600; }
 QLabel#status_lbl[status_state="error"]   { color: #f72585; font-style: normal; font-weight: normal; }
 /* --- ESM2 status indicator --- */
 QLabel#esm2_lbl { font-size: 10px; font-weight: 600; }
 QLabel#esm2_lbl[esm2_state="ready"]   { color: #4cc9f0; }
 QLabel#esm2_lbl[esm2_state="active"]  { color: #f72585; }
 QLabel#esm2_lbl[esm2_state="missing"] { color: #5a6787; }
 /* --- Protein info bar --- */
 QTextBrowser#info_bar {
     background: #16213e;
     border: 1px solid #2d3561;
     border-radius: 6px;
     padding: 4px 8px;
     font-size: 9pt;
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
 QToolButton#info_btn:hover { color: #4cc9f0; background: transparent; }
 /* --- Delete / danger buttons --- */
 QPushButton#delete_btn { background-color: #c1121f; color: #ffffff; }
 QPushButton#delete_btn:hover { background-color: #a50f1a; }
 QPushButton#delete_btn:pressed { background-color: #8a0c15; }
 QPushButton#danger_btn {
     background: transparent;
     color: #f72585;
     border: 1px solid #f72585;
     border-radius: 5px;
     padding: 6px 14px;
     font-weight: 600;
 }
 QPushButton#danger_btn:hover { background: #2a1020; }
 /* --- Vertical separator in fetch bar --- */
 QFrame#v_sep { color: #2d3561; }
 /* --- Muted placeholder / info labels --- */
 QLabel#placeholder_lbl { color: #6a7a9a; font-style: italic; }
"""

__all__ = [
    "LIGHT_THEME_CSS",
    "DARK_THEME_CSS",
    "NAMED_COLORS",
    "NAMED_COLORMAPS",
]
