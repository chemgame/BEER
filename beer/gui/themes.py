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
 QTreeWidget#graph_tree, QListWidget#report_nav {
     background-color: #f0f2fa;
     border: none;
     border-right: 1px solid #d0d4e0;
     font-size: 11px;
     outline: 0;
 }
 QTreeWidget#graph_tree::item { padding: 5px 6px; color: #4a5568; }
 QTreeWidget#graph_tree::item:selected { background-color: #4361ee; color: #ffffff; border-radius: 3px; }
 QTreeWidget#graph_tree::branch { background-color: #f0f2fa; }
 QListWidget#report_nav::item { padding: 8px 10px; color: #4a5568; }
 QListWidget#report_nav::item:selected { background-color: #4361ee; color: #ffffff; }
 QListWidget#report_nav::item:hover:!selected { background-color: #dce3f8; }
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
 QTreeWidget#graph_tree, QListWidget#report_nav {
     background-color: #16213e;
     border: none;
     border-right: 1px solid #2d3561;
     font-size: 11px;
     outline: 0;
 }
 QTreeWidget#graph_tree::item { padding: 5px 6px; color: #94a3b8; }
 QTreeWidget#graph_tree::item:selected { background-color: #4cc9f0; color: #1a1a2e; border-radius: 3px; }
 QTreeWidget#graph_tree::branch { background-color: #16213e; }
 QListWidget#report_nav::item { padding: 8px 10px; color: #94a3b8; }
 QListWidget#report_nav::item:selected { background-color: #4cc9f0; color: #1a1a2e; }
 QListWidget#report_nav::item:hover:!selected { background-color: #1a3a5c; }
"""

__all__ = [
    "LIGHT_THEME_CSS",
    "DARK_THEME_CSS",
    "NAMED_COLORS",
    "NAMED_COLORMAPS",
]
