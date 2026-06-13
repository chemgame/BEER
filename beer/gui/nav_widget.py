"""BEER left-sidebar navigation widget (PySide6)."""
from __future__ import annotations

import pathlib

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QListWidget, QListWidgetItem,
    QFrame, QStackedWidget,
)
from PySide6.QtCore import Qt, QSize

_ICON_DIR = pathlib.Path(__file__).parent / "icons"


class NavTabWidget(QWidget):
    """Left-sidebar navigation that is a drop-in replacement for QTabWidget.
    Implements the subset of QTabWidget API used in this app."""

    _NAV_ICONS = {
        "Analysis":            "\U0001f9ec",   # 🧬  sequence / biology
        "Report":              "\U0001f4cb",   # 📋  report / results
        "Summary":             "\U0001f4ca",   # 📊  bar chart / stats
        "Graphs":              "\U0001f4c8",   # 📈  line chart / plots
        "Structure":           "\U0001f52c",   # 🔬  microscope / 3-D
        "BLAST":               "\U0001f50d",   # 🔍  search
        "Compare":             "⚖️", # ⚖️  compare / scales
        "Multichain Analysis": "\U0001f9e9",   # 🧩  multichain / assembly
        "Truncation":          "✂️", # ✂️  truncation / cut
        "MSA":                 "\U0001f500",   # 🔀  multiple alignment
        "Protein Complex":     "⚛️", # ⚛️  complex / molecular
        "Fix PDB":             "\U0001F527",   # 🔧  fix / repair
        "Settings":            "⚙️", # ⚙️  settings
        "Help":                "\U0001f4d6",   # 📖  help / docs
    }

    # Monochrome SVG icon (Lucide, ISC) per tab — tinted to the theme. Falls
    # back to the emoji above if the SVG can't be rendered.
    _NAV_SVG = {
        "Analysis": "dna", "Report": "clipboard-list", "Graphs": "chart-line",
        "Structure": "microscope", "BLAST": "search", "Compare": "scale",
        "Multichain Analysis": "puzzle", "Truncation": "scissors", "MSA": "rows-3",
        "Protein Complex": "atom", "Fix PDB": "wrench", "Settings": "settings",
        "Help": "book-open",
    }

    # Insert a separator before these tab names
    _GROUP_BREAKS = {"BLAST", "Settings"}

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._icon_color = "#46506e"   # tint for the SVG nav icons (theme-driven)
        self.nav_list = QListWidget()
        self.nav_list.setObjectName("nav_bar")
        self.nav_list.setFixedWidth(180)
        self.nav_list.setIconSize(QSize(18, 18))
        self.nav_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(self.nav_list)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFrameShadow(QFrame.Shadow.Plain)
        sep.setObjectName("nav_sep")
        outer.addWidget(sep)

        self.stack = QStackedWidget()
        outer.addWidget(self.stack, 1)

        # Maps list-row index → stack page index (-1 for separator rows).
        self._row_to_stack: list[int] = []
        # Maps tab name → stack page index (stable; used by set_display_order).
        self._name_to_stack: dict[str, int] = {}

        self.nav_list.currentRowChanged.connect(self._on_row_changed)

    # ── internal ──────────────────────────────────────────────────────────────

    def _on_row_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._row_to_stack):
            return
        stack_idx = self._row_to_stack[row]
        if stack_idx >= 0:
            self.stack.setCurrentIndex(stack_idx)
        else:
            # Separator was somehow focused — skip to next selectable row.
            next_row = row + 1
            while next_row < len(self._row_to_stack) and self._row_to_stack[next_row] < 0:
                next_row += 1
            if next_row < len(self._row_to_stack):
                self.nav_list.setCurrentRow(next_row)

    def _make_icon(self, name: str, color: str):
        """Render the tab's Lucide SVG tinted to *color*, or None on failure."""
        svg_name = self._NAV_SVG.get(name)
        if not svg_name:
            return None
        try:
            from PySide6.QtSvg import QSvgRenderer
            from PySide6.QtGui import QPixmap, QPainter, QIcon
            svg = (_ICON_DIR / f"{svg_name}.svg").read_text(encoding="utf-8")
            svg = svg.replace("currentColor", color)
            renderer = QSvgRenderer(bytearray(svg, "utf-8"))
            pm = QPixmap(18, 18)
            pm.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pm)
            renderer.render(painter)
            painter.end()
            return QIcon(pm)
        except Exception:
            return None

    def _make_nav_item(self, name: str) -> QListWidgetItem:
        """Build a nav row: SVG icon + name, or emoji + name as a fallback."""
        icon = self._make_icon(name, self._icon_color)
        if icon is not None:
            item = QListWidgetItem(f"  {name}")
            item.setIcon(icon)
        else:
            item = QListWidgetItem(f"  {self._NAV_ICONS.get(name, '▸')}  {name}")
        item.setData(Qt.ItemDataRole.UserRole, name)
        return item

    def set_icon_color(self, color: str) -> None:
        """Re-tint all nav icons (called when the theme changes)."""
        self._icon_color = color
        for row in range(self.nav_list.count()):
            item = self.nav_list.item(row)
            name = item.data(Qt.ItemDataRole.UserRole)
            if name:
                icon = self._make_icon(name, color)
                if icon is not None:
                    item.setIcon(icon)

    def _add_separator(self) -> None:
        item = QListWidgetItem()
        item.setFlags(Qt.ItemFlag.NoItemFlags)
        item.setSizeHint(QSize(180, 14))
        self.nav_list.addItem(item)
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Plain)
        line.setObjectName("nav_sep_h")
        self.nav_list.setItemWidget(item, line)
        self._row_to_stack.append(-1)

    # ── public API ────────────────────────────────────────────────────────────

    def addTab(self, widget: QWidget, name: str) -> int:
        if name in self._GROUP_BREAKS:
            self._add_separator()

        self.nav_list.addItem(self._make_nav_item(name))
        idx = self.stack.addWidget(widget)
        self._row_to_stack.append(idx)
        self._name_to_stack[name] = idx

        # Select the very first real tab.
        if sum(1 for s in self._row_to_stack if s >= 0) == 1:
            self.nav_list.setCurrentRow(len(self._row_to_stack) - 1)
        return idx

    def set_display_order(self, ordered_names: list,
                          group_breaks: "set | None" = None) -> None:
        """Re-arrange the sidebar rows into a custom visual order WITHOUT changing
        stack page indices — so setCurrentIndex()/shortcut indices stay valid.

        ordered_names: tab names in the desired top-to-bottom order.
        group_breaks: names that get a separator inserted just before them.
        """
        group_breaks = group_breaks or set()
        cur_stack = self.currentIndex()
        self.nav_list.blockSignals(True)
        self.nav_list.clear()
        self._row_to_stack = []
        for name in ordered_names:
            if name not in self._name_to_stack:
                continue
            if name in group_breaks and self._row_to_stack:
                self._add_separator()
            self.nav_list.addItem(self._make_nav_item(name))
            self._row_to_stack.append(self._name_to_stack[name])
        # Append any tabs that were added but not listed (safety net).
        for name, sidx in self._name_to_stack.items():
            if sidx not in self._row_to_stack:
                self.nav_list.addItem(self._make_nav_item(name))
                self._row_to_stack.append(sidx)
        self.nav_list.blockSignals(False)
        self.setCurrentIndex(cur_stack if cur_stack >= 0 else 0)

    def row_for_stack(self, stack_idx: int) -> int:
        """Sidebar row currently showing stack page *stack_idx* (-1 if none)."""
        for row, s in enumerate(self._row_to_stack):
            if s == stack_idx:
                return row
        return -1

    def stack_for_name(self, name: str) -> int:
        """Stack page index for a tab name (-1 if not present)."""
        return self._name_to_stack.get(name, -1)

    def setCurrentIndex(self, idx: int) -> None:
        for row, stack_idx in enumerate(self._row_to_stack):
            if stack_idx == idx:
                self.nav_list.setCurrentRow(row)
                return

    def currentIndex(self) -> int:
        row = self.nav_list.currentRow()
        if 0 <= row < len(self._row_to_stack):
            return self._row_to_stack[row]
        return -1

    def currentWidget(self) -> QWidget:
        return self.stack.currentWidget()

    def widget(self, idx: int) -> QWidget:
        return self.stack.widget(idx)

    def count(self) -> int:
        return self.stack.count()


__all__ = ["NavTabWidget"]
