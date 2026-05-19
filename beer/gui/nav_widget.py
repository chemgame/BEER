"""BEER left-sidebar navigation widget (PySide6)."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QListWidget, QListWidgetItem,
    QFrame, QStackedWidget,
)
from PySide6.QtCore import Qt, QSize


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
        "Complex":             "⚛️", # ⚛️  complex / molecular
        "Settings":            "⚙️", # ⚙️  settings
        "Help":                "\U0001f4d6",   # 📖  help / docs
    }

    # Insert a separator before these tab names
    _GROUP_BREAKS = {"BLAST", "Settings"}

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.nav_list = QListWidget()
        self.nav_list.setObjectName("nav_bar")
        self.nav_list.setFixedWidth(180)
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
            if next_row < self.nav_list.count():
                self.nav_list.setCurrentRow(next_row)

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

        icon = self._NAV_ICONS.get(name, "▸")
        item = QListWidgetItem(f"  {icon}  {name}")
        self.nav_list.addItem(item)
        idx = self.stack.addWidget(widget)
        self._row_to_stack.append(idx)

        # Select the very first real tab.
        if sum(1 for s in self._row_to_stack if s >= 0) == 1:
            self.nav_list.setCurrentRow(len(self._row_to_stack) - 1)
        return idx

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
