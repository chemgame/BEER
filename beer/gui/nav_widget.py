"""BEER left-sidebar navigation widget (PySide6)."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QListWidget, QListWidgetItem,
    QFrame, QStackedWidget,
)
from PySide6.QtCore import Qt


class NavTabWidget(QWidget):
    """Left-sidebar navigation that is a drop-in replacement for QTabWidget.
    Implements the subset of QTabWidget API used in this app."""

    _NAV_ICONS = {
        "Analysis":            "\U0001f9ea",
        "Graphs":              "\U0001f4ca",
        "Structure":           "\U0001f52c",
        "BLAST":               "\U0001f50d",
        "Compare":             "\u2696",
        "Multichain Analysis": "\U0001f4cb",
        "Truncation":          "\u2702",
        "MSA":                 "\U0001f500",
        "Complex":             "\u269b",
        "Settings":            "\u2699",
        "Help":                "\u2753",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.nav_list = QListWidget()
        self.nav_list.setObjectName("nav_bar")
        self.nav_list.setFixedWidth(152)
        self.nav_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(self.nav_list)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFrameShadow(QFrame.Shadow.Plain)
        sep.setObjectName("nav_sep")
        outer.addWidget(sep)

        self.stack = QStackedWidget()
        outer.addWidget(self.stack, 1)

        self.nav_list.currentRowChanged.connect(self.stack.setCurrentIndex)

    def addTab(self, widget: QWidget, name: str) -> int:
        icon = self._NAV_ICONS.get(name, "\u25b8")
        item = QListWidgetItem(f"  {icon}  {name}")
        self.nav_list.addItem(item)
        idx = self.stack.addWidget(widget)
        if self.nav_list.count() == 1:
            self.nav_list.setCurrentRow(0)
        return idx

    def setCurrentIndex(self, idx: int):
        self.nav_list.setCurrentRow(idx)

    def currentIndex(self) -> int:
        return self.nav_list.currentRow()

    def currentWidget(self) -> QWidget:
        return self.stack.currentWidget()

    def widget(self, idx: int) -> QWidget:
        return self.stack.widget(idx)

    def count(self) -> int:
        return self.stack.count()


__all__ = ["NavTabWidget"]
