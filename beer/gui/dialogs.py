"""BEER GUI dialogs (PySide6)."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog, QFormLayout, QSpinBox, QLabel, QComboBox,
    QDialogButtonBox, QVBoxLayout, QHBoxLayout, QWidget,
    QGridLayout,
)

from beer.constants import VALID_AMINO_ACIDS


class MutationDialog(QDialog):
    """Simple dialog: pick a position and a replacement amino acid."""

    def __init__(self, seq: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Mutate Residue")
        self.setMinimumWidth(320)
        self._seq = seq
        layout = QFormLayout(self)
        layout.setSpacing(10)

        self.pos_spin = QSpinBox()
        self.pos_spin.setRange(1, len(seq))
        self.pos_spin.setValue(1)
        self.pos_spin.valueChanged.connect(self._update_current)
        layout.addRow("Position (1-based):", self.pos_spin)

        self.current_lbl = QLabel(seq[0] if seq else "?")
        self.current_lbl.setStyleSheet(
            "font-weight:700; color:#4361ee; font-family:monospace; font-size:14pt;"
        )
        layout.addRow("Current residue:", self.current_lbl)

        self.aa_combo = QComboBox()
        self.aa_combo.addItems(sorted(VALID_AMINO_ACIDS))
        layout.addRow("Replace with:", self.aa_combo)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)

    def _update_current(self, pos):
        aa = self._seq[pos - 1] if 0 <= pos - 1 < len(self._seq) else "?"
        self.current_lbl.setText(aa)

    def get_mutation(self):
        """Returns (position_0based, new_aa)."""
        return self.pos_spin.value() - 1, self.aa_combo.currentText()


class _FigureComposerDialog(QDialog):
    """Dialog to build a multi-panel figure from existing graph canvases."""

    _LAYOUTS = ["1\u00d71", "1\u00d72", "2\u00d71", "2\u00d72", "2\u00d73", "3\u00d72", "3\u00d73"]

    def __init__(self, available_titles: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Figure Composer")
        self.setMinimumSize(560, 400)
        layout = QVBoxLayout(self)

        top = QHBoxLayout()
        top.addWidget(QLabel("Layout:"))
        self.layout_combo = QComboBox()
        self.layout_combo.addItems(self._LAYOUTS)
        self.layout_combo.setCurrentText("2\u00d72")
        self.layout_combo.currentTextChanged.connect(self._rebuild_slots)
        top.addWidget(self.layout_combo)
        top.addStretch()
        layout.addLayout(top)

        self._available = ["\u2014 None \u2014"] + available_titles
        self._slots_frame = QWidget()
        self._slots_grid = QGridLayout(self._slots_frame)
        layout.addWidget(self._slots_frame)

        self._slot_combos: list = []
        self._rebuild_slots(self.layout_combo.currentText())

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _rebuild_slots(self, layout_str: str):
        try:
            nr, nc = [int(x) for x in layout_str.split("\u00d7")]
        except Exception:
            nr, nc = 2, 2
        # Clear old
        for c in self._slot_combos:
            c.setParent(None)
        self._slot_combos.clear()
        for i in range(nr * nc):
            cb = QComboBox()
            cb.addItems(self._available)
            self._slot_combos.append(cb)
            self._slots_grid.addWidget(QLabel(f"Panel {chr(ord('A') + i)}:"), i // nc, (i % nc) * 2)
            self._slots_grid.addWidget(cb, i // nc, (i % nc) * 2 + 1)

    def get_composition(self):
        layout_str = self.layout_combo.currentText()
        titles = []
        for cb in self._slot_combos:
            t = cb.currentText()
            titles.append(None if t == "\u2014 None \u2014" else t)
        return layout_str, titles


__all__ = ["MutationDialog", "_FigureComposerDialog"]
