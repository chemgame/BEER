"""BEER dialog classes."""
import os, json, re, difflib
import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QDialogButtonBox, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QComboBox, QSpinBox, QCheckBox, QListWidget, QListWidgetItem,
    QFileDialog, QMessageBox, QTextEdit, QPushButton, QTableWidget,
    QTableWidgetItem, QLineEdit, QScrollArea, QWidget, QGridLayout,
    QSizePolicy, QSplitter, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont


# ---------------------------------------------------------------------------
# MutationDialog
# ---------------------------------------------------------------------------

class MutationDialog(QDialog):
    """Dialog to pick a position and a replacement amino acid for a point mutation."""

    _AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")

    def __init__(self, seq_length: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Introduce Point Mutation")
        self.setMinimumWidth(320)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self._pos_spin = QSpinBox()
        self._pos_spin.setRange(1, max(1, seq_length))
        self._pos_spin.setValue(1)
        form.addRow("Position (1-based):", self._pos_spin)

        self._aa_combo = QComboBox()
        self._aa_combo.addItems(self._AA_LIST)
        form.addRow("New amino acid:", self._aa_combo)

        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_mutation(self):
        """Return (pos_0based, new_aa) or None if dialog was rejected."""
        pos_1based = self._pos_spin.value()
        new_aa = self._aa_combo.currentText()
        return (pos_1based - 1, new_aa)


# ---------------------------------------------------------------------------
# ComplexCalculatorDialog
# ---------------------------------------------------------------------------

def _parse_stoichiometry(stoich_str: str) -> dict:
    """Parse stoichiometry string like 'A2B1' or 'AB2C' into {chain_label: count}."""
    result = {}
    tokens = re.findall(r'([A-Z]+)(\d*)', stoich_str.upper())
    for label, count_str in tokens:
        if not label:
            continue
        count = int(count_str) if count_str else 1
        if label in result:
            result[label] += count
        else:
            result[label] = count
    return result


def _estimate_mw(seq: str) -> float:
    """Estimate molecular weight in Da from one-letter sequence."""
    mw_table = {
        'A': 89.09, 'R': 174.20, 'N': 132.12, 'D': 133.10, 'C': 121.16,
        'E': 147.13, 'Q': 146.15, 'G': 75.03, 'H': 155.16, 'I': 131.17,
        'L': 131.17, 'K': 146.19, 'M': 149.21, 'F': 165.19, 'P': 115.13,
        'S': 105.09, 'T': 119.12, 'W': 204.23, 'Y': 181.19, 'V': 117.15,
    }
    water = 18.02
    mw = sum(mw_table.get(aa, 110.0) for aa in seq.upper()) - (len(seq) - 1) * water
    return round(mw, 2)


def _estimate_ext_coeff(seq: str) -> int:
    """Estimate extinction coefficient at 280 nm (reduced cysteines)."""
    s = seq.upper()
    return s.count('W') * 5500 + s.count('Y') * 1490 + s.count('C') * 125


def _estimate_pi(seq: str) -> float:
    """Estimate isoelectric point using simple Henderson-Hasselbalch iteration."""
    pka = {'D': 3.9, 'E': 4.1, 'H': 6.0, 'C': 8.3, 'Y': 10.1, 'K': 10.5, 'R': 12.5}
    nterm_pka = 8.0
    cterm_pka = 3.1
    counts = {aa: seq.upper().count(aa) for aa in pka}

    def charge_at(pH):
        q = 1.0 / (1.0 + 10 ** (pH - nterm_pka))  # N-term
        q -= 1.0 / (1.0 + 10 ** (cterm_pka - pH))  # C-term
        for aa, pk in pka.items():
            n = counts[aa]
            if aa in ('D', 'E', 'C', 'Y'):
                q -= n / (1.0 + 10 ** (pk - pH))
            else:
                q += n / (1.0 + 10 ** (pH - pk))
        return q

    lo, hi = 0.0, 14.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if charge_at(mid) > 0:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2.0, 2)


class ComplexCalculatorDialog(QDialog):
    """Dialog for protein complex stoichiometry and property calculation."""

    result_ready = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Protein Complex Calculator")
        self.setMinimumSize(600, 500)
        self._result = {}

        layout = QVBoxLayout(self)

        # Input area
        input_label = QLabel(
            "Paste multi-FASTA sequences (one chain per entry). "
            "Use chain labels matching the stoichiometry string."
        )
        input_label.setWordWrap(True)
        layout.addWidget(input_label)

        self._fasta_edit = QTextEdit()
        self._fasta_edit.setPlaceholderText(
            ">A\nMSEQUENCEHERE\n>B\nANOTHERCHAIN\n..."
        )
        self._fasta_edit.setMinimumHeight(140)
        layout.addWidget(self._fasta_edit)

        stoich_layout = QHBoxLayout()
        stoich_layout.addWidget(QLabel("Stoichiometry:"))
        self._stoich_edit = QLineEdit()
        self._stoich_edit.setPlaceholderText("e.g. A2B1  or  AB2C")
        stoich_layout.addWidget(self._stoich_edit)
        layout.addLayout(stoich_layout)

        calc_btn = QPushButton("Calculate Complex Properties")
        calc_btn.clicked.connect(self._calculate)
        layout.addWidget(calc_btn)

        # Results table
        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(
            ["Chain ID", "Seq Length", "MW (Da)", "Ext. Coeff (280nm)", "pI"]
        )
        self._table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._table)

        # Summary label
        self._summary_label = QLabel("")
        self._summary_label.setWordWrap(True)
        font = QFont()
        font.setBold(True)
        self._summary_label.setFont(font)
        layout.addWidget(self._summary_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _parse_fasta(self, text: str) -> dict:
        """Parse multi-FASTA text into {name: sequence}."""
        chains = {}
        name = None
        seq_lines = []
        for line in text.splitlines():
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    chains[name] = "".join(seq_lines).upper()
                name = line[1:].split()[0]
                seq_lines = []
            elif line:
                seq_lines.append(line)
        if name is not None:
            chains[name] = "".join(seq_lines).upper()
        return chains

    def _calculate(self):
        fasta_text = self._fasta_edit.toPlainText().strip()
        stoich_str = self._stoich_edit.text().strip()

        if not fasta_text:
            QMessageBox.warning(self, "Input Required", "Please paste FASTA sequences.")
            return

        chains = self._parse_fasta(fasta_text)
        stoich = _parse_stoichiometry(stoich_str) if stoich_str else {k: 1 for k in chains}

        self._table.setRowCount(0)
        total_mw = 0.0
        total_ext = 0
        pi_values = []
        chain_data = []

        for chain_id, seq in chains.items():
            mw = _estimate_mw(seq)
            ext = _estimate_ext_coeff(seq)
            pi = _estimate_pi(seq)
            copies = stoich.get(chain_id, 1)

            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setItem(row, 0, QTableWidgetItem(f"{chain_id} (×{copies})"))
            self._table.setItem(row, 1, QTableWidgetItem(str(len(seq))))
            self._table.setItem(row, 2, QTableWidgetItem(f"{mw:,.2f}"))
            self._table.setItem(row, 3, QTableWidgetItem(str(ext)))
            self._table.setItem(row, 4, QTableWidgetItem(f"{pi:.2f}"))

            total_mw += mw * copies
            total_ext += ext * copies
            pi_values.append(pi)
            chain_data.append({
                "chain_id": chain_id, "copies": copies,
                "seq_len": len(seq), "mw": mw, "ext_coeff": ext, "pi": pi,
            })

        pi_range = f"{min(pi_values):.2f} – {max(pi_values):.2f}" if pi_values else "N/A"
        self._summary_label.setText(
            f"Complex MW: {total_mw:,.2f} Da  |  "
            f"Combined ε₂₈₀: {total_ext}  |  "
            f"pI range: {pi_range}"
        )

        self._result = {
            "chains": chain_data,
            "stoichiometry": stoich,
            "total_mw": total_mw,
            "total_ext_coeff": total_ext,
            "pi_range": (min(pi_values), max(pi_values)) if pi_values else (0, 0),
        }

    def _on_accept(self):
        if self._result:
            self.result_ready.emit(self._result)
        self.accept()


# ---------------------------------------------------------------------------
# FigureComposerDialog
# ---------------------------------------------------------------------------

_LAYOUT_OPTIONS = ["1×1", "1×2", "2×1", "2×2", "2×3", "3×2", "3×3"]


class FigureComposerDialog(QDialog):
    """Dialog for multi-panel figure export."""

    composition_ready = pyqtSignal(str, list)

    def __init__(self, available_graphs: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Figure Composer")
        self.setMinimumSize(700, 500)
        self._available_graphs = available_graphs
        self._slot_combos = []

        main_layout = QHBoxLayout(self)

        # Left panel: available graphs list
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.addWidget(QLabel("Available Graphs:"))
        self._graph_list = QListWidget()
        for g in available_graphs:
            self._graph_list.addItem(QListWidgetItem(g))
        left_layout.addWidget(self._graph_list)
        main_layout.addWidget(left_widget, 1)

        # Right panel: layout selector + slot combos + preview
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        layout_row = QHBoxLayout()
        layout_row.addWidget(QLabel("Layout:"))
        self._layout_combo = QComboBox()
        self._layout_combo.addItems(_LAYOUT_OPTIONS)
        self._layout_combo.currentIndexChanged.connect(self._rebuild_slots)
        layout_row.addWidget(self._layout_combo)
        layout_row.addStretch()
        right_layout.addLayout(layout_row)

        self._slots_widget = QWidget()
        self._slots_grid = QGridLayout(self._slots_widget)
        right_layout.addWidget(self._slots_widget)

        # Preview placeholder
        self._preview_label = QLabel("[ Preview area — select graphs above ]")
        self._preview_label.setAlignment(Qt.AlignCenter)
        self._preview_label.setFrameShape(QFrame.StyledPanel)
        self._preview_label.setMinimumHeight(120)
        right_layout.addWidget(self._preview_label)

        compose_btn = QPushButton("Compose & Export")
        compose_btn.clicked.connect(self._on_compose)
        right_layout.addWidget(compose_btn)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        right_layout.addWidget(buttons)

        main_layout.addWidget(right_widget, 2)

        self._rebuild_slots()

    def _parse_layout(self, layout_str: str):
        """Return (rows, cols) from a layout string like '2×3'."""
        parts = layout_str.replace("x", "×").split("×")
        rows = int(parts[0].strip())
        cols = int(parts[1].strip())
        return rows, cols

    def _rebuild_slots(self):
        # Clear existing slot combos
        for i in reversed(range(self._slots_grid.count())):
            widget = self._slots_grid.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        self._slot_combos = []

        layout_str = self._layout_combo.currentText()
        rows, cols = self._parse_layout(layout_str)
        none_option = "— None —"
        options = [none_option] + self._available_graphs

        for r in range(rows):
            for c in range(cols):
                combo = QComboBox()
                combo.addItems(options)
                self._slots_grid.addWidget(QLabel(f"Slot ({r+1},{c+1}):"), r * 2, c * 2)
                self._slots_grid.addWidget(combo, r * 2 + 1, c * 2)
                self._slot_combos.append(combo)

    def _on_compose(self):
        layout_str = self._layout_combo.currentText()
        titles = []
        for combo in self._slot_combos:
            text = combo.currentText()
            titles.append(text if text != "— None —" else None)
        self.composition_ready.emit(layout_str, titles)
        self._preview_label.setText(
            f"Layout: {layout_str}  |  Slots: {[t or 'None' for t in titles]}"
        )

    def get_composition(self):
        """Return (layout_str, list_of_titles_or_None)."""
        layout_str = self._layout_combo.currentText()
        titles = []
        for combo in self._slot_combos:
            text = combo.currentText()
            titles.append(text if text != "— None —" else None)
        return layout_str, titles


# ---------------------------------------------------------------------------
# MSADialog
# ---------------------------------------------------------------------------

def _simple_progressive_msa(seqs: list) -> list:
    """Very simple progressive pairwise MSA fallback using difflib."""
    if not seqs:
        return []
    if len(seqs) == 1:
        return list(seqs)

    def align_pair(s1: str, s2: str):
        """Align two sequences using difflib opcodes, inserting gaps."""
        matcher = difflib.SequenceMatcher(None, s1, s2, autojunk=False)
        a_out, b_out = [], []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                a_out.append(s1[i1:i2])
                b_out.append(s2[j1:j2])
            elif tag == 'replace':
                block_a = s1[i1:i2]
                block_b = s2[j1:j2]
                max_len = max(len(block_a), len(block_b))
                a_out.append(block_a.ljust(max_len, '-'))
                b_out.append(block_b.ljust(max_len, '-'))
            elif tag == 'delete':
                block_a = s1[i1:i2]
                a_out.append(block_a)
                b_out.append('-' * len(block_a))
            elif tag == 'insert':
                block_b = s2[j1:j2]
                a_out.append('-' * len(block_b))
                b_out.append(block_b)
        return "".join(a_out), "".join(b_out)

    aligned = [seqs[0]]
    for seq in seqs[1:]:
        a_aln, b_aln = align_pair(aligned[0], seq)
        # Propagate gaps into previously aligned sequences
        updated = []
        for prev in aligned:
            new_prev = []
            pi = 0
            for ch in a_aln:
                if ch == '-':
                    new_prev.append('-')
                else:
                    new_prev.append(prev[pi] if pi < len(prev) else '-')
                    pi += 1
            updated.append("".join(new_prev))
        aligned = updated
        aligned.append(b_aln)

    # Pad all to same length
    max_len = max(len(s) for s in aligned)
    aligned = [s.ljust(max_len, '-') for s in aligned]
    return aligned


class MSADialog(QDialog):
    """Dialog for multiple sequence alignment input and conservation display."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Multiple Sequence Alignment")
        self.setMinimumSize(560, 440)
        self._alignment = []
        self._names = []

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(
            "Paste multi-FASTA sequences (aligned or unaligned):"
        ))

        self._fasta_edit = QTextEdit()
        self._fasta_edit.setPlaceholderText(
            ">seq1\nMACDEFGHIK\n>seq2\nMAC-EFGHIK\n..."
        )
        self._fasta_edit.setMinimumHeight(180)
        layout.addWidget(self._fasta_edit)

        self._prealigned_cb = QCheckBox("Input is pre-aligned")
        self._prealigned_cb.setChecked(False)
        layout.addWidget(self._prealigned_cb)

        align_btn = QPushButton("Align && Show Conservation")
        align_btn.clicked.connect(self._do_align)
        layout.addWidget(align_btn)

        self._result_label = QLabel("")
        self._result_label.setWordWrap(True)
        layout.addWidget(self._result_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _parse_fasta(self, text: str):
        names, seqs = [], []
        name, seq_lines = None, []
        for line in text.splitlines():
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    names.append(name)
                    seqs.append("".join(seq_lines).upper())
                name = line[1:].strip() or f"seq{len(names)+1}"
                seq_lines = []
            elif line:
                seq_lines.append(line)
        if name is not None:
            names.append(name)
            seqs.append("".join(seq_lines).upper())
        return names, seqs

    def _do_align(self):
        text = self._fasta_edit.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Input Required", "Please paste FASTA sequences.")
            return

        names, seqs = self._parse_fasta(text)
        if not seqs:
            QMessageBox.warning(self, "Parse Error", "Could not parse any sequences.")
            return

        is_aligned = self._prealigned_cb.isChecked()
        if is_aligned:
            aligned_seqs = seqs
        else:
            aligned_seqs = _simple_progressive_msa(seqs)

        self._alignment = aligned_seqs
        self._names = names

        # Compute conservation summary
        if aligned_seqs:
            aln_len = len(aligned_seqs[0])
            conserved = 0
            for col in range(aln_len):
                col_chars = set(s[col] for s in aligned_seqs if col < len(s)) - {'-'}
                if len(col_chars) == 1:
                    conserved += 1
            pct = 100.0 * conserved / aln_len if aln_len else 0
            self._result_label.setText(
                f"Aligned {len(aligned_seqs)} sequences, length {aln_len} columns. "
                f"Conserved positions: {conserved} ({pct:.1f}%)"
            )
        else:
            self._result_label.setText("Alignment produced no results.")

    def get_alignment(self):
        """Return (list_of_sequences, list_of_names, is_aligned)."""
        is_aligned = self._prealigned_cb.isChecked() or bool(self._alignment)
        return self._alignment, self._names, is_aligned


# ---------------------------------------------------------------------------
# TruncationDialog
# ---------------------------------------------------------------------------

class TruncationDialog(QDialog):
    """Dialog for truncation series settings."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Truncation Series Settings")
        self.setMinimumWidth(300)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self._step_spin = QSpinBox()
        self._step_spin.setRange(1, 20)
        self._step_spin.setValue(10)
        self._step_spin.setSuffix(" %")
        form.addRow("Step size:", self._step_spin)

        layout.addLayout(form)

        self._nterm_cb = QCheckBox("N-terminal truncations")
        self._nterm_cb.setChecked(True)
        layout.addWidget(self._nterm_cb)

        self._cterm_cb = QCheckBox("C-terminal truncations")
        self._cterm_cb.setChecked(True)
        layout.addWidget(self._cterm_cb)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_settings(self):
        """Return (step_pct, do_nterm, do_cterm)."""
        return (
            self._step_spin.value(),
            self._nterm_cb.isChecked(),
            self._cterm_cb.isChecked(),
        )
