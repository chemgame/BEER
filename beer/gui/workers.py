"""Background worker threads for BEER GUI."""
from PyQt5.QtCore import QThread, pyqtSignal


class AnalysisWorker(QThread):
    """Non-blocking protein analysis in a QThread. Emits finished(dict) or error(str)."""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, seq, pH, window_size, use_reducing, pka):
        super().__init__()
        self.seq = seq
        self.pH = pH
        self.window_size = window_size
        self.use_reducing = use_reducing
        self.pka = pka

    def run(self):
        try:
            from beer.analysis.core import AnalysisTools
            data = AnalysisTools.analyze_sequence(
                self.seq, self.pH, self.window_size,
                self.use_reducing, self.pka
            )
            self.finished.emit(data)
        except Exception as exc:
            self.error.emit(str(exc))
