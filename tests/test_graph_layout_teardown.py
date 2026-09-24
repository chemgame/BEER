"""Replacing a graph must delete the old widgets through Qt, not leave them to Python.

Widgets detached from a layout but kept alive by Python can be destroyed by a garbage
collection at any moment, including while Qt is building the next graph. That showed up as
attribute errors from inside the matplotlib toolbar.
"""
import gc

import pytest

pytest.importorskip("PySide6")


def _app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_clear_layout_deletes_widgets_and_closes_figures(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = _app()
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from PySide6.QtWidgets import QVBoxLayout, QWidget
    from shiboken6 import isValid

    from beer.gui.main_window import ProteinAnalyzerGUI as MainWindow

    host = QWidget()
    layout = QVBoxLayout(host)
    fig = plt.figure()
    canvas = FigureCanvasQTAgg(fig)
    inner = QWidget()
    layout.addWidget(canvas)
    layout.addWidget(inner)

    MainWindow._clear_layout(MainWindow, layout)
    assert layout.count() == 0
    from PySide6.QtCore import QEvent
    app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()
    gc.collect()
    assert not isValid(canvas), "old canvas should be destroyed by Qt"
    assert not isValid(inner), "old widget should be destroyed by Qt"
    assert not plt.fignum_exists(fig.number), "figure should be closed"


def test_toolbar_failure_leaves_a_usable_toolbar(monkeypatch):
    """A toolbar that cannot be built must not take the graph down with it."""
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    _app()
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from PySide6.QtWidgets import QToolBar

    from beer.gui import main_window as mw

    def boom(*a, **k):
        raise AttributeError("simulated Qt failure")

    monkeypatch.setattr(mw, "NavigationToolbar2QT", boom)

    from PySide6.QtWidgets import QWidget

    class _Host(QWidget):
        """Stands in for the main window: the helper only needs a parent widget."""
        def _save_canvas_figure(self, canvas):
            pass

    canvas = FigureCanvasQTAgg(plt.figure())
    bar = mw.ProteinAnalyzerGUI._make_graph_toolbar(_Host(), canvas)
    assert isinstance(bar, QToolBar)
    assert [a.text() for a in bar.actions()] == ["Save figure…"]
    plt.close("all")
