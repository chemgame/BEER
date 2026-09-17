"""`beer` with arguments must reach the CLI, never the GUI.

`beer --help`, a typo or an unknown flag goes to the CLI; only Qt options
launch the desktop window. These tests pin the routing without importing Qt.
"""
from __future__ import annotations

import sys

import pytest

from beer.gui import app


@pytest.fixture(autouse=True)
def _no_side_effects(monkeypatch):
    # main() normally enables faulthandler and writes ~/.beer/beer.log
    monkeypatch.setattr(app, "_harden_native_runtime", lambda: None)
    monkeypatch.setattr(app, "_setup_logging", lambda: None)


def _run_main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["beer", *argv])
    app.main()


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_help_prints_usage_and_exits_cleanly(monkeypatch, capsys, flag):
    with pytest.raises(SystemExit) as exc:
        _run_main(monkeypatch, flag)
    assert exc.value.code == 0
    assert "usage: beer" in capsys.readouterr().out


def test_unknown_flag_is_an_error_not_a_window(monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc:
        _run_main(monkeypatch, "--no-such-flag")
    assert exc.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_unknown_command_is_an_error(monkeypatch):
    with pytest.raises(SystemExit) as exc:
        _run_main(monkeypatch, "analyse-typo")
    assert exc.value.code == 2


@pytest.mark.parametrize("argv", [["--version"], ["-V"], ["version"]])
def test_version_forms(monkeypatch, capsys, argv):
    import beer
    _run_main(monkeypatch, *argv)
    assert capsys.readouterr().out.strip() == f"BEER v{beer.__version__}"


@pytest.mark.parametrize("arg", ["-platform", "-style", "-style=fusion", "-reverse",
                                 "-stylesheet=theme.qss"])
def test_qt_options_still_mean_gui(arg):
    assert app._is_qt_option(arg)


@pytest.mark.parametrize("arg", ["--help", "-h", "analyze", "--platform", "-x"])
def test_non_qt_arguments_go_to_the_cli(arg):
    assert not app._is_qt_option(arg)
