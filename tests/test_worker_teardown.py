"""Retiring a worker must not depend on what its completion signals carry.

The workers do not agree on signatures. AnalysisWorker declares finished(dict) and error(str).
AISectionWorker declares neither finished (so QThread's no-argument signal applies) nor a
one-argument error, using error(str, str) instead. A teardown callback written for one shape
raises TypeError on the other, and the exception fires inside the worker's own teardown, which
is where results such as the transmembrane Viterbi topology are handed back.
"""
import pytest


def _teardown_callback(registry, worker):
    """The callback shape used to retire a worker, matching _discard_worker."""
    return lambda *_a, w=worker: registry.discard(w)


@pytest.mark.parametrize(
    "emitted",
    [
        (),                                 # QThread.finished, no payload
        ({"seq": "ACDE"},),                 # AnalysisWorker.finished(dict)
        ("some error",),                    # AnalysisWorker.error(str)
        ("section", "some error"),          # AISectionWorker.error(str, str)
        ("section", [0.1, 0.2]),            # AISectionWorker.result_ready(str, list)
    ],
)
def test_teardown_callback_accepts_any_signal_arity(emitted):
    worker = object()
    registry = {worker}
    callback = _teardown_callback(registry, worker)
    callback(*emitted)
    assert worker not in registry, "worker was not retired for payload %r" % (emitted,)


def test_single_argument_callback_would_break_on_qthread_finished():
    """Why the variadic form is needed: the old shape fails QThread's bare finished."""
    worker = object()
    registry = {worker}
    old_style = lambda _data, w=worker: registry.discard(w)  # noqa: E731
    with pytest.raises(TypeError):
        old_style()
    assert worker in registry
