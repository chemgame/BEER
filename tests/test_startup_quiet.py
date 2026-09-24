"""BEER should start without the ESM C notices about optional CUDA kernels."""
import subprocess
import sys


def _run(env_extra):
    import os
    env = dict(os.environ)
    env.pop("BEER_LOG_ESM", None)
    env.update(env_extra)
    return subprocess.run([sys.executable, "-c", "import beer"], capture_output=True, text=True,
                          env=env)


def test_import_is_quiet_by_default():
    out = _run({})
    assert out.returncode == 0, out.stderr
    assert "Transformer Engine is not installed" not in out.stderr
    assert "neither xformers nor flash-attn" not in out.stderr


def test_notices_can_be_switched_back_on():
    out = _run({"BEER_LOG_ESM": "1"})
    assert out.returncode == 0, out.stderr
    assert "Transformer Engine is not installed" in out.stderr
