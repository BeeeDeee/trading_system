"""The judge sandbox: no network, read-only repo except the rendered cards and knowledge, writable LAB_HOME."""

import shutil
import subprocess
import sys

import pytest

from lab.framework import sandbox
from lab.framework.paths import sandbox_paths

pytestmark = pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap not installed")

PROBE = """
import socket, sys, pathlib
home, strategies = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
(home / "ok.txt").write_text("written")
for what, f in (("network", lambda: socket.create_connection(("1.1.1.1", 53), timeout=3)),
                ("strategies", lambda: (strategies / "evil.py").write_text("x")),
                ("homedir", lambda: (pathlib.Path.home() / "evil.txt").write_text("x"))):
    try:
        f()
        print(what, "ALLOWED")
    except OSError:
        print(what, "blocked")
"""


def test_sandbox_blocks_network_and_writes(tmp_path):
    paths = sandbox_paths(tmp_path)
    paths.home.mkdir(parents=True, exist_ok=True)
    out = subprocess.run(sandbox.command(paths, [sys.executable, "-c", PROBE, str(paths.home),
                                                 str(paths.strategies)]), capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["network", "blocked", "strategies", "blocked", "homedir", "blocked"]
    assert (paths.home / "ok.txt").read_text() == "written"
