"""The bwrap sandbox for the judge phase: strategy code runs without network and can write only lab state.

Production (`LAB_SANDBOX=1`, set by the systemd unit) runs `lab tick --phase judge` inside bubblewrap:
- the whole file system read-only (`--ro-bind / /`), fresh /tmp, /dev, /proc,
- no network (`--unshare-net`), own PID namespace, killed with its parent,
- writable only: LAB_HOME (lab.db, cache, data) and the repo's `lab/hypotheses`, `lab/knowledge` (rendered
  cards and knowledge) - the strategies, gates.yaml, the catalog and the framework stay read-only.
RAM and CPU limits come from the systemd unit (MemoryMax, CPUQuota). The ingest phase (Archivist fetchers,
network) runs outside, in the orchestrator process.
"""

import os
import shutil
import subprocess
import sys

from lab.framework.paths import LabPaths

ENV_KEEP = ("LAB_HOME", "LAB_DATA_ROOT", "LAB_WORKSPACES", "PATH", "HOME", "LANG")


def command(paths: LabPaths, argv: list[str]) -> list[str]:
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        raise RuntimeError("bwrap not installed (apt install bubblewrap)")
    cmd = [bwrap, "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp",
           "--unshare-net", "--unshare-pid", "--die-with-parent", "--new-session"]
    for rw in (paths.home, paths.lab / "hypotheses", paths.lab / "knowledge"):
        rw.mkdir(parents=True, exist_ok=True)
        cmd += ["--bind", str(rw), str(rw)]
    return cmd + ["--"] + argv


def judge(paths: LabPaths, timeout_s: float = 3 * 3600) -> tuple[int, str]:
    """Run `lab tick --phase judge` in the sandbox. Returns (exit code, output)."""
    argv = [sys.executable, "-m", "lab.framework.cli", "tick", "--phase", "judge"]
    env = {k: os.environ[k] for k in ENV_KEEP if k in os.environ} | {"LAB_HOME": str(paths.home)}
    p = subprocess.run(command(paths, argv), capture_output=True, text=True, timeout=timeout_s, env=env,
                       cwd=str(paths.lab.parent))
    return p.returncode, (p.stdout + p.stderr)[-20000:]
