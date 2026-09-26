"""Phase 5: one-time final evaluation of the frozen methodology on the holdout (spec §9.7, §15).

Usage: python scripts/final_evaluation.py sharadar_YYYY-MM-DD

1. requires a clean git tree (the evaluated code is exactly the current commit),
2. freezes the methodology (hash of frozen_defaults.yaml + commit + Phase 3 references),
3. opens the vault once for this snapshot (logged in runs/vault/vault.log),
4. runs the grid over the whole history and the walk-forward through the holdout; the
   2005-2019 part must reproduce the pre-registered Phase 3 run.
Re-running after a crash resumes inside the same logged session; a different commit is refused.
"""

import os
import subprocess
import sys
from pathlib import Path

from qlab.pipeline import CONFIG, FINAL_ENV, load_config, vault_for
from qlab.validation.registry import config_hash, current_commit
from qlab.validation.vault import VaultError

DEV_GRID_RUN = "a03e443786b83e08"   # Phase 2 grid on 2000-2019
DEV_WFO_RUN = "e166501421d4bd05"    # Phase 3 walk-forward (pre-registered)

snapshot = sys.argv[1]
dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout
if dirty.strip():
    sys.exit(f"working tree is not clean:\n{dirty}")
commit = current_commit()
cfg = load_config()
methodology = {"frozen_defaults": CONFIG.read_text(), "commit": commit,
               "dev_grid_run": DEV_GRID_RUN, "dev_wfo_run": DEV_WFO_RUN}
m_hash = config_hash(methodology)
vault = vault_for(cfg)

if vault.lock_path.exists():
    print(f"methodology already frozen: {vault.lock_path.read_text()}")
else:
    vault.freeze(m_hash)
    print(f"frozen methodology {m_hash} at commit {commit}")
try:
    vault.check_final_session(m_hash, snapshot, commit)
    print("resuming the logged final session")
except VaultError:
    vault.open_final(m_hash, snapshot, commit)
    print(f"vault opened for {snapshot} (logged)")

env = {**os.environ, FINAL_ENV: m_hash}
py = [sys.executable]
grid = subprocess.run(py + ["scripts/run_grid.py", snapshot, "--final"], env=env,
                      capture_output=True, text=True)
print(grid.stdout[-2000:], grid.stderr[-2000:])
if grid.returncode:
    sys.exit("final grid run failed")
run_dir = Path(grid.stdout.strip().splitlines()[-1].split()[1])
wfo = subprocess.run(py + ["scripts/run_wfo.py", snapshot, run_dir.name, "--final",
                           "--replicate", DEV_WFO_RUN], env=env, capture_output=True, text=True)
Path("runs/final_wfo.log").write_text(wfo.stdout + wfo.stderr)
print(wfo.stdout[-6000:], wfo.stderr[-3000:])
if wfo.returncode:
    sys.exit("final walk-forward failed")
