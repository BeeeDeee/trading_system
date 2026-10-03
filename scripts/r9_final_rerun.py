"""Rerun of scripts/r9_final.py after it crashed AFTER the vault opened (json could not encode numpy bool).

The frozen script is executed unchanged (same methodology hash, same commit). Only two things differ:
json.dumps converts numpy scalars, and Vault.open_final logs a 'rerun' line instead of refusing the
second open (it still requires the original open of the same hash/commit/snapshot). Logged in
docs/research9/PREREGISTRATION.md §10.
"""

import json
import runpy
import sys
from datetime import datetime, timezone

import numpy as np

from qlab.validation import vault as V

_dumps = json.dumps


def dumps(obj, **kw):
    kw.setdefault("default", lambda o: o.item() if isinstance(o, np.generic) else str(o))
    return _dumps(obj, **kw)


def open_again(self, methodology_hash, snapshot_id, commit):
    self.check_final_session(methodology_hash, snapshot_id, commit)          # original open must exist
    with self.log_path.open("a") as f:
        f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "snapshot_id": snapshot_id,
                            "commit": commit, "methodology_hash": methodology_hash,
                            "rerun": "crash after open: json could not encode numpy bool"}) + "\n")


json.dumps = dumps
V.Vault.open_final = open_again
sys.argv = ["scripts/r9_final.py", *sys.argv[1:]]
runpy.run_path("scripts/r9_final.py", run_name="__main__")
