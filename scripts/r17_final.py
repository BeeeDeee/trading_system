"""Research 17, the one test of the fixed LEAD rule on altcoins (prereg §3-9). Runs ONCE through the vault.

    uv run python scripts/r17_final.py binance_2026-10-03 freeze
    QLAB_R17_METHODOLOGY=<hash> uv run python scripts/r17_final.py binance_2026-10-03 run

Every follower starts with an empty book at 2020-05-01 00:00 UTC (prereg §7); indicators use earlier data.
"""

import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from qlab.research17 import evaluate as E
from qlab.research17.setup import FINAL_ENV, START, load, registry, vault
from qlab.validation.registry import current_commit

snapshot, action = sys.argv[1], sys.argv[2]
FILES = ["docs/research17/PREREGISTRATION.md", "scripts/r17_final.py",
         "src/qlab/research15/strategy.py", "src/qlab/research15/universe.py",
         *sorted(str(f) for f in Path("src/qlab/research16").glob("*.py")),
         *sorted(str(f) for f in Path("src/qlab/research17").glob("*.py"))]


def methodology_hash() -> str:
    h = hashlib.sha256()
    for f in FILES:
        h.update(f.encode() + b"\0" + Path(f).read_bytes() + b"\0")
    return h.hexdigest()


if action == "freeze":
    print(f"frozen: {(mh := methodology_hash())}")
    vault().freeze(mh)
    sys.exit(0)
if action != "run":
    raise SystemExit(__doc__)
mh = os.environ.get(FINAL_ENV, "")
if mh != methodology_hash():
    raise SystemExit("methodology changed since freeze (or env var missing)")
vault().open_final(mh, snapshot, current_commit())
registry().record("candidates", {"study": "research17", **asdict(E.P)}, note="r17 LEAD K3 long on altcoins")
registry().record("methodology_eval", {"study": "research17", "stage": "final", "hash": mh}, note="r17 one-shot")

manifest = json.loads((Path("data/raw") / snapshot / "r15_manifest.json").read_text())
members = manifest["members"]
first_month = f"{START.year}-{START.month:02d}"
syms = sorted({s for m, v in members.items() if m >= first_month for s in v})
out = {"snapshot": snapshot, "commit": current_commit(), "methodology_hash": mh} | E.run(
    lambda sym, market: load(snapshot, sym, market), syms, members, verbose=True)
Path("docs/research17/results").mkdir(parents=True, exist_ok=True)
Path("docs/research17/results/final.json").write_text(json.dumps(out, indent=1, default=float))
print(json.dumps({k: v for k, v in out.items() if k != "robust"}, indent=1, default=float))
