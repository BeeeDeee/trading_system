"""Freeze and update the separate BTC/H2 paper mix tracking (not H1/H2 decision rules).

    uv run python scripts/r18_mix_forward.py freeze
    uv run python scripts/r18_mix_forward.py status
"""

import hashlib
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from qlab.research17 import forward as F
from qlab.research18 import paper_mixes as M, setup as T
from qlab.validation.registry import TrialRegistry, current_commit

LOCK = Path("docs/research18/paper_mix_lock.json")
RULE_FILES = ["docs/research18/BTC_H2_PAPER.md", "src/qlab/research18/paper_mixes.py",
              "scripts/r18_mix_forward.py"]


def rule_hash() -> str:
    h = hashlib.sha256()
    for name in RULE_FILES:
        h.update(name.encode() + b"\0" + Path(name).read_bytes() + b"\0")
    return h.hexdigest()


def check_lock() -> None:
    if not LOCK.exists():
        raise SystemExit("paper mix rules are not frozen: run `freeze` first")
    lock = json.loads(LOCK.read_text())
    if lock.get("rule_hash") != rule_hash():
        raise SystemExit("paper mix rules changed since freeze; create a new preregistered series")


action = sys.argv[1] if len(sys.argv) > 1 else ""
if action == "freeze":
    if LOCK.exists():
        raise SystemExit(f"{LOCK} already exists")
    LOCK.write_text(json.dumps({"rule_hash": rule_hash(), "files": RULE_FILES,
                                "commit": current_commit(), "frozen_at": datetime.now(timezone.utc).isoformat()}, indent=1))
    TrialRegistry(T.REGISTRY).record(
        "candidates", {"study": "research18_btc_h2_paper_mixes", "period_start": M.START.isoformat(),
                       "variants": M.VARIANTS, "btc_weight": M.BTC_WEIGHT, "h2_weight": M.H2_WEIGHT,
                       "rv_window": M.RV_WINDOW, "rv_target": M.RV_TARGET, "dd_window": M.DD_WINDOW,
                       "dd_trigger": M.DD_TRIGGER, "dd_reduced": M.DD_REDUCED,
                       "dd_resume_sma": M.DD_RESUME_SMA, "rebalance_cost": M.REBALANCE_COST},
        n_configs=len(M.VARIANTS), note="Separate descriptive paper tracking of 3 BTC/H2 portfolio mixes")
    print(f"frozen {LOCK}: {json.loads(LOCK.read_text())['rule_hash']}")
    raise SystemExit(0)
if action != "status":
    raise SystemExit(__doc__)
check_lock()

files = sorted(T.RESULTS.glob("status_2*.json"))
if not files:
    raise SystemExit("base research 18 status is not available yet")
base = json.loads(files[-1].read_text())
asof = date.fromisoformat(base["asof"])
if asof < M.START:
    print(f"waiting: paper mixes start {M.START}, latest base status is {asof}")
    raise SystemExit(0)

# Only BTC spot is needed; load the exact checked forward panel source used by research 17.
btc = F.build_panel(F.kline_files("spot", "BTCUSDT"))
btc = F.truncate(btc, asof)
out = M.run(base, btc) | {"commit": current_commit(), "rule_hash": json.loads(LOCK.read_text())["rule_hash"]}
dest = T.RESULTS / f"mix_status_{asof}.json"
T.RESULTS.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps(out, indent=1, default=float))
print(json.dumps({k: v for k, v in out.items() if k not in ("daily",)}, indent=1))
print(f"saved {dest}")
