"""Research 20 forward test of F1 (H1b) and F2 (H2) (docs/research20/PREREGISTRATION.md). Data come from the daily
snapshots (`scripts/r20_download.py`, cron); this script only computes.

    uv run python scripts/r20_forward.py freeze    # once: lock the rule files, register N = 2 (commit forward_lock.json)
    uv run python scripts/r20_forward.py status    # descriptive metrics up to the last complete day (never a decision)
    uv run python scripts/r20_forward.py final     # the pre-registered decision, only at the dates of §6
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

from qlab.research17.forward import rule_hash
from qlab.research20 import evaluate as E
from qlab.research20 import forward as F
from qlab.research20 import setup as S
from qlab.validation.registry import TrialRegistry, current_commit

RULE_FILES = sorted(str(q) for q in Path("src/qlab/research20").glob("*.py")) + sorted(
    str(q) for q in Path("src/qlab/research19").glob("*.py")) + [
    "src/qlab/engine/vector.py", "src/qlab/research5/sim.py", "docs/research20/PREREGISTRATION.md",
    "scripts/r20_forward.py"]
action = sys.argv[1] if len(sys.argv) > 1 else ""


def check_lock() -> dict:
    if not S.LOCK.exists():
        raise SystemExit("rule files are not frozen: run `freeze` first")
    lock = json.loads(S.LOCK.read_text())
    if lock["rule_hash"] != rule_hash(lock["files"]):
        raise SystemExit("rule files changed since the freeze (forward_lock.json); see PREREGISTRATION.md §12")
    return lock


if action == "freeze":
    if S.LOCK.exists():
        raise SystemExit(f"{S.LOCK} already exists")
    S.LOCK.parent.mkdir(parents=True, exist_ok=True)
    S.LOCK.write_text(json.dumps({"rule_hash": rule_hash(RULE_FILES), "files": RULE_FILES, "commit": current_commit(),
                                  "frozen_at": datetime.now(timezone.utc).isoformat()}, indent=1))
    TrialRegistry(S.REGISTRY).record("candidates", {"study": "research20", "hypotheses": ["F1_h1b", "F2_h2"]},
                                     n_configs=2, note="r20 forward: F1 = research 19 H1b, F2 = H2")
    print(f"frozen: {json.loads(S.LOCK.read_text())['rule_hash']}")
    sys.exit(0)
if action not in ("status", "final"):
    raise SystemExit(__doc__)
lock = check_lock()
p, cash, tickers, t0 = F.build_panel()
end = len(p.dates)
prov = {"commit": current_commit(), "rule_hash": rule_hash(lock["files"]), "created_at": datetime.now(timezone.utc).isoformat()}
S.RESULTS.mkdir(parents=True, exist_ok=True)

if action == "status":
    out = F.forward_metrics(p, cash, tickers, t0, end) | prov | {"note": "descriptive; the decision is made only by `final`"}
    dest = S.RESULTS / f"forward_{p.dates[-1]}.json"
else:
    done = sorted(S.RESULTS.glob("decision_*.json"))
    look = len(done) + 1
    if look > len(S.LOOKS_MONTHS):
        raise SystemExit("all decisions were made")
    look_day = E.look_dates(S.T0, p.dates)[look - 1]
    if look_day is None or np.datetime64(look_day, "D") > p.dates[-1]:
        raise SystemExit(f"decision {look} is allowed from the close of {look_day}; data end {p.dates[-1]}")
    ld = np.datetime64(look_day, "D")
    end = int(np.searchsorted(p.dates, ld, side="right"))
    m = F.forward_metrics(p, cash, tickers, t0, end)
    prev = json.loads(done[-1].read_text())["states"] if done else {}
    open_ = [k for k in ("F1", "F2") if prev.get(k) not in ("zamitnuta", "potvrzena")]
    hyps = {}
    for k in open_:
        h = m[k]
        if k == "F1":
            hyps[k] = {"n_enough": h["n_days"] >= S.MIN_DAYS_F1, "main_x1": h["sharpe_x1"], "up90": h["up90"], "p": h["p"],
                       "plan_bar": h["sharpe_x1.5"] > 0, "point_ok": h["sharpe_x1"] > 0}
        else:
            hyps[k] = {"n_enough": h["n_trades"] >= S.MIN_TRADES_F2[look - 1], "main_x1": h["mean_trade_x1"], "up90": h["up90"],
                       "p": h["p"], "plan_bar": h["sharpe_x1.5"] > 0 and h["mean_trade_x1.5"] > 0,
                       "point_ok": h["sharpe_x1"] > 0 and h["mean_trade_x1"] > 0}
    states = {**{k: prev[k] for k in prev if k not in open_}, **E.decide(look, hyps)}
    out = m | {"look": look, "look_day": str(look_day), "inputs": hyps, "states": states} | prov
    dest = S.RESULTS / f"decision_{look}.json"
    if dest.exists():
        raise SystemExit(f"{dest} exists: the decision is made once")
dest.write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
print(f"{dest}\n" + json.dumps({k: out[k] for k in ("F1", "F2", "SPY")}, indent=1, default=str))
