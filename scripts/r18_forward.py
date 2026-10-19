"""Research 18 forward test of H1 and H2 (docs/research18/PREREGISTRATION.md). Data come from the research 17
downloader (`scripts/r17_forward.py download`, run daily by cron), this script only computes.

    uv run python scripts/r18_forward.py freeze    # once: lock the rule files, register N = 2 (commit forward_lock.json)
    uv run python scripts/r18_forward.py status    # descriptive metrics up to the last common day (never a decision)
    uv run python scripts/r18_forward.py final     # the pre-registered decision, only at the dates of §6
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from qlab.research17 import forward as F
from qlab.research18 import evaluate as E, setup as T
from qlab.validation.registry import TrialRegistry, current_commit

RULE_FILES = [*F.RULE_FILES, "src/qlab/research17/variants.py", "src/qlab/research18/setup.py",
              "src/qlab/research18/rules.py", "src/qlab/research18/evaluate.py"]
action = sys.argv[1] if len(sys.argv) > 1 else ""


def check_lock() -> None:
    if not T.LOCK.exists():
        raise SystemExit("rule files are not frozen: run `freeze` first")
    lock = json.loads(T.LOCK.read_text())
    if lock["rule_hash"] != F.rule_hash(lock["files"]):
        raise SystemExit("rule files changed since the freeze (forward_lock.json); see PREREGISTRATION.md §12")


if action == "freeze":
    if T.LOCK.exists():
        raise SystemExit(f"{T.LOCK} already exists")
    T.LOCK.parent.mkdir(parents=True, exist_ok=True)
    T.LOCK.write_text(json.dumps({"rule_hash": F.rule_hash(RULE_FILES), "files": RULE_FILES, "commit": current_commit(),
                                  "frozen_at": datetime.now(timezone.utc).isoformat()}, indent=1))
    TrialRegistry(T.REGISTRY).record("candidates", {"study": "research18", "hypotheses": ["H1_lag0", "H2_btc200"]},
                                     n_configs=2, note="r18 forward: H1 V05_lag0, H2 V01 + BTC > MA200")
    print(f"frozen: {json.loads(T.LOCK.read_text())['rule_hash']}")
    sys.exit(0)
if action not in ("status", "final"):
    raise SystemExit(__doc__)
check_lock()

members = json.loads((F.RAW / "members.json").read_text())["members"]
panels = F.load_panels(sorted({s for v in members.values() for s in v}))
asof = min(F.last_day(panels[k]) for k in ("BTCUSDT@spot", "BTCUSDT@perp", "ETHUSDT@perp"))
if asof < T.T0:
    raise SystemExit(f"data end {asof} < start {T.T0}: nothing to evaluate yet")
provenance = {"commit": current_commit(), "rule_hash": F.rule_hash(json.loads(T.LOCK.read_text())["files"]),
              "data_hash": F.data_hash(), "created_at": datetime.now(timezone.utc).isoformat()}
T.RESULTS.mkdir(parents=True, exist_ok=True)

if action == "status":
    out = E.evaluate(panels, members, asof) | provenance | {"note": "descriptive; the decision is made only by `final`"}
    dest = T.RESULTS / f"status_{asof}.json"
else:
    first = T.RESULTS / "decision_1.json"
    if not first.exists():
        look, alive = 1, list(E.H.HYPOTHESES)
    else:
        prev = json.loads(first.read_text())["decisions"]
        look, alive = 2, [h for h, d in prev.items() if d == "inconclusive"]
        if not alive:
            raise SystemExit("every hypothesis was decided at the first look: nothing to extend")
    dest = T.RESULTS / f"decision_{look}.json"
    if dest.exists():
        raise SystemExit(f"{dest} already exists: the decision is made once")
    end = T.END[look]
    if asof < end:
        raise SystemExit(f"data end {asof} < {end}: decision {look} is not due")
    out = E.evaluate(panels, members, end) | provenance
    out["look"], out["alive"] = look, alive
    out["decisions"] = E.decide(out["metrics"], alive, look)
    out.pop("provisional_decision", None)
dest.write_text(json.dumps(out, indent=1, default=float))
print(json.dumps({k: v for k, v in out.items() if k not in ("daily", "trades", "per_symbol", "followers", "by_year")},
                 indent=1, default=float)[:6000])
