"""Research 17 forward paper test (docs/research17/FORWARD.md).

    uv run python scripts/r17_forward.py freeze      # once: lock the rule files (commit forward_lock.json)
    uv run python scripts/r17_forward.py download    # new data up to yesterday (UTC), idempotent
    uv run python scripts/r17_forward.py status      # descriptive results up to the last common day
    uv run python scripts/r17_forward.py final       # the decision (once, after 2027-03-31, >= 30 events)

Run `download` + `status` weekly and commit the new `results/forward_<asof>.json`.
"""

import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from qlab.research17 import forward as F
from qlab.validation.registry import current_commit

action = sys.argv[1] if len(sys.argv) > 1 else ""
RESULTS = Path("docs/research17/results")


def check_lock() -> None:
    if not F.LOCK.exists():
        raise SystemExit("rule files are not frozen: run `freeze` first")
    lock = json.loads(F.LOCK.read_text())
    if lock["rule_hash"] != F.rule_hash(lock["files"]):
        raise SystemExit("rule files changed since the freeze (forward_lock.json); see FORWARD.md §9")


if action == "freeze":
    if F.LOCK.exists():
        raise SystemExit(f"{F.LOCK} already exists")
    F.LOCK.write_text(json.dumps({"rule_hash": F.rule_hash(), "files": F.RULE_FILES, "commit": current_commit(),
                                  "frozen_at": datetime.now(timezone.utc).isoformat()}, indent=1))
    print(f"frozen: {json.loads(F.LOCK.read_text())['rule_hash']}")
    sys.exit(0)
if action not in ("download", "status", "final"):
    raise SystemExit(__doc__)
check_lock()
today = datetime.now(timezone.utc).date()

if action == "download":
    mem = F.refresh_members(today)
    syms = sorted({s for v in mem["members"].values() for s in v})
    print(f"eligible perps {mem['eligible']}, followers {len(syms)}, membership errors {len(mem['errors'])}")
    errors, last = list(mem["errors"]), today - timedelta(days=1)
    for sym in ["BTCUSDT", "ETHUSDT", *syms]:
        try:
            for market in ("perp", "spot") if sym == "BTCUSDT" else ("perp",):
                r = F.fetch_days(market, sym, F.FORWARD_START, last)
                print(r, flush=True)
            if not (F.BASE_RAW / "perp_15m" / sym).exists():            # a member new since the snapshot
                F.fetch_months("perp_15m", sym, F.RAW, None, F.LAST_BASE_MONTH)
                F.fetch_months("funding", sym, F.RAW, None, F.LAST_BASE_MONTH)
            print(F.fetch_funding(sym), flush=True)
        except Exception as e:
            errors.append({"symbol": sym, "error": f"{type(e).__name__}: {e}"})
    print(f"errors: {len(errors)}")
    for e in errors:
        print(e)
    sys.exit(1 if errors else 0)

members = json.loads((F.RAW / "members.json").read_text())["members"]
syms = sorted({s for v in members.values() for s in v})
panels = F.load_panels(syms)
core = [panels[k] for k in ("BTCUSDT@spot", "BTCUSDT@perp", "ETHUSDT@perp")]
asof = min(F.last_day(d) for d in core)
if action == "final" and asof < F.FORWARD_END:
    raise SystemExit(f"data end {asof} < {F.FORWARD_END}: the forward test is not over")
asof = min(asof, F.FORWARD_END) if action == "final" else asof
out = F.evaluate(panels, members, asof) | {"commit": current_commit(), "rule_hash": F.rule_hash(),
                                           "data_hash": F.data_hash(), "created_at": datetime.now(timezone.utc).isoformat()}
RESULTS.mkdir(parents=True, exist_ok=True)
if action == "final":
    dest = RESULTS / "forward_final.json"
    if dest.exists():
        raise SystemExit(f"{dest} already exists: the decision is made once")
    if out["n_events"] < F.MIN_EVENTS:
        raise SystemExit(f"only {out['n_events']} events < {F.MIN_EVENTS}: extend the test by 3 months (FORWARD.md §4)")
    out["decision"] = "success" if out["primary_criterion"]["meets"] else "rejected"
else:
    dest = RESULTS / f"forward_{asof}.json"
dest.write_text(json.dumps(out, indent=1, default=float))
print(json.dumps({k: v for k, v in out.items() if k not in ("members_used", "per_symbol", "scenarios")},
                 indent=1, default=float))
