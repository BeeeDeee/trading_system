"""Research 23, final holdout evaluation 2023-01 -> 2026-09 (prereg §8-9). Runs ONCE through the vault, only for
the setups that passed the development gate (docs/research23/results/dev.json).

    uv run python scripts/r23_final.py binance_2026-10-03 freeze
    QLAB_R23_METHODOLOGY=<hash> uv run python scripts/r23_final.py binance_2026-10-03 run
"""

import hashlib
import json
import os
import sys
from pathlib import Path

from qlab.research16.engine import holm
from qlab.research23 import engine as G, evaluate as E, setups as SU
from qlab.research23.data import FINAL_ENV, HOLDOUT, SUBPERIODS, load_final, registry, vault
from qlab.validation.registry import current_commit
from qlab.validation.stats import deflated_sharpe

N_EARLIER = 46                              # research 15-17 candidates on the same BTC/ETH data
snapshot, action = sys.argv[1], sys.argv[2]
FILES = ["docs/research23/PREREGISTRATION.md", "docs/research23/results/dev.json", "scripts/r23_final.py",
         *sorted(str(f) for f in Path("src/qlab/research23").glob("*.py"))]
dev = json.loads(Path("docs/research23/results/dev.json").read_text())
if not dev["passed_gate"]:
    raise SystemExit("no setup passed the development gate: the holdout is not opened (prereg §7)")


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
registry().record("methodology_eval", {"study": "research23", "stage": "final", "hash": mh}, note="r23 one-time holdout")
assets = {s: E.prepare(d) for s, d in load_final(snapshot).items()}
t = next(iter(assets.values()))["t"]
start, end = E.start_index(assets, HOLDOUT[0]), len(t)
by_id = {v.id: v for v in SU.grid()}
n_grid = len(by_id)

fams, pvals = {}, {}
for s in dev["passed_gate"]:
    P = by_id[dev["families"][s]["P"]]
    sigs = E.signals(assets, P)
    rp = E.run(assets, P, sigs, start, end)
    sp = E.full_stats(rp)
    mean, lo, hi = E.trade_ci(rp.trades())
    nl = E.null(assets, rp, start, end)
    pvals[s] = nl["p"]
    s_x2 = E.stats(E.run(assets, P, sigs, start, end, cost=2 * G.COST).daily)["sharpe"]
    s_d1 = E.stats(E.run(assets, P, sigs, start, end, delay=1).daily)["sharpe"]
    sub = {k: E.stats(E.window(rp.dates, rp.daily, *w))["sharpe"] for k, w in SUBPERIODS.items()}
    per_asset = sp["per_asset_mean_trade"]
    crit = {
        "1_sharpe_and_trade_ci": {"sharpe": sp["sharpe"], "mean_trade_ci90": [mean, lo, hi],
                                  "pass": bool(sp["sharpe"] > 0 and mean > 0 and lo > 0)},
        "2_null_holm": {"B_RND": nl},
        "3_cost_x2": {"sharpe": s_x2, "pass": bool(s_x2 > 0)},
        "4_delay_1_candle": {"sharpe": s_d1, "pass": bool(s_d1 > 0)},
        "5_subperiods": {"sharpe": sub, "pass": bool(all(x > 0 for x in sub.values()))},
        "6_btc_and_eth_trade": {"mean_trade": per_asset,
                                "pass": bool(all((per_asset.get(a) or -1) > 0 for a in assets))},
    }
    fams[s] = {"P": P.id, "criteria": crit, "P_stats": sp, "by_year": E.by_year(rp.dates, rp.daily),
               "sides": E.sides(rp), "gross_mean_trade": E.gross_mean(rp),
               "dsr_N20": deflated_sharpe(rp.daily, n_grid, dev["sr_daily_var"]),
               "dsr_N66": deflated_sharpe(rp.daily, n_grid + N_EARLIER, dev["sr_daily_var"])}
hl = holm(pvals)
for s, x in fams.items():
    x["criteria"]["2_null_holm"] |= hl[s]
    x["n_pass"] = int(sum(c["pass"] for c in x["criteria"].values()))
    x["verdict"] = "forward_candidate" if x["n_pass"] == 6 else "rejected"
    print(s, x["P"], x["verdict"], x["n_pass"], flush=True)

# robustness (prereg §9), after §8: all 20 variants, gross, sides, hours, volatility terciles
allv = {}
for v in SU.grid():
    sigs = E.signals(assets, v)
    r = E.run(assets, v, sigs, start, end)
    allv[v.id] = E.full_stats(r) | {"gross_mean_trade": E.gross_mean(r), "sides": E.sides(r),
                                    "by_year": E.by_year(r.dates, r.daily), "by_hour_utc": E.by_hour(assets, r),
                                    "by_vol_tercile": E.by_vol_tercile(assets, r)}
    print(f"{v.id:18s} n={allv[v.id].get('n_trades', 0):6d} sharpe={allv[v.id]['sharpe']:6.2f}", flush=True)

out = {"snapshot": snapshot, "commit": current_commit(), "methodology_hash": mh, "k": len(fams), "families": fams,
       "robust": {"all_variants": allv}}
Path("docs/research23/results/final.json").write_text(json.dumps(out, indent=1, default=float))
print(json.dumps({s: {"P": x["P"], "verdict": x["verdict"], "n_pass": x["n_pass"]} for s, x in fams.items()}, indent=1))
