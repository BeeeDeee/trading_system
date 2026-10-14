"""Research 16, final holdout evaluation (prereg §8-9). Runs ONCE through the vault, only for the families that
passed the development gate (docs/research16/results/dev.json).

    uv run python scripts/r16_final.py binance_2026-10-03 freeze
    QLAB_R16_METHODOLOGY=<hash> uv run python scripts/r16_final.py binance_2026-10-03 run

Every run starts with an empty book at 2022-01-01 00:00 UTC (prereg §7); indicators use earlier data.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

from qlab.research15 import strategy as S
from qlab.research15.setup import SECONDARY_START, BASKET_WEIGHT, epoch_15m
from qlab.research16 import engine as G, evaluate as E, families as F
from qlab.research16.setup import FINAL_ENV, HOLDOUT, SUBPERIODS, load_final, registry, vault
from qlab.validation.registry import current_commit
from qlab.validation.stats import deflated_sharpe

N_R15 = 16                                  # candidates of research 15 on the same data (cumulative DSR)
snapshot, action = sys.argv[1], sys.argv[2]
FILES = ["docs/research16/PREREGISTRATION.md", "docs/research16/results/dev.json", "scripts/r16_final.py",
         *sorted(str(f) for f in Path("src/qlab/research16").glob("*.py"))]
dev = json.loads(Path("docs/research16/results/dev.json").read_text())
if not dev["passed_gate"]:
    raise SystemExit("no family passed the development gate: the holdout is not opened (prereg §7)")


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
registry().record("methodology_eval", {"study": "research16", "stage": "final", "hash": mh},
                  note="r16 one-time holdout")
spot, perp = load_final(snapshot), load_final(snapshot, market="perp")
assets = E.build_assets(spot, perp)
t = next(iter(assets.values())).t
start, end = int(epoch_15m(HOLDOUT[0]) - t[0]), len(t)
by_id = {v.id: v for v in F.grid()}

fams, pvals = {}, {}
for fam in dev["passed_gate"]:
    P = by_id[dev["families"][fam]["P"]]
    cs = F.signals(assets, P)
    rp = E.run(assets, cs, start, end)
    sp = E.full_stats(rp)
    mean, lo, hi = E.trade_ci(rp.all_trades())
    nl = E.null(assets, P, rp, start, end)
    pvals[fam] = nl["p"]
    s_x2 = E.stats(E.run(assets, cs, start, end, cost=2 * S.COST).daily)["sharpe"]
    s_d1 = E.stats(E.run(assets, cs, start, end, delay=1).daily)["sharpe"]
    sub = {k: E.stats(E.window(rp.dates, rp.daily, *w))["sharpe"] for k, w in SUBPERIODS.items()}
    per_asset = {s: float(b.ret.mean()) for s, b in rp.books.items() if len(b.ret)}
    need_assets = ("ETHUSDT",) if fam == "LEAD" else tuple(assets)
    crit = {
        "1_sharpe_and_trade_ci": {"sharpe": sp["sharpe"], "mean_trade_ci90": [mean, lo, hi],
                                  "pass": bool(sp["sharpe"] > 0 and mean > 0 and lo > 0)},
        "2_null_holm": {"B_RND": nl},                                       # pass filled in after Holm
        "3_cost_x2": {"sharpe": s_x2, "pass": bool(s_x2 > 0)},
        "4_delay_1_candle": {"sharpe": s_d1, "pass": bool(s_d1 > 0)},
        "5_subperiods": {"sharpe": sub, "pass": bool(all(x > 0 for x in sub.values()))},
        "6_btc_and_eth_trade": {"mean_trade": per_asset, "applies_to": list(need_assets),
                                "pass": bool(all(per_asset.get(s, -1) > 0 for s in need_assets))},
    }
    fams[fam] = {"P": P.id, "criteria": crit, "P_stats": sp, "by_year": E.by_year(rp.dates, rp.daily),
                 "sides": E.sides(rp),
                 "no_funding": E.stats(E.run(assets, cs, start, end, use_funding=False).daily)["sharpe"],
                 "dsr_N30": deflated_sharpe(rp.daily, len(F.grid()), dev["sr_daily_var"]),
                 "dsr_N46": deflated_sharpe(rp.daily, len(F.grid()) + N_R15, dev["sr_daily_var"])}
holm = G.holm(pvals)
for fam, x in fams.items():
    x["criteria"]["2_null_holm"] |= holm[fam]
    x["n_pass"] = int(sum(c["pass"] for c in x["criteria"].values()))
    x["verdict"] = "pass" if x["n_pass"] == 6 else "rejected"
    print(fam, x["P"], x["verdict"], x["n_pass"], flush=True)

# robustness (prereg §9), computed after §8: all 30 variants, perp prices, secondary basket of passing families
allv = {}
for v in F.grid():
    r = E.run(assets, F.signals(assets, v), start, end)
    allv[v.id] = E.full_stats(r)
pa = {s: G.Asset.build(perp[s]) for s in perp}
pt = next(iter(pa.values())).t
perp_runs = {}
for fam, x in fams.items():
    P = by_id[x["P"]]
    cs = F.signals(pa, P)
    perp_runs[fam] = {"holdout": E.full_stats(E.run(pa, cs, int(epoch_15m(HOLDOUT[0]) - pt[0]), len(pt))),
                      "2020_2026": E.full_stats(E.run(pa, cs, E.start_index(pa, P, HOLDOUT[0].replace(year=2020)),
                                                      len(pt)))}
basket = {}
passing = [f for f, x in fams.items() if x["verdict"] == "pass" and f not in ("LEAD", "CME")]
if passing:
    manifest = json.loads((Path("data/raw") / snapshot / "r15_manifest.json").read_text())
    for fam in passing:
        P = by_id[fams[fam]["P"]]
        d, dd, tr = E.run_basket(((sym, G.Asset.build(load_final(snapshot, (sym,), "perp")[sym]))
                                  for sym in manifest["secondary"]), manifest["members"], P, SECONDARY_START,
                                 BASKET_WEIGHT)
        basket[fam] = {"stats": E.stats(dd, tr), "by_year": E.by_year(d, dd)}
_, hold = E.run_hold(assets, start, end)
_, h4 = E.run_4h(assets, "LS", start, end)

out = {"snapshot": snapshot, "commit": current_commit(), "methodology_hash": mh, "k": len(fams), "families": fams,
       "B_HOLD": E.stats(hold), "B_4H_LS": E.stats(h4),
       "robust": {"all_variants": allv, "perp_prices": perp_runs, "secondary_basket_2020-05_2026-09": basket}}
Path("docs/research16/results/final.json").write_text(json.dumps(out, indent=1, default=float))
print(json.dumps({f: {"P": x["P"], "verdict": x["verdict"], "n_pass": x["n_pass"]} for f, x in fams.items()}, indent=1))
