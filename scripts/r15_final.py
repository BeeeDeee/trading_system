"""Research 15, final holdout evaluation (prereg §8-9). Runs ONCE through the vault.

    uv run python scripts/r15_final.py binance_2026-10-03 freeze
    QLAB_R15_METHODOLOGY=<hash> uv run python scripts/r15_final.py binance_2026-10-03 run

Every run starts with an empty book at 2022-01-01 00:00 UTC (prereg §7); indicators use earlier data.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

from qlab.research15 import evaluate as E, strategy as S
from qlab.research15.setup import (FINAL_ENV, HOLDOUT, SECONDARY_START, SUBPERIODS, epoch_15m,
                                   load_final, registry, vault)
from qlab.validation.registry import current_commit
from qlab.validation.stats import deflated_sharpe

N_RND = 500
snapshot, action = sys.argv[1], sys.argv[2]
FILES = ["docs/research15/PREREGISTRATION.md", "docs/research15/results/dev.json", "scripts/r15_final.py",
         *sorted(str(f) for f in Path("src/qlab/research15").glob("*.py"))]


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
registry().record("methodology_eval", {"study": "research15", "stage": "final", "hash": mh},
                  note="r15 one-time holdout")
dev = json.loads(Path("docs/research15/results/dev.json").read_text())
spot, perp = load_final(snapshot), load_final(snapshot, market="perp")
assets = E.build_assets(spot, perp)
t = next(iter(assets.values())).t
start, end = int(epoch_15m(HOLDOUT[0]) - t[0]), len(t)
P = {c.id: c for c in S.grid()}[dev["P"]]

rp = E.run(assets, P, start, end)
sp = E.stats(rp.daily, rp.all_trades())
mean, lo, hi = E.trade_ci(rp.all_trades())
rnd = np.array([E.stats(E.run_random(assets, P, rp, start, end, seed).daily)["sharpe"] for seed in range(N_RND)])
no15 = E.run(assets, P, start, end, confirm=False)
no4h = E.run(assets, P, start, end, use_regime=False)
s_no15, s_no4h = E.stats(no15.daily, no15.all_trades()), E.stats(no4h.daily, no4h.all_trades())
s_x2 = E.stats(E.run(assets, P, start, end, cost=2 * S.COST).daily)
sub = {k: E.stats(E.window(rp.dates, rp.daily, *v))["sharpe"] for k, v in SUBPERIODS.items()}
per_asset = {s: float(b.ret.mean()) if len(b.ret) else float("nan") for s, b in rp.books.items()}
dsr = deflated_sharpe(rp.daily, len(S.grid()), dev["sr_daily_var"])

criteria = {
    "1_sharpe_and_trade_ci": {"sharpe": sp["sharpe"], "mean_trade_ci90": [mean, lo, hi],
                              "pass": bool(sp["sharpe"] > 0 and mean > 0 and lo > 0)},
    "2_above_rnd_p95": {"sharpe": sp["sharpe"], "rnd_p95": float(np.quantile(rnd, 0.95)),
                        "pass": bool(sp["sharpe"] > np.quantile(rnd, 0.95))},
    "3_above_no15_and_no4h_trade": {"P": sp.get("mean_trade"), "B_NO15": s_no15.get("mean_trade"),
                                    "B_NO4H": s_no4h.get("mean_trade"),
                                    "pass": bool(sp.get("mean_trade", -1) > s_no15.get("mean_trade", 1)
                                                 and sp.get("mean_trade", -1) > s_no4h.get("mean_trade", 1))},
    "4_cost_x2": {"sharpe": s_x2["sharpe"], "pass": bool(s_x2["sharpe"] > 0)},
    "5_subperiods": {"sharpe": sub, "pass": bool(all(v > 0 for v in sub.values()))},
    "6_btc_and_eth_trade": {"mean_trade": per_asset, "pass": bool(all(v > 0 for v in per_asset.values()))},
}
_, h4 = E.run_4h(assets, P.side, start, end)
_, h4_ls = E.run_4h(assets, "LS", start, end)
_, hold = E.run_hold(assets, start, end)
sides = {}
for name, sd in (("long", 1), ("short", -1)):
    tr = np.concatenate([b.ret[b.sides == sd] for b in rp.books.values()])
    sides[name] = {"n": int(len(tr)), "mean": float(tr.mean()) if len(tr) else None}

# perp prices instead of spot (prereg §9): holdout and the whole perp history 2020-01 -> 2026-09
pa = {s: S.Asset.build(perp[s]) for s in perp}
pt = next(iter(pa.values())).t
rpp = E.run(pa, P, int(epoch_15m(HOLDOUT[0]) - pt[0]), len(pt))
rpf = E.run(pa, P, 0, len(pt))

# secondary basket (prereg §9): monthly point-in-time top-10 perps, descriptive only, own prices and funding
manifest = json.loads((Path("data/raw") / snapshot / "r15_manifest.json").read_text())
sec_dates, sec_daily, sec_tr = E.run_basket(
    ((sym, S.Asset.build(load_final(snapshot, (sym,), "perp")[sym])) for sym in manifest["secondary"]),
    manifest["members"], P, SECONDARY_START)

out = {
    "snapshot": snapshot, "commit": current_commit(), "methodology_hash": mh, "P": P.id,
    "criteria": criteria, "n_pass": int(sum(c["pass"] for c in criteria.values())),
    "P_stats": sp | E.trade_breakdown(rp), "dsr": dsr,
    "B_RND": {"median": float(np.median(rnd)), "p95": float(np.quantile(rnd, 0.95)),
              "share_below_P": float((rnd < sp["sharpe"]).mean())},
    "B_NO15": s_no15, "B_NO4H": s_no4h, "B_4H_same_side": E.stats(h4), "B_4H_LS": E.stats(h4_ls),
    "B_HOLD": E.stats(hold),
    "robust": {
        "all_candidates": {c.id: E.stats((r := E.run(assets, c, start, end)).daily, r.all_trades())
                           | E.trade_breakdown(r) for c in S.grid()},
        "by_year": E.by_year(rp.dates, rp.daily),
        "delay_1_candle": E.stats((r := E.run(assets, P, start, end, delay=1)).daily, r.all_trades()),
        "no_funding": E.stats((r := E.run(assets, P, start, end, use_funding=False)).daily, r.all_trades()),
        "sides": sides,
        "perp_prices_holdout": E.stats(rpp.daily, rpp.all_trades()),
        "perp_prices_2020_2026": E.stats(rpf.daily, rpf.all_trades()),
        "secondary_basket_2020-05_2026-09": E.stats(sec_daily, sec_tr),
        "secondary_by_year": E.by_year(sec_dates, sec_daily),
    },
}
Path("docs/research15/results/final.json").write_text(json.dumps(out, indent=1, default=float))
print(json.dumps({k: v for k, v in out.items() if k != "robust"}, indent=1, default=float))
