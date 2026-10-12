"""Research 15, development stage (prereg §7): all 16 candidates on 2018-02 -> 2021-12, selection of P.

    uv run python scripts/r15_dev.py binance_2026-10-03

Every candidate is recorded in the registry before anything is computed. Writes docs/research15/results/dev.json.
"""

import json
import sys
from pathlib import Path

import numpy as np

from qlab.research15 import evaluate as E, strategy as S
from qlab.research15.setup import DEV, epoch_15m, load_dev, registry
from qlab.validation.stats import sharpe

N_RND = 500
snapshot = sys.argv[1]
reg = registry()
for cfg in S.grid():
    reg.record("candidates", cfg.as_dict(), note=f"r15 dev {cfg.id}")

spot, perp = load_dev(snapshot), load_dev(snapshot, market="perp")
assets = E.build_assets(spot, perp)
t0 = next(iter(assets.values())).t[0]
start, end = int(epoch_15m(DEV[0]) - t0), len(next(iter(assets.values())).t)

cands, sr_daily = {}, []
for cfg in S.grid():
    r = E.run(assets, cfg, start, end)
    cands[cfg.id] = E.stats(r.daily, r.all_trades()) | E.trade_breakdown(r) | {
        "per_asset_mean_trade": {s: float(b.ret.mean()) if len(b.ret) else None for s, b in r.books.items()}}
    sr_daily.append(sharpe(r.daily))
    print(cfg.id, {k: round(v, 4) for k, v in cands[cfg.id].items() if isinstance(v, float)})
P = max(S.grid(), key=lambda c: cands[c.id]["sharpe"])          # ties -> first in grid order (max is stable)
rp = E.run(assets, P, start, end)
rnd = np.array([E.stats(E.run_random(assets, P, rp, start, end, seed).daily)["sharpe"] for seed in range(N_RND)])
no15 = E.run(assets, P, start, end, confirm=False)
no4h = E.run(assets, P, start, end, use_regime=False)
_, h4 = E.run_4h(assets, P.side, start, end)
_, hold = E.run_hold(assets, start, end)
mean, lo, hi = E.trade_ci(rp.all_trades())
out = {
    "snapshot": snapshot, "period": [str(DEV[0]), str(DEV[1])], "candidates": cands, "P": P.id,
    "sr_daily_var": float(np.var(sr_daily, ddof=1)),
    "P_trade_ci90": [mean, lo, hi],
    "B_RND": {"median": float(np.median(rnd)), "p95": float(np.quantile(rnd, 0.95)),
              "share_below_P": float((rnd < cands[P.id]["sharpe"]).mean())},
    "B_NO15": E.stats(no15.daily, no15.all_trades()),
    "B_NO4H": E.stats(no4h.daily, no4h.all_trades()),
    "B_4H": E.stats(h4),
    "B_HOLD": E.stats(hold),
    "P_by_year": E.by_year(rp.dates, rp.daily),
    "P_cost_x2": E.stats(E.run(assets, P, start, end, cost=2 * S.COST).daily),
}
Path("docs/research15/results").mkdir(parents=True, exist_ok=True)
Path("docs/research15/results/dev.json").write_text(json.dumps(out, indent=1, default=float))
print(json.dumps({k: v for k, v in out.items() if k != "candidates"}, indent=1, default=float))
