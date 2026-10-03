"""Research 8, development stage (prereg §4.3, §5-6): 17 candidates on 2020-01-06..2022-12-31, choose P2.

    uv run python scripts/r8_dev.py binance_2026-10-03     -> docs/research8/results/dev.json

Every candidate is recorded in runs/registry/research8.sqlite before it is computed.
"""

import json
import sys
from pathlib import Path

import numpy as np

from qlab.research8 import report as R, strategy as S
from qlab.research8.setup import DEV, index_of, load_dev, registry
from qlab.research8.sim import Costs, simulate
from qlab.validation.stats import sharpe

snapshot = sys.argv[1]
p = load_dev(snapshot)
i0 = index_of(p, DEV[0])
reg = registry()
out = {"stage": "dev", "period": [str(DEV[0]), str(DEV[1])], "candidates": {}}
sr_daily = {}
for cfg in S.grid():
    if not reg.has(cfg.as_dict()):
        reg.record("candidates", cfg.as_dict(), note=f"r8 dev {cfg.id}")
    tg = {t: w for t, w in S.targets(p, cfg).items() if t >= i0}
    res = {}
    for scen, c in (("base", Costs()), ("stress", Costs(mult=2.0))):
        r = simulate(p, tg, s=cfg.s, costs=c, liq_price=cfg.liq_price)
        sl = slice(i0, None)
        res[scen] = R.summary(r.ret[sl], p.tbill[sl], r.funding[sl], r.costs[sl], r.turnover[sl])
        res[scen]["liquidations"] = len([x for x in r.liquidations if x[0] >= i0])
        res[scen]["delistings"] = len([x for x in r.delistings if x[0] >= i0])
        res[scen]["pair_mismatch_exits"] = len([x for x in r.mismatches if x[0] >= i0])
        if scen == "base":
            sr_daily[cfg.id] = sharpe(r.ret[sl] - p.tbill[sl])
    out["candidates"][cfg.id] = {"config": cfg.as_dict(), **res}
    b = res["base"]
    print(f"{cfg.id:<18} excess {b['excess_ann'] * 100:+6.2f} %  SR {b['sharpe_excess']:+5.2f}  "
          f"MDD {b['max_dd'] * 100:6.1f} %  funding {b['funding_ann'] * 100:5.1f} %  costs {b['costs_ann'] * 100:4.1f} %  "
          f"liq {b['liquidations']}  stress {res['stress']['excess_ann'] * 100:+6.2f} %")

best = max(out["candidates"], key=lambda k: (out["candidates"][k]["base"]["sharpe_excess"],
                                             -out["candidates"][k]["base"]["turnover_ann"]))
out["P1"] = "S0"
out["P2"] = best
out["sr_variance_daily"] = float(np.var(list(sr_daily.values()), ddof=1))
out["n_trials"] = reg.total_configs("candidates")
Path("docs/research8/results").mkdir(parents=True, exist_ok=True)
Path("docs/research8/results/dev.json").write_text(json.dumps(out, indent=1))
print(f"\nP1 = S0, P2 = {best}; N = {out['n_trials']}, var(SR daily) = {out['sr_variance_daily']:.3e}")
