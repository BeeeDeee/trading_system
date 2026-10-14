"""Research 16, development stage (prereg §7): all 30 variants on 2018-02 -> 2021-12 (FUND from 2020-04),
selection of P_f per family and the development gate.

    uv run python scripts/r16_dev.py binance_2026-10-03

Every variant is recorded in the registry before anything is computed. Writes docs/research16/results/dev.json.
"""

import json
import sys
from pathlib import Path

import numpy as np

from qlab.research15 import strategy as S
from qlab.research16 import evaluate as E, families as F
from qlab.research15.setup import epoch_15m
from qlab.research16.setup import DEV, load_dev, registry
from qlab.validation.stats import sharpe

MIN_TRADES = 30
snapshot = sys.argv[1]
reg = registry()
for v in F.grid():
    reg.record("candidates", v.as_dict(), note=f"r16 dev {v.id}")

assets = E.build_assets(load_dev(snapshot), load_dev(snapshot, market="perp"))
end = len(next(iter(assets.values())).t)

cands, runs, sr_daily = {}, {}, []
for v in F.grid():
    start = E.start_index(assets, v, DEV[0])
    cs = F.signals(assets, v)
    r = E.run(assets, cs, start, end)
    runs[v.id] = (v, cs, r, start)
    cands[v.id] = E.full_stats(r) | {"start": str(r.dates[0]),
                                      "sharpe_cost_x2": E.stats(E.run(assets, cs, start, end, cost=2 * S.COST).daily)["sharpe"],
                                      "sharpe_delay_1": E.stats(E.run(assets, cs, start, end, delay=1).daily)["sharpe"]}
    sr_daily.append(sharpe(r.daily))
    print(v.id, {k: round(x, 4) for k, x in cands[v.id].items() if isinstance(x, float)}, flush=True)

families = {}
for fam in F.FAMILIES:
    vs = [v for v in F.grid() if v.family == fam and cands[v.id].get("n_trades", 0) >= MIN_TRADES]
    if not vs:
        families[fam] = {"P": None, "gate": False, "reason": f"no variant with >= {MIN_TRADES} trades"}
        continue
    P = max(vs, key=lambda v: cands[v.id]["sharpe"])           # ties -> first in grid order
    _, cs, r, start = runs[P.id]
    nl = E.null(assets, P, r, start, end)
    mean, lo, hi = E.trade_ci(r.all_trades())
    gate = bool(cands[P.id]["sharpe_cost_x2"] > 0 and nl["p"] < 0.05)
    families[fam] = {"P": P.id, "sharpe": cands[P.id]["sharpe"], "sharpe_cost_x2": cands[P.id]["sharpe_cost_x2"],
                     "B_RND": nl, "trade_ci90": [mean, lo, hi], "by_year": E.by_year(r.dates, r.daily),
                     "sides": E.sides(r), "gate": gate}
    print(fam, json.dumps(families[fam], default=float), flush=True)

t0 = next(iter(assets.values())).t[0]
_, hold = E.run_hold(assets, int(epoch_15m(DEV[0]) - t0), end)
out = {"snapshot": snapshot, "period": [str(DEV[0]), str(DEV[1])], "min_trades": MIN_TRADES,
       "candidates": cands, "families": families, "passed_gate": [f for f, x in families.items() if x["gate"]],
       "sr_daily_var": float(np.nanvar(sr_daily, ddof=1)), "B_HOLD": E.stats(hold)}
Path("docs/research16/results").mkdir(parents=True, exist_ok=True)
Path("docs/research16/results/dev.json").write_text(json.dumps(out, indent=1, default=float))
print(json.dumps({k: v for k, v in out.items() if k not in ("candidates", "families")}, indent=1, default=float))
