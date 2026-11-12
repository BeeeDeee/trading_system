"""Research 23, development stage 2020-01 -> 2022-12 (prereg §7). All 20 variants are registered first.

    uv run python scripts/r23_dev.py binance_2026-10-03
"""

import json
import sys
from pathlib import Path

import numpy as np

from qlab.research23 import engine as G, evaluate as E, setups as SU
from qlab.research23.data import DEV, load_dev, registry
from qlab.validation.stats import sharpe

snapshot = sys.argv[1]
reg = registry()
grid = SU.grid()
for v in grid:
    reg.record("candidates", v.as_dict(), note=f"r23 dev {v.id}")
assets = {s: E.prepare(d) for s, d in load_dev(snapshot).items()}
t = next(iter(assets.values()))["t"]
start, end = E.start_index(assets, DEV[0]), len(t)

variants, sr_daily, runs = {}, [], {}
for v in grid:
    sigs = E.signals(assets, v)
    r = E.run(assets, v, sigs, start, end)
    runs[v.id] = (v, sigs, r)
    st = E.full_stats(r)
    st |= {"gross_mean_trade": E.gross_mean(r), "sides": E.sides(r),
           "sharpe_cost_x2": E.stats(E.run(assets, v, sigs, start, end, cost=2 * G.COST).daily)["sharpe"],
           "sharpe_gross": E.stats(E.run(assets, v, sigs, start, end, cost=0.0).daily)["sharpe"]}
    variants[v.id] = st
    sr_daily.append(sharpe(r.daily))
    print(f"{v.id:18s} n={st.get('n_trades', 0):6d} sharpe={st['sharpe']:6.2f} mean={st.get('mean_trade', 0) * 1e4:7.2f}bp "
          f"gross={(st['gross_mean_trade'] or 0) * 1e4:6.2f}bp", flush=True)

families, passed = {}, []
for s in SU.SETUPS:
    ids = [v.id for v in grid if v.setup == s and v.k == 1.0]
    best = max(ids, key=lambda i: variants[i]["sharpe"])
    v, sigs, r = runs[best]
    nl = E.null(assets, r, start, end)
    st = variants[best]
    gate = {"sharpe_gt_0": st["sharpe"] > 0, "mean_trade_gt_0": st.get("mean_trade", -1) > 0,
            "n_ge_200": st.get("n_trades", 0) >= 200, "null_p_le_0.10": nl["p"] <= 0.10}
    ok = all(gate.values())
    families[s] = {"P": best, "B_RND": nl, "gate": gate, "pass": ok, "ranking": sorted(ids, key=lambda i: -variants[i]["sharpe"])}
    if ok:
        passed.append(s)
    print(s, best, "PASS" if ok else "fail", gate, nl, flush=True)

out = {"snapshot": snapshot, "stage": [str(DEV[0]), str(DEV[1])], "variants": variants, "families": families,
       "passed_gate": passed, "sr_daily_var": float(np.var(sr_daily, ddof=1)), "n_registered": len(grid)}
Path("docs/research23/results").mkdir(parents=True, exist_ok=True)
Path("docs/research23/results/dev.json").write_text(json.dumps(out, indent=1, default=float))
print("passed gate:", passed)
