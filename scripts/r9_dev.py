"""Research 9, development stage (prereg §4-6): 26 candidates on 2018-04-01..2021-12-31, choose P_T and P_X.

    uv run python scripts/r9_dev.py binance_2026-10-03     -> docs/research9/results/dev.json
"""

import json
import sys
from pathlib import Path

import numpy as np

from qlab.engine.vector import simulate
from qlab.research9 import panel as P, strategy as S
from qlab.research9.setup import DEV, index_of, load_dev, registry, symbols
from qlab.validation.metrics import max_drawdown
from qlab.validation.stats import sharpe

ANN = np.sqrt(365)
snapshot = sys.argv[1]
p, extra = load_dev(snapshot)
syms = symbols(snapshot)
btc, eth = syms.index("BTCUSDT"), syms.index("ETHUSDT")
qv = np.asarray(extra["qv"])
i0 = index_of(p, DEV[0])
first = i0 - 1                                       # first decision after the close before the dev start
cost, cost2 = P.cost_rate(qv), P.cost_rate(qv, 2.0)
sl = slice(i0, None)


def stats(r: np.ndarray) -> dict:
    nav = np.cumprod(1 + r)
    return {"cagr": float(nav[-1] ** (365 / len(r)) - 1), "sharpe": float(sharpe(r) * ANN),
            "vol": float(r.std(ddof=1) * ANN), "max_dd": float(max_drawdown(nav)[0])}


bench = {"T": simulate(p, S.monthly_5050(p, btc, eth, first), cost).returns,
         "X": simulate(p, S.ew_universe_decisions(p, qv, S.UNIVERSE_N, first), cost).returns}
btc_hold = simulate(p, S.hold(p, btc, first), cost).returns
out = {"stage": "dev", "period": [str(DEV[0]), str(DEV[1])],
       "benchmarks": {"T_5050": stats(bench["T"][sl]), "X_ew20": stats(bench["X"][sl]), "BTC_hold": stats(btc_hold[sl])},
       "candidates": {}}
reg = registry()
act_sr = {}
for cfg in S.grid():
    if not reg.has(cfg.as_dict()):
        reg.record("candidates", cfg.as_dict(), note=f"r9 dev {cfg.id}")
    r = S.run(p, qv, cfg, cost, first, btc, eth)
    r2 = S.run(p, qv, cfg, cost2, first, btc, eth)
    b = bench[cfg.family][sl]
    s = stats(r.returns[sl])
    s.update({"sharpe_diff": s["sharpe"] - out["benchmarks"]["T_5050" if cfg.family == "T" else "X_ew20"]["sharpe"],
              "stress_sharpe": stats(r2.returns[sl])["sharpe"], "turnover_ann": float(r.turnover[sl].mean() * 365),
              "costs_ann": float(r.costs[sl].mean() * 365), "exposure": float(r.exposure[sl].mean())})
    act_sr[cfg.id] = sharpe(r.returns[sl] - b)
    out["candidates"][cfg.id] = {"config": cfg.as_dict(), **s}
    print(f"{cfg.id:<16} CAGR {s['cagr'] * 100:+7.1f} %  SR {s['sharpe']:5.2f}  ΔSR {s['sharpe_diff']:+5.2f}  "
          f"MDD {s['max_dd'] * 100:6.1f} %  vol {s['vol'] * 100:5.1f} %  costs {s['costs_ann'] * 100:4.1f} %  "
          f"expo {s['exposure']:.2f}  stress SR {s['stress_sharpe']:5.2f}")
for k, v in out["benchmarks"].items():
    print(f"{k:<16} CAGR {v['cagr'] * 100:+7.1f} %  SR {v['sharpe']:5.2f}  MDD {v['max_dd'] * 100:6.1f} %")


def best(fam: str) -> str:
    c = {k: v for k, v in out["candidates"].items() if v["config"]["family"] == fam}
    return max(c, key=lambda k: (c[k]["sharpe"], -c[k]["turnover_ann"]))


out["P_T"], out["P_X"] = best("T"), best("X")
out["sr_variance_active_daily"] = float(np.var(list(act_sr.values()), ddof=1))
out["n_trials"] = reg.total_configs("candidates")
Path("docs/research9/results").mkdir(parents=True, exist_ok=True)
Path("docs/research9/results/dev.json").write_text(json.dumps(out, indent=1))
print(f"\nP_T = {out['P_T']}, P_X = {out['P_X']}, N = {out['n_trials']}")
