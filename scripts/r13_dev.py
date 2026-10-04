"""Research 13, development stage (prereg §5-7): 24 candidates on 2018-04-01..2021-12-31, choose P.

    uv run python scripts/r13_dev.py binance_2026-10-03     -> docs/research13/results/dev.json

All 24 candidates are recorded in the registry before the first one is computed. P = highest Sharpe,
a tie goes to the first candidate in grid order (decision log 2026-10-03). B_RND is reported for P only,
descriptively (the criterion is on the holdout).
"""

import json
import sys
from pathlib import Path

import numpy as np

from qlab.engine.vector import simulate as simulate_vector
from qlab.research9 import panel as P9, strategy as S9
from qlab.research13 import strategy as S
from qlab.research13.setup import DEV, index_of, load_dev, registry, symbols
from qlab.validation.metrics import max_drawdown
from qlab.validation.stats import sharpe

ANN = np.sqrt(365)
N_RND = 500
snapshot = sys.argv[1]
p, extra = load_dev(snapshot)
syms = symbols(snapshot)
btc = syms.index("BTCUSDT")
qv = np.asarray(extra["qv"])
i0 = index_of(p, DEV[0])
start, end = i0 - 1, p.shape[0]          # first decision after the close before the dev start, empty book
sl = slice(i0, None)                     # vector engine: full-length series
cost, cost2 = P9.cost_rate(qv), P9.cost_rate(qv, 2.0)
univ = S.universe_mask(p, qv, S9.UNIVERSE_N, S.excluded(syms))


def stats(r: np.ndarray) -> dict:
    nav = np.cumprod(1 + r)
    return {"cagr": float(nav[-1] ** (365 / len(r)) - 1), "sharpe": float(sharpe(r) * ANN),
            "vol": float(r.std(ddof=1) * ANN), "max_dd": float(max_drawdown(nav)[0])}


def trade_stats(tr) -> dict:
    n = tr.height
    if not n:
        return {"n_trades": 0}
    reasons = tr["reason"].value_counts()
    return {"n_trades": n, "win_rate": float((tr["net_ret"] > 0).mean()), "mean_net": float(tr["net_ret"].mean()),
            "median_held": float(tr["held"].median()),
            "exits": {r: int(c) for r, c in zip(reasons["reason"], reasons["count"])}}


ew = simulate_vector(p, S9.ew_universe_decisions(p, qv, S9.UNIVERSE_N, start), cost).returns[sl]
btc_hold = simulate_vector(p, S9.hold(p, btc, start), cost).returns[sl]
out = {"stage": "dev", "period": [str(DEV[0]), str(DEV[1])],
       "benchmarks": {"B_EW": stats(ew), "B_BTC": stats(btc_hold)}, "candidates": {}}

reg = registry()
grid = S.grid()
for cfg in grid:
    if not reg.has(cfg.as_dict()):
        reg.record("candidates", cfg.as_dict(), note=f"r13 dev {cfg.id}")

act_sr, runs = {}, {}
for cfg in grid:
    o = S.run(p, extra, cfg, cost, start, end, univ)
    r = o.returns[1:]                    # day `start` is the decision close before the dev start
    r2 = S.run(p, extra, cfg, cost2, start, end, univ).returns[1:]
    s = stats(r)
    s.update({"sharpe_diff": s["sharpe"] - out["benchmarks"]["B_EW"]["sharpe"], "stress_sharpe": stats(r2)["sharpe"],
              "turnover_ann": float(o.turnover[1:].mean() * 365), "costs_ann": float(o.costs[1:].mean() * 365),
              "exposure": float(o.exposure[1:].mean()), **trade_stats(o.trades)})
    act_sr[cfg.id] = sharpe(r - ew)
    runs[cfg.id] = o
    out["candidates"][cfg.id] = {"config": cfg.as_dict(), **s}
    print(f"{cfg.id:<22} CAGR {s['cagr'] * 100:+7.1f} %  SR {s['sharpe']:5.2f}  ΔSR {s['sharpe_diff']:+5.2f}  "
          f"MDD {s['max_dd'] * 100:5.1f} %  expo {s['exposure']:.2f}  costs {s['costs_ann'] * 100:4.1f} %  "
          f"trades {s['n_trades']:4d}  win {s.get('win_rate', 0) * 100:3.0f} %  stress SR {s['stress_sharpe']:5.2f}")
for k, v in out["benchmarks"].items():
    print(f"{k:<22} CAGR {v['cagr'] * 100:+7.1f} %  SR {v['sharpe']:5.2f}  MDD {v['max_dd'] * 100:5.1f} %")

P = max(out["candidates"], key=lambda k: out["candidates"][k]["sharpe"])     # first maximum = grid order
rnd = np.array([sharpe(S.random_benchmark(p, univ, runs[P], cost, start, end, seed).returns[1:]) * ANN
                for seed in range(N_RND)])
out["P"] = P
out["B_RND_for_P"] = {"n": N_RND, "median": float(np.median(rnd)), "p95": float(np.percentile(rnd, 95)),
                      "share_below_P": float((rnd < out["candidates"][P]["sharpe"]).mean())}
out["sr_variance_active_daily"] = float(np.var(list(act_sr.values()), ddof=1))
out["n_trials"] = reg.total_configs("candidates")
Path("docs/research13/results").mkdir(parents=True, exist_ok=True)
Path("docs/research13/results/dev.json").write_text(json.dumps(out, indent=1))
b = out["B_RND_for_P"]
print(f"\nP = {P} (SR {out['candidates'][P]['sharpe']:.2f}); B_RND median {b['median']:.2f}, p95 {b['p95']:.2f}, "
      f"P above {b['share_below_P'] * 100:.0f} % of random runs; N = {out['n_trials']}")
