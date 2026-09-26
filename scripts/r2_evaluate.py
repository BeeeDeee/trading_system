"""Research 2 / R2: H1 (fixed), H2 (walk-forward selection), benchmarks, statistics, robustness.

Usage: python scripts/r2_evaluate.py sharadar_YYYY-MM-DD
Output: data/derived/<snapshot>/research2/ (summary.json, yearly.parquet, returns.npy, members)
Everything follows docs/research2/PREREGISTRATION.md; both hypotheses are recorded in the
research-2 trial registry before any result is computed.
"""

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

from qlab.data.panel import load_panel
from qlab.engine.vector import Decisions, simulate
from qlab.pipeline import load_config
from qlab.schedule import period_starts
from qlab.selection.metrics import sharpe as col_sharpe
from qlab.selection.select import select
from qlab.strategies.allocation import (H1, AllocationConfig, allocation_decisions,
                                        allocation_grid, allocation_neighbours)
from qlab.validation.metrics import max_drawdown, summary
from qlab.validation.registry import TrialRegistry, config_hash
from qlab.validation.stats import bootstrap_ci, deflated_sharpe, spa_pvalue
from qlab.validation.wfo import SparseDecisions, make_folds, stitch_decisions

COST = 5e-4                       # per side, prereg §3
FIRST_DECISION = "2007-03-01"     # prereg §10 (a)
EVAL_START = "2010-01-01"         # prereg §6
SUBSTITUTES = {"EFA": "VEU", "EEM": "VWO", "IEF": "AGG", "GLD": "IAU", "DBC": "GSG", "VNQ": "IYR"}

snapshot = sys.argv[1]
t0 = time.time()
der = Path("data/derived") / snapshot
panel, extra = load_panel(der / "panel_etf", mmap=False)
names = json.loads((der / "panel_etf" / "tickers.json").read_text())
col = {t: i for i, t in enumerate(names["columns"])}
main = [col[t] for t in names["main"]]
alt = [col[SUBSTITUTES.get(t, t)] for t in names["main"]]
cash = np.asarray(extra["cash_ret"])
dates = panel.dates
first = int(np.searchsorted(dates, np.datetime64(FIRST_DECISION)))
ev = int(np.searchsorted(dates, np.datetime64(EVAL_START)))
decide = period_starts(dates, "M") & (np.arange(len(dates)) >= first)
out = der / "research2"
out.mkdir(parents=True, exist_ok=True)

cfg = load_config()
r2_cfg = copy.deepcopy(cfg)
r2_cfg["hard_filters"]["min_avg_positions"] = 1   # prereg §10 (c)
registry = TrialRegistry(Path("runs/registry/research2.sqlite"))
for h in ("H1", "H2"):
    trial = {"research": 2, "hypothesis": h, "prereg": "docs/research2/PREREGISTRATION.md v1.0"}
    if not registry.has(trial):
        registry.record("methodology_eval", trial, note=f"research 2 {h}")
n_meth = registry.n_meth


def run(cfg_: AllocationConfig, columns: list[int], cost: float = COST):
    d = allocation_decisions(panel, columns, cfg_, cash, decide)
    return d, simulate(panel, d, cost, cash)


def fixed_weights(weights: dict[int, float]):
    days = np.flatnonzero(decide)
    w = np.zeros((len(days), panel.shape[1]))
    for c, x in weights.items():
        w[:, c] = x
    return simulate(panel, Decisions(days, w), COST, cash)


# H1 and benchmarks ------------------------------------------------------------------------------
_, h1 = run(H1, main)
bench = {"60/40": fixed_weights({col["SPY"]: 0.6, col["IEF"]: 0.4}),
         "EW9": fixed_weights({c: 1 / 9 for c in main})}
spy = (1 + panel.ret_co[:, col["SPY"]]) * (1 + panel.ret_oc[:, col["SPY"]]) - 1

# H2: grid -> R -> walk-forward selection -> stitched portfolio ----------------------------------
grid = allocation_grid()
neigh = allocation_neighbours(grid)
R = np.zeros((len(dates), len(grid)))
TURN = np.zeros_like(R)
NPOS = np.zeros_like(R)
decisions = {}
for k, g in enumerate(grid):
    d, r = run(g, main)
    decisions[k] = SparseDecisions(d)
    R[:, k], TURN[:, k], NPOS[:, k] = r.returns, r.turnover, r.n_positions
folds = make_folds(dates, 2010, int(str(dates[-1])[:4]), first + 1, cfg["wfo"]["embargo_days"])
members, rows = [], []
for f in folds:
    sl = slice(f.train_start, f.train_end)
    pol = select(R[sl], cash[sl], dates[sl], TURN[sl], NPOS[sl], grid, neigh, r2_cfg, k=2)
    members.append([int(c) for c in pol.members])
    rows += [{"year": f.year, "rank": i + 1, "config": grid[c].key, "train_score": float(s),
              "n_passed": pol.n_passed} for i, (c, s) in enumerate(zip(pol.members, pol.scores))]
h2 = simulate(panel, stitch_decisions(folds, members, 0.5, decisions, panel.shape[1]), COST, cash)
pl.DataFrame(rows).write_parquet(out / "h2_members.parquet")

# Statistics on the evaluation period -------------------------------------------------------------
def sharpe_diff(x):
    s = col_sharpe(x[:, :2], x[:, 2])
    return float(s[0] - s[1])


def cagr_diff(x):
    g = np.exp(np.log1p(x[:, :2]).sum(axis=0) * 252 / len(x)) - 1
    return float(g[0] - g[1])


def evaluate(r: np.ndarray, sl: slice, n_trials: int, sim=None) -> dict:
    rf = cash[sl]
    res = {"metrics": summary(r[sl], rf, None if sim is None else sim.turnover[sl],
                              None if sim is None else sim.costs[sl],
                              None if sim is None else sim.exposure[sl]),
           "dsr": deflated_sharpe(r[sl] - rf, n_trials, 1.0 / (sl.stop - sl.start))}
    for b, br in (("60/40", bench["60/40"].returns), ("SPY", spy)):
        x = np.column_stack([r[sl], br[sl], rf])
        res[f"sharpe_diff_vs_{b}"] = bootstrap_ci(x, sharpe_diff, n_boot=1000, mean_block=21)
        res[f"cagr_diff_vs_{b}"] = bootstrap_ci(x, cagr_diff, n_boot=1000, mean_block=21)
        res[f"spa_p_vs_{b}"] = spa_pvalue((r[sl] - br[sl])[:, None], n_boot=1000)
    return res


full = slice(ev, len(dates))
stats = {"H1": evaluate(h1.returns, full, 1, h1), "H2": evaluate(h2.returns, full, n_meth, h2),
         "60/40": {"metrics": summary(bench["60/40"].returns[full], cash[full])},
         "SPY": {"metrics": summary(spy[full], cash[full])},
         "EW9": {"metrics": summary(bench["EW9"].returns[full], cash[full])}}

# Robustness (prereg §7) --------------------------------------------------------------------------
sr = lambda r, sl=full: float(col_sharpe(r[sl][:, None], cash[sl])[0])  # noqa: E731
rob = {f"costs_x{m}": sr(run(H1, main, COST * m)[1].returns) for m in (1, 2, 3)}
rob["substitutes"] = sr(run(H1, alt)[1].returns)
for L in (150, 250):
    rob[f"sma{L}"] = sr(run(AllocationConfig("iv126", f"sma{L}", "none"), main)[1].returns)
split = int(np.searchsorted(dates, np.datetime64("2020-01-01")))
for name, sl in (("2010-2019", slice(ev, split)), ("2020-2026", slice(split, len(dates)))):
    rob[f"period_{name}"] = {k: {"sharpe": sr(r, sl), "cagr": float(np.prod(1 + r[sl]) ** (252 / (sl.stop - sl.start)) - 1),
                                 "max_drawdown": max_drawdown(np.cumprod(1 + r[sl]))[0]}
                             for k, r in (("H1", h1.returns), ("H2", h2.returns),
                                          ("60/40", bench["60/40"].returns), ("SPY", spy))}

h1m, b = stats["H1"], stats["60/40"]["metrics"]
criteria = {"ci_sharpe_vs_6040_lower_gt_0": h1m["sharpe_diff_vs_60/40"][1] > 0,
            "max_drawdown_le_25pct": h1m["metrics"]["max_drawdown"] <= 0.25,
            "sharpe_costs_x3_gt_6040": rob["costs_x3"] > b["sharpe"]}

# Forward-test threshold (prereg §6): 10th percentile of rolling 252-day return differences.
cum = lambda r: np.cumprod(1 + r[full])  # noqa: E731
a, c = cum(h1.returns), cum(bench["60/40"].returns)
diff12 = a[252:] / a[:-252] - c[252:] / c[:-252]
forward_threshold = float(np.quantile(diff12, 0.10))

years = np.asarray(dates[full], dtype="datetime64[Y]").astype(int) + 1970
series = {"H1": h1.returns[full], "H2": h2.returns[full], "60/40": bench["60/40"].returns[full],
          "SPY": spy[full], "EW9": bench["EW9"].returns[full]}
yearly = pl.DataFrame([{"year": int(y), **{k: float(np.prod(1 + v[years == y]) - 1)
                                         for k, v in series.items()}} for y in np.unique(years)])
yearly.write_parquet(out / "yearly.parquet")
np.save(out / "returns.npy", np.column_stack(list(series.values())))
result = {"period": [str(dates[ev]), str(dates[-1])], "n_meth": n_meth, "stats": stats,
          "robustness": rob, "criteria_H1": {k: bool(v) for k, v in criteria.items()},
          "criteria_H1_all": bool(all(criteria.values())),
          "forward_threshold_12m_diff_vs_6040": forward_threshold,
          "h1_config": H1.key, "grid_size": len(grid), "folds": len(folds),
          "config_hash": config_hash({"h1": H1.key, "cost": COST, "first": FIRST_DECISION}),
          "seconds": round(time.time() - t0)}
(out / "summary.json").write_text(json.dumps(result, indent=2, default=float))
pl.Config.set_tbl_rows(40)
pl.Config.set_float_precision(3)
print(json.dumps(result, indent=2, default=float))
print(yearly)
print(pl.DataFrame(rows).group_by("config").len().sort("len", descending=True))
