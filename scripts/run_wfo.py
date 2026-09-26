"""Phase 3: walk-forward selection + stitched out-of-sample ensemble (spec §9.2-9.6).

Usage: python scripts/run_wfo.py sharadar_YYYY-MM-DD <grid_run_hash> [--universe sp500] [--delisting optimistic|pessimistic]
Output: data/derived/<snapshot>/wfo/<method_hash>/ (members, returns, summary.json).
Both pre-registered variants (K=2 primary, K=3 secondary) are recorded in the trial registry as
methodology evaluations BEFORE any result is computed.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

from qlab.benchmarks import equal_weight_targets
from qlab.engine.ledger import simulate_ledger
from qlab.engine.vector import simulate
from qlab.pipeline import dev_setup, spy_returns
from qlab.schedule import period_starts
from qlab.selection.metrics import sharpe as col_sharpe
from qlab.selection.ranking import neighbour_lists
from qlab.selection.select import select
from qlab.strategies.grid import build_grid
from qlab.strategies.run import decisions_for
from qlab.validation.metrics import summary
from qlab.validation.registry import TrialRegistry, config_hash
from qlab.validation.stats import bootstrap_ci, deflated_sharpe, sharpe, spa_pvalue
from qlab.validation.wfo import SparseDecisions, make_folds, stitch_decisions

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("snapshot")
ap.add_argument("grid_hash")
ap.add_argument("--universe", default="liq1000", choices=["liq1000", "sp500"])
ap.add_argument("--delisting", default="base", choices=["base", "optimistic", "pessimistic"])
args = ap.parse_args()
snapshot, grid_hash = args.snapshot, args.grid_hash
robustness = args.universe != "liq1000" or args.delisting != "base"
t0 = time.time()
setup = dev_setup(snapshot, universe=args.universe, delisting=args.delisting)
cfg, ctx, panel = setup.cfg, setup.ctx, setup.panel
der = Path("data/derived") / snapshot
gdir = der / "candidates" / grid_hash
cands = pl.read_parquet(gdir / "candidates.parquet")
R = np.load(gdir / "R.npy", mmap_mode="r")
TURN = np.load(gdir / "turnover.npy", mmap_mode="r")
NPOS = np.load(gdir / "n_pos.npy", mmap_mode="r")
off = setup.start  # R row 0 = panel row `start`
rf_all = np.asarray(ctx.cash_ret)
by_id = {c.candidate_id: c for c in build_grid(cfg["grid"])}
grid = [by_id[i] for i in cands["candidate_id"]]
neigh = neighbour_lists(grid, cfg["grid"])

w = cfg["wfo"]
folds = make_folds(panel.dates, w["first_test_year"], w["last_test_year"], setup.start,
                   w["embargo_days"])
variants = {"primary": cfg["ensemble"]["primary"]["k"], "secondary": cfg["ensemble"]["secondary"]["k"]}

registry = TrialRegistry(Path("runs/registry/trials.sqlite"))
method = {"grid_run": grid_hash, "hard_filters": cfg["hard_filters"], "ranking": cfg["ranking"],
          "wfo": w, "ensemble": cfg["ensemble"], "select": "qlab.selection.select v1"}
method_hash = config_hash(method)
for name, k in variants.items():
    trial = {**method, "variant": name, "k": k}
    if not registry.has(trial):  # robustness runs evaluate the same methodology on perturbed data
        registry.record("other" if robustness else "methodology_eval", trial,
                        note=f"WFO {name} K={k} ({method_hash})"
                        + (f" robustness {args.universe}/{args.delisting}" if robustness else ""))
n_meth = registry.n_meth
out = der / "wfo" / method_hash
out.mkdir(parents=True, exist_ok=True)

# 1. Selection per fold (only training-window rows reach select()).
policies = {name: [] for name in variants}
fold_rows = []
for f in folds:
    a, b = f.train_start - off, f.train_end - off
    win = dict(R=np.asarray(R[a:b]), rf=rf_all[f.train_start:f.train_end],
               dates=panel.dates[f.train_start:f.train_end], turnover=np.asarray(TURN[a:b]),
               n_pos=np.asarray(NPOS[a:b]))
    for name, k in variants.items():
        pol = select(win["R"], win["rf"], win["dates"], win["turnover"], win["n_pos"], grid, neigh,
                     cfg, k)
        policies[name].append(pol)
        for rank, (c, s) in enumerate(zip(pol.members, pol.scores), 1):
            g = grid[c]
            fold_rows.append({"variant": name, "year": f.year, "rank": rank, "col": int(c),
                              "family": g.family, "signal_key": g.signal_key, "top_n": g.top_n,
                              "weighting": g.weighting, "rebalance": g.rebalance,
                              "hysteresis": g.hysteresis, "overlay": g.overlay,
                              "train_score": float(s), "n_passed": pol.n_passed,
                              "n_clusters": pol.n_clusters})
members_df = pl.DataFrame(fold_rows)
members_df.write_parquet(out / "members.parquet")
print(f"selection done ({time.time() - t0:.0f}s)")

# 2. Point-in-time decisions of every selected candidate, then the stitched portfolio.
needed = sorted({int(c) for pols in policies.values() for p in pols for c in p.members})
decisions = {}
for c in sorted(needed, key=lambda c: grid[c].signal_key):
    decisions[c] = SparseDecisions(decisions_for(grid[c], ctx))
first, last = folds[0].test_start, folds[-1].test_end
oos = slice(first, last)
results = {}
for name, pols in policies.items():
    d = stitch_decisions(folds, [list(p.members) for p in pols], pols[0].weight_each, decisions,
                         panel.shape[1])
    results[name] = simulate(panel, d, ctx.cost_rate, ctx.cash_ret)
    if name == "primary":
        led = simulate_ledger(panel, d, ctx.cost_rate, ctx.cash_ret)
        parity = float(np.abs(led.sim.returns[oos] - results[name].returns[oos]).max())
        n_orders = int((led.orders["side"] != "delisting").sum())
print(f"portfolios done ({time.time() - t0:.0f}s)")

# 3. Benchmarks over the same out-of-sample years.
members_mask = np.asarray(ctx.universe)
decide = period_starts(panel.dates, "M") & (np.arange(setup.end) >= first - 1)
decide[first - 1] = True
decide[: first - 1] = False
ew = simulate(panel, equal_weight_targets(members_mask, decide), ctx.cost_rate, ctx.cash_ret)
spy = spy_returns(snapshot, panel.dates)
rf = rf_all[oos]
series = {"primary": results["primary"].returns[oos], "secondary": results["secondary"].returns[oos],
          "EW_UNIV": ew.returns[oos], "SPY_TR": spy[oos]}

# 4. Statistics.
def sharpe_diff(x):  # columns: strategy, benchmark, rf
    return float(col_sharpe(x[:, :2], x[:, 2])[0] - col_sharpe(x[:, :2], x[:, 2])[1])


def cagr_diff(x):
    g = np.exp(np.log1p(x[:, :2]).sum(axis=0) * 252 / len(x)) - 1
    return float(g[0] - g[1])


stats = {}
for name in variants:
    r = series[name]
    ex = r - rf
    s = {"metrics": summary(r, rf, results[name].turnover[oos], results[name].costs[oos],
                            results[name].exposure[oos]),
         "dsr": deflated_sharpe(ex, n_meth, 1.0 / len(ex)),
         "n_meth": n_meth}
    for bench in ("EW_UNIV", "SPY_TR"):
        x = np.column_stack([r, series[bench], rf])
        s[f"sharpe_diff_vs_{bench}"] = bootstrap_ci(x, sharpe_diff, n_boot=1000, mean_block=21)
        s[f"cagr_diff_vs_{bench}"] = bootstrap_ci(x, cagr_diff, n_boot=1000, mean_block=21)
        s[f"spa_p_vs_{bench}"] = spa_pvalue((r - series[bench])[:, None], n_boot=1000)
    stats[name] = s
for bench in ("EW_UNIV", "SPY_TR"):
    stats[bench] = {"metrics": summary(series[bench], rf)}

years = np.asarray(panel.dates[oos], dtype="datetime64[Y]").astype(int) + 1970
yearly = pl.DataFrame([{"year": int(y), **{k: float(np.prod(1 + v[years == y]) - 1)
                                         for k, v in series.items()}} for y in np.unique(years)])
yearly.write_parquet(out / "yearly.parquet")
np.save(out / "returns.npy", np.column_stack([series[k] for k in series]))

p = stats["primary"]
gate_dsr = p["dsr"] >= cfg["gates"]["phase3_dsr_min"]
gate_ci = p["sharpe_diff_vs_EW_UNIV"][1] > cfg["gates"]["phase3_ci_vs_ew_univ_lower_bound_min"]
result = {"method_hash": method_hash, "grid_run": grid_hash,
          "oos_period": [str(panel.dates[first]), str(panel.dates[last - 1])],
          "folds": len(folds), "n_meth": n_meth, "ledger_parity_primary": parity,
          "ledger_orders_primary": n_orders, "stats": stats,
          "gate": {"dsr": bool(gate_dsr), "ci_sharpe_vs_ew": bool(gate_ci),
                   "passed": bool(gate_dsr and gate_ci)},
          "seconds": round(time.time() - t0)}
(out / "summary.json").write_text(json.dumps(result, indent=2, default=float))
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_width_chars(250)
pl.Config.set_float_precision(3)
print(json.dumps(result, indent=2, default=float))
print(yearly)
print(members_df.filter(pl.col("variant") == "primary").select(
    "year", "rank", "family", "signal_key", "top_n", "weighting", "rebalance", "hysteresis",
    "overlay", "train_score", "n_passed", "n_clusters"))
