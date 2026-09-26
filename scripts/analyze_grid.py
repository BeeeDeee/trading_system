"""Phase 2 analysis of a grid run on the development period -> docs/PHASE2_REPORT.md tables.

Usage: python scripts/analyze_grid.py sharadar_YYYY-MM-DD <run_hash>
Full-period metrics are in-sample diagnostics (the grid is not selected on them); selection happens
walk-forward in Phase 3. Cost sensitivity uses r_k = r - (k - 1) * cost (first-order approximation).
"""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from qlab.data.panel import load_panel
from qlab.selection.metrics import window_summary
from qlab.selection.ranking import (correlation_clusters, hard_filter, neighbour_lists,
                                    robust_score)
from qlab.strategies.grid import build_grid
from qlab.validation.stats import deflated_sharpe, pbo_cscv

snapshot, run_hash = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(Path("configs/frozen_defaults.yaml").read_text())
der = Path("data/derived") / snapshot
out = der / "candidates" / run_hash
meta = json.loads((out / "meta.json").read_text())
done = np.load(out / "done.npy")
if not done.all():
    sys.exit(f"grid run incomplete: {done.sum()}/{len(done)}")
cands = pl.read_parquet(out / "candidates.parquet")
R = np.load(out / "R.npy", mmap_mode="r")
turnover = np.load(out / "turnover.npy", mmap_mode="r")
costs = np.load(out / "costs.npy", mmap_mode="r")
n_pos = np.load(out / "n_pos.npy", mmap_mode="r")
exposure = np.load(out / "exposure.npy", mmap_mode="r")

full, extra = load_panel(der / "panel_liq1000")
start = int(np.searchsorted(full.dates, np.datetime64(meta["dates"][0])))
end = start + meta["n_days"]
dates = full.dates[start:end]
rf = np.asarray(extra["cash_ret"][start:end])

grid = build_grid(cfg["grid"])
by_id = {c.candidate_id: c for c in grid}
grid = [by_id[i] for i in cands["candidate_id"]]  # column order of R

m = window_summary(R, rf, dates, turnover, n_pos, exposure)
m["is_regime"] = (cands["family"] == "regime_market").to_numpy()
m["costs_bps_annual"] = np.asarray(costs, dtype=np.float64).sum(axis=0) / (len(dates) / 252) * 1e4
neigh = neighbour_lists(grid, cfg["grid"])
score, neigh_med = robust_score(m["sharpe"], neigh)
passed = hard_filter(m, cfg["hard_filters"])
cost_sum = np.asarray(costs, dtype=np.float64)
sr_x3 = window_summary(np.asarray(R, dtype=np.float64) - 2 * cost_sum, rf, dates)["sharpe"]

table = cands.with_columns(
    sharpe=m["sharpe"], cagr=m["cagr"], max_drawdown=m["max_drawdown"],
    share_positive_years=m["share_positive_years"], turnover=m["turnover_annual"],
    avg_positions=m["avg_positions"], avg_exposure=m["avg_exposure"],
    costs_bps=m["costs_bps_annual"], neigh_median_sharpe=neigh_med, robust_score=score,
    passes_filters=passed, sharpe_costs_x3=sr_x3)
table.write_parquet(out / "metrics_full_dev.parquet")

families = (table.group_by("family").agg(
    candidates=pl.len(),
    median_sharpe=pl.col("sharpe").median(),
    median_neigh_sharpe=pl.col("neigh_median_sharpe").median(),
    best_robust=pl.col("robust_score").max(),
    median_cagr=pl.col("cagr").median(),
    median_mdd=pl.col("max_drawdown").median(),
    median_turnover=pl.col("turnover").median(),
    median_costs_bps=pl.col("costs_bps").median(),
    median_sharpe_x3=pl.col("sharpe_costs_x3").median(),
    pass_filters=pl.col("passes_filters").sum(),
).sort("median_neigh_sharpe", descending=True))

by_dim = {dim: table.group_by(dim).agg(median_sharpe=pl.col("sharpe").median(),
                                       median_mdd=pl.col("max_drawdown").median())
          .sort(dim) for dim in ("overlay", "rebalance", "top_n", "weighting", "hysteresis")}

pbo, logits = pbo_cscv(np.asarray(R, dtype=np.float64) - rf[:, None], n_blocks=16)
labels = correlation_clusters(R, cfg["ranking"]["dedup_correlation"])
n_eff = len(np.unique(labels))
best = int(np.argmax(m["sharpe"]))
sr_daily = m["sharpe"] / np.sqrt(252)
excess_best = np.asarray(R[:, best], dtype=np.float64) - rf
dsr_all = deflated_sharpe(excess_best, len(grid), float(np.var(sr_daily)))
dsr_eff = deflated_sharpe(excess_best, n_eff, float(np.var(sr_daily)))

top = (table.sort("robust_score", descending=True).head(15)
       .select("family", "signal_key", "top_n", "weighting", "rebalance", "hysteresis", "overlay",
               "sharpe", "neigh_median_sharpe", "cagr", "max_drawdown", "turnover", "costs_bps",
               "passes_filters"))

gate_family = families["median_neigh_sharpe"].max() > 0
gate_pbo = pbo <= cfg["gates"]["phase2_pbo_max"]
summary = {
    "period": meta["dates"], "candidates": len(grid), "effective_candidates": n_eff,
    "pbo": pbo, "pbo_logit_median": float(np.median(logits)),
    "best": {"family": grid[best].family, "signal_key": grid[best].signal_key,
             "sharpe": float(m["sharpe"][best]), "dsr_n_all": dsr_all, "dsr_n_eff": dsr_eff},
    "passing_hard_filters": int(passed.sum()),
    "gate_family_median_neigh_sharpe_gt_0": bool(gate_family), "gate_pbo": bool(gate_pbo),
}
(out / "phase2_summary.json").write_text(json.dumps(summary, indent=2, default=str))
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_width_chars(250)
pl.Config.set_float_precision(3)
print(json.dumps(summary, indent=2, default=str))
print(families)
for dim, df in by_dim.items():
    print(df)
print(top)
families.write_parquet(out / "families.parquet")
top.write_parquet(out / "top_robust.parquet")
