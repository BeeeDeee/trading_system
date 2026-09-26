"""Benchmarks on the development period (spec §8) + engine parity on real data.

Usage: python scripts/run_benchmarks.py sharadar_YYYY-MM-DD
Only data up to the vault boundary (2019-12-31) is used.
"""

import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from qlab.benchmarks import buy_and_hold_returns, equal_weight_targets
from qlab.data.panel import load_panel
from qlab.engine.costs import CostModel
from qlab.engine.ledger import simulate_ledger
from qlab.engine.vector import simulate
from qlab.schedule import period_starts
from qlab.validation.metrics import summary
from qlab.validation.vault import Vault

snapshot = sys.argv[1]
der = Path("data/derived") / snapshot
vault = Vault(date(2019, 12, 31), Path("runs/vault/frozen.lock"), Path("runs/vault/vault.log"))
full, extra = load_panel(der / "panel_liq1000")
end = vault.last_visible_index(full.dates)
panel = full.slice(end)
vault.check_dev_only(panel.dates)
start = int(np.searchsorted(panel.dates, np.datetime64("2000-01-01")))
dates = panel.dates
rf = np.asarray(extra["cash_ret"][:end])
members = np.asarray(extra["in_liq1000"][:end])
cost = CostModel().rate(np.asarray(extra["liq_rank"][:end]), dates)

t0 = time.time()
decide = period_starts(dates, "M") & (np.arange(end) >= start - 1)
decide[start - 1] = True
ew_targets = equal_weight_targets(members, decide)
ew = simulate(panel, ew_targets, cost, rf)
t_vec = time.time() - t0
t0 = time.time()
ledger = simulate_ledger(panel, ew_targets, cost, rf)
t_led = time.time() - t0
parity = float(np.abs(ledger.sim.returns - ew.returns).max())

bh = buy_and_hold_returns(panel, members[start - 1], start, entry_cost=float(np.nanmean(cost[start])))
funds = pl.read_parquet(Path("data/parquet") / snapshot / "funds.parquet",
                        columns=["ticker", "date", "closeadj"]).filter(pl.col("ticker") == "SPY").sort("date")
spy = (pl.DataFrame({"date": dates}).with_columns(pl.col("date").cast(pl.Date))
       .join(funds.with_columns(r=pl.col("closeadj").pct_change()), on="date", how="left")
       ["r"].fill_null(0.0).to_numpy())

s = slice(start, end)
rows = {
    "SPY_TR": summary(spy[s], rf[s]),
    "EW_UNIV (LIQ1000, monthly)": summary(ew.returns[s], rf[s], ew.turnover[s], ew.costs[s],
                                          ew.exposure[s]),
    "BH_UNIV (from 2000-01-03)": summary(bh[s], rf[s]),
    "CASH (T-bill)": summary(rf[s], rf[s]),
}
table = pl.DataFrame([{"benchmark": k, **{m: v[m] for m in (
    "cagr", "volatility", "sharpe", "max_drawdown", "max_drawdown_days")},
    "turnover_annual": v.get("turnover_annual"), "costs_bps_annual": v.get("costs_bps_annual")}
    for k, v in rows.items()])
pl.Config.set_tbl_width_chars(200)
pl.Config.set_float_precision(3)
print(table)
print(json.dumps({"period": f"{dates[start]}..{dates[end - 1]}", "vector_s": round(t_vec, 1),
                  "ledger_s": round(t_led, 1), "max_abs_return_diff_vector_vs_ledger": parity,
                  "avg_positions": float(ew.n_positions[s].mean())}, indent=2))
table.write_parquet(der / "benchmarks_dev.parquet")
