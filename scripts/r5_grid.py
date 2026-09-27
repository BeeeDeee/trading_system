"""Research 5 / R5.3: the 972-config STR-TF grid on the development period (pre-registration §8).

Usage: python scripts/r5_grid.py sharadar_YYYY-MM-DD
Output: runs/research5/grid.parquet (metrics per config), runs/research5/grid_returns.npy
(daily net returns, dev days x configs, float32, same order as grid.parquet)
"""

import itertools
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl

from qlab.research5.report import portfolio_stats, spy_returns, trade_stats
from qlab.research5.setup import load_stage, log_experiment, registry
from qlab.research5.strategy import GRID, Config, candidates, grid_configs, run

snapshot = sys.argv[1]
out_dir = Path("runs/research5")
out_dir.mkdir(parents=True, exist_ok=True)
ctx = load_stage(snapshot, "dev")
spy = spy_returns(ctx)
configs = grid_configs()
registry().record("candidates", {"grid": {k: list(v) for k, v in GRID.items()}, "stage": "dev"},
                  len(configs), "research5 dev grid")

index = {c: i for i, c in enumerate(configs)}
returns = np.lib.format.open_memmap(out_dir / "grid_returns.npy", mode="w+", dtype=np.float32,
                                    shape=(ctx.end - ctx.start, len(configs)))
rows = [None] * len(configs)
t0 = time.time()
signal_keys = itertools.product(GRID["lookback_ret"], GRID["trend_sma"], GRID["min_adv"],
                                GRID["entry_z"])
for k, (lb, tsma, adv, ez) in enumerate(signal_keys):
    base = Config(entry_z=ez, lookback_ret=lb, trend_sma=tsma, min_adv=adv)
    cands = candidates(ctx, base)
    for ex, mh, mp in itertools.product(GRID["exit_sma"], GRID["max_hold"], GRID["max_positions"]):
        cfg = replace(base, exit_sma=ex, max_hold=mh, max_positions=mp)
        res = run(ctx, cfg, cands)
        i = index[cfg]
        returns[:, i] = res.returns
        rows[i] = {**{f: getattr(cfg, f) for f in GRID}, **portfolio_stats(res, spy),
                   **trade_stats(res.trades)}
    print(f"{k + 1}/54 signal sets, {time.time() - t0:.0f}s", flush=True)
returns.flush()
grid = pl.DataFrame(rows)
grid.write_parquet(out_dir / "grid.parquet")
log_experiment("dev", {"grid": "972"}, {"n_configs": len(configs),
                                        "median_sharpe": float(grid["sharpe"].median())},
               "grid run")
print(f"done in {time.time() - t0:.0f}s")
