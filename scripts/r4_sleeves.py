"""Research 4 / R4.2: the sleeve library -> daily net returns and monthly targets, 2000-2026.

Usage: python scripts/r4_sleeves.py sharadar_YYYY-MM-DD
Output: data/derived/<snapshot>/research4/sleeves/
  returns.npy (T x S, NaN before a sleeve's first decision), turnover.npy, costs.npy,
  sleeves.parquet (name, family, kind, first decision row), targets.parquet (sleeve, k, col, w),
  days.npy (decision rows), scores.npz (stock scores on decision rows)
"""

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import polars as pl

from qlab.data.panel import load_panel
from qlab.engine.vector import Decisions, simulate
from qlab.pipeline import load_config
from qlab.research4.fundamentals import fundamental_scores
from qlab.research4.sleeves import (STOCK_SCORES, add_composites, cost_rates, etf_decisions,
                                    price_scores, sleeve_library, stock_decisions)
from qlab.schedule import period_starts

FIRST_DECISION = "2000-01-01"

snapshot = sys.argv[1]
t0 = time.time()
der = Path("data/derived") / snapshot
out = der / "research4" / "sleeves"
out.mkdir(parents=True, exist_ok=True)
panel, extra = load_panel(der / "panel_r4")
meta = json.loads((der / "panel_r4" / "columns.json").read_text())
etf_col, n0 = meta["etf"], meta["n_stock_cols"]
T, N = panel.shape
cash = np.asarray(extra["cash_ret"])
days = np.flatnonzero(period_starts(panel.dates, "M")
                      & (panel.dates >= np.datetime64(FIRST_DECISION)))
np.save(out / "days.npy", days)
universe = np.asarray(extra["in_liq1000"][days])

scores_path = out / "scores.npz"
if scores_path.exists():
    scores = dict(np.load(scores_path))
else:
    scores = price_scores(panel, days, etf_col["SPY"])
    print(f"price scores ({time.time() - t0:.0f}s)", flush=True)
    fund = fundamental_scores(Path("data/parquet") / snapshot, panel.assets[:n0], panel.dates, days)
    for k, v in fund.items():
        full = np.full((len(days), N), np.nan, dtype=np.float32)
        full[:, :n0] = v
        scores[k] = full
    print(f"fundamental scores ({time.time() - t0:.0f}s)", flush=True)
    add_composites(scores, universe)
    scores = {k: v.astype(np.float32) for k, v in scores.items()}
    np.savez(scores_path, **scores)
missing = set(STOCK_SCORES) - set(scores)
assert not missing, missing
for k in STOCK_SCORES:  # coverage: rankable universe members per decision day
    c = (np.isfinite(scores[k]) & universe).sum(axis=1)
    print(f"  {k:18s} coverage 2000: {c[:12].mean():6.0f}  2010: {c[120:132].mean():6.0f}  "
          f"2025: {c[-20:-8].mean():6.0f}")

cfg = load_config()
rate = cost_rates(extra["liq_rank"], panel.dates, cfg["costs"], list(etf_col.values()))

lib = sleeve_library()
S = len(lib)
R = np.full((T, S), np.nan)
TURN = np.zeros((T, S))
COST = np.zeros((T, S))
first_row, rows = [], []
for s, sl in enumerate(lib):
    if sl.kind == "stock":
        d = stock_decisions(scores[sl.score], universe, days, sl.top_n)
        first = 0
    elif sl.kind == "etf":
        d, first = etf_decisions(panel, etf_col[sl.ticker], days)
    else:
        d, first = Decisions(days, np.zeros((len(days), N))), 0
    start = days[first] + 1 if first < len(days) else T
    first_row.append(int(start))
    if start < T:
        sim = simulate(panel, d, rate, cash)
        R[start:, s] = sim.returns[start:]
        TURN[:, s], COST[:, s] = sim.turnover, sim.costs
    for k, w in enumerate(d.weights):
        nz = np.flatnonzero(w)
        rows.append(pl.DataFrame({"sleeve": np.full(len(nz), s, dtype=np.int32),
                                  "k": np.full(len(nz), first + k, dtype=np.int32),
                                  "col": nz.astype(np.int32), "w": w[nz]}))
    print(f"{s + 1}/{S} {sl.name} ({time.time() - t0:.0f}s)", flush=True)

np.save(out / "returns.npy", R)
np.save(out / "turnover.npy", TURN)
np.save(out / "costs.npy", COST)
pl.concat(rows).write_parquet(out / "targets.parquet")
pl.DataFrame([{**asdict(sl), "first_row": f} for sl, f in zip(lib, first_row)]).write_parquet(
    out / "sleeves.parquet")
print(f"done {S} sleeves ({time.time() - t0:.0f}s)")
