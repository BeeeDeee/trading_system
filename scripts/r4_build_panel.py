"""Research 4 / R4.1: LIQ1000 panel + ETF columns in one memory-mapped panel.

Usage: python scripts/r4_build_panel.py sharadar_YYYY-MM-DD
Output: data/derived/<snapshot>/panel_r4/ (LIQ1000 columns first, then ETFs; SPY stays in its
LIQ1000-panel column), extras in_liq1000, liq_rank, cash_ret and columns.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

from qlab.data.normalize import normalize_prices
from qlab.data.panel import _MATRICES, load_panel, panel_from_bars
from qlab.data.schema import DELISTINGS_SCHEMA, PRICES_SCHEMA
from qlab.research4.sleeves import ETFS

snapshot = sys.argv[1]
src = Path("data/parquet") / snapshot
der = Path("data/derived") / snapshot
out = der / "panel_r4"
out.mkdir(parents=True, exist_ok=True)

base, extra = load_panel(der / "panel_liq1000")
spy_perm = json.loads((der / "panel_liq1000" / "special_assets.json").read_text())["SPY"]
spy_col = int(np.searchsorted(base.assets, spy_perm))
add = [t for t in ETFS if t != "SPY"]

perm = (pl.read_parquet(src / "tickers.parquet")
        .filter((pl.col("table") == "SFP") & pl.col("ticker").is_in(add))
        .select("ticker", pl.col("permaticker").cast(pl.Int64)))
prices = (pl.read_parquet(src / "funds.parquet").filter(pl.col("ticker").is_in(add))
          .join(perm, on="ticker").select([pl.col(c).cast(t) for c, t in PRICES_SCHEMA.items()])
          .sort("permaticker", "date"))
bars = normalize_prices(prices, pl.DataFrame(schema=DELISTINGS_SCHEMA), base.dates.tolist())
etf = panel_from_bars(bars, base.dates.tolist())
assert (etf.dates == base.dates).all()
by_perm = dict(zip(perm["permaticker"].to_list(), perm["ticker"].to_list()))
etf_tickers = [by_perm[int(a)] for a in etf.assets]

n0, n1 = base.shape[1], etf.shape[1]
shape = (base.shape[0], n0 + n1)
np.save(out / "dates.npy", base.dates)
np.save(out / "assets.npy", np.concatenate([base.assets, etf.assets]))
for name in _MATRICES:
    a, b = getattr(base, name), getattr(etf, name)
    m = np.lib.format.open_memmap(out / f"{name}.npy", mode="w+", dtype=a.dtype, shape=shape)
    for s in range(0, shape[0], 1000):  # row chunks keep memory low
        m[s:s + 1000, :n0] = a[s:s + 1000]
        m[s:s + 1000, n0:] = b[s:s + 1000].astype(a.dtype)
    m.flush()
    del m
for name, fill in (("in_liq1000", False), ("liq_rank", np.nan)):
    a = extra[name]
    m = np.lib.format.open_memmap(out / f"extra_{name}.npy", mode="w+", dtype=a.dtype, shape=shape)
    for s in range(0, shape[0], 1000):
        m[s:s + 1000, :n0] = a[s:s + 1000]
        m[s:s + 1000, n0:] = fill
    m.flush()
    del m
np.save(out / "extra_cash_ret.npy", np.asarray(extra["cash_ret"]))
columns = {"SPY": spy_col, **{t: n0 + i for i, t in enumerate(etf_tickers)}}
(out / "columns.json").write_text(json.dumps({"etf": columns, "n_stock_cols": n0}, indent=2))
first = {t: str(base.dates[np.argmax(etf.listed[:, i])]) for i, t in enumerate(etf_tickers)}
print(f"panel_r4 {shape}; ETF first dates: {first}")
