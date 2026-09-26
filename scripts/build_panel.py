"""Dense, memory-mapped panel of all securities that were ever in LIQ1000 (spec §5, §7).

Usage: python scripts/build_panel.py sharadar_YYYY-MM-DD
Output: data/derived/<snapshot>/panel_liq1000/*.npy (+ extra_liq_rank, extra_in_liq1000,
extra_in_sp500, extra_cash_ret)
"""

import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from qlab.data.schema import DELISTED
from qlab.engine.costs import cash_returns

snapshot = sys.argv[1]
der = Path("data/derived") / snapshot
out = der / "panel_liq1000"
t0 = time.time()

liq = pl.read_parquet(der / "universe_liq1000.parquet")
sp = pl.read_parquet(der / "universe_sp500.parquet")
assets = np.sort(liq["permaticker"].unique().to_numpy())
con = duckdb.connect()
con.execute("SET memory_limit='1200MB'; SET threads=2; SET preserve_insertion_order=false")
dates = np.array([d for (d,) in con.sql(
    f"SELECT DISTINCT date FROM read_parquet('{der / 'bars' / '*.parquet'}') ORDER BY 1").fetchall()],
    dtype="datetime64[D]")
shape = (len(dates), len(assets))
print(f"panel {shape}, building...")

out.mkdir(parents=True, exist_ok=True)
spec = {"ret_co": (np.float64, 0.0), "ret_oc": (np.float64, 0.0), "tradable": (bool, False),
        "listed": (bool, False), "delisting": (bool, False), "close_u": (np.float32, np.nan),
        "dollar_volume": (np.float32, np.nan), "extra_liq_rank": (np.float32, np.nan),
        "extra_in_liq1000": (bool, False), "extra_in_sp500": (bool, False)}
mats = {}
for name, (dtype, fill) in spec.items():
    m = np.lib.format.open_memmap(out / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
    m[:] = fill
    mats[name] = m


def index(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    t = np.searchsorted(dates, df["date"].to_numpy().astype("datetime64[D]"))
    n = np.searchsorted(assets, df["permaticker"].to_numpy())
    return t, n


asset_filter = pl.col("permaticker").is_in(pl.Series(assets).implode())
for part in sorted((der / "bars").glob("part-*.parquet")):
    b = pl.read_parquet(part).filter(asset_filter)
    if b.is_empty():
        continue
    t, n = index(b)
    mats["ret_co"][t, n] = b["ret_co"].to_numpy()
    mats["ret_oc"][t, n] = b["ret_oc"].to_numpy()
    mats["tradable"][t, n] = b["tradable"].to_numpy()
    mats["listed"][t, n] = True
    mats["delisting"][t, n] = (b["status"] == DELISTED).to_numpy()
    mats["close_u"][t, n] = b["close_u"].to_numpy()
    mats["dollar_volume"][t, n] = b["dollar_volume"].to_numpy()

# Liquidity rank among all securities with trading volume (for the cost tiers).
con.register("assets", pl.DataFrame({"permaticker": assets}))
ranks = con.sql(f"""
    SELECT * FROM (
        SELECT date, permaticker,
               rank() OVER (PARTITION BY date ORDER BY dv_median DESC) AS liq_rank
        FROM read_parquet('{der / 'universe_features' / '*.parquet'}')
        WHERE dv_median > 0 AND NOT spac)
    WHERE permaticker IN (SELECT permaticker FROM assets)""").pl()
t, n = index(ranks)
mats["extra_liq_rank"][t, n] = ranks["liq_rank"].to_numpy()
t, n = index(liq)
mats["extra_in_liq1000"][t, n] = True
sp_in = sp.filter(asset_filter)
t, n = index(sp_in)
mats["extra_in_sp500"][t, n] = True
np.save(out / "dates.npy", dates)
np.save(out / "assets.npy", assets)

rf = pl.read_csv(sorted(Path("data/raw").glob("fred_*/DTB3.csv"))[-1], null_values=".",
                 schema_overrides={"DTB3": pl.Float64})
np.save(out / "extra_cash_ret.npy", cash_returns(
    dates, rf["observation_date"].str.to_date().to_numpy(), rf["DTB3"].to_numpy()))
for m in mats.values():
    m.flush()
size = sum(f.stat().st_size for f in out.glob("*.npy")) / 1e9
print(f"done in {time.time() - t0:.0f}s, {size:.2f} GB on disk, "
      f"sp500 members outside LIQ1000 assets: {sp.height - sp_in.height:,} rows")
