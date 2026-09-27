"""Research 5 / R5.1: panel of every common stock that ever passed the STR-TF universe filter at
the widest liquidity threshold (ADV20 >= 5M USD), plus SPY (pre-registration §3).

Usage: python scripts/r5_build_panel.py sharadar_YYYY-MM-DD
Output: data/derived/<snapshot>/panel_r5/ with the Panel matrices and extras
  high_u, low_u      unadjusted high/low (IBS, ATR)
  adv20              mean dollar volume of the last 20 bars (point in time)
  base_ok            close_u >= 5, >= 252 bars, not a SPAC at t (universe minus the ADV filter)
  sic                SIC code valid at t (0 = unknown)
  earn8k             8-K Item 2.02 (earnings release) filed on this trading day (or the weekend
                     before it)
  liq_rank           liquidity rank among all securities (research 1 cost model, sensitivity only)
  cash_ret           daily T-bill return
"""

import json
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from qlab.data.normalize import normalize_prices
from qlab.data.schema import DELISTED, DELISTINGS_SCHEMA, PRICES_SCHEMA
from qlab.engine.costs import cash_returns
from qlab.research5.universe import MIN_ADV_PANEL, r5_features

snapshot = sys.argv[1]
src = Path("data/parquet") / snapshot
der = Path("data/derived") / snapshot
out = der / "panel_r5"
feat_dir = der / "r5_features"
feat_dir.mkdir(parents=True, exist_ok=True)
t0 = time.time()

sic = pl.read_parquet(der / "sic_history.parquet")
parts = sorted((der / "bars").glob("part-*.parquet"))
for part in parts:
    r5_features(pl.read_parquet(part), sic).write_parquet(feat_dir / part.name)
feats = pl.scan_parquet(feat_dir / "*.parquet")
stock_assets = (feats.filter(pl.col("base_ok") & (pl.col("adv20") >= MIN_ADV_PANEL))
                .select(pl.col("permaticker").unique()).collect()["permaticker"].to_numpy())
print(f"features done in {time.time() - t0:.0f}s; {len(stock_assets):,} stocks ever eligible")

spy_perm = int(pl.read_parquet(src / "tickers.parquet").filter(
    (pl.col("table") == "SFP") & (pl.col("ticker") == "SPY"))["permaticker"].cast(pl.Int64).item())
assets = np.sort(np.append(stock_assets, spy_perm))
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
        "dollar_volume": (np.float32, np.nan), "extra_high_u": (np.float32, np.nan),
        "extra_low_u": (np.float32, np.nan), "extra_adv20": (np.float32, np.nan),
        "extra_base_ok": (bool, False), "extra_sic": (np.int16, 0),
        "extra_earn8k": (bool, False), "extra_liq_rank": (np.float32, np.nan)}
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
for part in parts:
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
    mats["extra_high_u"][t, n] = b["high_u"].to_numpy()
    mats["extra_low_u"][t, n] = b["low_u"].to_numpy()
    f = pl.read_parquet(feat_dir / part.name).filter(asset_filter)
    t, n = index(f)
    mats["extra_adv20"][t, n] = f["adv20"].to_numpy()
    mats["extra_base_ok"][t, n] = f["base_ok"].to_numpy()
    mats["extra_sic"][t, n] = f["sic"].fill_null(0).to_numpy()

spy_px = (pl.read_parquet(src / "funds.parquet").filter(pl.col("ticker") == "SPY")
          .with_columns(permaticker=pl.lit(spy_perm, dtype=pl.Int64))
          .select([pl.col(c).cast(t) for c, t in PRICES_SCHEMA.items()]).sort("date"))
spy_bars = normalize_prices(spy_px, pl.DataFrame(schema=DELISTINGS_SCHEMA), dates.tolist())
t, n = index(spy_bars)
for name in ("ret_co", "ret_oc", "tradable", "close_u", "dollar_volume"):
    mats[name][t, n] = spy_bars[name].to_numpy()
mats["listed"][t, n] = True
(out / "special_assets.json").write_text(json.dumps({"SPY": spy_perm}))

# Earnings releases: 8-K Item 2.02 (Sharadar event code 22) by filing date; a filing on a
# non-trading day belongs to the next trading day.
sep = (pl.read_parquet(src / "tickers.parquet").filter(pl.col("table") == "SEP")
       .select("ticker", pl.col("permaticker").cast(pl.Int64)))
earn = (pl.read_parquet(src / "events.parquet")
        .filter(pl.col("eventcodes").str.split("|").list.contains("22"))
        .join(sep, on="ticker").filter(asset_filter).select("permaticker", "date"))
t = np.searchsorted(dates, earn["date"].to_numpy().astype("datetime64[D]"), side="left")
ok = t < len(dates)
mats["extra_earn8k"][t[ok], np.searchsorted(assets, earn["permaticker"].to_numpy()[ok])] = True

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
mats["extra_liq_rank"][:, int(np.searchsorted(assets, spy_perm))] = 1

np.save(out / "dates.npy", dates)
np.save(out / "assets.npy", assets)
rf = pl.read_csv(sorted(Path("data/raw").glob("fred_*/DTB3.csv"))[-1], null_values=".",
                 schema_overrides={"DTB3": pl.Float64})
np.save(out / "extra_cash_ret.npy", cash_returns(
    dates, rf["observation_date"].str.to_date().to_numpy(), rf["DTB3"].to_numpy()))
for m in mats.values():
    m.flush()
size = sum(f.stat().st_size for f in out.glob("*.npy")) / 1e9
print(f"done in {time.time() - t0:.0f}s, {size:.2f} GB on disk; earnings 8-K rows {earn.height:,}")
