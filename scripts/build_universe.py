"""Point-in-time universes from bars (spec §5).

Usage: python scripts/build_universe.py sharadar_YYYY-MM-DD
Output in data/derived/<snapshot>/: universe_liq1000.parquet, universe_sp500.parquet
"""

import sys
import time
from pathlib import Path

import polars as pl

from qlab.universe import eligibility_features, rank_universe, sp500_intervals

snapshot = sys.argv[1]
der = Path("data/derived") / snapshot
t0 = time.time()
sic = pl.read_parquet(der / "sic_history.parquet")
feat_dir = der / "universe_features"
feat_dir.mkdir(exist_ok=True)
for part in sorted((der / "bars").glob("part-*.parquet")):
    eligibility_features(pl.read_parquet(part), sic).write_parquet(feat_dir / part.name)
liq = rank_universe(pl.scan_parquet(feat_dir / "*.parquet"), top_n=1000)
liq.write_parquet(der / "universe_liq1000.parquet")

tickers = pl.read_parquet(Path("data/parquet") / snapshot / "tickers.parquet").filter(
    pl.col("table") == "SEP").select(pl.col("permaticker").cast(pl.Int64), "ticker")
iv = sp500_intervals(pl.read_parquet(Path("data/parquet") / snapshot / "sp500.parquet"))
members = iv.join(tickers, on="ticker")
sp = (pl.scan_parquet(der / "bars" / "*.parquet").filter(pl.col("status") != "delisted")
      .select("permaticker", "date").join(members.lazy(), on="permaticker")
      .filter((pl.col("start").is_null() | (pl.col("date") >= pl.col("start")))
              & (pl.col("end").is_null() | (pl.col("date") < pl.col("end"))))
      .select("date", "permaticker").unique().sort("date", "permaticker").collect())
sp.write_parquet(der / "universe_sp500.parquet")

per_day = liq.group_by("date").len()
print(f"LIQ1000: {liq.height:,} rows, {liq['permaticker'].n_unique():,} securities ever, "
      f"members/day min {per_day['len'].min()} median {per_day['len'].median():.0f}")
sp_day = sp.group_by("date").len()
print(f"SP500_PIT: {sp['permaticker'].n_unique():,} securities ever, members/day min "
      f"{sp_day['len'].min()} median {sp_day['len'].median():.0f}; {time.time() - t0:.0f}s")
