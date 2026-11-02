"""Research 19: ETF panel (26 tickers of PLAN §3.2) from Sharadar funds, total return from closeadj.

Usage: python scripts/r19_build_etf_panel.py sharadar_YYYY-MM-DD
Output: data/derived/<snapshot>/panel_r19_etf/ (+ extra_cash_ret from FRED DTB3, tickers.json)
"""

import json
import sys
from pathlib import Path

import polars as pl

from qlab.data.normalize import normalize_prices
from qlab.data.panel import panel_from_bars, save_panel
from qlab.data.schema import DELISTINGS_SCHEMA, PRICES_SCHEMA
from qlab.engine.costs import cash_returns
from qlab.research19.setup import ALL_ETF, ETF_TREND, RSI_UNIVERSE, etf_panel_dir

snapshot = sys.argv[1]
src = Path("data/parquet") / snapshot
out = etf_panel_dir(snapshot)

calendar = (pl.read_parquet(src / "funds.parquet", columns=["ticker", "date"])
            .filter(pl.col("ticker") == "SPY")["date"].sort().to_list())
perm = (pl.read_parquet(src / "tickers.parquet")
        .filter((pl.col("table") == "SFP") & pl.col("ticker").is_in(ALL_ETF))
        .select("ticker", pl.col("permaticker").cast(pl.Int64)))
missing = sorted(set(ALL_ETF) - set(perm["ticker"].to_list()))
if missing:
    sys.exit(f"STOP: tickers missing in Sharadar SFP: {missing}")
prices = (pl.read_parquet(src / "funds.parquet").filter(pl.col("ticker").is_in(ALL_ETF))
          .join(perm, on="ticker").select([pl.col(c).cast(t) for c, t in PRICES_SCHEMA.items()])
          .sort("permaticker", "date"))
bars = normalize_prices(prices, pl.DataFrame(schema=DELISTINGS_SCHEMA), calendar)
panel = panel_from_bars(bars, calendar)
rf = pl.read_csv(sorted(Path("data/raw").glob("fred_*/DTB3.csv"))[-1], null_values=".",
                 schema_overrides={"DTB3": pl.Float64})
cash = cash_returns(panel.dates, rf["observation_date"].str.to_date().to_numpy(), rf["DTB3"].to_numpy())
save_panel(panel, out, extra={"cash_ret": cash})
by_perm = dict(zip(perm["permaticker"].to_list(), perm["ticker"].to_list()))
order = [by_perm[int(a)] for a in panel.assets]
(out / "tickers.json").write_text(json.dumps({"columns": order, "etf_trend": ETF_TREND,
                                              "rsi_universe": RSI_UNIVERSE}))
print(f"panel {panel.shape}, columns {order}")
print(bars.group_by("status").len())
