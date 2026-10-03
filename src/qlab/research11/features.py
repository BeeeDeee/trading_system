"""New point-in-time features of research 11 (prereg §4) on decision dates.

- Fundamentals (ART/ARQ): usable the trading day after the SEC filing date (`research4.fundamentals.asof_filings`),
  at most 400 days old.
- Insiders: filings with filing date in [t - 91 days, t).
- 13F (`holdings_ticker`): the quarter ending on D is usable from D + 46 days (45-day filing deadline);
  at most `H_MAX_AGE_DAYS` after that.
Each function returns {name: (D, N_stock) float32}, NaN = unknown.
"""

from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl

from qlab.research4.fundamentals import INSIDER_WINDOW_DAYS, asof_filings, scatter, with_year_ago

H_LAG_DAYS = 46
H_MAX_AGE_DAYS = 120
PREV_QUARTER_MAX_DAYS = 100
FUND = ["accruals", "net_issuance", "eps_surprise"]
INS = ["ins_sell", "ins_n_buyers", "ins_n_sellers", "ins_net_n"]
FLOW = ["io", "d_io", "d_holders", "d_breadth", "putcall"]


def ticker_cols(src: Path, stock_assets: np.ndarray) -> pl.DataFrame:
    tick = (pl.read_parquet(src / "tickers.parquet").filter(pl.col("table") == "SEP")
            .select("ticker", pl.col("permaticker").cast(pl.Int64))
            .filter(pl.col("permaticker").is_in(pl.Series(stock_assets).implode())))
    return tick.with_columns(col=pl.Series(np.searchsorted(stock_assets, tick["permaticker"].to_numpy()))
                             ).select("ticker", "col")


def marketcap(src: Path, tick: pl.DataFrame, dec_dates: list) -> pl.DataFrame:
    """date, col, mcap (USD) at the close of each decision date (Sharadar `daily`, millions -> USD)."""
    return (pl.scan_parquet(src / "daily.parquet")
            .filter(pl.col("date").is_in(pl.Series(dec_dates, dtype=pl.Date).implode()))
            .select("ticker", "date", (pl.col("marketcap") * 1e6).alias("mcap")).collect()
            .join(tick, on="ticker").select("date", "col", "mcap"))


def fundamental_features(src: Path, tick: pl.DataFrame, dec_dates: list, row_of: dict, shape: tuple,
                         mcap: np.ndarray) -> dict[str, np.ndarray]:
    art = (pl.scan_parquet(src / "fundamentals.parquet").filter(pl.col("dimension") == "ART")
           .select("ticker", "calendardate", pl.col("date").alias("filed"), "netinc", "ncfo", "assets", "sharesbas")
           .collect())
    art = with_year_ago(art, ["sharesbas"]).join(tick, on="ticker")
    art = art.with_columns(
        accruals=pl.when(pl.col("assets") > 0).then(-(pl.col("netinc") - pl.col("ncfo")) / pl.col("assets")),
        net_issuance=pl.when(pl.col("sharesbas_ya") > 0).then(-(pl.col("sharesbas") / pl.col("sharesbas_ya") - 1)))
    art = (art.sort("col", "filed", "calendardate").unique(["col", "filed"], keep="last", maintain_order=True)
           .select("col", "filed", "accruals", "net_issuance"))
    j = asof_filings(art, dec_dates)
    out = {k: scatter(j, k, row_of, shape) for k in ("accruals", "net_issuance")}
    arq = (pl.scan_parquet(src / "fundamentals.parquet").filter(pl.col("dimension") == "ARQ")
           .select("ticker", "calendardate", pl.col("date").alias("filed"), "netinc").collect())
    arq = with_year_ago(arq, ["netinc"]).join(tick, on="ticker")
    arq = arq.with_columns(dni=(pl.col("netinc") - pl.col("netinc_ya")).cast(pl.Float64))
    arq = (arq.sort("col", "filed", "calendardate").unique(["col", "filed"], keep="last", maintain_order=True)
           .select("col", "filed", "dni"))
    dni = scatter(asof_filings(arq, dec_dates), "dni", row_of, shape)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["eps_surprise"] = np.where(mcap > 0, dni / mcap, np.nan)
    return out


def insider_features(src: Path, tick: pl.DataFrame, dec_dates: list, row_of: dict, shape: tuple,
                     mcap: np.ndarray) -> dict[str, np.ndarray]:
    ins = (pl.scan_parquet(src / "insiders.parquet")
           .filter(pl.col("transactioncode").is_in(["P", "S"]))
           .select("ticker", pl.col("date").alias("filed"), "ownername", "transactioncode",
                   pl.coalesce(pl.col("transactionvalue"),
                               pl.col("transactionshares") * pl.col("transactionpricepershare"))
                   .cast(pl.Float64).alias("value"))
           .collect().join(tick, on="ticker"))
    sell, nb, ns = (np.full(shape, np.nan) for _ in range(3))
    for d, k in row_of.items():
        w = ins.filter((pl.col("filed") < d) & (pl.col("filed") >= d - timedelta(days=INSIDER_WINDOW_DAYS)))
        g = w.group_by("col").agg(
            sell=pl.col("value").filter((pl.col("transactioncode") == "S") & (pl.col("value") > 0)).sum(),
            nb=pl.col("ownername").filter(pl.col("transactioncode") == "P").n_unique(),
            ns=pl.col("ownername").filter(pl.col("transactioncode") == "S").n_unique())
        c = g["col"].to_numpy()
        sell[k, c], nb[k, c], ns[k, c] = g["sell"].to_numpy(), g["nb"].to_numpy(), g["ns"].to_numpy()
    # a stock with no filing in the window had no insider trades: 0, not unknown (insiders exist from 2008)
    first = np.datetime64(ins["filed"].min(), "D") + np.timedelta64(INSIDER_WINDOW_DAYS, "D")
    known = np.array([np.datetime64(d, "D") >= first for d in dec_dates])[:, None] & np.isfinite(mcap)
    sell, nb, ns = (np.where(known, np.nan_to_num(x), np.nan) for x in (sell, nb, ns))
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"ins_sell": np.where(mcap > 0, sell / mcap, np.nan), "ins_n_buyers": nb, "ins_n_sellers": ns,
                "ins_net_n": nb - ns}


def flow_table(src: Path, tick: pl.DataFrame) -> pl.DataFrame:
    """Per (col, quarter end D): flow features and `filed` = D + 45 days (usable from D + 46)."""
    ht = (pl.scan_parquet(src / "holdings_ticker.parquet")
          .select("date", "ticker", "shrholders", "shrvalue", "cllholders", "putholders").collect()
          .join(tick, on="ticker"))
    filers = (pl.scan_parquet(src / "holdings_investor.parquet").group_by("date")
              .agg(pl.len().alias("n_filers")).collect())
    qd = ht["date"].unique().sort()
    near = sorted({d - timedelta(days=i) for d in qd.to_list() for i in range(8)})   # quarter ends - 0..7 days
    mc = (pl.scan_parquet(src / "daily.parquet")
          .filter(pl.col("date").is_in(pl.Series(near, dtype=pl.Date).implode()))
          .select("ticker", "date", (pl.col("marketcap") * 1e6).alias("mcap")).collect()
          .join(tick, on="ticker").select("col", pl.col("date").alias("mdate"), "mcap").sort("mdate"))
    ht = ht.sort("date").join_asof(mc, left_on="date", right_on="mdate", by="col", strategy="backward",
                                    tolerance=timedelta(days=7), check_sortedness=False)
    ht = ht.join(filers, on="date", how="left").with_columns(
        io=pl.when(pl.col("mcap") > 0).then(pl.col("shrvalue") * 1e6 / pl.col("mcap")),
        breadth=pl.col("shrholders") / pl.col("n_filers"),
        putcall=pl.when(pl.col("shrholders") > 0)
        .then((pl.col("putholders") - pl.col("cllholders")) / pl.col("shrholders")))
    ht = ht.sort("col", "date").unique(["col", "date"], keep="last", maintain_order=True)
    prev = [pl.col(c).shift(1).over("col").alias(f"{c}_prev") for c in ("date", "io", "shrholders", "breadth")]
    ht = ht.with_columns(prev)
    fresh = (pl.col("date") - pl.col("date_prev")).dt.total_days() <= PREV_QUARTER_MAX_DAYS
    ht = ht.with_columns(
        d_io=pl.when(fresh).then(pl.col("io") - pl.col("io_prev")),
        d_holders=pl.when(fresh & (pl.col("shrholders") > 0) & (pl.col("shrholders_prev") > 0))
        .then((pl.col("shrholders") / pl.col("shrholders_prev")).log()),
        d_breadth=pl.when(fresh).then(pl.col("breadth") - pl.col("breadth_prev")),
        filed=pl.col("date") + timedelta(days=H_LAG_DAYS - 1))
    return ht.select("col", "filed", *FLOW)


def flow_features(src: Path, tick: pl.DataFrame, dec_dates: list, row_of: dict, shape: tuple
                  ) -> dict[str, np.ndarray]:
    j = asof_filings(flow_table(src, tick), dec_dates, max_age_days=H_MAX_AGE_DAYS + H_LAG_DAYS)
    return {k: scatter(j, k, row_of, shape) for k in FLOW}


def build(src: Path, stock_assets: np.ndarray, dec_dates: list) -> dict[str, np.ndarray]:
    row_of = {d: k for k, d in enumerate(dec_dates)}
    shape = (len(dec_dates), len(stock_assets))
    tick = ticker_cols(src, stock_assets)
    mcap = scatter(marketcap(src, tick, dec_dates), "mcap", row_of, shape)
    out = fundamental_features(src, tick, dec_dates, row_of, shape, mcap)
    out.update(insider_features(src, tick, dec_dates, row_of, shape, mcap))
    out.update(flow_features(src, tick, dec_dates, row_of, shape))
    return {k: np.where(np.isfinite(v), v, np.nan).astype(np.float32) for k, v in out.items()}
