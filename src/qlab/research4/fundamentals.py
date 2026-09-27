"""Point-in-time fundamental scores on decision days (research 4 pre-registration §3).

- `daily` ratios (pe, pb, evebitda, ps, marketcap) are taken at the close of the decision day.
- `fundamentals` rows (ART, as reported) become usable on the trading day after their SEC filing
  date: a row filed on day d is visible at decision day t only if d < t. Rows older than
  `MAX_AGE_DAYS` at t are ignored (stale).
- Insider purchases (transaction code P) count if filed in [t - 91 days, t).
"""

from pathlib import Path

import numpy as np
import polars as pl

MAX_AGE_DAYS = 400
INSIDER_WINDOW_DAYS = 91
ART_FIELDS = ["gp", "assets", "roe", "de", "revenue", "eps", "ncfdiv"]


def scatter(df: pl.DataFrame, value: str, row_of: dict, shape: tuple[int, int]) -> np.ndarray:
    """(D, N) float matrix from a frame with columns date, col, `value` (NaN where missing)."""
    out = np.full(shape, np.nan)
    rows = np.array([row_of[d] for d in df["date"].to_list()], dtype=int)
    out[rows, df["col"].to_numpy()] = df[value].cast(pl.Float64).fill_null(np.nan).to_numpy()
    return out


def asof_filings(filings: pl.DataFrame, decision_dates: list, max_age_days: int = MAX_AGE_DAYS
                 ) -> pl.DataFrame:
    """Latest filing per `col` with filing date < decision date (and not stale).

    `filings` has columns col, filed (date) and value columns. Returns one row per (date, col).
    """
    right = (filings.with_columns(avail=pl.col("filed") + pl.duration(days=1))
             .sort("avail"))
    cols = right["col"].unique().sort()
    left = (pl.DataFrame({"date": decision_dates}).cast({"date": pl.Date})
            .join(pl.DataFrame({"col": cols}), how="cross").sort("date"))
    j = left.join_asof(right, left_on="date", right_on="avail", by="col", strategy="backward",
                       check_sortedness=False)  # both sides sorted by their key above
    return j.filter(pl.col("filed").is_not_null()
                    & ((pl.col("date") - pl.col("filed")).dt.total_days() <= max_age_days))


def with_year_ago(art: pl.DataFrame, fields: list[str]) -> pl.DataFrame:
    """Attach `<field>_ya` = value of the as-reported row for calendardate one year earlier."""
    prev = art.select("ticker", pl.col("calendardate").dt.offset_by("1y").alias("calendardate"),
                      *[pl.col(f).alias(f"{f}_ya") for f in fields])
    return art.join(prev, on=["ticker", "calendardate"], how="left")


def fundamental_scores(src: Path, stock_assets: np.ndarray, dates: np.ndarray, days: np.ndarray
                       ) -> dict[str, np.ndarray]:
    """(D, N_stock) scores for value / quality / investment / growth / dividend / size / insider."""
    dec_dates = [d.item() for d in np.asarray(dates[days], dtype="datetime64[D]")]
    row_of = {d: k for k, d in enumerate(dec_dates)}
    shape = (len(days), len(stock_assets))
    tick = (pl.read_parquet(src / "tickers.parquet")
            .filter(pl.col("table") == "SEP")
            .select("ticker", pl.col("permaticker").cast(pl.Int64)))
    tick = tick.filter(pl.col("permaticker").is_in(pl.Series(stock_assets).implode()))
    tick = tick.with_columns(col=pl.Series(np.searchsorted(stock_assets,
                                                           tick["permaticker"].to_numpy())))
    tick = tick.select("ticker", "col")

    daily = (pl.scan_parquet(src / "daily.parquet")
             .filter(pl.col("date").is_in(pl.Series(dec_dates, dtype=pl.Date).implode()))
             .select("ticker", "date", "pe", "pb", "evebitda", "ps", "marketcap")
             .collect().join(tick, on="ticker"))
    inv = lambda c: pl.when(pl.col(c) != 0).then(1.0 / pl.col(c))  # noqa: E731
    daily = daily.with_columns(ep=inv("pe"), bp=inv("pb"), ebitda_ev=inv("evebitda"),
                               sp=inv("ps"), mcap=pl.col("marketcap") * 1e6)
    out = {"value_ep": scatter(daily, "ep", row_of, shape),
           "value_bp": scatter(daily, "bp", row_of, shape),
           "value_ebitda_ev": scatter(daily, "ebitda_ev", row_of, shape),
           "value_sp": scatter(daily, "sp", row_of, shape)}
    mcap = scatter(daily, "mcap", row_of, shape)
    out["size_small"], out["size_large"] = -mcap, mcap.copy()
    del daily

    art = (pl.scan_parquet(src / "fundamentals.parquet")
           .filter(pl.col("dimension") == "ART")
           .select("ticker", "calendardate", pl.col("date").alias("filed"), *ART_FIELDS)
           .collect())
    art = with_year_ago(art, ["assets", "revenue", "eps"]).join(tick, on="ticker")
    art = art.with_columns(
        gpa=pl.when(pl.col("assets") > 0).then(pl.col("gp") / pl.col("assets")),
        lowlev=-pl.col("de"),
        lowag=pl.when(pl.col("assets_ya") > 0).then(-(pl.col("assets") / pl.col("assets_ya") - 1)),
        rev_g=pl.when(pl.col("revenue_ya") > 0).then(pl.col("revenue") / pl.col("revenue_ya") - 1),
        eps_g=(pl.col("eps") - pl.col("eps_ya"))
        / pl.max_horizontal(pl.col("eps_ya").abs(), pl.lit(0.10)),
        div=-pl.col("ncfdiv").cast(pl.Float64))
    # one filing per (col, filed): keep the latest calendar period
    art = (art.sort("col", "filed", "calendardate")
           .unique(["col", "filed"], keep="last", maintain_order=True)
           .select("col", "filed", "gpa", "roe", "lowlev", "lowag", "rev_g", "eps_g", "div"))
    j = asof_filings(art, dec_dates)
    for key, c in (("quality_gpa", "gpa"), ("quality_roe", "roe"), ("quality_lowlev", "lowlev"),
                   ("invest_lowag", "lowag"), ("growth_rev", "rev_g"), ("growth_eps", "eps_g")):
        out[key] = scatter(j, c, row_of, shape)
    div = scatter(j, "div", row_of, shape)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["dividend"] = np.where(mcap > 0, div / mcap, np.nan)
    del art, j

    ins = (pl.scan_parquet(src / "insiders.parquet")
           .filter(pl.col("transactioncode") == "P")
           .select("ticker", pl.col("date").alias("filed"),
                   pl.coalesce(pl.col("transactionvalue"),
                               pl.col("transactionshares") * pl.col("transactionpricepershare"))
                   .cast(pl.Float64).alias("value"))
           .filter(pl.col("value") > 0).collect().join(tick, on="ticker"))
    buys = np.full(shape, np.nan)
    for d, k in row_of.items():
        w = (ins.filter((pl.col("filed") < d)
                        & (pl.col("filed") >= d - np.timedelta64(INSIDER_WINDOW_DAYS, "D").item()))
             .group_by("col").agg(pl.col("value").sum()))
        buys[k, w["col"].to_numpy()] = w["value"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        out["insider"] = np.where(mcap > 0, buys / mcap, np.nan)
    return {k: np.where(np.isfinite(v), v, np.nan).astype(np.float32) for k, v in out.items()}
