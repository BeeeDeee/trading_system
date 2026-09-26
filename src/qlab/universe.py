"""Point-in-time universes (spec §5).

LIQ-N: on each day t, securities with an unadjusted close >= min_price, at least `min_history`
bars, not a SPAC (SIC 6770 valid at t), ranked by the median dollar volume of the last
`dv_window` bars; the top N. Everything uses bars up to and including t.

Features are computed per security (so bars can be processed in chunks), the ranking across
securities per day afterwards.
"""

from datetime import date

import polars as pl

from qlab.data.schema import DELISTED

SPAC_SIC = 6770


def eligibility_features(bars: pl.DataFrame, sic: pl.DataFrame | None = None,
                         dv_window: int = 63) -> pl.DataFrame:
    """Per (permaticker, date): trailing median dollar volume, bar count and SPAC flag."""
    df = (bars.filter(pl.col("status") != DELISTED)
          .select("permaticker", "date", "close_u", "dollar_volume")
          .sort("permaticker", "date")
          .with_columns(
              dv_median=pl.col("dollar_volume").fill_null(0.0)
              .rolling_median(dv_window, min_samples=dv_window).over("permaticker"),
              n_bars=pl.int_range(1, pl.len() + 1).over("permaticker")))
    if sic is None:
        return df.with_columns(spac=pl.lit(False))
    spac_iv = sic.filter(pl.col("sic") == SPAC_SIC).select("permaticker", "valid_from", "valid_to")
    flagged = (df.select("permaticker", "date").join(spac_iv, on="permaticker")
               .filter((pl.col("date") >= pl.col("valid_from")) & (pl.col("date") < pl.col("valid_to")))
               .select("permaticker", "date").unique().with_columns(spac=pl.lit(True)))
    return (df.join(flagged, on=["permaticker", "date"], how="left")
            .with_columns(pl.col("spac").fill_null(False)))


def rank_universe(features: pl.DataFrame | pl.LazyFrame, top_n: int = 1000,
                  min_price: float = 5.0, min_history: int = 252) -> pl.DataFrame:
    """(date, permaticker, dv_rank) for the top `top_n` eligible securities of each day."""
    return (features.lazy()
            .filter((pl.col("close_u") >= min_price) & (pl.col("n_bars") >= min_history)
                    & ~pl.col("spac") & pl.col("dv_median").is_not_null()
                    & (pl.col("dv_median") > 0))
            .with_columns(dv_rank=pl.col("dv_median").rank("ordinal", descending=True).over("date"))
            .filter(pl.col("dv_rank") <= top_n)
            .select("date", "permaticker", pl.col("dv_rank").cast(pl.Int32))
            .sort("date", "dv_rank")
            .collect())


def sp500_intervals(sp500: pl.DataFrame) -> pl.DataFrame:
    """Membership intervals [start, end) per ticker from Sharadar's sp500 table.

    Walks the added/removed events backwards from the current constituents; `start`/`end` null
    means open-ended. Tickers are Sharadar tickers (map to permaticker via tickers).
    """
    current = set(sp500.filter(pl.col("action") == "current")["ticker"])
    events = (sp500.filter(pl.col("action").is_in(["added", "removed"]))
              .sort("date", descending=True))
    open_end: dict[str, date | None] = {tk: None for tk in current}
    rows = []
    for d, action, tk in events.select("date", "action", "ticker").iter_rows():
        if action == "added":
            rows.append((tk, d, open_end.pop(tk, None)))
        else:
            open_end[tk] = d
    rows += [(tk, None, end) for tk, end in open_end.items()]
    return pl.DataFrame(rows, schema={"ticker": pl.Utf8, "start": pl.Date, "end": pl.Date},
                        orient="row")
