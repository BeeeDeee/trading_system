"""Sharadar-style prices -> normalized `bars` (spec §4.4, §4.5).

Returns are total returns derived from `closeadj`. The open is brought onto the same basis with the
factor of the same day (`open_adj = open * closeadj / close`), as documented by Sharadar. On an
ex-dividend day this attributes the dividend proportionally to the overnight and intraday parts; the
close-to-close return `(1 + ret_co) * (1 + ret_oc) - 1` is exact.
"""

import numpy as np
import polars as pl

from qlab.data.schema import (ACTIVE, BARS_SCHEMA, DELISTED, DELISTINGS_SCHEMA, HALTED, NO_OPEN,
                              PRICES_SCHEMA, validate)

ACQUISITION = "acquisition"             # merger/acquisition; cash_per_share if known
SPAC_LIQUIDATION = "spac_liquidation"   # trust returned to holders
BANKRUPTCY = "bankruptcy"               # bankruptcy or liquidation of an operating company
PERFORMANCE = "performance"             # regulatory / exchange-driven delisting
VOLUNTARY = "voluntary"
UNKNOWN = "unknown"
DELISTING_KINDS = (ACQUISITION, SPAC_LIQUIDATION, BANKRUPTCY, PERFORMANCE, VOLUNTARY, UNKNOWN)


def terminal_return(kind: str, last_close_u: float, cash_per_share: float | None = None,
                    performance_ret: float = -0.30, low_price: float = 1.0) -> float:
    """Return realized by a holder from the last close to the delisting payout (spec §4.5)."""
    if kind not in DELISTING_KINDS:
        raise ValueError(f"unknown delisting kind {kind!r}")
    if kind == ACQUISITION:
        if cash_per_share is None or not np.isfinite(cash_per_share):
            return 0.0
        return cash_per_share / last_close_u - 1.0
    if kind == BANKRUPTCY:
        return -1.0
    if kind == PERFORMANCE or (kind == UNKNOWN and last_close_u < low_price):
        return performance_ret
    return 0.0  # SPAC liquidation, voluntary, unknown above the low-price threshold


def normalize_prices(prices: pl.DataFrame, delistings: pl.DataFrame,
                     calendar: list | np.ndarray | pl.Series) -> pl.DataFrame:
    """Build `bars` for the securities in `prices` on the trading `calendar`.

    Every calendar day between a security's first and last price row gets a row (missing days become
    HALTED). A security whose prices end before the calendar does gets one extra DELISTED row on the
    next session carrying its terminal return; securities without a delisting record are treated as
    UNKNOWN delistings.
    """
    validate(prices, PRICES_SCHEMA, "prices")
    validate(delistings, DELISTINGS_SCHEMA, "delistings")
    cal = pl.DataFrame({"date": pl.Series(calendar, dtype=pl.Date)}).unique().sort("date")
    cal = cal.with_row_index("day")
    cal_end = cal["date"][-1]

    px = (prices.join(cal, on="date", how="inner")
          .with_columns(k=pl.col("closeunadj") / pl.col("close"),
                        f=pl.col("closeadj") / pl.col("close")))
    if px.height != prices.height:
        raise ValueError("prices contain dates outside the calendar")

    span = px.group_by("permaticker").agg(first=pl.col("day").min(), last=pl.col("day").max())
    delisted = (span.join(cal.rename({"day": "last", "date": "last_date"}), on="last")
                .filter(pl.col("last_date") < cal_end)
                .join(delistings, on=["permaticker", "last_date"], how="left")
                .with_columns(pl.col("kind").fill_null(UNKNOWN)))
    span = span.join(delisted.select("permaticker", end=pl.col("last") + 1), on="permaticker",
                     how="left").with_columns(end=pl.coalesce("end", "last"))

    grid = (span.join(cal, how="cross")
            .filter(pl.col("day").is_between(pl.col("first"), pl.col("end")))
            .select("permaticker", "date", "day", is_delisting_row=pl.col("day") > pl.col("last")))
    df = (grid.join(px.drop("day"), on=["permaticker", "date"], how="left")
          .sort("permaticker", "date"))

    has_bar = pl.col("close").is_not_null()
    has_open = has_bar & (pl.col("open") > 0) & (pl.col("volume") > 0)
    prev_adj = pl.col("closeadj").forward_fill().shift(1).over("permaticker")
    open_adj = pl.col("open") * pl.col("f")
    first_row = pl.col("day") == pl.col("day").min().over("permaticker")

    df = df.with_columns(
        open_u=pl.col("open") * pl.col("k"),
        high_u=pl.col("high") * pl.col("k"),
        low_u=pl.col("low") * pl.col("k"),
        close_u=pl.col("closeunadj"),
        volume_u=pl.col("volume") / pl.col("k"),
        dollar_volume=pl.col("close") * pl.col("volume"),
        ret_co=pl.when(first_row).then(0.0)
        .when(has_open).then(open_adj / prev_adj - 1.0)
        .when(has_bar).then(pl.col("closeadj") / prev_adj - 1.0)
        .otherwise(0.0),
        ret_oc=pl.when(has_open).then(pl.col("closeadj") / open_adj - 1.0).otherwise(0.0),
        tradable=has_open,
        status=pl.when(pl.col("is_delisting_row")).then(pl.lit(DELISTED))
        .when(has_open).then(pl.lit(ACTIVE))
        .when(has_bar).then(pl.lit(NO_OPEN))
        .otherwise(pl.lit(HALTED)),
    )

    last_close = (df.filter(has_bar).group_by("permaticker")
                  .agg(last_close_u=pl.col("closeunadj").last()))
    terms = delisted.join(last_close, on="permaticker").select(
        "permaticker",
        terminal_ret=pl.struct("kind", "last_close_u", "cash_per_share").map_elements(
            lambda r: terminal_return(r["kind"], r["last_close_u"], r["cash_per_share"]),
            return_dtype=pl.Float64))
    df = (df.join(terms, on="permaticker", how="left")
          .with_columns(
              terminal_ret=pl.when(pl.col("is_delisting_row")).then(pl.col("terminal_ret")),
              ret_co=pl.when(pl.col("is_delisting_row")).then(pl.col("terminal_ret"))
              .otherwise(pl.col("ret_co")),
              ret_oc=pl.when(pl.col("is_delisting_row")).then(0.0).otherwise(pl.col("ret_oc")),
              tradable=pl.when(pl.col("is_delisting_row")).then(False)
              .otherwise(pl.col("tradable"))))

    bars = df.select([pl.col(c).cast(t) for c, t in BARS_SCHEMA.items()])
    validate(bars, BARS_SCHEMA, "bars")
    return bars
