"""Schemas of the normalized data layer (spec §4.4)."""

import polars as pl

# Row status in `bars`.
ACTIVE = "active"        # regular trading day
NO_OPEN = "no_open"      # close known, no usable open price -> not tradable, full day return in ret_co
HALTED = "halted"        # listed but no trading on this calendar day
DELISTED = "delisted"    # first session after the last trade; carries the terminal return
STATUSES = (ACTIVE, NO_OPEN, HALTED, DELISTED)

# Input price table in Sharadar SEP semantics: OHLC and volume split-adjusted, `closeadj`
# adjusted for splits, dividends and spinoffs, `closeunadj` unadjusted.
PRICES_SCHEMA = {
    "permaticker": pl.Int64,
    "date": pl.Date,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "closeadj": pl.Float64,
    "closeunadj": pl.Float64,
}

# One row per delisted security.
DELISTINGS_SCHEMA = {
    "permaticker": pl.Int64,
    "last_date": pl.Date,           # last trading day
    "kind": pl.Utf8,                # see qlab.data.normalize.DELISTING_KINDS
    "consideration_per_share": pl.Float64,  # acquisition: cash + stock value, unadjusted USD/share
}

# Normalized bars. Prices and volume are unadjusted; returns are total returns.
BARS_SCHEMA = {
    "permaticker": pl.Int64,
    "date": pl.Date,
    "open_u": pl.Float64,
    "high_u": pl.Float64,
    "low_u": pl.Float64,
    "close_u": pl.Float64,
    "volume_u": pl.Float64,
    "dollar_volume": pl.Float64,
    "ret_co": pl.Float64,           # previous close -> open (overnight)
    "ret_oc": pl.Float64,           # open -> close (intraday)
    "tradable": pl.Boolean,
    "status": pl.Utf8,
    "terminal_ret": pl.Float64,     # only on DELISTED rows
}


def validate(df: pl.DataFrame, schema: dict[str, pl.DataType], name: str) -> None:
    """Raise if `df` does not have exactly the columns and dtypes of `schema`."""
    actual = dict(df.schema)
    if actual != schema:
        missing = set(schema) - set(actual)
        extra = set(actual) - set(schema)
        wrong = {c: (actual[c], t) for c, t in schema.items() if c in actual and actual[c] != t}
        raise ValueError(f"{name}: schema mismatch; missing={missing} extra={extra} dtype={wrong}")
    keys = ["permaticker", "date"] if "date" in schema else ["permaticker"]
    if df.select(pl.struct(keys).is_duplicated().any()).item():
        raise ValueError(f"{name}: duplicate (permaticker, date) rows")
