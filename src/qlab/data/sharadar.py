"""Sharadar snapshot (Parquet) -> inputs of the normalization and `bars` on disk.

All heavy lifting runs in DuckDB with a memory limit; prices are normalized in chunks of securities
so the whole pipeline fits a small machine. See docs/DATA_FINDINGS.md for the data facts used here.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb
import polars as pl

from qlab.data.normalize import (ACQUISITION, BANKRUPTCY, PERFORMANCE, SPAC_LIQUIDATION, VOLUNTARY,
                                 normalize_prices)
from qlab.data.schema import DELISTINGS_SCHEMA, PRICES_SCHEMA

UNIVERSE_CATEGORIES = ("Domestic Common Stock", "Domestic Common Stock Primary Class")
SPAC_SIC = 6770
# Acquisition consideration further than this from the last close is treated as a data error
# (typically the acquirer's price is of a different share class); the payout falls back to 0 %.
MAX_CONSIDERATION_DEVIATION = 0.5


@dataclass(frozen=True)
class Snapshot:
    path: Path

    def table(self, name: str) -> str:
        return f"read_parquet('{self.path / f'{name}.parquet'}')"

    def connect(self, memory_limit: str = "1500MB", threads: int = 2) -> duckdb.DuckDBPyConnection:
        con = duckdb.connect()
        con.execute(f"SET memory_limit='{memory_limit}'; SET threads={threads}; "
                    "SET preserve_insertion_order=false")
        return con


def calendar(snap: Snapshot, con: duckdb.DuckDBPyConnection) -> list[date]:
    """Trading days = dates present in SEP (identical to SPY's dates, see the audit)."""
    return [d for (d,) in con.sql(f"SELECT DISTINCT date FROM {snap.table('stocks')} ORDER BY 1")
            .fetchall()]


def securities(snap: Snapshot, con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    cats = ", ".join(f"'{c}'" for c in UNIVERSE_CATEGORIES)
    return con.sql(f"""
        SELECT CAST(permaticker AS BIGINT) AS permaticker, ticker, category,
               TRY_CAST(siccode AS INTEGER) AS siccode
        FROM {snap.table('tickers')} WHERE "table" = 'SEP' AND category IN ({cats})
        ORDER BY permaticker""").pl()


def sic_history(snap: Snapshot, con: duckdb.DuckDBPyConnection, secs: pl.DataFrame) -> pl.DataFrame:
    """SIC code intervals [valid_from, valid_to) per security, from `sicchangefrom` actions.

    Before the first recorded change the SIC is that change's `from` value; after the last one it
    is the current SIC from tickers.
    """
    con.register("secs", secs.select("permaticker", "ticker", "siccode"))
    return con.sql(f"""
        WITH ch AS (
            SELECT s.permaticker, a.date, TRY_CAST(a.value AS INTEGER) AS sic_from
            FROM {snap.table('actions')} a JOIN secs s USING (ticker)
            WHERE a.action = 'sicchangefrom'),
        seq AS (
            SELECT permaticker, date AS valid_to, sic_from AS sic,
                   lag(date) OVER (PARTITION BY permaticker ORDER BY date) AS valid_from
            FROM ch)
        SELECT permaticker, coalesce(valid_from, DATE '1900-01-01') AS valid_from, valid_to, sic
        FROM seq
        UNION ALL
        SELECT s.permaticker, coalesce(max(ch.date), DATE '1900-01-01'), DATE '9999-12-31', s.siccode
        FROM secs s LEFT JOIN ch USING (permaticker) GROUP BY s.permaticker, s.siccode
        ORDER BY 1, 2""").pl()


def delistings(snap: Snapshot, con: duckdb.DuckDBPyConnection, secs: pl.DataFrame,
               sic: pl.DataFrame, end: date) -> pl.DataFrame:
    """One row per security whose prices end before `end`, classified by its actions on that day.

    Returns DELISTINGS_SCHEMA plus diagnostic columns (raw action, raw consideration).
    Securities with no delisting action are left out on purpose: normalization treats them as
    UNKNOWN.
    """
    con.register("secs", secs.select("permaticker", "ticker"))
    con.register("sic", sic)
    df = con.sql(f"""
        WITH last AS (
            SELECT s.permaticker, s.ticker, max(p.date) AS last_date,
                   arg_max(p.closeunadj, p.date) AS last_close
            FROM {snap.table('stocks')} p JOIN secs s USING (ticker)
            GROUP BY ALL HAVING max(p.date) < DATE '{end}'),
        acts AS (
            SELECT l.permaticker, l.last_date, l.last_close,
                   bool_or(a.action IN ('acquisitionby', 'mergerto')) AS acquired,
                   bool_or(a.action = 'bankruptcyliquidation') AS bankrupt,
                   bool_or(a.action = 'regulatorydelisting') AS regulatory,
                   bool_or(a.action = 'voluntarydelisting') AS voluntary,
                   bool_or(a.action = 'delisted') AS delisted,
                   sum(CASE WHEN a.action = 'acquisitioncash' THEN a.value END) AS cash,
                   sum(CASE WHEN a.action = 'acquisitionstock' THEN a.value END) AS ratio,
                   any_value(CASE WHEN a.action = 'acquisitionstock' THEN a.contraticker END) AS acq
            FROM last l JOIN {snap.table('actions')} a ON a.ticker = l.ticker AND a.date = l.last_date
            GROUP BY ALL),
        px AS (
            SELECT ticker, date, closeunadj FROM {snap.table('stocks')}
            UNION ALL SELECT ticker, date, closeunadj FROM {snap.table('funds')})
        SELECT a.*, p.closeunadj AS acq_price, h.sic
        FROM acts a
        LEFT JOIN px p ON p.ticker = a.acq AND p.date = a.last_date
        LEFT JOIN sic h ON h.permaticker = a.permaticker
             AND a.last_date >= h.valid_from AND a.last_date < h.valid_to
        WHERE a.acquired OR a.bankrupt OR a.regulatory OR a.voluntary OR a.delisted""").pl()

    consideration = (pl.col("cash").fill_null(0.0)
                     + (pl.col("ratio") * pl.col("acq_price")).fill_null(0.0))
    has_terms = pl.col("cash").is_not_null() | (pl.col("ratio").is_not_null()
                                                & pl.col("acq_price").is_not_null())
    plausible = ((consideration / pl.col("last_close") - 1.0).abs() <= MAX_CONSIDERATION_DEVIATION)
    spac = pl.col("sic") == SPAC_SIC
    return df.with_columns(
        raw_consideration=pl.when(has_terms).then(consideration),
        kind=pl.when(pl.col("acquired")).then(pl.lit(ACQUISITION))
        .when(pl.col("bankrupt") & spac).then(pl.lit(SPAC_LIQUIDATION))
        .when(pl.col("bankrupt")).then(pl.lit(BANKRUPTCY))
        .when(pl.col("regulatory")).then(pl.lit(PERFORMANCE))
        .when(pl.col("voluntary")).then(pl.lit(VOLUNTARY))
        .otherwise(pl.lit("unknown")),
    ).with_columns(
        consideration_per_share=pl.when(pl.col("acquired") & has_terms & plausible)
        .then(consideration),
    ).select(*DELISTINGS_SCHEMA, "last_close", "raw_consideration", "sic").sort("permaticker")


def prices(snap: Snapshot, con: duckdb.DuckDBPyConnection, secs: pl.DataFrame) -> pl.DataFrame:
    con.register("chunk", secs.select("permaticker", "ticker"))
    cols = ", ".join(f"p.{c}" for c in PRICES_SCHEMA if c not in ("permaticker",))
    df = con.sql(f"""SELECT c.permaticker, {cols} FROM {snap.table('stocks')} p
                     JOIN chunk c USING (ticker) ORDER BY 1, 2""").pl()
    return df.select([pl.col(c).cast(t) for c, t in PRICES_SCHEMA.items()])


def build_bars(snap: Snapshot, out_dir: Path, chunk_size: int = 1000) -> dict:
    """Normalize all universe-category securities into `out_dir/part-XXXX.parquet`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    con = snap.connect()
    cal = calendar(snap, con)
    secs = securities(snap, con)
    sic = sic_history(snap, con, secs)
    dl = delistings(snap, con, secs, sic, cal[-1])
    dl.write_parquet(out_dir.parent / "delistings.parquet")
    sic.write_parquet(out_dir.parent / "sic_history.parquet")
    n_rows = 0
    for k, start in enumerate(range(0, secs.height, chunk_size)):
        chunk = secs.slice(start, chunk_size)
        px = prices(snap, con, chunk)
        if px.is_empty():
            continue
        bars = normalize_prices(px, dl.select(*DELISTINGS_SCHEMA), cal)
        bars.write_parquet(out_dir / f"part-{k:04d}.parquet")
        n_rows += bars.height
    return {"securities": secs.height, "delistings": dl.height, "bars_rows": n_rows,
            "calendar_days": len(cal)}
