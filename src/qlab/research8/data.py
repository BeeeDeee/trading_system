"""Parse raw Binance ZIPs into tidy polars frames. Raw files are never modified.

Format quirks handled here (all seen in the files):
- kline CSVs have a header in newer months and none in older ones;
- spot kline timestamps switched from milliseconds to microseconds in 2025;
- funding `calc_time` carries a few ms of jitter (e.g. ...00001): rounded to the nearest second.
"""

import io
import re
import zipfile
from pathlib import Path

import polars as pl

KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
              "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
STABLE_BASES = {"USDC", "FDUSD", "TUSD", "BUSD", "DAI", "USDP", "PAX", "UST", "USTC", "EUR", "GBP", "AEUR",
                "USDE", "USD1", "RLUSD", "PYUSD", "WBTC", "WBETH", "BETH", "STETH", "XUSD", "EURI", "SUSD", "USDS"}
PERP_RE = re.compile(r"^[A-Z0-9]+USDT$")
MULT_RE = re.compile(r"^(1000000|1000|1M)([A-Z0-9]+)$")


def _csv_from_zip(path: Path) -> bytes:
    with zipfile.ZipFile(path) as z:
        return z.read(z.namelist()[0])


def read_klines(sym_dir: Path) -> pl.DataFrame:
    """Daily klines of one symbol: date, open, high, low, close, quote_volume (sorted, deduplicated)."""
    frames = []
    for f in sorted(sym_dir.glob("*.zip")):
        raw = _csv_from_zip(f)
        if not raw.strip():
            continue
        has_header = not raw[:1].isdigit()
        df = pl.read_csv(io.BytesIO(raw), has_header=has_header, new_columns=KLINE_COLS,
                         schema_overrides={c: pl.Float64 for c in KLINE_COLS})
        frames.append(df.select("open_time", "open", "high", "low", "close", "quote_volume"))
    if not frames:
        return pl.DataFrame(schema={"date": pl.Date, "open": pl.Float64, "high": pl.Float64, "low": pl.Float64,
                                    "close": pl.Float64, "quote_volume": pl.Float64})
    df = pl.concat(frames)
    ms = pl.when(pl.col("open_time") > 1e14).then(pl.col("open_time") / 1000).otherwise(pl.col("open_time"))
    return (df.with_columns(pl.from_epoch(ms.cast(pl.Int64), time_unit="ms").dt.date().alias("date"))
              .drop("open_time").unique("date", keep="first").sort("date")
              .select("date", "open", "high", "low", "close", "quote_volume"))


def read_funding(sym_dir: Path) -> pl.DataFrame:
    """Funding events of one symbol: ts (UTC, second precision), rate."""
    frames = []
    for f in sorted(sym_dir.glob("*.zip")):
        raw = _csv_from_zip(f)
        if not raw.strip():
            continue
        df = pl.read_csv(io.BytesIO(raw), has_header=not raw[:1].isdigit(),
                         new_columns=["calc_time", "interval_h", "rate"],
                         schema_overrides={"calc_time": pl.Float64, "interval_h": pl.Float64, "rate": pl.Float64})
        frames.append(df.select("calc_time", "rate"))
    if not frames:
        return pl.DataFrame(schema={"ts": pl.Datetime("ms"), "rate": pl.Float64})
    df = pl.concat(frames)
    sec = (pl.col("calc_time") / 1000).round(0).cast(pl.Int64)
    return (df.with_columns(pl.from_epoch(sec, time_unit="s").cast(pl.Datetime("ms")).alias("ts"))
              .select("ts", "rate").unique("ts", keep="first").sort("ts"))


def spot_pair(perp: str, spot_symbols: set[str]) -> tuple[str, float] | None:
    """Spot symbol and price multiplier for a USDT-M perp (perp price = mult x spot price).

    The identical spot symbol wins (Binance spot lists e.g. 1000SATSUSDT itself); otherwise the
    1000/1000000/1M prefix is stripped. Stablecoins and wrapped tokens are excluded.
    """
    if not PERP_RE.match(perp):
        return None
    base = perp[:-4]
    m = MULT_RE.match(base)
    if base in STABLE_BASES or (m and m.group(2) in STABLE_BASES):
        return None
    if perp in spot_symbols:
        return perp, 1.0
    if m and m.group(2) + "USDT" in spot_symbols:
        return m.group(2) + "USDT", 1_000_000.0 if m.group(1) in ("1000000", "1M") else 1000.0
    return None
