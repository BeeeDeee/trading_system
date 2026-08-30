"""Column contracts for data/raw/equity/. See docs/04-DATA_AND_UNIVERSE.md §4 and §6."""

from __future__ import annotations

OHLCV_COLUMNS: tuple[str, ...] = (
    "asset_id",
    "symbol",
    "session",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "dividend",
    "split_ratio",
)

ACTIONS_COLUMNS: tuple[str, ...] = (
    "asset_id",
    "ex_date",
    "action_type",
    "split_ratio",
    "cash_amount",
    "new_symbol",
)

TICKERS_COLUMNS: tuple[str, ...] = (
    "asset_id",
    "symbol",
    "exchange",
    "category",
    "sector",
    "is_etf",
    "listed_date",
    "delisted_date",
    "delist_reason",
)

EARNINGS_COLUMNS: tuple[str, ...] = (
    "asset_id",
    "earnings_date",
    "is_confirmed",
    "available_ts",
    "timing",
)

SNAPSHOT_KEYS: tuple[str, ...] = (
    "data_snapshot_id",
    "vendor",
    "created_utc",
    "symbols",
    "rows",
    "date_min",
    "date_max",
    "actions_rows",
    "earnings_rows",
)

CALENDAR_COLUMNS: tuple[str, ...] = (
    "session",
    "open_utc",
    "close_utc",
    "is_half_day",
    "session_index",
)

PANEL_COLUMNS: tuple[str, ...] = (
    "asset_id",
    "symbol",
    "ts",
    "session_index",
    "open",
    "high",
    "low",
    "close",
    "close_raw",
    "volume",
    "dollar_volume",
    "is_suspect",
)

UNIVERSE_SNAPSHOT_COLUMNS: tuple[str, ...] = (
    "ts",
    "asset_id",
    "symbol",
    "eligible",
    "reason",
    "adv_usd_60",
    "adv_rank",
    "spread_bps_est",
    "close_raw",
    "bars_available",
    "sector",
    "is_etf",
)

MIN_DELISTED_FRACTION = 0.15
