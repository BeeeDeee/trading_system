"""Build `data/reference/benchmark_1d.parquet` from processed SPY and VIX.

Schema: docs/04-DATA_AND_UNIVERSE.md §6.3 and domain BENCHMARK_COLUMNS.
VIX9D and VIX3M are not in the Sharadar equity ingest; they stay null until
a later source exists (sentiment vix_term is M4.3). Missing VIX is left NaN,
never filled.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

from scout.data.store import write_parquet_atomic
from scout.domain.market import BENCHMARK_COLUMNS, MarketPanel
from scout.utils.errors import ScoutDataError

VIX_SYMBOL: str = "^VIX"


def build_benchmark_frame(
    spy: pd.DataFrame,
    vix: pd.DataFrame | None,
) -> pd.DataFrame:
    """SPY session grid, left-joined to VIX close. No filling."""
    if spy.empty:
        raise ScoutDataError("SPY panel is empty; cannot build benchmark")
    needed = {"ts", "session_index", "open", "high", "low", "close"}
    missing = [c for c in needed if c not in spy.columns]
    if missing:
        raise ScoutDataError(f"SPY panel missing columns: {missing}")
    out = spy.loc[:, ["ts", "session_index", "open", "high", "low", "close"]].copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    before = len(out)
    if vix is None or vix.empty or "close" not in vix.columns or "ts" not in vix.columns:
        out["vix_close"] = float("nan")
    else:
        right = vix.loc[:, ["ts", "close"]].copy()
        right["ts"] = pd.to_datetime(right["ts"], utc=True)
        right = right.rename(columns={"close": "vix_close"})
        right = right.drop_duplicates(subset=["ts"], keep="last")
        merged = out.merge(right, on="ts", how="left", validate="many_to_one")
        if len(merged) != before:
            raise ScoutDataError("VIX merge duplicated SPY rows")
        out = merged
    out["vix9d_close"] = float("nan")
    out["vix3m_close"] = float("nan")
    out = out.loc[:, list(BENCHMARK_COLUMNS)]
    out["session_index"] = out["session_index"].astype("int32")
    for col in ("open", "high", "low", "close", "vix_close", "vix9d_close", "vix3m_close"):
        out[col] = out[col].astype("float64")
    return out.sort_values(["ts"], kind="mergesort").reset_index(drop=True)


def ticker_id(tickers: pd.DataFrame, symbol: str) -> str:
    hits = tickers.loc[tickers["symbol"].astype(str) == symbol, "asset_id"]
    if hits.empty:
        raise ScoutDataError(f"symbol {symbol!r} not in tickers")
    return str(hits.iloc[0])


def write_benchmark(path: Path, frame: pd.DataFrame) -> None:
    missing = [c for c in BENCHMARK_COLUMNS if c not in frame.columns]
    if missing:
        raise ScoutDataError(f"benchmark missing columns: {missing}")
    write_parquet_atomic(path, frame.loc[:, list(BENCHMARK_COLUMNS)])


def benchmark_path(processed_dir: Path) -> Path:
    return processed_dir.parent / "reference" / "benchmark_1d.parquet"


def panel_slice(panel: MarketPanel, asset_id: str) -> pd.DataFrame:
    return panel.frame.loc[panel.frame["asset_id"].astype(str) == asset_id].copy()
