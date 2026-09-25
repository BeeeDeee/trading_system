"""M1.3 acceptance: 1000 symbols x 25 years of daily bars loads under 15s and 600 MB."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scout.data.parquet_source import ParquetCandleSource
from scout.domain.market import MARKET_COLUMNS

N_SYMBOLS = 1_000
N_YEARS = 25
SESSIONS_PER_YEAR = 252
MAX_LOAD_SECONDS = 15.0
MAX_PANEL_BYTES = 600 * 1024 * 1024
FIRST_YEAR = 2000


def _write_year(path: Path, year: int, asset_ids: list[str]) -> None:
    n_sym = len(asset_ids)
    n = SESSIONS_PER_YEAR * n_sym
    day = np.arange(SESSIONS_PER_YEAR, dtype=np.int32)
    base = np.datetime64(f"{year}-01-03T21:00:00")
    ts_one = base + day.astype("timedelta64[D]")
    ts = np.repeat(ts_one, n_sym)
    ids = np.tile(np.array(asset_ids, dtype=object), SESSIONS_PER_YEAR)
    session_index = np.repeat(
        day + (year - FIRST_YEAR) * SESSIONS_PER_YEAR, n_sym
    ).astype(np.int32)
    price = np.full(n, 25.0, dtype=np.float64)
    volume = np.full(n, 1_000.0, dtype=np.float64)
    table = pa.table(
        {
            "asset_id": pa.array(ids, type=pa.dictionary(pa.int32(), pa.string())),
            "symbol": pa.array(ids, type=pa.dictionary(pa.int32(), pa.string())),
            "ts": pa.array(ts).cast(pa.timestamp("ms", tz="UTC")),
            "session_index": session_index,
            "open": price,
            "high": price + 0.5,
            "low": price - 0.5,
            "close": price,
            "close_raw": price,
            "volume": volume,
            "dollar_volume": price * volume,
            "is_suspect": np.zeros(n, dtype=bool),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


@pytest.fixture(scope="module")
def panel_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("panel_1k_25y")
    asset_ids = [f"S{i:04d}" for i in range(N_SYMBOLS)]
    panel_dir = root / "panel" / "1d"
    for offset in range(N_YEARS):
        year = FIRST_YEAR + offset
        _write_year(panel_dir / f"{year}.parquet", year, asset_ids)
    return root


def test_load_1000_symbols_25_years_budget(panel_root: Path) -> None:
    source = ParquetCandleSource(panel_root)
    asset_ids = [f"S{i:04d}" for i in range(N_SYMBOLS)]
    start = datetime(FIRST_YEAR, 1, 1, tzinfo=UTC)
    end = datetime(FIRST_YEAR + N_YEARS - 1, 12, 31, 23, 59, 59, tzinfo=UTC)
    t0 = time.perf_counter()
    panel = source.load_panel(asset_ids, "1d", start, end)
    elapsed = time.perf_counter() - t0
    frame = panel.frame
    nbytes = int(frame.memory_usage(deep=True).sum())
    expected_rows = N_SYMBOLS * N_YEARS * SESSIONS_PER_YEAR
    assert len(frame) == expected_rows
    assert list(frame.columns) == list(MARKET_COLUMNS)
    assert elapsed < MAX_LOAD_SECONDS, f"load took {elapsed:.3f}s (limit {MAX_LOAD_SECONDS}s)"
    assert nbytes < MAX_PANEL_BYTES, f"panel used {nbytes} bytes (limit {MAX_PANEL_BYTES})"
