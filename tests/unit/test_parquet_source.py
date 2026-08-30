from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from scout.data.parquet_source import ParquetCandleSource
from scout.data.store import write_parquet_atomic
from scout.domain.market import MARKET_COLUMNS
from scout.utils.errors import ScoutDataError

TS_A = datetime(2015, 6, 1, 20, 0, tzinfo=UTC)
TS_B = datetime(2015, 6, 2, 20, 0, tzinfo=UTC)
TS_C = datetime(2016, 1, 4, 21, 0, tzinfo=UTC)
TS_GAP = datetime(2015, 6, 3, 20, 0, tzinfo=UTC)


def _row(
    ts: datetime,
    asset_id: str,
    *,
    symbol: str | None = None,
    close: float = 10.0,
    session_index: int = 0,
) -> dict[str, object]:
    label = symbol if symbol is not None else asset_id
    return {
        "asset_id": asset_id,
        "symbol": label,
        "ts": ts,
        "session_index": session_index,
        "open": close,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "close_raw": close,
        "volume": 1_000.0,
        "dollar_volume": close * 1_000.0,
        "is_suspect": False,
    }


def _write_years(processed_dir: Path, rows: list[dict[str, object]]) -> None:
    frame = pd.DataFrame(rows, columns=list(MARKET_COLUMNS))
    years = pd.to_datetime(frame["ts"], utc=True).dt.year
    for year in sorted(int(y) for y in years.unique()):
        part = frame.loc[years == year].reset_index(drop=True)
        write_parquet_atomic(processed_dir / "panel" / "1d" / f"{year}.parquet", part)


def test_load_panel_sorted_by_ts_asset_id(tmp_path: Path) -> None:
    # Written in the opposite of (ts, asset_id) order. Symbol order would
    # put ZZZ before AAA; asset_id order puts A1 before Z9.
    rows = [
        _row(TS_B, "Z9", symbol="AAA", close=2.0, session_index=1),
        _row(TS_B, "A1", symbol="ZZZ", close=1.0, session_index=1),
        _row(TS_A, "Z9", symbol="AAA", close=2.0, session_index=0),
        _row(TS_A, "A1", symbol="ZZZ", close=1.0, session_index=0),
    ]
    _write_years(tmp_path, rows)
    source = ParquetCandleSource(tmp_path)
    panel = source.load_panel(["Z9", "A1"], "1d", TS_A, TS_B)
    frame = panel.frame
    pairs = list(zip(frame["ts"].tolist(), frame["asset_id"].astype(str).tolist(), strict=True))
    assert pairs == [
        (pd.Timestamp(TS_A), "A1"),
        (pd.Timestamp(TS_A), "Z9"),
        (pd.Timestamp(TS_B), "A1"),
        (pd.Timestamp(TS_B), "Z9"),
    ]
    assert list(frame.columns) == list(MARKET_COLUMNS)
    assert str(frame["close"].dtype) == "float64"


def test_missing_symbols_omitted_without_raising(tmp_path: Path) -> None:
    _write_years(tmp_path, [_row(TS_A, "A1"), _row(TS_A, "B2")])
    source = ParquetCandleSource(tmp_path)
    panel = source.load_panel(["A1", "MISSING", "ALSO_MISSING"], "1d", TS_A, TS_A)
    assert panel.asset_ids == ("A1",)
    assert "MISSING" not in set(panel.frame["asset_id"].astype(str))


def test_load_panel_deterministic(tmp_path: Path) -> None:
    _write_years(
        tmp_path,
        [
            _row(TS_A, "B2", close=20.0),
            _row(TS_A, "A1", close=10.0),
            _row(TS_B, "A1", close=11.0, session_index=1),
        ],
    )
    source = ParquetCandleSource(tmp_path)
    first = source.load_panel(["B2", "A1"], "1d", TS_A, TS_B).frame
    second = source.load_panel(["A1", "B2"], "1d", TS_A, TS_B).frame
    pd.testing.assert_frame_equal(first, second)
    third = ParquetCandleSource(tmp_path).load_panel(["A1", "B2"], "1d", TS_A, TS_B).frame
    pd.testing.assert_frame_equal(first, third)


def test_closed_bars_only_end_inclusive(tmp_path: Path) -> None:
    _write_years(
        tmp_path,
        [
            _row(TS_A, "A1", session_index=0),
            _row(TS_B, "A1", session_index=1),
            _row(TS_GAP, "A1", session_index=2),
        ],
    )
    source = ParquetCandleSource(tmp_path)
    panel = source.load_panel(["A1"], "1d", TS_A, TS_B)
    timestamps = list(panel.timestamps)
    assert timestamps == [pd.Timestamp(TS_A), pd.Timestamp(TS_B)]
    assert pd.Timestamp(TS_GAP) not in timestamps


def test_start_inclusive_and_gaps_stay_gaps(tmp_path: Path) -> None:
    _write_years(
        tmp_path,
        [
            _row(TS_A, "A1", session_index=0),
            _row(TS_GAP, "A1", session_index=2),
        ],
    )
    source = ParquetCandleSource(tmp_path)
    panel = source.load_panel(["A1"], "1d", TS_A, TS_GAP)
    dates = {pd.Timestamp(ts).tz_convert("UTC").date() for ts in panel.frame["ts"]}
    assert TS_B.date() not in dates
    assert len(panel.frame) == 2


def test_year_files_outside_range_are_not_required(tmp_path: Path) -> None:
    _write_years(tmp_path, [_row(TS_A, "A1"), _row(TS_C, "A1", session_index=100)])
    source = ParquetCandleSource(tmp_path)
    panel = source.load_panel(["A1"], "1d", TS_C, TS_C)
    assert list(panel.timestamps) == [pd.Timestamp(TS_C)]


def test_empty_symbol_list_returns_empty_panel(tmp_path: Path) -> None:
    _write_years(tmp_path, [_row(TS_A, "A1")])
    source = ParquetCandleSource(tmp_path)
    panel = source.load_panel([], "1d", TS_A, TS_B)
    assert panel.frame.empty
    assert list(panel.frame.columns) == list(MARKET_COLUMNS)


def test_missing_timeframe_directory_raises(tmp_path: Path) -> None:
    source = ParquetCandleSource(tmp_path)
    with pytest.raises(ScoutDataError, match="processed panel not found"):
        source.load_panel(["A1"], "1d", TS_A, TS_B)


def test_naive_bounds_raise(tmp_path: Path) -> None:
    _write_years(tmp_path, [_row(TS_A, "A1")])
    source = ParquetCandleSource(tmp_path)
    naive = datetime(2015, 6, 1, 20, 0)  # noqa: DTZ001
    with pytest.raises(ScoutDataError, match="start must be timezone-aware UTC"):
        source.load_panel(["A1"], "1d", naive, TS_B)
    with pytest.raises(ScoutDataError, match="end must be timezone-aware UTC"):
        source.load_panel(["A1"], "1d", TS_A, naive)


def test_available_range(tmp_path: Path) -> None:
    _write_years(
        tmp_path,
        [
            _row(TS_A, "A1"),
            _row(TS_B, "A1", session_index=1),
            _row(TS_C, "B2"),
        ],
    )
    source = ParquetCandleSource(tmp_path)
    span = source.available_range("A1", "1d")
    assert span is not None
    assert span[0] == TS_A
    assert span[1] == TS_B
    assert source.available_range("MISSING", "1d") is None
    assert source.available_range("A1", "4h") is None
