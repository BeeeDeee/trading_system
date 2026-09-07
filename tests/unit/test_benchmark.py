"""Benchmark panel from processed SPY + VIX. 04-DATA_AND_UNIVERSE.md §6.3."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from scout.cli.__main__ import main
from scout.data.benchmark import (
    build_benchmark_frame,
    ticker_id,
    write_benchmark,
)
from scout.data.store import write_json_atomic, write_parquet_atomic
from scout.domain.market import BENCHMARK_COLUMNS, MARKET_COLUMNS, BenchmarkPanel
from scout.utils.errors import ScoutDataError

START = datetime(2015, 1, 5, 21, 0, tzinfo=UTC)


def _spy_rows(n: int) -> pd.DataFrame:
    ts = pd.DatetimeIndex([START + pd.Timedelta(days=i) for i in range(n)], tz="UTC")
    return pd.DataFrame(
        {
            "asset_id": ["118691"] * n,
            "symbol": ["SPY"] * n,
            "ts": ts,
            "session_index": list(range(n)),
            "open": [200.0 + i for i in range(n)],
            "high": [201.0 + i for i in range(n)],
            "low": [199.0 + i for i in range(n)],
            "close": [200.5 + i for i in range(n)],
            "close_raw": [200.5 + i for i in range(n)],
            "volume": [1_000_000.0] * n,
            "dollar_volume": [2.0e8] * n,
            "is_suspect": [False] * n,
        }
    )


def _vix_rows(n: int, *, skip_last: bool = False) -> pd.DataFrame:
    k = n - 1 if skip_last else n
    ts = pd.DatetimeIndex([START + pd.Timedelta(days=i) for i in range(k)], tz="UTC")
    return pd.DataFrame(
        {
            "asset_id": ["111630"] * k,
            "symbol": ["^VIX"] * k,
            "ts": ts,
            "session_index": list(range(k)),
            "open": [15.0] * k,
            "high": [16.0] * k,
            "low": [14.0] * k,
            "close": [12.0 + i for i in range(k)],
            "close_raw": [12.0 + i for i in range(k)],
            "volume": [0.0] * k,
            "dollar_volume": [0.0] * k,
            "is_suspect": [False] * k,
        }
    )


def test_columns_match_benchmark_contract() -> None:
    out = build_benchmark_frame(_spy_rows(3), _vix_rows(3))
    assert list(out.columns) == list(BENCHMARK_COLUMNS)
    BenchmarkPanel(out)
    assert out["close"].tolist() == [200.5, 201.5, 202.5]
    assert out["vix_close"].tolist() == [12.0, 13.0, 14.0]
    assert out["vix9d_close"].isna().all()
    assert out["vix3m_close"].isna().all()


def test_missing_vix_session_is_nan_not_filled() -> None:
    out = build_benchmark_frame(_spy_rows(3), _vix_rows(3, skip_last=True))
    assert len(out) == 3
    assert out["vix_close"].iloc[0] == pytest.approx(12.0)
    assert pd.isna(out["vix_close"].iloc[2])


def test_empty_vix_leaves_nan_column() -> None:
    out = build_benchmark_frame(_spy_rows(2), None)
    assert len(out) == 2
    assert out["vix_close"].isna().all()


def test_empty_spy_raises() -> None:
    with pytest.raises(ScoutDataError, match="SPY panel is empty"):
        build_benchmark_frame(_spy_rows(0), _vix_rows(0))


def test_ticker_id_lookup() -> None:
    tickers = pd.DataFrame(
        {"asset_id": ["118691", "111630"], "symbol": ["SPY", "^VIX"]}
    )
    assert ticker_id(tickers, "SPY") == "118691"
    with pytest.raises(ScoutDataError, match="QQQ"):
        ticker_id(tickers, "QQQ")


def test_cli_build_benchmark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import os

    for key in list(os.environ):
        if key.startswith("SCOUT_"):
            monkeypatch.delenv(key, raising=False)
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    spy = _spy_rows(4)
    vix = _vix_rows(4)
    panel = pd.concat([spy, vix], ignore_index=True)
    write_parquet_atomic(
        processed / "panel" / "1d" / "2015.parquet",
        panel.loc[:, list(MARKET_COLUMNS)],
    )
    tickers = pd.DataFrame(
        {
            "asset_id": ["118691", "111630"],
            "symbol": ["SPY", "^VIX"],
            "exchange": ["NYSEARCA", "INDEX"],
            "category": ["ETF", "IDX"],
            "sector": ["", ""],
            "is_etf": [True, False],
            "listed_date": [date(1993, 1, 29), date(1990, 1, 2)],
            "delisted_date": [pd.NaT, pd.NaT],
            "delist_reason": ["", ""],
        }
    )
    write_parquet_atomic(raw / "tickers.parquet", tickers)
    write_json_atomic(
        raw / "SNAPSHOT.json",
        {
            "data_snapshot_id": "test",
            "vendor": "fixture",
            "created_utc": "2016-06-15T22:14:03Z",
            "symbols": 2,
            "rows": 8,
            "date_min": "2015-01-05",
            "date_max": "2015-01-08",
            "actions_rows": 0,
            "earnings_rows": 0,
        },
    )
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        f"""
data:
  vendor: fixture
  raw_dir: {raw.as_posix()}
  processed_dir: {processed.as_posix()}
  snapshot_id_path: {(raw / "SNAPSHOT.json").as_posix()}
""",
        encoding="utf-8",
    )
    code = main(["build-benchmark", "--config", str(overlay)])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    dest = processed.parent / "reference" / "benchmark_1d.parquet"
    assert dest.is_file()
    loaded = pd.read_parquet(dest)
    assert list(loaded.columns) == list(BENCHMARK_COLUMNS)
    assert len(loaded) == 4
    assert loaded["vix_close"].notna().all()
    write_benchmark(dest, loaded)
