"""Optimized labeling: same barriers/earnings, checkpoints, CLI progress."""

from __future__ import annotations

import math
import os
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from scout.cli.__main__ import main
from scout.config.schema import LabelingConfig
from scout.data.store import read_json, write_parquet_atomic
from scout.domain.enums import Direction, Regime, SetupOutcome, VolBucket
from scout.domain.market import BENCHMARK_COLUMNS, MARKET_COLUMNS, EarningsEvent
from scout.domain.setup import Setup
from scout.gates.eligibility import SessionLookup, earnings_in_window
from scout.scoring.labeling import (
    earnings_in_window_fast,
    label_setups,
    prepare_earnings,
    resolve_setup,
    resolve_setup_arrays,
)

REL = 1e-9
TS0 = datetime(2015, 1, 2, 21, 0, tzinfo=UTC)


def _ts(offset_days: int) -> datetime:
    return TS0 + timedelta(days=offset_days)


def _setup(**overrides: Any) -> Setup:
    kwargs: dict[str, Any] = {
        "symbol": "AAPL",
        "ts": TS0,
        "strategy_id": "donchian_breakout_v1",
        "direction": Direction.LONG,
        "regime": Regime.TREND_UP,
        "reference_price": 100.0,
        "stop_price": 95.0,
        "target_price": 110.0,
        "max_hold_bars": 10,
        "trigger_note": "test",
    }
    kwargs.update(overrides)
    return Setup(**kwargs)


def _forward(rows: list[tuple[datetime, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"ts": ts, "open": o, "high": h, "low": lo, "close": c} for ts, o, h, lo, c in rows]
    )


def _arrays(
    forward: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if forward.empty:
        empty = np.empty(0, dtype=np.float64)
        return np.empty(0, dtype=object), empty, empty, empty, empty
    ts = pd.DatetimeIndex(pd.to_datetime(forward["ts"], utc=True))
    return (
        ts.to_pydatetime(),
        forward["open"].to_numpy(dtype=np.float64),
        forward["high"].to_numpy(dtype=np.float64),
        forward["low"].to_numpy(dtype=np.float64),
        forward["close"].to_numpy(dtype=np.float64),
    )


def _assert_same_resolved(left: Any, right: Any) -> None:
    assert left.outcome is right.outcome
    assert left.bars_held == right.bars_held
    if left.exit_price is None:
        assert right.exit_price is None
    else:
        assert right.exit_price == pytest.approx(left.exit_price, rel=REL)
    if math.isnan(left.realised_r_gross):
        assert math.isnan(right.realised_r_gross)
    else:
        assert right.realised_r_gross == pytest.approx(left.realised_r_gross, rel=REL)
    assert left.entry_price == pytest.approx(right.entry_price, rel=REL, nan_ok=True)
    assert left.resolution_ts == right.resolution_ts


def _fast_resolve(setup: Setup, forward: pd.DataFrame) -> Any:
    ts_py, open_, high, low, close = _arrays(forward)
    return resolve_setup_arrays(
        setup,
        ts_py=ts_py,
        open_=open_,
        high=high,
        low=low,
        close=close,
        cfg=LabelingConfig(),
        vol_bucket=VolBucket.MID,
    )


def test_fast_resolve_matches_stop() -> None:
    setup = _setup()
    forward = _forward(
        [
            (_ts(1), 100.0, 101.0, 99.0, 100.5),
            (_ts(2), 100.5, 101.0, 95.0, 96.0),
        ]
    )
    slow = resolve_setup(setup, forward, LabelingConfig(), vol_bucket=VolBucket.MID)
    _assert_same_resolved(slow, _fast_resolve(setup, forward))
    assert slow.outcome is SetupOutcome.STOP


def test_fast_resolve_matches_target() -> None:
    setup = _setup()
    forward = _forward(
        [
            (_ts(1), 100.0, 101.0, 99.0, 100.5),
            (_ts(2), 100.5, 110.0, 100.0, 109.0),
        ]
    )
    slow = resolve_setup(setup, forward, LabelingConfig(), vol_bucket=VolBucket.MID)
    _assert_same_resolved(slow, _fast_resolve(setup, forward))


def test_fast_resolve_matches_both_barriers_stop() -> None:
    setup = _setup()
    forward = _forward([(_ts(1), 100.0, 110.0, 95.0, 100.0)])
    slow = resolve_setup(setup, forward, LabelingConfig(), vol_bucket=VolBucket.MID)
    _assert_same_resolved(slow, _fast_resolve(setup, forward))
    assert slow.outcome is SetupOutcome.STOP


def test_fast_resolve_matches_timeout() -> None:
    setup = _setup(max_hold_bars=2)
    forward = _forward(
        [
            (_ts(1), 100.0, 101.0, 99.0, 100.5),
            (_ts(2), 100.5, 103.0, 99.5, 102.0),
        ]
    )
    slow = resolve_setup(setup, forward, LabelingConfig(), vol_bucket=VolBucket.MID)
    _assert_same_resolved(slow, _fast_resolve(setup, forward))


def test_fast_resolve_matches_open() -> None:
    setup = _setup(max_hold_bars=5)
    empty = _forward([])
    _assert_same_resolved(
        resolve_setup(setup, empty, LabelingConfig(), vol_bucket=VolBucket.MID),
        _fast_resolve(setup, empty),
    )
    short = _forward(
        [
            (_ts(1), 100.0, 101.0, 99.0, 100.5),
            (_ts(2), 100.5, 102.0, 99.5, 101.0),
        ]
    )
    _assert_same_resolved(
        resolve_setup(setup, short, LabelingConfig(), vol_bucket=VolBucket.MID),
        _fast_resolve(setup, short),
    )


def _earnings_calendar() -> pd.DataFrame:
    sessions = pd.bdate_range(date(2015, 1, 5), periods=50, freq="B")
    close = pd.DatetimeIndex(sessions.tz_localize("UTC")) + pd.Timedelta(hours=21)
    return pd.DataFrame(
        {
            "session": [ts.date() for ts in sessions],
            "close_utc": close,
            "session_index": list(range(50)),
        }
    )


def _ts_at(calendar: pd.DataFrame, index: int) -> datetime:
    value = calendar["close_utc"].iloc[index]
    ts = pd.Timestamp(value).to_pydatetime()
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts


def test_fast_earnings_matches_gates() -> None:
    calendar = _earnings_calendar()
    lookup = SessionLookup(calendar)
    t_idx = 15
    ts = _ts_at(calendar, t_idx)
    max_hold = 21

    def _event(index: int, available_index: int | None = 0) -> EarningsEvent:
        available = None if available_index is None else _ts_at(calendar, available_index)
        session_d = calendar["session"].iloc[index]
        if not hasattr(session_d, "year"):
            session_d = pd.Timestamp(session_d).date()
        return EarningsEvent(
            asset_id="AAPL",
            earnings_date=session_d,
            is_confirmed=True,
            available_ts=available,
            timing="AMC",
        )

    cases: list[tuple[list[EarningsEvent], bool, int]] = [
        ([_event(t_idx + max_hold)], False, max_hold),
        ([_event(t_idx + max_hold + 1)], False, max_hold),
        ([_event(t_idx)], False, max_hold),
        ([_event(t_idx + 2)], False, max_hold),
        ([], False, max_hold),
        ([_event(t_idx + 10)], False, 5),
        ([_event(t_idx + 10)], False, 21),
        ([_event(t_idx + 2, available_index=t_idx + 1)], False, max_hold),
    ]
    for events, is_etf, hold in cases:
        slow = earnings_in_window(
            ts=ts,
            max_hold_bars=hold,
            events=events,
            is_etf=is_etf,
            calendar=calendar,
        )
        fast = earnings_in_window_fast(
            ts=ts,
            max_hold_bars=hold,
            events=prepare_earnings(events, lookup),
            is_etf=is_etf,
            lookup=lookup,
        )
        assert fast is slow, (events, is_etf, hold, slow, fast)
    assert (
        earnings_in_window(
            ts=ts, max_hold_bars=max_hold, events=[], is_etf=True, calendar=calendar
        )
        is False
    )
    assert (
        earnings_in_window_fast(
            ts=ts,
            max_hold_bars=max_hold,
            events=(),
            is_etf=True,
            lookup=lookup,
        )
        is False
    )


def _sort_labels(frame: pd.DataFrame) -> pd.DataFrame:
    cols = ["strategy_id", "asset_id", "setup_ts", "direction"]
    return frame.sort_values(cols, kind="mergesort").reset_index(drop=True)


@pytest.fixture
def isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith("SCOUT_"):
            monkeypatch.delenv(key, raising=False)


def _write_mini_dataset(tmp_path: Path) -> Path:
    n = 15
    timestamps = pd.bdate_range("2015-01-05", periods=n, freq="C").tz_localize("UTC")
    timestamps = timestamps + pd.Timedelta(hours=21)
    rows: list[dict[str, object]] = []
    for i, ts in enumerate(timestamps):
        px = 50.0 + i
        rows.append(
            {
                "asset_id": "A1",
                "symbol": "AAA",
                "ts": ts,
                "session_index": i,
                "open": px,
                "high": px + 1.0,
                "low": px - 1.0,
                "close": px,
                "close_raw": px,
                "volume": 1_000.0,
                "dollar_volume": px * 1_000.0,
                "is_suspect": False,
            }
        )
        spy = 200.0 + i * 0.1
        rows.append(
            {
                "asset_id": "SPY",
                "symbol": "SPY",
                "ts": ts,
                "session_index": i,
                "open": spy,
                "high": spy + 1.0,
                "low": spy - 1.0,
                "close": spy,
                "close_raw": spy,
                "volume": 1_000_000.0,
                "dollar_volume": spy * 1_000_000.0,
                "is_suspect": False,
            }
        )
    panel = pd.DataFrame(rows, columns=list(MARKET_COLUMNS))
    bench_rows = []
    for i, ts in enumerate(timestamps):
        spy = 200.0 + i * 0.1
        bench_rows.append(
            {
                "ts": ts,
                "session_index": i,
                "open": spy,
                "high": spy + 1.0,
                "low": spy - 1.0,
                "close": spy,
                "vix_close": 18.0,
                "vix9d_close": 17.0,
                "vix3m_close": 19.0,
            }
        )
    bench = pd.DataFrame(bench_rows, columns=list(BENCHMARK_COLUMNS))
    snaps = pd.DataFrame(
        {
            "ts": [timestamps[0], timestamps[0]],
            "asset_id": ["A1", "SPY"],
            "symbol": ["AAA", "SPY"],
            "eligible": [True, True],
            "reason": ["", ""],
            "adv_usd_60": [1e8, 1e10],
            "adv_rank": [1, 2],
            "spread_bps_est": [1.0, 0.5],
            "close_raw": [50.0, 200.0],
            "bars_available": [n, n],
            "sector": ["TECH", "ETF"],
            "is_etf": [False, True],
        }
    )
    calendar = pd.DataFrame(
        {
            "session": [ts.date() for ts in timestamps],
            "open_utc": timestamps - pd.Timedelta(hours=6.5),
            "close_utc": timestamps,
            "is_half_day": [False] * n,
            "session_index": list(range(n)),
        }
    )
    tickers = pd.DataFrame(
        {
            "asset_id": ["A1", "SPY"],
            "symbol": ["AAA", "SPY"],
            "exchange": ["NASDAQ", "NYSEARCA"],
            "category": ["Domestic Common Stock", "ETF"],
            "sector": ["TECH", "ETF"],
            "is_etf": [False, True],
            "listed_date": [None, None],
            "delisted_date": [None, None],
            "delist_reason": [None, None],
        }
    )
    earnings = pd.DataFrame(
        {
            "asset_id": pd.Series(dtype="object"),
            "earnings_date": pd.Series(dtype="object"),
            "is_confirmed": pd.Series(dtype="bool"),
            "available_ts": pd.Series(dtype="datetime64[ns, UTC]"),
            "timing": pd.Series(dtype="object"),
        }
    )
    raw_dir = tmp_path / "raw"
    processed = tmp_path / "processed"
    reference = tmp_path / "reference"
    snaps_path = tmp_path / "universe" / "snapshots.parquet"
    candidates = tmp_path / "candidates.txt"
    raw_dir.mkdir()
    write_parquet_atomic(processed / "panel" / "1d" / "2015.parquet", panel)
    write_parquet_atomic(reference / "benchmark_1d.parquet", bench)
    write_parquet_atomic(reference / "calendar_xnys.parquet", calendar)
    write_parquet_atomic(snaps_path, snaps)
    write_parquet_atomic(raw_dir / "tickers.parquet", tickers)
    write_parquet_atomic(raw_dir / "earnings.parquet", earnings)
    candidates.write_text("A1,AAA\nSPY,SPY\n", encoding="utf-8")
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        f"""
period:
  start: 2015-01-05T21:00:00Z
  warmup_end: 2015-07-06T21:00:00Z
  end: 2015-08-01T21:00:00Z
data:
  vendor: fixture
  raw_dir: {raw_dir.as_posix()}
  processed_dir: {processed.as_posix()}
  snapshot_id_path: {(raw_dir / "SNAPSHOT.json").as_posix()}
universe:
  candidates_file: {candidates.as_posix()}
  snapshots_path: {snaps_path.as_posix()}
labeling:
  labels_dir: {(tmp_path / "labels").as_posix()}
""",
        encoding="utf-8",
    )
    return overlay


def test_cli_label_writes_progress(
    tmp_path: Path,
    isolated_env: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    overlay = _write_mini_dataset(tmp_path)
    labels_dir = tmp_path / "labels_opt"
    code = main(["label", "--config", str(overlay), "--labels-dir", str(labels_dir)])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    progress = labels_dir / "progress.json"
    assert progress.is_file(), captured.out
    payload = read_json(progress)
    assert payload["status"] == "done"
    assert payload["command"] == "label"
    status = main(["label-status", "--labels-dir", str(labels_dir)])
    status_out = capsys.readouterr()
    assert status == 0, status_out.err
    assert "status=done" in status_out.out
    for sid in ("donchian_breakout_v1", "xsec_momentum_v1"):
        assert (labels_dir / f"setups_{sid}.parquet").is_file()


def test_checkpoint_skip_does_not_duplicate(tmp_path: Path, isolated_env: None) -> None:
    overlay = _write_mini_dataset(tmp_path)
    from scout.cli import label_setups as label_cli
    from scout.config.loader import load_config
    from scout.data.ingest import load_tickers, read_candidate_pairs
    from scout.data.parquet_source import ParquetCandleSource
    from scout.features.engine import compute_features
    from scout.strategies.registry import build_strategies

    cfg = load_config(overlay)
    strategies = list(build_strategies(cfg.strategies))
    candidates = read_candidate_pairs(Path(cfg.universe.candidates_file))
    asset_ids = [str(x) for x in candidates["asset_id"].tolist()]
    panel = ParquetCandleSource(cfg.data.processed_dir).load_panel(
        asset_ids,
        cfg.data.decision_timeframe,
        cfg.period.start,
        cfg.period.end,
    )
    snapshots = label_cli._load_snapshots(cfg)
    benchmark = label_cli._load_benchmark(Path(cfg.data.processed_dir))
    features = compute_features(panel, benchmark, snapshots, cfg.features)
    calendar = label_cli._load_calendar(cfg)
    is_etf = label_cli._etf_map(load_tickers(Path(cfg.data.raw_dir)))
    earnings = label_cli._load_earnings(Path(cfg.data.raw_dir))
    kwargs = {
        "snapshots": snapshots,
        "earnings_by_asset": earnings,
        "is_etf": is_etf,
        "calendar": calendar,
    }
    full = label_setups(features, panel, strategies, cfg.labeling, **kwargs)
    first_id = str(panel.frame["asset_id"].astype(str).iloc[0])
    rest = label_setups(
        features,
        panel,
        strategies,
        cfg.labeling,
        skip_asset_ids={first_id},
        **kwargs,
    )
    only_first = label_setups(
        features,
        panel,
        strategies,
        cfg.labeling,
        skip_asset_ids=set(panel.frame["asset_id"].astype(str).unique()) - {first_id},
        **kwargs,
    )
    combined = pd.concat([only_first, rest], ignore_index=True)
    pd.testing.assert_frame_equal(
        _sort_labels(full),
        _sort_labels(combined),
        check_dtype=True,
        rtol=1e-12,
        atol=1e-12,
    )
