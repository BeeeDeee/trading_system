"""Triple-barrier labeling: conservative tie, re-anchoring, no-edge sanity."""

from __future__ import annotations

import math
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from scout.cli.__main__ import main
from scout.config.schema import LabelingConfig
from scout.data.store import write_parquet_atomic
from scout.domain.enums import Direction, MarketRegime, Regime, SetupOutcome, VolBucket
from scout.domain.features import FeatureRow
from scout.domain.market import BENCHMARK_COLUMNS, MARKET_COLUMNS
from scout.domain.setup import Setup
from scout.scoring.labeling import (
    LABEL_COLUMNS,
    empty_label_frame,
    label_record,
    records_to_frame,
    resolve_setup,
)
from scout.utils.stats import bootstrap_ci

REL = 1e-9
TS0 = datetime(2015, 1, 2, 21, 0, tzinfo=UTC)


def _ts(offset_days: int) -> datetime:
    return TS0 + timedelta(days=offset_days)


def _cfg() -> LabelingConfig:
    return LabelingConfig()


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
        [
            {"ts": ts, "open": o, "high": h, "low": lo, "close": c}
            for ts, o, h, lo, c in rows
        ]
    )


def _feature_row(**overrides: Any) -> FeatureRow:
    kwargs: dict[str, Any] = {
        "symbol": "AAPL",
        "ts": TS0,
        "close": 100.0,
        "atr_14": 2.0,
        "atr_pct": 0.02,
        "vol_20": 0.2,
        "vol_60": 0.25,
        "ema_fast": 101.0,
        "ema_slow": 99.0,
        "ema_spread_atr": 1.0,
        "slope_atr_20": 0.5,
        "efficiency_ratio_20": 0.8,
        "efficiency_ratio_60": 0.7,
        "atr_percentile_1y": 0.4,
        "regime": Regime.TREND_UP,
        "vol_bucket": VolBucket.MID,
        "donchian_high_20": 102.0,
        "donchian_low_20": 90.0,
        "donchian_high_55": 105.0,
        "donchian_low_55": 85.0,
        "keltner_upper": 105.0,
        "keltner_lower": 93.0,
        "dist_to_high_atr": 2.5,
        "dist_to_low_atr": 7.5,
        "mom_252_skip21": 0.15,
        "mom_126_skip21": 0.08,
        "mom_21": 0.02,
        "mom_252_xs_pct": 0.50,
        "vol_xs_pct": 0.4,
        "xs_population": 400,
        "gap_atr": 0.1,
        "gap_abs_mean_20": 0.2,
        "overnight_var_share_60": 0.3,
        "market_regime": MarketRegime.RISK_ON,
        "spy_dd_252": 0.05,
        "spy_above_ma": True,
        "vix_close": 18.0,
        "beta_bench_90": 1.1,
        "corr_bench_90": 0.6,
        "bars_available": 400,
        "bars_since_gap": 50,
        "is_warm": True,
    }
    kwargs.update(overrides)
    return FeatureRow(**kwargs)


def test_stop_hit_gives_minus_one_r() -> None:
    # Entry at 100, risk 5, stop 95. Next bar wicks to the stop, not the target.
    setup = _setup()
    forward = _forward(
        [
            (_ts(1), 100.0, 101.0, 99.0, 100.5),
            (_ts(2), 100.5, 101.0, 95.0, 96.0),
        ]
    )
    resolved = resolve_setup(setup, forward, _cfg(), vol_bucket=VolBucket.MID)
    assert resolved.outcome is SetupOutcome.STOP
    assert resolved.exit_price == pytest.approx(95.0, rel=REL)
    assert resolved.realised_r_gross == pytest.approx(-1.0, rel=REL)
    assert resolved.bars_held == 2


def test_target_hit_gives_rr() -> None:
    # Intended R/R is (110-100)/(100-95) = 2. Clean target touch, no stop.
    setup = _setup()
    forward = _forward(
        [
            (_ts(1), 100.0, 101.0, 99.0, 100.5),
            (_ts(2), 100.5, 110.0, 100.0, 109.0),
        ]
    )
    resolved = resolve_setup(setup, forward, _cfg(), vol_bucket=VolBucket.MID)
    assert resolved.outcome is SetupOutcome.TARGET
    assert resolved.exit_price == pytest.approx(110.0, rel=REL)
    assert resolved.realised_r_gross == pytest.approx(setup.reward_risk_ratio, rel=REL)
    assert resolved.realised_r_gross == pytest.approx(2.0, rel=REL)


def test_both_barriers_same_bar_gives_stop() -> None:
    setup = _setup()
    forward = _forward([(_ts(1), 100.0, 110.0, 95.0, 100.0)])
    resolved = resolve_setup(setup, forward, _cfg(), vol_bucket=VolBucket.MID)
    assert resolved.outcome is SetupOutcome.STOP
    assert resolved.exit_price == pytest.approx(95.0, rel=REL)
    assert resolved.realised_r_gross == pytest.approx(-1.0, rel=REL)
    assert resolved.bars_held == 1


def test_timeout_exits_at_close() -> None:
    setup = _setup(max_hold_bars=2)
    close_px = 102.0
    forward = _forward(
        [
            (_ts(1), 100.0, 101.0, 99.0, 100.5),
            (_ts(2), 100.5, 103.0, 99.5, close_px),
        ]
    )
    resolved = resolve_setup(setup, forward, _cfg(), vol_bucket=VolBucket.MID)
    assert resolved.outcome is SetupOutcome.TIME
    assert resolved.exit_price == pytest.approx(close_px, rel=REL)
    assert resolved.realised_r_gross == pytest.approx((close_px - 100.0) / 5.0, rel=REL)
    assert resolved.bars_held == 2
    assert resolved.resolution_ts == _ts(2)


def test_insufficient_forward_data_gives_open() -> None:
    setup = _setup(max_hold_bars=5)
    empty = resolve_setup(setup, _forward([]), _cfg(), vol_bucket=VolBucket.MID)
    assert empty.outcome is SetupOutcome.OPEN
    assert empty.resolution_ts is None
    assert empty.exit_price is None

    short = _forward(
        [
            (_ts(1), 100.0, 101.0, 99.0, 100.5),
            (_ts(2), 100.5, 102.0, 99.5, 101.0),
        ]
    )
    unresolved = resolve_setup(setup, short, _cfg(), vol_bucket=VolBucket.MID)
    assert unresolved.outcome is SetupOutcome.OPEN
    assert unresolved.resolution_ts is None
    assert unresolved.exit_price is None
    assert math.isnan(unresolved.realised_r_gross)


def test_stop_target_reanchored_on_entry() -> None:
    # Decision close 100, stop 95, target 110 (rr=2). Next open gaps to 115.
    # Re-anchored stop=110, target=125. A naive keep-the-levels fill at 110
    # would realise -1 R (or an inflated R/R if the original stop were kept).
    setup = _setup()
    forward = _forward(
        [
            (_ts(1), 115.0, 116.0, 114.0, 115.0),
            (_ts(2), 116.0, 125.0, 115.0, 124.0),
        ]
    )
    resolved = resolve_setup(setup, forward, _cfg(), vol_bucket=VolBucket.MID)
    assert resolved.entry_price == pytest.approx(115.0, rel=REL)
    assert resolved.outcome is SetupOutcome.TARGET
    assert resolved.exit_price == pytest.approx(125.0, rel=REL)
    assert resolved.realised_r_gross == pytest.approx(setup.reward_risk_ratio, rel=REL)
    assert abs(resolved.entry_price - 110.0) == pytest.approx(setup.risk_per_unit, rel=REL)
    assert abs(125.0 - resolved.entry_price) == pytest.approx(
        setup.reward_per_unit, rel=REL
    )


def test_mae_mfe_bounds() -> None:
    setup = _setup()
    # Goes 2 against, then hits the 2R target.
    forward = _forward(
        [
            (_ts(1), 100.0, 101.0, 98.0, 99.0),
            (_ts(2), 99.0, 110.0, 98.5, 109.0),
        ]
    )
    resolved = resolve_setup(setup, forward, _cfg(), vol_bucket=VolBucket.MID)
    assert resolved.mae_r <= 0.0 <= resolved.mfe_r
    if resolved.outcome is SetupOutcome.STOP:
        assert resolved.mae_r <= -1.0 + 1e-9
    if resolved.outcome is SetupOutcome.TARGET:
        assert resolved.mfe_r >= resolved.setup.reward_risk_ratio - 1e-9
    assert resolved.outcome is SetupOutcome.TARGET
    assert resolved.mae_r == pytest.approx(-0.4, rel=REL)  # (98-100)/5
    assert resolved.mfe_r == pytest.approx(2.0, rel=REL)  # (110-100)/5


def test_resolution_ts_is_exit_bar() -> None:
    setup = _setup()
    exit_ts = _ts(3)
    forward = _forward(
        [
            (_ts(1), 100.0, 101.0, 99.0, 100.0),
            (_ts(2), 100.0, 101.0, 99.0, 100.0),
            (exit_ts, 100.0, 101.0, 95.0, 96.0),
        ]
    )
    resolved = resolve_setup(setup, forward, _cfg(), vol_bucket=VolBucket.MID)
    assert resolved.outcome is SetupOutcome.STOP
    assert resolved.resolution_ts == exit_ts
    assert resolved.entry_ts == _ts(1)


def test_join_adv_with_overlapping_asset_timestamps() -> None:
    """pandas 2.2 merge_asof requires `on` globally sorted, even with `by`."""
    from scout.scoring.labeling import _join_adv

    ts_a = datetime(2015, 1, 5, 21, 0, tzinfo=UTC)
    ts_b = datetime(2015, 1, 6, 21, 0, tzinfo=UTC)
    labels = records_to_frame(
        [
            label_record(
                resolve_setup(
                    _setup(symbol="AAA", ts=ts_a),
                    _forward([(ts_b, 100.0, 101.0, 95.0, 96.0)]),
                    _cfg(),
                    vol_bucket=VolBucket.MID,
                ),
                asset_id="B",
                feature_row=_feature_row(symbol="BBB", ts=ts_a),
                adv_usd_60=float("nan"),
                had_earnings_in_window=False,
            ),
            label_record(
                resolve_setup(
                    _setup(symbol="AAA", ts=ts_a),
                    _forward([(ts_b, 100.0, 101.0, 95.0, 96.0)]),
                    _cfg(),
                    vol_bucket=VolBucket.MID,
                ),
                asset_id="A",
                feature_row=_feature_row(symbol="AAA", ts=ts_a),
                adv_usd_60=float("nan"),
                had_earnings_in_window=False,
            ),
        ]
    )
    snaps = pd.DataFrame(
        {
            "ts": [ts_a, ts_a],
            "asset_id": ["A", "B"],
            "adv_usd_60": [1.0e8, 2.0e8],
        }
    )
    out = _join_adv(labels, snaps)
    by_id = out.set_index("asset_id")["adv_usd_60"]
    assert float(by_id.loc["A"]) == pytest.approx(1.0e8)
    assert float(by_id.loc["B"]) == pytest.approx(2.0e8)


def test_label_parquet_schema() -> None:
    setup = _setup()
    forward = _forward([(_ts(1), 100.0, 101.0, 95.0, 96.0)])
    resolved = resolve_setup(setup, forward, _cfg(), vol_bucket=VolBucket.MID)
    record = label_record(
        resolved,
        asset_id="AAPL_ID",
        feature_row=_feature_row(),
        adv_usd_60=1.5e8,
        had_earnings_in_window=False,
    )
    frame = records_to_frame([record])
    assert tuple(frame.columns) == LABEL_COLUMNS
    empty = empty_label_frame()
    assert tuple(empty.columns) == LABEL_COLUMNS


def test_no_edge_series_gives_near_zero_mean_r() -> None:
    """Master sanity: driftless GBM + 2:1 barriers ⇒ mean R ~ 0, win_rate ~ 1/3."""
    rng = np.random.default_rng(20260827)
    n = 4000
    overnight = rng.normal(0.0, 0.008, n)
    intraday = rng.normal(0.0, 0.008, n)
    open_px = np.empty(n, dtype=np.float64)
    close_px = np.empty(n, dtype=np.float64)
    open_px[0] = 100.0
    for i in range(n):
        close_px[i] = open_px[i] * np.exp(intraday[i])
        if i + 1 < n:
            open_px[i + 1] = close_px[i] * np.exp(overnight[i])
    high_px = np.maximum(open_px, close_px) * np.exp(np.abs(rng.normal(0.0, 0.002, n)))
    low_px = np.minimum(open_px, close_px) * np.exp(-np.abs(rng.normal(0.0, 0.002, n)))
    timestamps = pd.bdate_range("2000-01-03", periods=n, freq="C").tz_localize("UTC")
    timestamps = timestamps + pd.Timedelta(hours=21)

    rr = 2.0
    max_hold = 40
    cfg = _cfg()
    realised: list[float] = []
    wins = 0
    for i in range(20, n - max_hold - 1, 8):
        ref = float(close_px[i])
        if not np.isfinite(ref) or ref <= 0:
            continue
        risk = 0.02 * ref
        setup = _setup(
            ts=timestamps[i].to_pydatetime(),
            reference_price=ref,
            stop_price=ref - risk,
            target_price=ref + rr * risk,
            max_hold_bars=max_hold,
        )
        sl = slice(i + 1, i + 1 + max_hold)
        forward = pd.DataFrame(
            {
                "ts": timestamps[sl],
                "open": open_px[sl],
                "high": high_px[sl],
                "low": low_px[sl],
                "close": close_px[sl],
            }
        )
        resolved = resolve_setup(setup, forward, cfg, vol_bucket=VolBucket.MID)
        if resolved.outcome is SetupOutcome.OPEN:
            continue
        r = resolved.realised_r_gross
        realised.append(r)
        if r > 0:
            wins += 1

    assert len(realised) >= 200
    mean_r = float(np.mean(realised))
    lo, hi = bootstrap_ci(realised, seed=20260827)
    win_rate = wins / len(realised)
    expected_win = 1.0 / (1.0 + rr)
    assert lo <= 0.0 <= hi, (
        f"mean_r={mean_r:.4f} bootstrap CI=({lo:.4f}, {hi:.4f}) does not contain 0; leak"
    )
    assert win_rate == pytest.approx(expected_win, abs=0.08), (
        f"win_rate={win_rate:.4f} vs 1/(1+rr)={expected_win:.4f}"
    )


@pytest.fixture
def isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith("SCOUT_"):
            monkeypatch.delenv(key, raising=False)


def test_cli_label_writes_schema(
    tmp_path: Path,
    isolated_env: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
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
    labels_dir = tmp_path / "labels"
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
  labels_dir: {labels_dir.as_posix()}
""",
        encoding="utf-8",
    )
    code = main(["label", "--config", str(overlay)])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    for sid in ("donchian_breakout_v1", "xsec_momentum_v1"):
        path = labels_dir / f"setups_{sid}.parquet"
        assert path.is_file(), captured.out
        loaded = pd.read_parquet(path)
        assert list(loaded.columns) == list(LABEL_COLUMNS)
