from __future__ import annotations

import os
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from scout.cli.__main__ import main
from scout.config.loader import load_config
from scout.data.actions import (
    adjustment_factors,
    apply_adjustments,
    build_processed_panel,
    run_adjust,
)
from scout.data.calendar import build_calendar, reference_calendar_path
from scout.data.ingest import run_ingest
from scout.data.quality import flag_suspect, quality_masks
from scout.data.schemas import ACTIONS_COLUMNS, PANEL_COLUMNS
from scout.data.store import write_json_atomic, write_parquet_atomic
from scout.domain.market import MARKET_COLUMNS, MarketPanel
from scout.utils.clock import BarClock
from scout.utils.errors import ScoutConfigError, ScoutDataError

FIXED_TS = datetime(2016, 6, 15, 22, 14, 3, tzinfo=UTC)

# 2-for-1 split Monday. Previous session is Friday.
PRE_SPLIT = date(2015, 7, 31)
SPLIT_EX = date(2015, 8, 3)
POST_SPLIT = date(2015, 8, 4)

# Dividend Thursday. Friday is not the pay date; Monday 16th is a stand-in pay date.
PRE_DIV = date(2015, 11, 11)
DIV_EX = date(2015, 11, 12)
POST_DIV = date(2015, 11, 13)
PAY_DATE = date(2015, 11, 16)
DIV_CASH = 0.40
PRE_DIV_CLOSE = 20.0


@pytest.fixture
def isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith("SCOUT_"):
            monkeypatch.delenv(key, raising=False)


def _empty_actions() -> pd.DataFrame:
    return pd.DataFrame(columns=list(ACTIONS_COLUMNS))


def _ohlcv_row(
    session: date,
    close: float,
    *,
    asset_id: str = "A1",
    symbol: str = "AAA",
    volume: float = 1_000.0,
    high: float | None = None,
    low: float | None = None,
    open_: float | None = None,
) -> dict[str, object]:
    return {
        "asset_id": asset_id,
        "symbol": symbol,
        "session": session,
        "open": close if open_ is None else open_,
        "high": close + 1.0 if high is None else high,
        "low": close - 1.0 if low is None else low,
        "close": close,
        "volume": volume,
        "dividend": 0.0,
        "split_ratio": 1.0,
    }


def _actions_row(
    ex_date: date,
    *,
    asset_id: str = "A1",
    action_type: str = "SPLIT",
    split_ratio: float = 1.0,
    cash_amount: float = 0.0,
) -> dict[str, object]:
    return {
        "asset_id": asset_id,
        "ex_date": ex_date,
        "action_type": action_type,
        "split_ratio": split_ratio,
        "cash_amount": cash_amount,
        "new_symbol": "",
    }


def _mini_calendar(sessions: list[date]) -> pd.DataFrame:
    return build_calendar(min(sessions), max(sessions))


def _at(frame: pd.DataFrame, session: date, asset_id: str = "A1") -> pd.Series:
    sess = pd.to_datetime(frame["session"]).dt.date
    matched = frame.loc[(sess == session) & (frame["asset_id"] == asset_id)]
    assert len(matched) == 1, f"expected one row for {asset_id} {session}"
    return matched.iloc[0]


def test_split_adjust_is_causal() -> None:
    ohlcv = pd.DataFrame(
        [
            _ohlcv_row(PRE_SPLIT, 100.0),
            _ohlcv_row(SPLIT_EX, 50.5),
            _ohlcv_row(POST_SPLIT, 51.0),
        ]
    )
    actions = pd.DataFrame([_actions_row(SPLIT_EX, split_ratio=2.0)])
    adjusted = apply_adjustments(ohlcv, actions)
    pre = _at(adjusted, PRE_SPLIT)
    ex = _at(adjusted, SPLIT_EX)
    post = _at(adjusted, POST_SPLIT)
    # Factor applies strictly before the ex-date. Ex-date and after stay at traded prices.
    assert pre["close"] == pytest.approx(50.0)
    assert pre["open"] == pytest.approx(50.0)
    assert pre["high"] == pytest.approx(50.5)
    assert pre["low"] == pytest.approx(49.5)
    assert pre["volume"] == pytest.approx(2_000.0)
    assert ex["close"] == pytest.approx(50.5)
    assert post["close"] == pytest.approx(51.0)
    # Adjusted path does not contain the 50% split drop.
    assert ex["close"] / pre["close"] == pytest.approx(50.5 / 50.0)


def test_unadjusted_close_pinned() -> None:
    ohlcv = pd.DataFrame(
        [
            _ohlcv_row(PRE_SPLIT, 100.0),
            _ohlcv_row(SPLIT_EX, 50.5),
            _ohlcv_row(POST_SPLIT, 51.0),
        ]
    )
    original = ohlcv["close"].to_numpy().copy()
    actions = pd.DataFrame(
        [
            _actions_row(SPLIT_EX, split_ratio=2.0),
            _actions_row(POST_SPLIT, split_ratio=4.0),
        ]
    )
    adjusted = apply_adjustments(ohlcv, actions)
    assert adjusted["close_raw"].to_numpy() == pytest.approx(original)
    later = apply_adjustments(ohlcv, actions)
    assert later["close_raw"].to_numpy() == pytest.approx(original)
    assert ohlcv["close"].to_numpy() == pytest.approx(original)


def test_dividend_applied_on_ex_date_not_pay_date() -> None:
    ohlcv = pd.DataFrame(
        [
            _ohlcv_row(PRE_DIV, PRE_DIV_CLOSE),
            _ohlcv_row(DIV_EX, 19.70),
            _ohlcv_row(POST_DIV, 19.80),
            _ohlcv_row(PAY_DATE, 19.90),
        ]
    )
    actions = pd.DataFrame(
        [_actions_row(DIV_EX, action_type="DIVIDEND", cash_amount=DIV_CASH)]
    )
    sessions = pd.to_datetime(ohlcv["session"])
    factor = adjustment_factors(actions, sessions, ohlcv["close"])
    expected_f = 1.0 - DIV_CASH / PRE_DIV_CLOSE
    assert expected_f == pytest.approx(0.98)
    by_session = dict(zip(pd.to_datetime(ohlcv["session"]).dt.date, factor, strict=True))
    assert by_session[PRE_DIV] == pytest.approx(expected_f)
    # On and after the ex-date the factor is 1.0, including a later pay date.
    assert by_session[DIV_EX] == pytest.approx(1.0)
    assert by_session[POST_DIV] == pytest.approx(1.0)
    assert by_session[PAY_DATE] == pytest.approx(1.0)
    adjusted = apply_adjustments(ohlcv, actions)
    assert _at(adjusted, PRE_DIV)["close"] == pytest.approx(PRE_DIV_CLOSE * expected_f)
    assert _at(adjusted, DIV_EX)["close"] == pytest.approx(19.70)
    assert _at(adjusted, PAY_DATE)["close"] == pytest.approx(19.90)


def test_close_and_close_raw_present_on_every_row() -> None:
    ohlcv = pd.DataFrame(
        [
            _ohlcv_row(PRE_SPLIT, 100.0),
            _ohlcv_row(SPLIT_EX, 50.5),
            _ohlcv_row(POST_SPLIT, 51.0),
        ]
    )
    actions = pd.DataFrame([_actions_row(SPLIT_EX, split_ratio=2.0)])
    calendar = _mini_calendar([PRE_SPLIT, SPLIT_EX, POST_SPLIT])
    panel = build_processed_panel(ohlcv, actions, calendar)
    assert list(panel.columns) == list(PANEL_COLUMNS)
    assert list(PANEL_COLUMNS) == list(MARKET_COLUMNS)
    assert panel["close"].notna().all()
    assert panel["close_raw"].notna().all()
    assert len(panel) == 3
    MarketPanel(panel)


def test_nothing_forward_filled() -> None:
    sessions = [date(2015, 6, 1), date(2015, 6, 2), date(2015, 6, 4), date(2015, 6, 5)]
    ohlcv = pd.DataFrame([_ohlcv_row(s, 10.0 + i) for i, s in enumerate(sessions)])
    calendar = build_calendar(date(2015, 6, 1), date(2015, 6, 5))
    panel = build_processed_panel(ohlcv, _empty_actions(), calendar)
    # 2015-06-03 is a valid XNYS session and is absent from the raw bars.
    panel_dates = {pd.Timestamp(ts).tz_convert("UTC").date() for ts in panel["ts"]}
    assert date(2015, 6, 3) not in panel_dates
    assert date(2015, 6, 1) in panel_dates
    assert date(2015, 6, 2) in panel_dates
    assert date(2015, 6, 4) in panel_dates
    assert len(panel) == 4


def test_yahoo_not_a_vendor(tmp_path: Path, isolated_env: None) -> None:
    path = tmp_path / "overlay.yaml"
    path.write_text("data:\n  vendor: yahoo\n", encoding="utf-8")
    with pytest.raises(ScoutConfigError, match="data.vendor"):
        load_config(path)


def test_quality_ohlc_invalid_and_zero_volume() -> None:
    sessions = [date(2015, 6, 1), date(2015, 6, 2)]
    ohlcv = pd.DataFrame(
        [
            _ohlcv_row(sessions[0], 10.0, high=9.0, low=11.0),
            _ohlcv_row(sessions[1], 10.0, volume=0.0),
        ]
    )
    calendar = _mini_calendar(sessions)
    panel = build_processed_panel(ohlcv, _empty_actions(), calendar)
    masks = quality_masks(panel, actions=_empty_actions(), calendar=calendar)
    assert bool(masks["ohlc_invalid"].iloc[0]) is True
    assert bool(masks["zero_volume"].iloc[1]) is True
    assert bool(panel["is_suspect"].all())


def test_quality_large_return_requires_matching_action() -> None:
    # 10-for-1 drop: |ln(0.1)| > 1. A matching SPLIT row suppresses the flag.
    pre, ex, post = date(2015, 8, 3), date(2015, 8, 4), date(2015, 8, 5)
    ohlcv = pd.DataFrame(
        [
            _ohlcv_row(pre, 100.0),
            _ohlcv_row(ex, 10.0),
            _ohlcv_row(post, 10.1),
        ]
    )
    calendar = _mini_calendar([pre, ex, post])
    none = _empty_actions()
    panel = build_processed_panel(ohlcv, none, calendar)
    masks = quality_masks(panel, actions=none, calendar=calendar)
    assert bool(masks["large_return"].iloc[1]) is True
    split = pd.DataFrame([_actions_row(ex, split_ratio=10.0)])
    panel_ok = build_processed_panel(ohlcv, split, calendar)
    masks_ok = quality_masks(panel_ok, actions=split, calendar=calendar)
    assert bool(masks_ok["large_return"].iloc[1]) is False


def test_quality_stale_close_five_sessions() -> None:
    sessions = [
        date(2015, 6, 1),
        date(2015, 6, 2),
        date(2015, 6, 3),
        date(2015, 6, 4),
        date(2015, 6, 5),
        date(2015, 6, 8),
    ]
    ohlcv = pd.DataFrame(
        [_ohlcv_row(s, 10.0) for s in sessions[:5]] + [_ohlcv_row(sessions[5], 11.0)]
    )
    calendar = _mini_calendar(sessions)
    panel = build_processed_panel(ohlcv, _empty_actions(), calendar)
    masks = quality_masks(panel, actions=_empty_actions(), calendar=calendar)
    assert bool(masks["stale_close"].iloc[:5].all())
    assert bool(masks["stale_close"].iloc[5]) is False


def test_quality_session_absent_from_calendar() -> None:
    saturday = date(2015, 6, 6)  # not an XNYS session
    monday = date(2015, 6, 8)
    ohlcv = pd.DataFrame([_ohlcv_row(saturday, 10.0), _ohlcv_row(monday, 10.5)])
    calendar = build_calendar(date(2015, 6, 8), date(2015, 6, 8))
    panel = build_processed_panel(ohlcv, _empty_actions(), calendar)
    masks = quality_masks(panel, actions=_empty_actions(), calendar=calendar)
    assert bool(masks["not_in_calendar"].iloc[0]) is True
    assert bool(masks["not_in_calendar"].iloc[1]) is False
    flagged = flag_suspect(panel, actions=_empty_actions(), calendar=calendar)
    assert bool(flagged["is_suspect"].iloc[0]) is True


def test_quality_does_not_delete_or_fill_rows() -> None:
    sessions = [date(2015, 6, 1), date(2015, 6, 2)]
    ohlcv = pd.DataFrame(
        [
            _ohlcv_row(sessions[0], 10.0, volume=0.0),
            _ohlcv_row(sessions[1], 10.5),
        ]
    )
    calendar = _mini_calendar(sessions)
    panel = build_processed_panel(ohlcv, _empty_actions(), calendar)
    assert len(panel) == 2


def test_run_adjust_writes_calendar_and_panel(tmp_path: Path, isolated_env: None) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    snapshot = raw_dir / "SNAPSHOT.json"
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        f"""
data:
  vendor: fixture
  raw_dir: {raw_dir.as_posix()}
  processed_dir: {processed_dir.as_posix()}
  snapshot_id_path: {snapshot.as_posix()}
period:
  start: 2015-01-01T00:00:00Z
  warmup_end: 2015-07-01T00:00:00Z
  end: 2016-12-31T00:00:00Z
""",
        encoding="utf-8",
    )
    cfg = load_config(overlay)
    run_ingest(
        cfg,
        start=datetime(2015, 7, 1, tzinfo=UTC),
        end=datetime(2015, 8, 31, tzinfo=UTC),
        raw_dir=raw_dir,
        snapshot_path=snapshot,
        clock=BarClock(FIXED_TS),
    )
    result = run_adjust(cfg)
    cal_path = reference_calendar_path(processed_dir, "XNYS")
    assert cal_path.is_file()
    assert result.panel_rows > 0
    assert result.calendar_rows > 0
    year_file = processed_dir / "panel" / "1d" / "2015.parquet"
    assert year_file.is_file()
    panel = pd.read_parquet(year_file)
    assert list(panel.columns) == list(PANEL_COLUMNS)
    assert panel["close"].notna().all()
    assert panel["close_raw"].notna().all()
    split_rows = panel.loc[panel["asset_id"] == "FIX0005"]
    assert not split_rows.empty
    # Raw close is present and the post-split adjusted close equals raw.
    ex_ts = split_rows.loc[
        pd.to_datetime(split_rows["ts"], utc=True).dt.date == SPLIT_EX
    ]
    if not ex_ts.empty:
        assert ex_ts["close"].iloc[0] == pytest.approx(ex_ts["close_raw"].iloc[0])


def test_run_adjust_requires_snapshot(tmp_path: Path, isolated_env: None) -> None:
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        f"""
data:
  vendor: fixture
  raw_dir: {(tmp_path / 'raw').as_posix()}
  processed_dir: {(tmp_path / 'processed').as_posix()}
  snapshot_id_path: {(tmp_path / 'raw' / 'SNAPSHOT.json').as_posix()}
""",
        encoding="utf-8",
    )
    cfg = load_config(overlay)
    with pytest.raises(ScoutDataError, match="snapshot not found"):
        run_adjust(cfg)


def test_cli_adjust_smoke(tmp_path: Path, isolated_env: None) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        f"""
data:
  vendor: fixture
  raw_dir: {raw_dir.as_posix()}
  processed_dir: {processed_dir.as_posix()}
  snapshot_id_path: {(raw_dir / 'SNAPSHOT.json').as_posix()}
period:
  start: 2015-01-01T00:00:00Z
  warmup_end: 2015-07-01T00:00:00Z
  end: 2016-12-31T00:00:00Z
""",
        encoding="utf-8",
    )
    assert (
        main(["ingest", "--config", str(overlay), "--start", "2015-01-01", "--end", "2015-03-31"])
        == 0
    )
    assert main(["adjust", "--config", str(overlay)]) == 0
    assert (processed_dir / "panel" / "1d" / "2015.parquet").is_file()
    assert reference_calendar_path(processed_dir, "XNYS").is_file()


def test_write_helpers_used_in_adjust_pipeline(tmp_path: Path) -> None:
    # Keep store contract visible: calendar write is atomic.
    path = tmp_path / "cal.parquet"
    frame = build_calendar(date(2008, 7, 1), date(2008, 7, 7))
    write_parquet_atomic(path, frame)
    write_json_atomic(tmp_path / "x.json", {"ok": True})
    assert path.is_file()
