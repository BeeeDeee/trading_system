"""Point-in-time universe snapshots. Trailing data only; ineligible rows kept."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scout.cli.__main__ import main
from scout.config.schema import SpreadFloorConfig, UniverseConfig
from scout.data.schemas import UNIVERSE_SNAPSHOT_COLUMNS
from scout.data.store import write_parquet_atomic
from scout.domain.enums import RejectionReason
from scout.domain.market import MARKET_COLUMNS
from scout.universe.build import (
    _assert_no_lookahead,
    build_snapshots,
    lookup_snapshot,
    snapshot_timestamps,
    to_universe_snapshot,
)
from scout.universe.eligibility import first_failing_reason
from scout.universe.spread import corwin_schultz_bps, spread_bps_est, spread_floor_bps
from scout.utils.errors import ScoutLookaheadError

REL = 1e-9
# 2015-01-05 is a Monday. Consecutive calendar days keep weekday arithmetic simple.
START = datetime(2015, 1, 5, 21, 0, tzinfo=UTC)


def _uni_cfg(**overrides: object) -> UniverseConfig:
    defaults: dict[str, object] = {
        "snapshot_frequency": "monthly",
        "universe_size": 1000,
        "min_history_bars": 20,
        "min_bars_since_gap": 2,
        "min_price_usd": 5.0,
        "min_adv_usd": 1_000.0,
        "max_spread_bps": 50.0,
        "max_suspect_lookback": 5,
        "delisting_grace_bars": 3,
        "spread_window_bars": 5,
    }
    defaults.update(overrides)
    return UniverseConfig(**defaults)  # type: ignore[arg-type]


def _timestamps(n: int) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([START + timedelta(days=i) for i in range(n)], tz="UTC")


def _calendar(timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session": [ts.date() for ts in timestamps],
            "open_utc": timestamps - pd.Timedelta(hours=6, minutes=30),
            "close_utc": timestamps,
            "is_half_day": False,
            "session_index": np.arange(len(timestamps), dtype=np.int32),
        }
    )


def _tickers(
    pairs: list[tuple[str, str]], *, delisted: dict[str, date] | None = None
) -> pd.DataFrame:
    gone = delisted or {}
    rows = [
        {
            "asset_id": asset_id,
            "symbol": symbol,
            "exchange": "NYSE",
            "category": "Domestic Common Stock",
            "sector": "L1",
            "is_etf": False,
            "listed_date": date(2010, 1, 1),
            "delisted_date": gone.get(asset_id, pd.NaT),
            "delist_reason": "",
        }
        for asset_id, symbol in pairs
    ]
    return pd.DataFrame(rows)


def _candidates(pairs: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {"asset_id": [a for a, _ in pairs], "symbol": [s for _, s in pairs]}
    )


def _panel_rows(
    timestamps: pd.DatetimeIndex,
    series: dict[str, dict[str, object]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for i, ts in enumerate(timestamps):
        for asset_id, spec in series.items():
            n = int(spec.get("n", len(timestamps)))  # type: ignore[arg-type]
            if i >= n:
                continue
            close = float(spec.get("close", 50.0))  # type: ignore[arg-type]
            close_raw = float(spec.get("close_raw", close))  # type: ignore[arg-type]
            dollar = float(spec.get("dollar_volume", 10_000_000.0))  # type: ignore[arg-type]
            high = float(spec.get("high", close + 0.01))  # type: ignore[arg-type]
            low = float(spec.get("low", close - 0.01))  # type: ignore[arg-type]
            rows.append(
                {
                    "asset_id": asset_id,
                    "symbol": str(spec.get("symbol", asset_id)),
                    "ts": ts,
                    "session_index": i,
                    "open": close,
                    "high": high,
                    "low": low,
                    "close": close,
                    "close_raw": close_raw,
                    "volume": 1_000.0,
                    "dollar_volume": dollar,
                    "is_suspect": bool(spec.get("is_suspect", False)),
                }
            )
    return pd.DataFrame(rows, columns=list(MARKET_COLUMNS))


def _build(
    n: int,
    series: dict[str, dict[str, object]],
    *,
    cfg: UniverseConfig | None = None,
    pairs: list[tuple[str, str]] | None = None,
    delisted: dict[str, date] | None = None,
    panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    timestamps = _timestamps(n)
    used_pairs = pairs or [(aid, str(spec.get("symbol", aid))) for aid, spec in series.items()]
    frame = panel if panel is not None else _panel_rows(timestamps, series)
    return build_snapshots(
        frame,
        candidates=_candidates(used_pairs),
        tickers=_tickers(used_pairs, delisted=delisted),
        calendar=_calendar(timestamps),
        cfg=cfg or _uni_cfg(),
        start=timestamps[0].to_pydatetime(),
        end=timestamps[-1].to_pydatetime(),
    )


def test_corwin_schultz_golden_constant_range() -> None:
    # H=2, L=1 every bar. beta = 2 ln(2)^2, gamma = ln(2)^2 -> alpha = ln(2), spread = 2/3.
    n = 20
    high = pd.Series([2.0] * n)
    low = pd.Series([1.0] * n)
    got = corwin_schultz_bps(high, low, window=5)
    assert pd.isna(got.iloc[4])
    assert got.iloc[5] == pytest.approx(2.0 / 3.0 * 1e4, rel=REL)
    assert got.iloc[-1] == pytest.approx(6666.6666666667, rel=1e-9)


def test_spread_floor_applied_when_cs_is_zero() -> None:
    n = 20
    high = pd.Series([100.0] * n)
    low = pd.Series([100.0] * n)
    adv = pd.Series([10_000_000.0] * n)
    floors = SpreadFloorConfig()
    floor = spread_floor_bps(adv, floors)
    assert floor.iloc[-1] == pytest.approx(floors.tier_below)
    est = spread_bps_est(high, low, adv, _uni_cfg())
    assert est.iloc[-1] == pytest.approx(floors.tier_below)


def test_negative_corwin_schultz_clamps_to_floor() -> None:
    # Overnight gap, tiny daily range: alpha is negative, CS clamps to 0, floor binds.
    highs = [101.0, 110.0] * 15
    lows = [100.0, 109.0] * 15
    high = pd.Series(highs)
    low = pd.Series(lows)
    raw = corwin_schultz_bps(high, low, window=5)
    assert raw.iloc[-1] == pytest.approx(0.0)
    adv = pd.Series([15_000_000.0] * len(high))
    est = spread_bps_est(high, low, adv, _uni_cfg())
    assert est.iloc[-1] == pytest.approx(SpreadFloorConfig().tier_below)


def test_first_failure_is_insufficient_history() -> None:
    cfg = _uni_cfg()
    reason = first_failing_reason(
        bars_available=5,
        has_session_at_ts=False,
        is_delisted=True,
        suspect_in_lookback=True,
        bars_since_gap=0,
        close_raw=1.0,
        adv_usd_60=1.0,
        adv_rank=9_999,
        spread_bps_est=100.0,
        cfg=cfg,
    )
    assert reason is RejectionReason.INSUFFICIENT_HISTORY


def test_snapshot_uses_only_trailing_data() -> None:
    n = 120
    series = {
        "A1": {"symbol": "AAA", "dollar_volume": 20_000_000.0, "close": 50.0},
        "B2": {"symbol": "BBB", "dollar_volume": 5_000_000.0, "close": 40.0},
    }
    cfg = _uni_cfg(universe_size=1)
    baseline = _build(n, series, cfg=cfg)
    timestamps = _timestamps(n)
    april = datetime(2015, 4, 1, 21, 0, tzinfo=UTC)
    assert april in set(timestamps)
    cut = timestamps.get_loc(april)
    mutated = _panel_rows(timestamps, series)
    future = mutated["ts"] > timestamps[cut]
    mutated.loc[future, "dollar_volume"] = 1e12
    mutated.loc[future, "close_raw"] = 0.01
    mutated.loc[future, "close"] = 0.01
    altered = _build(n, series, cfg=cfg, panel=mutated)
    base_at = baseline.loc[baseline["ts"] == timestamps[cut]].sort_values("asset_id")
    alt_at = altered.loc[altered["ts"] == timestamps[cut]].sort_values("asset_id")
    assert list(base_at["eligible"]) == list(alt_at["eligible"])
    assert list(base_at["reason"]) == list(alt_at["reason"])
    assert list(base_at["adv_rank"]) == list(alt_at["adv_rank"])
    assert np.allclose(base_at["adv_usd_60"].to_numpy(), alt_at["adv_usd_60"].to_numpy())


def test_ineligible_rows_recorded() -> None:
    n = 80
    series = {
        "A1": {"symbol": "AAA", "dollar_volume": 20_000_000.0},
        "THIN": {"symbol": "ZZZ", "dollar_volume": 10.0},
    }
    out = _build(n, series, cfg=_uni_cfg(universe_size=10, min_adv_usd=1_000_000.0))
    thin = out.loc[out["asset_id"] == "THIN"]
    assert not thin.empty
    assert bool((~thin["eligible"]).any())
    assert "" not in set(thin.loc[~thin["eligible"], "reason"])


def test_snapshot_lookup_rounds_backward() -> None:
    n = 10
    series = {"A1": {"symbol": "AAA"}}
    cfg = _uni_cfg(snapshot_frequency="weekly", min_history_bars=1, min_bars_since_gap=0)
    out = _build(n, series, cfg=cfg)
    monday = START
    wednesday = START + timedelta(days=2)
    assert monday.weekday() == 0
    assert wednesday.weekday() == 2
    found = lookup_snapshot(out, wednesday)
    assert not found.empty
    ts = pd.to_datetime(found["ts"].iloc[0], utc=True)
    assert ts == pd.Timestamp(monday)


def test_eligible_symbols_sorted() -> None:
    n = 120
    pairs = [("C3", "CCC"), ("A1", "AAA"), ("B2", "BBB")]
    series = {
        aid: {"symbol": sym, "dollar_volume": 20_000_000.0 + i * 1_000.0}
        for i, (aid, sym) in enumerate(pairs)
    }
    cfg = _uni_cfg(universe_size=10)
    out = _build(n, series, cfg=cfg, pairs=pairs)
    april = datetime(2015, 4, 1, 21, 0, tzinfo=UTC)
    slice_ = out.loc[out["ts"] == april]
    snap = to_universe_snapshot(slice_)
    assert snap.eligible_symbols == tuple(sorted(snap.eligible_symbols))
    assert snap.eligible_symbols == tuple(
        sorted(s for s in ("AAA", "BBB", "CCC") if s in snap.eligible_symbols)
    )


def test_min_price_uses_close_raw() -> None:
    n = 120
    series = {
        "A1": {
            "symbol": "AAA",
            "close": 40.0,
            "close_raw": 4.0,
            "dollar_volume": 20_000_000.0,
        },
        "B2": {
            "symbol": "BBB",
            "close": 40.0,
            "close_raw": 40.0,
            "dollar_volume": 19_000_000.0,
        },
    }
    out = _build(n, series, cfg=_uni_cfg(universe_size=10, min_price_usd=5.0))
    april = datetime(2015, 4, 1, 21, 0, tzinfo=UTC)
    row = out.loc[(out["ts"] == april) & (out["asset_id"] == "A1")].iloc[0]
    assert row["close_raw"] == pytest.approx(4.0)
    assert not bool(row["eligible"])
    assert row["reason"] == RejectionReason.LOW_PRICE.value
    other = out.loc[(out["ts"] == april) & (out["asset_id"] == "B2")].iloc[0]
    assert bool(other["eligible"])


def test_adv_rank_is_binding_size_constraint() -> None:
    n = 120
    series = {
        "A1": {"symbol": "AAA", "dollar_volume": 40_000_000.0},
        "B2": {"symbol": "BBB", "dollar_volume": 30_000_000.0},
        "C3": {"symbol": "CCC", "dollar_volume": 20_000_000.0},
        "D4": {"symbol": "DDD", "dollar_volume": 10_000_000.0},
    }
    out = _build(n, series, cfg=_uni_cfg(universe_size=2, min_adv_usd=1_000.0))
    april = datetime(2015, 4, 1, 21, 0, tzinfo=UTC)
    rows = out.loc[out["ts"] == april].set_index("asset_id")
    assert int(rows.loc["A1", "adv_rank"]) == 1
    assert int(rows.loc["B2", "adv_rank"]) == 2
    assert bool(rows.loc["A1", "eligible"])
    assert bool(rows.loc["B2", "eligible"])
    assert not bool(rows.loc["C3", "eligible"])
    assert rows.loc["C3", "reason"] == RejectionReason.NOT_IN_UNIVERSE.value
    assert rows.loc["D4", "reason"] == RejectionReason.NOT_IN_UNIVERSE.value


def test_column_set_matches_schema() -> None:
    out = _build(40, {"A1": {"symbol": "AAA"}}, cfg=_uni_cfg(min_history_bars=1))
    assert list(out.columns) == list(UNIVERSE_SNAPSHOT_COLUMNS)


def test_candidate_order_does_not_change_ranks() -> None:
    n = 120
    series = {
        "A1": {"symbol": "AAA", "dollar_volume": 40_000_000.0},
        "B2": {"symbol": "BBB", "dollar_volume": 10_000_000.0},
    }
    cfg = _uni_cfg(universe_size=2)
    forward = _build(n, series, cfg=cfg, pairs=[("A1", "AAA"), ("B2", "BBB")])
    backward = _build(n, series, cfg=cfg, pairs=[("B2", "BBB"), ("A1", "AAA")])
    april = datetime(2015, 4, 1, 21, 0, tzinfo=UTC)
    a = forward.loc[forward["ts"] == april].sort_values("asset_id").reset_index(drop=True)
    b = backward.loc[backward["ts"] == april].sort_values("asset_id").reset_index(drop=True)
    assert list(a["adv_rank"]) == list(b["adv_rank"])
    assert list(a["eligible"]) == list(b["eligible"])


def test_delisted_ticker_is_stale_even_with_a_bar() -> None:
    n = 80
    series = {"A1": {"symbol": "AAA", "dollar_volume": 20_000_000.0}}
    delist_on = date(2015, 2, 1)
    out = _build(n, series, cfg=_uni_cfg(), delisted={"A1": delist_on})
    feb = datetime(2015, 2, 1, 21, 0, tzinfo=UTC)
    row = out.loc[(out["ts"] == feb) & (out["asset_id"] == "A1")].iloc[0]
    assert not bool(row["eligible"])
    assert row["reason"] in {
        RejectionReason.STALE_DATA.value,
        RejectionReason.INSUFFICIENT_HISTORY.value,
    }


def test_cli_prints_eligible_count_per_year(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import os

    for key in list(os.environ):
        if key.startswith("SCOUT_"):
            monkeypatch.delenv(key, raising=False)
    n = 40
    timestamps = _timestamps(n)
    pairs = [(f"A{i}", f"S{i}") for i in range(7)]
    series = {aid: {"symbol": sym, "dollar_volume": 20_000_000.0} for aid, sym in pairs}
    panel = _panel_rows(timestamps, series)
    raw_dir = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw_dir.mkdir()
    tickers = _tickers(pairs, delisted={"A0": date(2016, 1, 1), "A1": date(2016, 6, 1)})
    write_parquet_atomic(raw_dir / "tickers.parquet", tickers)
    write_parquet_atomic(processed / "panel" / "1d" / "2015.parquet", panel)
    write_parquet_atomic(tmp_path / "reference" / "calendar_xnys.parquet", _calendar(timestamps))
    candidates = tmp_path / "candidates.txt"
    candidates.write_text(
        "\n".join(f"{aid},{sym}" for aid, sym in pairs) + "\n", encoding="utf-8"
    )
    snaps = tmp_path / "universe" / "snapshots.parquet"
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        f"""
period:
  start: 2015-01-05T21:00:00Z
  warmup_end: 2015-07-06T00:00:00Z
  end: 2015-08-01T00:00:00Z
data:
  vendor: fixture
  raw_dir: {raw_dir.as_posix()}
  processed_dir: {processed.as_posix()}
  snapshot_id_path: {(raw_dir / "SNAPSHOT.json").as_posix()}
universe:
  candidates_file: {candidates.as_posix()}
  snapshots_path: {snaps.as_posix()}
  snapshot_frequency: monthly
  min_history_bars: 1
  min_bars_since_gap: 0
  min_adv_usd: 1
""",
        encoding="utf-8",
    )
    code = main(["build-universe", "--config", str(overlay)])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert "eligible_by_year" in captured.out
    assert "2015" in captured.out
    assert snaps.is_file()
    loaded = pd.read_parquet(snaps)
    assert list(loaded.columns) == list(UNIVERSE_SNAPSHOT_COLUMNS)


def test_lookahead_assertion_trips_on_future_bar() -> None:
    frame = pd.DataFrame(
        {
            "ts": [START],
            "panel_ts": [START + timedelta(days=1)],
        }
    )
    with pytest.raises(ScoutLookaheadError, match="ts > snapshot ts"):
        _assert_no_lookahead(frame)


def test_weekly_and_monthly_grids() -> None:
    timestamps = _timestamps(40)
    cal = _calendar(timestamps)
    monthly = snapshot_timestamps(
        cal, "monthly", timestamps[0].to_pydatetime(), timestamps[-1].to_pydatetime()
    )
    weekly = snapshot_timestamps(
        cal, "weekly", timestamps[0].to_pydatetime(), timestamps[-1].to_pydatetime()
    )
    assert monthly[0] == timestamps[0]
    assert datetime(2015, 2, 1, 21, 0, tzinfo=UTC) in set(monthly)
    assert weekly[0] == timestamps[0]
    assert len(weekly) > len(monthly)
