from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, is_dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from scout.domain.audit import DecisionRecord
from scout.domain.costs import CostEstimate
from scout.domain.edge import MIN_BIN_SAMPLES, BinKey, BinStats, EdgeTable
from scout.domain.enums import (
    ActionType,
    Direction,
    MarketRegime,
    OrderType,
    Regime,
    RejectionReason,
    RunMode,
    SetupOutcome,
    VolBucket,
)
from scout.domain.execution import Fill, OrderIntent
from scout.domain.features import FEATURE_COLUMNS, FeaturePanel, FeatureRow
from scout.domain.market import (
    BENCHMARK_COLUMNS,
    MARKET_COLUMNS,
    Asset,
    Bar,
    BenchmarkPanel,
    CorporateAction,
    EarningsEvent,
    MarketPanel,
    Session,
)
from scout.domain.opportunity import Opportunity
from scout.domain.portfolio import PortfolioState, Position, TradeDecision
from scout.domain.results import BacktestResult, ClosedTrade
from scout.domain.sentiment import SentimentObservation, SentimentView
from scout.domain.setup import ResolvedSetup, Setup
from scout.domain.universe import UniverseEntry, UniverseSnapshot

TS = datetime(2020, 1, 2, 21, 0, tzinfo=UTC)
TS2 = datetime(2020, 1, 3, 21, 0, tzinfo=UTC)


def _setup(**overrides: Any) -> Setup:
    kwargs: dict[str, Any] = {
        "symbol": "AAPL",
        "ts": TS,
        "strategy_id": "donchian_breakout_v1",
        "direction": Direction.LONG,
        "regime": Regime.TREND_UP,
        "reference_price": 100.0,
        "stop_price": 95.0,
        "target_price": 110.0,
        "max_hold_bars": 10,
        "trigger_note": "close>dc_high_55",
    }
    kwargs.update(overrides)
    return Setup(**kwargs)


def _feature_row(**overrides: Any) -> FeatureRow:
    kwargs: dict[str, Any] = {
        "symbol": "AAPL",
        "ts": TS,
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
        "mom_252_xs_pct": 0.92,
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
        "bars_available": 300,
        "bars_since_gap": 50,
        "is_warm": True,
    }
    kwargs.update(overrides)
    return FeatureRow(**kwargs)


def _bin_stats(**overrides: Any) -> BinStats:
    kwargs: dict[str, Any] = {
        "key": BinKey("donchian_breakout_v1", Direction.LONG, VolBucket.MID),
        "as_of": TS,
        "n": MIN_BIN_SAMPLES,
        "mean_r": 0.2,
        "std_r": 1.0,
        "ev_r_lcb": 0.1,
        "mean_bars_held": 8.0,
        "win_rate": 0.4,
        "avg_win_r": 1.5,
        "avg_loss_r": -1.0,
        "target_rate": 0.3,
        "stop_rate": 0.5,
        "time_rate": 0.2,
        "median_r": 0.1,
        "p05_r": -1.2,
        "p95_r": 2.0,
    }
    kwargs.update(overrides)
    return BinStats(**kwargs)


def _market_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ts, session_index in ((TS, 0), (TS2, 1)):
        for asset_id, close in (("AAA", 10.0), ("BBB", 20.0)):
            rows.append(
                {
                    "asset_id": asset_id,
                    "symbol": asset_id,
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
            )
    return rows


def _market_panel() -> MarketPanel:
    return MarketPanel(pd.DataFrame(_market_rows(), columns=list(MARKET_COLUMNS)))


def _position(**overrides: Any) -> Position:
    kwargs: dict[str, Any] = {
        "symbol": "AAPL",
        "direction": Direction.LONG,
        "qty": Decimal("10"),
        "entry_price": Decimal("100"),
        "entry_ts": TS,
        "stop_price": Decimal("95"),
        "target_price": Decimal("110"),
        "max_hold_bars": 10,
        "bars_held": 1,
        "strategy_id": "donchian_breakout_v1",
        "cluster": "INFO_TECH",
        "beta_bench_90": 1.0,
        "client_order_id": "run-AAPL-1-e",
        "realised_fees_usd": Decimal("0"),
        "dividends_usd": Decimal("0"),
        "borrow_usd": Decimal("0"),
        "funding_paid_usd": Decimal("0"),
    }
    kwargs.update(overrides)
    return Position(**kwargs)


DOMAIN_DATACLASSES: tuple[type[Any], ...] = (
    Asset,
    CorporateAction,
    EarningsEvent,
    Session,
    Bar,
    FeatureRow,
    UniverseEntry,
    UniverseSnapshot,
    Setup,
    ResolvedSetup,
    BinKey,
    BinStats,
    CostEstimate,
    Opportunity,
    Position,
    PortfolioState,
    TradeDecision,
    OrderIntent,
    Fill,
    SentimentObservation,
    SentimentView,
    DecisionRecord,
    ClosedTrade,
    BacktestResult,
)


def test_every_dataclass_is_frozen_and_slotted() -> None:
    for cls in DOMAIN_DATACLASSES:
        assert is_dataclass(cls), cls
        params = cls.__dataclass_params__  # type: ignore[attr-defined]
        assert params.frozen, cls
        assert params.slots, cls


def test_setup_rejects_stop_on_wrong_side_long() -> None:
    with pytest.raises(ValueError, match="losing side"):
        _setup(stop_price=105.0)


def test_setup_rejects_stop_on_wrong_side_short() -> None:
    with pytest.raises(ValueError, match="losing side"):
        _setup(direction=Direction.SHORT, stop_price=90.0, target_price=80.0)


def test_setup_rejects_non_positive_risk() -> None:
    with pytest.raises(ValueError, match="risk_per_unit"):
        _setup(stop_price=100.0)


def test_setup_rejects_naive_datetime() -> None:
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        _setup(ts=datetime(2020, 1, 2, 21, 0))  # noqa: DTZ001


def test_setup_rejects_target_on_wrong_side() -> None:
    with pytest.raises(ValueError, match="winning side"):
        _setup(target_price=90.0)


def test_setup_allows_none_target() -> None:
    setup = _setup(target_price=None)
    assert setup.target_price is None
    assert setup.risk_per_unit == 5.0
    with pytest.raises(ValueError, match="target_price is None"):
        _ = setup.reward_per_unit


def test_setup_derived_ratios() -> None:
    setup = _setup()
    assert setup.risk_per_unit == 5.0
    assert setup.reward_per_unit == 10.0
    assert setup.reward_risk_ratio == 2.0


def test_setup_is_frozen() -> None:
    setup = _setup()
    with pytest.raises(FrozenInstanceError):
        setup.stop_price = 90.0  # type: ignore[misc]


@given(
    ref=st.floats(min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False),
    offset=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
)
def test_setup_post_init_rejects_stop_on_wrong_side(ref: float, offset: float) -> None:
    with pytest.raises(ValueError, match="losing side"):
        _setup(reference_price=ref, stop_price=ref + offset, target_price=ref + 2 * offset)


def test_direction_sign() -> None:
    assert Direction.LONG.sign == 1
    assert Direction.SHORT.sign == -1


def test_market_panel_as_of_returns_only_rows_at_or_before_ts() -> None:
    panel = _market_panel()
    view = panel.as_of(TS)
    assert set(view.frame["ts"].unique()) == {pd.Timestamp(TS)}
    assert (view.frame["ts"] <= pd.Timestamp(TS)).all()
    assert len(view.frame) == 2
    full = panel.as_of(TS2)
    assert len(full.frame) == 4
    empty = panel.as_of(datetime(2019, 12, 31, tzinfo=UTC))
    assert empty.frame.empty


def test_market_panel_as_of_includes_the_requested_bar() -> None:
    panel = _market_panel()
    view = panel.as_of(TS)
    assert view.frame["ts"].max() == pd.Timestamp(TS)


def test_market_panel_latest_one_row_per_asset() -> None:
    panel = _market_panel()
    latest = panel.latest(TS2)
    assert len(latest) == 2
    assert set(latest["asset_id"].astype(str)) == {"AAA", "BBB"}
    assert (latest["ts"] == pd.Timestamp(TS2)).all()


def test_market_panel_slice_symbol_ts_indexed() -> None:
    panel = _market_panel()
    sliced = panel.slice_symbol("AAA")
    assert list(sliced.index) == [pd.Timestamp(TS), pd.Timestamp(TS2)]
    assert (sliced["asset_id"].astype(str) == "AAA").all()


def test_market_panel_timestamps_unique_sorted() -> None:
    panel = _market_panel()
    ts = panel.timestamps
    assert ts.is_unique
    assert ts.is_monotonic_increasing


def test_market_panel_rejects_wrong_columns() -> None:
    frame = pd.DataFrame(_market_rows())
    frame = frame.drop(columns=["is_suspect"])
    with pytest.raises(ValueError, match="columns must be"):
        MarketPanel(frame)


def test_market_panel_rejects_naive_ts_column() -> None:
    rows = _market_rows()
    for row in rows:
        row["ts"] = row["ts"].replace(tzinfo=None)
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        MarketPanel(pd.DataFrame(rows, columns=list(MARKET_COLUMNS)))


def test_benchmark_panel_as_of() -> None:
    frame = pd.DataFrame(
        [
            {
                "ts": TS,
                "session_index": 0,
                "open": 300.0,
                "high": 301.0,
                "low": 299.0,
                "close": 300.5,
                "vix_close": 18.0,
                "vix9d_close": 17.0,
                "vix3m_close": 19.0,
            },
            {
                "ts": TS2,
                "session_index": 1,
                "open": 301.0,
                "high": 302.0,
                "low": 300.0,
                "close": 301.5,
                "vix_close": 19.0,
                "vix9d_close": 18.0,
                "vix3m_close": 20.0,
            },
        ],
        columns=list(BENCHMARK_COLUMNS),
    )
    panel = BenchmarkPanel(frame)
    assert len(panel.as_of(TS).frame) == 1
    assert len(panel.as_of(TS2).frame) == 2


def test_feature_panel_columns_match_feature_row() -> None:
    assert tuple(f.name for f in fields(FeatureRow)) == FEATURE_COLUMNS


def test_feature_panel_row_roundtrip() -> None:
    row = _feature_row()
    frame = pd.DataFrame([row.to_dict()])
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    panel = FeaturePanel(frame)
    got = panel.row(TS, "AAPL")
    assert got.symbol == "AAPL"
    assert got.regime is Regime.TREND_UP
    assert got.market_regime is MarketRegime.RISK_ON
    assert got.close == 100.0


def test_universe_eligible_symbols_sorted() -> None:
    entries = {
        "BBB": UniverseEntry("BBB", True, None, 1e8, 5.0, 200, 1000.0),
        "AAA": UniverseEntry("AAA", True, None, 1e8, 5.0, 200, 1000.0),
        "CCC": UniverseEntry(
            "CCC", False, RejectionReason.LOW_LIQUIDITY, 1e6, 20.0, 50, 10.0
        ),
    }
    snap = UniverseSnapshot(TS, entries)
    assert snap.eligible_symbols == ("AAA", "BBB")


def test_universe_entry_reason_iff_ineligible() -> None:
    with pytest.raises(ValueError, match="None iff eligible"):
        UniverseEntry("AAA", True, RejectionReason.NO_SETUP, 1.0, 1.0, 1, 1.0)
    with pytest.raises(ValueError, match="requires a RejectionReason"):
        UniverseEntry("AAA", False, None, 1.0, 1.0, 1, 1.0)


def test_bin_stats_is_usable() -> None:
    assert _bin_stats().is_usable is True
    assert _bin_stats(n=MIN_BIN_SAMPLES - 1).is_usable is False
    assert _bin_stats(std_r=0.0).is_usable is False
    assert _bin_stats(mean_r=float("nan")).is_usable is False


def test_edge_table_lookup_rounds_backward() -> None:
    key = BinKey("donchian_breakout_v1", Direction.LONG, VolBucket.MID)
    april = datetime(2023, 4, 3, 20, 0, tzinfo=UTC)
    may = datetime(2023, 5, 1, 20, 0, tzinfo=UTC)
    table = EdgeTable(
        [
            _bin_stats(key=key, as_of=april, mean_r=0.1),
            _bin_stats(key=key, as_of=may, mean_r=0.2),
        ]
    )
    hit = table.lookup(key, datetime(2023, 4, 17, 20, 0, tzinfo=UTC))
    assert hit is not None
    assert hit.as_of == april
    assert hit.mean_r == 0.1
    assert table.lookup(key, datetime(2023, 4, 1, 20, 0, tzinfo=UTC)) is None


def test_edge_table_save_load(tmp_path: Any) -> None:
    key = BinKey("xsec_momentum_v1", Direction.SHORT, VolBucket.HIGH)
    original = EdgeTable([_bin_stats(key=key, n=200)])
    path = tmp_path / "edge.parquet"
    original.save(path)
    loaded = EdgeTable.load(path)
    got = loaded.lookup(key, TS)
    assert got is not None
    assert got.n == 200
    assert got.key == key


def test_cost_estimate_is_valid() -> None:
    good = CostEstimate(
        symbol="AAPL",
        ts=TS,
        notional_usd=10_000.0,
        expected_bars_held=8.0,
        entry_fee_bps=0.5,
        exit_fee_bps=0.5,
        spread_bps=4.0,
        slippage_bps=1.0,
        impact_bps=2.0,
        funding_bps=-0.1,
        total_bps=7.9,
        cost_usd=7.9,
        cost_r=0.05,
    )
    assert good.is_valid is True
    bad = CostEstimate(
        symbol="AAPL",
        ts=TS,
        notional_usd=10_000.0,
        expected_bars_held=8.0,
        entry_fee_bps=0.5,
        exit_fee_bps=0.5,
        spread_bps=4.0,
        slippage_bps=1.0,
        impact_bps=2.0,
        funding_bps=0.0,
        total_bps=8.0,
        cost_usd=8.0,
        cost_r=0.0,
    )
    assert bad.is_valid is False


def test_position_open_risk_never_negative() -> None:
    pos = _position()
    assert pos.open_risk_usd(Decimal("100")) == Decimal("50")
    assert pos.open_risk_usd(Decimal("95")) == Decimal("0")
    assert pos.open_risk_usd(Decimal("90")) == Decimal("0")
    short = _position(direction=Direction.SHORT, stop_price=Decimal("105"))
    assert short.open_risk_usd(Decimal("100")) == Decimal("50")
    assert short.open_risk_usd(Decimal("110")) == Decimal("0")


def test_portfolio_state_drawdown_and_day_loss() -> None:
    state = PortfolioState(
        ts=TS,
        equity_usd=Decimal("90"),
        cash_usd=Decimal("90"),
        positions={},
        peak_equity_usd=Decimal("100"),
        realised_pnl_today_usd=Decimal("-10"),
        day_start_equity_usd=Decimal("100"),
        trades_today=0,
        bars_since_breaker=10_000,
    )
    assert state.drawdown_pct() == pytest.approx(0.10)
    assert state.day_loss_pct() == pytest.approx(0.10)
    assert state.open_risk_pct == 0.0


def test_portfolio_heat_at_entry() -> None:
    pos = _position()
    state = PortfolioState(
        ts=TS,
        equity_usd=Decimal("100000"),
        cash_usd=Decimal("99000"),
        positions={"AAPL": pos},
        peak_equity_usd=Decimal("100000"),
        realised_pnl_today_usd=Decimal("0"),
        day_start_equity_usd=Decimal("100000"),
        trades_today=1,
        bars_since_breaker=10_000,
    )
    assert state.open_risk_pct == pytest.approx(0.0005)
    assert state.cluster_risk_pct("INFO_TECH") == pytest.approx(0.0005)
    assert state.cluster_risk_pct("ENERGY") == 0.0
    assert state.net_beta_exposure_pct == pytest.approx(0.0005)


def test_net_beta_weights_by_beta_bench_90() -> None:
    pos = _position(beta_bench_90=2.0)
    state = PortfolioState(
        ts=TS,
        equity_usd=Decimal("100000"),
        cash_usd=Decimal("99000"),
        positions={"AAPL": pos},
        peak_equity_usd=Decimal("100000"),
        realised_pnl_today_usd=Decimal("0"),
        day_start_equity_usd=Decimal("100000"),
        trades_today=1,
        bars_since_breaker=10_000,
    )
    # open_risk 50 / 100000 * beta 2.0
    assert state.net_beta_exposure_pct == pytest.approx(0.001)


def test_trade_decision_reason_iff_rejected() -> None:
    setup = _setup()
    opp = Opportunity(
        symbol="AAPL",
        ts=TS,
        strategy_id="donchian_breakout_v1",
        direction=Direction.LONG,
        setup=setup,
        ev_net_r=0.1,
        ev_per_bar_r=0.01,
        ev_r_lcb=0.2,
        ev_r_point=0.3,
        cost_r=0.1,
        expected_bars_held=8.0,
        bin_key=BinKey("donchian_breakout_v1", Direction.LONG, VolBucket.MID),
        bin_n=200,
        bin_std_r=1.0,
        bin_win_rate=0.4,
        regime=Regime.TREND_UP,
        vol_bucket=VolBucket.MID,
        adv_usd_30=1e8,
        spread_bps_est=5.0,
        beta_bench_90=1.0,
        cluster="INFO_TECH",
        atr_pct=0.02,
    )
    with pytest.raises(ValueError, match="iff accepted"):
        TradeDecision(
            opportunity=opp,
            accepted=True,
            rejection_reason=RejectionReason.BELOW_EV_THRESHOLD,
            rank=1,
            qty=None,
            entry_order_type=None,
            intended_notional_usd=None,
            risk_usd=None,
            size_multiplier=None,
            sentiment_multiplier=None,
            client_order_id=None,
            final_cost=None,
        )


def test_sentiment_available_ts_not_before_event() -> None:
    with pytest.raises(ValueError, match="available_ts"):
        SentimentObservation(
            symbol="AAPL",
            source="vix_term",
            event_ts=TS2,
            available_ts=TS,
            raw_score=0.0,
            score=0.0,
            source_weight=1.0,
            sample_size=1,
            payload_hash="x",
        )


def test_sentiment_view_is_neutral() -> None:
    stale = SentimentView("AAPL", TS, 0.2, 0.9, 10, 6.0, 1, 0.1, True)
    assert stale.is_neutral is True
    empty = SentimentView("AAPL", TS, 0.0, 0.0, 0, 0.0, 0, 0.0, False)
    assert empty.is_neutral is True
    live = SentimentView("AAPL", TS, -0.2, 0.8, 10, 1.0, 1, 0.1, False)
    assert live.is_neutral is False


def test_resolved_setup_open_has_no_resolution_ts() -> None:
    with pytest.raises(ValueError, match="OPEN"):
        ResolvedSetup(
            setup=_setup(),
            entry_ts=TS2,
            entry_price=101.0,
            resolution_ts=TS2,
            exit_price=None,
            outcome=SetupOutcome.OPEN,
            bars_held=0,
            realised_r_gross=0.0,
            mae_r=0.0,
            mfe_r=0.0,
            vol_bucket=VolBucket.MID,
        )


def test_asset_rejects_naive_listed_at() -> None:
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        Asset(
            asset_id="1",
            symbol="AAPL",
            exchange="NASDAQ",
            quote_currency="USD",
            is_etf=False,
            cluster="INFO_TECH",
            tick_size=Decimal("0.01"),
            step_size=Decimal("1"),
            min_notional_usd=Decimal("0"),
            listed_at=datetime(1980, 12, 12),  # noqa: DTZ001
            delisted_at=None,
            delist_reason=None,
        )


def test_bar_and_session_utc() -> None:
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        Bar(
            asset_id="1",
            symbol="AAPL",
            ts=datetime(2020, 1, 2, 21, 0),  # noqa: DTZ001
            session_index=0,
            open=1.0,
            high=1.0,
            low=1.0,
            close=1.0,
            close_raw=1.0,
            volume=1.0,
            dollar_volume=1.0,
            is_suspect=False,
        )
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        Session(
            session=TS.date(),
            open_utc=datetime(2020, 1, 2, 14, 30),  # noqa: DTZ001
            close_utc=TS,
            is_half_day=False,
            session_index=0,
        )


def test_feature_row_to_dict_units_and_enums() -> None:
    d = _feature_row().to_dict()
    assert d["regime"] == "TREND_UP"
    assert d["spy_above_ma"] is True
    assert isinstance(d["ts"], str)
    assert d["xs_population"] == 400
    assert set(d) == set(FEATURE_COLUMNS)


def test_closed_trade_and_backtest_result_construct() -> None:
    trade = ClosedTrade(
        symbol="AAPL",
        strategy_id="donchian_breakout_v1",
        direction=Direction.LONG,
        entry_ts=TS,
        exit_ts=TS2,
        entry_price=Decimal("100"),
        exit_price=Decimal("110"),
        qty=Decimal("10"),
        bars_held=1,
        outcome=SetupOutcome.TARGET,
        gross_pnl_usd=Decimal("100"),
        fees_usd=Decimal("1"),
        dividends_usd=Decimal("0"),
        borrow_usd=Decimal("0"),
        funding_usd=Decimal("0"),
        net_pnl_usd=Decimal("99"),
        realised_r=2.0,
        mae_r=-0.1,
        mfe_r=2.0,
        regime=Regime.TREND_UP,
        vol_bucket=VolBucket.MID,
        cluster="INFO_TECH",
        ev_net_r_at_entry=0.15,
    )
    idx = pd.DatetimeIndex([TS, TS2], tz="UTC")
    result = BacktestResult(
        run_id="r1",
        config_hash="abc",
        started_at=TS,
        period_start=TS,
        period_end=TS2,
        mode=RunMode.BACKTEST,
        trades=(trade,),
        equity_curve=pd.Series([100.0, 101.0], index=idx),
        benchmark_curve=pd.Series([1.0, 1.01], index=idx),
        metrics={"sharpe": 0.5},
        n_decisions_considered=10,
        n_rejections_by_reason={RejectionReason.NO_SETUP: 8},
    )
    assert result.mode is RunMode.BACKTEST
    assert ActionType.SPLIT.value == "SPLIT"
    assert OrderType.MARKET.value == "MARKET"
