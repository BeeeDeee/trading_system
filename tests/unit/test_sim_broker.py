"""SimBroker fill rules: next-open entry, gap-through stop, clipped target."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pandas as pd
import pytest

from scout.backtest.ledger import apply_entry, apply_exit, initial_state
from scout.backtest.sim_broker import SimBroker
from scout.config.schema import CostsConfig, PortfolioConfig
from scout.domain.costs import CostEstimate
from scout.domain.edge import MIN_BIN_SAMPLES, BinKey
from scout.domain.enums import (
    ActionType,
    Direction,
    OrderType,
    Regime,
    RejectionReason,
    VolBucket,
)
from scout.domain.execution import OrderIntent
from scout.domain.market import MARKET_COLUMNS, CorporateAction, MarketPanel
from scout.domain.opportunity import Opportunity
from scout.domain.portfolio import TradeDecision
from scout.domain.ports import Broker
from scout.domain.setup import Setup

TS0 = datetime(2015, 1, 2, 21, 0, tzinfo=UTC)
TS1 = TS0 + timedelta(days=1)
TS2 = TS0 + timedelta(days=2)
REL = 1e-9
QTY = Decimal("10")
START = Decimal("10000")


def _panel(rows: list[tuple[datetime, float, float, float, float]]) -> MarketPanel:
    records: list[dict[str, Any]] = []
    for i, (ts, o, h, lo, c) in enumerate(rows):
        records.append(
            {
                "asset_id": "AAPL",
                "symbol": "AAPL",
                "ts": ts,
                "session_index": i,
                "open": o,
                "high": h,
                "low": lo,
                "close": c,
                "close_raw": c,
                "volume": 1_000.0,
                "dollar_volume": c * 1_000.0,
                "is_suspect": False,
            }
        )
    return MarketPanel(pd.DataFrame(records, columns=list(MARKET_COLUMNS)))


def _costs() -> CostsConfig:
    return CostsConfig(commission_per_share_usd=0.0, commission_min_usd=0.0)


def _broker(panel: MarketPanel, actions: tuple[CorporateAction, ...] = ()) -> SimBroker:
    return SimBroker(_costs(), PortfolioConfig(), panel=panel, actions=actions)


def _setup(*, direction: Direction = Direction.LONG) -> Setup:
    if direction is Direction.LONG:
        stop, target = 95.0, 110.0
    else:
        stop, target = 105.0, 90.0
    return Setup(
        symbol="AAPL",
        ts=TS0,
        strategy_id="donchian_breakout_v1",
        direction=direction,
        regime=Regime.TREND_UP,
        reference_price=100.0,
        stop_price=stop,
        target_price=target,
        max_hold_bars=10,
        trigger_note="test",
    )


def _decision(*, direction: Direction = Direction.LONG) -> TradeDecision:
    setup = _setup(direction=direction)
    opp = Opportunity(
        symbol="AAPL",
        ts=TS0,
        strategy_id=setup.strategy_id,
        direction=direction,
        setup=setup,
        ev_net_r=0.15,
        ev_per_bar_r=0.015,
        ev_r_lcb=0.20,
        ev_r_point=0.30,
        cost_r=0.05,
        expected_bars_held=10.0,
        bin_key=BinKey(setup.strategy_id, direction, VolBucket.MID),
        bin_n=MIN_BIN_SAMPLES,
        bin_std_r=1.0,
        bin_win_rate=0.4,
        regime=Regime.TREND_UP,
        vol_bucket=VolBucket.MID,
        adv_usd_30=1e8,
        spread_bps_est=1.0,
        beta_bench_90=1.0,
        cluster="INFO_TECH",
        atr_pct=0.02,
    )
    return TradeDecision(
        opportunity=opp,
        accepted=True,
        rejection_reason=None,
        rank=1,
        qty=QTY,
        entry_order_type=OrderType.MARKET,
        intended_notional_usd=QTY * Decimal("100"),
        risk_usd=Decimal("50"),
        size_multiplier=1.0,
        sentiment_multiplier=1.0,
        client_order_id="oid-e",
        final_cost=None,
    )


def _intents(
    *,
    direction: Direction = Direction.LONG,
    stop: Decimal = Decimal("95"),
    target: Decimal = Decimal("110"),
) -> tuple[OrderIntent, OrderIntent, OrderIntent]:
    close_side = Direction.SHORT if direction is Direction.LONG else Direction.LONG
    entry = OrderIntent(
        client_order_id="oid-e",
        symbol="AAPL",
        side=direction,
        order_type=OrderType.MARKET,
        qty=QTY,
        limit_price=None,
        stop_price=None,
        reduce_only=False,
        created_ts=TS0,
        correlation_id="corr-1",
    )
    stop_i = OrderIntent(
        client_order_id="oid-s",
        symbol="AAPL",
        side=close_side,
        order_type=OrderType.STOP_MARKET,
        qty=QTY,
        limit_price=None,
        stop_price=stop,
        reduce_only=True,
        created_ts=TS0,
        correlation_id="corr-1",
    )
    target_i = OrderIntent(
        client_order_id="oid-t",
        symbol="AAPL",
        side=close_side,
        order_type=OrderType.TAKE_PROFIT_MARKET,
        qty=QTY,
        limit_price=target,
        stop_price=None,
        reduce_only=True,
        created_ts=TS0,
        correlation_id="corr-1",
    )
    return entry, stop_i, target_i


def test_sim_broker_satisfies_protocol() -> None:
    broker = _broker(_panel([(TS0, 100.0, 101.0, 99.0, 100.0)]))
    assert isinstance(broker, Broker)
    assert Broker not in SimBroker.__mro__


def test_entry_fills_next_session_open() -> None:
    # Decision close is 100; next session opens at 102. Fill must be 102, not 100.
    panel = _panel(
        [
            (TS0, 99.0, 101.0, 98.0, 100.0),
            (TS1, 102.0, 103.0, 101.0, 102.5),
        ]
    )
    broker = _broker(panel)
    fill, corr = broker.submit_bracket(*_intents(), decision=_decision())
    assert corr == "corr-1"
    assert fill is not None
    assert fill.ts == TS1
    assert fill.price == Decimal("102")
    assert fill.ts != TS0
    assert fill.price != Decimal("100")


def test_gap_through_stop_loses_more_than_one_r() -> None:
    panel = _panel(
        [
            (TS0, 100.0, 101.0, 99.0, 100.0),
            (TS1, 100.0, 101.0, 99.0, 100.0),
            (TS2, 90.0, 91.0, 89.0, 90.5),
        ]
    )
    broker = _broker(panel)
    decision = _decision()
    fill, _corr = broker.submit_bracket(*_intents(), decision=decision)
    assert fill is not None
    assert fill.price == Decimal("100")

    state = apply_entry(initial_state(START, TS0), decision, fill, "corr-1")
    assert broker.poll_fills(TS1) == ()
    exits = broker.poll_fills(TS2)
    assert len(exits) == 1
    assert exits[0].exit_reason == "STOP"
    assert exits[0].price == Decimal("90")

    pos = state.positions["AAPL"]
    _state, trade = apply_exit(state, pos, exits[0], TS2, last_mark=Decimal("100"))
    # (90 - 100) / 5 = -2.0 R; a gap through the stop is worse than -1 R.
    assert trade.realised_r < -1.0
    assert trade.realised_r == pytest.approx(-2.0, rel=REL)
    assert trade.outcome.value == "STOP"


def test_target_never_fills_better_than_target() -> None:
    panel = _panel(
        [
            (TS0, 100.0, 101.0, 99.0, 100.0),
            (TS1, 100.0, 101.0, 99.0, 100.0),
            (TS2, 115.0, 116.0, 114.0, 115.0),
        ]
    )
    broker = _broker(panel)
    decision = _decision()
    fill, _corr = broker.submit_bracket(*_intents(), decision=decision)
    assert fill is not None
    assert broker.poll_fills(TS1) == ()
    exits = broker.poll_fills(TS2)
    assert len(exits) == 1
    assert exits[0].exit_reason == "TARGET"
    assert exits[0].price == Decimal("110")
    assert exits[0].price < Decimal("115")


def test_no_next_bar_no_entry() -> None:
    panel = _panel([(TS0, 100.0, 101.0, 99.0, 100.0)])
    broker = _broker(panel)
    fill, corr = broker.submit_bracket(*_intents(), decision=_decision())
    assert fill is None
    assert corr == "corr-1"
    assert broker.last_entry_rejection is RejectionReason.DATA_GAP
    assert broker.bracket("AAPL") is None
    state = initial_state(START, TS0)
    assert state.positions == {}


def test_dividend_credits_long() -> None:
    panel = _panel(
        [
            (TS0, 100.0, 101.0, 99.0, 100.0),
            (TS1, 100.0, 101.0, 99.0, 100.0),
            (TS2, 99.5, 100.5, 99.0, 100.0),
        ]
    )
    action = CorporateAction(
        asset_id="AAPL",
        ex_date=TS2.date(),
        action_type=ActionType.DIVIDEND,
        split_ratio=1.0,
        cash_amount=0.50,
        new_symbol=None,
    )
    broker = _broker(panel, actions=(action,))
    decision = _decision()
    fill, _corr = broker.submit_bracket(*_intents(), decision=decision)
    assert fill is not None
    state = apply_entry(initial_state(START, TS0), decision, fill, "corr-1")
    cash_before = state.cash_usd
    state = broker.mark(state, panel, TS2)
    assert state.positions["AAPL"].dividends_usd == Decimal("5")
    # $0.50 * 10 shares credited; MTM at close 100 restores the inventory term.
    assert state.cash_usd == cash_before + Decimal("5")
    implied = Decimal("10") * Decimal("100")
    assert state.equity_usd == state.cash_usd + implied


def test_both_barriers_same_bar_gives_stop() -> None:
    panel = _panel(
        [
            (TS0, 100.0, 101.0, 99.0, 100.0),
            (TS1, 100.0, 101.0, 99.0, 100.0),
            (TS2, 100.0, 111.0, 94.0, 100.0),
        ]
    )
    broker = _broker(panel)
    fill, _corr = broker.submit_bracket(*_intents(), decision=_decision())
    assert fill is not None
    assert broker.poll_fills(TS1) == ()
    exits = broker.poll_fills(TS2)
    assert len(exits) == 1
    assert exits[0].exit_reason == "STOP"
    assert exits[0].price == Decimal("95")


def test_stop_target_reanchored_on_entry() -> None:
    """Overnight gap must not change intended R. Same rule as labeling.

    Decision: close 100, stop 95, target 110 (risk 5, reward 10).
    Fill at 102 → working stop 97, target 112.
    A bar that only reaches 96 hits the re-anchored stop and would miss 95.
    """
    panel = _panel(
        [
            (TS0, 100.0, 101.0, 99.0, 100.0),
            (TS1, 102.0, 103.0, 101.0, 102.0),
            (TS2, 100.0, 103.0, 96.0, 100.0),
        ]
    )
    broker = _broker(panel)
    decision = _decision()
    fill, _corr = broker.submit_bracket(*_intents(), decision=decision)
    assert fill is not None
    assert fill.price == Decimal("102")
    bracket = broker.bracket("AAPL")
    assert bracket is not None
    assert bracket.stop_price == Decimal("97")
    assert bracket.target_price == Decimal("112")

    state = apply_entry(initial_state(START, TS0), decision, fill, "corr-1")
    pos = state.positions["AAPL"]
    assert pos.stop_price == Decimal("97")
    assert pos.target_price == Decimal("112")

    assert broker.poll_fills(TS1) == ()
    exits = broker.poll_fills(TS2)
    assert len(exits) == 1
    assert exits[0].exit_reason == "STOP"
    assert exits[0].price == Decimal("97")


def test_exit_fill_has_no_extra_slippage() -> None:
    """Variant A: modelled slip lives entirely on the entry fill. A stop
    that trades at the stop level fills at the stop, not worse.
    """
    panel = _panel(
        [
            (TS0, 100.0, 101.0, 99.0, 100.0),
            (TS1, 100.0, 101.0, 99.0, 100.0),
            (TS2, 100.0, 101.0, 95.0, 100.0),
        ]
    )
    cost = CostEstimate(
        symbol="AAPL",
        ts=TS0,
        notional_usd=1000.0,
        expected_bars_held=10.0,
        entry_fee_bps=0.0,
        exit_fee_bps=0.0,
        spread_bps=0.0,
        slippage_bps=20.0,
        impact_bps=0.0,
        funding_bps=0.0,
        total_bps=20.0,
        cost_usd=2.0,
        cost_r=0.04,
    )
    broker = _broker(panel)
    decision = _decision()
    fill, _corr = broker.submit_bracket(*_intents(), cost=cost, decision=decision)
    assert fill is not None
    assert fill.price == Decimal("100.2")
    bracket = broker.bracket("AAPL")
    assert bracket is not None
    assert bracket.stop_price == Decimal("95.2")
    assert broker.poll_fills(TS1) == ()
    exits = broker.poll_fills(TS2)
    assert len(exits) == 1
    assert exits[0].exit_reason == "STOP"
    assert exits[0].price == Decimal("95.2")


def test_entry_fill_uses_close_raw_scale() -> None:
    """Adjusted open 1e-7 with close_raw $30 fills at $30, not 1e-7."""
    records: list[dict[str, Any]] = []
    rows = [
        (TS0, 1.0e-7, 1.1e-7, 0.9e-7, 1.0e-7, 30.0),
        (TS1, 1.0e-7, 1.1e-7, 0.9e-7, 1.0e-7, 30.0),
    ]
    for i, (ts, o, h, lo, c, raw) in enumerate(rows):
        records.append(
            {
                "asset_id": "AAPL",
                "symbol": "AAPL",
                "ts": ts,
                "session_index": i,
                "open": o,
                "high": h,
                "low": lo,
                "close": c,
                "close_raw": raw,
                "volume": 1_000.0,
                "dollar_volume": raw * 1_000.0,
                "is_suspect": False,
            }
        )
    panel = MarketPanel(pd.DataFrame(records, columns=list(MARKET_COLUMNS)))
    broker = _broker(panel)
    fill, _corr = broker.submit_bracket(*_intents(), decision=_decision())
    assert fill is not None
    assert fill.price == pytest.approx(Decimal("30"))
    assert fill.qty == QTY
