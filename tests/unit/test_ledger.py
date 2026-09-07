"""Ledger: Decimal cash accounting, equity identity, zero-equity halt."""

from __future__ import annotations

from dataclasses import fields, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pandas as pd
import pytest

from scout.backtest.ledger import (
    accrue_borrow,
    apply_dividend,
    apply_entry,
    apply_exit,
    apply_split,
    initial_state,
    mark,
)
from scout.domain.costs import CostEstimate
from scout.domain.edge import MIN_BIN_SAMPLES, BinKey
from scout.domain.enums import Direction, OrderType, Regime, VolBucket
from scout.domain.execution import Fill
from scout.domain.market import MARKET_COLUMNS, MarketPanel
from scout.domain.opportunity import Opportunity
from scout.domain.portfolio import PortfolioState, Position, TradeDecision
from scout.domain.results import ClosedTrade
from scout.domain.setup import Setup
from scout.utils.errors import ScoutError

TS0 = datetime(2015, 1, 2, 21, 0, tzinfo=UTC)
TS1 = TS0 + timedelta(days=1)
TS2 = TS0 + timedelta(days=2)
REL = 1e-9
START = Decimal("10000")
QTY = Decimal("10")
ENTRY = Decimal("100")
STOP = Decimal("95")
TARGET = Decimal("110")
FEE = Decimal("1")


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
        intended_notional_usd=QTY * ENTRY,
        risk_usd=Decimal("50"),
        size_multiplier=1.0,
        sentiment_multiplier=1.0,
        client_order_id="oid-e",
        final_cost=_cost(),
    )


def _cost() -> CostEstimate:
    return CostEstimate(
        symbol="AAPL",
        ts=TS0,
        notional_usd=1000.0,
        expected_bars_held=10.0,
        entry_fee_bps=10.0,
        exit_fee_bps=10.0,
        spread_bps=0.0,
        slippage_bps=0.0,
        impact_bps=0.0,
        funding_bps=0.0,
        total_bps=20.0,
        cost_usd=2.0,
        cost_r=0.04,
    )


def _fill(
    *,
    price: Decimal,
    ts: datetime,
    fee: Decimal = FEE,
    side: Direction = Direction.LONG,
    exit_reason: str | None = None,
    client_order_id: str = "oid-e",
) -> Fill:
    return Fill(
        client_order_id=client_order_id,
        symbol="AAPL",
        side=side,
        qty=QTY,
        price=price,
        fee_usd=fee,
        ts=ts,
        is_maker=False,
        exit_reason=exit_reason,
    )


def _money_fields(obj: object) -> list[tuple[str, object]]:
    out: list[tuple[str, object]] = []
    for f in fields(obj):  # type: ignore[arg-type]
        name = f.name
        if name.endswith("_usd") or name.endswith("_price") or name in {"qty"}:
            if name in {"mae_r", "mfe_r", "realised_r", "ev_net_r_at_entry"}:
                continue
            out.append((name, getattr(obj, name)))
    return out


def test_equity_identity() -> None:
    panel = _panel(
        [
            (TS0, 100.0, 101.0, 99.0, 100.0),
            (TS1, 100.0, 102.0, 99.0, 101.0),
            (TS2, 101.0, 103.0, 100.0, 102.0),
        ]
    )
    state = initial_state(START, TS0)
    assert state.equity_usd == state.cash_usd

    state = apply_entry(state, _decision(), _fill(price=ENTRY, ts=TS1), "corr")
    pos = state.positions["AAPL"]
    implied = Decimal(pos.direction.sign) * pos.qty * pos.entry_price
    assert state.equity_usd == state.cash_usd + implied

    state = mark(state, panel, TS1)
    implied = Decimal(pos.direction.sign) * pos.qty * Decimal("101")
    assert state.equity_usd == state.cash_usd + implied

    state = apply_dividend(state, "AAPL", Decimal("0.50"), TS1)
    assert state.equity_usd == state.cash_usd + implied

    pos = state.positions["AAPL"]
    exit_fill = _fill(
        price=TARGET,
        ts=TS2,
        side=Direction.SHORT,
        exit_reason="TARGET",
        client_order_id="oid-t",
    )
    state, _trade = apply_exit(state, pos, exit_fill, TS2, last_mark=Decimal("101"))
    assert state.positions == {}
    assert state.equity_usd == state.cash_usd


def test_apply_entry_identity_not_three_term_add() -> None:
    """Decimal is not associative at prec=28. cash + implied + notional drifts."""
    messy = replace(
        initial_state(START, TS0),
        cash_usd=Decimal("80297.76471136582818206050732"),
        equity_usd=Decimal("107320.9919555519281845850412"),
    )
    state = apply_entry(messy, _decision(), _fill(price=ENTRY, ts=TS1), "corr")
    implied = state.equity_usd - state.cash_usd
    assert state.equity_usd == state.cash_usd + implied


def test_decimal_throughout() -> None:
    panel = _panel(
        [
            (TS0, 100.0, 101.0, 99.0, 100.0),
            (TS1, 100.0, 101.0, 99.0, 100.0),
        ]
    )
    state = initial_state(START, TS0)
    state = apply_entry(state, _decision(), _fill(price=ENTRY, ts=TS1), "corr")
    state = mark(state, panel, TS1)
    pos = state.positions["AAPL"]
    state, trade = apply_exit(
        state,
        pos,
        _fill(price=STOP, ts=TS1, side=Direction.SHORT, exit_reason="STOP"),
        TS1,
        last_mark=Decimal("100"),
    )
    for label, obj in (
        ("state", state),
        ("trade", trade),
        ("fill", _fill(price=ENTRY, ts=TS1)),
    ):
        for name, value in _money_fields(obj):
            assert type(value) is Decimal, f"{label}.{name} is {type(value)}"
    assert type(pos.qty) is Decimal
    assert type(pos.entry_price) is Decimal
    assert type(pos.stop_price) is Decimal
    assert type(pos.target_price) is Decimal
    assert type(pos.realised_fees_usd) is Decimal
    assert type(pos.dividends_usd) is Decimal
    assert type(pos.borrow_usd) is Decimal
    assert type(pos.funding_paid_usd) is Decimal
    assert type(state.peak_equity_usd) is Decimal
    assert type(state.realised_pnl_today_usd) is Decimal
    assert type(state.day_start_equity_usd) is Decimal


def test_round_trip_pnl() -> None:
    """Short 10 @ 100, cover @ 90. Fees $1+$1, dividend $5 paid, borrow $0.50.

    gross = -1 * (90 - 100) * 10 = 100
    net   = 100 - 2 - 0.50 + (-5) = 92.50
    """
    state = initial_state(START, TS0)
    state = apply_entry(
        state,
        _decision(direction=Direction.SHORT),
        _fill(price=ENTRY, ts=TS1, side=Direction.SHORT),
        "corr",
    )
    # Cash: 10000 - (-1)*1000 - 1 = 10999
    assert state.cash_usd == Decimal("10999")
    assert state.equity_usd == Decimal("9999")

    state = apply_dividend(state, "AAPL", Decimal("0.50"), TS1)
    assert state.positions["AAPL"].dividends_usd == Decimal("-5")
    assert state.cash_usd == Decimal("10994")

    state = accrue_borrow(state, "AAPL", Decimal("0.50"), TS1)
    assert state.positions["AAPL"].borrow_usd == Decimal("0.50")
    assert state.cash_usd == Decimal("10993.50")

    pos = state.positions["AAPL"]
    cover = _fill(
        price=Decimal("90"),
        ts=TS2,
        side=Direction.LONG,
        exit_reason="TARGET",
        client_order_id="oid-t",
    )
    state, trade = apply_exit(state, pos, cover, TS2, last_mark=ENTRY)
    assert trade.gross_pnl_usd == Decimal("100")
    assert trade.fees_usd == Decimal("2")
    assert trade.dividends_usd == Decimal("-5")
    assert trade.borrow_usd == Decimal("0.50")
    assert trade.net_pnl_usd == Decimal("92.50")
    assert trade.realised_r == pytest.approx(1.85, rel=REL)
    assert state.cash_usd == Decimal("10092.50")
    assert state.equity_usd == state.cash_usd
    assert state.positions == {}


def test_zero_equity_halts() -> None:
    panel = _panel(
        [
            (TS0, 100.0, 101.0, 99.0, 100.0),
            (TS1, 100.0, 101.0, 99.0, 100.0),
            (TS2, 0.0, 0.01, 0.0, 0.0),
        ]
    )
    state = initial_state(Decimal("1000"), TS0)
    state = apply_entry(
        state,
        _decision(),
        Fill(
            client_order_id="oid-e",
            symbol="AAPL",
            side=Direction.LONG,
            qty=QTY,
            price=ENTRY,
            fee_usd=Decimal("0"),
            ts=TS1,
            is_maker=False,
            exit_reason=None,
        ),
        "corr",
    )
    assert state.cash_usd == Decimal("0")
    assert state.equity_usd == Decimal("1000")
    with pytest.raises(ScoutError, match="account blew up"):
        mark(state, panel, TS2)


def test_decimal_fields_on_initial_state() -> None:
    state = initial_state(START, TS0)
    for name, value in _money_fields(state):
        assert type(value) is Decimal, name
    assert isinstance(state, PortfolioState)
    assert isinstance(_fill(price=ENTRY, ts=TS1), Fill)
    assert ClosedTrade.__dataclass_params__.frozen  # type: ignore[attr-defined]
    assert Position.__dataclass_params__.frozen  # type: ignore[attr-defined]


def test_apply_exit_forwards_regime_and_vol_bucket() -> None:
    state = initial_state(START, TS0)
    state = apply_entry(state, _decision(), _fill(price=ENTRY, ts=TS1), "corr")
    pos = state.positions["AAPL"]
    _state, trade = apply_exit(
        state,
        pos,
        _fill(price=TARGET, ts=TS2, side=Direction.SHORT, exit_reason="TARGET"),
        TS2,
        last_mark=ENTRY,
        regime=Regime.TREND_UP,
        vol_bucket=VolBucket.HIGH,
        ev_net_r_at_entry=0.15,
    )
    assert trade.regime is Regime.TREND_UP
    assert trade.vol_bucket is VolBucket.HIGH
    assert trade.ev_net_r_at_entry == pytest.approx(0.15)


def test_apply_split_preserves_notional() -> None:
    state = initial_state(START, TS0)
    state = apply_entry(state, _decision(), _fill(price=ENTRY, ts=TS1), "corr")
    pos = state.positions["AAPL"]
    notional = pos.qty * pos.entry_price
    cash = state.cash_usd
    equity = state.equity_usd
    state = apply_split(state, "AAPL", Decimal("2"), TS2)
    pos = state.positions["AAPL"]
    assert pos.qty == QTY * Decimal("2")
    assert pos.entry_price == ENTRY / Decimal("2")
    assert pos.qty * pos.entry_price == notional
    assert state.cash_usd == cash
    assert state.equity_usd == equity
