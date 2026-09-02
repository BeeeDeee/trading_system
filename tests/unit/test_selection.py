"""Portfolio selection: provisional caps, cluster/beta/top-N, breakers, kill switch."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pandas as pd
import pytest

from scout.backtest.sim_broker import SimBroker
from scout.config.schema import CostsConfig, PortfolioConfig, RiskConfig, SentimentConfig
from scout.domain.edge import MIN_BIN_SAMPLES, BinKey
from scout.domain.enums import Direction, OrderType, Regime, RejectionReason, VolBucket
from scout.domain.execution import OrderIntent
from scout.domain.market import MARKET_COLUMNS, Asset, MarketPanel
from scout.domain.opportunity import Opportunity
from scout.domain.portfolio import PortfolioState, Position, TradeDecision
from scout.domain.setup import Setup
from scout.portfolio.breakers import breaker_tripped
from scout.portfolio.selection import select_and_size, simulate_add
from scout.portfolio.sizing import heat_multiplier

TS = datetime(2015, 6, 15, 20, 0, tzinfo=UTC)
TS0 = datetime(2015, 1, 2, 21, 0, tzinfo=UTC)
TS1 = TS0 + timedelta(days=1)
EQUITY = Decimal("100000")
QTY = Decimal("10")


def _setup(**overrides: Any) -> Setup:
    direction = overrides.get("direction", Direction.LONG)
    kwargs: dict[str, Any] = {
        "symbol": "AAPL",
        "ts": TS,
        "strategy_id": "donchian_breakout_v1",
        "direction": direction,
        "regime": Regime.TREND_UP,
        "reference_price": 100.0,
        "stop_price": 95.0 if direction is Direction.LONG else 105.0,
        "target_price": 110.0 if direction is Direction.LONG else 90.0,
        "max_hold_bars": 21,
        "trigger_note": "test",
    }
    kwargs.update(overrides)
    if "stop_price" not in overrides:
        kwargs["stop_price"] = 95.0 if kwargs["direction"] is Direction.LONG else 105.0
    if "target_price" not in overrides:
        kwargs["target_price"] = 110.0 if kwargs["direction"] is Direction.LONG else 90.0
    return Setup(**kwargs)


def _opportunity(**overrides: Any) -> Opportunity:
    setup = overrides.pop("setup", None)
    if setup is None:
        setup = _setup(
            symbol=str(overrides.get("symbol", "AAPL")),
            direction=overrides.get("direction", Direction.LONG),
        )
    kwargs: dict[str, Any] = {
        "symbol": setup.symbol,
        "ts": setup.ts,
        "strategy_id": setup.strategy_id,
        "direction": setup.direction,
        "setup": setup,
        "ev_net_r": 0.40,
        "ev_per_bar_r": 0.04,
        "ev_r_lcb": 0.50,
        "ev_r_point": 0.55,
        "cost_r": 0.10,
        "expected_bars_held": 10.0,
        "bin_key": BinKey(setup.strategy_id, setup.direction, VolBucket.MID),
        "bin_n": MIN_BIN_SAMPLES,
        "bin_std_r": 1.0,
        "bin_win_rate": 0.4,
        "regime": Regime.TREND_UP,
        "vol_bucket": VolBucket.MID,
        "adv_usd_30": 1_000_000_000.0,
        "spread_bps_est": 1.0,
        "beta_bench_90": 1.0,
        "cluster": "INFO_TECH",
        "atr_pct": 0.02,
    }
    kwargs.update(overrides)
    return Opportunity(**kwargs)


def _asset(symbol: str, cluster: str) -> Asset:
    return Asset(
        asset_id=symbol,
        symbol=symbol,
        exchange="NASDAQ",
        quote_currency="USD",
        is_etf=False,
        cluster=cluster,
        tick_size=Decimal("0.01"),
        step_size=Decimal("1"),
        min_notional_usd=Decimal("0"),
        listed_at=None,
        delisted_at=None,
        delist_reason=None,
    )


def _state(**overrides: Any) -> PortfolioState:
    kwargs: dict[str, Any] = {
        "ts": TS,
        "equity_usd": EQUITY,
        "cash_usd": EQUITY,
        "positions": {},
        "peak_equity_usd": EQUITY,
        "realised_pnl_today_usd": Decimal("0"),
        "day_start_equity_usd": EQUITY,
        "trades_today": 0,
        "bars_since_breaker": 10_000,
    }
    kwargs.update(overrides)
    return PortfolioState(**kwargs)


def _position(
    *, symbol: str = "AAPL", open_risk: Decimal = Decimal("400"), **overrides: Any
) -> Position:
    kwargs: dict[str, Any] = {
        "symbol": symbol,
        "direction": Direction.LONG,
        "qty": open_risk,
        "entry_price": Decimal("100"),
        "entry_ts": TS,
        "stop_price": Decimal("99"),
        "target_price": Decimal("110"),
        "max_hold_bars": 21,
        "bars_held": 0,
        "strategy_id": "donchian_breakout_v1",
        "cluster": "INFO_TECH",
        "beta_bench_90": 1.0,
        "client_order_id": f"{symbol}-1-e",
        "realised_fees_usd": Decimal("0"),
        "dividends_usd": Decimal("0"),
        "borrow_usd": Decimal("0"),
        "funding_paid_usd": Decimal("0"),
    }
    kwargs.update(overrides)
    return Position(**kwargs)


def _select(
    ranked: list[Opportunity],
    *,
    state: PortfolioState | None = None,
    cfg: PortfolioConfig | None = None,
    risk_cfg: RiskConfig | None = None,
) -> tuple[TradeDecision, ...]:
    assets = {o.symbol: _asset(o.symbol, o.cluster) for o in ranked}
    return select_and_size(
        ranked,
        state if state is not None else _state(),
        {},
        assets,
        cfg if cfg is not None else PortfolioConfig(),
        risk_cfg if risk_cfg is not None else RiskConfig(),
        SentimentConfig(enabled=False),
        CostsConfig(commission_per_share_usd=0.0, commission_min_usd=0.0),
        0.05,
    )


def test_caps_checked_against_provisional_state() -> None:
    # Each 0.4% risk passes a 1.0% heat cap; three together are 1.2%.
    cfg = PortfolioConfig(
        max_portfolio_heat_pct=0.010,
        heat_taper_start=1.0,
        max_cluster_risk_pct=0.010,
        max_positions_per_cluster=6,
        top_n=3,
        max_positions=6,
    )
    ranked = [
        _opportunity(symbol="AAPL", cluster="INFO_TECH", setup=_setup(symbol="AAPL")),
        _opportunity(symbol="JPM", cluster="FINANCE", setup=_setup(symbol="JPM")),
        _opportunity(symbol="XOM", cluster="ENERGY", setup=_setup(symbol="XOM")),
    ]
    decisions = _select(ranked, cfg=cfg)
    assert [d.accepted for d in decisions] == [True, True, False]
    assert decisions[2].rejection_reason is RejectionReason.PORTFOLIO_HEAT_CAP


def test_cluster_cap() -> None:
    cfg = PortfolioConfig(
        max_positions_per_cluster=2,
        max_cluster_risk_pct=0.020,
        max_portfolio_heat_pct=0.020,
        heat_taper_start=1.0,
        top_n=3,
        max_positions=6,
    )
    ranked = [
        _opportunity(symbol="AAA", cluster="L1", setup=_setup(symbol="AAA")),
        _opportunity(symbol="BBB", cluster="L1", setup=_setup(symbol="BBB")),
        _opportunity(symbol="CCC", cluster="L1", setup=_setup(symbol="CCC")),
    ]
    decisions = _select(ranked, cfg=cfg)
    accepted = [d for d in decisions if d.accepted]
    assert len(accepted) == 2
    assert decisions[2].rejection_reason is RejectionReason.CLUSTER_CAP


def test_beta_cap_credits_hedge() -> None:
    cfg = PortfolioConfig(
        max_net_beta_pct=0.010,
        max_portfolio_heat_pct=0.020,
        heat_taper_start=1.0,
        max_cluster_risk_pct=0.020,
        max_positions_per_cluster=6,
        top_n=3,
    )
    long_a = _opportunity(
        symbol="AAPL",
        cluster="INFO_TECH",
        beta_bench_90=1.5,
        setup=_setup(symbol="AAPL"),
    )
    long_b = _opportunity(
        symbol="MSFT",
        cluster="INFO_TECH",
        beta_bench_90=1.5,
        setup=_setup(symbol="MSFT"),
    )
    two_longs = _select([long_a, long_b], cfg=cfg)
    assert two_longs[0].accepted is True
    assert two_longs[1].accepted is False
    assert two_longs[1].rejection_reason is RejectionReason.BETA_CAP

    short_b = _opportunity(
        symbol="MSFT",
        cluster="FINANCE",
        direction=Direction.SHORT,
        beta_bench_90=1.5,
        setup=_setup(symbol="MSFT", direction=Direction.SHORT),
    )
    hedge = _select([long_a, short_b], cfg=cfg)
    assert hedge[0].accepted is True
    assert hedge[1].accepted is True
    qty0 = hedge[0].qty
    qty1 = hedge[1].qty
    risk0 = hedge[0].risk_usd
    risk1 = hedge[1].risk_usd
    assert qty0 is not None and qty1 is not None
    assert risk0 is not None and risk1 is not None
    state = _state()
    after_long = simulate_add(state, long_a, qty0, risk0)
    after_hedge = simulate_add(after_long, short_b, qty1, risk1)
    assert after_hedge.net_beta_exposure_pct < after_long.net_beta_exposure_pct


def test_heat_taper() -> None:
    cfg = PortfolioConfig()
    used_levels = (0.50, 0.60, 0.70, 0.80, 0.90)
    multipliers: list[float] = []
    for used in used_levels:
        open_risk = Decimal(str(used)) * Decimal(str(cfg.max_portfolio_heat_pct)) * EQUITY
        state = _state(positions={"AAPL": _position(open_risk=open_risk)})
        multipliers.append(heat_multiplier(state, cfg))
    assert multipliers[0] == 1.0
    assert multipliers[1] == 1.0
    assert multipliers[1:] == sorted(multipliers[1:], reverse=True)
    assert all(0.0 < m <= 1.0 for m in multipliers)


def test_every_opportunity_gets_a_decision() -> None:
    cfg = PortfolioConfig(top_n=2, max_positions=6)
    ranked = [
        _opportunity(symbol="AAPL", cluster="INFO_TECH", setup=_setup(symbol="AAPL")),
        _opportunity(symbol="MSFT", cluster="INFO_TECH", setup=_setup(symbol="MSFT")),
        _opportunity(symbol="JPM", cluster="FINANCE", setup=_setup(symbol="JPM")),
        _opportunity(symbol="XOM", cluster="ENERGY", setup=_setup(symbol="XOM")),
    ]
    decisions = _select(ranked, cfg=cfg)
    assert len(decisions) == len(ranked)
    assert [d.rank for d in decisions] == [1, 2, 3, 4]
    for d in decisions:
        if d.accepted:
            assert d.size_multiplier is not None
            assert 0.0 < d.size_multiplier <= 1.0


def test_rejection_before_top_n_recorded() -> None:
    cfg = PortfolioConfig(top_n=3, max_positions=6, max_positions_per_cluster=6)
    ranked = [
        _opportunity(symbol="A", cluster="INFO_TECH", setup=_setup(symbol="A")),
        _opportunity(symbol="B", cluster="FINANCE", setup=_setup(symbol="B")),
        _opportunity(symbol="C", cluster="ENERGY", setup=_setup(symbol="C")),
        _opportunity(symbol="D", cluster="HEALTH", setup=_setup(symbol="D")),
    ]
    decisions = _select(ranked, cfg=cfg)
    assert decisions[3].rank == 4
    assert decisions[3].accepted is False
    assert decisions[3].rejection_reason is RejectionReason.BELOW_TOP_N


def test_breaker_blocks_entries_not_exits() -> None:
    tripped = _state(equity_usd=Decimal("80000"), cash_usd=Decimal("80000"))
    assert tripped.drawdown_pct() >= 0.15
    assert breaker_tripped(tripped, RiskConfig()) is RejectionReason.CIRCUIT_BREAKER

    opp = _opportunity()
    decisions = _select([opp], state=tripped)
    assert decisions[0].accepted is False
    assert decisions[0].rejection_reason is RejectionReason.CIRCUIT_BREAKER

    rows = [
        {
            "asset_id": "AAPL",
            "symbol": "AAPL",
            "ts": ts,
            "session_index": i,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "close_raw": 100.0,
            "volume": 1_000.0,
            "dollar_volume": 100_000.0,
            "is_suspect": False,
        }
        for i, ts in enumerate((TS0, TS1))
    ]
    panel = MarketPanel(pd.DataFrame(rows, columns=list(MARKET_COLUMNS)))
    broker = SimBroker(
        CostsConfig(commission_per_share_usd=0.0, commission_min_usd=0.0),
        PortfolioConfig(),
        panel=panel,
    )
    entry = OrderIntent(
        client_order_id="oid-e",
        symbol="AAPL",
        side=Direction.LONG,
        order_type=OrderType.MARKET,
        qty=QTY,
        limit_price=None,
        stop_price=None,
        reduce_only=False,
        created_ts=TS0,
        correlation_id="corr-1",
    )
    stop = OrderIntent(
        client_order_id="oid-s",
        symbol="AAPL",
        side=Direction.SHORT,
        order_type=OrderType.STOP_MARKET,
        qty=QTY,
        limit_price=None,
        stop_price=Decimal("95"),
        reduce_only=True,
        created_ts=TS0,
        correlation_id="corr-1",
    )
    target = OrderIntent(
        client_order_id="oid-t",
        symbol="AAPL",
        side=Direction.SHORT,
        order_type=OrderType.TAKE_PROFIT_MARKET,
        qty=QTY,
        limit_price=Decimal("110"),
        stop_price=None,
        reduce_only=True,
        created_ts=TS0,
        correlation_id="corr-1",
    )
    fill, _corr = broker.submit_bracket(entry, stop, target)
    assert fill is not None
    broker.poll_fills(TS1)
    exit_fill = broker.close_position("AAPL", "TIME")
    assert exit_fill is not None
    assert exit_fill.exit_reason == "TIME"


def test_kill_switch_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCOUT_TRADING_ENABLED", "0")
    decisions = _select([_opportunity()], risk_cfg=RiskConfig(trading_enabled=True))
    assert len(decisions) == 1
    assert decisions[0].accepted is False
    assert decisions[0].rejection_reason is RejectionReason.KILL_SWITCH
