"""Position sizing: risk formula, step rounding down, notional and ADV caps."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from scout.config.schema import PortfolioConfig
from scout.domain.edge import MIN_BIN_SAMPLES, BinKey
from scout.domain.enums import Direction, Regime, VolBucket
from scout.domain.market import Asset
from scout.domain.opportunity import Opportunity
from scout.domain.portfolio import PortfolioState
from scout.domain.setup import Setup
from scout.portfolio.sizing import heat_multiplier, liquidity_multiplier, size_position
from scout.utils.decimals import to_decimal

TS = datetime(2015, 6, 15, 20, 0, tzinfo=UTC)
EQUITY = Decimal("100000")


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
        "max_hold_bars": 21,
        "trigger_note": "test",
    }
    kwargs.update(overrides)
    return Setup(**kwargs)


def _opportunity(**overrides: Any) -> Opportunity:
    setup = overrides.pop("setup", None) or _setup()
    kwargs: dict[str, Any] = {
        "symbol": setup.symbol,
        "ts": setup.ts,
        "strategy_id": setup.strategy_id,
        "direction": setup.direction,
        "setup": setup,
        "ev_net_r": 0.20,
        "ev_per_bar_r": 0.02,
        "ev_r_lcb": 0.25,
        "ev_r_point": 0.30,
        "cost_r": 0.05,
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


def _asset(**overrides: Any) -> Asset:
    kwargs: dict[str, Any] = {
        "asset_id": "AAPL",
        "symbol": "AAPL",
        "exchange": "NASDAQ",
        "quote_currency": "USD",
        "is_etf": False,
        "cluster": "INFO_TECH",
        "tick_size": Decimal("0.01"),
        "step_size": Decimal("1"),
        "min_notional_usd": Decimal("0"),
        "listed_at": None,
        "delisted_at": None,
        "delist_reason": None,
    }
    kwargs.update(overrides)
    return Asset(**kwargs)


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


def test_risk_based_size() -> None:
    # equity 100k * 0.4% = $400 risk; stop $5; qty = 80 exactly, no rounding.
    opp = _opportunity(setup=_setup(reference_price=100.0, stop_price=95.0))
    qty, risk = size_position(opp, _state(), _asset(), 1.0, PortfolioConfig())
    assert qty == Decimal("80")
    assert risk == Decimal("400.00")
    assert qty * Decimal("5") == EQUITY * Decimal("0.004")


def test_step_rounding_always_down() -> None:
    # $400 / $6 = 66.666… shares. Rounding up would be 67 and $402 of risk.
    opp = _opportunity(setup=_setup(reference_price=100.0, stop_price=94.0))
    qty, risk = size_position(opp, _state(), _asset(), 1.0, PortfolioConfig())
    assert qty == Decimal("66")
    assert risk == Decimal("396")
    assert risk < EQUITY * Decimal("0.004")
    assert (qty + 1) * Decimal("6") > EQUITY * Decimal("0.004")


def test_notional_cap_binds_in_low_vol() -> None:
    # Tiny stop (0.1) ⇒ risk formula wants 4000 shares / $400k notional.
    # max_position_notional_pct 0.25 ⇒ $25k / $100 = 250 shares.
    setup = _setup(reference_price=100.0, stop_price=99.9)
    opp = _opportunity(setup=setup, atr_pct=0.001)
    qty, risk = size_position(opp, _state(), _asset(), 1.0, PortfolioConfig())
    assert qty == Decimal("250")
    assert qty * Decimal("100") == EQUITY * Decimal("0.25")
    assert risk == Decimal("25.0")
    assert risk < EQUITY * Decimal("0.004")


def test_adv_cap_binds() -> None:
    # Thin book: 0.5% of $100k ADV = $500 notional = 5 shares at $100.
    opp = _opportunity(
        setup=_setup(reference_price=100.0, stop_price=95.0),
        adv_usd_30=100_000.0,
    )
    qty, risk = size_position(opp, _state(), _asset(), 1.0, PortfolioConfig())
    assert qty == Decimal("5")
    assert qty * Decimal("100") == Decimal("500")
    assert risk == Decimal("25")


def test_multiplier_never_above_one() -> None:
    cfg = PortfolioConfig()
    empty = _state()
    assert 0.0 < heat_multiplier(empty, cfg) <= 1.0
    opp = _opportunity(adv_usd_30=1.0)
    assert 0.0 < liquidity_multiplier(opp, cfg) <= 1.0
    opp_full = _opportunity(adv_usd_30=cfg.liquidity_full_size_adv_usd)
    assert liquidity_multiplier(opp_full, cfg) == 1.0


@given(adv=st.floats(min_value=1.0, max_value=1e12, allow_nan=False, allow_infinity=False))
def test_liquidity_multiplier_in_unit_interval(adv: float) -> None:
    m = liquidity_multiplier(_opportunity(adv_usd_30=adv), PortfolioConfig())
    assert 0.0 < m <= 1.0


@given(
    equity=st.decimals(min_value=1000, max_value=10_000_000, places=2),
    atr_pct=st.floats(min_value=0.001, max_value=0.20, allow_nan=False, allow_infinity=False),
    risk_frac=st.floats(min_value=0.001, max_value=0.02, allow_nan=False, allow_infinity=False),
)
def test_sizing_never_exceeds_risk_mandate(
    equity: Decimal, atr_pct: float, risk_frac: float
) -> None:
    entry = 100.0
    stop = entry * (1.0 - atr_pct)
    opp = _opportunity(
        setup=_setup(reference_price=entry, stop_price=stop),
        atr_pct=atr_pct,
        adv_usd_30=1e12,
    )
    cfg = PortfolioConfig(
        risk_fraction_per_trade=risk_frac,
        max_position_notional_pct=10.0,
        max_pct_of_adv=1.0,
    )
    state = _state(
        equity_usd=equity,
        cash_usd=equity,
        peak_equity_usd=equity,
        day_start_equity_usd=equity,
    )
    _qty, risk = size_position(opp, state, _asset(), 1.0, cfg)
    mandate = equity * to_decimal(risk_frac)
    assert risk <= mandate * Decimal("1.0001")
