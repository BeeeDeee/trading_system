"""Execution cost model: IBKR-style commission, square-root impact, shorts pay."""

from __future__ import annotations

import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from scout.config.loader import load_config
from scout.config.schema import CostsConfig
from scout.costs.model import estimate_cost
from scout.domain.costs import CostEstimate
from scout.domain.enums import Direction, Regime
from scout.domain.setup import Setup
from scout.domain.universe import UniverseEntry
from scout.utils.errors import ScoutConfigError

TS = datetime(2015, 6, 15, 20, 0, tzinfo=UTC)

# 08-COSTS.md §3.1 Example A — liquid large cap, long.
RISK_CAPITAL = 400.0
STOP_ATR = 3.0
ATR_PCT_A = 0.018
NOTIONAL_A = RISK_CAPITAL / (STOP_ATR * ATR_PCT_A)  # 400 / 0.054
PRICE_RAW_A = 230.0
ADV_A = 1_500_000_000.0
SPREAD_A = 1.0
BARS_HELD = 21.0

# Hand-computed Example A (08-COSTS.md §3.1), exact inputs not the rounded display.
# shares = notional/230; per-share commission 32.206*0.0035 = $0.113 < $0.35 min
# per_side = $0.35; commission_bps = 2*0.35/notional*1e4 = 0.945
# spread round-trip = 2.0
# slip_vol = 0.02 * 0.018 * 1e4 = 3.6; slippage = 2*(1.0+3.6) = 9.2
# sqrt(notional/1.5e9) = 1/450; impact = 180/450 = 0.4
# total_bps = 12.545; cost_r = 12.545e-4 / 0.054
_COMMISSION_BPS_A = 0.945
_SPREAD_RT_A = 2.0
_SLIPPAGE_BPS_A = 9.2
_IMPACT_BPS_A = 0.4
_TOTAL_BPS_A = _COMMISSION_BPS_A + _SPREAD_RT_A + _SLIPPAGE_BPS_A + _IMPACT_BPS_A
_COST_R_A = _TOTAL_BPS_A / 1e4 / (STOP_ATR * ATR_PCT_A)
_DOC_COST_R_A = 0.023


def _setup(**overrides: Any) -> Setup:
    kwargs: dict[str, Any] = {
        "symbol": "AAPL",
        "ts": TS,
        "strategy_id": "donchian_breakout_v1",
        "direction": Direction.LONG,
        "regime": Regime.TREND_UP,
        "reference_price": PRICE_RAW_A,
        "stop_price": PRICE_RAW_A * (1.0 - STOP_ATR * ATR_PCT_A),
        "target_price": None,
        "max_hold_bars": 21,
        "trigger_note": "test_cost",
    }
    kwargs.update(overrides)
    if "stop_price" not in overrides and kwargs["direction"] is Direction.SHORT:
        kwargs["stop_price"] = PRICE_RAW_A * (1.0 + STOP_ATR * ATR_PCT_A)
    return Setup(**kwargs)


def _entry(**overrides: Any) -> UniverseEntry:
    kwargs: dict[str, Any] = {
        "symbol": "AAPL",
        "eligible": True,
        "reason": None,
        "adv_usd_30": ADV_A,
        "spread_bps_est": SPREAD_A,
        "bars_available": 400,
        "listed_days": 1000.0,
    }
    kwargs.update(overrides)
    return UniverseEntry(**kwargs)


def _estimate(
    *,
    setup: Setup | None = None,
    universe_entry: UniverseEntry | None = None,
    notional_usd: float = NOTIONAL_A,
    risk_capital_usd: float = RISK_CAPITAL,
    expected_bars_held: float = BARS_HELD,
    cfg: CostsConfig | None = None,
    atr_pct: float = ATR_PCT_A,
    price_raw: float = PRICE_RAW_A,
    borrow_bps_per_year: float | None = None,
    dividend_yield_annual: float | None = None,
) -> CostEstimate:
    return estimate_cost(
        setup if setup is not None else _setup(),
        universe_entry if universe_entry is not None else _entry(),
        notional_usd,
        risk_capital_usd,
        expected_bars_held,
        cfg if cfg is not None else CostsConfig(),
        atr_pct=atr_pct,
        price_raw=price_raw,
        borrow_bps_per_year=borrow_bps_per_year,
        dividend_yield_annual=dividend_yield_annual,
    )


def test_worked_example() -> None:
    est = _estimate()
    assert est.is_valid is True
    assert est.entry_fee_bps == pytest.approx(_COMMISSION_BPS_A / 2.0, rel=1e-12)
    assert est.exit_fee_bps == pytest.approx(_COMMISSION_BPS_A / 2.0, rel=1e-12)
    assert est.spread_bps == pytest.approx(_SPREAD_RT_A, rel=1e-12)
    assert est.slippage_bps == pytest.approx(_SLIPPAGE_BPS_A, rel=1e-12)
    assert est.impact_bps == pytest.approx(_IMPACT_BPS_A, rel=1e-12)
    assert est.funding_bps == pytest.approx(0.0, abs=1e-15)
    assert est.total_bps == pytest.approx(_TOTAL_BPS_A, rel=1e-12)
    assert est.cost_r == pytest.approx(_COST_R_A, rel=1e-12)
    assert est.cost_r == pytest.approx(_DOC_COST_R_A, abs=0.001)


def test_cost_r_invariant_to_risk_fraction() -> None:
    # $0.35 ticket minimum is size-dependent; zero it so the residual is impact only.
    cfg = CostsConfig(commission_min_usd=0.0)
    full = _estimate(cfg=cfg)
    half = _estimate(
        notional_usd=NOTIONAL_A / 2.0,
        risk_capital_usd=RISK_CAPITAL / 2.0,
        cfg=cfg,
    )
    assert full.cost_r != 0.0
    rel = abs(half.cost_r - full.cost_r) / abs(full.cost_r)
    assert rel < 0.01


def test_borrow_sign() -> None:
    long_est = _estimate(
        setup=_setup(direction=Direction.LONG),
        borrow_bps_per_year=30.0,
        dividend_yield_annual=0.0,
    )
    short_est = _estimate(
        setup=_setup(direction=Direction.SHORT),
        borrow_bps_per_year=30.0,
        dividend_yield_annual=0.0,
    )
    years_held = BARS_HELD / 252.0
    expected_borrow_bps = 30.0 * years_held
    assert long_est.funding_bps == pytest.approx(0.0, abs=1e-15)
    assert short_est.funding_bps == pytest.approx(expected_borrow_bps, rel=1e-12)
    assert short_est.cost_r > long_est.cost_r


def test_dividend_sign() -> None:
    long_est = _estimate(
        setup=_setup(direction=Direction.LONG),
        borrow_bps_per_year=0.0,
        dividend_yield_annual=0.015,
    )
    short_est = _estimate(
        setup=_setup(direction=Direction.SHORT),
        borrow_bps_per_year=0.0,
        dividend_yield_annual=0.015,
    )
    years_held = BARS_HELD / 252.0
    expected_div_bps = 0.015 * 1e4 * years_held
    assert long_est.funding_bps == pytest.approx(0.0, abs=1e-15)
    assert short_est.funding_bps == pytest.approx(expected_div_bps, rel=1e-12)
    assert short_est.cost_r > long_est.cost_r


def test_impact_sqrt_law() -> None:
    base = _estimate(notional_usd=NOTIONAL_A)
    quad = _estimate(notional_usd=4.0 * NOTIONAL_A)
    assert base.impact_bps > 0.0
    assert quad.impact_bps / base.impact_bps == pytest.approx(2.0, rel=1e-12)


def test_spread_floor_applied() -> None:
    zero = _estimate(universe_entry=_entry(spread_bps_est=0.0))
    negative = _estimate(universe_entry=_entry(spread_bps_est=-5.0))
    positive = _estimate(universe_entry=_entry(spread_bps_est=1.0))
    assert negative.spread_bps == pytest.approx(0.0, abs=1e-15)
    assert negative.cost_r == pytest.approx(zero.cost_r, rel=1e-12)
    assert negative.cost_r < positive.cost_r


def test_cost_multiplier_below_one_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key in list(os.environ):
        if key.startswith("SCOUT_"):
            monkeypatch.delenv(key, raising=False)
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text("costs:\n  cost_multiplier: 0.9\n", encoding="utf-8")
    with pytest.raises(ScoutConfigError, match="cost_multiplier"):
        load_config(overlay)

    cfg = CostsConfig(cost_multiplier=0.9)
    with pytest.raises(ScoutConfigError, match="cost_multiplier"):
        _estimate(cfg=cfg)


def test_cost_components_non_negative() -> None:
    long_est = _estimate()
    short_est = _estimate(setup=_setup(direction=Direction.SHORT))
    for est in (long_est, short_est):
        assert math.isfinite(est.entry_fee_bps) and est.entry_fee_bps >= 0.0
        assert math.isfinite(est.exit_fee_bps) and est.exit_fee_bps >= 0.0
        assert math.isfinite(est.spread_bps) and est.spread_bps >= 0.0
        assert math.isfinite(est.slippage_bps) and est.slippage_bps >= 0.0
        assert math.isfinite(est.impact_bps) and est.impact_bps >= 0.0
        assert math.isfinite(est.total_bps) and est.total_bps >= 0.0
        assert math.isfinite(est.funding_bps) and est.funding_bps >= 0.0
        assert est.is_valid is True
