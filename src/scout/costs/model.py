"""Round-trip execution cost. One implementation for backtest, paper, and live."""

from __future__ import annotations

import math

from scout.config.schema import CostsConfig
from scout.domain.costs import CostEstimate
from scout.domain.enums import Direction
from scout.domain.setup import Setup
from scout.domain.universe import UniverseEntry
from scout.utils.errors import ScoutConfigError

_SESSIONS_PER_YEAR = 252.0
_BPS = 1e4
_MIN_ADV_USD = 1.0


def estimate_cost(
    setup: Setup,
    universe_entry: UniverseEntry,
    notional_usd: float,
    risk_capital_usd: float,
    expected_bars_held: float,
    cfg: CostsConfig,
    *,
    atr_pct: float,
    price_raw: float,
    borrow_bps_per_year: float | None = None,
    dividend_yield_annual: float | None = None,
) -> CostEstimate:
    """Formulas in 08-COSTS.md. Same function in backtest and live."""
    if cfg.cost_multiplier < 1.0:
        raise ScoutConfigError("costs.cost_multiplier: must be >= 1.0")

    shares = notional_usd / price_raw
    per_share_usd = cfg.commission_per_share_usd * shares
    per_side_usd = min(
        max(per_share_usd, cfg.commission_min_usd),
        cfg.commission_max_pct_of_notional * notional_usd,
    )
    entry_fee_bps = per_side_usd / notional_usd * _BPS
    exit_fee_bps = entry_fee_bps

    # A negative Corwin-Schultz estimate must not credit the trade (Rule 4).
    one_way_spread_bps = max(universe_entry.spread_bps_est, 0.0)
    spread_bps = 2.0 * one_way_spread_bps

    slip_vol_bps = cfg.slippage_vol_coef * atr_pct * _BPS
    slippage_bps = 2.0 * (cfg.slippage_fixed_bps + slip_vol_bps)

    adv_usd = universe_entry.adv_usd_30
    participation = notional_usd / max(adv_usd, _MIN_ADV_USD)
    impact_bps = cfg.impact_coef * atr_pct * _BPS * math.sqrt(participation)

    if setup.direction is Direction.SHORT:
        years_held = expected_bars_held / _SESSIONS_PER_YEAR
        borrow_rate = (
            cfg.borrow_bps_per_year_default
            if borrow_bps_per_year is None
            else borrow_bps_per_year
        )
        div_yield = (
            cfg.dividend_yield_default
            if dividend_yield_annual is None
            else dividend_yield_annual
        )
        borrow_bps = borrow_rate * years_held
        dividend_bps = div_yield * _BPS * years_held
        funding_bps = borrow_bps + dividend_bps
    else:
        funding_bps = 0.0

    mult = cfg.cost_multiplier
    entry_fee_bps *= mult
    exit_fee_bps *= mult
    spread_bps *= mult
    slippage_bps *= mult
    impact_bps *= mult
    funding_bps *= mult

    total_bps = (
        entry_fee_bps
        + exit_fee_bps
        + spread_bps
        + slippage_bps
        + impact_bps
        + funding_bps
    )
    cost_usd = total_bps / _BPS * notional_usd
    cost_r = cost_usd / risk_capital_usd
    return CostEstimate(
        symbol=setup.symbol,
        ts=setup.ts,
        notional_usd=notional_usd,
        expected_bars_held=expected_bars_held,
        entry_fee_bps=entry_fee_bps,
        exit_fee_bps=exit_fee_bps,
        spread_bps=spread_bps,
        slippage_bps=slippage_bps,
        impact_bps=impact_bps,
        funding_bps=funding_bps,
        total_bps=total_bps,
        cost_usd=cost_usd,
        cost_r=cost_r,
    )
