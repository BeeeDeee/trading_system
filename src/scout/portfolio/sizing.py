"""Risk-based position sizing. The float → Decimal boundary starts here."""

from __future__ import annotations

import math
from decimal import Decimal

from scout.config.schema import PortfolioConfig
from scout.domain.market import Asset
from scout.domain.opportunity import Opportunity
from scout.domain.portfolio import PortfolioState
from scout.utils.decimals import round_qty, round_usd, to_decimal
from scout.utils.errors import ScoutError


def size_position(
    opp: Opportunity,
    state: PortfolioState,
    asset: Asset,
    multiplier: float,
    cfg: PortfolioConfig,
) -> tuple[Decimal, Decimal]:
    """Return (qty, actual_risk_usd). qty is step-rounded DOWN."""
    equity = state.equity_usd
    risk_frac = Decimal(str(cfg.risk_fraction_per_trade))
    mult = Decimal(str(multiplier))

    risk_usd = round_usd(equity * risk_frac * mult)

    entry = to_decimal(opp.setup.reference_price)
    stop = to_decimal(opp.setup.stop_price)
    risk_per_unit = abs(entry - stop)
    if risk_per_unit <= 0:
        raise ScoutError("setup with non-positive risk reached sizing")

    qty_raw = risk_usd / risk_per_unit

    # Notional caps, applied BEFORE step rounding.
    max_notional = min(
        equity * Decimal(str(cfg.max_position_notional_pct)),
        Decimal(str(opp.adv_usd_30 * cfg.max_pct_of_adv)),
    )
    if entry > 0:
        qty_raw = min(qty_raw, max_notional / entry)

    qty = round_qty(qty_raw, asset.step_size)
    actual_risk = qty * risk_per_unit
    return qty, actual_risk


def heat_multiplier(state: PortfolioState, cfg: PortfolioConfig) -> float:
    """Taper size as the portfolio fills up. Returns (0, 1]."""
    used = state.open_risk_pct / cfg.max_portfolio_heat_pct
    if used <= cfg.heat_taper_start:
        return 1.0
    denom = 1.0 - cfg.heat_taper_start
    if denom <= 0.0:
        return cfg.heat_min_multiplier
    remaining = (1.0 - used) / denom
    return min(1.0, max(cfg.heat_min_multiplier, remaining))


def liquidity_multiplier(opp: Opportunity, cfg: PortfolioConfig) -> float:
    """Taper size for symbols near the liquidity floor. Returns (0, 1]."""
    ratio = opp.adv_usd_30 / cfg.liquidity_full_size_adv_usd
    return min(1.0, max(cfg.liquidity_min_multiplier, math.sqrt(ratio)))
