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


def to_raw_distance(
    adj_from: float,
    adj_to: float,
    *,
    raw_price: Decimal,
) -> Decimal:
    """Map `|adj_from - adj_to|` onto the share units of `raw_price`.

    Distances are taken in Decimal (`str(float)`) so `100.0 - 99.9` is `0.1`.
    Setup stops live in adjusted space; `raw_price` is `close_raw` (ADR-017).
    """
    ref = to_decimal(adj_from)
    if ref <= 0:
        raise ScoutError("adjusted reference_price must be positive")
    if raw_price <= 0:
        raise ScoutError("price_raw must be positive")
    dist = abs(ref - to_decimal(adj_to))
    return dist * (raw_price / ref)


def size_position(
    opp: Opportunity,
    state: PortfolioState,
    asset: Asset,
    multiplier: float,
    cfg: PortfolioConfig,
    *,
    price_raw: float | None = None,
) -> tuple[Decimal, Decimal]:
    """Return (qty, actual_risk_usd). qty is step-rounded DOWN.

    `price_raw` is the unadjusted decision close (`close_raw`). When omitted,
    `setup.reference_price` is used (tests where adj == raw).
    """
    equity = state.equity_usd
    risk_frac = Decimal(str(cfg.risk_fraction_per_trade))
    mult = Decimal(str(multiplier))

    risk_usd = round_usd(equity * risk_frac * mult)

    entry_raw = to_decimal(
        opp.setup.reference_price if price_raw is None else price_raw
    )
    risk_per_unit = to_raw_distance(
        opp.setup.reference_price,
        opp.setup.stop_price,
        raw_price=entry_raw,
    )
    if risk_per_unit <= 0:
        raise ScoutError("setup with non-positive risk reached sizing")

    qty_raw = risk_usd / risk_per_unit

    # Notional caps, applied BEFORE step rounding.
    max_notional = min(
        equity * Decimal(str(cfg.max_position_notional_pct)),
        Decimal(str(opp.adv_usd_30 * cfg.max_pct_of_adv)),
    )
    if entry_raw > 0:
        qty_raw = min(qty_raw, max_notional / entry_raw)

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
