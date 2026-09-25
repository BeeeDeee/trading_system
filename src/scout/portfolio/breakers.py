"""Kill switch and circuit breakers. Block opening risk only, never exits."""

from __future__ import annotations

import os

from scout.config.schema import RiskConfig
from scout.domain.enums import RejectionReason
from scout.domain.portfolio import PortfolioState

# Must match scout.config.loader._parse_bool.
_ENV_FALSE = frozenset({"0", "false", "no", "off", ""})


def trading_enabled(cfg: RiskConfig) -> bool:
    """Env wins over config and is re-read on every call, not cached at startup."""
    raw = os.environ.get("SCOUT_TRADING_ENABLED")
    if raw is not None:
        return raw.strip().lower() not in _ENV_FALSE
    return cfg.trading_enabled


def breaker_tripped(state: PortfolioState, cfg: RiskConfig) -> RejectionReason | None:
    """Return CIRCUIT_BREAKER if a risk limit is breached, else None."""
    if state.day_loss_pct() >= cfg.max_daily_loss_pct:
        return RejectionReason.CIRCUIT_BREAKER
    if state.drawdown_pct() >= cfg.max_drawdown_pct:
        return RejectionReason.CIRCUIT_BREAKER
    if state.trades_today >= cfg.max_trades_per_day:
        return RejectionReason.CIRCUIT_BREAKER
    if state.bars_since_breaker < cfg.breaker_cooldown_bars:
        return RejectionReason.CIRCUIT_BREAKER
    return None
