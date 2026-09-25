import math
from dataclasses import dataclass
from datetime import datetime

from scout.domain._checks import require_aware


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """Round-trip execution cost. `cost_r` is the only field the ranking uses;
    the rest exist so a bad cost estimate can be diagnosed instead of guessed at.
    """

    symbol: str
    ts: datetime
    notional_usd: float  # the notional this estimate was computed for
    expected_bars_held: float

    entry_fee_bps: float
    exit_fee_bps: float
    spread_bps: float  # full spread crossed once per side
    slippage_bps: float
    impact_bps: float
    funding_bps: float  # signed: positive = we pay
    total_bps: float  # sum of the above

    cost_usd: float  # total_bps / 1e4 * notional_usd
    cost_r: float  # cost_usd / risk_capital_usd

    def __post_init__(self) -> None:
        require_aware(self.ts, "ts")

    @property
    def is_valid(self) -> bool:
        """All components finite, non-negative except funding, cost_r > 0."""
        nonneg = (
            self.entry_fee_bps,
            self.exit_fee_bps,
            self.spread_bps,
            self.slippage_bps,
            self.impact_bps,
            self.total_bps,
            self.cost_usd,
        )
        if any((not math.isfinite(x)) or x < 0.0 for x in nonneg):
            return False
        return math.isfinite(self.funding_bps) and math.isfinite(self.cost_r) and self.cost_r > 0.0
