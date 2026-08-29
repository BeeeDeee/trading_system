from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from scout.domain._checks import require_aware
from scout.domain.costs import CostEstimate
from scout.domain.enums import Direction, OrderType, RejectionReason
from scout.domain.opportunity import Opportunity


@dataclass(frozen=True, slots=True)
class Position:
    symbol: str
    direction: Direction
    qty: Decimal  # base units, already step-rounded
    entry_price: Decimal
    entry_ts: datetime
    stop_price: Decimal
    target_price: Decimal
    max_hold_bars: int
    bars_held: int
    strategy_id: str
    cluster: str
    client_order_id: str  # idempotency key of the entry
    realised_fees_usd: Decimal
    dividends_usd: Decimal  # cash dividends received (longs) or paid (shorts)
    borrow_usd: Decimal  # short-stock borrow accrued
    funding_paid_usd: Decimal  # crypto (M7); zero for equities

    def __post_init__(self) -> None:
        require_aware(self.entry_ts, "entry_ts")

    def open_risk_usd(self, mark: Decimal) -> Decimal:
        """Distance from mark to stop, times qty. Never negative: if the stop
        has moved through the mark, risk is 0.
        """
        if self.direction is Direction.LONG:
            dist = mark - self.stop_price
        else:
            dist = self.stop_price - mark
        if dist <= 0:
            return Decimal("0")
        return dist * abs(self.qty)

    def notional_usd(self, mark: Decimal) -> Decimal:
        return abs(self.qty) * mark


@dataclass(frozen=True, slots=True)
class PortfolioState:
    ts: datetime
    equity_usd: Decimal  # cash + unrealised
    cash_usd: Decimal
    positions: Mapping[str, Position]

    peak_equity_usd: Decimal  # for drawdown circuit breaker
    realised_pnl_today_usd: Decimal
    day_start_equity_usd: Decimal
    trades_today: int
    bars_since_breaker: int  # large sentinel when no breaker has tripped

    def __post_init__(self) -> None:
        require_aware(self.ts, "ts")

    def _heat_at_entry(self) -> Decimal:
        # Methods must be pure functions of this object's own fields. Marks are
        # not stored on Position; entry_price is the only price on the object.
        total = Decimal("0")
        for pos in self.positions.values():
            total += pos.open_risk_usd(pos.entry_price)
        return total

    @property
    def open_risk_pct(self) -> float:
        """Portfolio heat: sum of open_risk_usd / equity_usd."""
        if self.equity_usd == 0:
            return 0.0
        return float(self._heat_at_entry() / self.equity_usd)

    @property
    def gross_exposure_pct(self) -> float:
        if self.equity_usd == 0:
            return 0.0
        total = Decimal("0")
        for pos in self.positions.values():
            total += pos.notional_usd(pos.entry_price)
        return float(total / self.equity_usd)

    @property
    def net_beta_exposure_pct(self) -> float:
        # Position has no beta field. Signed open-risk / equity, implicit beta 1.0.
        if self.equity_usd == 0:
            return 0.0
        total = Decimal("0")
        for pos in self.positions.values():
            signed = Decimal(pos.direction.sign) * pos.open_risk_usd(pos.entry_price)
            total += signed
        return float(total / self.equity_usd)

    def cluster_risk_pct(self, cluster: str) -> float:
        if self.equity_usd == 0:
            return 0.0
        total = Decimal("0")
        for pos in self.positions.values():
            if pos.cluster == cluster:
                total += pos.open_risk_usd(pos.entry_price)
        return float(total / self.equity_usd)

    def drawdown_pct(self) -> float:
        """(peak_equity_usd - equity_usd) / peak_equity_usd, floored at 0."""
        if self.peak_equity_usd <= 0:
            return 0.0
        dd = (self.peak_equity_usd - self.equity_usd) / self.peak_equity_usd
        return max(0.0, float(dd))

    def day_loss_pct(self) -> float:
        """(day_start_equity_usd - equity_usd) / day_start_equity_usd, floored
        at 0. Includes unrealised, so an open losing position can trip the
        daily breaker before it is closed. That is intended.
        """
        if self.day_start_equity_usd <= 0:
            return 0.0
        loss = (self.day_start_equity_usd - self.equity_usd) / self.day_start_equity_usd
        return max(0.0, float(loss))


@dataclass(frozen=True, slots=True)
class TradeDecision:
    """The portfolio stage's verdict on one Opportunity."""

    opportunity: Opportunity
    accepted: bool
    rejection_reason: RejectionReason | None  # None iff accepted
    rank: int  # 1-based, pre-portfolio-gate

    # populated only when accepted
    qty: Decimal | None
    entry_order_type: OrderType | None
    intended_notional_usd: Decimal | None
    risk_usd: Decimal | None
    size_multiplier: float | None  # product of all penalties, (0, 1]
    sentiment_multiplier: float | None  # the sentiment component of the above
    client_order_id: str | None
    final_cost: CostEstimate | None

    def __post_init__(self) -> None:
        if self.accepted != (self.rejection_reason is None):
            raise ValueError("rejection_reason must be None iff accepted")
