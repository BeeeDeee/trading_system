"""Simple portfolio aggregation: one target from the primary signal."""

from __future__ import annotations

from decimal import Decimal, ROUND_DOWN

from app.domain.models.enums import Direction
from app.domain.models.state import PortfolioState
from app.domain.models.trading import TargetPosition, TradingSignal


class SimplePortfolioManager:
    """Proposes sizes. Risk Manager may only reduce or reject."""

    def __init__(self) -> None:
        self.mark_price: Decimal | None = None

    def aggregate(
        self, signals: list[TradingSignal], portfolio_state: PortfolioState
    ) -> list[TargetPosition]:
        if not signals:
            return []
        if self.mark_price is None or self.mark_price <= 0:
            raise ValueError("mark_price must be set before aggregate()")

        signal = signals[0]
        if signal.direction == Direction.LONG and signal.target_exposure > 0:
            notional = portfolio_state.equity * signal.target_exposure
            qty = (notional / self.mark_price).quantize(Decimal("0.000001"), rounding=ROUND_DOWN)
        else:
            qty = Decimal(0)

        return [
            TargetPosition(
                instrument=signal.instrument,
                target_quantity=qty,
                reason=f"{signal.strategy_id}:{signal.direction.value}",
                contributing_strategy_ids=(signal.strategy_id,),
            )
        ]
