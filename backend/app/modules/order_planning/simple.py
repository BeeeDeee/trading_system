"""Convert approved targets into order intents."""

from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

from app.domain.models.enums import OrderSide, OrderType, RiskDecisionStatus
from app.domain.models.state import PortfolioState
from app.domain.models.trading import OrderIntent, RiskDecision, TargetPosition


class SimpleOrderPlanner:
    def __init__(self, account_id: str = "local-dev") -> None:
        self.account_id = account_id

    def _current_qty(self, instrument: str, portfolio_state: PortfolioState) -> Decimal:
        for pos in portfolio_state.positions:
            if pos.instrument == instrument:
                return pos.quantity
        return Decimal(0)

    def plan(
        self,
        target: TargetPosition,
        risk_decision: RiskDecision,
        portfolio_state: PortfolioState,
    ) -> list[OrderIntent]:
        if risk_decision.status not in (
            RiskDecisionStatus.APPROVE,
            RiskDecisionStatus.MODIFY,
        ):
            return []
        if risk_decision.approved_quantity is None:
            return []

        desired = risk_decision.approved_quantity
        current = self._current_qty(target.instrument, portfolio_state)
        delta = desired - current
        if delta == 0:
            return []

        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        qty = abs(delta)
        intent_id = str(uuid4())
        client_order_id = f"bt-{intent_id}"
        return [
            OrderIntent(
                intent_id=intent_id,
                client_order_id=client_order_id,
                account_id=self.account_id,
                instrument=target.instrument,
                side=side,
                quantity=qty,
                order_type=OrderType.MARKET,
                risk_decision_id=risk_decision.decision_id,
                reduce_only=desired == 0 or (current > 0 and delta < 0 and desired >= 0),
                strategy_id=(
                    target.contributing_strategy_ids[0]
                    if target.contributing_strategy_ids
                    else None
                ),
            )
        ]
