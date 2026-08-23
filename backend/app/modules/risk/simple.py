"""Risk gate: constrain proposed size; never increase."""

from __future__ import annotations

from decimal import Decimal, ROUND_DOWN
from uuid import uuid4

from app.domain.models.enums import RiskDecisionStatus
from app.domain.models.state import AccountState, PortfolioState
from app.domain.models.trading import RiskDecision, TargetPosition


class SimpleRiskManager:
    def __init__(
        self,
        max_position_fraction: Decimal = Decimal("0.95"),
        allow_short: bool = False,
        mark_price: Decimal | None = None,
    ) -> None:
        self.max_position_fraction = max_position_fraction
        self.allow_short = allow_short
        self.mark_price = mark_price

    def evaluate(
        self,
        target: TargetPosition,
        portfolio_state: PortfolioState,
        account_state: AccountState,
    ) -> RiskDecision:
        del account_state
        decision_id = str(uuid4())
        ts = portfolio_state.positions[0].updated_at if portfolio_state.positions else None

        # Use a timestamp from caller via portfolio equity update path — engine stamps later.
        from datetime import datetime, timezone

        now = ts or datetime.now(timezone.utc)

        qty = target.target_quantity
        if qty < 0 and not self.allow_short:
            return RiskDecision(
                decision_id=decision_id,
                status=RiskDecisionStatus.REJECT,
                target=target,
                timestamp=now,
                reason="shorts_disabled",
                rule_id="no_short",
            )

        if self.mark_price is None or self.mark_price <= 0:
            return RiskDecision(
                decision_id=decision_id,
                status=RiskDecisionStatus.REJECT,
                target=target,
                timestamp=now,
                reason="missing_mark_price",
                rule_id="mark_price",
            )

        max_notional = portfolio_state.equity * self.max_position_fraction
        max_qty = (max_notional / self.mark_price).quantize(
            Decimal("0.000001"), rounding=ROUND_DOWN
        )
        if abs(qty) > max_qty:
            return RiskDecision(
                decision_id=decision_id,
                status=RiskDecisionStatus.MODIFY,
                target=target,
                timestamp=now,
                approved_quantity=max_qty if qty > 0 else -max_qty,
                reason="max_position_fraction",
                rule_id="max_position",
            )

        return RiskDecision(
            decision_id=decision_id,
            status=RiskDecisionStatus.APPROVE,
            target=target,
            timestamp=now,
            approved_quantity=qty,
            reason="ok",
            rule_id=None,
        )
