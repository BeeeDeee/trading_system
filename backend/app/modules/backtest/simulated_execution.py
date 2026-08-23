"""Simulated market fills for backtesting."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from uuid import uuid4

from app.domain.models.enums import ExecutionStatus, OrderSide
from app.domain.models.trading import ExecutionReport, OrderIntent


class SimulatedExecution:
    def __init__(self, fee_bps: Decimal = Decimal("10"), slippage_bps: Decimal = Decimal("5")) -> None:
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps

    def fill(self, intent: OrderIntent, mark_price: Decimal, timestamp) -> ExecutionReport:
        slip = mark_price * (self.slippage_bps / Decimal(10000))
        if intent.side == OrderSide.BUY:
            px = mark_price + slip
        else:
            px = mark_price - slip
        px = px.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        notional = px * intent.quantity
        fee = (notional * self.fee_bps / Decimal(10000)).quantize(
            Decimal("0.00000001"), rounding=ROUND_HALF_UP
        )
        return ExecutionReport(
            order_id=str(uuid4()),
            client_order_id=intent.client_order_id,
            status=ExecutionStatus.FILLED,
            timestamp=timestamp,
            filled_quantity=intent.quantity,
            average_price=px,
            fee=fee,
        )
