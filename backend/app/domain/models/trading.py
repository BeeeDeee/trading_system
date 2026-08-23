"""Trading decision and execution domain models."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from app.domain.models.enums import (
    Direction,
    ExecutionStatus,
    OrderSide,
    OrderType,
    RiskDecisionStatus,
)


@dataclass(frozen=True, slots=True)
class TradingSignal:
    strategy_id: str
    instrument: str
    direction: Direction
    confidence: Decimal
    target_exposure: Decimal
    timestamp: datetime
    signal_id: str | None = None


@dataclass(frozen=True, slots=True)
class TargetPosition:
    instrument: str
    target_quantity: Decimal
    reason: str
    contributing_strategy_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RiskDecision:
    decision_id: str
    status: RiskDecisionStatus
    target: TargetPosition
    timestamp: datetime
    approved_quantity: Decimal | None = None
    reason: str | None = None
    rule_id: str | None = None


@dataclass(frozen=True, slots=True)
class OrderIntent:
    intent_id: str
    client_order_id: str
    account_id: str
    instrument: str
    side: OrderSide
    quantity: Decimal
    order_type: OrderType
    risk_decision_id: str
    price: Decimal | None = None
    reduce_only: bool = False
    strategy_id: str | None = None


@dataclass(frozen=True, slots=True)
class ExecutionReport:
    order_id: str
    status: ExecutionStatus
    timestamp: datetime
    filled_quantity: Decimal = Decimal(0)
    average_price: Decimal | None = None
    fee: Decimal | None = None
    reason: str | None = None
    client_order_id: str | None = None
