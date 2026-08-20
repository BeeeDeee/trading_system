"""Local, external, and derived state models."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from app.domain.models.enums import ExecutionStatus


@dataclass(frozen=True, slots=True)
class OrderState:
    order_id: str
    intent_id: str
    instrument: str
    status: ExecutionStatus
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class PositionState:
    instrument: str
    quantity: Decimal
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class AccountState:
    equity: Decimal
    available_capital: Decimal
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class PortfolioState:
    equity: Decimal
    positions: tuple[PositionState, ...]
    realized_pnl: Decimal = Decimal(0)
    unrealized_pnl: Decimal = Decimal(0)


@dataclass(frozen=True, slots=True)
class ExternalOrder:
    external_order_id: str
    instrument: str
    status: ExecutionStatus
    observed_at: datetime


@dataclass(frozen=True, slots=True)
class ExternalOrderAcknowledgement:
    external_order_id: str
    accepted: bool
    observed_at: datetime
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class ExternalFill:
    external_fill_id: str
    external_order_id: str
    instrument: str
    quantity: Decimal
    price: Decimal
    observed_at: datetime


@dataclass(frozen=True, slots=True)
class ExternalAccountSnapshot:
    account_state: AccountState
    positions: tuple[PositionState, ...] = ()
    open_orders: tuple[ExternalOrder, ...] = ()


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    reconciled_at: datetime
    differences: tuple[str, ...] = ()
    is_consistent: bool = True


@dataclass(frozen=True, slots=True)
class StrategyConfiguration:
    strategy_id: str
    is_active: bool
    parameters: Mapping[str, object] = field(default_factory=dict)
