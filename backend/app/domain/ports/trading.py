"""Decision, execution, and account ports."""

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from app.domain.models.state import (
    AccountState,
    ExternalAccountSnapshot,
    ExternalFill,
    ExternalOrder,
    ExternalOrderAcknowledgement,
    PortfolioState,
)
from app.domain.models.trading import (
    ExecutionReport,
    OrderIntent,
    RiskDecision,
    TargetPosition,
    TradingSignal,
)


class PortfolioManager(Protocol):
    def aggregate(
        self, signals: Sequence[TradingSignal], portfolio_state: PortfolioState
    ) -> Sequence[TargetPosition]: ...


class RiskManager(Protocol):
    def evaluate(
        self,
        target: TargetPosition,
        portfolio_state: PortfolioState,
        account_state: AccountState,
    ) -> RiskDecision: ...


class OrderPlanner(Protocol):
    def plan(
        self,
        target: TargetPosition,
        risk_decision: RiskDecision,
        portfolio_state: PortfolioState,
    ) -> Sequence[OrderIntent]: ...


class TradingEngine(Protocol):
    async def execute(self, intent: OrderIntent) -> ExecutionReport: ...

    async def cancel(self, order_id: str) -> ExecutionReport: ...


class ExecutionPort(Protocol):
    async def submit(self, intent: OrderIntent) -> ExternalOrderAcknowledgement: ...

    async def cancel(self, order_id: str) -> ExternalOrderAcknowledgement: ...

    async def get_open_orders(self) -> Sequence[ExternalOrder]: ...

    async def get_fills(self, since: datetime) -> Sequence[ExternalFill]: ...


class AccountPort(Protocol):
    async def get_account_snapshot(self) -> ExternalAccountSnapshot: ...
