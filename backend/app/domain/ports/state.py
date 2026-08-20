"""State derivation and reconciliation ports."""

from collections.abc import Sequence
from typing import Protocol

from app.domain.models.state import (
    AccountState,
    ExternalAccountSnapshot,
    OrderState,
    PortfolioState,
    PositionState,
    ReconciliationResult,
)


class ReconciliationService(Protocol):
    def reconcile(
        self,
        local_orders: Sequence[OrderState],
        local_positions: Sequence[PositionState],
        external_snapshot: ExternalAccountSnapshot,
    ) -> ReconciliationResult: ...


class PortfolioView(Protocol):
    def derive(
        self,
        account_state: AccountState,
        position_state: Sequence[PositionState],
        order_state: Sequence[OrderState],
    ) -> PortfolioState: ...
