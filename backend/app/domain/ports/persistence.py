"""Persistence repository ports."""

from collections.abc import Sequence
from typing import Protocol

from app.domain.models.state import OrderState, PositionState, StrategyConfiguration
from app.domain.models.trading import ExecutionReport


class OrderRepository(Protocol):
    async def save(self, order_state: OrderState) -> None: ...

    async def get(self, order_id: str) -> OrderState | None: ...


class PositionRepository(Protocol):
    async def save(self, position_state: PositionState) -> None: ...

    async def get_all(self) -> Sequence[PositionState]: ...


class TradeRepository(Protocol):
    async def save(self, report: ExecutionReport) -> None: ...


class StrategyRepository(Protocol):
    async def get_active(self) -> Sequence[StrategyConfiguration]: ...
