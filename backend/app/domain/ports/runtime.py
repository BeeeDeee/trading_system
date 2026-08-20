"""Runtime and pipeline ports."""

from typing import Protocol

from app.domain.models.market import MarketSnapshot
from app.domain.models.runtime import PipelineResult, RuntimeCommand, RuntimeCommandResult


class TradingRuntime(Protocol):
    async def submit(self, command: RuntimeCommand) -> RuntimeCommandResult: ...


class TradingPipeline(Protocol):
    async def process(self, snapshot: MarketSnapshot) -> PipelineResult: ...
