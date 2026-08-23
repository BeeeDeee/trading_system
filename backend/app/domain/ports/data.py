"""Market-data and analysis ports."""

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from app.domain.models.market import FeatureSet, MarketSnapshot, RawMarketData
from app.domain.models.state import PortfolioState
from app.domain.models.trading import TradingSignal


class MarketDataPort(Protocol):
    async def get_snapshot(self, instrument: str, timestamp: datetime) -> RawMarketData: ...


class Normalizer(Protocol):
    def normalize(self, raw_data: RawMarketData) -> MarketSnapshot: ...


class FeatureEngine(Protocol):
    def calculate(self, snapshot: MarketSnapshot) -> FeatureSet: ...


class Strategy(Protocol):
    def evaluate(
        self,
        snapshot: MarketSnapshot,
        features: FeatureSet,
        portfolio: PortfolioState,
    ) -> Sequence[TradingSignal]: ...
