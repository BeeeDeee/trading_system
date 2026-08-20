"""Market-data and analysis ports."""

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from app.domain.models.market import FeatureSet, MarketSnapshot, RawMarketData, RegimeState
from app.domain.models.state import PortfolioState
from app.domain.models.trading import TradingSignal


class MarketDataPort(Protocol):
    async def get_snapshot(self, instrument: str, timestamp: datetime) -> RawMarketData: ...


class Normalizer(Protocol):
    def normalize(self, raw_data: RawMarketData) -> MarketSnapshot: ...


class FeatureEngine(Protocol):
    def calculate(self, snapshot: MarketSnapshot) -> FeatureSet: ...


class RegimeDetector(Protocol):
    def detect(self, snapshot: MarketSnapshot, features: FeatureSet) -> RegimeState: ...


class Strategy(Protocol):
    def evaluate(
        self,
        snapshot: MarketSnapshot,
        features: FeatureSet,
        regime: RegimeState,
        portfolio: PortfolioState,
    ) -> Sequence[TradingSignal]: ...
