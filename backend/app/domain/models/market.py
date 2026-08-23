"""Market and analysis domain models."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class RawMarketData:
    source: str
    instrument: str
    observed_at: datetime
    payload: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    instrument: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal | None = None
    is_valid: bool = True

    @property
    def price(self) -> Decimal:
        return self.close


@dataclass(frozen=True, slots=True)
class FeatureSet:
    instrument: str
    timestamp: datetime
    feature_version: str
    values: Mapping[str, Decimal] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RegimeState:
    instrument: str
    timestamp: datetime
    regime: str
    trend: str
    volatility: str
    confidence: Decimal | None = None
    detector_version: str | None = None


@dataclass(frozen=True, slots=True)
class CandleBar:
    """Normalized OHLCV bar used by historical loaders and the backtester."""

    instrument: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
