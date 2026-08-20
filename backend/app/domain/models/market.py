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
    price: Decimal
    volume: Decimal | None = None
    is_valid: bool = True


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
