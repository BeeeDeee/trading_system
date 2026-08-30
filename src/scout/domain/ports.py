"""The five substitutable Protocols. Implementations do not subclass these."""

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol, runtime_checkable

from scout.domain.audit import DecisionRecord
from scout.domain.enums import Regime
from scout.domain.execution import Fill, OrderIntent
from scout.domain.features import FeatureRow
from scout.domain.market import MarketPanel
from scout.domain.portfolio import PortfolioState
from scout.domain.sentiment import SentimentObservation
from scout.domain.setup import Setup


@runtime_checkable
class CandleSource(Protocol):
    """Supplies closed OHLCV bars. The only source of price truth."""

    def load_panel(
        self,
        symbols: Sequence[str],
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> MarketPanel:
        """Return all CLOSED bars in [start, end] for the given symbols."""

    def available_range(
        self, symbol: str, timeframe: str
    ) -> tuple[datetime, datetime] | None:
        """First and last available bar close times, or None if unknown."""


@runtime_checkable
class Strategy(Protocol):
    """Pure setup detection. Sees one symbol's features at one instant."""

    @property
    def strategy_id(self) -> str: ...

    @property
    def allowed_regimes(self) -> frozenset[Regime]: ...

    @property
    def required_warmup_bars(self) -> int: ...

    def detect(self, row: FeatureRow) -> Setup | None: ...


@runtime_checkable
class SentimentSource(Protocol):
    """Supplies point-in-time sentiment observations."""

    @property
    def source_id(self) -> str: ...

    def observations(
        self,
        symbols: Sequence[str],
        start: datetime,
        end: datetime,
    ) -> tuple[SentimentObservation, ...]: ...


@runtime_checkable
class Broker(Protocol):
    """Order placement and account state. The only component that talks to an exchange."""

    def submit_bracket(
        self,
        entry: OrderIntent,
        stop: OrderIntent,
        target: OrderIntent,
    ) -> tuple[Fill | None, str]: ...

    def close_position(self, symbol: str, reason: str) -> Fill | None: ...

    def portfolio_state(self, ts: datetime) -> PortfolioState: ...

    def poll_fills(self, ts: datetime) -> tuple[Fill, ...]: ...


@runtime_checkable
class DecisionSink(Protocol):
    """Persists the audit trail. Narrow on purpose."""

    def write(self, records: Sequence[DecisionRecord]) -> None: ...

    def flush(self) -> None: ...
