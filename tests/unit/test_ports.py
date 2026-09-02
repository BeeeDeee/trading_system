"""Runtime Protocol checks. Concrete classes must not subclass the Protocol."""

from pathlib import Path

from scout.backtest.sim_broker import SimBroker
from scout.config.schema import CostsConfig, PortfolioConfig
from scout.data.parquet_source import ParquetCandleSource
from scout.domain.ports import (
    Broker,
    CandleSource,
    DecisionSink,
    SentimentSource,
    Strategy,
)
from scout.sentiment.null_source import NullSentimentSource
from scout.storage.decision_sink import NullDecisionSink, ParquetDecisionSink
from scout.strategies.donchian_breakout import DonchianBreakout
from scout.strategies.xsec_momentum import XSecMomentum


def test_parquet_candle_source_satisfies_protocol(tmp_path: Path) -> None:
    source = ParquetCandleSource(tmp_path)
    assert isinstance(source, CandleSource)
    # Structural typing only — implementations must not inherit the Protocol.
    assert CandleSource not in ParquetCandleSource.__mro__


def test_xsec_momentum_satisfies_protocol() -> None:
    assert isinstance(XSecMomentum(), Strategy)
    assert Strategy not in XSecMomentum.__mro__


def test_donchian_breakout_satisfies_protocol() -> None:
    assert isinstance(DonchianBreakout(), Strategy)
    assert Strategy not in DonchianBreakout.__mro__


def test_sim_broker_satisfies_protocol() -> None:
    broker = SimBroker(CostsConfig(), PortfolioConfig())
    assert isinstance(broker, Broker)
    assert Broker not in SimBroker.__mro__


def test_null_sentiment_source_satisfies_protocol() -> None:
    source = NullSentimentSource()
    assert isinstance(source, SentimentSource)
    assert SentimentSource not in NullSentimentSource.__mro__


def test_null_decision_sink_satisfies_protocol() -> None:
    sink = NullDecisionSink()
    assert isinstance(sink, DecisionSink)
    assert DecisionSink not in NullDecisionSink.__mro__


def test_parquet_decision_sink_satisfies_protocol(tmp_path: Path) -> None:
    sink = ParquetDecisionSink(tmp_path / "decisions.parquet")
    assert isinstance(sink, DecisionSink)
    assert DecisionSink not in ParquetDecisionSink.__mro__

