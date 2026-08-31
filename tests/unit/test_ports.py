"""Runtime Protocol checks. Concrete classes must not subclass the Protocol."""

from pathlib import Path

from scout.data.parquet_source import ParquetCandleSource
from scout.domain.ports import CandleSource, Strategy
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
