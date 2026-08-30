"""Runtime Protocol checks. Concrete classes must not subclass the Protocol."""

from pathlib import Path

from scout.data.parquet_source import ParquetCandleSource
from scout.domain.ports import CandleSource


def test_parquet_candle_source_satisfies_protocol(tmp_path: Path) -> None:
    source = ParquetCandleSource(tmp_path)
    assert isinstance(source, CandleSource)
    # Structural typing only — implementations must not inherit the Protocol.
    assert CandleSource not in ParquetCandleSource.__mro__
