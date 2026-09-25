"""Decision audit trail. Parquet row groups; a null sink for tests."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import fields
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from scout.domain.audit import DecisionRecord

_RECORD_FIELDS = tuple(f.name for f in fields(DecisionRecord))

# Fixed schema so a flush of all-null optionals does not lock the file to
# Arrow null types that later accepted rows cannot write.
_SCHEMA = pa.schema(
    [
        ("run_id", pa.string()),
        ("decision_ts", pa.timestamp("ns", tz="UTC")),
        ("symbol", pa.string()),
        ("strategy_id", pa.string()),
        ("stage", pa.string()),
        ("accepted", pa.bool_()),
        ("rejection_reason", pa.string()),
        ("direction", pa.string()),
        ("regime", pa.string()),
        ("vol_bucket", pa.string()),
        ("reference_price", pa.float64()),
        ("stop_price", pa.float64()),
        ("target_price", pa.float64()),
        ("reward_risk_ratio", pa.float64()),
        ("bin_key", pa.string()),
        ("bin_n", pa.float64()),
        ("ev_r_point", pa.float64()),
        ("ev_r_lcb", pa.float64()),
        ("cost_r", pa.float64()),
        ("ev_net_r", pa.float64()),
        ("ev_per_bar_r", pa.float64()),
        ("rank", pa.float64()),
        ("sentiment_score", pa.float64()),
        ("sentiment_confidence", pa.float64()),
        ("sentiment_multiplier", pa.float64()),
        ("portfolio_heat_pct", pa.float64()),
        ("cluster", pa.string()),
        ("size_multiplier", pa.float64()),
        ("qty", pa.float64()),
        ("notional_usd", pa.float64()),
        ("adv_usd_30", pa.float64()),
        ("spread_bps_est", pa.float64()),
        ("features_json", pa.string()),
    ]
)


class NullDecisionSink:
    """In-memory sink. Tests inspect `.records` after flush."""

    def __init__(self) -> None:
        self.records: list[DecisionRecord] = []
        self._buffer: list[DecisionRecord] = []

    def write(self, records: Sequence[DecisionRecord]) -> None:
        self._buffer.extend(records)

    def flush(self) -> None:
        self.records.extend(self._buffer)
        self._buffer.clear()


class ParquetDecisionSink:
    """Buffers DecisionRecords and writes row groups every `flush_every` cycles."""

    def __init__(self, path: str | Path, *, flush_every: int = 500) -> None:
        self._path = Path(path)
        self._flush_every = max(1, flush_every)
        self._buffer: list[DecisionRecord] = []
        self._cycles = 0
        self._writer: pq.ParquetWriter | None = None

    def write(self, records: Sequence[DecisionRecord]) -> None:
        self._buffer.extend(records)
        self._cycles += 1
        if self._cycles >= self._flush_every:
            self.flush()

    def flush(self) -> None:
        self._cycles = 0
        if not self._buffer:
            return
        frame = _records_to_frame(self._buffer)
        self._buffer.clear()
        table = pa.Table.from_pandas(frame, schema=_SCHEMA, preserve_index=False)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self._path, _SCHEMA)
        self._writer.write_table(table)

    def close(self) -> None:
        self.flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None


def _records_to_frame(records: Sequence[DecisionRecord]) -> pd.DataFrame:
    rows = [_record_to_row(rec) for rec in records]
    return pd.DataFrame(rows, columns=list(_RECORD_FIELDS))


def _record_to_row(rec: DecisionRecord) -> dict[str, object]:
    out: dict[str, object] = {}
    for name in _RECORD_FIELDS:
        value = getattr(rec, name)
        if isinstance(value, datetime):
            out[name] = value
        elif isinstance(value, Enum):
            out[name] = value.value
        elif isinstance(value, Decimal):
            out[name] = float(value)
        else:
            out[name] = value
    return out
