"""Decision audit trail. Parquet row groups; a null sink for tests."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import fields
from datetime import datetime
from enum import Enum
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from scout.domain.audit import DecisionRecord

_RECORD_FIELDS = tuple(f.name for f in fields(DecisionRecord))


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
        table = pa.Table.from_pandas(frame, preserve_index=False)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self._path, table.schema)
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
        else:
            out[name] = value
    return out
