"""Atomic Parquet and JSON writes. Temp file in the destination directory, then replace."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]


def _tmp_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def _cleanup(tmp: Path) -> None:
    if tmp.exists():
        tmp.unlink()


def write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    """Write `frame` to `path` so a kill mid-write cannot truncate the destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    try:
        table = pa.Table.from_pandas(frame, preserve_index=False)
        pq.write_table(table, tmp)
        os.replace(tmp, path)
    except BaseException:
        _cleanup(tmp)
        raise


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON with explicit UTF-8. Same atomic contract as Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    try:
        text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        _cleanup(tmp)
        raise


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def read_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return loaded
