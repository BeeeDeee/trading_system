"""ParquetCandleSource: the in-loop reader of data/processed/panel/<tf>/<year>.parquet."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.dataset as ds  # type: ignore[import-untyped]

from scout.domain.market import MARKET_COLUMNS, MarketPanel
from scout.utils.errors import ScoutDataError


class ParquetCandleSource:
    """Reads the processed panel. Constructor-injected; never constructed inside the loop."""

    def __init__(self, processed_dir: str | Path) -> None:
        self._processed_dir = Path(processed_dir)

    def load_panel(
        self,
        symbols: Sequence[str],
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> MarketPanel:
        start_utc = _require_aware_utc(start, "start")
        end_utc = _require_aware_utc(end, "end")
        asset_ids = _unique_ids(symbols)
        if not asset_ids or end_utc < start_utc:
            return MarketPanel(_empty_market_frame())
        panel_dir = self._panel_dir(timeframe)
        paths = _year_paths(panel_dir, start_utc, end_utc)
        if not paths:
            return MarketPanel(_empty_market_frame())
        table = _read_panel_table(paths, asset_ids)
        if table.num_rows == 0:
            return MarketPanel(_empty_market_frame())
        table = _filter_ts(table, start_utc, end_utc)
        if table.num_rows == 0:
            return MarketPanel(_empty_market_frame())
        frame = table.to_pandas(ignore_metadata=True)
        return MarketPanel(frame)

    def available_range(
        self, symbol: str, timeframe: str
    ) -> tuple[datetime, datetime] | None:
        panel_dir = self._processed_dir / "panel" / timeframe
        if not panel_dir.is_dir():
            return None
        paths = _all_year_paths(panel_dir)
        if not paths:
            return None
        table = _read_range_table(paths, symbol)
        if table.num_rows == 0:
            return None
        ts = table.column("ts")
        first_py = pc.min(ts).as_py()
        last_py = pc.max(ts).as_py()
        if first_py is None or last_py is None:
            return None
        return _to_utc_datetime(first_py), _to_utc_datetime(last_py)

    def _panel_dir(self, timeframe: str) -> Path:
        panel_dir = self._processed_dir / "panel" / timeframe
        if not panel_dir.is_dir():
            raise ScoutDataError(
                f"processed panel not found for timeframe {timeframe!r}: {panel_dir}"
            )
        return panel_dir


def _require_aware_utc(value: datetime, name: str) -> datetime:
    if value.tzinfo is None:
        raise ScoutDataError(f"{name} must be timezone-aware UTC")
    return value.astimezone(UTC)


def _unique_ids(symbols: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in symbols:
        asset_id = str(raw)
        if asset_id in seen:
            continue
        seen.add(asset_id)
        out.append(asset_id)
    return out


def _year_paths(panel_dir: Path, start: datetime, end: datetime) -> list[Path]:
    y0 = start.year
    y1 = end.year
    paths: list[Path] = []
    for year in range(y0, y1 + 1):
        path = panel_dir / f"{year}.parquet"
        if path.is_file():
            paths.append(path)
    return paths


def _all_year_paths(panel_dir: Path) -> list[Path]:
    paths = [
        path
        for path in panel_dir.glob("*.parquet")
        if path.stem.isdigit() and path.is_file()
    ]
    return sorted(paths, key=lambda path: int(path.stem))


def _read_panel_table(paths: Sequence[Path], asset_ids: Sequence[str]) -> pa.Table:
    dataset = ds.dataset([str(p) for p in paths], format="parquet")
    filt = ds.field("asset_id").isin(list(asset_ids))
    try:
        return dataset.to_table(columns=list(MARKET_COLUMNS), filter=filt)
    except (OSError, pa.ArrowInvalid, pa.ArrowTypeError, KeyError) as exc:
        joined = ", ".join(str(p) for p in paths)
        raise ScoutDataError(f"failed reading processed panel files [{joined}]: {exc}") from exc


def _read_range_table(paths: Sequence[Path], asset_id: str) -> pa.Table:
    dataset = ds.dataset([str(p) for p in paths], format="parquet")
    filt = ds.field("asset_id") == asset_id
    try:
        return dataset.to_table(columns=["ts"], filter=filt)
    except (OSError, pa.ArrowInvalid, pa.ArrowTypeError, KeyError) as exc:
        joined = ", ".join(str(p) for p in paths)
        raise ScoutDataError(
            f"failed reading processed panel files for available_range [{joined}]: {exc}"
        ) from exc


def _filter_ts(table: pa.Table, start: datetime, end: datetime) -> pa.Table:
    ts = table.column("ts")
    try:
        start_s = pa.scalar(pd.Timestamp(start), type=ts.type)
        end_s = pa.scalar(pd.Timestamp(end), type=ts.type)
        mask = pc.and_(pc.greater_equal(ts, start_s), pc.less_equal(ts, end_s))
    except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError, ValueError) as exc:
        raise ScoutDataError(f"failed applying [start, end] filter on ts: {exc}") from exc
    return table.filter(mask)


def _to_utc_datetime(value: object) -> datetime:
    ts = pd.Timestamp(value)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    out = ts.to_pydatetime()
    if not isinstance(out, datetime):
        raise ScoutDataError(f"ts scalar is not a datetime: {type(out)!r}")
    return out


def _empty_market_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "asset_id": pd.Series(dtype="object"),
            "symbol": pd.Series(dtype="object"),
            "ts": pd.Series(dtype="datetime64[ns, UTC]"),
            "session_index": pd.Series(dtype="int32"),
            "open": pd.Series(dtype="float64"),
            "high": pd.Series(dtype="float64"),
            "low": pd.Series(dtype="float64"),
            "close": pd.Series(dtype="float64"),
            "close_raw": pd.Series(dtype="float64"),
            "volume": pd.Series(dtype="float64"),
            "dollar_volume": pd.Series(dtype="float64"),
            "is_suspect": pd.Series(dtype="bool"),
        }
    )
