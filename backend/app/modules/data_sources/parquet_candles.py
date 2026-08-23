"""Historical candle load/save helpers (Parquet)."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pandas as pd

from app.domain.models.market import CandleBar


REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")


def candles_to_frame(candles: list[CandleBar]) -> pd.DataFrame:
    rows = [
        {
            "timestamp": c.timestamp,
            "open": float(c.open),
            "high": float(c.high),
            "low": float(c.low),
            "close": float(c.close),
            "volume": float(c.volume),
            "instrument": c.instrument,
        }
        for c in candles
    ]
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        frame = frame.reset_index(drop=True)
    return frame


def save_candles_parquet(candles: list[CandleBar], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = candles_to_frame(candles)
    frame.to_parquet(path, index=False)
    return path


def load_candles_parquet(path: Path, instrument: str | None = None) -> list[CandleBar]:
    if not path.exists():
        raise FileNotFoundError(
            f"Candle file not found: {path}. Run: python -m app.cli.download_data"
        )
    frame = pd.read_parquet(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Parquet missing columns {missing}: {path}")

    frame = frame.sort_values("timestamp").reset_index(drop=True)
    candles: list[CandleBar] = []
    for row in frame.itertuples(index=False):
        ts = row.timestamp
        if isinstance(ts, pd.Timestamp):
            ts = ts.to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        else:
            ts = ts.astimezone(timezone.utc)
        inst = instrument or getattr(row, "instrument", None) or "BTC/USDT"
        candles.append(
            CandleBar(
                instrument=str(inst),
                timestamp=ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts)),
                open=Decimal(str(row.open)),
                high=Decimal(str(row.high)),
                low=Decimal(str(row.low)),
                close=Decimal(str(row.close)),
                volume=Decimal(str(row.volume)),
            )
        )
    return candles
