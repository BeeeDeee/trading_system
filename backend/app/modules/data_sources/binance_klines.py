"""Public Binance kline download (no API key required)."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import httpx

from app.domain.models.market import CandleBar

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


def _to_binance_symbol(instrument: str) -> str:
    return instrument.replace("/", "").upper()


def download_binance_klines(
    instrument: str = "BTC/USDT",
    interval: str = "1h",
    start: datetime | None = None,
    end: datetime | None = None,
    limit_per_request: int = 1000,
) -> list[CandleBar]:
    """Download OHLCV candles from Binance public REST API."""
    symbol = _to_binance_symbol(instrument)
    start_ms = int((start or datetime(2019, 1, 1, tzinfo=timezone.utc)).timestamp() * 1000)
    end_ms = int((end or datetime.now(timezone.utc)).timestamp() * 1000)

    candles: list[CandleBar] = []
    cursor = start_ms

    with httpx.Client(timeout=30.0) as client:
        while cursor < end_ms:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": limit_per_request,
            }
            response = client.get(BINANCE_KLINES_URL, params=params)
            response.raise_for_status()
            batch = response.json()
            if not batch:
                break
            for item in batch:
                open_time_ms = int(item[0])
                ts = datetime.fromtimestamp(open_time_ms / 1000, tz=timezone.utc)
                candles.append(
                    CandleBar(
                        instrument=instrument,
                        timestamp=ts,
                        open=Decimal(str(item[1])),
                        high=Decimal(str(item[2])),
                        low=Decimal(str(item[3])),
                        close=Decimal(str(item[4])),
                        volume=Decimal(str(item[5])),
                    )
                )
            last_open = int(batch[-1][0])
            next_cursor = last_open + 1
            if next_cursor <= cursor:
                break
            cursor = next_cursor
            if len(batch) < limit_per_request:
                break

    # Deduplicate by timestamp
    by_ts = {c.timestamp: c for c in candles}
    return [by_ts[k] for k in sorted(by_ts)]
