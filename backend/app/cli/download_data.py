"""CLI: download BTC/USDT 1h candles to Parquet."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from app.infrastructure.configuration.settings import load_backtest_config, repo_root
from app.modules.data_sources.binance_klines import download_binance_klines
from app.modules.data_sources.parquet_candles import save_candles_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description="Download OHLCV candles for backtests")
    parser.add_argument("--instrument", default="BTC/USDT")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--start", default="2019-01-01", help="UTC date YYYY-MM-DD")
    parser.add_argument(
        "--out",
        default=None,
        help="Output parquet path (default: config/backtest.yaml data_path)",
    )
    args = parser.parse_args()

    config = load_backtest_config()
    out = Path(args.out) if args.out else config.data_path
    if not out.is_absolute():
        out = repo_root() / out

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    print(f"Downloading {args.instrument} {args.interval} from {start.date()} ...")
    candles = download_binance_klines(
        instrument=args.instrument, interval=args.interval, start=start
    )
    save_candles_parquet(candles, out)
    print(f"Wrote {len(candles)} bars -> {out}")


if __name__ == "__main__":
    main()
