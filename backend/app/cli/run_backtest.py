"""CLI: run the EMA-cross reference backtest."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.infrastructure.configuration.settings import load_backtest_config
from app.modules.backtest.engine import BacktestEngine, write_backtest_outputs
from app.modules.data_sources.parquet_candles import load_candles_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bar-driven backtest")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to backtest YAML (default: <repo>/config/backtest.yaml)",
    )
    args = parser.parse_args()
    config_path = Path(args.config) if args.config else None
    config = load_backtest_config(config_path)

    print(f"Loading candles from {config.data_path}")
    candles = load_candles_parquet(config.data_path, instrument=config.instrument)
    print(f"Bars: {len(candles)} | strategy={config.strategy_id}")

    engine = BacktestEngine(config)
    result = engine.run(candles)
    out = write_backtest_outputs(result, config.results_dir, config)

    print(f"Run id: {result.run_id}")
    print(f"Metrics: {result.metrics}")
    print(f"Outputs: {out}")


if __name__ == "__main__":
    main()
