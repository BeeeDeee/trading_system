"""Backtest configuration loading."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    """Repository root (parent of backend/)."""
    return Path(__file__).resolve().parents[4]


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    instrument: str
    timeframe: str
    account_id: str
    data_path: Path
    results_dir: Path
    initial_equity: Decimal
    fee_bps: Decimal
    slippage_bps: Decimal
    strategy_id: str
    fast_period: int
    slow_period: int
    target_fraction: Decimal
    max_position_fraction: Decimal
    allow_short: bool
    warm_up_bars: int


def load_backtest_config(path: Path | None = None) -> BacktestConfig:
    root = repo_root()
    config_path = path or (root / "config" / "backtest.yaml")
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    strategy = raw.get("strategy") or {}
    risk = raw.get("risk") or {}

    data_path = Path(raw["data_path"])
    if not data_path.is_absolute():
        data_path = root / data_path
    results_dir = Path(raw["results_dir"])
    if not results_dir.is_absolute():
        results_dir = root / results_dir

    return BacktestConfig(
        instrument=str(raw["instrument"]),
        timeframe=str(raw["timeframe"]),
        account_id=str(raw.get("account_id", "local-dev")),
        data_path=data_path,
        results_dir=results_dir,
        initial_equity=Decimal(str(raw["initial_equity"])),
        fee_bps=Decimal(str(raw.get("fee_bps", "10"))),
        slippage_bps=Decimal(str(raw.get("slippage_bps", "5"))),
        strategy_id=str(strategy.get("id", "ema_cross")),
        fast_period=int(strategy.get("fast_period", 12)),
        slow_period=int(strategy.get("slow_period", 26)),
        target_fraction=Decimal(str(strategy.get("target_fraction", "0.95"))),
        max_position_fraction=Decimal(str(risk.get("max_position_fraction", "0.95"))),
        allow_short=bool(risk.get("allow_short", False)),
        warm_up_bars=int(raw.get("warm_up_bars", 30)),
    )
