"""Smoke tests for the backtest vertical slice."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from app.domain.models.market import CandleBar
from app.infrastructure.configuration.settings import BacktestConfig
from app.modules.backtest.engine import BacktestEngine
from app.modules.data_sources.parquet_candles import load_candles_parquet, save_candles_parquet
from app.modules.features.ema_features import EmaFeatureEngine
from app.domain.models.market import MarketSnapshot


def _synthetic_candles(n: int = 80) -> list[CandleBar]:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles: list[CandleBar] = []
    price = Decimal("40000")
    for i in range(n):
        # Gentle up then down to trigger crosses
        if i < n // 2:
            price += Decimal("50")
        else:
            price -= Decimal("40")
        ts = start + timedelta(hours=i)
        candles.append(
            CandleBar(
                instrument="BTC/USDT",
                timestamp=ts,
                open=price,
                high=price + Decimal("10"),
                low=price - Decimal("10"),
                close=price,
                volume=Decimal("1"),
            )
        )
    return candles


def test_ema_features_produce_spread() -> None:
    engine = EmaFeatureEngine(fast_period=3, slow_period=5)
    candles = _synthetic_candles(20)
    last = None
    for c in candles:
        snap = MarketSnapshot(
            instrument=c.instrument,
            timestamp=c.timestamp,
            open=c.open,
            high=c.high,
            low=c.low,
            close=c.close,
            volume=c.volume,
        )
        last = engine.calculate(snap)
    assert last is not None
    assert "ema_fast" in last.values
    assert "ema_slow" in last.values
    assert "ema_spread" in last.values


def test_parquet_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "bars.parquet"
    candles = _synthetic_candles(10)
    save_candles_parquet(candles, path)
    loaded = load_candles_parquet(path, instrument="BTC/USDT")
    assert len(loaded) == 10
    assert loaded[0].instrument == "BTC/USDT"


def test_backtest_engine_runs() -> None:
    candles = _synthetic_candles(80)
    config = BacktestConfig(
        instrument="BTC/USDT",
        timeframe="1h",
        account_id="test",
        data_path=Path("unused.parquet"),
        results_dir=Path("results"),
        initial_equity=Decimal("10000"),
        fee_bps=Decimal("10"),
        slippage_bps=Decimal("5"),
        strategy_id="ema_cross",
        fast_period=5,
        slow_period=10,
        target_fraction=Decimal("0.95"),
        max_position_fraction=Decimal("0.95"),
        allow_short=False,
        warm_up_bars=15,
    )
    result = BacktestEngine(config).run(candles)
    assert result.metrics["bars"] > 0
    assert result.equity_curve
