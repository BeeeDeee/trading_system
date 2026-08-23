"""Bar-driven backtest engine sharing the live decision pipeline modules."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pandas as pd

from app.domain.models.market import CandleBar, MarketSnapshot
from app.domain.models.state import AccountState, PortfolioState, PositionState
from app.domain.models.trading import ExecutionReport
from app.infrastructure.configuration.settings import BacktestConfig
from app.modules.backtest.simulated_execution import SimulatedExecution
from app.modules.features.ema_features import EmaFeatureEngine
from app.modules.order_planning.simple import SimpleOrderPlanner
from app.modules.portfolio.simple import SimplePortfolioManager
from app.modules.risk.simple import SimpleRiskManager
from app.modules.strategies.ema_cross import EmaCrossStrategy


@dataclass
class BacktestResult:
    run_id: str
    trades: list[dict] = field(default_factory=list)
    equity_curve: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    output_dir: Path | None = None


def _position_qty(portfolio: PortfolioState, instrument: str) -> Decimal:
    for pos in portfolio.positions:
        if pos.instrument == instrument:
            return pos.quantity
    return Decimal(0)


def _with_position(
    portfolio: PortfolioState, instrument: str, quantity: Decimal, ts: datetime
) -> PortfolioState:
    others = tuple(p for p in portfolio.positions if p.instrument != instrument)
    positions = others
    if quantity != 0:
        positions = others + (PositionState(instrument=instrument, quantity=quantity, updated_at=ts),)
    return PortfolioState(
        equity=portfolio.equity,
        positions=positions,
        realized_pnl=portfolio.realized_pnl,
        unrealized_pnl=portfolio.unrealized_pnl,
    )


class BacktestEngine:
    def __init__(self, config: BacktestConfig) -> None:
        self.config = config
        self.features = EmaFeatureEngine(config.fast_period, config.slow_period)
        self.strategy = EmaCrossStrategy(config.strategy_id, config.target_fraction)
        self.portfolio_mgr = SimplePortfolioManager()
        self.risk = SimpleRiskManager(
            max_position_fraction=config.max_position_fraction,
            allow_short=config.allow_short,
        )
        self.planner = SimpleOrderPlanner(account_id=config.account_id)
        self.execution = SimulatedExecution(config.fee_bps, config.slippage_bps)

    def run(self, candles: list[CandleBar]) -> BacktestResult:
        if len(candles) < self.config.warm_up_bars + 2:
            raise ValueError("Not enough candles for warm-up and execution")

        run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-ema-cross")
        equity = self.config.initial_equity
        cash = equity
        instrument = self.config.instrument
        portfolio = PortfolioState(equity=equity, positions=())
        avg_entry: Decimal | None = None
        realized = Decimal(0)

        trades: list[dict] = []
        equity_curve: list[dict] = []

        # Warm-up features without trading
        for bar in candles[: self.config.warm_up_bars]:
            snap = _bar_to_snapshot(bar)
            self.features.calculate(snap)

        # Decision on bar i close; fill at bar i+1 open (no lookahead)
        for i in range(self.config.warm_up_bars, len(candles) - 1):
            bar = candles[i]
            next_bar = candles[i + 1]
            snapshot = _bar_to_snapshot(bar)
            features = self.features.calculate(snapshot)

            mark = snapshot.close
            qty = _position_qty(portfolio, instrument)
            unrealized = Decimal(0)
            if qty != 0 and avg_entry is not None:
                unrealized = (mark - avg_entry) * qty
            equity = cash + (qty * mark)
            portfolio = PortfolioState(
                equity=equity,
                positions=portfolio.positions,
                realized_pnl=realized,
                unrealized_pnl=unrealized,
            )

            signals = self.strategy.evaluate(snapshot, features, portfolio)
            self.portfolio_mgr.mark_price = mark
            targets = self.portfolio_mgr.aggregate(list(signals), portfolio)

            fill_reports: list[ExecutionReport] = []
            for target in targets:
                self.risk.mark_price = mark
                account = AccountState(
                    equity=equity, available_capital=cash, updated_at=bar.timestamp
                )
                decision = self.risk.evaluate(target, portfolio, account)
                decision = replace(decision, timestamp=bar.timestamp)
                intents = self.planner.plan(target, decision, portfolio)
                for intent in intents:
                    report = self.execution.fill(intent, next_bar.open, next_bar.timestamp)
                    fill_reports.append(report)

                    fill_qty = report.filled_quantity
                    fill_px = report.average_price or next_bar.open
                    fee = report.fee or Decimal(0)
                    signed = fill_qty if intent.side.value == "buy" else -fill_qty

                    # Update cash / position / realized PnL
                    prev_qty = _position_qty(portfolio, instrument)
                    new_qty = prev_qty + signed
                    if intent.side.value == "buy":
                        cash -= fill_qty * fill_px + fee
                        if prev_qty == 0:
                            avg_entry = fill_px
                        elif prev_qty > 0:
                            avg_entry = (
                                ((avg_entry or fill_px) * prev_qty) + (fill_px * fill_qty)
                            ) / new_qty
                    else:
                        cash += fill_qty * fill_px - fee
                        if prev_qty > 0 and avg_entry is not None:
                            realized += (fill_px - avg_entry) * fill_qty
                        if new_qty == 0:
                            avg_entry = None

                    portfolio = _with_position(portfolio, instrument, new_qty, next_bar.timestamp)
                    trades.append(
                        {
                            "timestamp": next_bar.timestamp.isoformat(),
                            "side": intent.side.value,
                            "quantity": str(fill_qty),
                            "price": str(fill_px),
                            "fee": str(fee),
                            "client_order_id": intent.client_order_id,
                            "position_after": str(new_qty),
                        }
                    )

            qty = _position_qty(portfolio, instrument)
            mark_close = next_bar.close
            equity = cash + qty * mark_close
            equity_curve.append(
                {
                    "timestamp": next_bar.timestamp.isoformat(),
                    "equity": float(equity),
                    "cash": float(cash),
                    "position": float(qty),
                    "price": float(mark_close),
                }
            )

        metrics = _compute_metrics(equity_curve, self.config.initial_equity)
        return BacktestResult(
            run_id=run_id, trades=trades, equity_curve=equity_curve, metrics=metrics
        )


def _bar_to_snapshot(bar: CandleBar) -> MarketSnapshot:
    return MarketSnapshot(
        instrument=bar.instrument,
        timestamp=bar.timestamp,
        open=bar.open,
        high=bar.high,
        low=bar.low,
        close=bar.close,
        volume=bar.volume,
        is_valid=True,
    )


def _compute_metrics(equity_curve: list[dict], initial: Decimal) -> dict:
    if not equity_curve:
        return {"total_return": 0.0, "max_drawdown": 0.0, "bars": 0}
    eq = pd.Series([row["equity"] for row in equity_curve], dtype=float)
    total_return = float(eq.iloc[-1] / float(initial) - 1.0)
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return {
        "total_return": total_return,
        "max_drawdown": float(dd.min()),
        "final_equity": float(eq.iloc[-1]),
        "bars": int(len(eq)),
        "trades_hint": "see trades.csv",
    }


def write_backtest_outputs(result: BacktestResult, results_dir: Path, config: BacktestConfig) -> Path:
    out = results_dir / result.run_id
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result.trades).to_csv(out / "trades.csv", index=False)
    pd.DataFrame(result.equity_curve).to_csv(out / "equity.csv", index=False)
    (out / "metrics.json").write_text(
        json.dumps(result.metrics, indent=2) + "\n", encoding="utf-8"
    )
    # Minimal config dump
    (out / "config.yaml").write_text(
        f"instrument: {config.instrument}\n"
        f"timeframe: {config.timeframe}\n"
        f"strategy: {config.strategy_id}\n"
        f"initial_equity: {config.initial_equity}\n",
        encoding="utf-8",
    )
    result.output_dir = out
    return out
