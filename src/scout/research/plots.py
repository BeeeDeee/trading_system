"""The six required run plots. ≥150 DPI, timezone-aware axes where time is plotted."""

from __future__ import annotations

from datetime import UTC
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from scout.domain.results import BacktestResult
from scout.research.diagnostics import calibration_buckets


def write_all_plots(plots_dir: Path, result: BacktestResult, *, dpi: int) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    equity = _as_utc_series(result.equity_curve, "equity")
    bench = _as_utc_series(result.benchmark_curve, "benchmark")
    if equity.empty:
        equity = pd.Series(
            [1.0], index=pd.DatetimeIndex([result.period_start], tz="UTC")
        )
    eq_norm = _normalise(equity)
    bench_norm = eq_norm if bench.empty else _normalise(bench.reindex(eq_norm.index).ffill())
    peak = eq_norm.cummax()
    dd = (eq_norm / peak) - 1.0
    _equity_png(plots_dir / "equity.png", eq_norm, bench_norm, dpi=dpi)
    _drawdown_png(plots_dir / "drawdown.png", dd, dpi=dpi)
    _summary_png(plots_dir / "summary.png", eq_norm, dd, dpi=dpi)
    _calibration_png(plots_dir / "calibration.png", result, dpi=dpi)
    _monthly_png(plots_dir / "monthly_returns.png", eq_norm, dpi=dpi)
    _regime_png(plots_dir / "regime_breakdown.png", result, dpi=dpi)


def apply_utc_xaxis(ax: Any) -> None:
    ax.xaxis_date(tz=UTC)
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(mdates.AutoDateLocator(tz=UTC))  # type: ignore[no-untyped-call]
    )


def _equity_png(
    path: Path, eq_norm: pd.Series, bench_norm: pd.Series, *, dpi: int
) -> None:
    final_pct = float(eq_norm.iloc[-1] / eq_norm.iloc[0] - 1.0) * 100.0
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(eq_norm.index, eq_norm.to_numpy(), label="strategy")
    ax.plot(bench_norm.index, bench_norm.to_numpy(), label="SPY")
    ax.legend()
    ax.set_title(f"Equity  final {final_pct:.1f}%")
    ax.set_ylabel("growth (start=1)")
    apply_utc_xaxis(ax)
    fig.autofmt_xdate()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _drawdown_png(path: Path, dd: pd.Series, *, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(dd.index, dd.to_numpy(), 0.0, color="C3", alpha=0.4)
    ax.plot(dd.index, dd.to_numpy(), color="C3")
    max_dd = float(dd.min()) if len(dd) else 0.0
    max_dd_ts = dd.idxmin() if len(dd) else None
    ax.set_title(f"Drawdown  max {max_dd:.1%} at {max_dd_ts}")
    apply_utc_xaxis(ax)
    fig.autofmt_xdate()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _summary_png(path: Path, eq_norm: pd.Series, dd: pd.Series, *, dpi: int) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(eq_norm.index, eq_norm.to_numpy())
    axes[0].set_ylabel("equity")
    apply_utc_xaxis(axes[0])
    axes[1].fill_between(dd.index, dd.to_numpy(), 0.0, color="C3", alpha=0.4)
    axes[1].set_ylabel("drawdown")
    apply_utc_xaxis(axes[1])
    fig.autofmt_xdate()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _calibration_png(path: Path, result: BacktestResult, *, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    if result.trades:
        trades = pd.DataFrame(
            {
                "ev_net_r_at_entry": [t.ev_net_r_at_entry for t in result.trades],
                "realised_r": [t.realised_r for t in result.trades],
            }
        )
        buckets = calibration_buckets(trades)
        if not buckets.empty:
            yerr = np.vstack(
                [
                    np.maximum(buckets["mean_realised_r"] - buckets["ci_low"], 0.0),
                    np.maximum(buckets["ci_high"] - buckets["mean_realised_r"], 0.0),
                ]
            )
            ax.errorbar(
                buckets["predicted_mid"],
                buckets["mean_realised_r"],
                yerr=yerr,
                fmt="o",
                capsize=3,
            )
        ax.scatter(
            trades["ev_net_r_at_entry"],
            trades["realised_r"],
            s=12,
            alpha=0.35,
            color="grey",
        )
    ax.axhline(0.0, color="grey", linewidth=0.8)
    ax.axvline(0.0, color="grey", linewidth=0.8)
    ax.set_xlabel("predicted ev_net_r")
    ax.set_ylabel("realised R")
    ax.set_title("calibration")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _monthly_png(path: Path, eq_norm: pd.Series, *, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    monthly = eq_norm.resample("ME").last().pct_change().dropna()
    if monthly.empty:
        ax.set_title("monthly returns (empty)")
    else:
        grid = pd.DataFrame(
            {
                "year": monthly.index.year,
                "month": monthly.index.month,
                "r": monthly.to_numpy(),
            }
        )
        pivot = grid.pivot_table(index="year", columns="month", values="r", aggfunc="sum")
        ax.imshow(pivot.to_numpy(), aspect="auto", cmap="RdYlGn")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(y) for y in pivot.index])
        ax.set_xticks(range(12))
        ax.set_xticklabels([str(i) for i in range(1, 13)])
        ax.set_title("monthly returns")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _regime_png(path: Path, result: BacktestResult, *, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    if result.trades:
        _bar_means(
            axes[0],
            [t.regime.value for t in result.trades],
            [t.realised_r for t in result.trades],
            "regime",
        )
        _bar_means(
            axes[1],
            [t.vol_bucket.value for t in result.trades],
            [t.realised_r for t in result.trades],
            "vol bucket",
        )
    else:
        axes[0].set_title("regime")
        axes[1].set_title("vol bucket")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _bar_means(ax: Any, labels: list[str], values: list[float], title: str) -> None:
    grouped: dict[str, list[float]] = {}
    for lab, val in zip(labels, values, strict=True):
        grouped.setdefault(lab, []).append(val)
    keys = sorted(grouped)
    means = [sum(grouped[k]) / len(grouped[k]) for k in keys]
    ax.bar(keys, means)
    ax.set_ylabel("mean realised R")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)


def _as_utc_series(series: pd.Series, name: str) -> pd.Series:
    if series.empty:
        return pd.Series(dtype="float64", name=name)
    out = series.copy()
    out.name = name
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index, utc=True))
    return out.sort_index()


def _normalise(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    first = float(series.iloc[0])
    if first == 0.0 or not np.isfinite(first):
        return series
    return series / first
