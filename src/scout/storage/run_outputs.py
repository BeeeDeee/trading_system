"""Write the per-run artifact directory. One registry row per run, always."""

from __future__ import annotations

import csv
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd  # type: ignore[import-untyped]
import yaml  # type: ignore[import-untyped]

from scout.config.hashing import config_hash
from scout.config.loader import git_tree_is_clean
from scout.config.schema import ScoutConfig
from scout.domain.enums import SetupOutcome
from scout.domain.results import BacktestResult, ClosedTrade
from scout.utils.clock import Clock, WallClock

_REGISTRY_COLUMNS: tuple[str, ...] = (
    "trial_id",
    "run_id",
    "timestamp_utc",
    "git_sha",
    "config_hash",
    "split",
    "period_start",
    "period_end",
    "strategies",
    "n_trades",
    "mean_r",
    "sharpe",
    "max_dd_pct",
    "total_return_pct",
    "notes",
)

_REGISTRY_PATH_DEFAULT = Path("experiments") / "registry.csv"


def results_dir(cfg: ScoutConfig, run_id: str) -> Path:
    return Path(cfg.audit.results_dir) / run_id


def make_run_id(strategy_slug: str, clock: Clock | None = None) -> str:
    """`YYYYMMDD-HHMMSS-<strategy-slug>`. Timestamp comes from Clock, never wall-clock."""
    now = (clock or WallClock()).now()
    stamp = now.astimezone(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{strategy_slug}"


def write_run_outputs(
    result: BacktestResult,
    run_id: str,
    cfg: ScoutConfig,
    *,
    notes: str = "",
    clock: Clock | None = None,
    used_bins: pd.DataFrame | None = None,
    funnel: pd.DataFrame | None = None,
) -> Path:
    """Write every artifact in 11-BACKTEST_ENGINE.md §8 and append registry.csv."""
    out = results_dir(cfg, run_id)
    out.mkdir(parents=True, exist_ok=True)
    digest = result.config_hash or config_hash(cfg)
    _write_text(out / "config_hash.txt", digest + "\n")
    _write_resolved_config(out / "config.yaml", cfg)
    metrics = dict(result.metrics)
    _write_json(out / "metrics.json", metrics)
    _write_trades(out / "trades.csv", result.trades)
    _write_equity(out / "equity.csv", result.equity_curve, result.benchmark_curve)
    if funnel is None:
        funnel = _funnel_from_result(result)
    _write_csv(out / "funnel.csv", funnel)
    bins = used_bins if used_bins is not None else pd.DataFrame()
    _write_csv(out / "bins.csv", bins)
    _ensure_run_log(out / "run.log")
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    _write_plots(plots, result, dpi=cfg.research.plot_dpi)
    _append_registry(cfg, result, run_id, notes=notes, clock=clock)
    return out


def _write_resolved_config(path: Path, cfg: ScoutConfig) -> None:
    payload = cfg.model_dump(mode="json")
    text = yaml.safe_dump(payload, sort_keys=True, allow_unicode=True)
    _atomic_text(path, text)


def _write_trades(path: Path, trades: Sequence[ClosedTrade]) -> None:
    rows: list[dict[str, object]] = []
    for trade in trades:
        rows.append(
            {
                "symbol": trade.symbol,
                "strategy_id": trade.strategy_id,
                "direction": trade.direction.value,
                "entry_ts": trade.entry_ts.isoformat(),
                "exit_ts": trade.exit_ts.isoformat(),
                "entry_price": str(trade.entry_price),
                "exit_price": str(trade.exit_price),
                "qty": str(trade.qty),
                "bars_held": trade.bars_held,
                "outcome": trade.outcome.value,
                "gross_pnl_usd": str(trade.gross_pnl_usd),
                "fees_usd": str(trade.fees_usd),
                "dividends_usd": str(trade.dividends_usd),
                "borrow_usd": str(trade.borrow_usd),
                "funding_usd": str(trade.funding_usd),
                "net_pnl_usd": str(trade.net_pnl_usd),
                "realised_r": trade.realised_r,
                "mae_r": trade.mae_r,
                "mfe_r": trade.mfe_r,
                "regime": trade.regime.value,
                "vol_bucket": trade.vol_bucket.value,
                "cluster": trade.cluster,
                "ev_net_r_at_entry": trade.ev_net_r_at_entry,
            }
        )
    frame = pd.DataFrame(rows)
    _write_csv(path, frame)


def _write_equity(path: Path, equity: pd.Series, benchmark: pd.Series) -> None:
    eq = _as_utc_series(equity, "equity")
    bench = _as_utc_series(benchmark, "benchmark")
    joined = pd.DataFrame({"equity": eq}).join(bench, how="left")
    peak = joined["equity"].cummax()
    dd = (peak - joined["equity"]) / peak.replace(0.0, float("nan"))
    joined["drawdown_pct"] = dd.fillna(0.0).clip(lower=0.0)
    out = joined.reset_index()
    if out.columns[0] != "ts":
        out = out.rename(columns={out.columns[0]: "ts"})
    _write_csv(path, out)


def _funnel_from_result(result: BacktestResult) -> pd.DataFrame:
    rows = [
        {"stage": "REJECT", "rejection_reason": reason.value, "n": n}
        for reason, n in sorted(
            result.n_rejections_by_reason.items(), key=lambda kv: kv[0].value
        )
    ]
    accepted = result.n_decisions_considered - sum(result.n_rejections_by_reason.values())
    rows.append({"stage": "ACCEPTED", "rejection_reason": "", "n": max(0, accepted)})
    return pd.DataFrame(rows)


def _write_plots(plots: Path, result: BacktestResult, *, dpi: int) -> None:
    equity = _as_utc_series(result.equity_curve, "equity")
    bench = _as_utc_series(result.benchmark_curve, "benchmark")
    if equity.empty:
        equity = pd.Series([1.0], index=pd.DatetimeIndex([result.period_start], tz="UTC"))
    eq_norm = _normalise(equity)
    bench_norm = _normalise(bench.reindex(eq_norm.index).ffill()) if not bench.empty else eq_norm
    peak = eq_norm.cummax()
    dd = (eq_norm / peak) - 1.0
    final_pct = float(eq_norm.iloc[-1] / eq_norm.iloc[0] - 1.0) * 100.0

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(eq_norm.index, eq_norm.to_numpy(), label="strategy")
    ax.plot(bench_norm.index, bench_norm.to_numpy(), label="SPY")
    ax.legend()
    ax.set_title(f"Equity  final {final_pct:.1f}%")
    ax.set_ylabel("growth (start=1)")
    fig.autofmt_xdate()
    fig.savefig(plots / "equity.png", dpi=dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(dd.index, dd.to_numpy(), 0.0, color="C3", alpha=0.4)
    ax.plot(dd.index, dd.to_numpy(), color="C3")
    max_dd = float(dd.min()) if len(dd) else 0.0
    max_dd_ts = dd.idxmin() if len(dd) else None
    ax.set_title(f"Drawdown  max {max_dd:.1%} at {max_dd_ts}")
    fig.autofmt_xdate()
    fig.savefig(plots / "drawdown.png", dpi=dpi)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(eq_norm.index, eq_norm.to_numpy())
    axes[0].set_ylabel("equity")
    axes[1].fill_between(dd.index, dd.to_numpy(), 0.0, color="C3", alpha=0.4)
    axes[1].set_ylabel("drawdown")
    fig.autofmt_xdate()
    fig.savefig(plots / "summary.png", dpi=dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    pred = [t.ev_net_r_at_entry for t in result.trades]
    realised = [t.realised_r for t in result.trades]
    if pred:
        ax.scatter(pred, realised, s=12, alpha=0.7)
    ax.axhline(0.0, color="grey", linewidth=0.8)
    ax.axvline(0.0, color="grey", linewidth=0.8)
    ax.set_xlabel("predicted ev_net_r")
    ax.set_ylabel("realised R")
    ax.set_title("calibration")
    fig.savefig(plots / "calibration.png", dpi=dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    monthly = eq_norm.resample("ME").last().pct_change().dropna()
    if monthly.empty:
        ax.set_title("monthly returns (empty)")
    else:
        years = monthly.index.year
        months = monthly.index.month
        grid = pd.DataFrame({"year": years, "month": months, "r": monthly.to_numpy()})
        pivot = grid.pivot_table(index="year", columns="month", values="r", aggfunc="sum")
        ax.imshow(pivot.to_numpy(), aspect="auto", cmap="RdYlGn")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(y) for y in pivot.index])
        ax.set_xticks(range(12))
        ax.set_xticklabels([str(i) for i in range(1, 13)])
        ax.set_title("monthly returns")
    fig.savefig(plots / "monthly_returns.png", dpi=dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    if result.trades:
        by_reg: dict[str, list[float]] = {}
        for trade in result.trades:
            by_reg.setdefault(trade.regime.value, []).append(trade.realised_r)
        labels = sorted(by_reg)
        means = [sum(by_reg[k]) / len(by_reg[k]) for k in labels]
        ax.bar(labels, means)
    ax.set_ylabel("mean realised R")
    ax.set_title("regime breakdown")
    fig.savefig(plots / "regime_breakdown.png", dpi=dpi)
    plt.close(fig)


def _append_registry(
    cfg: ScoutConfig,
    result: BacktestResult,
    run_id: str,
    *,
    notes: str,
    clock: Clock | None,
) -> None:
    path = Path(cfg.research.registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_registry(path)
    trial_id = 1 if not existing else max(int(r["trial_id"]) for r in existing) + 1
    now = (clock or WallClock()).now().astimezone(UTC)
    n_trades = len(result.trades)
    mean_r = _mean([t.realised_r for t in result.trades])
    row = {
        "trial_id": str(trial_id),
        "run_id": run_id,
        "timestamp_utc": now.isoformat(),
        "git_sha": _git_sha(),
        "config_hash": result.config_hash,
        "split": cfg.period.split.value,
        "period_start": result.period_start.isoformat(),
        "period_end": result.period_end.isoformat(),
        "strategies": ",".join(
            sc.strategy_id for sc in cfg.strategies if sc.enabled
        ),
        "n_trades": str(n_trades),
        "mean_r": _fmt(mean_r),
        "sharpe": _fmt(float(result.metrics.get("sharpe", float("nan")))),
        "max_dd_pct": _fmt(float(result.metrics.get("max_dd_pct", float("nan")))),
        "total_return_pct": _fmt(float(result.metrics.get("total_return_pct", float("nan")))),
        "notes": notes,
    }
    write_header = not path.is_file()
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(_REGISTRY_COLUMNS))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise


def compute_headline_metrics(
    trades: Sequence[ClosedTrade],
    equity: pd.Series,
) -> dict[str, float]:
    """Minimal headline metrics. Full set is M3.5."""
    eq = _as_utc_series(equity, "equity")
    n_trades = float(len(trades))
    realised = [t.realised_r for t in trades]
    mean_r = _mean(realised)
    total_return = 0.0
    max_dd = 0.0
    sharpe = float("nan")
    if not eq.empty:
        start = float(eq.iloc[0])
        end = float(eq.iloc[-1])
        if start > 0.0:
            total_return = end / start - 1.0
        peak = eq.cummax()
        dd = (peak - eq) / peak.replace(0.0, float("nan"))
        max_dd = float(dd.max()) if dd.notna().any() else 0.0
        rets = eq.pct_change().dropna()
        if len(rets) > 1:
            std = float(rets.std(ddof=1))
            if std > 0.0:
                sharpe = float(rets.mean()) / std * math.sqrt(252.0)
    wins = sum(1 for t in trades if t.realised_r > 0.0)
    return {
        "n_trades": n_trades,
        "mean_r": mean_r,
        "win_rate": (wins / n_trades) if n_trades else float("nan"),
        "sharpe": sharpe,
        "max_dd_pct": max_dd,
        "total_return_pct": total_return,
        "n_target": float(sum(1 for t in trades if t.outcome is SetupOutcome.TARGET)),
        "n_stop": float(sum(1 for t in trades if t.outcome is SetupOutcome.STOP)),
        "n_time": float(sum(1 for t in trades if t.outcome is SetupOutcome.TIME)),
    }


def _read_registry(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    sha = result.stdout.strip() if result.returncode == 0 else "unknown"
    if not git_tree_is_clean():
        return f"{sha}--dirty"
    return sha


def _as_utc_series(series: pd.Series, name: str) -> pd.Series:
    if series.empty:
        return pd.Series(dtype="float64", name=name)
    out = series.copy()
    out.name = name
    idx = pd.DatetimeIndex(pd.to_datetime(out.index, utc=True))
    out.index = idx
    return out.sort_index()


def _normalise(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    first = float(series.iloc[0])
    if first == 0.0 or not math.isfinite(first):
        return series
    return series / first


def _mean(values: Sequence[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return float("nan")
    return sum(finite) / len(finite)


def _fmt(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.10g}"


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        frame.to_csv(tmp, index=False)
        tmp.replace(path)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n"
    _atomic_text(path, text)


def _write_text(path: Path, text: str) -> None:
    _atomic_text(path, text)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise


def _ensure_run_log(path: Path) -> None:
    if path.is_file():
        return
    _atomic_text(path, "")
