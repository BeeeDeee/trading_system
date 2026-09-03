"""Write the per-run artifact directory. One registry row per run, always."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import yaml  # type: ignore[import-untyped]

from scout.config.hashing import config_hash
from scout.config.schema import ScoutConfig
from scout.domain.enums import SetupOutcome
from scout.domain.results import BacktestResult, ClosedTrade
from scout.utils.clock import Clock, WallClock


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
    from scout.research.metrics import compute_metrics
    from scout.research.plots import write_all_plots
    from scout.research.registry import append_trial, current_git_sha

    out = results_dir(cfg, run_id)
    out.mkdir(parents=True, exist_ok=True)
    digest = result.config_hash or config_hash(cfg)
    _write_text(out / "config_hash.txt", digest + "\n")
    _write_resolved_config(out / "config.yaml", cfg)
    _write_trades(out / "trades.csv", result.trades)
    _write_equity(out / "equity.csv", result.equity_curve, result.benchmark_curve)
    if funnel is None:
        funnel = _funnel_from_result(result)
    _write_csv(out / "funnel.csv", funnel)
    bins = used_bins if used_bins is not None else pd.DataFrame()
    _write_csv(out / "bins.csv", bins)
    _ensure_run_log(out / "run.log")
    trades_df = pd.read_csv(out / "trades.csv")
    equity_df = pd.read_csv(out / "equity.csv")
    metrics = compute_metrics(
        trades_df,
        equity_df,
        registry_path=Path(cfg.research.registry_path),
        bootstrap_iterations=cfg.research.bootstrap_iterations,
        seed=cfg.run.seed,
        split=cfg.period.split.value,
    )
    _write_json(out / "metrics.json", metrics)
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    write_all_plots(plots, result, dpi=cfg.research.plot_dpi)
    now = (clock or WallClock()).now().astimezone(UTC)
    append_trial(
        Path(cfg.research.registry_path),
        run_id=run_id,
        git_sha=current_git_sha(),
        config_hash=result.config_hash,
        split=cfg.period.split.value,
        period_start=result.period_start.isoformat(),
        period_end=result.period_end.isoformat(),
        strategies=",".join(sc.strategy_id for sc in cfg.strategies if sc.enabled),
        n_trades=int(metrics.get("n_trades", 0.0) or 0),
        mean_r=float(metrics.get("mean_r", float("nan"))),
        sharpe=float(metrics.get("sharpe", float("nan"))),
        max_dd_pct=float(metrics.get("max_dd_pct", float("nan"))),
        total_return_pct=float(metrics.get("total_return_pct", float("nan"))),
        notes=notes,
        timestamp_utc=now,
    )
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


def compute_headline_metrics(
    trades: Sequence[ClosedTrade],
    equity: pd.Series,
) -> dict[str, float]:
    """Headline metrics for the in-memory BacktestResult. The full set is
    computed from trades.csv and equity.csv in research.metrics.
    """
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


def _as_utc_series(series: pd.Series, name: str) -> pd.Series:
    if series.empty:
        return pd.Series(dtype="float64", name=name)
    out = series.copy()
    out.name = name
    idx = pd.DatetimeIndex(pd.to_datetime(out.index, utc=True))
    out.index = idx
    return out.sort_index()


def _mean(values: Sequence[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return float("nan")
    return sum(finite) / len(finite)


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
