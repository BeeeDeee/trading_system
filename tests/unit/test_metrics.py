"""Hand-computed metrics on a 20-point equity curve and 20 trades.

Expected values are derived from the formulas in 12-RESEARCH_PROTOCOL.md §4,
not from calling compute_metrics.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from scipy.stats import kurtosis, skew

from scout.research.metrics import BARS_PER_YEAR, compute_metrics
from scout.research.registry import REGISTRY_COLUMNS, append_trial
from scout.utils.stats import deflated_sharpe

REL = 1e-9

# 19 bar returns: ten +2%, nine -1%. Equity starts at 100.
_RETS = (0.02,) * 10 + (-0.01,) * 9
_START = datetime(2020, 1, 2, tzinfo=UTC)


def _equity_frame() -> pd.DataFrame:
    eq = [100.0]
    for ret in _RETS:
        eq.append(eq[-1] * (1.0 + ret))
    idx = [_START + timedelta(days=i) for i in range(20)]
    return pd.DataFrame({"ts": idx, "equity": eq, "benchmark": eq})


def _trade_rows() -> pd.DataFrame:
    # ten +1 R, eight -1 R, one +0.5 R, one -0.5 R.
    realised = [1.0] * 10 + [-1.0] * 8 + [0.5, -0.5]
    rows = []
    for i, r in enumerate(realised):
        entry = _START + timedelta(days=i)
        exit_ts = entry + timedelta(days=4)
        rows.append(
            {
                "symbol": f"S{i:02d}",
                "strategy_id": "xsec_momentum_v1",
                "direction": "LONG" if i % 2 == 0 else "SHORT",
                "entry_ts": entry.isoformat(),
                "exit_ts": exit_ts.isoformat(),
                "entry_price": "100",
                "exit_price": "110" if r > 0 else "90",
                "qty": "10",
                "bars_held": 5,
                "outcome": "TARGET" if r > 0 else "STOP",
                "gross_pnl_usd": "10",
                "fees_usd": "1",
                "dividends_usd": "0",
                "borrow_usd": "0",
                "funding_usd": "0",
                "net_pnl_usd": "9",
                "realised_r": r,
                "mae_r": -0.5,
                "mfe_r": 1.0,
                "regime": "TREND_UP",
                "vol_bucket": "MID",
                "cluster": "INFO_TECH",
                "ev_net_r_at_entry": 0.1,
            }
        )
    return pd.DataFrame(rows)


def test_total_return_pct_20_point() -> None:
    # 100 * 1.02^10 * 0.99^9 / 100 - 1.
    expected = (1.02**10) * (0.99**9) - 1.0
    m = compute_metrics(_trade_rows(), _equity_frame())
    assert m["total_return_pct"] == pytest.approx(expected, rel=REL)


def test_cagr_pct_20_point() -> None:
    total = (1.02**10) * (0.99**9)
    years = 19.0 / 365.25
    expected = total ** (1.0 / years) - 1.0
    m = compute_metrics(_trade_rows(), _equity_frame())
    assert m["cagr_pct"] == pytest.approx(expected, rel=REL)


def test_sharpe_20_point() -> None:
    mean = 0.11 / 19.0
    var = sum((r - mean) ** 2 for r in _RETS) / 18.0
    expected = mean / math.sqrt(var) * math.sqrt(BARS_PER_YEAR)
    m = compute_metrics(_trade_rows(), _equity_frame())
    assert m["sharpe"] == pytest.approx(expected, rel=REL)


def test_sortino_20_point() -> None:
    mean = 0.11 / 19.0
    # Downside deviation: RMS of min(r, 0) over all 19 returns, MAR = 0.
    down = math.sqrt(sum(min(r, 0.0) ** 2 for r in _RETS) / 19.0)
    expected = mean / down * math.sqrt(BARS_PER_YEAR)
    m = compute_metrics(_trade_rows(), _equity_frame())
    assert m["sortino"] == pytest.approx(expected, rel=REL)


def test_max_drawdown_pct_20_point() -> None:
    # Peak at bar 10 (after the last +2%). Then nine -1% bars:
    # 1 - 0.99^9 = 1 - 0.9135172474836408.
    expected = 1.0 - (0.99**9)
    m = compute_metrics(_trade_rows(), _equity_frame())
    assert m["max_drawdown_pct"] == pytest.approx(expected, rel=REL)


def test_max_drawdown_duration_days_20_point() -> None:
    # Underwater from 2020-01-13 (index 11) through 2020-01-21 (index 19): 8 days.
    m = compute_metrics(_trade_rows(), _equity_frame())
    assert m["max_drawdown_duration_days"] == pytest.approx(8.0, rel=REL)


def test_calmar_20_point() -> None:
    total = (1.02**10) * (0.99**9)
    years = 19.0 / 365.25
    cagr = total ** (1.0 / years) - 1.0
    max_dd = 1.0 - (0.99**9)
    m = compute_metrics(_trade_rows(), _equity_frame())
    assert m["calmar"] == pytest.approx(cagr / max_dd, rel=REL)


def test_ulcer_index_20_point() -> None:
    eq = [100.0]
    for ret in _RETS:
        eq.append(eq[-1] * (1.0 + ret))
    peak = eq[0]
    dds = []
    for x in eq:
        peak = max(peak, x)
        dds.append((peak - x) / peak)
    expected = math.sqrt(sum(d * d for d in dds) / len(dds))
    m = compute_metrics(_trade_rows(), _equity_frame())
    assert m["ulcer_index"] == pytest.approx(expected, rel=REL)


def test_trade_statistics_20_trades() -> None:
    m = compute_metrics(_trade_rows(), _equity_frame())
    assert m["n_trades"] == pytest.approx(20.0, rel=REL)
    assert m["win_rate"] == pytest.approx(11.0 / 20.0, rel=REL)
    assert m["avg_win_r"] == pytest.approx(10.5 / 11.0, rel=REL)
    assert m["avg_loss_r"] == pytest.approx(-8.5 / 9.0, rel=REL)
    assert m["mean_r"] == pytest.approx(0.1, rel=REL)
    assert m["median_r"] == pytest.approx(0.75, rel=REL)
    rs = [1.0] * 10 + [-1.0] * 8 + [0.5, -0.5]
    mean = 0.1
    std = math.sqrt(sum((x - mean) ** 2 for x in rs) / 19.0)
    assert m["std_r"] == pytest.approx(std, rel=REL)
    assert m["profit_factor"] == pytest.approx(10.5 / 8.5, rel=REL)
    assert m["expectancy_r"] == pytest.approx(0.1, rel=REL)
    assert m["payoff_ratio"] == pytest.approx((10.5 / 11.0) / (8.5 / 9.0), rel=REL)
    assert m["max_consecutive_losses"] == pytest.approx(8.0, rel=REL)
    assert m["avg_bars_held"] == pytest.approx(5.0, rel=REL)
    assert m["mae_r_mean"] == pytest.approx(-0.5, rel=REL)
    assert m["mfe_r_mean"] == pytest.approx(1.0, rel=REL)


def test_cost_drag_pct_20_trades() -> None:
    # 20 * $1 fees / 20 * $10 gross = 0.1. _pct is a fraction.
    m = compute_metrics(_trade_rows(), _equity_frame())
    assert m["total_fees_usd"] == pytest.approx(20.0, rel=REL)
    assert m["total_funding_usd"] == pytest.approx(0.0, rel=REL)
    assert m["cost_drag_pct"] == pytest.approx(0.1, rel=REL)


def test_activity_metrics_20_point() -> None:
    m = compute_metrics(_trade_rows(), _equity_frame())
    years = 19.0 / 365.25
    assert m["trades_per_month"] == pytest.approx(20.0 / (years * 12.0), rel=REL)
    # Each trade covers entry..exit inclusive (5 calendar days) on a 20-day grid.
    # Distinct exposed dates: days 0-4, 1-5, ..., 19-23 clipped to the equity
    # index (days 0-19) → every bar from 0 through 19 is covered. exposure = 1.
    assert m["exposure_pct"] == pytest.approx(1.0, rel=REL)
    # 11 winners at 100/110, 9 losers at 100/90.
    # one-way notional = 11*(1000+1100)/2 + 9*(1000+900)/2 = 20100.
    eq = [100.0]
    for ret in _RETS:
        eq.append(eq[-1] * (1.0 + ret))
    mean_eq = sum(eq) / 20.0
    expected_to = 20_100.0 / mean_eq / years
    assert m["turnover_annual"] == pytest.approx(expected_to, rel=REL)


def test_mean_r_ci_contains_mean() -> None:
    m = compute_metrics(_trade_rows(), _equity_frame())
    assert m["mean_r_ci_low"] < m["mean_r"] < m["mean_r_ci_high"]


def test_p_value_all_positive_is_zero() -> None:
    trades = _trade_rows()
    trades = trades.assign(realised_r=1.0)
    m = compute_metrics(trades, _equity_frame())
    assert m["p_value_mean_r"] == pytest.approx(0.0, abs=1e-12)


def test_deflated_sharpe_reads_registry_trial_count(tmp_path: Path) -> None:
    registry = tmp_path / "registry.csv"
    clock_ts = datetime(2024, 1, 1, tzinfo=UTC)
    for i in range(4):
        append_trial(
            registry,
            run_id=f"r{i}",
            git_sha="abc",
            config_hash="h",
            split="DEVELOPMENT",
            period_start="2004-01-01T00:00:00+00:00",
            period_end="2017-12-31T00:00:00+00:00",
            strategies="xsec_momentum_v1",
            n_trades=10,
            mean_r=0.1,
            sharpe=1.0,
            max_dd_pct=0.1,
            total_return_pct=0.2,
            notes="t",
            timestamp_utc=clock_ts,
        )
    m = compute_metrics(
        _trade_rows(),
        _equity_frame(),
        registry_path=registry,
        split="DEVELOPMENT",
    )
    assert m["n_trials"] == pytest.approx(5.0, rel=REL)
    mean = 0.11 / 19.0
    var = sum((r - mean) ** 2 for r in _RETS) / 18.0
    std = math.sqrt(var)
    per_bar = mean / std
    expected = deflated_sharpe(
        per_bar,
        n_trials=5,
        n_obs=19,
        skew=float(skew(_RETS, bias=True)),
        kurtosis=float(kurtosis(_RETS, fisher=False, bias=True)),
    )
    assert m["deflated_sharpe"] == pytest.approx(expected, rel=REL)


def test_registry_columns_match_protocol() -> None:
    assert REGISTRY_COLUMNS == (
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


def test_apply_utc_xaxis() -> None:
    import matplotlib.pyplot as plt

    from scout.research.plots import apply_utc_xaxis

    fig, ax = plt.subplots()
    apply_utc_xaxis(ax)
    locator = ax.xaxis.get_major_locator()
    assert getattr(locator, "tz", None) is UTC
    plt.close(fig)
