"""Single-run diagnostics that make the estimator falsifiable.

Re-run sensitivities (z, cost_multiplier, tie-rule, LCB method, earnings gate,
factor loadings) belong to the M4 robustness suite. This module produces the
tables that can be computed from one run's trades, equity, and bins.
"""

from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from scout.utils.stats import bootstrap_ci


def calibration_buckets(trades: pd.DataFrame, n_buckets: int = 10) -> pd.DataFrame:
    """Decile buckets of predicted ev_net_r vs mean realised R, with bootstrap CI."""
    if trades.empty or "ev_net_r_at_entry" not in trades.columns:
        return pd.DataFrame(
            columns=[
                "bucket",
                "predicted_mid",
                "mean_realised_r",
                "ci_low",
                "ci_high",
                "n",
            ]
        )
    pred = pd.to_numeric(trades["ev_net_r_at_entry"], errors="coerce")
    realised = pd.to_numeric(trades["realised_r"], errors="coerce")
    frame = pd.DataFrame({"pred": pred, "realised": realised}).dropna()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "bucket",
                "predicted_mid",
                "mean_realised_r",
                "ci_low",
                "ci_high",
                "n",
            ]
        )
    k = min(n_buckets, max(1, int(frame["pred"].nunique())))
    try:
        frame = frame.assign(
            bucket=pd.qcut(frame["pred"], q=k, duplicates="drop")
        )
    except ValueError:
        frame = frame.assign(bucket=pd.cut(frame["pred"], bins=k))
    rows = []
    for i, (interval, group) in enumerate(frame.groupby("bucket", observed=True), start=1):
        values = group["realised"].to_numpy(dtype=np.float64)
        lo, hi = (
            bootstrap_ci(values.tolist())
            if values.size
            else (float("nan"), float("nan"))
        )
        left = float(getattr(interval, "left", float("nan")))
        right = float(getattr(interval, "right", float("nan")))
        rows.append(
            {
                "bucket": i,
                "predicted_mid": 0.5 * (left + right),
                "mean_realised_r": float(values.mean()) if values.size else float("nan"),
                "ci_low": lo,
                "ci_high": hi,
                "n": int(values.size),
            }
        )
    return pd.DataFrame(rows)


def cold_start_profile(trades: pd.DataFrame) -> pd.DataFrame:
    """Trades per month over the run. Confirms the early drought and that it ends."""
    if trades.empty or "entry_ts" not in trades.columns:
        return pd.DataFrame(columns=["month", "n_trades"])
    ts = pd.to_datetime(trades["entry_ts"], utc=True)
    month = ts.dt.to_period("M")
    counts = month.value_counts().sort_index()
    return pd.DataFrame(
        {"month": counts.index.astype(str), "n_trades": counts.to_numpy()}
    )


def gap_decomposition(trades: pd.DataFrame) -> pd.DataFrame:
    """STOP-outcome realised R: how often the R framework's -1 bound is breached."""
    if trades.empty:
        return pd.DataFrame(
            columns=["n_stop", "mean_r", "p05_r", "frac_below_minus_1_5"]
        )
    outcome = trades["outcome"].astype(str) if "outcome" in trades.columns else None
    r = pd.to_numeric(trades["realised_r"], errors="coerce")
    if outcome is not None:
        r = r[outcome == "STOP"]
    r = r.dropna()
    if r.empty:
        return pd.DataFrame(
            {
                "n_stop": [0],
                "mean_r": [float("nan")],
                "p05_r": [float("nan")],
                "frac_below_minus_1_5": [float("nan")],
            }
        )
    return pd.DataFrame(
        {
            "n_stop": [int(r.size)],
            "mean_r": [float(r.mean())],
            "p05_r": [float(r.quantile(0.05))],
            "frac_below_minus_1_5": [float((r < -1.5).mean())],
        }
    )


def bin_realised_table(bins: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Join realised trade counts and mean R onto the bins used this run."""
    if bins.empty:
        return bins
    out = bins.copy()
    if trades.empty or "ev_net_r_at_entry" not in trades.columns:
        out = out.assign(realised_n=0, realised_mean_r=float("nan"))
        return out
    out = out.assign(realised_n=len(trades), realised_mean_r=float(
        pd.to_numeric(trades["realised_r"], errors="coerce").mean()
    ))
    return out
