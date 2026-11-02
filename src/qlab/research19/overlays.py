"""H3 overlays (prereg §4.4): applied to return series of sleeves (a, b) or to target weights (c)."""

import numpy as np

from qlab.engine.vector import Decisions
from qlab.research19 import setup as S
from qlab.research19.signals import realized_vol, sma


def rolling_std(r: np.ndarray, n: int, ann: float) -> np.ndarray:
    """Std (ddof=1) of the last n values up to t times sqrt(ann); NaN for t < n-1."""
    r = np.asarray(r, dtype=float)
    out = np.full(len(r), np.nan)
    if len(r) >= n:
        w = np.lib.stride_tricks.sliding_window_view(r, n)
        out[n - 1:] = w.std(axis=1, ddof=1) * np.sqrt(ann)
    return out


def h3a_overlay(r_h2: np.ndarray, expo_h2: np.ndarray, r_spy: np.ndarray, r_cash: np.ndarray,
                tom: np.ndarray, cost_bps: float = S.ETF_COST_BPS) -> np.ndarray:
    """r = r_H2 + (1 - expo_H2[t-1]) * TOM[t] * (r_SPY[t] - r_cash[t]) - cost, cost = bps * |change of the invested
    SPY part| on the days the window opens or closes. `expo_H2[t-1]` is the invested fraction at the close of t-1."""
    r_h2, expo = np.asarray(r_h2, float), np.asarray(expo_h2, float)
    free = np.r_[1.0, 1.0 - expo[:-1]]
    inv = free * np.asarray(tom, float)
    inv_prev = np.r_[0.0, inv[:-1]]
    changes = np.asarray(tom, bool) != np.r_[False, np.asarray(tom, bool)[:-1]]
    cost = cost_bps / 1e4 * np.abs(inv - inv_prev) * changes
    return r_h2 + inv * (np.asarray(r_spy, float) - np.asarray(r_cash, float)) - cost


def h3b_vol_target(r: np.ndarray, r_cash: np.ndarray, target: float = S.VOL_TARGET_PORTFOLIO,
                   window: int = S.VOL_TARGET_WINDOW, ann: float = S.ETF_YEAR,
                   cost_bps: float = S.ETF_COST_BPS) -> tuple[np.ndarray, np.ndarray]:
    """k_t = min(1, target / sigma20_t) from returns up to t; r_new[t+1] = k_t r[t+1] + (1-k_t) cash[t+1] - cost.
    Returns (new returns, k) where k[t] is the scale applied to day t+1's return (k[T-1] unused)."""
    r = np.asarray(r, float)
    sig = rolling_std(r, window, ann)
    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.where(np.isfinite(sig) & (sig > 0), np.minimum(1.0, target / sig), 1.0)
    k_used = np.r_[1.0, k[:-1]]                       # scale in force on day t was decided at the close of t-1
    k_prev = np.r_[1.0, k_used[:-1]]
    out = k_used * r + (1.0 - k_used) * np.asarray(r_cash, float) - cost_bps / 1e4 * np.abs(k_used - k_prev)
    return out, k


def crash_state(P_bench: np.ndarray, valid: np.ndarray, ann: float) -> np.ndarray:
    """(T,) bool: P < SMA200 and sigma20 > 1.5 * median(sigma20 over the last 252 days, incl. t).
    P_bench and valid are 1-D (the benchmark: BTC for H1a, SPY for H1b)."""
    P, v = np.asarray(P_bench, float)[:, None], np.asarray(valid, bool)[:, None]
    s200 = sma(P, v, 200)[:, 0]
    sig = realized_vol(P, v, S.VOL_TARGET_WINDOW, ann)[:, 0]
    med = np.full(len(sig), np.nan)
    n = S.CRASH_MEDIAN_WINDOW
    if len(sig) >= n:
        w = np.lib.stride_tricks.sliding_window_view(sig, n)
        with np.errstate(all="ignore"):
            med[n - 1:] = np.nanmedian(np.where(np.isfinite(w), w, np.nan), axis=1)
    with np.errstate(invalid="ignore"):
        return (P[:, 0] < s200) & (sig > S.CRASH_VOL_MULT * med)


def crash_scale(crash: np.ndarray) -> np.ndarray:
    """Multiplier of the sleeve's target weights decided at the close of t (0.5 in a crash state)."""
    return np.where(np.asarray(crash, bool), S.CRASH_SCALE, 1.0)


def scale_decisions(d: Decisions, scale: np.ndarray) -> Decisions:
    return Decisions(d.days, d.weights * np.asarray(scale, float)[d.days][:, None])
