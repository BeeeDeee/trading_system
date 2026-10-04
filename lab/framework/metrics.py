"""Metrics used by the gates. Sharpe ratios are annualized with the calendar's own periods per year."""

import numpy as np

from qlab.validation.metrics import max_drawdown


def periods_per_year(dates: np.ndarray) -> float:
    d = np.asarray(dates, dtype="datetime64[D]")
    years = (d[-1] - d[0]).astype(int) / 365.25
    return (len(d) - 1) / years if years > 0 else 252.0


def sharpe(r: np.ndarray, ppy: float) -> float:
    r = np.asarray(r, dtype=float)
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(ppy)) if sd > 0 else 0.0


def max_dd(r: np.ndarray) -> float:
    return max_drawdown(np.cumprod(1.0 + np.asarray(r, dtype=float)))[0]


def cagr(r: np.ndarray, ppy: float) -> float:
    years = len(r) / ppy
    return float(np.prod(1.0 + np.asarray(r)) ** (1 / years) - 1) if years > 0 else float("nan")


def bootstrap_mean_lower(x: np.ndarray, level: float, n_boot: int, mean_block: float, seed: int = 0) -> float:
    """One-sided lower bound of the mean at `level` under the stationary bootstrap (Politis-Romano).
    Vectorized version of qlab.validation.stats.stationary_bootstrap_indices (same distribution)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    offsets = np.arange(n)
    for b in range(n_boot):
        starts = rng.random(n) < 1.0 / mean_block
        starts[0] = True
        block = np.cumsum(starts) - 1
        first = np.flatnonzero(starts)
        pos = rng.integers(n, size=len(first))
        idx = (pos[block] + offsets - first[block]) % n
        means[b] = x[idx].mean()
    return float(np.quantile(means, 1.0 - level))


def bootstrap_sharpe_diff_lower(r: np.ndarray, b: np.ndarray, level: float, n_boot: int, mean_block: float,
                                seed: int = 0) -> float:
    """One-sided lower bound of Sharpe(r) - Sharpe(b) (per period, paired rows resampled together)."""
    x = np.column_stack([r, b]).astype(float)
    n = len(x)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    offsets = np.arange(n)
    for k in range(n_boot):
        starts = rng.random(n) < 1.0 / mean_block
        starts[0] = True
        block = np.cumsum(starts) - 1
        first = np.flatnonzero(starts)
        idx = (rng.integers(n, size=len(first))[block] + offsets - first[block]) % n
        s = x[idx]
        sd = s.std(axis=0, ddof=1)
        sr = np.where(sd > 0, s.mean(axis=0) / np.where(sd > 0, sd, 1), 0.0)
        diffs[k] = sr[0] - sr[1]
    return float(np.quantile(diffs, 1.0 - level))


def block_log_return_share(r: np.ndarray, n_blocks: int) -> tuple[np.ndarray, float]:
    """Log return of each block and the largest block's share of the total (inf if the total is <= 0)."""
    blocks = np.array([np.log1p(np.asarray(b)).sum() for b in np.array_split(np.asarray(r, dtype=float), n_blocks)])
    total = float(blocks.sum())
    return blocks, float(blocks.max() / total) if total > 0 else float("inf")


def position_entries(held: np.ndarray, eps: float = 1e-6) -> int:
    """Number of times any instrument goes from flat to a position (long or short) at a close."""
    on = np.abs(held) > eps
    return int((on[1:] & ~on[:-1]).sum() + on[0].sum())

