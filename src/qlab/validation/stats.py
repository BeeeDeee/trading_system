"""Overfitting statistics (spec §9.6): PSR/DSR, PBO via CSCV, stationary block bootstrap.

References: Bailey & López de Prado (2012, 2014) for PSR/DSR, Bailey, Borwein, López de Prado &
Zhu (2017) for PBO/CSCV, Politis & Romano (1994) for the stationary bootstrap.
All Sharpe ratios here are per period (not annualized) unless stated otherwise.
"""

from collections.abc import Callable
from itertools import combinations
from math import comb, e, sqrt
from statistics import NormalDist

import numpy as np

EULER_GAMMA = 0.5772156649015329
_N = NormalDist()


def sharpe(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else 0.0


def probabilistic_sharpe(returns: np.ndarray, sr_benchmark: float = 0.0) -> float:
    """P(true SR > sr_benchmark) given sample length, skewness and kurtosis of `returns`."""
    r = np.asarray(returns, dtype=float)
    n = len(r)
    sr = sharpe(r)
    z = (r - r.mean()) / r.std(ddof=0)
    skew, kurt = float((z ** 3).mean()), float((z ** 4).mean())
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    return _N.cdf((sr - sr_benchmark) * sqrt(n - 1) / sqrt(max(denom, 1e-12)))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum of `n_trials` Sharpe estimates of zero-skill strategies."""
    if n_trials <= 1:
        return 0.0
    return sqrt(sr_variance) * ((1 - EULER_GAMMA) * _N.inv_cdf(1 - 1 / n_trials)
                                + EULER_GAMMA * _N.inv_cdf(1 - 1 / (n_trials * e)))


def deflated_sharpe(returns: np.ndarray, n_trials: int, sr_variance: float) -> float:
    """PSR against the Sharpe ratio expected from the best of `n_trials` unskilled trials.

    `sr_variance` is the variance of the per-period Sharpe ratios across the trials.
    """
    return probabilistic_sharpe(returns, expected_max_sharpe(n_trials, sr_variance))


def pbo_cscv(returns: np.ndarray, n_blocks: int = 16) -> tuple[float, np.ndarray]:
    """Probability of backtest overfitting via combinatorially symmetric cross-validation.

    `returns` is T x N (days x candidates). For every split of the blocks into halves, the candidate
    with the best in-sample Sharpe is ranked out of sample; PBO is the share of splits where it
    lands at or below the median. Returns (PBO, logits).
    """
    r = np.asarray(returns, dtype=float)
    if n_blocks % 2 or n_blocks < 2:
        raise ValueError("n_blocks must be even")
    n_days, n_cand = r.shape
    edges = np.linspace(0, n_days, n_blocks + 1).astype(int)
    # Per-block sufficient statistics, so every split costs O(N) instead of O(T*N).
    s1 = np.array([r[a:b].sum(axis=0) for a, b in zip(edges[:-1], edges[1:])])
    s2 = np.array([(r[a:b] ** 2).sum(axis=0) for a, b in zip(edges[:-1], edges[1:])])
    cnt = np.diff(edges).astype(float)

    def block_sharpe(blocks):
        n = cnt[blocks].sum()
        m = s1[blocks].sum(axis=0) / n
        var = (s2[blocks].sum(axis=0) - n * m * m) / (n - 1)
        return m / np.sqrt(np.maximum(var, 1e-300))

    all_blocks = set(range(n_blocks))
    splits = combinations(range(n_blocks), n_blocks // 2)
    logits = np.empty(comb(n_blocks, n_blocks // 2))
    for k, is_blocks in enumerate(splits):
        is_idx = list(is_blocks)
        oos_idx = sorted(all_blocks - set(is_blocks))
        best = int(np.argmax(block_sharpe(is_idx)))
        oos = block_sharpe(oos_idx)
        rank = (oos < oos[best]).sum() + 0.5 * ((oos == oos[best]).sum() + 1)  # 1..N, mid ties
        w = rank / (n_cand + 1)
        logits[k] = np.log(w / (1 - w))
    return float((logits <= 0).mean()), logits


def stationary_bootstrap_indices(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """One resample of 0..n-1: blocks with geometric lengths (mean `mean_block`), wrapping around."""
    starts = rng.random(n) < 1.0 / mean_block
    starts[0] = True
    idx = np.empty(n, dtype=int)
    pos = 0
    for i in range(n):
        pos = int(rng.integers(n)) if starts[i] else (pos + 1) % n
        idx[i] = pos
    return idx


def bootstrap_ci(data: np.ndarray, stat: Callable[[np.ndarray], float], n_boot: int = 1000,
                 mean_block: float = 21.0, alpha: float = 0.10,
                 seed: int = 0) -> tuple[float, float, float]:
    """Point estimate and (1 - alpha) percentile interval of `stat` under the stationary bootstrap.

    `data` is T or T x k (rows are resampled together, e.g. strategy and benchmark returns).
    """
    x = np.asarray(data, dtype=float)
    rng = np.random.default_rng(seed)
    boots = np.array([stat(x[stationary_bootstrap_indices(len(x), mean_block, rng)])
                      for _ in range(n_boot)])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(stat(x)), float(lo), float(hi)


def spa_pvalue(excess: np.ndarray, n_boot: int = 2000, mean_block: float = 21.0,
               seed: int = 0) -> float:
    """Hansen (2005) SPA_c p-value for H0: no column of `excess` has a positive mean.

    `excess` is T x k: daily returns of k strategies minus the benchmark.
    """
    d = np.asarray(excess, dtype=float)
    d = d[:, None] if d.ndim == 1 else d
    n = len(d)
    rng = np.random.default_rng(seed)
    means = d.mean(axis=0)
    boot_means = np.array([d[stationary_bootstrap_indices(n, mean_block, rng)].mean(axis=0)
                           for _ in range(n_boot)])
    sd = np.sqrt(n) * boot_means.std(axis=0)
    sd = np.where(sd > 0, sd, np.inf)
    t_obs = max(float(np.max(np.sqrt(n) * means / sd)), 0.0)
    threshold = -np.sqrt(2 * np.log(np.log(n)))
    recentred = np.where(np.sqrt(n) * means / sd >= threshold, means, 0.0)
    t_boot = np.maximum(np.max(np.sqrt(n) * (boot_means - recentred) / sd, axis=1), 0.0)
    return float((t_boot >= t_obs).mean())
