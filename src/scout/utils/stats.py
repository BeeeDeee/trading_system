from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm, ttest_ind  # type: ignore[import-untyped]

from scout.utils.errors import ScoutError

# Paper uses 0.5772156649; this is the same constant to more digits.
_EULER_MASCHERONI = 0.5772156649015328606

DEFAULT_BOOTSTRAP_ITERATIONS = 2000
DEFAULT_BOOTSTRAP_SEED = 20260827
DEFAULT_CONFIDENCE = 0.90

ArrayLike = Sequence[float] | NDArray[np.floating[Any]]
Statistic = Callable[[NDArray[np.float64]], float]


def bootstrap_ci(
    values: ArrayLike,
    *,
    statistic: Statistic | None = None,
    n_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
) -> tuple[float, float]:
    """Two-sided bootstrap CI of `statistic` (default: the mean)."""
    stats = _bootstrap_stats(values, statistic, n_iterations, seed)
    alpha = 1.0 - confidence
    lo = float(np.quantile(stats, alpha / 2.0))
    hi = float(np.quantile(stats, 1.0 - alpha / 2.0))
    return lo, hi


def bootstrap_lcb(
    values: ArrayLike,
    *,
    n_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    quantile: float = 0.10,
) -> float:
    """Lower confidence bound: `quantile` of bootstrap resample means."""
    stats = _bootstrap_stats(values, None, n_iterations, seed)
    return float(np.quantile(stats, quantile))


def welch_ttest(a: ArrayLike, b: ArrayLike) -> tuple[float, float]:
    """Two-sided Welch's t-test. Returns `(t_statistic, p_value)`."""
    arr_a = _finite_1d(a, "a")
    arr_b = _finite_1d(b, "b")
    if arr_a.size < 2 or arr_b.size < 2:
        raise ScoutError("Welch's t-test requires at least two observations in each sample")
    result = ttest_ind(arr_a, arr_b, equal_var=False)
    return float(result.statistic), float(result.pvalue)


def wilson_ci(
    successes: int,
    n: int,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        raise ScoutError(f"wilson_ci requires n > 0, got {n}")
    if successes < 0 or successes > n:
        raise ScoutError(f"successes must be in [0, n], got {successes} of {n}")
    if not 0.0 < confidence < 1.0:
        raise ScoutError(f"confidence must be in (0, 1), got {confidence}")
    z = float(norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    margin = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    return centre - margin, centre + margin


def deflated_sharpe(
    sharpe: float,
    n_trials: int,
    n_obs: int,
    skew: float,
    kurtosis: float,
    *,
    var_sr: float | None = None,
) -> float:
    """Bailey & López de Prado (2014) Deflated Sharpe Ratio.

    Returns the probability that the true Sharpe exceeds zero, given that this
    was the best of `n_trials` attempts. `kurtosis` is Pearson (normal = 3).
    `sharpe` and `var_sr` must be in the same units as the observations counted
    by `n_obs` — do not mix annualised Sharpe with a per-period variance.

    `var_sr` is the variance of the trial Sharpes. When omitted it is estimated
    from this series' moments (Lo 2002). The paper's numerical example supplies
    it separately.
    """
    if n_trials < 1:
        raise ScoutError(f"n_trials must be >= 1, got {n_trials}")
    if n_obs < 2:
        raise ScoutError(f"n_obs must be >= 2, got {n_obs}")
    se_non_normal = math.sqrt(
        max(
            1.0 - skew * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe * sharpe,
            0.0,
        )
    )
    if var_sr is None:
        var = (se_non_normal * se_non_normal) / (n_obs - 1)
    else:
        if var_sr < 0.0:
            raise ScoutError(f"var_sr must be non-negative, got {var_sr}")
        var = var_sr
    sr0 = 0.0 if n_trials == 1 or var == 0.0 else math.sqrt(var) * _expected_max_z(n_trials)
    denom = se_non_normal / math.sqrt(n_obs - 1)
    if denom == 0.0:
        return 1.0 if sharpe > sr0 else 0.0
    return float(norm.cdf((sharpe - sr0) / denom))


def _expected_max_z(n_trials: int) -> float:
    # Inverse-normal is -inf at 0; the EVT approximation is for N >= 2.
    z1 = float(norm.ppf(1.0 - 1.0 / n_trials))
    z2 = float(norm.ppf(1.0 - 1.0 / (n_trials * math.e)))
    return (1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2


def _bootstrap_stats(
    values: ArrayLike,
    statistic: Statistic | None,
    n_iterations: int,
    seed: int,
) -> NDArray[np.float64]:
    arr = _finite_1d(values, "values")
    if arr.size < 1:
        raise ScoutError("bootstrap requires at least one observation")
    if n_iterations < 1:
        raise ScoutError(f"n_iterations must be >= 1, got {n_iterations}")
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(n_iterations, arr.size), replace=True)
    out = np.empty(n_iterations, dtype=np.float64)
    if statistic is None:
        out[:] = samples.mean(axis=1)
    else:
        for i in range(n_iterations):
            out[i] = statistic(samples[i])
    return out


def _finite_1d(values: ArrayLike, name: str) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ScoutError(f"{name} must be 1-d, got shape {arr.shape}")
    return arr
