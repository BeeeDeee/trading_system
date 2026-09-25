from __future__ import annotations

import math

import numpy as np
import pytest

from scout.utils.errors import ScoutError
from scout.utils.stats import (
    bootstrap_ci,
    bootstrap_lcb,
    deflated_sharpe,
    welch_ttest,
    wilson_ci,
)


def test_bootstrap_deterministic() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(size=50)
    a = bootstrap_ci(values, seed=20260827)
    b = bootstrap_ci(values, seed=20260827)
    c = bootstrap_ci(values, seed=1)
    assert a == b
    assert a != c


def test_bootstrap_lcb_deterministic() -> None:
    values = [0.1, 0.2, -0.05, 0.3]
    assert bootstrap_lcb(values, seed=7) == bootstrap_lcb(values, seed=7)


def test_bootstrap_ci_contains_mean() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(loc=0.5, scale=1.0, size=500)
    lo, hi = bootstrap_ci(values, seed=0)
    assert lo < float(values.mean()) < hi


def test_welch_ttest_hand_computed() -> None:
    # A=[1,2,3], B=[2,3,4]; sample variances are both 1, n=3.
    # t = (2-3) / sqrt(1/3 + 1/3) = -sqrt(3/2)
    # Welch df = 4.
    t, p = welch_ttest([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])
    assert t == pytest.approx(-math.sqrt(1.5), rel=1e-9)
    # Two-sided Student-t, df=4, |t|=sqrt(3/2).
    # P(|T_4| > 1.224744871) = 0.287864 (hand-evaluated from the t CDF).
    assert p == pytest.approx(0.287864, abs=1e-6)


def test_wilson_ci_hand_computed() -> None:
    # n=10, 8 successes, 95% CI. z = Φ^{-1}(0.975) ≈ 1.95996398454
    # centre = (p + z²/(2n)) / (1 + z²/n)
    #        = (0.8 + 3.8414588/20) / (1 + 3.8414588/10)
    #        = 0.99207294 / 1.38414588 ≈ 0.716740
    # margin = z √(p(1-p)/n + z²/(4n²)) / (1 + z²/n) ≈ 0.226578
    lo, hi = wilson_ci(8, 10, confidence=0.95)
    assert lo == pytest.approx(0.490162, abs=1e-6)
    assert hi == pytest.approx(0.943318, abs=1e-6)


def test_deflated_sharpe_paper_example() -> None:
    """Bailey & López de Prado (2014), JPM, 'A Numerical Example'.

    Annualised SR* = 2.5 on T=1250 daily observations (250 per year),
    N=100 independent trials, V[{SR}] = 1/(2*250) = 0.002 (non-annualised),
    skew = -3, Pearson kurtosis = 10.

    Per-period SR* = 2.5 / sqrt(250).
    SR0 = sqrt(V) * ((1-gamma) * Phi^{-1}(1-1/N)
          + gamma * Phi^{-1}(1-1/(N e))) ~= 0.1132
    DSR = Phi[(SR*-SR0) * sqrt(T-1)
          / sqrt(1 - g3*SR* + ((g4-1)/4)*SR*^2)] = 0.9004
    """
    sr_daily = 2.5 / math.sqrt(250.0)
    var_sr = 1.0 / (2.0 * 250.0)
    dsr = deflated_sharpe(
        sr_daily,
        n_trials=100,
        n_obs=1250,
        skew=-3.0,
        kurtosis=10.0,
        var_sr=var_sr,
    )
    assert dsr == pytest.approx(0.9004, abs=5e-5)


def test_deflated_sharpe_paper_secondary_n46() -> None:
    # Same example: N=46 yields DSR ≈ 0.9505, just above 95%.
    sr_daily = 2.5 / math.sqrt(250.0)
    var_sr = 1.0 / (2.0 * 250.0)
    dsr = deflated_sharpe(
        sr_daily,
        n_trials=46,
        n_obs=1250,
        skew=-3.0,
        kurtosis=10.0,
        var_sr=var_sr,
    )
    assert dsr == pytest.approx(0.9505, abs=5e-5)


def test_deflated_sharpe_paper_secondary_normal_n88() -> None:
    # Same Sharpe, Normal returns (skew 0, kurtosis 3): DSR ≈ 0.9505 at N=88.
    sr_daily = 2.5 / math.sqrt(250.0)
    var_sr = 1.0 / (2.0 * 250.0)
    dsr = deflated_sharpe(
        sr_daily,
        n_trials=88,
        n_obs=1250,
        skew=0.0,
        kurtosis=3.0,
        var_sr=var_sr,
    )
    assert dsr == pytest.approx(0.9505, abs=5e-5)


def test_welch_rejects_tiny_samples() -> None:
    with pytest.raises(ScoutError, match="at least two"):
        welch_ttest([1.0], [1.0, 2.0])


def test_wilson_rejects_bad_n() -> None:
    with pytest.raises(ScoutError, match="n > 0"):
        wilson_ci(0, 0)
