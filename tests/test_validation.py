import sqlite3
from datetime import date

import numpy as np
import pytest

from qlab.validation.metrics import max_drawdown, summary
from qlab.validation.registry import TrialRegistry, config_hash
from qlab.validation.stats import (bootstrap_ci, deflated_sharpe, expected_max_sharpe, pbo_cscv,
                                   probabilistic_sharpe, sharpe, stationary_bootstrap_indices)
from qlab.validation.vault import Vault, VaultError


def test_probabilistic_sharpe_extremes():
    rng = np.random.default_rng(0)
    assert probabilistic_sharpe(rng.normal(0.002, 0.01, 2500)) > 0.99
    assert probabilistic_sharpe(rng.normal(-0.002, 0.01, 2500)) < 0.01
    assert 0.2 < probabilistic_sharpe(rng.normal(0.0, 0.01, 2500)) < 0.8


def test_expected_max_sharpe_matches_monte_carlo():
    rng = np.random.default_rng(1)
    n_days, n_trials = 1000, 200
    maxima = [max(sharpe(rng.normal(0, 0.01, n_days)) for _ in range(n_trials)) for _ in range(40)]
    expected = expected_max_sharpe(n_trials, 1.0 / n_days)
    assert np.mean(maxima) == pytest.approx(expected, rel=0.1)


def test_deflated_sharpe_penalizes_many_trials():
    r = np.random.default_rng(2).normal(0.0006, 0.01, 2500)
    one = deflated_sharpe(r, 1, 1 / 2500)
    many = deflated_sharpe(r, 1000, 1 / 2500)
    assert one == pytest.approx(probabilistic_sharpe(r))
    assert many < one - 0.3


def test_pbo_noise_is_near_half_and_skill_is_near_zero():
    rng = np.random.default_rng(3)
    noise = rng.normal(0, 0.01, (2000, 50))
    pbo_noise, logits = pbo_cscv(noise, n_blocks=10)
    assert 0.3 < pbo_noise < 0.7 and len(logits) == 252
    skilled = noise.copy()
    skilled[:, 7] += 0.002
    assert pbo_cscv(skilled, n_blocks=10)[0] < 0.05


def test_stationary_bootstrap_indices():
    rng = np.random.default_rng(4)
    idx = stationary_bootstrap_indices(5000, 20, rng)
    assert idx.min() >= 0 and idx.max() < 5000
    breaks = np.count_nonzero(np.diff(idx) != 1)
    assert 5000 / (breaks + 1) == pytest.approx(20, rel=0.25)


def test_bootstrap_ci_covers_the_mean():
    rng = np.random.default_rng(5)
    x = rng.normal(0.001, 0.01, 2000)
    est, lo, hi = bootstrap_ci(x, np.mean, n_boot=400, mean_block=5)
    assert lo < 0.001 < hi and lo < est < hi
    assert (hi - lo) == pytest.approx(2 * 1.645 * 0.01 / np.sqrt(2000), rel=0.3)
    pair = np.column_stack([x, x - 0.0005])
    est, lo, hi = bootstrap_ci(pair, lambda d: d[:, 0].mean() - d[:, 1].mean(), n_boot=200)
    assert est == pytest.approx(0.0005) and lo == pytest.approx(0.0005)  # paired rows move together


def test_registry_is_append_only(tmp_path):
    reg = TrialRegistry(tmp_path / "trials.sqlite")
    reg.record("candidates", {"grid": "v1"}, n_configs=3000)
    reg.record("methodology_eval", {"K": 2})
    reg.record("methodology_eval", {"K": 3})
    assert reg.total_configs() == 3002 and reg.n_meth == 2
    with sqlite3.connect(reg.path) as con, pytest.raises(sqlite3.DatabaseError):
        con.execute("DELETE FROM trials")
    with sqlite3.connect(reg.path) as con, pytest.raises(sqlite3.DatabaseError):
        con.execute("UPDATE trials SET n_configs = 1")
    with pytest.raises(ValueError):
        reg.record("peek_at_holdout", {})
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})


def test_vault(tmp_path):
    v = Vault(date(2019, 12, 31), tmp_path / "frozen.lock", tmp_path / "vault.log")
    dates = np.array(["2019-12-30", "2019-12-31", "2020-01-02"], dtype="datetime64[D]")
    assert v.last_visible_index(dates) == 2
    v.check_dev_only(dates[:2])
    with pytest.raises(VaultError):
        v.check_dev_only(dates)
    with pytest.raises(VaultError):
        v.open_final("abc", "snap1", "c0")      # not frozen
    v.freeze("abc")
    with pytest.raises(VaultError):
        v.open_final("xyz", "snap1", "c0")      # different methodology
    v.open_final("abc", "snap1", "c0")
    with pytest.raises(VaultError):
        v.open_final("abc", "snap1", "c1")      # second look at the same data
    assert len(v.log()) == 1


def test_metrics():
    assert max_drawdown(np.array([1.0, 1.2, 0.9, 1.0, 1.3])) == (pytest.approx(0.25), 2)
    r = np.full(504, 0.0004)
    s = summary(r, cash_ret=np.full(504, 0.0001), turnover=np.full(504, 0.01),
                costs=np.full(504, 1e-5), exposure=np.ones(504))
    assert s["cagr"] == pytest.approx(1.0004 ** 252 - 1)
    assert s["max_drawdown"] == 0 and s["turnover_annual"] == pytest.approx(2.52)
    assert s["costs_bps_annual"] == pytest.approx(25.2)


def test_spa_pvalue():
    from qlab.validation.stats import spa_pvalue
    rng = np.random.default_rng(9)
    assert spa_pvalue(rng.normal(0.002, 0.01, (2000, 2)), n_boot=300) < 0.05
    assert spa_pvalue(rng.normal(0.0, 0.01, (2000, 3)), n_boot=300) > 0.1
