import dataclasses

import numpy as np
import pytest

from lab.framework.data import synthetic
from strategy import (PARAMS, beta_ivol, book_from_scores, build_features, cohort_book, daily_returns,
                      gbm_params, liq30, target_weights, tradfi_features)


def _view(T=915, seed=0):
    """Synthetic crypto-like view: MKT renamed BTCUSDT, dispersed quote volume, carried FRED series."""
    d = synthetic()
    rng = np.random.default_rng(seed)
    N = len(d.instruments)
    names = tuple(list(d.instruments[:-1]) + ["BTCUSDT"])
    sl = slice(0, T)
    dv = np.exp(rng.normal(16, 1.5, size=(1, N)) + rng.normal(0, 0.3, size=(T, N)))
    vix = np.exp(np.log(18) + np.cumsum(rng.normal(0, 0.05, T)))
    ry = 0.5 + np.cumsum(rng.normal(0, 0.03, T))
    wk = np.arange(T) % 7 >= 5                  # weekend rows carry the last value
    for s in (vix, ry):
        for t in np.flatnonzero(wk):
            if t > 0:
                s[t] = s[t - 1]
    uni = np.ones((T, N), bool)
    uni[:, :8] = False                          # 23 members
    uni[700:, 10] = False                       # a member that leaves the universe
    return dataclasses.replace(
        d, dates=np.asarray(d.dates)[sl], instruments=names, asset_class=tuple(d.asset_class[:N]),
        ret_co=np.asarray(d.ret_co)[sl], ret_oc=np.asarray(d.ret_oc)[sl], tradable=np.asarray(d.tradable)[sl],
        listed=np.asarray(d.listed)[sl], delisting=np.asarray(d.delisting)[sl], close=np.asarray(d.close)[sl],
        dollar_volume=dv, universe=uni, extras={},
        series={"fred_macro.VIXCLS": vix, "fred_macro.DFII10": ry}, cash_ret=d.cash_ret)


def test_integration_book_mechanics():
    v = _view()
    X, scored, uni, dvix5, dry5 = build_features(v)
    assert X.shape == (915, len(v.instruments), 6)
    assert not scored[:45].any() and scored[45:].sum(axis=1).min() >= 20
    # date-level features identical across scored coins of a date, z-scored coin features centred
    row = 300
    m = scored[row]
    assert np.allclose(X[row, m, 3], X[row, m, 3][0]) and abs(np.mean(X[row, m, 0])) < 1e-6
    assert np.all(np.isnan(X[row, ~m]))
    w = target_weights(v, PARAMS)
    g = w.sum(axis=1)
    assert np.all(w >= 0) and np.all(g <= 1.0 + 1e-9)
    assert np.all(w[~uni] == 0.0)
    first = int(np.argmax(g > 0))
    assert first > 45                            # cash until the first model fit
    assert np.allclose(g[first + 4:], 1.0)       # always invested once five cohorts are live
    # point-in-time: truncating the view does not change earlier rows
    cut = 760
    vc = _view()
    vc = dataclasses.replace(
        vc, dates=vc.dates[:cut], ret_co=vc.ret_co[:cut], ret_oc=vc.ret_oc[:cut], tradable=vc.tradable[:cut],
        listed=vc.listed[:cut], delisting=vc.delisting[:cut], close=vc.close[:cut],
        dollar_volume=vc.dollar_volume[:cut], universe=vc.universe[:cut],
        series={k: s[:cut] for k, s in vc.series.items()})
    wc = target_weights(vc, PARAMS)
    assert np.allclose(wc, w[:cut], atol=1e-12)
    # the n_estimators parameter reaches the model
    w50 = target_weights(v, {"n_estimators": 50, "top": 0.2})
    assert not np.array_equal(w50, w)


# ------------------------------------------------------------------ returns and regression
def test_daily_returns_needs_two_prices():
    close = np.array([[1.0], [np.nan], [1.0], [1.1]])
    listed = np.ones((4, 1), bool)
    ret_co = np.array([[0.0], [0.0], [0.0], [0.05]])
    ret_oc = np.array([[0.0], [0.0], [0.0], [0.0]])
    r = daily_returns(ret_co, ret_oc, close, listed)
    assert np.isnan(r[0, 0]) and np.isnan(r[1, 0]) and np.isnan(r[2, 0])
    assert r[3, 0] == pytest.approx(0.05)


def test_beta_exact_linear_relation_and_min_pairs():
    rng = np.random.default_rng(0)
    T = 80
    f = rng.normal(0, 0.03, T)
    f[0] = np.nan
    noise = rng.normal(0, 0.01, T)
    r = np.column_stack([2.0 * f + 0.001, 0.5 * f + noise, f])
    beta, ivol = beta_ivol(r, f)
    # 45 valid pairs are first available on row 45 (row 0 is NaN)
    assert np.all(np.isnan(beta[:45]))
    assert np.isfinite(beta[45]).all()
    assert beta[70, 0] == pytest.approx(2.0, abs=1e-9)
    assert ivol[70, 0] == pytest.approx(0.0, abs=1e-7)
    # second column against a direct OLS over the last 60 rows
    y = r[11:71, 1]
    x = f[11:71]
    b = np.polyfit(x, y, 1)[0]
    res = y - np.polyval(np.polyfit(x, y, 1), x)
    assert beta[70, 1] == pytest.approx(b, rel=1e-8)
    assert ivol[70, 1] == pytest.approx(np.sqrt(np.sum(res ** 2) / (60 - 2)), rel=1e-6)


def test_beta_skips_missing_rows_and_point_in_time():
    rng = np.random.default_rng(1)
    T = 120
    f = rng.normal(0, 0.03, T)
    r = (1.5 * f + rng.normal(0, 0.01, T))[:, None].copy()
    r[50:70, 0] = np.nan                       # 20 missing rows inside the window
    beta, _ = beta_ivol(r, f)
    assert np.isfinite(beta[49, 0])            # 50 pairs in rows 0..49
    assert np.isnan(beta[85, 0])               # only 40 pairs in rows 26..85 -> missing
    assert np.isnan(beta[113, 0])              # 44 pairs in rows 54..113
    assert np.isfinite(beta[114, 0])           # 45 pairs in rows 55..114
    b_full, _ = beta_ivol(r, f)
    b_cut, _ = beta_ivol(r[:100], f[:100])
    assert np.array_equal(b_full[:100], b_cut, equal_nan=True)


# ------------------------------------------------------------------ liquidity
def test_liq30_median_over_last_30_rows():
    v = np.arange(1.0, 61.0)[:, None]
    v[40, 0] = np.nan
    out = liq30(v)
    assert out[0, 0] == pytest.approx(np.log(1.0))
    assert out[29, 0] == pytest.approx(np.log(np.median(np.arange(1.0, 31.0))))
    w = np.delete(np.arange(31.0, 61.0), 40 - 30)
    assert out[59, 0] == pytest.approx(np.log(np.median(w)))
    z = liq30(np.zeros((5, 1)))
    assert np.all(np.isnan(z))


# ------------------------------------------------------------------ TradFi date features
def test_tradfi_counts_published_observations_not_rows():
    # 7 business observations with weekend carries in between; DFII10 repeats a value once
    vix = np.array([20, 21, 22, 22, 22, 23, 24, 25, 26, 26, 26, 30.0])
    ry = np.array([0.5, 0.6, 0.7, 0.7, 0.7, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0, 1.1])
    # publications: rows 0,1,2,5(ry repeats, vix new),6,7,8,11 ; rows 3,4,9,10 are carries
    dvix5, dry5, lvix = tradfi_features(vix, ry)
    assert np.all(np.isnan(dvix5[:7]))         # fewer than 6 observations before row 7
    assert dvix5[7] == pytest.approx(np.log(25) - np.log(20))
    assert dry5[7] == pytest.approx(0.9 - 0.5)
    assert dvix5[8] == pytest.approx(np.log(26) - np.log(21))
    assert dvix5[10] == pytest.approx(np.log(26) - np.log(21))  # carried over the weekend
    assert dry5[8] == pytest.approx(1.0 - 0.6)
    assert dvix5[11] == pytest.approx(np.log(30) - np.log(22))
    assert dry5[11] == pytest.approx(1.1 - 0.7)
    assert lvix[11] == pytest.approx(np.log(30))


def test_tradfi_leading_nan():
    vix = np.array([np.nan, np.nan, 20, 21, 22, 23, 24, 25.0])
    ry = np.array([np.nan, np.nan, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    dvix5, dry5, lvix = tradfi_features(vix, ry)
    assert np.isnan(lvix[0])
    assert np.all(np.isnan(dvix5[:7]))
    assert dvix5[7] == pytest.approx(np.log(25) - np.log(20))


# ------------------------------------------------------------------ book
def test_cohort_book_sums_last_five_cohorts():
    c = np.zeros((8, 2))
    c[0, 0] = 1.0
    c[2, 1] = 1.0
    b = cohort_book(c, 5)
    assert np.allclose(b[:5, 0], 0.2) and b[5, 0] == 0.0
    assert b[1, 1] == 0.0 and np.allclose(b[2:7, 1], 0.2) and b[7, 1] == 0.0


def _scores(T, N, rng):
    s = rng.normal(size=(T, N))
    s[:3] = np.nan                              # no scores yet: cash
    return s


def test_book_top_fraction_hold_and_universe_exit():
    rng = np.random.default_rng(2)
    T, N = 12, 20
    s = _scores(T, N, rng)
    uni = np.ones((T, N), bool)
    w = book_from_scores(s, uni, 0.2)
    assert np.all(w[:3] == 0.0)
    assert np.all(w >= 0) and np.all(w.sum(axis=1) <= 1.0 + 1e-12)
    assert w[7].sum() == pytest.approx(1.0)     # five live cohorts
    assert w[4].sum() == pytest.approx(0.4)     # two live cohorts
    top4 = np.argsort(-s[3])[:4]
    assert np.allclose(w[3, top4], 0.05) and np.count_nonzero(w[3]) == 4
    w30 = book_from_scores(s, uni, 0.3)
    assert np.count_nonzero(w30[3]) == 6        # top parameter changes the cohort
    uni2 = uni.copy()
    uni2[5, top4[0]] = False
    w2 = book_from_scores(s, uni2, 0.2)
    assert w2[5, top4[0]] == 0.0                # closed at the first row outside the universe


def test_fewer_than_ten_scored_names_is_cash():
    rng = np.random.default_rng(3)
    T, N = 6, 20
    s = rng.normal(size=(T, N))
    s[2, 9:] = np.nan                           # only 9 scored names on row 2
    w = book_from_scores(s, np.ones((T, N), bool), 0.2)
    assert w[1].sum() == pytest.approx(0.4)
    assert w[2].sum() == pytest.approx(0.4)     # cohorts of rows 0, 1 live; row 2's cohort is cash
    assert w[3].sum() == pytest.approx(0.6)


def test_params():
    assert set(PARAMS) == {"n_estimators", "top"}
    p = gbm_params({"n_estimators": 50, "top": 0.2})
    assert p["n_estimators"] == 50 and p["num_leaves"] == 8 and p["min_child_samples"] == 200
    assert p["learning_rate"] == 0.05 and p["subsample"] == 0.8
    assert callable(target_weights)
