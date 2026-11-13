import dataclasses
import functools

import numpy as np
import pytest
from lab.framework.data import synthetic

from strategy import PARAMS, target_weights
from strategy import (rolling_sum, top_n_universe, daily_returns, coin_features, legs, cohort_book,
                      model_scores, HOLD, COHORT_LEG)


def _slice_view(d, n):
    T = len(d.dates)
    kw = {}
    for name, v in dataclasses.asdict(d).items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == T:
            kw[name] = v[:n]
        elif isinstance(v, dict):
            kw[name] = {k: (a[:n] if isinstance(a, np.ndarray) and a.ndim >= 1 and a.shape[0] == T else a)
                          for k, a in v.items()}
    return dataclasses.replace(d, **kw)


@functools.lru_cache(maxsize=None)
def _view():
    d = synthetic()
    T = len(d.dates)
    names = ("BTCUSDT.P",) + tuple(d.instruments[1:])
    vix = 20.0 + np.random.default_rng(0).normal(0, 1, T)
    vix[:10] = np.nan                                          # leading NaNs of the macro series
    d = dataclasses.replace(d, instruments=names, series={"fred_macro.VIXCLS": vix})
    return _slice_view(d, 900)


@functools.lru_cache(maxsize=None)
def _scores(n_est):
    p = dict(PARAMS)
    p["n_estimators"] = n_est
    return model_scores(_view(), p)


def test_end_to_end_book():
    d = _view()
    score, member = _scores(PARAMS["n_estimators"])
    W = target_weights(d, PARAMS)
    T, N = W.shape
    assert W.shape == (len(d.dates), len(d.instruments))
    assert np.all(np.isfinite(W))
    assert member.sum(axis=1).max() == 20
    scored = np.flatnonzero(np.isfinite(score).sum(axis=1) >= 10)
    assert scored.size > 50                                   # model reached its first fit on 900 rows
    first = scored[0]
    assert np.allclose(W[:first], 0.0)                        # cash before the first model fit
    assert np.abs(W[first]).sum() > 0
    assert np.all(np.abs(W).sum(axis=1) <= 1.0 + 1e-9)
    assert np.all(W[~member] == 0.0)                          # only members are held
    assert np.isfinite(score[~member]).sum() == 0
    full = np.abs(W[first + HOLD:]).sum(axis=1)
    assert np.allclose(full.max(), 1.0)
    # point-in-time: truncating the view does not change earlier rows
    cut = first + 40
    Wc = target_weights(_slice_view(d, cut), PARAMS)
    assert np.allclose(Wc, W[:cut])


def test_n_estimators_changes_scores_and_top_changes_book():
    s1, _ = _scores(100)
    s2, _ = _scores(50)
    both = np.isfinite(s1) & np.isfinite(s2)
    assert both.any() and not np.allclose(s1[both], s2[both])
    lg1, sh1, k1 = legs(s1, 0.2)
    lg2, sh2, k2 = legs(s1, 0.25)
    assert (k2 >= k1).all() and (k2 > k1).any()


def test_params_match_card():
    assert PARAMS == {"n_estimators": 100, "top": 0.2}


def test_rolling_sum_short_view_and_values():
    x = np.arange(1.0, 6.0)[:, None]          # 1..5
    out = rolling_sum(x, 3)
    assert np.all(np.isnan(out[:2]))
    assert out[2:, 0].tolist() == [6.0, 9.0, 12.0]
    assert np.all(np.isnan(rolling_sum(x, 30)))   # view shorter than the window: NaN, no error


def test_top_n_universe_ranking_and_late_listing():
    T, N = 40, 4
    dv = np.tile(np.array([4.0, 3.0, 2.0, 1.0]), (T, 1))
    listed = np.ones((T, N), dtype=bool)
    dv[:20, 3] = np.nan                      # coin 3 listed late with huge volume
    listed[:20, 3] = False
    dv[20:, 3] = 100.0
    u = top_n_universe(dv, listed, 2, 5)
    assert not u[:4].any()                   # window not full yet
    assert u[4].tolist() == [True, True, False, False]
    # coin 3 needs 5 full days of volume after listing (rows 20..24)
    assert u[23].tolist() == [True, True, False, False]
    assert u[24].tolist() == [True, False, False, True]
    # not listed on day t -> not a member
    listed2 = listed.copy()
    listed2[30, 0] = False
    u2 = top_n_universe(dv, listed2, 2, 5)
    assert u2[30].tolist() == [False, True, False, True]


def test_daily_returns_validity():
    ret_co = np.array([[0.0], [0.1], [0.0]])
    ret_oc = np.array([[0.0], [0.1], [0.05]])
    listed = np.array([[True], [True], [True]])
    r, v = daily_returns(ret_co, ret_oc, listed)
    assert v[:, 0].tolist() == [False, True, True]
    assert r[1, 0] == pytest.approx(0.21)
    assert r[2, 0] == pytest.approx(0.05)


def test_coin_features_by_hand():
    T, N = 70, 3
    rng = np.random.default_rng(1)
    rb = rng.normal(0, 0.03, T)
    r = np.zeros((T, N))
    r[:, 0] = rb                              # BTC
    r[:, 1] = 2.0 * rb + 0.001                # beta exactly 2
    r[:, 2] = rng.normal(0, 0.05, T)
    valid = np.ones((T, N), dtype=bool)
    valid[0] = False
    r[0] = 0.0
    valid[:65, 2] = valid[:65, 2]             # all valid for coin 2 too
    ret21, ret5, vol30, beta, btc21, elig = coin_features(r, valid, 0)
    t = T - 1
    assert ret21[t, 1] == pytest.approx(np.prod(1 + r[t - 20:t + 1, 1]) - 1)
    assert ret5[t, 2] == pytest.approx(np.prod(1 + r[t - 4:t + 1, 2]) - 1)
    assert vol30[t, 2] == pytest.approx(np.std(np.log1p(r[t - 29:t + 1, 2]), ddof=1))
    assert beta[t, 1] == pytest.approx(2.0)
    assert beta[t, 0] == 1.0
    assert btc21[t] == pytest.approx(ret21[t, 0])
    # 60 valid returns needed: row 0 invalid, so the first eligible row is 60
    assert not elig[59].any() and elig[60].all()
    assert np.isnan(beta[59]).all()


def test_coin_features_missing_history():
    T, N = 80, 2
    r = np.full((T, N), 0.01)
    valid = np.ones((T, N), dtype=bool)
    valid[0] = False
    valid[30, 1] = False                       # a gap in coin 1
    r[~valid] = 0.0
    *_, elig = coin_features(r, valid, 0)
    # coin 0 (BTC): eligible from row 60; coin 1 has a gap at row 30, so it needs rows 31..90 -> never here
    assert not elig[59, 0] and elig[60:, 0].all()
    assert not elig[:, 1].any()


def test_legs_sizes_and_threshold():
    T, N = 3, 20
    score = np.full((T, N), np.nan)
    score[0, :9] = np.arange(9.0)               # 9 scored -> no book
    score[1, :10] = np.arange(10.0)             # 10 scored, top 0.2 -> 2
    score[2, :] = np.arange(20.0)               # 20 scored, top 0.2 -> 4
    lg, sh, k = legs(score, 0.2)
    assert k.tolist() == [0, 2, 4]
    assert not lg[0].any() and not sh[0].any()
    assert np.flatnonzero(lg[1]).tolist() == [8, 9]
    assert np.flatnonzero(sh[1]).tolist() == [0, 1]
    assert np.flatnonzero(lg[2]).tolist() == [16, 17, 18, 19]
    assert np.flatnonzero(sh[2]).tolist() == [0, 1, 2, 3]
    _, _, k15 = legs(score, 0.15)              # 0.15*10=1.5 -> at least 2; 0.15*20=3
    assert k15.tolist() == [0, 2, 3]
    _, _, k25 = legs(score, 0.25)              # 2.5 -> 3 (half up); 5
    assert k25.tolist() == [0, 3, 5]


def test_cohort_book_holds_five_days_and_empties_slot():
    T, N = 12, 4
    lg = np.zeros((T, N), dtype=bool)
    sh = np.zeros((T, N), dtype=bool)
    k = np.zeros(T, dtype=np.int64)
    lg[2, 0] = True
    sh[2, 1] = True
    k[2] = 1
    alive = np.ones((T, N), dtype=bool)
    W = cohort_book(lg, sh, k, alive)
    assert np.allclose(W[2:2 + HOLD, 0], COHORT_LEG)
    assert np.allclose(W[2:2 + HOLD, 1], -COHORT_LEG)
    assert np.allclose(W[:2], 0.0) and np.allclose(W[2 + HOLD:], 0.0)
    # coin 0 leaves the universe on row 4 and comes back on row 5: slot stays empty
    alive2 = alive.copy()
    alive2[4, 0] = False
    W2 = cohort_book(lg, sh, k, alive2)
    assert np.allclose(W2[2:4, 0], COHORT_LEG)
    assert np.allclose(W2[4:, 0], 0.0)
    assert np.allclose(W2[2:2 + HOLD, 1], -COHORT_LEG)


def test_cohort_book_netting_and_gross():
    T, N = 10, 20
    rng = np.random.default_rng(3)
    score = rng.normal(size=(T, N))
    lg, sh, k = legs(score, 0.25)
    W = cohort_book(lg, sh, k, np.ones((T, N), dtype=bool))
    gross = np.abs(W).sum(axis=1)
    assert np.all(gross <= 1.0 + 1e-12)
    assert np.allclose(W.sum(axis=1), 0.0)
    assert gross[HOLD - 1] > 0.2
