import dataclasses

import numpy as np
import pytest
from lab.framework.data import synthetic

from strategy import PARAMS, target_weights
from strategy import build_book, build_features, market_return, stock_features, vix_state, mkt21_state
from strategy import _compound, _rsum


def test_params_match_card():
    assert PARAMS == {"n_estimators": 100, "top": 0.2}
    assert callable(target_weights)


# ------------------------------------------------------------------ book
def _scores(T, N, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(T, N))


def test_book_flat_below_min_names_and_before_scores():
    T, N = 6, 60
    score = np.full((T, N), np.nan)
    score[3:, :] = _scores(3, N)
    uni = np.ones((T, N), bool)
    uni[3, 20:] = False                       # only 20 members scored on row 3 -> flat
    decision = np.array([1, 0, 0, 1, 0, 1], bool)
    w = build_book(score, uni, decision, 0.2)
    assert np.all(w[0] == 0.0)                # no score yet: flat
    assert np.all(np.isnan(w[1])) and np.all(np.isnan(w[2]))
    assert np.all(w[3] == 0.0)                # 20 < 50 scored members
    assert np.isclose(w[5].sum(), 1.0)
    assert (w[5] > 0).sum() == 12             # K = round(0.2 * 60)
    assert np.all(w[5][w[5] > 0] == pytest.approx(1 / 12))


def test_book_top_k_and_buffer():
    N = 100
    uni = np.ones((2, N), bool)
    decision = np.array([True, True])
    s0 = -np.arange(N, dtype=float)           # rank = column + 1
    s1 = s0.copy()
    # at t=1 move held names 0..19 down: cols 0..9 to ranks 31..40 (inside 2*0.2*100 = 40), cols 10..19 out
    new_rank = np.concatenate([np.arange(30, 40), np.arange(60, 70), np.arange(0, 30), np.arange(40, 60),
                               np.arange(70, 100)])
    s1 = -new_rank.astype(float)
    score = np.vstack([s0, s1])
    w = build_book(score, uni, decision, 0.2)
    held0 = np.flatnonzero(w[0] > 0)
    assert list(held0) == list(range(20))
    held1 = set(np.flatnonzero(w[1] > 0))
    assert len(held1) == 20
    assert set(range(10)) <= held1            # buffer keeps cols 0..9 (ranks 31..40)
    assert not (set(range(10, 20)) & held1)   # ranks 61..70 dropped
    # filled by the 10 best not-held names: cols 20..29 (ranks 1..10)
    assert set(range(20, 30)) <= held1
    assert np.all(w[1][w[1] > 0] == pytest.approx(1 / 20))


def test_buffer_keeps_at_most_k():
    N = 100
    uni = np.ones((2, N), bool)
    decision = np.array([True, True])
    s0 = -np.arange(N, dtype=float)           # hold cols 0..19
    s1 = s0.copy()
    s1[50:70] = 100.0 + np.arange(20)         # 20 new names jump to the top
    w = build_book(np.vstack([s0, s1]), uni, decision, 0.2)
    held1 = set(np.flatnonzero(w[1] > 0))
    # old names 0..19 now rank 21..40: all inside the buffer, they are kept (at most K = 20)
    assert held1 == set(range(20))


def test_leaver_closed_between_decisions():
    N = 60
    T = 4
    score = np.tile(-np.arange(N, dtype=float), (T, 1))
    uni = np.ones((T, N), bool)
    uni[2, 0] = False                         # held col 0 leaves on row 2
    decision = np.array([True, False, False, True])
    w = build_book(score, uni, decision, 0.2)
    assert np.all(np.isnan(w[1]))
    assert w[2, 0] == 0.0
    assert np.isnan(w[2, 1:]).all()
    assert np.nansum(w[3]) == pytest.approx(1.0)
    for row in w:
        assert np.nansum(np.abs(row)) <= 1.0 + 1e-12
        assert np.all(row[np.isfinite(row)] >= 0)


def test_top_param_changes_book():
    N = 100
    score = _scores(1, N)
    uni = np.ones((1, N), bool)
    d = np.array([True])
    a = build_book(score, uni, d, 0.1)
    b = build_book(score, uni, d, 0.3)
    assert (a[0] > 0).sum() == 10 and (b[0] > 0).sum() == 30


def test_non_member_never_bought():
    N = 80
    score = _scores(1, N, 3)
    uni = np.ones((1, N), bool)
    best = int(np.argmax(score[0]))
    uni[0, best] = False
    w = build_book(score, uni, np.array([True]), 0.2)
    assert w[0, best] == 0.0
    assert (w[0] > 0).sum() == 16             # round(0.2 * 79) = 16


def _synthetic_with_vix(with_universe=True, with_vix=True):
    d = synthetic()
    ser = dict(d.series)
    ser.pop("fred_macro.VIXCLS", None)
    if with_vix:
        ser["fred_macro.VIXCLS"] = 20.0 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.05, len(d.dates))))
    uni = np.asarray(d.listed, bool).copy() if with_universe else None
    return dataclasses.replace(d, universe=uni, series=ser)


def test_features_complete_on_synthetic():
    d = _synthetic_with_vix()
    X, uni = build_features(d)
    assert X.shape == (len(d.dates), len(d.instruments), 6)
    comp = np.isfinite(X).all(axis=2)
    assert not (comp & ~uni).any()                       # never scores a non-member
    assert not comp[:270].any()                          # 21 + 252 sessions of market history needed
    assert comp[400:][uni[400:]].mean() > 0.9
    # date-level states are identical across members of a date, z-scored features have mean ~0
    t = 1000
    row = X[t][comp[t]]
    assert np.allclose(row[:, 4], row[0, 4]) and np.allclose(row[:, 5], row[0, 5])
    assert abs(row[:, 0].mean()) < 1e-4


def test_missing_inputs_raise():
    with pytest.raises(Exception):
        build_features(_synthetic_with_vix(with_universe=False))
    with pytest.raises(KeyError):
        build_features(_synthetic_with_vix(with_vix=False))


# ------------------------------------------------------------------ features
def test_rsum_and_compound():
    x = np.arange(1.0, 7.0)[:, None]
    s = _rsum(x, 3)
    assert np.isnan(s[:2]).all()
    assert s[2, 0] == 6 and s[5, 0] == 15
    r = np.array([0.1, 0.1, np.nan, 0.1, 0.1, 0.1])[:, None]
    c = _compound(r, 2)
    assert np.isnan(c[0, 0]) and c[1, 0] == pytest.approx(0.21)
    assert np.isnan(c[2, 0]) and np.isnan(c[3, 0]) and c[5, 0] == pytest.approx(0.21)


def test_market_return_members_only():
    r = np.array([[0.01, 0.03, 0.5], [np.nan, 0.02, 0.5]])
    uni = np.array([[True, True, False], [True, True, False]])
    m = market_return(r, uni)
    assert m[0] == pytest.approx(0.02) and m[1] == pytest.approx(0.02)


def test_beta_res_ivol_exact():
    rng = np.random.default_rng(1)
    T = 200
    m = rng.normal(0, 0.01, T)
    eps = rng.normal(0, 0.02, T)
    r = np.column_stack([1.5 * m + eps, 0.5 * m])
    r[:150, 1] = np.nan                       # late listing: only 50 valid days, < 100
    res5, res21x, ivol, beta = stock_features(r, m)
    t = T - 1
    xs, ys = m[t - 125:t + 1], r[t - 125:t + 1, 0]
    b = np.polyfit(xs, ys, 1)[0]
    assert beta[t, 0] == pytest.approx(b, rel=1e-8)
    assert np.isnan(beta[t, 1]) and np.isnan(res5[t, 1]) and np.isnan(ivol[t, 1])
    assert np.isnan(beta[124, 0]) and np.isfinite(beta[125, 0])
    rs = np.prod(1 + r[t - 4:t + 1, 0]) - 1
    rm = np.prod(1 + m[t - 4:t + 1]) - 1
    assert res5[t, 0] == pytest.approx(rs - b * rm, rel=1e-8)
    rs21 = np.prod(1 + r[t - 25:t - 4, 0]) - 1
    rm21 = np.prod(1 + m[t - 25:t - 4]) - 1
    assert res21x[t, 0] == pytest.approx(rs21 - b * rm21, rel=1e-8)
    z = r[t - 62:t + 1, 0] - b * m[t - 62:t + 1]
    assert ivol[t, 0] == pytest.approx(np.std(z, ddof=1), rel=1e-6)


def test_features_point_in_time():
    rng = np.random.default_rng(2)
    T, N = 400, 5
    m = rng.normal(0, 0.01, T)
    r = m[:, None] + rng.normal(0, 0.02, (T, N))
    full = stock_features(r, m)
    cut = 300
    part = stock_features(r[:cut + 1], m[:cut + 1])
    for a, b in zip(full, part):
        np.testing.assert_allclose(a[:cut + 1], b, equal_nan=True)
    v = 15 + rng.random(T) * 10
    np.testing.assert_allclose(vix_state(v)[:cut + 1], vix_state(v[:cut + 1]), equal_nan=True)
    np.testing.assert_allclose(mkt21_state(m)[:cut + 1], mkt21_state(m[:cut + 1]), equal_nan=True)


def test_vix_state_values():
    v = np.concatenate([[np.nan] * 3, np.full(251, 20.0), [40.0]])
    s = vix_state(v)
    assert np.isnan(s[:254]).all()            # window must hold 252 known values
    assert s[254] == pytest.approx(np.log(2.0))


def test_mkt21_state_value():
    rng = np.random.default_rng(4)
    m = rng.normal(0, 0.01, 300)
    s = mkt21_state(m)
    m21 = np.array([np.prod(1 + m[i - 20:i + 1]) - 1 if i >= 20 else np.nan for i in range(300)])
    t = 299
    expect = m21[t] / np.std(m21[t - 251:t + 1], ddof=1)
    assert s[t] == pytest.approx(expect, rel=1e-8)
    assert np.isnan(s[270]) and np.isfinite(s[271])   # first full 252-window of 21-day returns ends at 20 + 251

