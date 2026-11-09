import numpy as np
import pytest

from strategy import target_weights, PARAMS
from strategy import (_day_changes, _factor_shock, _exposures, _legs, _book, _close_exits, _gate,
                      _stock_returns, _market, _compute, _ffill, _events_from_legs, REG_DAYS)


def test_params_match_card():
    assert PARAMS == {"shock_z": 1.5, "top": 0.2}
    assert callable(target_weights)


def test_day_changes_gap_goes_to_next_value_day():
    lv = np.array([np.nan, 1.0, np.nan, 1.5, 1.5, 2.0])
    d = _day_changes(lv)
    # d[s] = known level at s+1 minus known level at s; the last day is not known yet
    assert np.isnan(d[0]) and np.isnan(d[-1])
    np.testing.assert_allclose(d[1:5], [0.0, 0.5, 0.0, 0.5])
    np.testing.assert_allclose(_ffill(lv)[1:], [1.0, 1.0, 1.5, 1.5, 2.0])


def test_stock_returns_and_market():
    co = np.array([[0.0, 0.0], [0.01, 0.02], [0.0, -0.01], [0.02, 0.0]])
    oc = np.array([[0.0, 0.0], [0.01, 0.0], [0.03, 0.01], [0.0, 0.01]])
    listed = np.array([[1, 0], [1, 0], [1, 1], [1, 1]], dtype=bool)
    close = np.where(listed, 10.0, np.nan)
    r, v = _stock_returns(co, oc, listed, close)
    assert not v[0].any()
    assert v[1, 0] and not v[1, 1] and not v[2, 1]       # late listing: first return on its second day
    assert v[3, 1]
    np.testing.assert_allclose(r[1, 0], 1.01 * 1.01 - 1)
    uni = np.array([[1, 0], [1, 0], [1, 1], [0, 1]], dtype=bool)
    m = _market(r, v, uni)
    assert np.isnan(m[0])
    np.testing.assert_allclose(m[2], r[2, 0])            # col 1 not valid yet on row 2
    np.testing.assert_allclose(m[3], r[3, 1])            # col 0 not a member on row 3


def test_factor_shock_matches_direct_ols():
    rng = np.random.default_rng(1)
    T = 300
    mkt = rng.normal(0, 0.01, T)
    dF = 0.1 + 0.7 * mkt + rng.normal(0, 0.05, T)
    a, b, shock, z = _factor_shock(dF, mkt)
    t = 280
    bb, aa = np.polyfit(mkt[t - 252:t], dF[t - 252:t], 1)
    res = dF[t - 252:t] - aa - bb * mkt[t - 252:t]
    np.testing.assert_allclose([a[t], b[t]], [aa, bb], rtol=1e-8)
    np.testing.assert_allclose(shock[t], res[-10:].sum(), rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(z[t], res[-10:].sum() / (res.std(ddof=1) * np.sqrt(10)), rtol=1e-7)
    assert np.isnan(z[199]) and np.isfinite(z[200])      # needs 200 valid pairs ending t-1
    # day t's change is not used at row t
    dF2 = dF.copy()
    dF2[t:] = 99.0
    a2, b2, s2, z2 = _factor_shock(dF2, mkt)
    np.testing.assert_allclose(z2[:t + 1], z[:t + 1], equal_nan=True)


def _exposure_setup():
    rng = np.random.default_rng(2)
    T, N = 330, 4
    mkt = rng.normal(0, 0.01, T)
    dF = np.vstack([0.3 * mkt + rng.normal(0, 0.05, T), rng.normal(0, 0.004, T), rng.normal(0, 0.02, T)])
    ab = [_factor_shock(dF[f], mkt) for f in range(3)]
    a = np.vstack([x[0] for x in ab])
    b = np.vstack([x[1] for x in ab])
    true_b = rng.normal(0, 1, (N, 5))
    r = np.zeros((T, N))
    m = 260
    E = dF - a[:, m:m + 1] - b[:, m:m + 1] * mkt[None, :]
    X = np.column_stack([np.ones(T), mkt, E.T])
    for j in range(N):
        r[:, j] = 0.01 * X @ true_b[j] + rng.normal(0, 0.01, T)
    valid = np.ones((T, N), dtype=bool)
    valid[0] = False
    valid[:200, 3] = False                                 # stock 3: only 60 valid returns in the window
    r = np.where(valid, r, np.nan)
    month = np.zeros(T, dtype=bool)
    month[[0, 100, 260, 290]] = True
    return r, valid, mkt, dF, a, b, month, X, m


def test_exposures_monthly_ols_and_ivol():
    r, valid, mkt, dF, a, b, month, X, m = _exposure_setup()
    betas, ivol = _exposures(r, valid, mkt, dF, a, b, month)
    s0 = m - REG_DAYS
    for j in range(3):
        coef = np.linalg.lstsq(X[s0:m], r[s0:m, j], rcond=None)[0]
        for i in range(4):
            np.testing.assert_allclose(betas[i][m, j], coef[1 + i], rtol=1e-4, atol=1e-5)
            assert betas[i][289, j] == betas[i][m, j]       # constant within the month
            assert betas[i][290, j] != betas[i][m, j]       # re-estimated next month
        t = 275
        res = r[t - 63:t, j] - X[t - 63:t, 1:] @ coef[1:]
        np.testing.assert_allclose(ivol[t, j], res.std(ddof=1), rtol=1e-4)
    assert np.all(np.isnan(betas[0][:m]))                   # first month start with 252 sessions is 260
    assert np.all(np.isnan(betas[0][m:, 3]))                # fewer than 200 valid returns: not scored
    assert np.all(np.isnan(ivol[m:, 3]))


def test_legs_and_book():
    T, N = 3, 12
    score = np.tile(np.arange(N, dtype=float), (T, 1))
    uni = np.ones((T, N), dtype=bool)
    uni[:, 11] = False
    beta = np.ones((T, N))
    beta[:, [9, 10]] = 1.5
    beta[:, [0, 1]] = 0.5
    gate = np.array([False, True, True])
    legs = _legs(score, uni, beta, gate, np.array([0, 1]), 0.2, min_names=10)
    assert legs[0][0].size == 0
    assert set(legs[1][0]) == {9, 10} and set(legs[1][1]) == {0, 1}
    W = _book(legs, beta, T, N)
    assert np.all(W[0] == 0) and np.all(np.isnan(W[2]))
    # L = 0.5 / 2.0 = 0.25 -> clipped to 0.35
    np.testing.assert_allclose(W[1, [9, 10]], 0.175)
    np.testing.assert_allclose(W[1, [0, 1]], -0.325)
    assert W[1, 11] == 0 and abs(np.abs(W[1]).sum() - 1.0) < 1e-12
    # unclipped beta balance
    beta2 = np.ones((T, N))
    beta2[:, [0, 1]] = 0.8
    W2 = _book(_legs(score, uni, beta2, gate, np.array([1]), 0.2, min_names=10), beta2, T, N)
    L, S = W2[1, W2[1] > 0].sum(), -W2[1, W2[1] < 0].sum()
    np.testing.assert_allclose(L * 1.0, S * 0.8)
    np.testing.assert_allclose(L + S, 1.0)
    # top changes the legs, the min-names floor makes the book flat
    l3 = _legs(score, uni, beta, gate, np.array([1]), 0.3, min_names=10)
    assert l3[1][0].size == 3
    l4 = _legs(score, uni, beta, gate, np.array([1]), 0.2, min_names=12)
    assert l4[1][0].size == 0
    # unscored names (NaN score) are not candidates
    sc = score.copy()
    sc[1, 10] = np.nan
    l5 = _legs(sc, uni, beta, gate, np.array([1]), 0.2, min_names=10)
    assert set(l5[1][0]) == {8, 9}


def test_diagnostic_events_entries_and_sign():
    T, N = 4, 6
    e = np.zeros(0, dtype=np.int64)
    legs = {0: (np.array([0, 1]), np.array([4, 5])), 2: (np.array([1, 2]), np.array([5, 3])), 3: (e, e)}
    imp = np.array([[1.0, 1.0, 1.0, -1.0, -1.0, 1.0]] * T)
    imp[2, 2] = -0.5                                        # new long entry with disagreeing sign: no event
    uni = np.ones((T, N), dtype=bool)
    trad = np.ones((T, N), dtype=bool)
    mask, side = _events_from_legs(legs, np.array([0, 2, 3]), imp, uni, trad, T, N)
    assert set(np.flatnonzero(mask[0])) == {0, 1, 4}          # col 5 short with positive implied: no event
    np.testing.assert_array_equal(side[0, [0, 1, 4]], [1, 1, -1])
    assert set(np.flatnonzero(mask[2])) == {3}                # 1 and 5 stayed in their legs, 2 disagrees
    assert side[2, 3] == -1
    assert not mask[1].any() and not mask[3].any()
    trad[0, 0] = False
    uni[0, 4] = False
    mask2, _ = _events_from_legs(legs, np.array([0, 2, 3]), imp, uni, trad, T, N)
    assert set(np.flatnonzero(mask2[0])) == {1}


def test_close_exits():
    T, N = 8, 3
    W = np.full((T, N), np.nan)
    W[0] = [0.5, -0.5, 0.0]
    W[5] = [0.0, -0.5, 0.5]
    week = np.zeros(T, dtype=bool)
    week[[0, 5]] = True
    uni = np.ones((T, N), dtype=bool)
    uni[2:4, 0] = False
    uni[6, 2] = False
    out = _close_exits(W, uni, week)
    assert out[2, 0] == 0 and out[3, 0] == 0
    assert np.isnan(out[2, 1]) and np.isnan(out[4, 0]) and np.isnan(out[1, 0])
    assert out[6, 2] == 0 and np.isnan(out[6, 1])
    np.testing.assert_allclose(out[5], W[5])


def test_gate():
    z = np.array([[0.5, np.nan, -1.6, np.nan], [1.2, 2.1, 0.0, np.nan], [np.nan, 0.1, 0.2, np.nan]])
    np.testing.assert_array_equal(_gate(z, 1.5), [False, True, True, False])
    np.testing.assert_array_equal(_gate(z, 1.0), [True, True, True, False])
    np.testing.assert_array_equal(_gate(z, 2.0), [False, True, False, False])


def _random_inputs(T=420, N=6, seed=3):
    rng = np.random.default_rng(seed)
    co = rng.normal(0, 0.005, (T, N))
    oc = rng.normal(0, 0.01, (T, N))
    listed = np.ones((T, N), dtype=bool)
    listed[:150, 5] = False                                 # late listing
    close = np.where(listed, 20.0, np.nan)
    r, valid = _stock_returns(co, oc, listed, close)
    uni = listed.copy()
    uni[300:, 2] = False
    dv = np.where(listed, rng.uniform(1e6, 1e7, (T, N)), np.nan)
    rate = np.cumsum(rng.normal(0, 0.05, T)) + 4.0
    rate[rng.random(T) < 0.03] = np.nan
    eur = np.exp(np.cumsum(rng.normal(0, 0.005, T)))
    cred = 1.0 + np.cumsum(rng.normal(0, 0.01, T))
    levels = (rate, -np.log(eur), cred)
    month = np.zeros(T, dtype=bool)
    month[::21] = True
    return r, valid, uni, dv, levels, month


def test_compute_is_point_in_time():
    r, valid, uni, dv, levels, month = _random_inputs()
    full = _compute(r, valid, uni, dv, levels, month)
    assert any(np.isfinite(f[-1]).any() for f in full["feats"])
    assert np.isfinite(full["z"][:, -1]).all()
    for cut in (300, 330, 377):
        lv = tuple(x[:cut + 1] for x in levels)
        tr = _compute(r[:cut + 1], valid[:cut + 1], uni[:cut + 1], dv[:cut + 1], lv, month[:cut + 1])
        for f_full, f_tr in zip(full["feats"], tr["feats"]):
            np.testing.assert_allclose(f_tr, f_full[:cut + 1], rtol=1e-5, atol=1e-7, equal_nan=True)
        np.testing.assert_allclose(tr["z"], full["z"][:, :cut + 1], equal_nan=True)
        # perturb the future: rows <= cut unchanged
        rng = np.random.default_rng(cut)
        r2 = r.copy()
        r2[cut + 1:] = rng.normal(0, 0.05, r2[cut + 1:].shape)
        lv2 = tuple(np.concatenate([x[:cut + 1], rng.normal(0, 1, len(x) - cut - 1)]) for x in levels)
        pt = _compute(r2, valid, uni, dv, lv2, month)
        for f_full, f_pt in zip(full["feats"], pt["feats"]):
            np.testing.assert_allclose(f_pt[:cut + 1], f_full[:cut + 1], rtol=1e-5, atol=1e-7, equal_nan=True)


def test_compute_feature_definitions():
    r, valid, uni, dv, levels, month = _random_inputs()
    c = _compute(r, valid, uni, dv, levels, month)
    t, j = 400, 1
    beta_mkt = c["beta_mkt"][t, j]
    stock10 = np.prod(1 + r[t - 9:t + 1, j]) - 1
    mkt10 = np.prod(1 + c["mkt"][t - 9:t + 1]) - 1
    np.testing.assert_allclose(c["feats"][3][t, j], stock10 - beta_mkt * mkt10, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(c["feats"][5][t, j], np.log(dv[t - 62:t + 1, j].mean()), rtol=1e-5)
    for f in range(3):
        np.testing.assert_allclose(c["feats"][f][t, j], c["betas"][1 + f][t, j] * c["shock"][f, t], rtol=1e-5)
        assert np.isfinite(c["feats"][f][t, j])
    np.testing.assert_allclose(c["implied_sum"][t, j], sum(c["feats"][f][t, j] for f in range(3)), rtol=1e-5)
    # late-listed stock has no exposure until it has 200 valid returns in a 252-session window
    assert np.all(np.isnan(c["beta_mkt"][:357, 5]))
    assert np.isfinite(c["beta_mkt"][-1, 5])
