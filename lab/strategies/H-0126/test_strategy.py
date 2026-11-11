import numpy as np
import pytest

from strategy import PARAMS, target_weights
from strategy import (book, coverage, credit_shock_z, last_change_rows, legs, market_return, rows_since_change,
                      ready_mask, shock_inputs, stock_features)


def test_params_match_card():
    assert PARAMS == {"n_estimators": 100, "z_news": 1.0}
    import strategy as S
    assert (S.HORIZON, S.REG_WIN, S.VOL_WIN, S.BETA_WIN, S.BETA_MIN) == (5, 252, 63, 252, 200)
    assert (S.MIN_NEWS_DATES, S.MIN_SCORED, S.MIN_LEG, S.RETRAIN_EVERY) == (40, 60, 3, 252)
    assert S.GBM_FIXED == {"num_leaves": 15, "learning_rate": 0.05, "min_child_samples": 200}


def test_rows_since_change_and_staleness():
    x = np.array([np.nan, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    out = rows_since_change(x)
    assert out[0] > 10 ** 6
    assert list(out[1:]) == [0, 1, 0, 1, 2, 3, 4, 5, 6]


def test_shock_inputs_alignment():
    T = 12
    baa = np.arange(T, dtype=float) * 2.0        # row u holds the day u-1 value
    y10 = np.arange(T, dtype=float) * 0.5
    rm = np.full(T, 0.01)
    rm[3] = 0.10
    dS, dY, M = shock_inputs(baa, y10, rm, k=5)
    assert np.all(np.isnan(dS[:5]))
    assert np.allclose(dS[5:], 5 * 1.5)
    assert np.allclose(dY[5:], 5 * 0.5)
    # M[u] compounds sessions u-5..u-1: row 3 enters at u = 4..8, not at u = 3 or u = 9
    assert np.isclose(M[8], 1.10 * 1.01 ** 4 - 1)
    assert np.isclose(M[9], 1.01 ** 5 - 1)
    assert np.isclose(M[5], 1.01 ** 4 * 1.10 - 1)


def _shock_series(T, seed=0, jump=0.0, jump_row=None):
    rng = np.random.default_rng(seed)
    rm = rng.normal(0, 0.01, T)
    y10 = 4 + np.cumsum(rng.normal(0, 0.05, T))
    spread = 2 + np.cumsum(rng.normal(0, 0.02, T))
    if jump_row is not None:
        spread[jump_row:] += jump
    return spread + y10, y10, rm


def test_credit_shock_z_warmup_sign_and_point_in_time():
    T = 400
    baa, y10, rm = _shock_series(T, jump=1.0, jump_row=350)
    rows = np.ones(T, dtype=bool)
    z = credit_shock_z(baa, y10, rm, rows, win=252, k=5)
    assert np.all(np.isnan(z[:251 + 5]))           # dS needs 5 rows, regression 252 windows
    assert np.isfinite(z[300])
    assert z[352] > 3.0                             # credit-specific widening -> large positive z
    z_cut = credit_shock_z(baa[:353], y10[:353], rm[:353], rows[:353], win=252, k=5)
    assert np.allclose(z_cut, z[:353], equal_nan=True)


def test_credit_shock_stale_series_no_shock():
    T = 320
    baa, y10, rm = _shock_series(T, seed=1)
    baa = baa.copy()
    baa[300:] = baa[299]                            # last new Baa value on row 299
    z = credit_shock_z(baa, y10, rm, np.ones(T, dtype=bool), win=252, k=5)
    assert np.isfinite(z[304])                      # 5 sessions without a new value is still fine
    assert np.isnan(z[305])                         # more than 5: no shock


def test_credit_shock_only_on_requested_rows():
    T = 320
    baa, y10, rm = _shock_series(T, seed=2)
    rows = np.zeros(T, dtype=bool)
    rows[300] = True
    z = credit_shock_z(baa, y10, rm, rows)
    assert np.isfinite(z[300]) and np.isnan(np.delete(z, 300)).all()


def test_coverage_rules():
    ebit = np.array([100.0, 100.0, -1000.0, 50.0, np.nan, np.nan, 100.0])
    ie = np.array([10.0, 1.0, 10.0, np.nan, 5.0, np.nan, 0.0])
    c = coverage(ebit, ie)
    assert c[0] == 10.0 and c[1] == 30.0 and c[2] == -5.0
    assert c[3] == 30.0 and np.isnan(c[4]) and c[5] == 30.0 and c[6] == 30.0


def test_market_return_members_only():
    r = np.array([[0.01, 0.03, 0.5], [np.nan, 0.02, 0.04]], dtype=np.float32)
    uni = np.array([[True, True, False], [True, True, True]])
    rm = market_return(r, uni)
    assert np.isclose(rm[0], 0.02) and np.isclose(rm[1], 0.03)


def test_last_change_rows():
    x = np.array([[np.nan, 1.0], [5.0, 1.0], [5.0, 2.0], [np.nan, 2.0], [7.0, 2.0]], dtype=np.float32)
    a = last_change_rows(x)
    assert list(a[:, 0]) == [-1, 1, 1, 1, 4]        # a NaN row keeps the anchor; callers require a finite value
    assert list(a[:, 1]) == [0, 0, 2, 2, 2]


def _feature_inputs(T=300, N=4, seed=3):
    rng = np.random.default_rng(seed)
    rm = rng.normal(0, 0.01, T)
    betas = np.array([0.5, 1.0, 1.5, 2.0])[:N]
    r = (rm[:, None] * betas[None, :] + rng.normal(0, 0.005, (T, N))).astype(np.float32)
    g = (1.0 + r).astype(np.float32)
    mc = np.full((T, N), np.nan, dtype=np.float32)
    mc[200:, :] = 1000.0
    debt = np.full((T, N), 500.0, dtype=np.float32)
    debt[:, 1] = np.nan                                  # missing debt counts as 0
    ebit = np.full((T, N), 100.0, dtype=np.float32)
    ie = np.full((T, N), 10.0, dtype=np.float32)
    return r, rm, g, mc, debt, ebit, ie, betas


def test_stock_features_hand_values():
    T = 300
    r, rm, g, mc, debt, ebit, ie, betas = _feature_inputs(T)
    anchor = last_change_rows(mc)
    t = 280
    F = stock_features(t, r, rm, g, mc, anchor, debt, ebit, ie, -1.5, np.ones(4, dtype=bool))
    lev, cov, vol, slev, resp, beta = F
    E = 1000.0 * np.prod(g[200:t + 1].astype(np.float64), axis=0)
    assert np.allclose(lev[[0, 2, 3]], (500.0 / (500.0 + E))[[0, 2, 3]], rtol=1e-6)
    assert lev[1] == 0.0
    assert np.allclose(cov, 10.0)
    assert np.allclose(vol, r[t - 62:t + 1].astype(np.float64).std(axis=0), rtol=1e-6)
    assert np.allclose(slev, -1.5 * lev)
    x = rm[t - 251:t + 1]
    for j in range(4):
        assert np.isclose(beta[j], np.polyfit(x, r[t - 251:t + 1, j].astype(np.float64), 1)[0], rtol=1e-6)
    assert np.allclose(beta, betas, atol=0.2)
    own = np.prod(1.0 + r[t - 5:t + 1].astype(np.float64), axis=0) - 1
    mk = np.prod(1.0 + rm[t - 5:t + 1]) - 1
    assert np.allclose(resp, own - beta * mk)


def test_stock_features_missing_inputs_not_scored():
    T = 300
    r, rm, g, mc, debt, ebit, ie, _ = _feature_inputs(T)
    anchor = last_change_rows(mc)
    r = r.copy()
    r[100:120, 0] = np.nan                               # 20 missing days: only 232 valid beta days, still ok
    r[50:120, 1] = np.nan                                # 70 missing: 182 < 200 -> no beta
    mc = mc.copy()
    mc[:, 2] = np.nan                                    # no marketcap -> not scored
    anchor = last_change_rows(mc)
    members = np.array([True, True, True, False])
    F = stock_features(280, r, rm, g, mc, anchor, debt, ebit, ie, 1.0, members)
    assert np.isfinite(F[:, 0]).all()
    assert np.isnan(F[:, 1]).all() and np.isnan(F[:, 2]).all() and np.isnan(F[:, 3]).all()
    # before 200 valid days nobody has a beta
    F0 = stock_features(150, r, rm, g, mc, anchor, debt, ebit, ie, 1.0, np.ones(4, dtype=bool))
    assert np.isnan(F0).all()


def test_legs_terciles_and_minimums():
    n = 90
    beta = np.arange(n, dtype=float)                     # terciles: cols 0-29, 30-59, 60-89
    score = np.tile(np.arange(30, dtype=float), 3)       # within each tercile score rises with column
    member = np.ones(n, dtype=bool)
    lo, sh = legs(score, beta, member)
    assert sorted(lo) == [27, 28, 29, 57, 58, 59, 87, 88, 89]
    assert sorted(sh) == [0, 1, 2, 30, 31, 32, 60, 61, 62]
    # 75 scored: terciles of 25 -> 2 per side < 3 -> all skipped
    m2 = member.copy()
    m2[75:] = False
    lo2, sh2 = legs(score, beta, m2)
    assert len(lo2) == 0 and len(sh2) == 0
    # fewer than 60 scored -> flat
    m3 = member.copy()
    m3[59:] = False
    assert len(legs(score, beta, m3)[0]) == 0
    # non-members and NaN scores are never in a leg
    s4 = score.copy()
    s4[29] = np.nan
    m4 = member.copy()
    m4[89] = False
    lo4, _ = legs(s4, beta, m4)
    assert 29 not in lo4 and 89 not in lo4


def test_book_holds_closes_and_flat_on_quiet_weeks():
    T, n = 12, 90
    beta = np.tile(np.arange(n, dtype=float), (T, 1))
    score = np.tile(np.tile(np.arange(30, dtype=float), 3), (T, 1))
    uni = np.ones((T, n), dtype=bool)
    uni[3:, 89] = False                                  # a long leaves the universe during the hold
    dec = np.zeros(T, dtype=bool)
    dec[[1, 6, 11]] = True
    news = np.zeros(T, dtype=bool)
    news[[1, 11]] = True
    W = book(score, beta, dec, news, uni)
    assert np.isnan(W[0]).all()
    assert np.isclose(W[1][W[1] > 0].sum(), 0.5) and np.isclose(W[1][W[1] < 0].sum(), -0.5)
    assert np.isclose(np.abs(W[1]).sum(), 1.0)
    assert np.isnan(W[2]).all()
    assert W[3, 89] == 0.0 and np.isnan(np.delete(W[3], 89)).all()
    assert np.isnan(W[4]).all()
    assert (W[6] == 0).all()                             # quiet decision date: flat
    assert np.isnan(W[7:11]).all()
    assert 89 not in np.flatnonzero(W[11])               # not a member any more
    assert np.isclose(np.abs(W[11]).sum(), 1.0)


def test_ready_mask_counts_realized_news_dates():
    has = np.zeros(30, dtype=bool)
    has[[2, 4, 10]] = True
    rdy = ready_mask(has, 5, 2)
    # second news date (row 4) has its label realized at row 4 + 5 + 1 = 10
    assert not rdy[:10].any() and rdy[10:].all()
    assert not ready_mask(has, 5, 4).any()


def test_z_news_changes_news_set():
    T = 700
    baa, y10, rm = _shock_series(T, seed=5)
    z = credit_shock_z(baa, y10, rm, np.ones(T, dtype=bool))
    fin = np.isfinite(z)
    n075 = (np.abs(z[fin]) >= 0.75).sum()
    n15 = (np.abs(z[fin]) >= 1.5).sum()
    assert n075 > n15 > 0
