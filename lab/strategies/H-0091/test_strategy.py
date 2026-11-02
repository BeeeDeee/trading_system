import numpy as np
import pytest

from strategy import target_weights, PARAMS
from strategy import daily_returns, market_adjusted, prior_sd, overhang, event_sides, book


class Stub:
    def __init__(self, ret_co, ret_oc, close, dv, listed, tradable, uni, ea):
        self.ret_co = ret_co
        self.ret_oc = ret_oc
        self.close = close
        self.dollar_volume = dv
        self.listed = listed
        self.tradable = tradable
        self.universe = uni
        self.extras = {"ea_reaction_day": ea}
        T, N = ret_co.shape
        self.instruments = tuple("E%d" % j for j in range(N))


def make(T=400, N=30, seed=0, n_events=12, event_row=300):
    rng = np.random.default_rng(seed)
    ret_co = rng.normal(0, 0.004, (T, N))
    ret_oc = rng.normal(0.0003, 0.012, (T, N)) + rng.normal(0, 0.01, (1, N)) * 0.05
    close = 50 * np.cumprod((1 + ret_co) * (1 + ret_oc), axis=0)
    dv = rng.uniform(1e6, 5e6, (T, N))
    listed = np.ones((T, N), bool)
    tradable = np.ones((T, N), bool)
    uni = np.ones((T, N), bool)
    ea = np.zeros((T, N))
    ea[:50] = np.nan
    js = np.arange(min(n_events, N))
    event_row = min(event_row, T - 1)
    ret_oc[event_row, js] = np.where(js % 2 == 0, 0.08, -0.08)
    ea[event_row, js] = 1.0
    return Stub(ret_co, ret_oc, close, dv, listed, tradable, uni, ea)


# ---------- brute-force reference (plain loops, written from the card) ----------

def ref_sides(d, ref_window, z_min):
    T, N = d.ret_co.shape
    r = (1 + d.ret_co) * (1 + d.ret_oc) - 1
    valid = np.zeros((T, N), bool)
    for t in range(1, T):
        for j in range(N):
            valid[t, j] = d.listed[t, j] and np.isfinite(d.close[t, j]) and np.isfinite(d.close[t - 1, j])
    a = np.full((T, N), np.nan)
    for t in range(T):
        mem = [j for j in range(N) if valid[t, j] and d.universe[t, j]]
        if not mem:
            continue
        m = np.mean([r[t, j] for j in mem])
        for j in range(N):
            if valid[t, j]:
                a[t, j] = r[t, j] - m
    P = np.ones((T, N))
    for t in range(T):
        for j in range(N):
            g = 1 + (r[t, j] if valid[t, j] else 0.0)
            P[t, j] = (P[t - 1, j] if t else 1.0) * g
    cgo = np.full((T, N), np.nan)
    for s in range(T):
        for j in range(N):
            ks = [k for k in range(s - ref_window + 1, s + 1) if k >= 0 and np.isfinite(d.dollar_volume[k, j])
                  and d.dollar_volume[k, j] > 0 and np.isfinite(d.close[k, j]) and d.listed[k, j]]
            if len(ks) >= 0.8 * ref_window and np.isfinite(d.close[s, j]):
                R = sum(d.dollar_volume[k, j] * P[k, j] for k in ks) / sum(d.dollar_volume[k, j] for k in ks)
                cgo[s, j] = P[s, j] / R - 1
    side = np.zeros((T, N), np.int8)
    for t in range(1, T):
        pool = [cgo[t - 1, j] for j in range(N) if d.universe[t - 1, j] and np.isfinite(cgo[t - 1, j])]
        if not pool:
            continue
        q1, q2 = np.quantile(pool, [1 / 3, 2 / 3])
        for j in range(N):
            if not (d.extras["ea_reaction_day"][t, j] == 1 and d.universe[t, j] and d.tradable[t, j]):
                continue
            w = [a[k, j] for k in range(max(t - 63, 0), t) if np.isfinite(a[k, j])]
            if len(w) < 40 or not np.isfinite(a[t, j]):
                continue
            z = a[t, j] / np.std(w, ddof=1)
            if abs(z) < z_min or not np.isfinite(cgo[t - 1, j]):
                continue
            if cgo[t - 1, j] >= q2:
                side[t, j] = 1
            elif cgo[t - 1, j] <= q1:
                side[t, j] = -1
    return side


# ---------- tests ----------

def test_params_match_card():
    assert PARAMS == {"ref_window": 252, "z_min": 1.0}


def test_market_adjusted_by_hand():
    r = np.array([[0.01, 0.03, 0.10, np.nan]])
    valid = np.isfinite(r)
    uni = np.array([[True, True, False, True]])
    a = market_adjusted(r, valid, uni)
    np.testing.assert_allclose(a[0, :3], [-0.01, 0.01, 0.08])
    assert np.isnan(a[0, 3])


def test_daily_returns_compound_and_first_day():
    d = make(T=5, N=2)
    d.close[0, 1] = np.nan
    r, valid = daily_returns(d)
    assert not valid[0].any()  # no previous close on row 0
    assert not valid[1, 1]     # no close on t-1
    np.testing.assert_allclose(r[2, 0], (1 + d.ret_co[2, 0]) * (1 + d.ret_oc[2, 0]) - 1)


def test_prior_sd_window_and_min_valid():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(120, 2))
    a[:, 1] = np.nan
    a[70:110, 1] = rng.normal(size=40)
    sd = prior_sd(a)
    for t in (40, 63, 64, 100, 119):
        np.testing.assert_allclose(sd[t, 0], np.std(a[max(t - 63, 0):t, 0], ddof=1))
    assert np.isnan(sd[39, 0])      # only 39 prior values
    assert np.isnan(sd[109, 1])     # 39 valid values in the window
    np.testing.assert_allclose(sd[110, 1], np.std(a[70:110, 1], ddof=1))


def test_overhang_by_hand():
    # one stock, W = 4: P = 1, 1.1, 1.21, 1.21 (r = 0, .1, .1, 0); dv = 1, 1, 2, 0 (last day no volume)
    T = 5
    ret_co = np.zeros((T, 1))
    ret_oc = np.array([[0.0], [0.1], [0.1], [0.0], [0.0]])
    close = np.ones((T, 1))
    dv = np.array([[1.0], [1.0], [2.0], [0.0], [1.0]])
    d = Stub(ret_co, ret_oc, close, dv, np.ones((T, 1), bool), np.ones((T, 1), bool), np.ones((T, 1), bool),
             np.zeros((T, 1)))
    r, valid = daily_returns(d)
    cgo = overhang(r, valid, d, 4)
    # s = 3: sessions 0..3, valid volume 3 of 4 (75 % < 80 %) -> NaN
    assert np.isnan(cgo[3, 0])
    # s = 2: sessions -1..2 -> only 3 valid of 4 -> NaN
    assert np.isnan(cgo[2, 0])
    # W = 3, s = 2: sessions 0..2 all valid: R = (1*1 + 1*1.1 + 2*1.21) / 4
    cgo3 = overhang(r, valid, d, 3)
    np.testing.assert_allclose(cgo3[2, 0], 1.21 / ((1 + 1.1 + 2 * 1.21) / 4) - 1)
    # W = 5, s = 4: sessions 0..4, 4 of 5 valid (80 %) -> value
    cgo5 = overhang(r, valid, d, 5)
    np.testing.assert_allclose(cgo5[4, 0], 1.21 / ((1 + 1.1 + 2 * 1.21 + 1.21) / 5) - 1)


def test_book_by_hand():
    T, N = 10, 8
    side = np.zeros((T, N), np.int8)
    side[2, :3] = 1
    side[2, 3:6] = -1
    uni = np.ones((T, N), bool)
    tr = np.ones((T, N), bool)
    w = book(side, uni, tr)
    assert np.all(w[:2] == 0)
    for t in range(2, 7):  # active on closes 2..6 (held open 3 .. open 8)
        np.testing.assert_allclose(w[t, :3], 0.5 / 3)
        np.testing.assert_allclose(w[t, 3:6], -0.5 / 3)
    assert np.all(w[7:] == 0)
    # latest event wins: stock 0 flips to short on row 4 -> 2 longs, 4 shorts -> flat (fewer than 3 longs)
    side2 = side.copy()
    side2[4, 0] = -1
    w2 = book(side2, uni, tr)
    assert np.all(w2[4:7] == 0)
    np.testing.assert_allclose(w2[2:4, 0], 0.5 / 3)
    # add a long on row 4: 3 longs (1, 2, 6), 4 shorts (0, 3, 4, 5)
    side2[4, 6] = 1
    w3 = book(side2, uni, tr)
    np.testing.assert_allclose(w3[4, [1, 2, 6]], 0.5 / 3)
    np.testing.assert_allclose(w3[4, [0, 3, 4, 5]], -0.5 / 4)
    # stock 0's short of row 4 lasts through row 8, stocks 1, 2 expire after row 6
    assert np.all(w3[7:9] == 0)  # fewer than 3 per side
    # leaving the universe closes the position at the next open and can flatten the book
    uni2 = uni.copy()
    uni2[3, 1] = False
    w4 = book(side, uni2, tr)
    assert np.all(w4[3] == 0)
    np.testing.assert_allclose(w4[4, :3], 0.5 / 3)
    tr2 = tr.copy()
    tr2[5, 4] = False
    w5 = book(side, uni, tr2)
    assert np.all(w5[5] == 0)


def test_end_to_end_matches_reference():
    d = make()
    for prm in ({"ref_window": 126, "z_min": 1.0}, {"ref_window": 252, "z_min": 1.5}):
        got = event_sides(d, prm)
        exp = ref_sides(d, prm["ref_window"], prm["z_min"])
        np.testing.assert_array_equal(got, exp)
        assert (got != 0).sum() >= 6


def test_weights_long_short_neutral_and_gross():
    d = make()
    w = target_weights(d, PARAMS)
    assert w.shape == d.ret_co.shape
    assert np.all(np.isfinite(w))
    assert np.all(np.abs(w).sum(axis=1) <= 1 + 1e-12)
    on = np.abs(w).sum(axis=1) > 0
    assert on.any()
    np.testing.assert_allclose(w[on].sum(axis=1), 0, atol=1e-12)
    np.testing.assert_allclose(np.abs(w[on]).sum(axis=1), 1.0)
    # the book is on exactly at closes event_row .. event_row + 4
    assert set(np.flatnonzero(on)) == set(range(300, 305))


def test_side_ignores_news_sign():
    d = make()
    side = event_sides(d, PARAMS)
    j = np.flatnonzero(side[300])
    signs = np.sign(d.ret_oc[300, j])
    assert set(side[300, j][signs > 0]) | set(side[300, j][signs < 0]) <= {1, -1}
    # flipping the news sign does not change the side
    d2 = make()
    d2.ret_oc[300, :12] *= -1
    np.testing.assert_array_equal(event_sides(d2, PARAMS)[300], side[300])


def test_parameters_change_output():
    d = make(seed=5)
    base = event_sides(d, PARAMS)
    hi_z = event_sides(d, {"ref_window": 252, "z_min": 50.0})
    assert (base != 0).sum() > 0 and (hi_z != 0).sum() == 0
    win = [event_sides(d, {"ref_window": w, "z_min": 1.0}) for w in (126, 252, 504)]
    # 504 > 300 rows of history at the event: no overhang value, no events
    assert (win[2] != 0).sum() == 0
    assert not np.array_equal(win[0], win[1]) or (win[0] != 0).sum() > 0


def test_missing_flag_late_listing_and_nonmember():
    d = make()
    d.extras["ea_reaction_day"][300, 0] = np.nan          # missing flag: no event
    d.universe[300, 1] = False                             # not a member on t: no event
    d.tradable[300, 2] = False                             # not tradable on t: no event
    late = 3                                               # listed only from row 280: < 40 sd values
    d.listed[:280, late] = False
    d.close[:280, late] = np.nan
    d.dollar_volume[:280, late] = np.nan
    side = event_sides(d, {"ref_window": 126, "z_min": 1.0})
    assert np.all(side[300, :4] == 0)
    assert (side[300] != 0).sum() > 0


def test_point_in_time_truncation_and_perturbation():
    d = make()
    w = target_weights(d, PARAMS)
    rng = np.random.default_rng(1)
    for cut in (120, 299, 300, 302):
        dt = Stub(d.ret_co[:cut + 1], d.ret_oc[:cut + 1], d.close[:cut + 1], d.dollar_volume[:cut + 1],
                  d.listed[:cut + 1], d.tradable[:cut + 1], d.universe[:cut + 1],
                  d.extras["ea_reaction_day"][:cut + 1])
        np.testing.assert_array_equal(target_weights(dt, PARAMS), w[:cut + 1])
        dp = make()
        dp.ret_oc[cut + 1:] = rng.normal(0, 0.05, dp.ret_oc[cut + 1:].shape)
        dp.dollar_volume[cut + 1:] = rng.uniform(1, 9e9, dp.dollar_volume[cut + 1:].shape)
        dp.extras["ea_reaction_day"][cut + 1:] = 1.0
        np.testing.assert_array_equal(target_weights(dp, PARAMS)[:cut + 1], w[:cut + 1])


def test_requires_universe_and_flag():
    d = make(T=80, N=5)
    d.universe = None
    with pytest.raises(ValueError):
        target_weights(d, PARAMS)
    d = make(T=80, N=5)
    d.extras = {}
    with pytest.raises(KeyError):
        target_weights(d, PARAMS)


def test_deterministic():
    d = make()
    np.testing.assert_array_equal(target_weights(d, PARAMS), target_weights(d, PARAMS))
