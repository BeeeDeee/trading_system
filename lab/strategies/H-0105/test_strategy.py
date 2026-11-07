import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import PARAMS, target_weights
import strategy as S


def make_view(T=400, n_groups=2, per_group=11, seed=3, late=None):
    """Two groups of stocks driven by a strong group factor, so the 10 peers of a stock are its own group."""
    rng = np.random.default_rng(seed)
    N = n_groups * per_group
    fco = rng.normal(0, 0.01, (T, n_groups))
    foc = rng.normal(0, 0.01, (T, n_groups))
    grp = np.repeat(np.arange(n_groups), per_group)
    co = fco[:, grp] + rng.normal(0, 0.002, (T, N))
    oc = foc[:, grp] + rng.normal(0, 0.002, (T, N))
    listed = np.ones((T, N), dtype=bool)
    if late is not None:
        col, row = late
        listed[:row, col] = False
        co[:row, col] = 0.0
        oc[:row, col] = 0.0
    close = np.where(listed, 50.0 * np.cumprod((1 + co) * (1 + oc), axis=0), np.nan)
    dv = np.where(listed, rng.uniform(1e7, 2e7, (T, N)), np.nan)
    uni = listed.copy()
    ea = np.zeros((T, N))
    ea[::40, :] = 1.0              # releases every 40 rows on all stocks
    k8 = np.zeros((T, N))
    k8[5, :] = 1.0
    sp = np.zeros((T, N))
    sp[:, :5] = 1.0
    liq = np.tile(np.arange(1, N + 1, dtype=float), (T, 1))
    dates = (np.arange(T) + 12000).astype("datetime64[D]")
    return DataView(
        dates=dates, instruments=tuple("E%d" % (k + 1) for k in range(N)),
        asset_class=tuple("us_equity" for _ in range(N)), ret_co=co, ret_oc=oc,
        tradable=listed.copy(), listed=listed, delisting=np.zeros((T, N), dtype=bool), close=close,
        dollar_volume=dv, universe=uni,
        extras={"ea_reaction_day": ea, "nonearn_8k_reaction_day": k8, "sp500_member": sp, "liq_rank": liq},
        series={}, cash_ret=None)


# ------------------------------------------------------------------------------------- peers and residuals
def test_peers_are_own_group():
    d = make_view()
    co, oc, cc, vio, vcc = S._returns(d)
    mem = np.arange(22)
    js, P = S._peers(cc, vcc, mem, 300)
    assert list(js) == list(range(22))
    for row, j in zip(P, js):
        grp = j // 11
        assert set(row) == set(range(grp * 11, grp * 11 + 11)) - {j}


def test_peers_need_200_common_sessions():
    d = make_view(T=400)
    co, oc, cc, vio, vcc = S._returns(d)
    js, P = S._peers(cc, vcc, np.arange(22), 150)       # only 151 rows of history
    assert js.size == 0


def test_zg_by_hand():
    d = make_view(T=400)
    co, oc, cc, vio, vcc = S._returns(d)
    zg, zi, prior = S._peer_features(d, co, oc, cc, vio, vcc)
    starts = np.flatnonzero(S.period_starts(d.dates, "M"))
    f = int(starts[starts >= 300][0])
    t = f + 3
    j = 0
    peers = list(range(1, 11))
    g = co[:, j] - co[:, peers].mean(axis=1)
    i = oc[:, j] - oc[:, peers].mean(axis=1)
    c = cc[:, j] - cc[:, peers].mean(axis=1)
    assert zg[t, j] == pytest.approx(g[t] / np.std(g[t - 63:t], ddof=1), rel=1e-9)
    assert zi[t, j] == pytest.approx(i[t] / np.std(i[t - 63:t], ddof=1), rel=1e-9)
    assert prior[t, j] == pytest.approx(c[t - 21:t].sum(), rel=1e-9)


def test_no_peer_features_before_first_month_with_history():
    d = make_view(T=400)
    co, oc, cc, vio, vcc = S._returns(d)
    zg, _, _ = S._peer_features(d, co, oc, cc, vio, vcc)
    assert np.isnan(zg[:200]).all()


def test_late_listing_has_no_peers_that_month():
    d = make_view(T=400, late=(3, 280))
    co, oc, cc, vio, vcc = S._returns(d)
    zg, _, _ = S._peer_features(d, co, oc, cc, vio, vcc)
    assert np.isnan(zg[:, 3]).all()                       # < 200 valid sessions by row 399
    assert np.isfinite(zg[350:, 0]).all()


# ----------------------------------------------------------------------------------------- other features
def test_abnormal_dollar_volume_by_hand():
    d = make_view(T=120)
    uni = np.asarray(d.universe)
    adv = S._abnormal_dollar_volume(d, uni)
    dv = d.dollar_volume
    t, j = 100, 4
    assert adv[t, j] == pytest.approx(np.log(dv[t, j] / np.median(dv[t - 63:t, j])), rel=1e-12)
    assert np.isnan(adv[49, j])                            # fewer than 50 prior values
    assert np.isfinite(adv[50, j])


def test_realized_vol_window_includes_t():
    d = make_view(T=120)
    co, oc, cc, vio, vcc = S._returns(d)
    rv = S._realized_vol(cc, vcc)
    assert rv[100, 2] == pytest.approx(np.std(cc[38:101, 2], ddof=1), rel=1e-6)


def test_since_release_capped_and_unknown_at_start():
    ea = np.zeros((200, 2))
    ea[:10, 1] = np.nan
    ea[100, 0] = 1.0
    ea[0, 0] = 1.0
    s = S._since_release(ea, 0)
    assert s[0, 0] == 0 and s[30, 0] == 30 and s[99, 0] == 63 and s[105, 0] == 5
    assert np.isnan(s[5, 1]) and np.isnan(s[72, 1]) and s[73, 1] == 63


def test_flags_window_and_dataset_start():
    ea = np.zeros((30, 3))
    ea[10, 0] = 1.0            # dataset starts at row 10
    ea[12, 1] = np.nan
    ok, start = S._flags_ok(ea, 5)
    assert start == 10
    assert not ok[14, 2] and ok[15, 2]                     # window t-5..t must start at or after row 10
    assert not ok[17, 1] and ok[18, 1]                     # a NaN flag inside t-5..t blocks the row
    assert not ok[15, 0] and ok[16, 0]                     # a release inside t-5..t blocks the row


def test_nonearn_window_is_t_minus_1_to_t():
    k8 = np.zeros((10, 1))
    k8[2, 0] = 1.0
    k8[6, 0] = 1.0
    ok, _ = S._flags_ok(k8, 1)
    assert [bool(v) for v in ok[:, 0]] == [False, False, False, False, True, True, False, False, True, True]


# ------------------------------------------------------------------------------------------------- sides
def test_sides_against_trailing_median():
    T, N = 300, 3
    score = np.full((T, N), np.nan)
    cand = np.zeros((T, N), dtype=bool)
    score[0, 0], score[0, 1] = 1.0, 2.0
    score[1, 0] = 5.0
    score[260, 0], score[260, 1], score[260, 2] = 0.1, 0.2, 0.3
    cand[[0, 0, 1, 260, 260, 260], [0, 1, 0, 0, 1, 2]] = True
    score[2, 2] = 9.0                                      # scored but not a candidate: ignored
    side, sc = S.sides(score, cand)
    assert side[0, 0] == -1 and side[0, 1] == 1           # median 1.5
    assert side[1, 0] == 1                                 # median of 1, 2, 5 = 2
    assert side[260, 0] == -1 and side[260, 1] == -1 and side[260, 2] == 1   # rows 0, 1 left the window
    assert np.isnan(side[2, 2]) and not sc[2, 2]


# -------------------------------------------------------------------------------------------- slot book
def test_slot_held_five_sessions_and_hedged():
    T, N = 12, 6
    uni = np.ones((T, N), dtype=bool)
    liq = np.tile(np.arange(1, N + 1, dtype=float), (T, 1))
    side = np.full((T, N), np.nan)
    sc = np.zeros((T, N), dtype=bool)
    side[2, 5], sc[2, 5] = 1.0, True
    w = S.slot_book(side, sc, uni, liq)
    assert np.allclose(w[2:7, 5], 0.03) and w[7, 5] == 0 and w[1, 5] == 0
    assert np.allclose(w[2:7, :5], -0.03 / 5)              # fewer than 100 non-slot members: all of them
    assert np.all(w[7:] == 0) and np.all(w[:2] == 0)


def test_hedge_uses_100_most_liquid_without_slot():
    T, N = 3, 150
    uni = np.ones((T, N), dtype=bool)
    liq = np.tile(np.arange(1, N + 1, dtype=float), (T, 1))
    side = np.full((T, N), np.nan)
    sc = np.zeros((T, N), dtype=bool)
    side[0, 0], sc[0, 0] = -1.0, True                     # the most liquid stock holds a slot
    w = S.slot_book(side, sc, uni, liq)
    assert w[0, 0] == pytest.approx(-0.03)
    assert np.allclose(w[0, 1:101], 0.03 / 100) and np.all(w[0, 101:] == 0)


def test_slot_closed_when_stock_leaves_universe():
    T, N = 12, 4
    uni = np.ones((T, N), dtype=bool)
    uni[5, 0] = False
    liq = np.tile(np.arange(1, N + 1, dtype=float), (T, 1))
    side = np.full((T, N), np.nan)
    sc = np.zeros((T, N), dtype=bool)
    side[2, 0], sc[2, 0] = 1.0, True
    w = S.slot_book(side, sc, uni, liq)
    assert np.allclose(w[2:5, 0], 0.03) and np.all(w[5:, 0] == 0)
    assert np.all(w[5:] == 0)


def test_opposite_slots_net_flat_no_hedge_and_gross_cap():
    T, N = 8, 60
    uni = np.ones((T, N), dtype=bool)
    liq = np.tile(np.arange(1, N + 1, dtype=float), (T, 1))
    side = np.full((T, N), np.nan)
    sc = np.zeros((T, N), dtype=bool)
    side[0, 0], sc[0, 0] = 1.0, True
    side[0, 1], sc[0, 1] = -1.0, True
    w = S.slot_book(side, sc, uni, liq)
    assert np.all(w[0, 2:] == 0) and w[0, 0] == pytest.approx(0.03) and w[0, 1] == pytest.approx(-0.03)
    side2 = np.full((T, N), np.nan)
    sc2 = np.zeros((T, N), dtype=bool)
    side2[0, :40], sc2[0, :40] = 1.0, True                # slot gross 1.2 plus hedge 1.2
    w2 = S.slot_book(side2, sc2, uni, liq)
    assert np.abs(w2[0]).sum() == pytest.approx(1.0)
    assert w2[0, 0] == pytest.approx(0.03 / 2.4)
    assert w2[0].sum() == pytest.approx(0.0, abs=1e-12)


# ------------------------------------------------------------------------------- candidates, end to end
def test_candidates_respect_flags_and_gap_z():
    d = make_view(T=400)
    cand, X, y = S.build_inputs(d, PARAMS)
    co, oc, cc, vio, vcc = S._returns(d)
    zg, _, _ = S._peer_features(d, co, oc, cc, vio, vcc)
    ea = d.extras["ea_reaction_day"]
    rows, cols = np.nonzero(cand)
    for t, j in zip(rows, cols):
        assert abs(zg[t, j]) >= PARAMS["gap_z"]
        assert np.all(ea[t - 5:t + 1, j] == 0)
    assert np.isnan(X[~cand]).all() and np.isfinite(X[cand]).all()
    lo, _, _ = S.build_inputs(d, dict(PARAMS, gap_z=1.0))
    assert lo.sum() > cand.sum() and np.all(lo[cand])
    # rows of an earnings window are never candidates
    assert not cand[40:46].any() and not cand[80:86].any()


def test_build_inputs_point_in_time():
    d = make_view(T=400)
    cand, X, _ = S.build_inputs(d, dict(PARAMS, gap_z=1.0))
    cut = 333
    dt = make_view(T=400)
    for arr in (dt.ret_co, dt.ret_oc, dt.close, dt.dollar_volume, dt.extras["ea_reaction_day"]):
        arr[cut + 1:] = arr[cut + 1:] * 3.0 + 0.01         # future rows replaced by something else
    c2, X2, _ = S.build_inputs(dt, dict(PARAMS, gap_z=1.0))
    assert cand[:cut + 1].sum() > 0
    assert np.array_equal(cand[:cut + 1], c2[:cut + 1])
    assert np.array_equal(np.isnan(X[:cut + 1]), np.isnan(X2[:cut + 1]))
    assert np.array_equal(X[:cut + 1][np.isfinite(X[:cut + 1])], X2[:cut + 1][np.isfinite(X2[:cut + 1])])


def test_target_weights_shape_and_flat_before_model():
    d = make_view(T=300)
    w = target_weights(d, PARAMS)
    assert w.shape == (300, 22)
    assert np.all(np.abs(w).sum(axis=1) <= 1.0 + 1e-12)
    assert np.all(w == 0)                                  # far fewer candidates than a training sample


def test_params_read():
    assert PARAMS == {"gap_z": 2.5, "n_estimators": 100}
    assert S.gbm_params({"gap_z": 2.5, "n_estimators": 50})["n_estimators"] == 50
    assert S.gbm_params(PARAMS)["num_leaves"] == 15


def test_missing_input_raises():
    d = make_view(T=100)
    del d.extras["nonearn_8k_reaction_day"]
    with pytest.raises(KeyError):
        S.build_inputs(d, PARAMS)
