import dataclasses
import math

import numpy as np
import pytest

from strategy import PARAMS, target_weights, ea_window, halfday_features, valid_sessions, _lag, weekly_with_exits
from strategy import MIN_SCORED, book
from strategy import tail_events as diag_tail_events


def _skew_g1_adj(x):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    d = x - x.mean()
    m2 = (d ** 2).mean()
    m3 = (d ** 3).mean()
    return m3 / m2 ** 1.5 * math.sqrt(n * (n - 1)) / (n - 2)


def _panel(T=30, N=3, seed=1):
    rng = np.random.default_rng(seed)
    ret_co = rng.normal(0, 0.01, (T, N))
    ret_oc = rng.normal(0, 0.02, (T, N)) + rng.exponential(0.01, (T, N)) * (rng.random((T, N)) < 0.2)
    close = np.full((T, N), 10.0)
    listed = np.ones((T, N), dtype=bool)
    return ret_co, ret_oc, close, listed


def test_params_match_card():
    assert PARAMS == {"n_estimators": 100, "tail": 0.1}


def test_halfday_features_match_hand_computation():
    ret_co, ret_oc, close, listed = _panel()
    rskew, rvol, onshare, n = halfday_features(ret_co, ret_oc, close, listed)
    t = 25
    # row 0 has no previous price -> session 0 invalid; window t-20..t is fully valid at t=25
    co = np.log1p(ret_co[t - 20:t + 1, 0])
    oc = np.log1p(ret_oc[t - 20:t + 1, 0])
    x = np.concatenate([co, oc])
    assert n[t, 0] == 42
    assert rskew[t, 0] == pytest.approx(_skew_g1_adj(x), rel=1e-9)
    assert rvol[t, 0] == pytest.approx(x.std(ddof=1), rel=1e-9)
    assert onshare[t, 0] == pytest.approx((co ** 2).sum() / (x ** 2).sum(), rel=1e-9)
    assert 0.0 <= onshare[t, 0] <= 1.0


def test_min_half_day_count_and_dropped_sessions():
    ret_co, ret_oc, close, listed = _panel()
    # first session has no previous price -> rows 0..17 have < 36 half-days
    rskew, rvol, onshare, n = halfday_features(ret_co, ret_oc, close, listed)
    assert n[18, 0] == 36 and np.isfinite(rskew[18, 0])
    assert n[17, 0] == 34 and np.isnan(rskew[17, 0]) and np.isnan(rvol[17, 0]) and np.isnan(onshare[17, 0])
    # a session with a missing half is dropped entirely; 4 dropped sessions -> 34 half-days -> not scored
    rc = ret_co.copy()
    rc[22:26, 1] = np.nan
    rskew2, _, _, n2 = halfday_features(rc, ret_oc, close, listed)
    assert n2[25, 1] == 34 and np.isnan(rskew2[25, 1])
    # 3 dropped sessions -> 36 -> scored, equals the skew of the remaining 36 half-days
    rc = ret_co.copy()
    rc[23:26, 1] = np.nan
    rskew3, _, _, n3 = halfday_features(rc, ret_oc, close, listed)
    assert n3[25, 1] == 36
    keep = np.r_[5:23]
    x = np.concatenate([np.log1p(ret_co[keep, 1]), np.log1p(ret_oc[keep, 1])])
    assert rskew3[25, 1] == pytest.approx(_skew_g1_adj(x), rel=1e-9)


def test_late_listing_needs_previous_price():
    ret_co, ret_oc, close, listed = _panel()
    close[:10, 2] = np.nan
    listed[:10, 2] = False
    ok = valid_sessions(ret_co, ret_oc, close, listed)
    assert not ok[:11, 2].any() and ok[11:, 2].all()
    rskew, _, _, n = halfday_features(ret_co, ret_oc, close, listed)
    assert np.isnan(rskew[:28, 2]).all() and np.isfinite(rskew[28, 2])


def test_halfday_features_point_in_time():
    ret_co, ret_oc, close, listed = _panel(T=40)
    full = halfday_features(ret_co, ret_oc, close, listed)
    cut = 27
    part = halfday_features(ret_co[:cut], ret_oc[:cut], close[:cut], listed[:cut])
    for a, b in zip(full, part):
        np.testing.assert_array_equal(a[:cut], b)


def test_ea_window():
    ea = np.zeros((30, 2))
    ea[:5, 1] = np.nan
    ea[3, 0] = 1.0
    ea[10, 1] = 1.0
    out = ea_window(ea, 21)
    assert out[2, 0] == 0 and out[3, 0] == 1 and out[23, 0] == 1 and out[24, 0] == 0
    assert out[:10, 1].sum() == 0 and out[10, 1] == 1 and out[30 - 1, 1] == 1
    # values other than exactly 1 (and NaN) do not count
    ea2 = np.full((5, 1), 2.0)
    assert ea_window(ea2, 21).sum() == 0


def test_lag():
    x = np.arange(5.0)
    np.testing.assert_array_equal(_lag(x, 2, -1.0), [-1, -1, 0, 1, 2])


def test_tail_events_by_hand():
    T, N = 6, 5
    w = np.zeros((T, N))
    days = np.array([1, 0, 0, 1, 0, 1], dtype=bool)
    w[0] = [0.5, -0.5, 0, 0, 0]
    w[3] = [0.5, 0, -0.5, 0, 0]       # S0 stays long (no event), S2 enters short
    w[5] = [0, 0.5, -0.5, 0, 0]       # S1 enters long, S2 stays short
    score = np.ones((T, N))
    ret5 = np.tile(np.array([0.0, 0.01, 0.02, 0.03, 0.04]), (T, 1))
    uni = np.ones((T, N), dtype=bool)
    trad = np.ones((T, N), dtype=bool)
    tail_events = diag_tail_events
    mask, side = tail_events(w, score, ret5, days, uni, trad)
    # row 0: no previous decision -> both enter, but S0 has ret5 at the minimum (below p20) -> excluded
    assert mask[0].tolist() == [False, True, False, False, False] and side[0, 1] == -1
    assert mask[3].tolist() == [False, False, True, False, False] and side[3, 2] == -1
    assert mask[5].tolist() == [False, True, False, False, False] and side[5, 1] == 1
    assert not mask[[1, 2, 4]].any()
    # non-member or untradable -> no event
    uni2 = uni.copy()
    uni2[3, 2] = False
    m2, _ = tail_events(w, score, ret5, days, uni2, trad)
    assert not m2[3, 2]


def test_weekly_with_exits():
    T, N = 7, 3
    days = np.array([1, 0, 0, 0, 1, 0, 0], dtype=bool)
    w_all = np.zeros((T, N))
    w_all[0] = [0.5, -0.5, 0.0]
    w_all[4] = [0.0, -0.5, 0.5]
    w_all[2] = [9, 9, 9]               # non-decision rows of w_all are ignored
    uni = np.ones((T, N), dtype=bool)
    uni[2, 0] = False                  # S0 held long, leaves the universe on row 2 -> closed on row 2
    uni[5:, 1] = False                 # S1 held short since row 4, leaves on row 5 -> closed on row 5
    uni[1, 2] = False                  # S2 not held on row 1 -> nothing
    w = weekly_with_exits(w_all, days, uni)
    np.testing.assert_array_equal(w[0], [0.5, -0.5, 0.0])
    assert np.isnan(w[1]).all()
    assert w[2, 0] == 0.0 and np.isnan(w[2, 1:]).all()
    assert np.isnan(w[3]).all()         # S0 back in the universe: NaN keeps the closed (zero) position
    np.testing.assert_array_equal(w[4], [0.0, -0.5, 0.5])
    assert w[5, 1] == 0.0 and np.isnan(w[5, 0]) and np.isnan(w[5, 2])


@dataclasses.dataclass
class _Mini:
    dates: np.ndarray
    universe: np.ndarray


def test_book_tail_and_min_names():
    from lab.framework.data import synthetic
    dates = synthetic().dates[:10]
    T, N = 10, 200
    rng = np.random.default_rng(5)
    score = rng.normal(size=(T, N))
    uni = np.ones((T, N), dtype=bool)
    d = _Mini(dates=dates, universe=uni)
    w1, _ = book(score, d, {**PARAMS, "tail": 0.1})
    w2, _ = book(score, d, {**PARAMS, "tail": 0.2})
    r = 3
    assert (w1[r] > 0).sum() == 20 and (w1[r] < 0).sum() == 20
    assert (w2[r] > 0).sum() == 40 and (w2[r] < 0).sum() == 40
    assert np.nansum(w1[r][w1[r] > 0]) == pytest.approx(0.5) and np.nansum(w1[r][w1[r] < 0]) == pytest.approx(-0.5)
    # longs are the highest scores
    assert score[r][w1[r] > 0].min() > score[r][w1[r] == 0].max()
    # fewer than 100 scored members -> flat
    s2 = score.copy()
    s2[:, 99:] = np.nan
    w3, _ = book(s2, d, PARAMS)
    assert np.nansum(np.abs(w3)) == 0.0
    s2[:, 99] = 0.0
    w4, _ = book(s2, d, PARAMS)
    assert np.nansum(np.abs(w4[r])) == pytest.approx(1.0)


def test_synthetic_end_to_end():
    from lab.framework.data import synthetic
    base = synthetic()
    T0, N0 = len(base.dates), len(base.instruments)
    rng = np.random.default_rng(3)
    uni0 = np.isfinite(np.asarray(base.close, dtype=float)) & np.asarray(base.listed, dtype=bool)
    uni0 &= rng.random((T0, N0)) < 0.97          # some membership churn
    ea = (rng.random((T0, N0)) < 0.016).astype(np.float32)
    ea[:300, :5] = np.nan
    extras = dict(base.extras or {})
    extras["ea_reaction_day"] = ea
    data = dataclasses.replace(base, universe=uni0, extras=extras)
    w = target_weights(data, PARAMS)
    T, N = len(data.dates), len(data.instruments)
    assert w.shape == (T, N)
    gross = np.nansum(np.abs(w), axis=1)
    assert (gross <= 1.0 + 1e-9).all()
    # nothing before the first refit has enough history (expanding window, refit every 126 rows)
    assert np.nansum(np.abs(w[:126]), axis=1).max() == 0.0
    # only members are held on decision rows
    if data.universe is not None:
        held = np.nan_to_num(w, nan=0.0) != 0
        assert not (held & ~np.asarray(data.universe, dtype=bool)).any()
    if N < MIN_SCORED:
        # fewer than 100 scorable members on every date -> the card's rule says flat
        assert np.nansum(np.abs(w)) == 0.0
    else:
        # the tail parameter changes the book
        w2 = target_weights(data, {**PARAMS, "tail": 0.2})
        assert not np.array_equal(np.nan_to_num(w, nan=7.0), np.nan_to_num(w2, nan=7.0))
