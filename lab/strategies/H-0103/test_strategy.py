import numpy as np
import pytest
from lab.framework.data import synthetic
from strategy import (PARAMS, build_book, eligible, lag, lagged_median_strict, raw_features,
                      rolling_all, rolling_std_strict, target_weights, tcn_ridge_events)


class FakeView:
    """Minimal stand-in exposing the fields raw_features reads."""

    def __init__(self, ret_co, ret_oc, dv, listed):
        self.ret_co = ret_co
        self.ret_oc = ret_oc
        self.dollar_volume = dv
        self.listed = listed


def test_params_match_card():
    assert PARAMS == {"seq_len": 10, "top": 0.2}


def test_lag_and_rolling_all():
    x = np.arange(6.0)[:, None]
    l2 = lag(x, 2)
    assert np.isnan(l2[:2]).all() and l2[2, 0] == 0.0 and l2[5, 0] == 3.0
    assert np.isnan(lag(x, 10)).all()
    ok = np.array([1, 1, 1, 0, 1, 1, 1], dtype=bool)[:, None]
    ra = rolling_all(ok, 3)[:, 0]
    assert ra.tolist() == [False, False, True, False, False, False, True]
    # view shorter than the window
    assert not rolling_all(ok[:2], 3).any()


def test_rolling_std_strict():
    rng = np.random.default_rng(1)
    r = rng.standard_normal((40, 2)) * 0.01
    r[30, 1] = np.nan
    sd = rolling_std_strict(r, 21)
    assert np.isnan(sd[:20]).all()
    assert sd[20, 0] == pytest.approx(np.std(r[0:21, 0], ddof=1), rel=1e-9)
    assert sd[39, 0] == pytest.approx(np.std(r[19:40, 0], ddof=1), rel=1e-9)
    assert np.isnan(sd[30:40, 1]).all() and np.isfinite(sd[29, 1])


def test_lagged_median_strict_excludes_today():
    x = np.arange(1.0, 81.0)[:, None] * np.ones((1, 2))
    x[70, 1] = np.nan
    m = lagged_median_strict(x, 63)
    assert np.isnan(m[:63]).all()
    # row 63 uses rows 0..62 -> values 1..63, median 32
    assert m[63, 0] == 32.0
    assert m[79, 0] == np.median(x[16:79, 0])
    assert np.isfinite(m[70, 1]) and np.isnan(m[71:80, 1]).all()
    assert np.isnan(lagged_median_strict(x[:50], 63)).all()


def test_raw_features_by_hand():
    T, N = 80, 2
    ret_co = np.full((T, N), 0.01)
    ret_oc = np.full((T, N), -0.02)
    dv = np.full((T, N), 1e6)
    dv[70, 0] = 2e6
    listed = np.ones((T, N), dtype=bool)
    listed[:30, 1] = False                    # late listing
    f1, f2, f3, f4, f5 = raw_features(FakeView(ret_co, ret_oc, dv, listed))
    assert f1[5, 0] == -0.02 and f2[5, 0] == 0.01
    assert np.isnan(f1[10, 1]) and np.isnan(f2[10, 1])
    r = 1.01 * 0.98 - 1
    assert f4[4, 0] == pytest.approx((1 + r) ** 5 - 1)
    assert np.isnan(f4[3, 0]) and np.isnan(f4[33, 1]) and np.isfinite(f4[34, 1])
    assert f5[20, 0] == pytest.approx(0.0, abs=1e-12) and np.isnan(f5[49, 1]) and np.isfinite(f5[50, 1])
    assert f3[70, 0] == pytest.approx(np.log(2.0))
    assert f3[69, 0] == pytest.approx(0.0)
    assert np.isnan(f3[62, 0]) and f3[63, 0] == pytest.approx(0.0)
    assert np.isnan(f3[:, 1]).all()           # listed at row 30: needs rows 30..92 before t = 93 > T


def test_raw_features_point_in_time():
    rng = np.random.default_rng(2)
    T, N = 120, 3
    ret_co = rng.standard_normal((T, N)) * 0.01
    ret_oc = rng.standard_normal((T, N)) * 0.01
    dv = np.exp(rng.standard_normal((T, N)))
    listed = np.ones((T, N), dtype=bool)
    full = raw_features(FakeView(ret_co, ret_oc, dv, listed))
    cut = 90
    part = raw_features(FakeView(ret_co[:cut], ret_oc[:cut], dv[:cut], listed[:cut]))
    for a, b in zip(full, part):
        np.testing.assert_array_equal(a[:cut], b)


def _scores(T, N, rng):
    return rng.standard_normal((T, N))


def test_eligibility_high_vol_half():
    score = np.array([[1.0, 2.0, 3.0, np.nan, 5.0]])
    vol = np.array([[0.1, 0.2, 0.3, 0.9, 0.4]])
    uni = np.array([[True, True, True, True, False]])
    seq_ok = np.ones((1, 5), dtype=bool)
    e = eligible(score, vol, uni, seq_ok)
    # scored members: cols 0,1,2 (col 3 no score, col 4 not a member); median vol 0.2
    assert e[0].tolist() == [False, True, True, False, False]


def test_book_weights_and_flat_rules():
    rng = np.random.default_rng(3)
    T, N = 6, 50
    score = _scores(T, N, rng)
    elig = np.ones((T, N), dtype=bool)
    elig[3, 19:] = False                      # only 19 eligible -> flat
    uni = np.ones((T, N), dtype=bool)
    dec = np.array([True, False, False, True, True, False])
    w = build_book(score, elig, dec, uni, 0.2)
    assert np.isnan(w[1]).all() and np.isnan(w[2]).all() and np.isnan(w[5]).all()
    r0 = w[0]
    assert (r0 > 0).sum() == 10 and (r0 < 0).sum() == 10
    assert r0[r0 > 0].sum() == pytest.approx(0.5) and r0[r0 < 0].sum() == pytest.approx(-0.5)
    order = np.argsort(score[0])
    assert (r0[order[-10:]] > 0).all() and (r0[order[:10]] < 0).all()
    assert np.allclose(w[3], 0.0)
    for t in (0, 4):
        assert np.nansum(np.abs(w[t])) <= 1.0 + 1e-9
    w3 = build_book(score, elig, dec, uni, 0.3)
    assert (w3[0] > 0).sum() == 15


def test_book_exits_when_leaving_universe():
    rng = np.random.default_rng(4)
    T, N = 5, 40
    score = _scores(T, N, rng)
    elig = np.ones((T, N), dtype=bool)
    uni = np.ones((T, N), dtype=bool)
    dec = np.array([True, False, False, False, True])
    held = int(np.argmax(score[0]))
    uni[2, held] = False
    elig[2, held] = False
    w = build_book(score, elig, dec, uni, 0.2)
    assert w[0, held] > 0
    assert np.isnan(w[1, held]) and w[2, held] == 0.0
    others = np.delete(np.arange(N), held)
    assert np.isnan(w[2, others]).all()


def test_events_tcn_only():
    T, N = 1, 40
    st = np.arange(N, dtype=float)[None, :]
    sr = st.copy()
    sr[0, 39], sr[0, 0] = sr[0, 0], sr[0, 39]   # ridge swaps the extreme names
    vol = np.ones((T, N))
    uni = np.ones((T, N), dtype=bool)
    seq_ok = np.ones((T, N), dtype=bool)
    dec = np.array([True])
    tradable = np.ones((T, N), dtype=bool)
    mask, side = tcn_ridge_events(st, sr, vol, uni, seq_ok, dec, tradable, 0.2)
    assert mask[0].sum() == 2
    assert mask[0, 39] and side[0, 39] == 1.0
    assert mask[0, 0] and side[0, 0] == -1.0
    mask2, _ = tcn_ridge_events(st, st, vol, uni, seq_ok, dec, tradable, 0.2)
    assert not mask2.any()
    mask3, _ = tcn_ridge_events(st, sr, vol, uni, seq_ok, np.array([False]), tradable, 0.2)
    assert not mask3.any()


def test_synthetic_end_to_end():
    data = synthetic()
    w = target_weights(data, PARAMS)
    assert w.shape == (len(data.dates), len(data.instruments))
    g = np.nansum(np.abs(w), axis=1)
    assert (g <= 1.0 + 1e-9).all()
