import numpy as np
import pytest

from strategy import target_weights, PARAMS
from strategy import _rsum, _features, _legs, _ranks, _leg_gross, _betas, _book, _market


def test_rsum_full_window_and_nan():
    x = np.arange(10, dtype=float).reshape(10, 1)
    x[6, 0] = np.nan
    out = _rsum(x, 3)
    assert np.isnan(out[:2, 0]).all()
    assert out[2, 0] == pytest.approx(0 + 1 + 2)
    assert out[5, 0] == pytest.approx(3 + 4 + 5)
    assert np.isnan(out[6:9, 0]).all()          # windows containing the NaN
    assert out[9, 0] == pytest.approx(7 + 8 + 9)
    assert np.isnan(_rsum(x[:2], 3)).all()       # view shorter than the window


def _toy_inputs(T=80, N=6, seed=0):
    rng = np.random.default_rng(seed)
    scale = np.linspace(0.005, 0.03, N)
    rco = rng.normal(0, 1, (T, N)) * scale * 0.5
    roc = rng.normal(0, 1, (T, N)) * scale
    r = (1 + rco) * (1 + roc) - 1
    dv = rng.uniform(1e6, 2e6, (T, N))
    uni = np.ones((T, N), dtype=bool)
    return uni, rco, roc, r, dv


def test_features_windows_and_high_vol_half():
    uni, rco, roc, r, dv = _toy_inputs()
    feats, elig = _features(uni, rco, roc, r, dv)
    T, N = r.shape
    # no eligible row before the 63-day window is full
    assert not elig[:62].any()
    t = 70
    # high-vol half: vol >= median of members -> the 3 most volatile of 6 (by realized vol)
    vol = np.std(r[t - 62:t + 1], axis=0, ddof=1)
    expect = vol >= np.median(vol)
    assert (elig[t] == expect).all()
    j = np.flatnonzero(elig[t])
    # feature 1: 5-day intraday sum, demeaned over the eligible set
    s1 = roc[t - 4:t + 1].sum(axis=0)
    assert np.allclose(feats[0][t, j], s1[j] - s1[j].mean())
    # feature 3: total return over sessions t-9..t-5, demeaned
    pw = np.prod(1 + r[t - 9:t - 4], axis=0) - 1
    assert np.allclose(feats[2][t, j], pw[j] - pw[j].mean())
    # feature 4: log(5-day mean dv / 63-day mean dv)
    f4 = np.log(dv[t - 4:t + 1].mean(axis=0) / dv[t - 62:t + 1].mean(axis=0))
    assert np.allclose(feats[3][t, j], f4[j])
    assert np.allclose(feats[4][t, j], np.log(vol[j]))
    # non-eligible are NaN
    assert np.isnan(feats[0][t, ~elig[t]]).all()


def test_features_non_member_and_missing_data():
    uni, rco, roc, r, dv = _toy_inputs()
    uni[:, 5] = False                    # most volatile stock not a member
    r = r.copy()
    r[68, 4] = np.nan                    # missing day inside the window
    feats, elig = _features(uni, rco, roc, r, dv)
    assert not elig[:, 5].any()
    assert not elig[68:68 + 63, 4].any()


def test_ranks_and_legs_buffer():
    n = 21
    s = np.arange(n, dtype=float)[::-1].copy()   # name j has rank n-1-j
    ok = np.ones(n, dtype=bool)
    pos, cnt = _ranks(s, ok)
    assert cnt == n and pos[0] == 20 and pos[20] == 0
    no = np.zeros(n, dtype=bool)
    long_, short_ = _legs(pos, cnt, ok, no, no, 0.15, 0.4)
    # entry 0.15: rank >= 17 (names 0..3) long; rank <= 3 (names 17..20) short
    assert set(np.flatnonzero(long_)) == {0, 1, 2, 3}
    assert set(np.flatnonzero(short_)) == {17, 18, 19, 20}
    held_l = no.copy()
    held_l[[7, 9, 18]] = True      # ranks 13 (keep, >=12), 11 (drop), 2 (switch to short)
    held_s = no.copy()
    held_s[[12, 14]] = True        # ranks 8 (keep, <=8), 6 (keep)
    long_, short_ = _legs(pos, cnt, ok, held_l, held_s, 0.15, 0.4)
    assert set(np.flatnonzero(long_)) == {0, 1, 2, 3, 7}
    assert set(np.flatnonzero(short_)) == {12, 14, 17, 18, 19, 20}
    assert not (long_ & short_).any()
    # parameters change the legs
    l2, s2 = _legs(pos, cnt, ok, held_l, held_s, 0.2, 0.4)
    assert l2.sum() > long_.sum()
    l3, s3 = _legs(pos, cnt, ok, held_l, held_s, 0.15, 0.3)
    assert not l3[7] and not s3[12]
    # ineligible names never enter
    ok2 = ok.copy()
    ok2[0] = False
    pos2, c2 = _ranks(s, ok2)
    l4, _ = _legs(pos2, c2, ok2, held_l, held_s, 0.15, 0.4)
    assert not l4[0]


def test_leg_gross():
    assert _leg_gross(1.0, 1.0) == pytest.approx((0.5, 0.5))
    assert _leg_gross(0.8, 1.2) == pytest.approx((0.6, 0.4))
    assert _leg_gross(1.5, 0.5) == pytest.approx((0.3, 0.7))
    gl, gs = _leg_gross(0.1, 2.0)
    assert (gl, gs) == pytest.approx((0.7, 0.3))
    assert sum(_leg_gross(-1.0, 0.5)) == pytest.approx(1.0)


def test_betas():
    rng = np.random.default_rng(1)
    T = 300
    m = rng.normal(0, 0.01, T)
    r = np.column_stack([2 * m, 0.5 * m + 0.001, m.copy()])
    r[:200, 2] = np.nan            # only 100 valid obs in the last 252 -> beta 1
    b = _betas(r, m, T - 1, np.array([0, 1, 2]))
    assert b == pytest.approx([2.0, 0.5, 1.0])
    b0 = _betas(r, m, 50, np.array([0]))  # fewer than 126 rows of history -> 1
    assert b0 == pytest.approx([1.0])


def test_book_sizes_buffer_and_flat():
    rng = np.random.default_rng(2)
    T, N = 300, 40
    r = rng.normal(0, 0.01, (T, N))
    uni = np.ones((T, N), dtype=bool)
    elig = np.ones((T, N), dtype=bool)
    score = rng.normal(0, 1, (T, N))
    score[:100] = np.nan                       # before the first fit: flat
    days = np.arange(0, T, 5)
    W, ev, side = _book(score, elig, r, uni, days, 0.15, 0.4)
    for d in days:
        g = np.abs(W[d]).sum()
        if d < 100:
            assert g == 0
        else:
            assert g == pytest.approx(1.0)
            lw = W[d][W[d] > 0]
            sw = W[d][W[d] < 0]
            assert len(lw) >= 6 and len(sw) >= 6
            assert np.allclose(lw, lw[0]) and np.allclose(sw, sw[0])
            assert 0.3 - 1e-6 <= lw.sum() <= 0.7 + 1e-6
            assert g <= 1.0
    # events only on new entries, side matches the leg
    assert ev[100].sum() == 12
    assert (np.sign(W[ev]) == side[ev]).all()
    for k in range(1, len(days)):
        d, p = days[k], days[k - 1]
        if d > 100:
            cont = (np.sign(W[d]) == np.sign(W[p])) & (W[d] != 0)
            assert not ev[d][cont].any()
    # fewer than 5 names in a leg -> flat
    elig_small = np.zeros((T, N), dtype=bool)
    elig_small[:, :20] = True                  # 20 names: 3 per leg at entry 0.15
    W2, ev2, _ = _book(score, elig_small, r, uni, days, 0.15, 0.4)
    assert np.abs(W2).sum() == 0               # legs never reach 5 names (buffer starts empty)
    assert not ev2.any()


def test_market_mean_of_members():
    r = np.array([[0.01, 0.03, np.nan], [0.02, 0.04, 0.06]])
    uni = np.array([[True, True, True], [True, False, True]])
    assert _market(r, uni) == pytest.approx([0.02, 0.04])


def test_full_pipeline_synthetic():
    import dataclasses
    from lab.framework.data import synthetic
    data = synthetic()
    with pytest.raises(ValueError):
        target_weights(data, PARAMS)            # liq_n card: a missing universe must fail loudly
    data = dataclasses.replace(data, universe=np.asarray(data.listed, dtype=bool).copy())
    w1 = np.asarray(target_weights(data, PARAMS))
    w2 = np.asarray(target_weights(data, PARAMS))
    assert w1.shape == (len(data.dates), len(data.instruments))
    assert np.array_equal(np.nan_to_num(w1, nan=9.0), np.nan_to_num(w2, nan=9.0))
    rows = ~np.isnan(w1).all(axis=1)
    assert rows.any()
    assert (np.nansum(np.abs(w1[rows]), axis=1) <= 1.0 + 1e-9).all()
