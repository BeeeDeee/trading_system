import numpy as np
import pytest

from strategy import (PARAMS, target_weights, volume_multiple, row_nanmedian, ea_clear, spike_days,
                      anchors, weekly_book, close_leavers, _rolling_min_prev)


def test_params_match_card():
    assert PARAMS == {"leg": 0.2, "n_estimators": 100, "spike_mult": 4}


def test_volume_multiple_median_and_min_valid():
    T = 70
    dv = np.full((T, 3), np.nan)
    dv[:, 0] = 10.0
    dv[63, 0] = 40.0                       # median of 63 tens = 10 -> m = 4
    dv[:, 1] = np.arange(T, dtype=float) + 1.0   # window rows 0..62 -> values 1..63, median 32
    dv[:, 2] = 5.0
    dv[0:14, 2] = np.nan                   # 14 missing in 0..62 -> 49 valid -> undefined at row 63
    need = np.ones((T, 3), bool)
    m = volume_multiple(dv, need)
    assert np.all(np.isnan(m[:63]))
    assert m[63, 0] == pytest.approx(4.0)
    assert m[63, 1] == pytest.approx(64.0 / 32.0)
    assert np.isnan(m[63, 2])
    # row 64 of col 2: window rows 1..63 -> 13 missing -> 50 valid, median 5
    assert m[64, 2] == pytest.approx(1.0)


def test_volume_multiple_even_count_median():
    T = 64
    dv = np.full((T, 1), np.nan)
    vals = np.arange(50, dtype=float) + 1.0        # 1..50 in rows 13..62
    dv[13:63, 0] = vals
    dv[63, 0] = 51.0
    m = volume_multiple(dv, np.ones((T, 1), bool))
    assert m[63, 0] == pytest.approx(51.0 / 25.5)


def test_volume_multiple_need_mask():
    T = 70
    dv = np.full((T, 2), 10.0)
    need = np.ones((T, 2), bool)
    need[:, 1] = False
    m = volume_multiple(dv, need)
    assert np.isfinite(m[65, 0]) and np.isnan(m[65, 1])


def test_rolling_min_prev():
    x = np.array([[5.0], [3.0], [np.nan], [7.0], [1.0], [9.0]])
    r = _rolling_min_prev(x, 3)
    assert np.isnan(r[0, 0])
    assert r[1, 0] == 5.0
    assert r[2, 0] == 3.0
    assert r[4, 0] == 3.0          # rows 1..3
    assert r[5, 0] == 1.0          # rows 2..4


def test_row_nanmedian():
    x = np.array([[1.0, 3.0, np.nan, 2.0], [np.nan] * 4, [4.0, 1.0, np.nan, np.nan]])
    out = row_nanmedian(x)
    assert out[0] == 2.0 and np.isnan(out[1]) and out[2] == 2.5


def test_ea_clear_window():
    T = 20
    ea = np.zeros((T, 2))
    ea[10, 0] = 1.0
    ea[:5, 1] = np.nan
    ok = ea_clear(ea)
    assert not ok[:3].any() and not ok[T - 3:].any()
    assert ok[6, 0] and not ok[7, 0] and not ok[13, 0] and ok[14, 0]
    assert not ok[7, 1] and ok[8, 1]


def _spike_market(T=120, N=12):
    dv = np.full((T, N), 100.0)
    ea = np.zeros((T, N))
    uni = np.ones((T, N), bool)
    return dv, ea, uni


def test_spike_rules():
    dv, ea, uni = _spike_market()
    dv[80, 0] = 500.0                  # m = 5 -> spike at mult 4
    dv[80, 1] = 350.0                  # m = 3.5 -> not at 4, yes at 3
    dv[90, 2] = 600.0
    ea[92, 2] = 1.0                    # earnings within s+3 -> excluded
    dv[95, 3] = 600.0
    ea[92, 3] = np.nan                 # NaN within s-3 -> excluded
    sp, m = spike_days(dv, ea, uni, 4.0)
    assert sp[80, 0] and not sp[80, 1] and not sp[90, 2] and not sp[95, 3]
    sp3, _ = spike_days(dv, ea, uni, 3.0)
    assert sp3[80, 1]
    # market-wide day: most members spike -> median m >= 1.5 -> nothing counts
    dv2, ea2, uni2 = _spike_market()
    dv2[100, :] = 500.0
    sp2, _ = spike_days(dv2, ea2, uni2, 4.0)
    assert not sp2[100].any()
    # the market median uses universe members only
    uni2[:, 1:] = False
    uni2[:, 0] = True
    dv2[100, 0] = 100.0
    sp2, _ = spike_days(dv2, ea2, uni2, 4.0)
    assert sp2[100, 1:].all()


def _days(T):
    # consecutive weekdays from day number 0 (1970-01-01 is a Thursday): use weekdays only
    d = []
    x = 0
    while len(d) < T:
        if (x + 3) % 7 < 5:
            d.append(x)
        x += 1
    return np.array(d, dtype=np.int64)


def test_anchor_window_largest_m_and_ties():
    T = 400
    dn = _days(T)
    spike = np.zeros((T, 2), bool)
    m = np.ones((T, 2), np.float32)
    s1, s2 = 10, 15
    spike[s1, 0] = True
    m[s1, 0] = 6.0
    spike[s2, 0] = True
    m[s2, 0] = 5.0
    spike[s1, 1] = True
    m[s1, 1] = 5.0
    spike[s2, 1] = True
    m[s2, 1] = 5.0
    brow = anchors(dn, spike, m)
    for t in range(T):
        d1 = dn[t] - (dn[s1] + 366)
        d2 = dn[t] - (dn[s2] + 366)
        exp0 = s1 if abs(d1) <= 14 else (s2 if abs(d2) <= 14 else -1)
        exp1 = s2 if abs(d2) <= 14 else (s1 if abs(d1) <= 14 else -1)
        assert brow[t, 0] == exp0
        assert brow[t, 1] == exp1
    assert (brow[:, 0] >= 0).sum() > 0


def test_weekly_book_legs():
    T, N = 3, 30
    score = np.tile(np.arange(N, dtype=float), (T, 1))
    elig = np.ones((T, N), bool)
    dec = np.array([True, False, True])
    elig[2, 9:] = False                    # only 9 eligible -> flat
    w = weekly_book(score, elig, dec, 0.2)
    assert np.all(np.isnan(w[1]))
    k = 6
    assert np.allclose(w[0, N - k:], 0.5 / k) and np.allclose(w[0, :k], -0.5 / k)
    assert np.allclose(w[0, k:N - k], 0.0)
    assert np.nansum(np.abs(w[0])) == pytest.approx(1.0) and np.nansum(w[0]) == pytest.approx(0.0)
    assert np.all(w[2] == 0.0)
    w1 = weekly_book(score, np.ones((T, N), bool), dec, 0.1)
    assert (w1[0] > 0).sum() == 3
    # small n: k = max(2, floor(0.1 * 10)) = 2
    s10 = score.copy()
    s10[:, 10:] = np.nan
    w2 = weekly_book(s10, np.ones((T, N), bool), dec, 0.1)
    assert (w2[0] > 0).sum() == 2 and (w2[0] < 0).sum() == 2
    assert w2[0, 9] == pytest.approx(0.25) and w2[0, 0] == pytest.approx(-0.25)


@pytest.fixture(scope="module")
def syn():
    import dataclasses
    from lab.framework.data import synthetic
    base = synthetic()
    T, N = len(base.dates), len(base.instruments)
    rng = np.random.default_rng(7)
    listed = np.asarray(base.listed, bool)
    uni = listed & (rng.random((T, N)) < 0.9)
    ea = np.where(rng.random((T, N)) < 0.015, 1.0, 0.0).astype(np.float32)
    ea[:40] = np.nan
    dv = np.where(listed, 1e6 * np.exp(0.2 * rng.standard_normal((T, N))), np.nan)
    spikes = (rng.random((T, N)) < 0.01) & listed
    dv[spikes] *= 6.0
    extras = dict(base.extras)
    extras["ea_reaction_day"] = ea
    return dataclasses.replace(base, universe=uni, extras=extras, dollar_volume=dv.astype(np.float32))


def test_clock_features_point_in_time_and_range(syn):
    from strategy import clock_features
    if syn.universe is None or "ea_reaction_day" not in syn.extras:
        pytest.skip("synthetic market lacks universe or 8-K flag")
    win, c, g, lm, rs, vol = clock_features(syn, PARAMS)
    if win.any():
        assert np.nanmin(c) >= -14 and np.nanmax(c) <= 14
        assert np.all(np.isnan(c[~win]))
        assert np.all(np.asarray(syn.universe, bool)[win])
        assert np.all(np.nan_to_num(lm[win], nan=np.log(4.0)) >= np.log(4.0) - 1e-5)


def test_target_weights_book_shape(syn):
    if syn.universe is None or "ea_reaction_day" not in syn.extras:
        pytest.skip("synthetic market lacks universe or 8-K flag")
    w = target_weights(syn, PARAMS)
    T, N = len(syn.dates), len(syn.instruments)
    assert w.shape == (T, N)
    gross = np.nansum(np.abs(w), axis=1)
    assert np.all(gross <= 1.0 + 1e-9)
    uni = np.asarray(syn.universe, bool)
    assert not np.any((np.nan_to_num(w) != 0) & ~uni)
    full = np.all(np.isfinite(w), axis=1)
    for t in np.nonzero(full & (gross > 0))[0]:
        assert np.sum(w[t]) == pytest.approx(0.0, abs=1e-9)
        assert gross[t] == pytest.approx(1.0)
    # weights only on window stocks
    from strategy import clock_features
    win = clock_features(syn, PARAMS)[0]
    assert not np.any((np.nan_to_num(w) != 0) & full[:, None] & ~win)


def test_spike_mult_changes_window(syn):
    from strategy import clock_features
    if syn.universe is None or "ea_reaction_day" not in syn.extras:
        pytest.skip("synthetic market lacks universe or 8-K flag")
    a = clock_features(syn, dict(PARAMS, spike_mult=3))[0]
    b = clock_features(syn, dict(PARAMS, spike_mult=5))[0]
    assert a.sum() >= b.sum()


def test_close_leavers():
    T, N = 5, 3
    w = np.full((T, N), np.nan)
    w[0] = [0.5, -0.5, 0.0]
    w[3] = [0.0, 0.5, -0.5]
    dec = np.array([True, False, False, True, False])
    uni = np.ones((T, N), bool)
    uni[2, 0] = False
    uni[4, 2] = False
    uni[1, 2] = False                       # not held -> untouched
    out = close_leavers(w, dec, uni)
    assert out[2, 0] == 0.0 and np.isnan(out[2, 1]) and np.isnan(out[1, 2])
    assert out[4, 2] == 0.0 and np.isnan(out[4, 1])
