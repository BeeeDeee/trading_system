import dataclasses

import numpy as np
import pytest

from strategy import target_weights, PARAMS, select_cohort, combine_cohorts, WINDOW, CAP
from lab.framework.data import synthetic


KEY = "macro_release_calendar_1d.next_release_sessions"
EVERY = 15


def _view():
    d = synthetic()
    T = len(d.dates)
    t = np.arange(T)
    s = (EVERY - (t % EVERY)).astype(float)      # release day on every row t with t % 15 == 0
    s[:5] = np.nan                               # leading NaN before the calendar starts
    return dataclasses.replace(d, series={KEY: s})


def test_end_to_end_cohorts_on_release_rows():
    d = _view()
    w = target_weights(d, PARAMS)
    T, N = w.shape
    assert T == len(d.dates) and N == len(d.instruments)
    dec = ~np.all(np.isnan(w), axis=1)
    rows = np.flatnonzero(dec)
    # decisions only on row 0, on release rows (start) and release row + hold_days (expiry)
    h = PARAMS["hold_days"]
    assert all(r == 0 or r % EVERY == 0 or (r - h) % EVERY == 0 for r in rows)
    assert len(rows) > 100
    g = np.nansum(np.abs(w), axis=1)
    assert g.max() <= 1.0 + 1e-9
    starts = rows[(rows % EVERY == 0) & (rows > 0)]
    held = w[starts]
    nz = (held != 0).sum(axis=1)
    assert np.all((nz == 0) | (nz == 2 * PARAMS["top_k"]))
    assert np.all(np.abs(held) <= 0.25 + 1e-12)
    # dollar-neutral up to the cap
    assert np.all(held.sum(axis=1) < 0.5)
    assert np.all(held[nz > 0].clip(min=0).sum(axis=1) <= 0.5 + 1e-12)
    # expiry rows go flat (non-overlapping cohorts with 15-row spacing)
    exp = rows[((rows - h) % EVERY == 0) & (rows > h)]
    np.testing.assert_allclose(w[exp], 0.0)


def test_end_to_end_point_in_time():
    d = _view()
    w = target_weights(d, PARAMS)
    for cut in (300, 1201, 2500):
        dc = dataclasses.replace(
            d, dates=d.dates[:cut], ret_co=d.ret_co[:cut], ret_oc=d.ret_oc[:cut],
            tradable=d.tradable[:cut], listed=d.listed[:cut], delisting=d.delisting[:cut],
            close=d.close[:cut], dollar_volume=d.dollar_volume[:cut],
            universe=None if d.universe is None else d.universe[:cut],
            extras={k: v[:cut] for k, v in (d.extras or {}).items()},
            series={KEY: d.series[KEY][:cut]},
            cash_ret=None if d.cash_ret is None else d.cash_ret[:cut])
        wc = target_weights(dc, PARAMS)
        np.testing.assert_array_equal(wc, w[:cut])


def test_missing_series_raises():
    d = dataclasses.replace(synthetic(), series={})
    with pytest.raises(KeyError):
        target_weights(d, PARAMS)


def _rv(T, N, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0, 0.01, size=(T, N)) * (1.0 + np.arange(N))[None, :] * 0.2
    valid = np.ones((T, N), dtype=bool)
    valid[0] = False
    r[0] = np.nan
    return r, valid


def test_select_requires_full_window():
    r, valid = _rv(100, 10)
    assert select_cohort(r, valid, WINDOW - 1, 3) is None
    assert select_cohort(r, valid, WINDOW, 3) is None        # row 0 invalid -> only 62 valid
    assert select_cohort(r, valid, WINDOW + 1, 3) is not None


def test_select_min_eligible():
    r, valid = _rv(100, 9)
    t = 80
    assert select_cohort(r, valid, t, 3) is not None          # 9 = 2*3+3
    valid2 = valid.copy()
    valid2[t - 5, 0] = False                                  # one missing day in window
    assert select_cohort(r, valid2, t, 3) is None             # 8 eligible
    valid3 = valid.copy()
    valid3[t, 4] = False                                      # no return on t
    assert select_cohort(r, valid3, t, 3) is None
    assert select_cohort(r, valid2, t, 2) is not None         # 8 >= 7


def test_ranking_and_inverse_vol_weights():
    r, valid = _rv(100, 10, seed=3)
    t = 90
    w, longs, shorts = select_cohort(r, valid, t, 3)
    sig = np.std(r[t - WINDOW:t], axis=0, ddof=1)
    z = r[t] / sig
    order = np.argsort(-z, kind="stable")
    assert set(longs) == set(order[:3])
    assert set(shorts) == set(order[-3:])
    for leg, sgn in ((longs, 1.0), (shorts, -1.0)):
        inv = 1.0 / sig[leg]
        expect = np.minimum(0.5 * inv / inv.sum(), CAP) * sgn
        np.testing.assert_allclose(w[leg], expect)
    others = np.setdiff1d(np.arange(10), np.concatenate([longs, shorts]))
    assert np.all(w[others] == 0)
    assert np.abs(w).sum() <= 1.0 + 1e-12


def test_cap_keeps_excess_in_cash():
    r, valid = _rv(100, 10, seed=1)
    t = 90
    r[t - WINDOW:t, 0] *= 0.01           # very low vol -> would dominate its leg
    r[t, 0] = 0.5                        # huge z -> long leg
    w, longs, _ = select_cohort(r, valid, t, 3)
    assert 0 in longs
    assert w[0] == pytest.approx(CAP)
    assert w[w > 0].sum() < 0.5


def test_combine_hold_and_average():
    T, N = 20, 3
    W = np.array([[0.5, -0.5, 0.0], [0.0, 0.5, -0.5]])
    out = combine_cohorts(np.array([3, 5]), W, T, 4)
    np.testing.assert_allclose(out[0], 0.0)
    assert np.all(np.isnan(out[1:3]))
    np.testing.assert_allclose(out[3], W[0])          # cohort 1 active rows 3..6
    assert np.all(np.isnan(out[4]))
    np.testing.assert_allclose(out[5], W.mean(axis=0))  # both active 5..6
    assert np.all(np.isnan(out[6]))
    np.testing.assert_allclose(out[7], W[1])          # cohort 1 expired after row 6
    np.testing.assert_allclose(out[9], 0.0)           # cohort 2 active 5..8
    assert np.all(np.isnan(out[10:]))


def test_hold_days_changes_output():
    W = np.array([[0.5, -0.5]])
    a = combine_cohorts(np.array([2]), W, 15, 3)
    b = combine_cohorts(np.array([2]), W, 15, 5)
    np.testing.assert_allclose(a[5], 0.0)
    assert np.all(np.isnan(b[5]))
    np.testing.assert_allclose(b[7], 0.0)


def test_top_k_changes_selection():
    r, valid = _rv(100, 12, seed=5)
    w3 = select_cohort(r, valid, 90, 3)[0]
    w2 = select_cohort(r, valid, 90, 2)[0]
    assert (w3 != 0).sum() == 6 and (w2 != 0).sum() == 4


def test_params_names():
    assert PARAMS == {"hold_days": 5, "top_k": 3}
