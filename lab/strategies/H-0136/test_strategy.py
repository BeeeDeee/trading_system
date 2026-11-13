import dataclasses

import numpy as np
import pytest
from lab.framework.data import synthetic

from strategy import target_weights, PARAMS, daily_returns, reaction_z, build_book, MAX_HOLD, MIN_SLOTS


def _flat(T, N):
    z = np.full((T, N), np.nan)
    ev = np.zeros((T, N), dtype=bool)
    uni = np.ones((T, N), dtype=bool)
    listed = np.ones((T, N), dtype=bool)
    return z, ev, uni, listed


# ---------------------------------------------------------------- returns

def test_daily_returns_compound_and_validity():
    co = np.array([[0.0], [0.01], [0.02], [0.0]])
    oc = np.array([[0.0], [0.02], [-0.01], [0.0]])
    listed = np.array([[True], [True], [True], [False]])
    close = np.array([[10.0], [10.3], [10.4], [np.nan]])
    r, valid = daily_returns(co, oc, listed, close)
    assert not valid[0, 0]                                  # no previous close
    assert r[1, 0] == pytest.approx(1.01 * 1.02 - 1)
    assert r[2, 0] == pytest.approx(1.02 * 0.99 - 1)
    assert not valid[3, 0] and np.isnan(r[3, 0])            # not listed


# ---------------------------------------------------------------- reaction z

def _reaction_market(T=70, N=4):
    r = np.zeros((T, N))
    r[1:68, 0] = np.where(np.arange(1, 68) % 2 == 0, 0.01, -0.01)
    r[68, 0] = 0.05
    r[69, 0] = 0.05
    valid = np.ones((T, N), dtype=bool)
    valid[0] = False
    r[0] = np.nan
    uni = np.ones((T, N), dtype=bool)
    ea = np.zeros((T, N))
    ea[69, 0] = 1.0
    return r, valid, uni, ea


def test_reaction_z_by_hand():
    r, valid, uni, ea = _reaction_market()
    z, ev = reaction_z(r, valid, uni, ea)
    # window rows 8..67: abnormal = 0.75 * (+-0.01), 30 of each sign, mean 0
    s = 0.0075 * np.sqrt(60.0 / 59.0)
    a = (1.05 ** 2 - 1) * 0.75
    assert ev[69, 0] and ev.sum() == 1
    assert z[69, 0] == pytest.approx(a / (s * np.sqrt(2.0)))
    assert np.isnan(z[68, 0])


def test_window_excludes_reaction_sessions_and_uses_t_minus_2():
    r, valid, uni, ea = _reaction_market()
    z0, _ = reaction_z(r, valid, uni, ea)
    r2 = r.copy()
    r2[69, 1] = 0.0        # unchanged
    r2[7, 0] = 0.5         # row 7 is outside the window 8..67
    z1, _ = reaction_z(r2, valid, uni, ea)
    assert z1[69, 0] == pytest.approx(z0[69, 0])
    r3 = r.copy()
    r3[67, 0] = 0.03       # row t-2 is inside the window
    z3, _ = reaction_z(r3, valid, uni, ea)
    assert z3[69, 0] != pytest.approx(z0[69, 0])


def test_release_ignored_when_not_member_or_short_history():
    r, valid, uni, ea = _reaction_market()
    uni2 = uni.copy()
    uni2[69, 0] = False
    _, ev = reaction_z(r, valid, uni2, ea)
    assert not ev.any()
    valid2 = valid.copy()
    valid2[8:29, 0] = False            # 21 invalid days -> 39 valid < 40
    r2 = np.where(valid2, r, np.nan)
    _, ev = reaction_z(r2, valid2, uni, ea)
    assert not ev.any()
    valid3 = valid.copy()
    valid3[8:28, 0] = False            # 40 valid -> enough
    r3 = np.where(valid3, r, np.nan)
    _, ev = reaction_z(r3, valid3, uni, ea)
    assert ev[69, 0]


def test_nan_flag_is_not_a_release():
    r, valid, uni, ea = _reaction_market()
    ea[:60, 0] = np.nan
    _, ev = reaction_z(r, valid, uni, ea)
    assert ev[69, 0] and ev.sum() == 1


def test_reaction_point_in_time():
    r, valid, uni, ea = _reaction_market()
    ea[40, 1] = 1.0
    z, ev = reaction_z(r, valid, uni, ea)
    zt, evt = reaction_z(r[:41], valid[:41], uni[:41], ea[:41])
    assert np.array_equal(ev[:41], evt) and np.allclose(z[:41], zt, equal_nan=True)
    rr = r.copy()
    rr[41:] = 0.3
    zp, _ = reaction_z(rr, valid, uni, ea)
    assert np.allclose(z[:41], zp[:41], equal_nan=True)


def test_target_weights_on_synthetic_market():
    base = synthetic()
    with pytest.raises(ValueError):
        target_weights(base, PARAMS)                          # no universe: fail loudly
    T, N = len(base.dates), len(base.instruments)
    uni = np.asarray(base.listed, dtype=bool).copy()
    with pytest.raises(KeyError):
        target_weights(dataclasses.replace(base, universe=uni, extras={}), PARAMS)
    rng = np.random.default_rng(1)
    ea = (rng.random((T, N)) < 1.0 / 63).astype(np.float32)
    data = dataclasses.replace(base, universe=uni, extras={"ea_reaction_day": ea})
    w = target_weights(data, PARAMS)
    assert w.shape == (T, N)
    assert np.all(np.abs(w).sum(axis=1) <= 1.0 + 1e-12) and w.min() >= 0
    assert np.all(w[~uni] == 0)                               # only members held
    assert (w > 0).any()
    w15 = target_weights(data, {"z_threshold": 1.5})
    assert (w15 > 0).sum() > (w > 0).sum()


# ---------------------------------------------------------------- book

def test_entry_hold_and_timeout():
    T, N = 120, 3
    z, ev, uni, listed = _flat(T, N)
    ev[5, 0] = True
    z[5, 0] = 2.5
    w = build_book(z, ev, uni, listed, 2.0)
    assert w[4, 0] == 0
    assert w[5, 0] == pytest.approx(1.0 / MIN_SLOTS)     # decided at close of 5, filled at open of 6
    assert w[5 + MAX_HOLD - 1, 0] > 0
    assert w[5 + MAX_HOLD, 0] == 0                        # sold at the open after the 80th session


def test_confirmation_restarts_clock_and_failure_exits():
    T, N = 200, 2
    z, ev, uni, listed = _flat(T, N)
    ev[5, 0], z[5, 0] = True, 2.5
    ev[50, 0], z[50, 0] = True, 2.1      # confirms
    w = build_book(z, ev, uni, listed, 2.0)
    assert w[100, 0] > 0 and w[129, 0] > 0 and w[130, 0] == 0
    ev[60, 0], z[60, 0] = True, 1.0      # next release fails
    w = build_book(z, ev, uni, listed, 2.0)
    assert w[59, 0] > 0 and w[60, 0] == 0 and w[61:, 0].sum() == 0


def test_ignored_release_keeps_position():
    T, N = 100, 1
    z, ev, uni, listed = _flat(T, N)
    ev[5, 0], z[5, 0] = True, 3.0
    w = build_book(z, ev, uni, listed, 2.0)   # a NaN release (ev False) at row 30 changes nothing
    assert w[30, 0] > 0 and w[84, 0] > 0 and w[85, 0] == 0


def test_leaving_universe_sells():
    T, N = 100, 1
    z, ev, uni, listed = _flat(T, N)
    ev[5, 0], z[5, 0] = True, 3.0
    uni[30, 0] = False
    w = build_book(z, ev, uni, listed, 2.0)
    assert w[29, 0] > 0 and w[30:, 0].sum() == 0


def test_equal_weight_above_twenty_and_gross():
    T, N = 40, 25
    z, ev, uni, listed = _flat(T, N)
    ev[3, :] = True
    z[3, :] = 2.2
    ev[10, :5] = True
    z[10, :5] = -1.0
    w = build_book(z, ev, uni, listed, 2.0)
    assert np.allclose(w[3], 1.0 / 25)
    assert np.allclose(w[10, 5:], 1.0 / 20) and np.all(w[10, :5] == 0)
    ev[12, 5:15] = True
    z[12, 5:15] = 0.0
    w = build_book(z, ev, uni, listed, 2.0)
    assert np.allclose(w[12, 15:], 1.0 / 20) and w[12].sum() == pytest.approx(0.5)
    assert np.all(w.sum(axis=1) <= 1.0 + 1e-12) and np.all(w >= 0)


def test_threshold_parameter_changes_entries():
    T, N = 20, 1
    z, ev, uni, listed = _flat(T, N)
    ev[3, 0], z[3, 0] = True, 1.8
    assert build_book(z, ev, uni, listed, 2.0).sum() == 0
    assert build_book(z, ev, uni, listed, 1.5)[3, 0] > 0
    z[3, 0] = 2.0
    assert build_book(z, ev, uni, listed, 2.0)[3, 0] > 0     # z >= threshold enters


def test_params():
    assert PARAMS == {"z_threshold": 2.0}
