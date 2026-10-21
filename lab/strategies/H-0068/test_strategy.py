import dataclasses
import numpy as np
import pytest
from lab.framework.data import DataView, synthetic
from strategy import target_weights, PARAMS
from strategy import (formation_return, winner_flag, reaction_ok, active_slots, book,
                      compute_events, daily_returns)
T, N = 130, 150
EV = 100                     # event row
W, V, X, Z, Y = 149, 125, 148, 147, 10


def _dates(n):
    return synthetic().dates[:n]


def make_view(r=None, ea=None, universe=None, liq=None, tradable=None, listed=None):
    rng = np.random.default_rng(7)
    if r is None:
        r = rng.normal(0.0, 0.002, (T, N))
        drift = 0.001 * np.arange(N)
        r[70:95] += drift[None, :]
        r[EV, W] = 0.0
        r[EV, V] = 0.0
        r[EV, X] = 0.5           # extreme reaction
    if ea is None:
        ea = np.zeros((T, N))
        ea[EV, [W, V, X, Y]] = 1.0
        ea[EV, Z] = np.nan       # NaN counts as no event
    if universe is None:
        universe = np.ones((T, N), dtype=bool)
    if liq is None:
        liq = np.tile(np.arange(1, N + 1, dtype=np.float64), (T, 1))
    if tradable is None:
        tradable = np.ones((T, N), dtype=bool)
    if listed is None:
        listed = np.ones((T, N), dtype=bool)
    return DataView(
        dates=_dates(r.shape[0]),
        instruments=tuple("E%d" % (i + 1) for i in range(r.shape[1])),
        asset_class=tuple("us_equity" for _ in range(r.shape[1])),
        ret_co=np.zeros_like(r), ret_oc=r.copy(), tradable=tradable, listed=listed,
        delisting=np.zeros(r.shape, dtype=bool), close=np.full(r.shape, 50.0),
        dollar_volume=np.full(r.shape, 1e8), universe=universe,
        extras={"ea_reaction_day": ea, "liq_rank": liq}, series={}, cash_ret=None)


def truncate(v, k):
    return dataclasses.replace(
        v, dates=v.dates[:k], ret_co=v.ret_co[:k], ret_oc=v.ret_oc[:k], tradable=v.tradable[:k],
        listed=v.listed[:k], delisting=v.delisting[:k], close=v.close[:k], dollar_volume=v.dollar_volume[:k],
        universe=v.universe[:k], extras={key: val[:k] for key, val in v.extras.items()})


P = {"formation_days": 20, "winner_quantile": 0.2}


# ---------- formation return ----------

def test_formation_window_and_coverage():
    Tn = 40
    r = np.full((Tn, 3), 0.01)
    valid = np.ones((Tn, 3), dtype=bool)
    valid[20:22, 1] = False            # 2 missing in some windows -> 18/20 = 90 % still ok
    valid[20:23, 2] = False            # 3 missing -> 17/20 < 90 %
    F = formation_return(np.where(valid, r, 0.0), valid, 10)
    # window rows t-15 .. t-6 needs t >= 15
    assert np.all(np.isnan(F[:15]))
    assert F[15, 0] == pytest.approx(1.01 ** 10 - 1)
    F20 = formation_return(np.where(valid, r, 0.0), valid, 20)
    assert np.all(np.isnan(F20[:25]))
    assert F20[30, 0] == pytest.approx(1.01 ** 20 - 1)
    assert F20[30, 1] == pytest.approx(1.01 ** 18 - 1)    # 18 valid of 20: eligible
    assert np.isnan(F20[30, 2])                          # 17 of 20: not eligible
    # the five sessions before t and t itself are excluded; t-6 is included
    r2 = np.zeros((Tn, 1))
    v2 = np.ones((Tn, 1), dtype=bool)
    r2[25, 0] = 0.5
    F2 = formation_return(r2, v2, 10)
    assert F2[30, 0] == pytest.approx(0.0)    # row 25 = t-5 excluded
    assert F2[31, 0] == pytest.approx(0.5)    # row 25 = t-6 included
    assert F2[40 - 1, 0] == pytest.approx(0.5)  # t=39: window 24..33
    assert F2[35, 0] == pytest.approx(0.5)


def test_daily_return_compounds_co_and_oc():
    co = np.array([[0.1], [np.nan]])
    oc = np.array([[0.1], [0.2]])
    r, valid = daily_returns(co, oc, np.ones((2, 1), dtype=bool), np.ones((2, 1)))
    assert r[0, 0] == pytest.approx(0.21)
    assert not valid[1, 0] and r[1, 0] == 0.0


# ---------- winner flag ----------

def test_winner_quantile_members_only():
    F = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 100.0, np.nan]])
    member = np.ones((1, 12), dtype=bool)
    member[0, 10] = False
    w = winner_flag(F, member, 0.2)
    # 0.8 quantile of 0..9 = 7.2 -> 8 and 9; the non-member with F=100 is ignored
    assert list(np.flatnonzero(w[0])) == [8, 9]
    w3 = winner_flag(F, member, 0.3)   # 0.7 quantile = 6.3 -> 7, 8, 9
    assert list(np.flatnonzero(w3[0])) == [7, 8, 9]
    assert not winner_flag(np.full((1, 3), np.nan), np.ones((1, 3), dtype=bool), 0.2).any()


# ---------- reaction filter ----------

def test_reaction_filter_z_and_min_obs():
    Tn, a = 80, 0.01
    sign = np.where(np.arange(Tn) % 2 == 0, 1.0, -1.0)
    r = np.zeros((Tn, 2))
    r[:, 0] = a * sign
    r[:, 1] = -a * sign                 # equal-weight mean is 0 -> e = r
    valid = np.ones((Tn, 2), dtype=bool)
    member = np.ones((Tn, 2), dtype=bool)
    sd = a * np.sqrt(60.0 / 59.0)       # 30 x +a and 30 x -a over rows t-65..t-6
    t = 70
    r[t, 0], r[t, 1] = 1.99 * sd, -1.99 * sd
    ok = reaction_ok(r, valid, member)
    assert ok[t, 0] and ok[t, 1]
    r[t, 0], r[t, 1] = 2.01 * sd, -2.01 * sd
    ok = reaction_ok(r, valid, member)
    assert not ok[t, 0] and not ok[t, 1]
    assert not ok[:65].any()            # sd window not available yet
    # fewer than 40 valid residuals -> not ok
    r[t, 0], r[t, 1] = 0.0, 0.0
    v2 = valid.copy()
    v2[10:31, 0] = False                # 21 of rows 5..64 missing -> 39 valid
    ok2 = reaction_ok(np.where(v2, r, 0.0), v2, member)
    assert not ok2[t, 0]


# ---------- slots ----------

def test_slots_hold_restart_and_early_close():
    Tn = 20
    ev = np.zeros((Tn, 3), dtype=bool)
    ev[2, 0] = True
    ev[2, 1] = True
    ev[5, 1] = True                      # re-trigger restarts the clock
    ev[2, 2] = True
    member = np.ones((Tn, 3), dtype=bool)
    trad = np.ones((Tn, 3), dtype=bool)
    member[4, 2] = False                 # leaves the universe on row 4
    act = active_slots(ev, member, trad)
    assert list(np.flatnonzero(act[:, 0])) == [2, 3, 4, 5, 6]          # closed at row 7 (open of t+6)
    assert list(np.flatnonzero(act[:, 1])) == [2, 3, 4, 5, 6, 7, 8, 9]
    assert list(np.flatnonzero(act[:, 2])) == [2, 3]
    trad2 = trad.copy()
    trad2[3, 0] = False
    act2 = active_slots(ev, member, trad2)
    assert list(np.flatnonzero(act2[:, 0])) == [2]


# ---------- book ----------

def test_book_weights():
    Nn = 150
    act = np.zeros((3, Nn), dtype=bool)
    act[1, [0, 5, 7]] = True
    act[2, 100:120] = True
    member = np.ones((3, Nn), dtype=bool)
    member[1, 1] = False
    liq = np.tile(np.arange(1, Nn + 1, dtype=np.float64), (3, 1))
    w = book(act, member, liq)
    assert np.all(w[0] == 0.0)
    assert np.allclose(w[1, [0, 5, 7]], -0.04)
    longs = np.flatnonzero(w[1] > 0)
    expect = [j for j in range(Nn) if j not in (0, 1, 5, 7)][:100]
    assert list(longs) == expect
    assert np.allclose(w[1, longs], 0.12 / 100)
    assert w[1].sum() == pytest.approx(0.0)
    # 20 slots: -min(0.04, 0.5/20) = -0.025, gross = 1
    assert np.allclose(w[2, 100:120], -0.025)
    assert list(np.flatnonzero(w[2] > 0)) == list(range(100))
    assert np.abs(w[2]).sum() == pytest.approx(1.0)
    assert w[2].sum() == pytest.approx(0.0)


# ---------- end to end ----------

def test_events_end_to_end():
    v = make_view()
    ev = compute_events(v, P)
    assert set(np.flatnonzero(ev[EV])) == {W, V}      # X extreme, Z NaN flag, Y not a winner
    assert ev.sum() == 2
    ev1 = compute_events(v, {"formation_days": 20, "winner_quantile": 0.1})
    assert set(np.flatnonzero(ev1[EV])) == {W}


def test_weights_end_to_end():
    v = make_view()
    w = target_weights(v, P)
    assert w.shape == (T, N)
    assert np.all(w[:EV] == 0.0)
    for t in range(EV, EV + 5):
        assert np.allclose(w[t, [W, V]], -0.04)
        assert np.allclose(w[t, :100], 0.08 / 100)
        assert w[t].sum() == pytest.approx(0.0)
    assert np.all(w[EV + 5:] == 0.0)
    assert np.all(np.abs(w).sum(axis=1) <= 1.0 + 1e-12)
    w1 = target_weights(v, {"formation_days": 20, "winner_quantile": 0.1})
    assert np.allclose(w1[EV, W], -0.04) and w1[EV, V] == 0.0
    assert np.allclose(w1[EV, :100], 0.04 / 100)


def test_not_member_no_event_and_no_weight():
    uni = np.ones((T, N), dtype=bool)
    uni[EV, V] = False
    uni[EV + 2, W] = False
    v = make_view(universe=uni)
    w = target_weights(v, P)
    assert w[EV, V] == 0.0
    assert np.allclose(w[EV:EV + 2, W], -0.04)
    assert np.all(w[EV + 2:, W] == 0.0)
    assert np.all(w[EV + 2:] == 0.0)
    assert np.all(w[~uni] == 0.0)


def test_point_in_time_truncation():
    v = make_view()
    w = target_weights(v, P)
    for k in (EV - 1, EV + 1, EV + 3, T - 10):
        wk = target_weights(truncate(v, k), P)
        assert np.array_equal(wk, w[:k])


def test_primary_params_and_missing_inputs():
    assert PARAMS == {"formation_days": 252, "winner_quantile": 0.2}
    v = make_view()
    with pytest.raises(KeyError):
        target_weights(dataclasses.replace(v, extras={"liq_rank": v.extras["liq_rank"]}), P)
    with pytest.raises(ValueError):
        target_weights(dataclasses.replace(v, universe=None), P)
