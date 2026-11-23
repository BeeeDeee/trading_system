import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS


def make_view(v, n, uni=None, trad=None, lr=None):
    T, N = v.shape
    dates = (np.arange(T) + 14613).astype("datetime64[D]")
    if uni is None:
        uni = np.ones((T, N), dtype=bool)
    if trad is None:
        trad = np.ones((T, N), dtype=bool)
    if lr is None:
        lr = np.tile(np.arange(1, N + 1, dtype=np.float32), (T, 1))
    z = np.zeros((T, N))
    return DataView(
        dates=dates,
        instruments=tuple("E%d" % i for i in range(N)),
        asset_class=tuple("us_equity" for _ in range(N)),
        ret_co=z.copy(),
        ret_oc=z.copy(),
        tradable=trad,
        listed=np.ones((T, N), dtype=bool),
        delisting=np.zeros((T, N), dtype=bool),
        close=np.full((T, N), 50.0),
        dollar_volume=np.full((T, N), 1e8),
        universe=uni,
        extras={"ins_buy_value_91d": v.astype(np.float32), "ins_buy_n_91d": n.astype(np.float32),
                "liq_rank": lr.astype(np.float32)},
        series={},
        cash_ret=np.zeros(T),
    )


def base(T=20, N=60):
    return np.zeros((T, N)), np.zeros((T, N))


def test_single_event_slot_and_hedge():
    v, n = base()
    j = 55
    v[3:, j] = 600000.0
    n[3:, j] = 1
    w = target_weights(make_view(v, n), PARAMS)
    # row 0 decided flat, rows 1-2 no decision
    assert np.all(w[0] == 0)
    assert np.all(np.isnan(w[1:3]))
    # slot open on decision rows 3..7 (enter open 4, exit open 9)
    assert w[3, j] == pytest.approx(0.08)
    assert np.all(np.isnan(w[4:8]))
    assert np.all(w[8] == 0)
    # hedge: 50 lowest liq_rank (cols 0..49), excludes slot stock
    assert np.allclose(w[3, :50], -0.08 / 50)
    assert np.all(w[3, 50:55] == 0) and np.all(w[3, 56:] == 0)
    assert np.nansum(w[3]) == pytest.approx(0.0)


def test_event_conditions():
    v, n = base()
    v[5:, 0] = 600000.0          # crossing without a new filing count: no event
    v[5:, 1] = 600000.0
    n[5:, 1] = 2                 # valid event at row 5
    v[:, 2] = np.nan
    v[6:, 2] = 600000.0          # NaN on t-1: no event
    n[6:, 2] = 1
    v[5:, 3] = 400000.0          # below X: no event
    n[5:, 3] = 1
    w = target_weights(make_view(v, n), PARAMS)
    assert w[5, 1] == pytest.approx(0.08)
    assert w[5, 0] < 0 and w[5, 3] < 0          # hedge legs, not slots
    assert np.all(np.isnan(w[6]))               # no event on row 6


def test_refire_only_after_falling_below():
    v, n = base(T=30)
    v[2:, 50] = 600000.0
    n[2:, 50] = 1
    v[4:, 50] = 900000.0                        # more buying, still above: no new event
    n[4:, 50] = 2
    v[12:, 50] = 100000.0                       # aged out
    n[12:, 50] = 1
    v[15:, 50] = 700000.0                       # new crossing
    n[15:, 50] = 2
    w = target_weights(make_view(v, n), PARAMS)
    change_rows = [s for s in range(30) if not np.all(np.isnan(w[s]))]
    assert change_rows == [0, 2, 7, 15, 20]


def test_non_member_or_untradable_no_event_and_early_close():
    v, n = base()
    v[3:, 55] = 600000.0
    n[3:, 55] = 1
    uni = np.ones(v.shape, dtype=bool)
    uni[3, 55] = False
    w = target_weights(make_view(v, n, uni=uni), PARAMS)
    assert np.all(w[0] == 0) and np.all(np.isnan(w[1:]))
    # early close when untradable on row 5
    trad = np.ones(v.shape, dtype=bool)
    trad[5, 55] = False
    w = target_weights(make_view(v, n, trad=trad), PARAMS)
    assert w[3, 55] == pytest.approx(0.08)
    assert np.all(np.isnan(w[4]))
    assert np.all(w[5] == 0)
    assert np.all(np.isnan(w[6:]))


def test_cap_over_five_slots_and_fewer_hedge_names():
    T, N = 15, 10
    v, n = np.zeros((T, N)), np.zeros((T, N))
    v[2:, :7] = 600000.0
    n[2:, :7] = 1
    w = target_weights(make_view(v, n), PARAMS)
    assert np.allclose(w[2, :7], 0.40 / 7)
    assert np.allclose(w[2, 7:], -0.40 / 3)   # only 3 eligible members
    assert np.abs(w[2]).sum() == pytest.approx(0.80)
    assert np.all(w[7] == 0)


def test_hedge_names_leaving_universe_are_zeroed():
    v, n = base()
    v[3:, 55] = 600000.0
    n[3:, 55] = 1
    uni = np.ones(v.shape, dtype=bool)
    uni[5:, 0] = False
    w = target_weights(make_view(v, n, uni=uni), PARAMS)
    assert w[5, 0] == 0.0
    assert np.all(np.isnan(np.delete(w[5], 0)))
    assert np.all(w[8] == 0)


def test_param_changes_events_and_gross():
    v, n = base()
    v[3:, 55] = 300000.0
    n[3:, 55] = 1
    w1 = target_weights(make_view(v, n), {"min_buy_usd": 500000})
    w2 = target_weights(make_view(v, n), {"min_buy_usd": 250000})
    assert np.all(np.isnan(w1[3]))
    assert w2[3, 55] == pytest.approx(0.08)
    assert np.nansum(np.abs(w2), axis=1).max() <= 0.80 + 1e-12


def test_point_in_time_truncation():
    rng = np.random.default_rng(1)
    T, N = 80, 60
    v = np.cumsum(rng.random((T, N)) < 0.05, axis=0) * 200000.0 % 1200000.0
    n = np.cumsum(rng.random((T, N)) < 0.05, axis=0).astype(float)
    full = target_weights(make_view(v, n), PARAMS)
    for cut in (10, 33, 61):
        part = target_weights(make_view(v[:cut], n[:cut]), PARAMS)
        assert np.array_equal(np.isnan(part), np.isnan(full[:cut]))
        assert np.allclose(np.nan_to_num(part), np.nan_to_num(full[:cut]))
