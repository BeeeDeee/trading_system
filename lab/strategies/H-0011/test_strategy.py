import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS

T = 300
N = 12
EV = 280          # breakout day of stock 0 in the base scenario


def make(ret_oc=None, close=None, universe=None, tradable=None):
    names = tuple("S%02d" % i for i in range(N))
    dates = (np.arange(T) + 10957).astype("datetime64[D]")
    ro = np.zeros((T, N)) if ret_oc is None else ret_oc
    cl = np.ones((T, N)) if close is None else close
    un = np.ones((T, N), dtype=bool) if universe is None else universe
    tr = np.ones((T, N), dtype=bool) if tradable is None else tradable
    return DataView(
        dates=dates, instruments=names, asset_class=tuple("us_equity" for _ in names),
        ret_co=np.zeros((T, N)), ret_oc=ro, tradable=tr, listed=np.ones((T, N), dtype=bool),
        delisting=np.zeros((T, N), dtype=bool), close=cl, dollar_volume=np.ones((T, N)),
        universe=un, extras={}, series={}, cash_ret=np.zeros(T),
    )


def base_returns(high_day=100, ev_ret=0.06):
    r = np.zeros((T, N))
    r[high_day, 0] = 0.10      # old high 1.10
    r[high_day + 1, 0] = -0.05  # back below: 1.045
    r[EV, 0] = ev_ret          # 1.045 * 1.06 = 1.1077 > 1.10
    r[EV, 1:4] = 0.10          # three bigger movers so stock 0 is not in the top decile
    return r


def test_breakout_long_hold_and_short_leg():
    w = target_weights(make(base_returns()), PARAMS)
    assert w.shape == (T, N)
    assert np.all(w[:EV] == 0)
    for t in range(EV, EV + PARAMS["hold_days"]):
        assert w[t, 0] == pytest.approx(0.5)
        assert np.allclose(w[t, 1:], -0.5 / (N - 1))
    assert np.all(w[EV + PARAMS["hold_days"]:] == 0)
    assert np.all(np.abs(w).sum(axis=1) <= 1 + 1e-12)
    assert np.allclose(w.sum(axis=1), 0)


def test_top_decile_day_excluded():
    w = target_weights(make(base_returns(ev_ret=0.2)), PARAMS)
    assert np.all(w == 0)


def test_gap_days_staleness():
    r = base_returns(high_day=EV - 15)  # high set 15 days before the breakout
    w21 = target_weights(make(r), {"gap_days": 21, "hold_days": 5})
    w10 = target_weights(make(r), {"gap_days": 10, "hold_days": 5})
    assert np.all(w21 == 0)
    assert w10[EV, 0] == pytest.approx(0.5)
    # boundary: high set exactly gap_days before t counts as stale
    r = base_returns(high_day=EV - 21)
    assert target_weights(make(r), {"gap_days": 21, "hold_days": 5})[EV, 0] == pytest.approx(0.5)
    assert np.all(target_weights(make(r), {"gap_days": 22, "hold_days": 5}) == 0)


def test_hold_days_changes_length():
    r = base_returns()
    for h in (3, 10):
        w = target_weights(make(r), {"gap_days": 21, "hold_days": h})
        assert np.all(w[EV:EV + h, 0] > 0)
        assert np.all(w[EV + h:] == 0)


def test_high_outside_window_is_not_the_anchor():
    # high at day 20 lies outside t-252..t-1 for t=280; the anchor is the flat 1.045 level which
    # was last touched at t-1, so it is not stale -> no event
    r = base_returns(high_day=20)
    assert np.all(target_weights(make(r), PARAMS) == 0)


def test_min_observations_late_listing():
    r = base_returns()
    for first, expect in ((40, True), (41, False)):   # window 28..279: 240 vs 239 priced days
        cl = np.ones((T, N))
        cl[:first, 0] = np.nan
        w = target_weights(make(r, close=cl), PARAMS)
        assert (w[EV, 0] > 0) == expect


def test_no_price_on_event_day():
    cl = np.ones((T, N))
    cl[EV, 0] = np.nan
    w = target_weights(make(base_returns(), close=cl), PARAMS)
    assert np.all(w[:, 0] >= 0) and w[EV, 0] == 0


def test_universe_and_tradable_required_for_event():
    un = np.ones((T, N), dtype=bool)
    un[EV, 0] = False
    assert np.all(target_weights(make(base_returns(), universe=un), PARAMS)[:, 0] >= 0)
    assert target_weights(make(base_returns(), universe=un), PARAMS)[EV, 0] == 0
    tr = np.ones((T, N), dtype=bool)
    tr[EV, 0] = False
    assert target_weights(make(base_returns(), tradable=tr), PARAMS)[EV, 0] == 0


def test_dropout_closes_long_and_nonmembers_not_held():
    un = np.ones((T, N), dtype=bool)
    un[EV + 2:, 0] = False     # leaves LIQ-500 two days after the event
    un[:, 11] = False          # never a member
    w = target_weights(make(base_returns(), universe=un), PARAMS)
    assert np.all(w[EV:EV + 2, 0] == pytest.approx(0.5))
    assert np.all(w[EV + 2:, 0] == 0)   # lab contract: only members may be held
    assert np.all(w[:, 11] == 0)
    assert np.allclose(w[EV:EV + 2, 1:11], -0.5 / 10)
    assert np.all(w[EV + 2:] == 0)      # long set empty -> flat


def test_new_event_extends_holding():
    r = base_returns()
    r[EV + 2, 0] = 0.01        # new high 1.1188 > 1.1077; but the old high is 2 days old -> not stale
    w = target_weights(make(r), PARAMS)
    assert np.all(w[EV + 5:] == 0)
    # a second stale breakthrough by another stock overlaps: both long, 0.25 each
    r = base_returns()
    r[150, 5] = 0.10
    r[151, 5] = -0.05
    r[EV + 2, 5] = 0.06
    r[EV + 2, 6:9] = 0.10
    w = target_weights(make(r), PARAMS)
    assert w[EV + 2, 0] == pytest.approx(0.25) and w[EV + 2, 5] == pytest.approx(0.25)
    assert w[EV + 5, 0] == pytest.approx(-0.5 / 11) and w[EV + 5, 5] == pytest.approx(0.5)
    assert np.all(w[EV + 7:] == 0)


def test_point_in_time_truncation():
    r = base_returns()
    full = target_weights(make(r), PARAMS)
    d = make(r)
    cut = EV + 2
    trunc = DataView(
        dates=d.dates[:cut], instruments=d.instruments, asset_class=d.asset_class,
        ret_co=d.ret_co[:cut], ret_oc=d.ret_oc[:cut], tradable=d.tradable[:cut], listed=d.listed[:cut],
        delisting=d.delisting[:cut], close=d.close[:cut], dollar_volume=d.dollar_volume[:cut],
        universe=d.universe[:cut], extras={}, series={}, cash_ret=d.cash_ret[:cut],
    )
    assert np.array_equal(target_weights(trunc, PARAMS), full[:cut])


def test_missing_universe_raises():
    d = make(base_returns(), universe=None)
    d2 = DataView(
        dates=d.dates, instruments=d.instruments, asset_class=d.asset_class, ret_co=d.ret_co,
        ret_oc=d.ret_oc, tradable=d.tradable, listed=d.listed, delisting=d.delisting, close=d.close,
        dollar_volume=d.dollar_volume, universe=None, extras={}, series={}, cash_ret=d.cash_ret,
    )
    with pytest.raises(ValueError):
        target_weights(d2, PARAMS)
