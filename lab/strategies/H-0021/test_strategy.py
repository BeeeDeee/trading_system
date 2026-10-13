import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS


def make_view(T=260, N=6, first_rows=None, universe=None, tradable=None, ret=None, start_day=12786):
    """Weekday calendar from an epoch-day number; columns S0..; first_rows[j] = first bar row."""
    days = np.arange(start_day, start_day + 2 * T)
    days = days[(days + 3) % 7 < 5][:T]          # 1970-01-01 was a Thursday
    dates = days.astype("datetime64[D]")
    if first_rows is None:
        first_rows = [0] * N
    listed = np.zeros((T, N), dtype=bool)
    for j, f in enumerate(first_rows):
        if f is not None:
            listed[f:, j] = True
    close = np.where(listed, 10.0, np.nan)
    if ret is None:
        rng = np.random.default_rng(0)
        ret = rng.normal(0, 0.01, size=(T, N))
    ret_oc = np.where(listed, ret, 0.0)
    ret_co = np.zeros((T, N))
    if tradable is None:
        tradable = listed.copy()
    if universe is None:
        universe = listed.copy()
    return DataView(
        dates=dates,
        instruments=tuple("S%d" % j for j in range(N)),
        asset_class=tuple("us_equity" for _ in range(N)),
        ret_co=ret_co,
        ret_oc=ret_oc,
        tradable=tradable,
        listed=listed,
        delisting=np.zeros((T, N), dtype=bool),
        close=close,
        dollar_volume=np.where(listed, 1e7, np.nan),
        universe=universe,
        extras={},
        series={},
        cash_ret=np.zeros(T),
    )


def event_row(dates, f, lead):
    d = dates.astype(np.int64)
    return int(np.searchsorted(d, d[f] + 180 - lead, side="left"))


def test_single_event_window_and_weights():
    data = make_view(first_rows=[0, 0, 0, 0, 0, 40])
    p = dict(PARAMS)
    W = target_weights(data, p)
    t = event_row(data.dates, 40, p["lead_days"])
    # distance on the previous row is below the threshold
    assert (data.dates[t - 1] - data.dates[40]).astype(int) < 176 <= (data.dates[t] - data.dates[40]).astype(int)
    assert np.all(W[:t] == 0)
    for s in range(t, t + p["hold_days"]):
        assert W[s, 5] == pytest.approx(-0.25)
        # beta window: rows 41..t, > 60 rows, so beta is estimated; hedge = 0.25 * beta over 5 names
        assert np.all(W[s, :5] > 0)
        assert np.allclose(W[s, :5], W[s, 0])
        assert 0.25 * 0.5 - 1e-12 <= W[s, :5].sum() <= 0.25 * 2.0 + 1e-12
    assert np.all(W[t + p["hold_days"]:] == 0)


def test_pre_existing_listing_never_fires():
    data = make_view(first_rows=[0, 0, 0, 0, 0, 29])
    W = target_weights(data, PARAMS)
    assert np.all(W == 0)
    data = make_view(first_rows=[0, 0, 0, 0, 0, 30])
    W = target_weights(data, PARAMS)
    assert (W[:, 5] < 0).sum() == PARAMS["hold_days"]


def test_nonmember_at_event_row_never_fires_later():
    T, N = 260, 6
    first = [0, 0, 0, 0, 0, 40]
    data0 = make_view(first_rows=first)
    t = event_row(data0.dates, 40, PARAMS["lead_days"])
    uni = np.asarray(data0.listed).copy()
    uni[t, 5] = False
    data = make_view(first_rows=first, universe=uni)
    W = target_weights(data, PARAMS)
    assert np.all(W == 0)


def test_beta_default_when_short_history():
    # first bar at 40, event ~ row 40+122; history >= 60 -> estimated beta. Make the stock's
    # returns exactly 2x market -> clipped 2.0 hedge notional = 0.5
    T, N = 260, 6
    rng = np.random.default_rng(1)
    mkt = rng.normal(0, 0.01, size=T)
    ret = np.tile(mkt[:, None], (1, N))
    ret[:, 5] = 3.0 * mkt     # beta > 2 -> clipped
    data = make_view(first_rows=[0, 0, 0, 0, 0, 40], ret=ret)
    W = target_weights(data, PARAMS)
    t = event_row(data.dates, 40, PARAMS["lead_days"])
    # market includes S5 itself: beta = 3 / ((5+3)/6) = 2.25 -> clipped to 2.0 (but not on rows before S5 lists)
    assert W[t, :5].sum() == pytest.approx(0.25 * 2.0)
    # with no variance in the stock relative to market and equal returns, beta = 1
    ret2 = np.tile(mkt[:, None], (1, N))
    data2 = make_view(first_rows=[0, 0, 0, 0, 0, 40], ret=ret2)
    W2 = target_weights(data2, PARAMS)
    assert W2[t, :5].sum() == pytest.approx(0.25 * 1.0)


def test_beta_min_history_falls_back_to_one():
    # gaps in the bars: delete most rows of S5 so that fewer than 60 valid rows remain
    T, N = 260, 6
    rng = np.random.default_rng(2)
    mkt = rng.normal(0, 0.01, size=T)
    ret = np.tile(mkt[:, None], (1, N))
    ret[:, 5] = 0.6 * mkt
    data0 = make_view(first_rows=[0, 0, 0, 0, 0, 40], ret=ret)
    t = event_row(data0.dates, 40, PARAMS["lead_days"])
    W0 = target_weights(data0, PARAMS)
    assert W0[t, :5].sum() < 0.25 * 0.99   # estimated beta < 1
    close = np.asarray(data0.close).copy()
    close[41:t - 20, 5] = np.nan            # leaves < 60 rows with a bar in the window
    data = DataView(dates=data0.dates, instruments=data0.instruments, asset_class=data0.asset_class,
                    ret_co=data0.ret_co, ret_oc=data0.ret_oc, tradable=data0.tradable, listed=data0.listed,
                    delisting=data0.delisting, close=close, dollar_volume=data0.dollar_volume,
                    universe=data0.universe, extras={}, series={}, cash_ret=data0.cash_ret)
    W = target_weights(data, PARAMS)
    assert W[t, :5].sum() == pytest.approx(0.25)


def test_two_overlapping_events_and_gross():
    first = [0, 0, 0, 0, 40, 41]
    data = make_view(first_rows=first)
    W = target_weights(data, PARAMS)
    t1 = event_row(data.dates, 40, 4)
    t2 = event_row(data.dates, 41, 4)
    assert t2 > t1
    # when both are active each short is 0.25 (=min(0.25, 0.5/2)), hedge excludes both
    both = W[t2]
    assert both[4] == pytest.approx(-0.25) or np.abs(both).sum() == pytest.approx(1.0)
    assert both[5] == both[4]
    assert np.all(both[:4] > 0)
    assert np.all(np.abs(W).sum(axis=1) <= 1.0 + 1e-9)


def test_three_events_short_weight_is_budget_over_n():
    first = [0, 0, 0, 0, 0, 0, 0, 40, 40, 40]
    ret = np.zeros((260, 10))   # zero variance -> beta 1
    data = make_view(N=10, first_rows=first, ret=ret)
    W = target_weights(data, PARAMS)
    t = event_row(data.dates, 40, 4)
    assert np.allclose(W[t, 7:], -0.5 / 3)
    assert W[t, :7].sum() == pytest.approx(0.5)
    assert np.abs(W[t]).sum() == pytest.approx(1.0)


def test_exit_when_event_stock_untradable():
    first = [0, 0, 0, 0, 0, 40]
    data0 = make_view(first_rows=first)
    t = event_row(data0.dates, 40, 4)
    trad = np.asarray(data0.tradable).copy()
    trad[t + 2, 5] = False
    data = make_view(first_rows=first, tradable=trad)
    W = target_weights(data, PARAMS)
    assert W[t, 5] < 0 and W[t + 1, 5] < 0
    assert np.all(W[t + 2:] == 0)


def test_params_change_output():
    data = make_view(first_rows=[0, 0, 0, 0, 0, 40])
    base = target_weights(data, PARAMS)
    p = dict(PARAMS, hold_days=3)
    W = target_weights(data, p)
    assert (W[:, 5] < 0).sum() == 3 and (base[:, 5] < 0).sum() == 5
    p = dict(PARAMS, lead_days=6)
    W = target_weights(data, p)
    t6 = event_row(data.dates, 40, 6)
    t4 = event_row(data.dates, 40, 4)
    assert t6 <= t4
    assert W[t6, 5] < 0
    assert not np.array_equal(W, base) or t6 == t4


def test_point_in_time_truncation():
    data = make_view(first_rows=[0, 0, 0, 30, 45, 40, ])
    full = target_weights(data, PARAMS)
    for cut in (100, 160, 165, 170, 200):
        sub = DataView(dates=data.dates[:cut], instruments=data.instruments, asset_class=data.asset_class,
                       ret_co=data.ret_co[:cut], ret_oc=data.ret_oc[:cut], tradable=data.tradable[:cut],
                       listed=data.listed[:cut], delisting=data.delisting[:cut], close=data.close[:cut],
                       dollar_volume=data.dollar_volume[:cut], universe=data.universe[:cut], extras={},
                       series={}, cash_ret=data.cash_ret[:cut])
        assert np.array_equal(target_weights(sub, PARAMS), full[:cut])


def test_hedge_member_leaving_universe_is_dropped():
    first = [0, 0, 0, 0, 0, 40]
    data0 = make_view(first_rows=first)
    t = event_row(data0.dates, 40, 4)
    uni = np.asarray(data0.listed).copy()
    uni[t + 1:, 0] = False
    data = make_view(first_rows=first, universe=uni)
    W = target_weights(data, PARAMS)
    assert W[t, 0] > 0
    assert np.all(W[t + 1:, 0] == 0)
    assert W[t + 1, 1] == pytest.approx(W[t, 1])   # no refresh of the basket
