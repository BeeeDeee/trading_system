import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS
import strategy as S


def day(y, m, d):
    return S._ymd(y, m, d).astype("datetime64[D]")


# ---------------------------------------------------------------- calendar (hand-checked dates)
def test_holidays_rules():
    hol = S.xnys_holidays(2003, 2008)
    for y, m, d in [(2003, 5, 26), (2003, 11, 27), (2004, 4, 9), (2004, 7, 5), (2004, 12, 24),
                    (2008, 3, 21), (2005, 1, 17), (2007, 2, 19), (2006, 9, 4), (2008, 5, 26)]:
        assert day(y, m, d) in hol, (y, m, d)
    # New Year 2005 fell on a Saturday: not observed on Friday 2004-12-31
    assert day(2004, 12, 31) not in hol
    assert day(2003, 6, 20) not in hol


def test_rebalance_and_decision_dates():
    reb, dec = S.rebalance_calendar(2003, 2008, 20)
    reb = list(reb)
    dec = list(dec)
    i = reb.index(day(2003, 6, 20))           # third Friday of June 2003
    assert dec[i] == day(2003, 5, 22)          # 20 sessions back, skipping Memorial Day
    i = reb.index(day(2003, 12, 19))
    assert dec[i] == day(2003, 11, 20)         # skipping Thanksgiving
    # 2008-03-21 (third Friday) was Good Friday: R is the Thursday before
    assert day(2008, 3, 20) in reb and day(2008, 3, 21) not in reb
    reb15, dec15 = S.rebalance_calendar(2003, 2003, 15)
    i = list(reb15).index(day(2003, 6, 20))
    assert dec15[i] == day(2003, 5, 30)


# ---------------------------------------------------------------- hand-made market
N_STOCK = 30


def make_view(n_rows=None):
    start = day(2003, 1, 2)
    end = day(2003, 12, 31)
    hol = S.xnys_holidays(2002, 2004)
    alld = start + np.arange(int((end - start).astype(np.int64)) + 1)
    dates = alld[np.is_busday(alld, holidays=hol)]
    T = len(dates)
    names = ("SPY",) + tuple("E%d" % (j + 1) for j in range(N_STOCK))
    N = len(names)
    rp = int(np.searchsorted(dates, day(2003, 3, 21)))
    fb = int(np.searchsorted(dates, day(2003, 4, 30)))
    mc = np.full((T, N), np.nan, dtype=np.float32)
    pr = np.full((T, N), np.nan, dtype=np.float32)
    g = np.zeros(N_STOCK)
    g[:24] = 0.01 * (np.arange(24) + 1)        # 0.01 .. 0.24 -> eligible
    g[24] = 0.003                              # below G_MIN
    g[25] = 0.30                               # above G_MAX
    g[26:] = 0.10
    for j in range(N_STOCK):
        c = j + 1
        mc[20:, c] = 1000.0
        pr[20:, c] = 10.0
        if j != 26:                            # stock 26: no new filing after R_prev
            mc[fb:, c] = 12.0 * 100.0 * (1.0 + g[j])
            pr[fb:, c] = 12.0
    mc[:, 30] = np.nan                         # stock 29: no fundamentals
    close = np.full((T, N), 50.0)
    close[40:, 29] = 25.0                      # stock 28: 2:1 split inside the guard window
    close[3:, 24] = 25.0                       # stock 23: split long before the S_old filing
    universe = np.ones((T, N), dtype=bool)
    universe[:, 28] = False                    # stock 27: not an S&P 500 member
    v = DataView(
        dates=dates, instruments=names, asset_class=("us_etf",) + ("us_equity",) * N_STOCK,
        ret_co=np.zeros((T, N)), ret_oc=np.zeros((T, N)), tradable=np.ones((T, N), dtype=bool),
        listed=np.ones((T, N), dtype=bool), delisting=np.zeros((T, N), dtype=bool),
        close=close, dollar_volume=np.ones((T, N)), universe=universe,
        extras={"sf1_arq_marketcap": mc, "sf1_arq_price": pr}, series={}, cash_ret=np.zeros(T))
    return v


def rows_of(v):
    e = int(np.searchsorted(v.dates, day(2003, 5, 22)))
    x = int(np.searchsorted(v.dates, day(2003, 6, 20)))
    return e, x


def test_selection_and_window_weights():
    v = make_view()
    W = target_weights(v, PARAMS)
    e, x = rows_of(v)
    spy = v.col("SPY")
    elig = [v.col("E%d" % (j + 1)) for j in range(24)]
    assert W[e, spy] == pytest.approx(0.8)
    assert np.allclose(W[e, elig], 0.2 / 24)
    others = [c for c in range(1, len(v.instruments)) if c not in elig]
    assert np.all(W[e, others] == 0)
    assert np.all(np.isnan(W[e + 1:x]))
    assert W[x, spy] == 1.0 and np.all(W[x, 1:] == 0)
    assert np.all(W[:e, spy] == 1.0) and np.all(W[:e, 1:] == 0)
    dec = ~np.isnan(W).all(axis=1)
    assert np.all(np.nansum(np.abs(W[dec]), axis=1) <= 1.0 + 1e-12)
    assert np.all(np.nan_to_num(W) >= 0)
    # September and December windows: no new filing since the previous R -> no tilt
    assert np.isfinite(W[x:]).all() and np.all(W[x:, spy] == 1.0)


def test_fewer_than_20_eligible_no_tilt():
    v = make_view()
    v.universe[:, 1:6] = False                 # 24 -> 19 eligible
    W = target_weights(v, PARAMS)
    assert np.isfinite(W).all()
    assert np.all(W[:, v.col("SPY")] == 1.0)


def test_exit_inside_window_goes_to_spy():
    v = make_view()
    e, x = rows_of(v)
    c0 = v.col("E1")
    v.universe[e + 3:, c0] = False
    W = target_weights(v, PARAMS)
    assert W[e + 3, c0] == 0.0
    assert W[e + 3, v.col("SPY")] == pytest.approx(0.8 + 0.2 / 24)
    assert np.isnan(W[e + 3, v.col("E2")])
    assert np.all(np.isnan(W[e + 4:x]))


def test_window_param_moves_decision_row():
    v = make_view()
    W15 = target_weights(v, {"window_sessions": 15})
    e15 = int(np.searchsorted(v.dates, day(2003, 5, 30)))
    assert W15[e15, v.col("SPY")] == pytest.approx(0.8)
    W20 = target_weights(v, PARAMS)
    assert not np.array_equal(np.isnan(W15), np.isnan(W20))


def test_point_in_time_truncation():
    v = make_view()
    full = target_weights(v, PARAMS)
    e, x = rows_of(v)
    for t in [e - 1, e, e + 1, e + 5, x - 1, x, x + 3]:
        tv = DataView(
            dates=v.dates[:t + 1], instruments=v.instruments, asset_class=v.asset_class,
            ret_co=v.ret_co[:t + 1], ret_oc=v.ret_oc[:t + 1], tradable=v.tradable[:t + 1],
            listed=v.listed[:t + 1], delisting=v.delisting[:t + 1], close=v.close[:t + 1],
            dollar_volume=v.dollar_volume[:t + 1], universe=v.universe[:t + 1],
            extras={k: a[:t + 1] for k, a in v.extras.items()}, series={},
            cash_ret=v.cash_ret[:t + 1])
        part = target_weights(tv, PARAMS)
        assert np.array_equal(part, full[:t + 1], equal_nan=True), t
