import numpy as np
import pytest

from strategy import target_weights, PARAMS
from strategy import (
    _days_from_ymd, week_end_rows, anchor_index, mcap_at, pool_and_ratio, cohort_size, cohort,
    id_rank, active_sets, betas, market_return,
)
from lab.framework.data import DataView


def _dates(ymd):
    y = [a for a, _, _ in ymd]
    m = [b for _, b, _ in ymd]
    d = [c for _, _, c in ymd]
    return _days_from_ymd(y, m, d).astype("datetime64[D]")


# ---------------------------------------------------------------- calendar
def test_week_end_normal_and_good_friday():
    ymd = [(2019, 4, 15), (2019, 4, 16), (2019, 4, 17), (2019, 4, 18),          # Good Friday 4-19
           (2019, 4, 22), (2019, 4, 23), (2019, 4, 24), (2019, 4, 25), (2019, 4, 26)]
    got = week_end_rows(_dates(ymd))
    assert got.tolist() == [False, False, False, True, False, False, False, False, True]


@pytest.mark.parametrize("thursday", [
    (2015, 7, 2),    # July 4 on Saturday -> observed Friday July 3
    (2014, 7, 3),    # July 4 on Friday
    (2020, 12, 24),  # Christmas on Friday
    (2010, 12, 23),  # Christmas on Saturday -> observed Friday Dec 24
    (2015, 12, 31),  # New Year 2016 on Friday
    (2008, 3, 20),   # Good Friday 2008-03-21
    (2016, 3, 24),   # Good Friday 2016-03-25
])
def test_week_end_thursday_before_friday_holiday(thursday):
    assert week_end_rows(_dates([thursday])).tolist() == [True]


def test_week_end_new_year_on_saturday_not_observed():
    # 2022-01-01 is a Saturday: NYSE trades on Friday 2021-12-31, Thursday is not the week end
    got = week_end_rows(_dates([(2021, 12, 30), (2021, 12, 31)]))
    assert got.tolist() == [False, True]


def test_week_end_is_point_in_time():
    d = np.busday_offset(_dates([(2012, 1, 2)])[0], np.arange(300))
    full = week_end_rows(d)
    for cut in (5, 37, 151, 299):
        assert (week_end_rows(d[:cut]) == full[:cut]).all()


# ---------------------------------------------------------------- market cap
def test_anchored_market_cap():
    mc = np.array([[np.nan, 50.0], [np.nan, 50.0], [100.0, 50.0], [100.0, 50.0],
                   [100.0, 60.0], [200.0, 60.0], [200.0, np.nan]], dtype=np.float32)
    tri = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.1, 1.0],
                    [1.21, 2.0], [1.21, 3.0], [1.331, 3.0]])
    a = anchor_index(mc)
    assert a[:, 0].tolist() == [-1, -1, 2, 2, 2, 5, 5]
    assert a[:, 1].tolist() == [-1, -1, -1, -1, 4, 4, 4]      # censored at the view start
    assert mcap_at(4, mc, tri, a)[0] == pytest.approx(121.0)
    assert mcap_at(6, mc, tri, a)[0] == pytest.approx(220.0)
    assert np.isnan(mcap_at(3, mc, tri, a)[1])                 # unknown anchor
    assert mcap_at(5, mc, tri, a)[1] == pytest.approx(90.0)
    assert np.isnan(mcap_at(6, mc, tri, a)[1])                 # stale / missing


# ---------------------------------------------------------------- pool and cohort
def test_pool_rules():
    member = np.array([True, True, True, True, False])
    bn = np.zeros((6, 5))
    bn[5] = [1, 0, 2, 1, 1]
    bn[0, 3] = 1                       # no new row for col 3
    bv = np.zeros((6, 5))
    bv[5] = [10, 5, 0, 7, 9]           # col 2 new rows but zero value
    mc = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    pool, ratio = pool_and_ratio(5, member, bn, bv, mc)
    assert pool.tolist() == [True, False, False, False, False]
    assert ratio[0] == pytest.approx(0.1)
    p2, _ = pool_and_ratio(4, member, bn, bv, mc)                   # t < 5: no pool
    assert not p2.any()
    bn2 = bn.copy(); bn2[0, 0] = np.nan                             # NaN insider field
    assert not pool_and_ratio(5, member, bn2, bv, mc)[0][0]
    mc2 = mc.copy(); mc2[0] = np.nan                                # NaN market cap
    assert not pool_and_ratio(5, member, bn, bv, mc2)[0][0]


def test_cohort_size():
    assert cohort_size(2, 0.2) == 0
    assert cohort_size(3, 0.2) == 1
    assert cohort_size(10, 0.2) == 2
    assert cohort_size(13, 0.2) == 3
    assert cohort_size(5, 0.3) == 2
    assert cohort_size(25, 0.1) == 3          # half up
    assert cohort_size(3, 0.5) == 1           # capped at n // 2
    assert cohort_size(30, 0.1) == 3


def test_cohort_order_and_ties():
    names = ("E10", "E9", "E100", "E5", "E7", "E8")
    rk = id_rank(names)
    assert rk.tolist() == [4, 3, 5, 0, 1, 2]
    pool = np.ones(6, dtype=bool)
    ratio = np.array([0.5, 0.5, 0.1, 0.3, 0.05, 0.2])
    top, bot = cohort(pool, ratio, 0.2, rk)          # k = 1
    assert top.tolist() == [1]                       # tie 0.5: E9 < E10
    assert bot.tolist() == [4]
    top, bot = cohort(pool, ratio, 0.3, rk)          # k = 2
    assert top.tolist() == [1, 0]
    assert bot.tolist() == [2, 4]


# ---------------------------------------------------------------- hold clock
def test_active_sets_hold_reentry_drop():
    R, N = 7, 3
    elig = np.ones((R, N), dtype=bool)
    e = [np.array([], dtype=np.int64) for _ in range(R)]
    e[0] = np.array([0, 1])
    e[1] = np.array([0])          # re-entry restarts col 0
    elig[1, 1] = False            # col 1 leaves the universe at r = 1
    e[2] = np.array([2])
    act = active_sets(e, elig, 2)
    assert act[:, 0].tolist() == [True, True, True, False, False, False, False]
    assert act[:, 1].tolist() == [True, False, False, False, False, False, False]
    assert act[:, 2].tolist() == [False, False, True, True, False, False, False]
    act4 = active_sets(e, elig, 4)
    assert act4[:, 0].tolist() == [True, True, True, True, True, False, False]


# ---------------------------------------------------------------- beta
def test_beta_and_min_obs():
    rng = np.random.default_rng(1)
    T = 300
    m = rng.normal(0, 0.01, T)
    r = np.full((T, 3), np.nan)
    r[:, 0] = 2.0 * m + 0.001
    r[-100:, 1] = 0.7 * m[-100:]          # only 100 observations
    r[:, 2] = 0.7 * m
    b = betas(T - 1, [0, 1, 2], r, m)
    assert b[0] == pytest.approx(2.0)
    assert b[1] == 1.0
    assert b[2] == pytest.approx(0.7)
    uni = np.ones((T, 3), dtype=bool)
    mr = market_return(np.nan_to_num(r[:, [0, 2]]), uni[:, :2])
    assert mr[5] == pytest.approx((2.0 * m[5] + 0.001 + 0.7 * m[5]) / 2)


# ---------------------------------------------------------------- end to end
def _view(T=40, N=8):
    start = _dates([(2019, 5, 6)])[0]                 # a Monday
    dates = np.busday_offset(start, np.arange(T)).astype("datetime64[D]")
    names = tuple("E%d" % (j + 1) for j in range(N))
    z = np.zeros((T, N))
    mc = np.full((T, N), 100.0, dtype=np.float32)
    mc[0] = np.nan                                    # first filing visible on row 1
    bn = np.zeros((T, N), dtype=np.float32)
    bv = np.zeros((T, N), dtype=np.float32)
    bn[7:, :4] = 1.0                                  # four new filings on row 7 (week 2)
    bv[7:, :4] = [10.0, 20.0, 30.0, 40.0]
    data = DataView(
        dates=dates, instruments=names, asset_class=tuple("us_equity" for _ in names),
        ret_co=z.copy(), ret_oc=z.copy(), tradable=np.ones((T, N), dtype=bool),
        listed=np.ones((T, N), dtype=bool), delisting=np.zeros((T, N), dtype=bool),
        close=np.full((T, N), 10.0), dollar_volume=np.full((T, N), 1e8),
        universe=np.ones((T, N), dtype=bool),
        extras={"ins_buy_value_91d": bv, "ins_buy_n_91d": bn, "sf1_arq_marketcap": mc},
        series={}, cash_ret=np.zeros(T),
    )
    return data


def test_end_to_end_weights():
    data = _view()
    W = target_weights(data, PARAMS)
    T, N = W.shape
    dec = [4, 9, 14, 19, 24, 29, 34, 39]
    assert np.isnan(np.delete(W, dec, axis=0)).all()
    assert (W[4] == 0).all()
    for t in (9, 14, 19, 24):                          # E4 held 4 rebalances
        assert W[t, 3] == pytest.approx(0.05)
        others = np.delete(W[t], 3)
        assert others == pytest.approx(np.full(N - 1, -0.05 / (N - 1)))   # beta 1.0 (< 126 obs)
        assert np.abs(W[t]).sum() <= 1.0 + 1e-12
    for t in (29, 34, 39):
        assert (W[t] == 0).all()


def test_end_to_end_params_change_output():
    data = _view()
    W2 = target_weights(data, {"hold_weeks": 2, "top_frac": 0.2})
    assert W2[14, 3] == pytest.approx(0.05) and (W2[19] == 0).all()
    W5 = target_weights(data, {"hold_weeks": 4, "top_frac": 0.5})   # k = 2: E4, E3
    assert W5[9, 3] == pytest.approx(0.05) and W5[9, 2] == pytest.approx(0.05)
    assert W5[9, 0] == pytest.approx(-0.10 / 6)


def test_non_member_dropped_and_late_listing():
    data = _view()
    data.universe[14:, 3] = False                     # E4 leaves LIQ-500 before rebalance 2
    W = target_weights(data, PARAMS)
    assert W[9, 3] == pytest.approx(0.05)
    assert (W[14] == 0).all()                         # nothing active -> flat, no hedge


def test_cohort_bottom_side_end_to_end_data():
    # the diagnostic's short side on the end-to-end view: bottom k of the same pool
    data = _view()
    t = 9
    mc = mcap_at(t, data.extras["sf1_arq_marketcap"], np.ones((40, 8)),
                 anchor_index(data.extras["sf1_arq_marketcap"]))
    pool, ratio = pool_and_ratio(t, np.ones(8, dtype=bool), data.extras["ins_buy_n_91d"],
                                 data.extras["ins_buy_value_91d"], mc)
    top, bot = cohort(pool, ratio, 0.2, id_rank(data.instruments))
    assert top.tolist() == [3] and bot.tolist() == [0]


def test_truncation_point_in_time():
    data = _view()
    full = target_weights(data, PARAMS)
    for cut in (10, 20, 26):
        d = _view(T=cut)
        w = target_weights(d, PARAMS)
        assert np.array_equal(np.isnan(w), np.isnan(full[:cut]))
        assert np.allclose(np.nan_to_num(w), np.nan_to_num(full[:cut]))
