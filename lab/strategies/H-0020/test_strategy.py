import dataclasses

import numpy as np
import pytest

from lab.framework.data import DataView, synthetic
from strategy import PARAMS, target_weights
import strategy as S


def day(y, m, d):
    """Calendar day as int days since the numpy epoch (arithmetic only)."""
    months = np.array([(y - 1970) * 12 + (m - 1)], dtype=np.int64)
    return int(months.astype("datetime64[M]").astype("datetime64[D]").astype(np.int64)[0]) + d - 1


def as_dates(ints):
    return np.asarray(ints, dtype=np.int64).astype("datetime64[D]")


def phase_of(y, m, d):
    ph, cy = S.schedule(as_dates([day(y, m, d)]))
    return int(ph[0])


# ----------------------------------------------------------------------------- calendar
def test_holidays():
    assert S._is_nyse_holiday(np.array([day(2003, 4, 18)]))[0]       # Good Friday
    assert S._is_nyse_holiday(np.array([day(2004, 4, 9)]))[0]        # Good Friday
    assert S._is_nyse_holiday(np.array([day(2008, 3, 21)]))[0]       # Good Friday (early Easter)
    assert S._is_nyse_holiday(np.array([day(2020, 12, 25)]))[0]      # Christmas
    assert S._is_nyse_holiday(np.array([day(2004, 12, 24)]))[0]      # Christmas observed on Friday
    assert S._is_nyse_holiday(np.array([day(2020, 1, 20)]))[0]       # MLK
    assert S._is_nyse_holiday(np.array([day(2026, 6, 19)]))[0]       # Juneteenth (from 2022)
    assert not S._is_nyse_holiday(np.array([day(2020, 6, 19)]))[0]  # no Juneteenth before 2022
    assert not S._is_nyse_holiday(np.array([day(2020, 1, 21)]))[0]
    assert not S._is_nyse_holiday(np.array([day(2003, 4, 17)]))[0]


def test_schedule_plain_month():
    # Jan 2020: third Friday = Jan 17; OE week Jan 13-17; POST Jan 20 (MLK) - 24
    assert phase_of(2020, 1, 9) == S.PH_NONE
    assert phase_of(2020, 1, 10) == S.PH_OE      # D1: Friday before the OE week
    assert phase_of(2020, 1, 16) == S.PH_OE
    assert phase_of(2020, 1, 17) == S.PH_POST    # D2: OE Friday (next trading day is Tue after MLK)
    assert phase_of(2020, 1, 23) == S.PH_POST
    assert phase_of(2020, 1, 24) == S.PH_NONE    # D3


def test_schedule_holiday_fridays():
    # Apr 2003: third Friday Apr 18 is Good Friday -> OE week ends Thursday Apr 17 (D2)
    assert phase_of(2003, 4, 16) == S.PH_OE
    assert phase_of(2003, 4, 17) == S.PH_POST
    # Apr 2004: Good Friday Apr 9 is the Friday before the OE week -> D1 is Thursday Apr 8
    assert phase_of(2004, 4, 7) == S.PH_NONE
    assert phase_of(2004, 4, 8) == S.PH_OE
    # Dec 2020: POST week Dec 21-25, Christmas Friday -> D3 is Thursday Dec 24
    assert phase_of(2020, 12, 23) == S.PH_POST
    assert phase_of(2020, 12, 24) == S.PH_NONE


def test_cycle_ids_link_oe_and_post():
    ph, cy = S.schedule(as_dates([day(2020, 1, 10), day(2020, 1, 17)]))
    assert ph[0] == S.PH_OE and ph[1] == S.PH_POST
    assert cy[0] == cy[1] == day(2020, 1, 17)


# ----------------------------------------------------------------------------- hand-made market
BETAS = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0])


def make_view(T=300, n=8, seed=0):
    start = day(2018, 1, 1)   # a Monday
    dates = np.busday_offset(as_dates([start])[0], np.arange(T), roll="forward")
    rng = np.random.default_rng(seed)
    m = rng.normal(0.0, 0.01, T)
    ret_oc = (m[:, None] * BETAS[None, :n]).astype(np.float64)
    ret_co = np.zeros((T, n))
    dv = np.tile((np.arange(n) + 1.0) * 1e6, (T, 1))
    ones = np.ones((T, n), dtype=bool)
    return DataView(
        dates=dates,
        instruments=tuple("S%02d" % j for j in range(n)),
        asset_class=tuple("us_equity" for _ in range(n)),
        ret_co=ret_co,
        ret_oc=ret_oc,
        tradable=ones.copy(),
        listed=ones.copy(),
        delisting=np.zeros((T, n), dtype=bool),
        close=np.full((T, n), 10.0),
        dollar_volume=dv,
        universe=ones.copy(),
    )


def decision_rows(w):
    return np.flatnonzero(~np.isnan(w).all(axis=1))


def weekday(dates):
    return (np.asarray(dates).astype("datetime64[D]").astype(np.int64) + 3) % 7


def dom(dates):
    x = np.asarray(dates).astype("datetime64[D]")
    return (x - x.astype("datetime64[M]").astype("datetime64[D]")).astype(np.int64) + 1


P2 = {"leg_n": 2, "lookback_days": 63}


def first_cycle_after(w, dates, t0):
    rows = [t for t in decision_rows(w) if t >= t0]
    d1 = [t for t in rows if 8 <= dom(dates[t:t + 1])[0] <= 14][0]
    d2 = [t for t in rows if t > d1][0]
    d3 = [t for t in rows if t > d2][0]
    return d1, d2, d3


def test_decision_calendar_and_nan_rows():
    v = make_view()
    w = target_weights(v, P2)
    rows = decision_rows(w)
    assert rows[0] == 0
    later = rows[1:]
    assert (weekday(v.dates[later]) == 4).all()                 # all Fridays in 2018 (no holiday Fridays)
    d = dom(v.dates[later])
    assert set(np.unique(((d - 1) // 7))) <= {1, 2, 3}          # days 8-14, 15-21, 22-28
    # non-decision rows are NaN entirely
    mask = np.ones(len(w), bool)
    mask[rows] = False
    assert np.isnan(w[mask]).all()


def test_oe_and_post_books_beta_neutral():
    v = make_view()
    w = target_weights(v, P2)
    d1, d2, d3 = first_cycle_after(w, v.dates, 140)
    # stock betas on the EW mean: 1/1.25 = 0.8 and 2/1.25 = 1.6 -> b_H = 1.6, b_L = 0.8
    g_h, g_l = 0.8 / 2.4, 1.6 / 2.4
    exp = np.zeros(8)
    exp[[6, 7]] = g_h / 2
    exp[[0, 1]] = -g_l / 2
    np.testing.assert_allclose(w[d1], exp, atol=1e-9)
    np.testing.assert_allclose(w[d2], -exp, atol=1e-9)
    np.testing.assert_allclose(w[d3], np.zeros(8), atol=0)
    assert abs(np.abs(w[d1]).sum() - 1.0) < 1e-9
    assert d3 - d2 == 5 and d2 - d1 == 5


def test_post_drops_untradable_and_rescales():
    v = make_view()
    w0 = target_weights(v, P2)
    d1, d2, d3 = first_cycle_after(w0, v.dates, 140)
    trad = v.tradable.copy()
    trad[d2, 7] = False
    v2 = dataclasses.replace(v, tradable=trad)
    w = target_weights(v2, P2)
    g_h, g_l = 0.8 / 2.4, 1.6 / 2.4
    exp = np.zeros(8)
    exp[6] = -g_h            # HIGH leg rescaled to its one remaining member
    exp[[0, 1]] = g_l / 2
    np.testing.assert_allclose(w[d2], exp, atol=1e-9)
    assert abs(np.abs(w[d2]).sum() - 1.0) < 1e-9
    np.testing.assert_allclose(w[d1], w0[d1])     # D1 groups unchanged


def test_groups_kept_from_d1():
    # a ranking change between D1 and D2 must not change the D2 stocks
    v = make_view()
    w0 = target_weights(v, P2)
    d1, d2, d3 = first_cycle_after(w0, v.dates, 140)
    dv = v.dollar_volume.copy()
    dv[d1 + 1:, 2] = 1e12
    w = target_weights(dataclasses.replace(v, dollar_volume=dv), P2)
    np.testing.assert_allclose(w[d2], w0[d2])


def test_short_history_beta_one_and_coverage():
    v = make_view()
    w = target_weights(v, P2)
    rows = decision_rows(w)
    d1_rows = [t for t in rows if 8 <= dom(v.dates[t:t + 1])[0] <= 14]
    # D1 rows before 0.8*63 rows of dollar volume exist: flat
    early = [t for t in d1_rows if t + 1 < 0.8 * 63]
    assert early and all(np.all(w[t] == 0) for t in early)
    # first D1 with coverage but < 126 return observations: betas = 1 -> equal legs 0.5 / 0.5
    mid = [t for t in d1_rows if 0.8 * 63 <= t + 1 < 126]
    assert mid
    t = mid[0]
    exp = np.zeros(8)
    exp[[6, 7]] = 0.25
    exp[[0, 1]] = -0.25
    np.testing.assert_allclose(w[t], exp, atol=1e-12)


def test_late_listing_and_nonmembers_excluded():
    v = make_view()
    w0 = target_weights(v, P2)
    d1, d2, d3 = first_cycle_after(w0, v.dates, 140)
    dv = v.dollar_volume.copy()
    dv[: d1 - 20, 7] = np.nan            # only 21 of 63 days with dollar volume -> not eligible
    uni = v.universe.copy()
    uni[d1, 0] = False                  # not a member at D1 -> not eligible
    w = target_weights(dataclasses.replace(v, dollar_volume=dv, universe=uni), P2)
    assert w[d1, 7] == 0 and w[d1, 0] == 0
    assert w[d1, 6] > 0 and w[d1, 5] > 0
    assert w[d1, 1] < 0 and w[d1, 2] < 0


def test_params_change_output():
    v = make_view()
    base = target_weights(v, P2)
    d1, _, _ = first_cycle_after(base, v.dates, 140)
    w1 = target_weights(v, {"leg_n": 1, "lookback_days": 63})
    assert np.count_nonzero(w1[d1]) == 2 and np.count_nonzero(base[d1]) == 4
    # lookback: stock 7 (normally the most traded) is barely traded in the last 21 rows before D1
    dv = v.dollar_volume.copy()
    dv[d1 - 20: d1 + 1, 7] = 1e3
    v2 = dataclasses.replace(v, dollar_volume=dv)
    ws = target_weights(v2, {"leg_n": 2, "lookback_days": 21})
    wl = target_weights(v2, {"leg_n": 2, "lookback_days": 126})
    assert ws[d1, 7] < 0           # 21-day mean: lowest -> LOW leg
    assert wl[d1, 7] > 0           # 126-day mean (105*8e6 + 21*1e3)/126 ~ 6.7e6: still top 2 -> HIGH leg


def test_point_in_time_truncation():
    v = make_view(T=300)
    w = target_weights(v, P2)
    for cut in (60, 131, 150, 201, 250):
        vt = DataView(
            dates=v.dates[:cut], instruments=v.instruments, asset_class=v.asset_class,
            ret_co=v.ret_co[:cut], ret_oc=v.ret_oc[:cut], tradable=v.tradable[:cut], listed=v.listed[:cut],
            delisting=v.delisting[:cut], close=v.close[:cut], dollar_volume=v.dollar_volume[:cut],
            universe=v.universe[:cut],
        )
        wt = target_weights(vt, P2)
        np.testing.assert_array_equal(np.isnan(wt), np.isnan(w[:cut]))
        np.testing.assert_allclose(np.nan_to_num(wt), np.nan_to_num(w[:cut]))


def test_synthetic_gross_and_missing_universe():
    d = synthetic()
    with pytest.raises(ValueError):
        target_weights(d, PARAMS)
    T, N = len(d.dates), len(d.instruments)
    du = dataclasses.replace(d, universe=np.ones((T, N), dtype=bool))
    w = target_weights(du, {"leg_n": 10, "lookback_days": 63})
    rows = decision_rows(w)
    assert len(rows) > 20
    gross = np.abs(w[rows]).sum(axis=1)
    assert (gross <= 1 + 1e-9).all()
    assert (gross > 0.99).sum() > 10
