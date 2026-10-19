import numpy as np
import pytest

from strategy import target_weights, PARAMS
from strategy import month_end_mask, week_end_mask, spread_percentile, nyse_holidays, _ymd_to_days, _easter
from lab.framework.data import DataView

INSTR = ("SPY", "IEF", "TLT", "SHY", "AGG")


def D(y, m, d):
    return _ymd_to_days(np.array([y]), np.array([m]), np.array([d]))[0]


def sessions(y0, m0, d0, n):
    """n consecutive weekdays that are not regular NYSE holidays, starting at y0-m0-d0."""
    start = D(y0, m0, d0)
    yrs = np.arange(y0 - 1, y0 + n // 200 + 3)
    hol = nyse_holidays(yrs)
    first = np.busday_offset(start, 0, roll="forward", holidays=hol)
    return np.busday_offset(first, np.arange(n), roll="forward", holidays=hol)


# ------------------------------------------------------------------------------------------- calendar
def test_easter_known_years():
    assert _easter(np.array([2010]))[0] == D(2010, 4, 4)
    assert _easter(np.array([2018]))[0] == D(2018, 4, 1)
    assert _easter(np.array([2024]))[0] == D(2024, 3, 31)


def test_month_end_regular_and_holidays():
    # ordinary month end: Thu 2019-01-31 yes, Wed 2019-01-30 no
    m = month_end_mask(np.array([D(2019, 1, 30), D(2019, 1, 31)]))
    assert list(m) == [False, True]
    # Memorial Day on May 31 2010: Friday May 28 is the last session of May
    assert month_end_mask(np.array([D(2010, 5, 28)]))[0]
    # Good Friday 2018-03-30: Thursday March 29 is the last session of March
    assert month_end_mask(np.array([D(2018, 3, 29)]))[0]
    # Friday 2019-05-24 before Memorial Day (May 27) is not a month end
    assert not month_end_mask(np.array([D(2019, 5, 24)]))[0]


def test_week_end():
    w = week_end_mask(np.array([D(2019, 1, 10), D(2019, 1, 11)]))       # Thu, Fri
    assert list(w) == [False, True]
    assert week_end_mask(np.array([D(2020, 7, 2)]))[0]                  # July 3 2020 observed holiday
    assert not week_end_mask(np.array([D(2015, 11, 25)]))[0]            # Wed before Thanksgiving: Fri open


def test_one_decision_per_month_on_sessions():
    d = sessions(2004, 1, 2, 800)
    m = month_end_mask(d)
    months = d.astype("datetime64[M]")
    # exactly one decision row in every complete month, and it is the last row of that month
    for mo in np.unique(months)[:-1]:
        idx = np.flatnonzero(months == mo)
        assert m[idx].sum() == 1
        assert m[idx[-1]]


def test_weekday_calendar_extra_rows_only_on_holiday_rows():
    # a calendar of all weekdays (like the synthetic market) has rows on holidays; a holiday row that is the
    # last weekday of the month is also flagged, every other month has exactly one decision
    start = D(2003, 10, 1)
    d = np.busday_offset(start, np.arange(4500), roll="forward")
    m = month_end_mask(d)
    hol = set(nyse_holidays(np.arange(2002, 2023)).tolist())
    months = d.astype("datetime64[M]")
    for mo in np.unique(months)[:-1]:
        idx = np.flatnonzero(months == mo)
        flagged = idx[m[idx]]
        if d[idx[-1]].tolist() in hol:
            assert len(flagged) == 2 and flagged[-1] == idx[-1]
        else:
            assert len(flagged) == 1 and flagged[0] == idx[-1]


# ------------------------------------------------------------------------------------------- percentile
def test_percentile_increasing_spread():
    n = 1000
    daaa = np.full(n, 5.0)
    dbaa = 5.0 + 0.01 * np.arange(1, n + 1)          # spread rises every day: latest is the maximum
    p = spread_percentile(daaa, dbaa, np.array([n - 1]), 800)
    assert p[0] == pytest.approx((799 + 0.5) / 800)
    # window covers everything available when window_days > count
    p = spread_percentile(daaa, dbaa, np.array([n - 1]), 1260)
    assert p[0] == pytest.approx((999 + 0.5) / 1000)


def test_percentile_min_obs_and_ties():
    n = 900
    daaa = np.full(n, 4.0)
    dbaa = 4.0 + np.where(np.arange(n) % 2 == 0, 1.0, 2.0)   # spreads alternate 1.0 / 2.0
    dbaa[-1] = 6.5                                           # latest spread 2.5 would be the max ...
    # fewer than 756 valid observations -> missing
    assert np.isnan(spread_percentile(daaa, dbaa, np.array([700]), 1260)[0])
    # row 899: latest spread 2.5, 899 below, 1 equal (itself)
    p = spread_percentile(daaa, dbaa, np.array([n - 1]), 1260)
    assert p[0] == pytest.approx((899 + 0.5) / 900)
    # row 898 (index even -> spread 1.0): 0 below, 449 equal (rows 0,2,...,898) -> 0.5*449/899
    dbaa2 = 4.0 + np.where(np.arange(n) % 2 == 0, 1.0, 2.0)
    p = spread_percentile(daaa, dbaa2, np.array([898]), 1260)
    assert p[0] == pytest.approx(0.5 * 450 / 899)


def test_percentile_float_noise_ties():
    # float32 storage: 6.10 - 5.20 and 7.00 - 6.10 must compare equal
    daaa = np.array([5.20, 6.10] * 400, dtype=np.float32)
    dbaa = np.array([6.10, 7.00] * 400, dtype=np.float32)
    p = spread_percentile(daaa, dbaa, np.array([799]), 1260)
    assert p[0] == pytest.approx(0.5)


def test_stale_and_leading_nan():
    n = 1000
    daaa = np.full(n, 5.0)
    dbaa = 5.0 + 0.01 * np.arange(n) + 1.0
    daaa[:100] = np.nan
    dbaa[:100] = np.nan
    # 900 valid observations by row 999
    assert np.isfinite(spread_percentile(daaa, dbaa, np.array([999]), 1260)[0])
    assert np.isnan(spread_percentile(daaa, dbaa, np.array([50]), 1260)[0])
    # forward-filled stale value: last new value on row 990
    a2, b2 = daaa.copy(), dbaa.copy()
    b2[991:] = b2[990]
    rows = np.array([994, 995])                       # age 5 sessions -> usable, age 6 -> missing
    p = spread_percentile(a2, b2, rows, 1260)
    assert np.isfinite(p[0]) and np.isnan(p[1])
    # NaN-gap representation: same answer
    a3, b3 = daaa.copy(), dbaa.copy()
    a3[991:] = np.nan
    b3[991:] = np.nan
    p = spread_percentile(a3, b3, rows, 1260)
    assert np.isfinite(p[0]) and np.isnan(p[1])


def test_window_param_changes_p():
    n = 2000
    rng = np.random.default_rng(1)
    daaa = np.full(n, 5.0)
    dbaa = 6.0 + np.round(np.cumsum(rng.normal(0, 0.02, n)), 2)
    r = np.array([n - 1])
    p1 = spread_percentile(daaa, dbaa, r, 756)[0]
    p2 = spread_percentile(daaa, dbaa, r, 1764)[0]
    assert np.isfinite(p1) and np.isfinite(p2) and p1 != p2


# ------------------------------------------------------------------------------------------- weights
def make_view(n, daaa, dbaa):
    dates = sessions(2003, 10, 1, n)
    T, N = n, len(INSTR)
    z = np.zeros((T, N))
    return DataView(
        dates=dates,
        instruments=INSTR,
        asset_class=tuple("us_etf" for _ in INSTR),
        ret_co=z.copy(),
        ret_oc=z.copy(),
        tradable=np.ones((T, N), dtype=bool),
        listed=np.ones((T, N), dtype=bool),
        delisting=np.zeros((T, N), dtype=bool),
        close=np.full((T, N), 100.0),
        dollar_volume=np.full((T, N), 1e9),
        universe=None,
        extras={},
        series={"fred_moody_yields_1d.DAAA": np.asarray(daaa, dtype=np.float64),
                "fred_moody_yields_1d.DBAA": np.asarray(dbaa, dtype=np.float64)},
        cash_ret=np.zeros(T),
    )


def test_weights_end_to_end():
    n = 1100
    daaa = np.full(n, 5.0)
    dbaa = 5.0 + 1.0 + 0.001 * np.arange(n)            # rising spread: p near 1 once 756 obs exist
    data = make_view(n, daaa, dbaa)
    w = target_weights(data, PARAMS)
    assert w.shape == (n, 5)
    dec = month_end_mask(data.dates)
    assert np.all(np.isnan(w[~dec]))                   # only month-end rows decide
    assert np.all(np.isfinite(w[dec]))
    assert np.allclose(w[dec].sum(axis=1), 1.0)
    assert np.all(w[dec] >= 0) and np.all(np.abs(w[dec]).sum(axis=1) <= 1 + 1e-12)
    assert np.all(w[dec][:, 2:] == 0)                  # TLT, SHY, AGG
    rows = np.flatnonzero(dec)
    early = rows[rows < 755]
    late = rows[rows >= 755]
    assert np.allclose(w[early, 0], 0.6) and np.allclose(w[early, 1], 0.4)
    t = late[-1]
    k = t + 1
    nobs = min(1260, k)
    p = (nobs - 1 + 0.5) / nobs
    assert w[t, 0] == pytest.approx(0.6 + 0.2 * (2 * p - 1))
    assert w[t, 1] == pytest.approx(1 - w[t, 0])
    # tilt parameter scales the deviation from 60 %
    w3 = target_weights(data, {"tilt": 0.3, "window_days": 1260})
    assert (w3[t, 0] - 0.6) == pytest.approx(1.5 * (w[t, 0] - 0.6))


def test_missing_signal_is_plain_6040_and_missing_series_raises():
    n = 1100
    daaa = np.full(n, np.nan)
    dbaa = np.full(n, np.nan)
    data = make_view(n, daaa, dbaa)
    w = target_weights(data, PARAMS)
    dec = month_end_mask(data.dates)
    assert np.allclose(w[dec, 0], 0.6) and np.allclose(w[dec, 1], 0.4)
    bad = make_view(n, daaa, dbaa)
    bad.series.pop("fred_moody_yields_1d.DBAA")
    with pytest.raises(KeyError):
        target_weights(bad, PARAMS)


def test_point_in_time_truncation():
    n = 1300
    rng = np.random.default_rng(3)
    daaa = 5.0 + np.round(np.cumsum(rng.normal(0, 0.03, n)), 2)
    dbaa = daaa + 1.0 + np.round(np.cumsum(rng.normal(0, 0.02, n)), 2)
    full = target_weights(make_view(n, daaa, dbaa), PARAMS)
    for cut in (800, 1000, 1250):
        part = target_weights(make_view(cut, daaa[:cut], dbaa[:cut]), PARAMS)
        assert np.array_equal(np.isnan(part), np.isnan(full[:cut]))
        assert np.allclose(np.nan_to_num(part), np.nan_to_num(full[:cut]))
