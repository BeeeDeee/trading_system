"""Multi-timeframe helpers: correct values and point-in-time (truncating the future changes nothing)."""

import numpy as np

from lab.framework import api

DAYS = np.arange(np.datetime64("2020-01-01"), np.datetime64("2020-06-30"))
DAYS = DAYS[np.is_busday(DAYS)]


def test_completed_month_value():
    x = np.arange(len(DAYS), dtype=float)
    v = api.completed_period_value(x, DAYS, "M")
    feb = np.flatnonzero(DAYS.astype("datetime64[M]") == np.datetime64("2020-02"))
    last_jan = np.flatnonzero(DAYS.astype("datetime64[M]") == np.datetime64("2020-01"))[-1]
    assert np.isnan(v[:feb[0]]).all() and (v[feb] == x[last_jan]).all()


def test_period_return_matches_month_ends():
    rng = np.random.default_rng(0)
    idx = np.cumprod(1 + rng.normal(0, 0.01, (len(DAYS), 2)), axis=0)
    r = api.period_return(idx, DAYS, "M", 2)
    t = np.flatnonzero(DAYS.astype("datetime64[M]") == np.datetime64("2020-05"))[3]
    m = DAYS.astype("datetime64[M]")
    end_apr = np.flatnonzero(m == np.datetime64("2020-04"))[-1]
    end_feb = np.flatnonzero(m == np.datetime64("2020-02"))[-1]
    assert np.allclose(r[t], idx[end_apr] / idx[end_feb] - 1)


def test_helpers_are_point_in_time():
    rng = np.random.default_rng(1)
    x = np.cumprod(1 + rng.normal(0, 0.01, (len(DAYS), 3)), axis=0)
    for f in (lambda d, a: api.completed_period_value(a, d, "W"), lambda d, a: api.period_return(a, d, "M", 1),
              lambda d, a: api.period_return(a, d, "Q", 1), lambda d, a: api.ema(a, 10)):
        full = f(DAYS, x)
        for cut in (5, 40, 77, len(DAYS) - 1):
            part = f(DAYS[:cut + 1], x[:cut + 1])
            assert np.allclose(part, full[:cut + 1], equal_nan=True)
