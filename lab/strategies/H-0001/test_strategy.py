import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import PARAMS, target_weights, easter_sunday, nyse_holidays

EPOCH = 1970


def _d(y, m, d):
    base = np.array([y - EPOCH], dtype=np.int64).astype("datetime64[Y]").astype("datetime64[M]")
    mon = base + np.array([m - 1], dtype=np.int64).astype("timedelta64[M]")
    return (mon.astype("datetime64[D]") + np.array([d - 1], dtype=np.int64).astype("timedelta64[D]"))[0]


def _trading_days(start, end, holidays):
    days = np.arange(start, end + np.timedelta64(1, "D"))
    wd = (days.astype(np.int64) + 3) % 7
    hol = np.array(holidays, dtype="datetime64[D]")
    return days[(wd < 5) & ~np.isin(days, hol)]


def _view(dates, close=None, tradable=None):
    T = len(dates)
    names = ("SPY", "IEF")
    if close is None:
        close = np.full((T, 2), 100.0)
    if tradable is None:
        tradable = np.isfinite(close)
    return DataView(
        dates=dates,
        instruments=names,
        asset_class=("us_etf", "us_etf"),
        ret_co=np.zeros((T, 2)),
        ret_oc=np.zeros((T, 2)),
        tradable=np.asarray(tradable, dtype=bool),
        listed=np.isfinite(close),
        delisting=np.zeros((T, 2), dtype=bool),
        close=close,
        dollar_volume=np.where(np.isfinite(close), 1e9, np.nan),
        universe=None,
        extras={},
        series={},
        cash_ret=np.zeros(T),
    )


# Oct 2019 .. Jan 2020 on the real XNYS calendar (hand-listed holidays)
HOL_A = [_d(2019, 11, 28), _d(2019, 12, 25), _d(2020, 1, 1), _d(2020, 1, 20)]
DATES_A = _trading_days(_d(2019, 10, 1), _d(2020, 1, 31), HOL_A)
# Feb .. Apr 2018: Good Friday is Mar 30 2018, the last weekday of March
HOL_B = [_d(2018, 2, 19), _d(2018, 3, 30)]
DATES_B = _trading_days(_d(2018, 2, 1), _d(2018, 4, 30), HOL_B)

# rows whose target (for the NEXT trading day) is SPY, primary k=2, m=3, derived by hand
SPY_A = [
    (2019, 10, 31), (2019, 11, 1), (2019, 11, 4),                     # +1 +2 +3 of Nov
    (2019, 11, 26), (2019, 11, 27),                                   # -2 Nov 27, -1 Nov 29 (Thanksgiving 28)
    (2019, 11, 29), (2019, 12, 2), (2019, 12, 3),                     # +1..+3 Dec 2,3,4
    (2019, 12, 27), (2019, 12, 30),                                   # -2 Dec 30, -1 Dec 31
    (2019, 12, 31), (2020, 1, 2), (2020, 1, 3),                       # +1..+3 Jan 2,3,6 (Jan 1 holiday)
    (2020, 1, 29), (2020, 1, 30), (2020, 1, 31),                      # -2 Jan 30, -1 Jan 31, +1 Feb 3
]


def _rows(dates, ymds):
    return np.array([int(np.nonzero(dates == _d(*x))[0][0]) for x in ymds])


def _expected(dates, spy_rows, first_active):
    T = len(dates)
    e = np.zeros((T, 2))
    e[first_active:, 1] = 1.0
    e[spy_rows, 0] = 1.0
    e[spy_rows, 1] = 0.0
    return e


def test_params_match_card():
    assert PARAMS == {"days_before": 2, "days_after": 3}


def test_easter_and_holidays():
    years = np.array([2000, 2008, 2011, 2018, 2019, 2024])
    want = [_d(2000, 4, 23), _d(2008, 3, 23), _d(2011, 4, 24), _d(2018, 4, 1), _d(2019, 4, 21), _d(2024, 3, 31)]
    assert list(easter_sunday(years)) == want
    h = set(nyse_holidays(2021, 2022).tolist())
    assert _d(2021, 12, 24).tolist() in h          # Christmas on Saturday -> Friday
    assert _d(2021, 7, 5).tolist() in h            # July 4 on Sunday -> Monday
    assert _d(2021, 12, 31).tolist() not in h      # New Year 2022 on Saturday is not observed
    assert _d(2021, 5, 31).tolist() in h           # Memorial day = last Monday of May
    assert _d(2021, 4, 2).tolist() in h            # Good Friday
    assert _d(2021, 11, 25).tolist() in h          # Thanksgiving
    assert _d(2021, 9, 6).tolist() in h            # Labor day


def test_primary_weights_by_hand():
    w = target_weights(_view(DATES_A), PARAMS)
    first_active = int(np.nonzero(DATES_A == _d(2019, 10, 31))[0][0])
    np.testing.assert_array_equal(w, _expected(DATES_A, _rows(DATES_A, SPY_A), first_active))


def test_good_friday_month_end():
    w = target_weights(_view(DATES_B), PARAMS)
    spy = _rows(DATES_B, [(2018, 2, 26), (2018, 2, 27), (2018, 2, 28), (2018, 3, 1), (2018, 3, 2),
                          (2018, 3, 27), (2018, 3, 28), (2018, 3, 29), (2018, 4, 2), (2018, 4, 3),
                          (2018, 4, 26), (2018, 4, 27), (2018, 4, 30)])
    first_active = int(np.nonzero(DATES_B == _d(2018, 2, 28))[0][0])
    e = _expected(DATES_B, spy, first_active)
    # Feb rows before the first active row hold nothing
    e[:first_active] = 0.0
    np.testing.assert_array_equal(w, e)


@pytest.mark.parametrize("name,value,added,removed", [
    ("days_before", 3, [(2019, 11, 25), (2019, 12, 26), (2020, 1, 28)], []),
    ("days_before", 1, [], [(2019, 11, 26), (2019, 12, 27), (2020, 1, 29)]),
    ("days_after", 4, [(2019, 11, 5), (2019, 12, 4), (2020, 1, 6)], []),
    ("days_after", 2, [], [(2019, 11, 4), (2019, 12, 3), (2020, 1, 3)]),
])
def test_param_changes(name, value, added, removed):
    p = dict(PARAMS)
    p[name] = value
    w = target_weights(_view(DATES_A), p)
    want = set(SPY_A) | set(added)
    want -= set(removed)
    first_active = int(np.nonzero(DATES_A == _d(2019, 10, 31))[0][0])
    np.testing.assert_array_equal(w, _expected(DATES_A, _rows(DATES_A, sorted(want)), first_active))
    assert not np.array_equal(w, target_weights(_view(DATES_A), PARAMS))


def test_entry_delayed_when_not_tradable():
    T = len(DATES_A)
    trad = np.ones((T, 2), dtype=bool)
    r = _rows(DATES_A, [(2019, 11, 26)])[0]
    trad[r, 1] = False
    w = target_weights(_view(DATES_A, tradable=trad), PARAMS)
    assert w[r].tolist() == [0.0, 1.0]           # keep IEF
    assert w[r + 1].tolist() == [1.0, 0.0]       # switch at next open where both trade


def test_exit_delayed_when_not_tradable():
    T = len(DATES_A)
    trad = np.ones((T, 2), dtype=bool)
    r = _rows(DATES_A, [(2019, 12, 4)])[0]
    trad[r, 0] = False
    w = target_weights(_view(DATES_A, tradable=trad), PARAMS)
    assert w[r].tolist() == [1.0, 0.0]           # keep SPY
    assert w[r + 1].tolist() == [0.0, 1.0]


def test_window_skipped_if_ended():
    T = len(DATES_A)
    trad = np.ones((T, 2), dtype=bool)
    a, b = _rows(DATES_A, [(2019, 11, 26), (2019, 12, 3)])
    trad[a:b + 1, 0] = False
    w = target_weights(_view(DATES_A, tradable=trad), PARAMS)
    assert np.all(w[a:b + 2, 1] == 1.0) and np.all(w[a:b + 2, 0] == 0.0)


def test_late_listing_starts_next_full_month():
    T = len(DATES_A)
    close = np.full((T, 2), 100.0)
    r_list = _rows(DATES_A, [(2019, 11, 13)])[0]
    close[:r_list, 1] = np.nan
    w = target_weights(_view(DATES_A, close=close), PARAMS)
    r_start = _rows(DATES_A, [(2019, 11, 29)])[0]  # decision for Dec 2, first day of the first full month
    assert np.all(w[:r_start] == 0.0)
    assert w[r_start].tolist() == [1.0, 0.0]
    assert np.all(w[r_start:].sum(axis=1) == 1.0)


def test_no_prices_holds_nothing():
    close = np.full((len(DATES_A), 2), 100.0)
    close[:, 1] = np.nan
    w = target_weights(_view(DATES_A, close=close), PARAMS)
    assert np.all(w == 0.0)


def test_gross_long_only_and_point_in_time():
    v = _view(DATES_A)
    w = target_weights(v, PARAMS)
    assert np.all(w >= 0.0) and np.all(w.sum(axis=1) <= 1.0 + 1e-12)
    for cut in (25, 40, 61, 70):
        wt = target_weights(_view(DATES_A[:cut]), PARAMS)
        np.testing.assert_array_equal(wt, w[:cut])
