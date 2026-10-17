import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import (target_weights, PARAMS, days_from_civil, civil_from_days, weekday,
                      sessions_remaining_in_month, xnys_holidays)

INSTR = ("SPY", "IEF", "TLT", "SHY", "LQD", "AGG", "TIP")
DT = np.dtype("datetime64[D]")


def d(y, m, dd):
    return int(days_from_civil(y, m, dd))


def as_dates(ints):
    return np.asarray(ints, dtype=np.int64).astype(DT)


def sessions(y0, m0, d0, y1, m1, d1):
    """Rule-based XNYS sessions between two dates (inclusive) as int days."""
    g = np.arange(d(y0, m0, d0), d(y1, m1, d1) + 1)
    gy, _, _ = civil_from_days(g)
    hol = xnys_holidays(np.arange(gy.min(), gy.max() + 1))
    return g[(weekday(g) < 5) & ~np.isin(g, hol)]


def make_view(day_ints, tradable=None):
    T, N = len(day_ints), len(INSTR)
    if tradable is None:
        tradable = np.ones((T, N), dtype=bool)
    zeros = np.zeros((T, N))
    return DataView(
        dates=as_dates(day_ints),
        instruments=INSTR,
        asset_class=tuple("us_etf" for _ in INSTR),
        ret_co=zeros.copy(),
        ret_oc=zeros.copy(),
        tradable=tradable,
        listed=np.ones((T, N), dtype=bool),
        delisting=np.zeros((T, N), dtype=bool),
        close=np.full((T, N), 100.0),
        dollar_volume=np.full((T, N), 1e9),
        universe=None,
        extras={},
        series={},
        cash_ret=np.zeros(T),
    )


def idx(day_ints, y, m, dd):
    return int(np.where(day_ints == d(y, m, dd))[0][0])


# ------------------------------------------------------------------ calendar
def test_civil_roundtrip():
    assert d(1970, 1, 1) == 0
    assert d(2000, 1, 1) == 10957
    assert int(weekday(d(2019, 11, 28))) == 3  # Thanksgiving 2019 is a Thursday
    z = np.arange(d(1999, 1, 1), d(2030, 12, 31))
    y, m, dd = civil_from_days(z)
    assert np.array_equal(days_from_civil(y, m, dd), z)


@pytest.mark.parametrize("year,n", [(2016, 252), (2017, 251), (2019, 252), (2022, 251)])
def test_sessions_per_year(year, n):
    assert len(sessions(year, 1, 1, year, 12, 31)) == n


def test_known_holidays():
    hol = set(xnys_holidays(np.array([2013, 2004, 2010, 2017, 2022, 2021])).tolist())
    assert d(2013, 3, 29) in hol       # Good Friday
    assert d(2004, 5, 31) in hol       # Memorial Day
    assert d(2010, 12, 24) in hol      # Christmas observed (Sat)
    assert d(2010, 12, 31) not in hol  # New Year on Saturday is not observed
    assert d(2017, 1, 2) in hol        # New Year observed (Sun)
    assert d(2022, 6, 20) in hol       # Juneteenth observed
    assert d(2021, 6, 18) not in hol   # no Juneteenth before 2022
    assert d(2019, 11, 29) not in hol  # day after Thanksgiving trades


def test_rem_values():
    pts = [(2013, 3, 27, 1), (2013, 3, 28, 0), (2004, 5, 27, 1), (2004, 5, 28, 0),
           (2019, 11, 26, 2), (2019, 11, 27, 1), (2010, 12, 30, 1), (2010, 12, 31, 0),
           (2019, 10, 1, 22), (2019, 10, 29, 2)]
    z = np.array([d(y, m, dd) for y, m, dd, _ in pts])
    rem = sessions_remaining_in_month(as_dates(z))
    assert rem.tolist() == [r for *_, r in pts]


def test_rem_independent_of_rows():
    # rem of a row must not depend on which other rows exist (truncation / gaps)
    z = sessions(2019, 10, 1, 2019, 12, 31)
    full = sessions_remaining_in_month(as_dates(z))
    for cut in (5, 20, 40, len(z) - 1):
        assert np.array_equal(sessions_remaining_in_month(as_dates(z[:cut])), full[:cut])
    sparse = z[::3]
    assert np.array_equal(sessions_remaining_in_month(as_dates(sparse)), full[::3])


# ------------------------------------------------------------------ weights
BASE = np.array([0.6, 0.4, 0, 0, 0, 0, 0])
TILT = np.array([0.6, 0, 0.4, 0, 0, 0, 0])


def test_weights_primary():
    z = sessions(2019, 10, 1, 2019, 12, 31)
    w = target_weights(make_view(z), PARAMS)
    assert w.shape == (len(z), len(INSTR))
    decided = ~np.isnan(w).all(axis=1)
    entries = [idx(z, 2019, 10, 29), idx(z, 2019, 11, 26), idx(z, 2019, 12, 27)]
    exits = [idx(z, 2019, 10, 31), idx(z, 2019, 11, 29), idx(z, 2019, 12, 31)]
    assert sorted(np.where(decided)[0].tolist()) == sorted([0] + entries + exits)
    assert np.allclose(w[0], BASE)
    for r in entries:
        assert np.allclose(w[r], TILT)
    for r in exits:
        assert np.allclose(w[r], BASE)
    dec = w[decided]
    assert np.all(dec >= 0)
    assert np.all(np.abs(dec).sum(axis=1) <= 1.0 + 1e-12)
    assert np.allclose(dec.sum(axis=1), 1.0)
    # peers never traded
    assert np.all(dec[:, 3:] == 0)


@pytest.mark.parametrize("window,entry_day", [(1, 30), (2, 29), (3, 28)])
def test_window_param(window, entry_day):
    z = sessions(2019, 10, 1, 2019, 10, 31)
    w = target_weights(make_view(z), {"window_sessions": window})
    r = idx(z, 2019, 10, entry_day)
    assert np.allclose(w[r], TILT)
    decided = np.where(~np.isnan(w).all(axis=1))[0].tolist()
    assert decided == [0, r, idx(z, 2019, 10, 31)]


def test_tlt_no_bar_skips_entry():
    z = sessions(2019, 10, 1, 2019, 10, 31)
    tr = np.ones((len(z), len(INSTR)), dtype=bool)
    r = idx(z, 2019, 10, 29)
    tr[r, INSTR.index("TLT")] = False
    w = target_weights(make_view(z, tr), PARAMS)
    assert np.isnan(w[r]).all()
    assert np.allclose(w[idx(z, 2019, 10, 31)], BASE)


def test_point_in_time_truncation():
    z = sessions(2019, 1, 1, 2019, 12, 31)
    full = target_weights(make_view(z), PARAMS)
    for cut in (10, 57, 100, 200):
        part = target_weights(make_view(z[:cut]), PARAMS)
        assert np.array_equal(np.isnan(part), np.isnan(full[:cut]))
        assert np.allclose(np.nan_to_num(part), np.nan_to_num(full[:cut]))


def test_missing_instrument_raises():
    z = sessions(2019, 10, 1, 2019, 10, 31)
    v = make_view(z)
    v2 = DataView(dates=v.dates, instruments=("SPY", "IEF", "SHY", "LQD", "AGG", "TIP", "X"),
                  asset_class=v.asset_class, ret_co=v.ret_co, ret_oc=v.ret_oc,
                  tradable=v.tradable, listed=v.listed, delisting=v.delisting, close=v.close,
                  dollar_volume=v.dollar_volume, universe=None, extras={}, series={},
                  cash_ret=v.cash_ret)
    with pytest.raises(Exception):
        target_weights(v2, PARAMS)
