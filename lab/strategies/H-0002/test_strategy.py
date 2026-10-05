import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS
from strategy import _nyse_holidays, _remaining_trading_days


def ymd(y, m, d):
    first = np.array((y - 1970) * 12 + (m - 1)).astype("datetime64[M]").astype("datetime64[D]")
    return (first.astype(np.int64) + (d - 1)).astype("datetime64[D]")


def trading_days(start, end, holidays):
    days = np.arange(start.astype(np.int64), end.astype(np.int64) + 1).astype("datetime64[D]")
    return days[np.is_busday(days, holidays=holidays)]


# Instruments deliberately in non-alphabetical order: columns must be found by name.
INSTR = ("IEF", "SPY")
I_IEF, I_SPY = 0, 1


def make_view(dates, ret_co=None, ret_oc=None, tradable=None, listed=None):
    T, N = len(dates), len(INSTR)
    z = np.zeros((T, N))
    ret_co = z.copy() if ret_co is None else ret_co
    ret_oc = z.copy() if ret_oc is None else ret_oc
    tradable = np.ones((T, N), dtype=bool) if tradable is None else tradable
    listed = np.ones((T, N), dtype=bool) if listed is None else listed
    return DataView(
        dates=dates,
        instruments=INSTR,
        asset_class=("us_etf",) * N,
        ret_co=ret_co,
        ret_oc=ret_oc,
        tradable=tradable,
        listed=listed,
        delisting=np.zeros((T, N), dtype=bool),
        close=np.where(listed, 100.0, np.nan),
        dollar_volume=np.where(listed, 1e9, np.nan),
        universe=None,
        extras={},
        series={},
        cash_ret=None,
    )


# Apr-Jun 2019 XNYS: Good Friday 19 Apr, Memorial Day 27 May.
HOL_2019 = [ymd(2019, 4, 19), ymd(2019, 5, 27)]
DATES = trading_days(ymd(2019, 3, 29), ymd(2019, 6, 28), HOL_2019)


def row(dates, y, m, d):
    idx = np.flatnonzero(dates == ymd(y, m, d))
    assert idx.size == 1
    return int(idx[0])


def base_returns():
    T = len(DATES)
    co = np.zeros((T, 2))
    oc = np.zeros((T, 2))
    # April: SPY underperforms -> D < 0 -> long SPY / short IEF.
    co[row(DATES, 2019, 4, 2), I_SPY] = -0.01
    # May: SPY outperforms -> D > 0 -> short SPY / long IEF.
    oc[row(DATES, 2019, 5, 1), I_SPY] = 0.01
    # June: no drift -> D == 0 -> flat.
    return co, oc


def test_params_names():
    assert PARAMS == {"window_days": 5}


def test_holiday_rules():
    expected_2019 = [ymd(2019, 1, 1), ymd(2019, 1, 21), ymd(2019, 2, 18), ymd(2019, 4, 19), ymd(2019, 5, 27),
                     ymd(2019, 7, 4), ymd(2019, 9, 2), ymd(2019, 11, 28), ymd(2019, 12, 25)]
    assert list(_nyse_holidays(np.array([2019]))) == expected_2019
    h21 = list(_nyse_holidays(np.array([2021])))
    assert ymd(2021, 7, 5) in h21 and ymd(2021, 12, 24) in h21  # Sunday / Saturday observed
    h22 = list(_nyse_holidays(np.array([2022])))
    assert ymd(2022, 1, 1) not in h22 and ymd(2021, 12, 31) not in h22  # Saturday New Year: no holiday
    h08 = list(_nyse_holidays(np.array([2008])))
    assert ymd(2008, 3, 21) in h08  # Good Friday, Easter 23 March 2008


def test_remaining_days_skip_holidays():
    rem, _ = _remaining_trading_days(DATES)
    assert rem[row(DATES, 2019, 5, 31)] == 0
    assert rem[row(DATES, 2019, 5, 28)] == 3
    assert rem[row(DATES, 2019, 5, 24)] == 4  # Memorial Day 27 May not counted
    assert rem[row(DATES, 2019, 5, 23)] == 5


def expected_month(W, dec, window, last, w_spy, w_ief):
    assert W[dec, I_SPY] == w_spy and W[dec, I_IEF] == w_ief
    for r in window:
        assert np.isnan(W[r]).all()
    assert (W[last] == 0).all()


def test_weights_primary():
    co, oc = base_returns()
    W = target_weights(make_view(DATES, co, oc), PARAMS)
    assert W.shape == (len(DATES), 2)
    d = lambda m, dd: row(DATES, 2019, m, dd)
    # April: last 5 trading days 24,25,26,29,30 -> decide 23 Apr.
    expected_month(W, d(4, 23), [d(4, 24), d(4, 25), d(4, 26), d(4, 29)], d(4, 30), 0.5, -0.5)
    # May: last 5 trading days 24,28,29,30,31 (27 May holiday) -> decide 23 May.
    expected_month(W, d(5, 23), [d(5, 24), d(5, 28), d(5, 29), d(5, 30)], d(5, 31), -0.5, 0.5)
    # June: D == 0 -> flat decision on 21 Jun.
    expected_month(W, d(6, 21), [d(6, 24), d(6, 25), d(6, 26), d(6, 27)], d(6, 28), 0.0, 0.0)
    # Flat on every row outside the windows, NaN only inside them.
    nan_rows = np.isnan(W).any(axis=1)
    assert nan_rows.sum() == 12
    assert (W[~nan_rows][:, I_SPY] != 0).sum() == 2
    assert np.all(np.abs(W[~nan_rows]).sum(axis=1) <= 1.0 + 1e-12)
    assert np.all(W[~nan_rows].sum(axis=1) == 0)  # dollar-neutral


def test_window_days_param_changes_decision_day():
    co, oc = base_returns()
    W = target_weights(make_view(DATES, co, oc), {"window_days": 3})
    d = lambda m, dd: row(DATES, 2019, m, dd)
    # May, K=3: window 29,30,31 -> decision 28 May.
    expected_month(W, d(5, 28), [d(5, 29), d(5, 30)], d(5, 31), -0.5, 0.5)
    assert (W[d(5, 23)] == 0).all()
    W7 = target_weights(make_view(DATES, co, oc), {"window_days": 7})
    # May, K=7: window 22,23,24,28,29,30,31 -> decision 21 May.
    assert W7[d(5, 21), I_SPY] == -0.5


def test_compounding():
    T = len(DATES)
    co = np.zeros((T, 2))
    oc = np.zeros((T, 2))
    # SPY +10% then -10% = -1% compounded (0 if summed); IEF -0.5%: D = -0.5% < 0 -> long SPY.
    oc[row(DATES, 2019, 5, 2), I_SPY] = 0.10
    oc[row(DATES, 2019, 5, 3), I_SPY] = -0.10
    oc[row(DATES, 2019, 5, 6), I_IEF] = -0.005
    W = target_weights(make_view(DATES, co, oc), PARAMS)
    assert W[row(DATES, 2019, 5, 23), I_SPY] == 0.5
    # Overnight return of the first day of the month counts (close of 30 Apr -> open of 1 May).
    co2 = np.zeros((T, 2))
    co2[row(DATES, 2019, 5, 1), I_IEF] = 0.02
    W2 = target_weights(make_view(DATES, co2, np.zeros((T, 2))), PARAMS)
    assert W2[row(DATES, 2019, 5, 23), I_SPY] == 0.5
    # Returns after the decision day do not count.
    co3 = np.zeros((T, 2))
    co3[row(DATES, 2019, 5, 24), I_IEF] = 0.02
    W3 = target_weights(make_view(DATES, co3, np.zeros((T, 2))), PARAMS)
    assert (W3[row(DATES, 2019, 5, 23)] == 0).all()


def test_missing_return_makes_month_flat():
    co, oc = base_returns()
    tr = np.ones((len(DATES), 2), dtype=bool)
    tr[row(DATES, 2019, 5, 10), I_IEF] = False
    W = target_weights(make_view(DATES, co, oc, tradable=tr), PARAMS)
    assert (W[row(DATES, 2019, 5, 23)] == 0).all()
    assert W[row(DATES, 2019, 4, 23), I_SPY] == 0.5  # other months unaffected
    # Missing anchor close (last day of April) also makes May flat.
    tr2 = np.ones((len(DATES), 2), dtype=bool)
    tr2[row(DATES, 2019, 4, 30), I_SPY] = False
    W2 = target_weights(make_view(DATES, co, oc, tradable=tr2), PARAMS)
    assert (W2[row(DATES, 2019, 5, 23)] == 0).all()
    # NaN return is missing too.
    oc3 = oc.copy()
    oc3[row(DATES, 2019, 5, 15), I_SPY] = np.nan
    W3 = target_weights(make_view(DATES, co, oc3), PARAMS)
    assert (W3[row(DATES, 2019, 5, 23)] == 0).all()


def test_entry_day_not_tradable_goes_flat():
    co, oc = base_returns()
    tr = np.ones((len(DATES), 2), dtype=bool)
    tr[row(DATES, 2019, 5, 24), I_IEF] = False
    W = target_weights(make_view(DATES, co, oc, tradable=tr), PARAMS)
    assert W[row(DATES, 2019, 5, 23), I_SPY] == -0.5  # decided before the entry day is known
    assert (W[row(DATES, 2019, 5, 24)] == 0).all()  # partial fill closed at the next open
    assert np.isnan(W[row(DATES, 2019, 5, 28)]).all()  # stays flat (holds 0) for the month


def test_late_listing():
    co, oc = base_returns()
    T = len(DATES)
    listed = np.ones((T, 2), dtype=bool)
    first = row(DATES, 2019, 4, 10)
    listed[:first, I_SPY] = False
    tr = listed.copy()
    co[:first, I_SPY] = 0.0
    W = target_weights(make_view(DATES, co, oc, tradable=tr, listed=listed), PARAMS)
    assert (W[row(DATES, 2019, 4, 23)] == 0).all()  # April incomplete -> flat
    assert W[row(DATES, 2019, 5, 23), I_SPY] == -0.5  # May complete


def test_first_month_without_anchor_is_flat():
    co, oc = base_returns()
    start = row(DATES, 2019, 4, 1)
    dv = make_view(DATES[start:], co[start:], oc[start:])
    W = target_weights(dv, PARAMS)
    assert (W[row(DATES[start:], 2019, 4, 23)] == 0).all()


def test_point_in_time_truncation_and_determinism():
    rng = np.random.default_rng(7)
    dates = trading_days(ymd(2017, 1, 3), ymd(2019, 12, 31), [])
    T = len(dates)
    co = rng.normal(0, 0.005, (T, 2))
    oc = rng.normal(0, 0.01, (T, 2))
    full = target_weights(make_view(dates, co, oc), PARAMS)
    again = target_weights(make_view(dates, co, oc), PARAMS)
    assert np.array_equal(full, again, equal_nan=True)
    for cut in [30, 100, 257, 400, 600]:
        part = target_weights(make_view(dates[:cut], co[:cut], oc[:cut]), PARAMS)
        assert np.array_equal(part, full[:cut], equal_nan=True)
    ok = ~np.isnan(full).any(axis=1)
    assert np.all(np.abs(full[ok]).sum(axis=1) <= 1.0 + 1e-12)
    assert np.all(np.isin(np.abs(full[ok]), [0.0, 0.5]))
    # Roughly one entry per month.
    n_dec = int((np.abs(full[ok]).sum(axis=1) == 1.0).sum())
    assert 30 <= n_dec <= 36
