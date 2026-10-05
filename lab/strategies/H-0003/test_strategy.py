import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS

MONDAY_2024_01_01 = 19723  # days since the epoch; (19723 + 3) % 7 == 0 -> Monday


def weekdays(n_weeks):
    days = []
    for k in range(n_weeks):
        for d in range(5):
            days.append(MONDAY_2024_01_01 + 7 * k + d)
    return np.array(days, dtype=np.int64).astype("datetime64[D]")


# 3 weeks, rows 4, 9, 14 are Fridays. Row 0 is SPY's first bar (no previous close -> no return).
R_SPY = np.array([0.0, 0.05, -0.05, 0.01, -0.01,
                  0.01, -0.01, 0.01, 0.05, -0.05,
                  0.0, 0.0, 0.0, 0.0, 0.0])
LOGR = np.log1p(R_SPY)


def make_view(r_spy=R_SPY, spy_close=None, ief_close=None, tradable=None, dates=None):
    T = len(r_spy)
    if dates is None:
        dates = weekdays(3)[:T]
    close = np.full((T, 2), 100.0)
    if spy_close is not None:
        close[:, 0] = spy_close
    if ief_close is not None:
        close[:, 1] = ief_close
    listed = np.isfinite(close)
    if tradable is None:
        tradable = listed.copy()
    ret_oc = np.zeros((T, 2))
    ret_oc[:, 0] = np.where(np.isfinite(close[:, 0]), r_spy, 0.0)
    return DataView(
        dates=dates,
        instruments=("SPY", "IEF"),
        asset_class=("us_etf", "us_etf"),
        ret_co=np.zeros((T, 2)),
        ret_oc=ret_oc,
        tradable=tradable,
        listed=listed,
        delisting=np.zeros((T, 2), dtype=bool),
        close=close,
        dollar_volume=np.where(listed, 1e9, np.nan),
        universe=None,
        extras={},
        series={},
        cash_ret=np.zeros(T),
    )


P = {"short_window_days": 2, "long_window_days": 4}


def decided_rows(w):
    return list(np.flatnonzero(~np.isnan(w).all(axis=1)))


def test_params_match_card():
    assert PARAMS == {"short_window_days": 21, "long_window_days": 252}


def test_decision_days_are_fridays():
    w = target_weights(make_view(r_spy=np.zeros(15)), P)
    assert decided_rows(w) == [4, 9, 14]


def test_holiday_friday_decides_on_next_row():
    d = np.delete(weekdays(3), 4)          # week 1 has no Friday row (exchange holiday)
    w = target_weights(make_view(r_spy=np.zeros(14), dates=d), P)
    # row 3 = Thursday wk1 is not knowable as the week's last day; row 4 = Monday wk2 is the late decision
    # for week 1, row 8 = Friday wk2, row 13 = Friday wk3
    assert decided_rows(w) == [4, 8, 13]


def test_weights_on_known_rows():
    w = target_weights(make_view(), P)
    assert w.shape == (15, 2)
    # row 4: short = std(log 1.01, log .99) is small vs long incl. +-5 % -> calm -> SPY
    assert list(w[4]) == [1.0, 0.0]
    # row 9: short = std(log 1.05, log .95) > long -> stressed -> IEF
    assert list(w[9]) == [0.0, 1.0]
    # row 14: all zero returns -> sigma_short == sigma_long == 0 -> calm (<=) -> SPY
    assert list(w[14]) == [1.0, 0.0]
    other = np.setdiff1d(np.arange(15), [4, 9, 14])
    assert np.isnan(w[other]).all()


def test_ratio_consistency_with_direct_std():
    w = target_weights(make_view(), P)
    for t in (4, 9, 14):
        ss = np.std(LOGR[t - 1:t + 1], ddof=1)
        sl = np.std(LOGR[t - 3:t + 1], ddof=1)
        assert (w[t, 0] == 1.0) == (ss <= sl)


def test_no_decision_before_long_window_full():
    # at row 4 there are only 4 valid returns (rows 1-4; row 0 has no previous close)
    w = target_weights(make_view(), {"short_window_days": 2, "long_window_days": 5})
    assert np.isnan(w[4]).all()
    assert not np.isnan(w[9]).any()


def test_no_decision_before_ief_priced():
    ief = np.full(15, 100.0)
    ief[:7] = np.nan                        # IEF listed from row 7
    w = target_weights(make_view(ief_close=ief), P)
    assert np.isnan(w[:9]).all()
    assert list(w[9]) == [0.0, 1.0]


def test_missing_spy_bar_extends_window():
    r = R_SPY.copy()
    r[6] = 0.2                              # would dominate the window if row 6 were used
    spy = np.full(15, 100.0)
    spy[6] = np.nan                         # vendor-error day dropped -> no bar, skipped
    w = target_weights(make_view(r_spy=r, spy_close=spy), P)
    lr = np.log1p(r)
    ss = np.std(lr[[8, 9]], ddof=1)
    sl = np.std(lr[[5, 7, 8, 9]], ddof=1)  # window extends back to row 5
    assert ss > sl
    assert list(w[9]) == [0.0, 1.0]
    sl_wrong = np.std(lr[[6, 7, 8, 9]], ddof=1)
    assert ss <= sl_wrong                   # using the dropped row would give the opposite state


def test_blocked_switch_is_reemitted_until_both_tradable():
    trad = np.ones((15, 2), dtype=bool)
    trad[10, 1] = False                     # IEF not tradable at the open after the row-9 switch
    trad[11, 1] = False
    w = target_weights(make_view(tradable=trad), P)
    assert list(w[10]) == [0.0, 1.0]
    assert list(w[11]) == [0.0, 1.0]
    assert np.isnan(w[12]).all()
    w0 = target_weights(make_view(), P)
    assert np.isnan(w0[10]).all()


def test_gross_and_long_only():
    w = target_weights(make_view(), P)
    rows = ~np.isnan(w).all(axis=1)
    assert (w[rows] >= 0).all()
    assert np.allclose(np.abs(w[rows]).sum(axis=1), 1.0)


def test_each_param_changes_output():
    base = target_weights(make_view(), P)
    # short == long -> sigma_short == sigma_long -> calm on row 9 too
    a = target_weights(make_view(), {"short_window_days": 4, "long_window_days": 4})
    assert list(a[9]) == [1.0, 0.0]
    b = target_weights(make_view(), {"short_window_days": 2, "long_window_days": 5})
    assert not np.array_equal(base, a, equal_nan=True)
    assert not np.array_equal(base, b, equal_nan=True)


def test_point_in_time_truncation():
    full = target_weights(make_view(), P)
    for cut in range(5, 15):
        part = target_weights(make_view(r_spy=R_SPY[:cut], dates=weekdays(3)[:cut]), P)
        assert np.array_equal(part, full[:cut], equal_nan=True)
