import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS

N_ROWS = 420
N_STK = 4


def _dates(n_rows, drop=None):
    days = np.arange(10000, 10000 + 2 * n_rows).astype("datetime64[D]")
    days = days[np.is_busday(days)]          # consecutive weekdays, no holidays
    if drop is not None:
        days = np.delete(days, drop)
    return days[:n_rows]


def _make(rev=None, ni=None, dv=None, universe=None, tradable=None, dates=None, n=N_STK):
    T = N_ROWS
    if dates is None:
        dates = _dates(T)
    T = len(dates)
    if rev is None:
        rev = np.full((T, n), np.nan, dtype=np.float32)
    if ni is None:
        ni = np.full((T, n), np.nan, dtype=np.float32)
    if dv is None:
        dv = np.ones((T, n), dtype=np.float32)
    if universe is None:
        universe = np.ones((T, n), dtype=bool)
    elif universe is False:
        universe = None
    if tradable is None:
        tradable = np.ones((T, n), dtype=bool)
    z = np.zeros((T, n))
    return DataView(
        dates=dates,
        instruments=tuple("S%02d" % i for i in range(n)),
        asset_class=tuple("us_equity" for _ in range(n)),
        ret_co=z.copy(),
        ret_oc=z.copy(),
        tradable=tradable,
        listed=np.ones((T, n), dtype=bool),
        delisting=np.zeros((T, n), dtype=bool),
        close=np.ones((T, n)),
        dollar_volume=dv,
        universe=universe,
        extras={"sf1_arq_revenue": rev, "sf1_arq_netinc": ni},
        series={},
        cash_ret=np.zeros(T),
    )


def _base(spike_row=90, spike=5.0, filing_row=100, n=N_STK):
    """Stock 0: revenue appears at filing_row (NaN -> value), volume spike at spike_row."""
    T = N_ROWS
    rev = np.full((T, n), np.nan, dtype=np.float32)
    rev[filing_row:, 0] = 1.0
    dv = np.ones((T, n), dtype=np.float32)
    dv[spike_row, 0] = spike
    return rev, dv


def _long_rows(w, j=0):
    return np.nonzero(w[:, j] > 0)[0].tolist()


def test_event_timing_and_weights():
    rev, dv = _base()
    w = target_weights(_make(rev=rev, dv=dv), PARAMS)
    # s* = row 90; +364 days = 52 weeks = 260 weekday rows -> r = row 350; lead 3 -> event at 347
    assert _long_rows(w) == [347, 348, 349, 350, 351]
    assert np.allclose(w[347, 0], 0.05)
    assert np.allclose(w[347, 1:], -0.05 / 3)
    assert np.all(w[352] == 0)
    assert np.all(w[:347] == 0)
    assert np.all(np.abs(w).sum(axis=1) <= 1.0)
    assert np.allclose(w.sum(axis=1), 0.0)


def test_lead_days_param():
    rev, dv = _base()
    p = dict(PARAMS, lead_days=2)
    w = target_weights(_make(rev=rev, dv=dv), p)
    assert _long_rows(w) == [348, 349, 350, 351, 352]
    p = dict(PARAMS, lead_days=4)
    w = target_weights(_make(rev=rev, dv=dv), p)
    assert _long_rows(w) == [346, 347, 348, 349, 350]


def test_min_spike_param():
    rev, dv = _base(spike=2.5)
    w = target_weights(_make(rev=rev, dv=dv), PARAMS)       # 2.5 >= 2.0 * 1
    assert _long_rows(w) == [347, 348, 349, 350, 351]
    w = target_weights(_make(rev=rev, dv=dv), dict(PARAMS, min_spike=3.0))
    assert _long_rows(w) == []


def test_spike_outside_window_no_event():
    rev, dv = _base(spike_row=70)    # window is rows 71..100
    w = target_weights(_make(rev=rev, dv=dv), PARAMS)
    assert np.all(w == 0)


def test_spike_on_filing_day_counts():
    rev, dv = _base(spike_row=100)
    w = target_weights(_make(rev=rev, dv=dv), PARAMS)
    assert _long_rows(w) == [357, 358, 359, 360, 361]


def test_no_sf1_no_event():
    _, dv = _base()
    w = target_weights(_make(dv=dv), PARAMS)
    assert np.all(w == 0)


def test_netinc_change_counts_and_value_to_nan_does_not():
    T = N_ROWS
    ni = np.full((T, N_STK), np.nan, dtype=np.float32)
    ni[20:, 0] = 1.0
    ni[100:, 0] = 2.0          # value -> value change at row 100
    dv = np.ones((T, N_STK), dtype=np.float32)
    dv[90, 0] = 5.0
    w = target_weights(_make(ni=ni, dv=dv), PARAMS)
    assert _long_rows(w) == [347, 348, 349, 350, 351]
    # value -> NaN at row 100 is not a filing
    ni2 = np.full((T, N_STK), np.nan, dtype=np.float32)
    ni2[20:100, 0] = 1.0
    w = target_weights(_make(ni=ni2, dv=dv), PARAMS)
    assert np.all(w == 0)


def test_amendment_within_40_days_ignored():
    T = N_ROWS
    rev = np.full((T, N_STK), np.nan, dtype=np.float32)
    rev[20:, 0] = 1.0           # first filing at row 20 (no spike in its window)
    rev[50:, 0] = 2.0           # 30 rows later: amendment, dropped
    dv = np.ones((T, N_STK), dtype=np.float32)
    dv[45, 0] = 5.0
    w = target_weights(_make(rev=rev, dv=dv), PARAMS)
    assert np.all(w == 0)
    rev[50:, 0] = 1.0
    rev[60:, 0] = 2.0           # 40 rows later: a new quarter
    dv[45, 0] = 1.0
    dv[55, 0] = 5.0
    w = target_weights(_make(rev=rev, dv=dv), PARAMS)
    assert _long_rows(w) == [312, 313, 314, 315, 316]


def test_untradable_during_hold_closes():
    rev, dv = _base()
    tr = np.ones((N_ROWS, N_STK), dtype=bool)
    tr[349, 0] = False
    w = target_weights(_make(rev=rev, dv=dv, tradable=tr), PARAMS)
    assert _long_rows(w) == [347, 348]
    assert w[349, 0] == 0 and w[350, 0] == 0
    assert np.all(w[349:] == 0)      # no active event -> flat book


def test_not_member_at_event_no_event_and_short_only_members():
    rev, dv = _base()
    uni = np.ones((N_ROWS, N_STK), dtype=bool)
    uni[347, 0] = False
    w = target_weights(_make(rev=rev, dv=dv, universe=uni), PARAMS)
    assert np.all(w == 0)
    uni = np.ones((N_ROWS, N_STK), dtype=bool)
    uni[:, 3] = False
    w = target_weights(_make(rev=rev, dv=dv, universe=uni), PARAMS)
    assert np.allclose(w[347], [0.05, -0.025, -0.025, 0.0])


def test_many_events_caps_and_neutral():
    n = 40
    T = N_ROWS
    rev = np.full((T, n), np.nan, dtype=np.float32)
    rev[100:, :20] = 1.0
    dv = np.ones((T, n), dtype=np.float32)
    dv[90, :20] = 5.0
    w = target_weights(_make(rev=rev, dv=dv, n=n), PARAMS)
    row = w[347]
    assert np.allclose(row[:20], 0.5 / 20)
    assert np.allclose(row[20:], -0.5 / 20)
    assert np.all(np.abs(w).sum(axis=1) <= 1.0)
    assert np.allclose(w.sum(axis=1), 0.0)


def test_holiday_on_event_day_fires_next_row():
    # drop the weekday at index 347 (a holiday): r is still the weekday that was row 350
    dates = _dates(N_ROWS, drop=347)
    rev, dv = _base()
    w = target_weights(_make(rev=rev, dv=dv, dates=dates), PARAMS)
    # the old row 348 is now row 347 and is the first row with r <= lead weekdays ahead
    assert _long_rows(w) == [347, 348, 349, 350, 351]


def test_point_in_time_truncation():
    rev, dv = _base()
    full = target_weights(_make(rev=rev, dv=dv), PARAMS)
    for cut in (100, 346, 347, 349):
        dates = _dates(N_ROWS)[: cut + 1]
        d = _make(rev=rev[: cut + 1], dv=dv[: cut + 1], dates=dates,
                  universe=np.ones((cut + 1, N_STK), dtype=bool),
                  tradable=np.ones((cut + 1, N_STK), dtype=bool))
        part = target_weights(d, PARAMS)
        assert np.array_equal(part, full[: cut + 1])


def test_requires_universe():
    rev, dv = _base()
    d = _make(rev=rev, dv=dv, universe=False)
    with pytest.raises(ValueError):
        target_weights(d, PARAMS)
