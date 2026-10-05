import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS, abnormal_volume, normalized_av, decision_rows, _weekday

N_STOCKS = 8
START = 10959          # day number of a Monday (2000-01-03)


def business_days(n_rows, start=START):
    d = np.arange(start, start + 2 * n_rows, dtype=np.int64)
    d = d[_weekday(d) < 5][:n_rows]
    return d


def make_view(days, dv, universe=None, tradable=None, no_universe=False):
    T, N = dv.shape
    dates = days.astype("datetime64[D]")
    if no_universe:
        universe = None
    elif universe is None:
        universe = np.ones((T, N), dtype=bool)
    if tradable is None:
        tradable = np.ones((T, N), dtype=bool)
    return DataView(
        dates=dates,
        instruments=tuple("E%d" % (i + 1) for i in range(N)),
        asset_class=tuple("us_equity" for _ in range(N)),
        ret_co=np.zeros((T, N)),
        ret_oc=np.zeros((T, N)),
        tradable=tradable,
        listed=np.ones((T, N), dtype=bool),
        delisting=np.zeros((T, N), dtype=bool),
        close=np.ones((T, N)),
        dollar_volume=dv,
        universe=universe,
        extras={"liq_rank": np.tile(np.arange(1, N + 1, dtype=float), (T, 1))},
        series={},
        cash_ret=np.zeros(T),
    )


def row_of(days, day):
    r = np.nonzero(days == day)[0]
    assert r.size == 1, "day is not a business day of the test calendar"
    return int(r[0])


def setup(n_rows=600, spike=5.0, ann_shift=0, q_shift=0, stock=0):
    """Constant dollar volume; stock `stock` spikes on the Wednesday 52 weeks (+ann_shift days) and
    13 weeks (+q_shift days) before the holding week following a Friday decision row t."""
    days = business_days(n_rows)
    dv = np.full((n_rows, N_STOCKS), 1e6)
    fridays = np.nonzero(_weekday(days) == 4)[0]
    t = int(fridays[fridays > 450][0])
    d1 = days[t] + 3
    wed = d1 + 2
    ra = row_of(days, wed - 364 + ann_shift)
    rq = row_of(days, wed - 91 + q_shift)
    dv[ra, stock] *= spike
    dv[rq, stock] *= spike
    return days, dv, t


def test_params_match_card():
    assert PARAMS == {"av_threshold": 2.0, "tol_days": 3}


def test_abnormal_volume_median_and_min_valid():
    T = 80
    dv = np.full((T, 2), 10.0)
    dv[:, 1] = np.nan
    dv[30:, 1] = 4.0          # stock 1 starts at row 30
    av = abnormal_volume(dv)
    assert np.all(np.isnan(av[:40, 0]))       # fewer than 40 valid baseline days
    assert np.allclose(av[40:, 0], 1.0)
    assert np.all(np.isnan(av[:70, 1]))       # needs 40 valid days, first at row 70
    assert np.allclose(av[70:, 1], 1.0)
    # median excludes day d itself and uses the baseline
    dv2 = np.full((T, 1), 10.0)
    dv2[50, 0] = 30.0
    av2 = abnormal_volume(dv2)
    assert av2[50, 0] == pytest.approx(3.0)
    assert av2[51, 0] == pytest.approx(1.0)   # one spike does not move the median
    # even-sized window: mean of the two middle values
    dv3 = np.full((T, 1), np.nan)
    dv3[:21, 0] = 1.0
    dv3[21:42, 0] = 3.0
    dv3[42, 0] = 4.0
    av3 = abnormal_volume(dv3)
    assert av3[42, 0] == pytest.approx(2.0)   # 42 valid days, median (1 + 3) / 2


def test_normalized_by_universe_median():
    av = np.array([[1.0, 2.0, 4.0, 100.0]])
    uni = np.array([[True, True, True, False]])
    avn = normalized_av(av, uni)
    assert np.allclose(avn, [[0.5, 1.0, 2.0, 50.0]])


def test_decision_calendar_fridays_and_catch_up():
    days = business_days(30)
    # drop the Friday of the second week (holiday)
    fri2 = np.nonzero(_weekday(days) == 4)[0][1]
    days_h = np.delete(days, fri2)
    dec, d1 = decision_rows(days_h)
    wd = _weekday(days_h[dec])
    # all Fridays plus the Monday after the holiday Friday
    assert np.sum(wd == 4) == np.sum(_weekday(days_h) == 4)
    catch = dec[wd != 4]
    assert catch.size == 1
    assert _weekday(days_h[catch[0]]) == 0 and days_h[catch[0] - 1] == days[fri2 - 1]
    # holding week Monday: next Monday for Fridays, the current Monday for the catch-up
    assert np.all(d1[wd == 4] == days_h[dec][wd == 4] + 3)
    assert d1[wd != 4][0] == days_h[catch[0]]


def test_predicted_announcer_long_rest_short():
    days, dv, t = setup()
    w = target_weights(make_view(days, dv), PARAMS)
    row = w[t]
    assert row[0] == pytest.approx(0.05)
    assert np.allclose(row[1:], -0.05 / (N_STOCKS - 1))
    assert row.sum() == pytest.approx(0.0)
    # non-decision rows are NaN
    wd = _weekday(days)
    assert np.all(np.isnan(w[wd != 4]))
    # gross <= 1 and dollar neutral on every decision row
    dec = w[wd == 4]
    assert np.all(np.abs(dec).sum(axis=1) <= 1.0 + 1e-12)
    assert np.allclose(dec.sum(axis=1), 0.0)
    # the neighbouring weeks are flat (annual window does not match)
    fr = np.nonzero(wd == 4)[0]
    k = int(np.nonzero(fr == t)[0][0])
    assert np.all(w[fr[k - 1]] == 0.0) and np.all(w[fr[k + 1]] == 0.0)


def test_needs_both_lags():
    days, dv, t = setup()
    rq = row_of(days, days[t] + 5 - 91)
    dv[rq, 0] = 1e6           # remove the quarterly spike
    w = target_weights(make_view(days, dv), PARAMS)
    assert np.all(w[t] == 0.0)
    days, dv, t = setup()
    ra = row_of(days, days[t] + 5 - 364)
    dv[ra, 0] = 1e6           # remove the annual spike
    w = target_weights(make_view(days, dv), PARAMS)
    assert np.all(w[t] == 0.0)


def test_tol_days_changes_annual_window():
    # annual spike on the Monday before the window's Monday: D1 - 364 - 7 + ... use a -5 day shift
    days, dv, t = setup(ann_shift=-7)   # Wednesday one week earlier: D1-364-5 = outside tol 3, inside tol 5
    w3 = target_weights(make_view(days, dv), {"av_threshold": 2.0, "tol_days": 3})
    w5 = target_weights(make_view(days, dv), {"av_threshold": 2.0, "tol_days": 5})
    w1 = target_weights(make_view(days, dv), {"av_threshold": 2.0, "tol_days": 1})
    assert np.all(w3[t] == 0.0)
    assert np.all(w1[t] == 0.0)
    assert w5[t, 0] == pytest.approx(0.05)


def test_quarterly_window_fixed_10_days():
    days, dv, t = setup(q_shift=-12)    # D1-91-10 = Wednesday-12: inside
    w = target_weights(make_view(days, dv), PARAMS)
    assert w[t, 0] == pytest.approx(0.05)
    days, dv, t = setup(q_shift=-14)    # Monday-91-12 = outside the quarterly window
    w = target_weights(make_view(days, dv), PARAMS)
    assert w[t, 0] <= 0.0


def test_av_threshold_changes_events():
    days, dv, t = setup(spike=2.5)
    w2 = target_weights(make_view(days, dv), {"av_threshold": 2.0, "tol_days": 3})
    w3 = target_weights(make_view(days, dv), {"av_threshold": 3.0, "tol_days": 3})
    assert w2[t, 0] == pytest.approx(0.05)
    assert np.all(w3[t] == 0.0)


def test_event_must_be_local_maximum():
    days, dv, t = setup()
    ra = row_of(days, days[t] + 5 - 364)
    dv[ra + 10, 0] = 1e6 * 8.0          # a bigger spike 10 rows later: the first is no event
    w = target_weights(make_view(days, dv), PARAMS)
    assert w[t, 0] <= 0.0


def test_long_leg_scaling_many_names():
    days = business_days(600)
    n = 40
    dv = np.full((600, n), 1e6)
    fridays = np.nonzero(_weekday(days) == 4)[0]
    t = int(fridays[fridays > 450][0])
    wed = days[t] + 5
    ra = row_of(days, wed - 364)
    rq = row_of(days, wed - 91)
    dv[ra, :12] *= 5.0
    dv[rq, :12] *= 5.0
    w = target_weights(make_view(days, dv), PARAMS)
    assert np.allclose(w[t, :12], 0.5 / 12)
    assert np.allclose(w[t, 12:], -0.5 / 28)
    assert np.abs(w[t]).sum() == pytest.approx(1.0)


def test_eligibility_universe_tradable_history():
    days, dv, t = setup()
    uni = np.ones_like(dv, dtype=bool)
    trad = np.ones_like(dv, dtype=bool)
    uni[t, 1] = False                    # not a member at the decision close
    trad[t, 2] = False                   # not tradable
    dv[t - 150:t - 40, 3] = np.nan       # AVn undefined on much of the last 380 days
    w = target_weights(make_view(days, dv, universe=uni, tradable=trad), PARAMS)
    assert w[t, 1] == 0.0 and w[t, 2] == 0.0 and w[t, 3] == 0.0
    assert w[t, 0] == pytest.approx(0.05)
    assert np.allclose(w[t, 4:], -0.05 / (N_STOCKS - 4))
    # the announcer itself must be a member
    uni2 = np.ones_like(dv, dtype=bool)
    uni2[t, 0] = False
    w2 = target_weights(make_view(days, dv, universe=uni2), PARAMS)
    assert np.all(w2[t] == 0.0)


def test_history_before_universe_entry_counts():
    days, dv, t = setup()
    uni = np.ones_like(dv, dtype=bool)
    uni[: t - 5, 0] = False              # stock 0 joins the universe a week before the decision
    w = target_weights(make_view(days, dv, universe=uni), PARAMS)
    assert w[t, 0] == pytest.approx(0.05)


def test_warm_up_is_flat_and_point_in_time():
    days, dv, t = setup()
    rng = np.random.default_rng(7)
    dv = dv * np.exp(0.3 * rng.standard_normal(dv.shape))
    full = target_weights(make_view(days, dv), PARAMS)
    fr = np.nonzero(_weekday(days) == 4)[0]
    assert np.all(full[fr[fr < 280]] == 0.0)          # no annual history yet
    for cut in (t - 30, t, t + 5, 560):
        part = target_weights(make_view(days[: cut + 1], dv[: cut + 1]), PARAMS)
        np.testing.assert_array_equal(part, full[: cut + 1])
    dv2 = dv.copy()
    dv2[t + 1:] = rng.uniform(1e5, 1e8, size=dv2[t + 1:].shape)
    pert = target_weights(make_view(days, dv2), PARAMS)
    np.testing.assert_array_equal(pert[: t + 1], full[: t + 1])


def test_missing_universe_fails_loudly():
    days, dv, t = setup()
    view = make_view(days, dv, no_universe=True)
    with pytest.raises(Exception):
        target_weights(view, PARAMS)
