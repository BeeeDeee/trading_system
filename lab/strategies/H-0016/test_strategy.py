import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import PARAMS, target_weights, turn_of_month_flags, xnys_holidays


def ymd(y, m, d):
    ms = np.array([y - 1970], dtype=np.int64).astype("datetime64[Y]").astype("datetime64[M]")
    ms = ms + np.array([m - 1], dtype=np.int64).astype("timedelta64[M]")
    return (ms.astype("datetime64[D]") + np.array([d - 1], dtype=np.int64).astype("timedelta64[D]"))[0]


def weekdays(start, end):
    n = int((end - start).astype(np.int64)) + 1
    days = start + np.arange(n, dtype=np.int64).astype("timedelta64[D]")
    return days[np.is_busday(days)]


# ------------------------------------------------------------------------------------------- calendar
def test_holidays_2020_2022():
    hol = set(xnys_holidays(2020, 2022).tolist())
    expected = [
        (2020, 1, 1), (2020, 1, 20), (2020, 2, 17), (2020, 4, 10), (2020, 5, 25), (2020, 7, 3),
        (2020, 9, 7), (2020, 11, 26), (2020, 12, 25),
        (2021, 1, 1), (2021, 1, 18), (2021, 2, 15), (2021, 4, 2), (2021, 5, 31), (2021, 7, 5),
        (2021, 9, 6), (2021, 11, 25), (2021, 12, 24),
        (2022, 1, 17), (2022, 2, 21), (2022, 4, 15), (2022, 5, 30), (2022, 6, 20), (2022, 7, 4),
        (2022, 9, 5), (2022, 11, 24), (2022, 12, 26),
    ]
    assert hol == {ymd(*e).tolist() for e in expected}
    # Saturday New Year (2022-01-01): the Friday before stays open
    assert ymd(2021, 12, 31).tolist() not in hol


def _flags(post_days, start=ymd(2020, 9, 1), end=ymd(2022, 1, 31)):
    dates = weekdays(start, end)
    entry, exit_ = turn_of_month_flags(dates, 2, post_days)
    return dates, entry, exit_


def test_entry_is_third_to_last_scheduled_day():
    dates, entry, _ = _flags(3)
    got = set(dates[entry].tolist())
    for e in [(2020, 10, 28), (2020, 11, 25), (2020, 12, 29), (2021, 3, 29), (2021, 5, 26),
              (2021, 12, 29), (2020, 9, 28)]:
        assert ymd(*e).tolist() in got
    # exactly one entry per month
    months = dates[entry].astype("datetime64[M]")
    assert len(np.unique(months)) == len(months) == len(np.unique(dates.astype("datetime64[M]")))


@pytest.mark.parametrize("post_days,expected", [
    (2, [(2020, 11, 3), (2021, 1, 5), (2021, 4, 5), (2022, 1, 4)]),
    (3, [(2020, 11, 4), (2021, 1, 6), (2021, 4, 6), (2022, 1, 5)]),
    (4, [(2020, 11, 5), (2021, 1, 7), (2021, 4, 7), (2022, 1, 6)]),
])
def test_exit_is_post_days_th_trading_day(post_days, expected):
    dates, _, exit_ = _flags(post_days)
    got = set(dates[exit_].tolist())
    for e in expected:
        assert ymd(*e).tolist() in got
    months = dates[exit_].astype("datetime64[M]")
    assert len(np.unique(months)) == len(months)


def test_exit_fires_when_scheduled_day_missing_from_data():
    dates = weekdays(ymd(2020, 10, 1), ymd(2020, 11, 30))
    dates = dates[dates != ymd(2020, 11, 4)]          # unscheduled closure on the 3rd trading day
    _, exit_ = turn_of_month_flags(dates, 2, 3)
    assert ymd(2020, 11, 5).tolist() in set(dates[exit_].tolist())


def test_flags_do_not_depend_on_future_rows():
    dates, entry, exit_ = _flags(3)
    for cut in [10, 41, 57, 100, 200]:
        e2, x2 = turn_of_month_flags(dates[:cut], 2, 3)
        assert np.array_equal(e2, entry[:cut]) and np.array_equal(x2, exit_[:cut])


# --------------------------------------------------------------------------------------------- weights
N = 10
BETAS = np.linspace(0.2, 2.0, N)


def make_view(start=ymd(2019, 6, 3), end=ymd(2021, 1, 29), seed=0):
    dates = weekdays(start, end)
    T = len(dates)
    rng = np.random.default_rng(seed)
    f = rng.normal(0.0, 0.01, T)
    ret_oc = (f[:, None] * BETAS[None, :]).astype(np.float64)
    ret_co = np.zeros((T, N))
    ones = np.ones((T, N), bool)
    return dict(
        dates=dates, instruments=tuple("E%d" % (i + 1) for i in range(N)),
        asset_class=tuple("us_equity" for _ in range(N)),
        ret_co=ret_co, ret_oc=ret_oc, tradable=ones.copy(), listed=ones.copy(),
        delisting=np.zeros((T, N), bool), close=np.full((T, N), 50.0), dollar_volume=np.full((T, N), 1e8),
        universe=ones.copy(), extras={}, series={}, cash_ret=np.zeros(T),
    )


def build(**kw):
    return DataView(**kw)


P = {"beta_lookback": 60, "post_days": 3}


def test_params_names():
    assert set(PARAMS) == {"beta_lookback", "post_days"}
    assert PARAMS["beta_lookback"] == 252 and PARAMS["post_days"] == 3


def test_weights_top_and_bottom_quintile():
    kw = make_view()
    data = build(**kw)
    W = target_weights(data, P)
    entry, exit_ = turn_of_month_flags(kw["dates"], 2, 3)
    rows = np.flatnonzero(entry)
    late = [t for t in rows if t >= 60]
    assert late
    for t in late:
        w = W[t]
        assert np.allclose(w[[8, 9]], 0.25) and np.allclose(w[[0, 1]], -0.25)
        assert np.allclose(w[2:8], 0.0)
        assert abs(w.sum()) < 1e-12 and np.abs(w).sum() <= 1.0 + 1e-12
    assert np.all(W[exit_] == 0.0)
    # everything else is NaN (hold, no rebalancing)
    other = ~(entry | exit_)
    assert np.all(np.isnan(W[other]))
    # each entry is followed by an exit before the next entry
    seq = [("e" if entry[t] else "x") for t in range(len(entry)) if entry[t] or exit_[t]]
    for a, b in zip(seq, seq[1:]):
        assert not (a == "e" and b == "e")


def test_before_lookback_full_is_flat():
    kw = make_view()
    W = target_weights(build(**kw), P)
    entry, _ = turn_of_month_flags(kw["dates"], 2, 3)
    first = np.flatnonzero(entry)[0]          # 2019-06-26, row ~17 < 0.8 * 60
    assert first < 48
    assert np.all(W[first] == 0.0)


def test_universe_and_tradable_and_late_listing():
    kw = make_view()
    T = len(kw["dates"])
    kw["universe"][:, 9] = False                         # highest beta not a member
    kw["tradable"][:, 0] = False                         # lowest beta never tradable
    kw["listed"][: T - 30, 8] = False                    # second highest listed only in the last 30 rows
    kw["tradable"][: T - 30, 8] = False
    W = target_weights(build(**kw), P)
    entry, _ = turn_of_month_flags(kw["dates"], 2, 3)
    t = np.flatnonzero(entry)[-1]
    # eligible: 1..7 -> 7 stocks, quintile = 1 stock each side
    assert W[t, 7] == pytest.approx(0.5) and W[t, 1] == pytest.approx(-0.5)
    assert W[t, 9] == 0.0 and W[t, 0] == 0.0 and W[t, 8] == 0.0


def test_beta_lookback_changes_eligibility():
    kw = make_view()
    T = len(kw["dates"])
    kw["listed"][: T - 50, 9] = False
    kw["tradable"][: T - 50, 9] = False
    entry, _ = turn_of_month_flags(kw["dates"], 2, 3)
    t = np.flatnonzero(entry)[-1]
    assert T - 1 - t < 20
    W60 = target_weights(build(**kw), {"beta_lookback": 60, "post_days": 3})
    W120 = target_weights(build(**kw), {"beta_lookback": 120, "post_days": 3})
    assert W60[t, 9] > 0          # >= 48 obs in a 60-day window
    assert W120[t, 9] == 0.0      # < 96 obs in a 120-day window


def test_post_days_moves_exit():
    kw = make_view()
    W3 = target_weights(build(**kw), P)
    W4 = target_weights(build(**kw), {"beta_lookback": 60, "post_days": 4})
    z3 = np.flatnonzero(np.all(W3 == 0.0, axis=1))
    z4 = np.flatnonzero(np.all(W4 == 0.0, axis=1))
    assert not np.array_equal(z3, z4)


def test_truncation_invariance_of_weights():
    kw = make_view()
    W = target_weights(build(**kw), P)
    for cut in [70, 150, 333]:
        kc = {k: (v[:cut] if isinstance(v, np.ndarray) else v) for k, v in kw.items()}
        Wc = target_weights(build(**kc), P)
        assert np.array_equal(np.isnan(Wc), np.isnan(W[:cut]))
        assert np.allclose(np.nan_to_num(Wc), np.nan_to_num(W[:cut]))


def test_missing_universe_raises():
    kw = make_view()
    kw["universe"] = None
    with pytest.raises(ValueError):
        target_weights(build(**kw), P)
