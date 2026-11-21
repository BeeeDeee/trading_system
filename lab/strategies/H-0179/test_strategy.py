import numpy as np
import pytest

from strategy import target_weights, PARAMS

PAIRS = (("FXA", "AU"), ("FXB", "GB"), ("FXC", "CA"), ("FXE", "DE"), ("FXF", "CH"), ("FXY", "JP"))
TICKERS = [tk for tk, _ in PAIRS]
KEYS = [k for _, k in PAIRS]
PREFIX = "fred_oecd_10y_yields_1d."
# slopes of the monthly yield (pp per month): FXA rises most, FXY falls most
SLOPES = {"AU": 0.03, "GB": 0.02, "CA": 0.01, "DE": 0.0, "CH": -0.01, "JP": -0.02}


class View:
    """Minimal stand-in exposing the DataView fields the strategy reads."""

    def __init__(self, dates, instruments, tradable, listed, series, universe=None):
        self.dates = dates
        self.instruments = tuple(instruments)
        self.tradable = tradable
        self.listed = listed
        self.series = series
        self.universe = universe

    def col(self, name):
        return self.instruments.index(name)


def weekdays(start_day, n_days):
    d = np.arange(start_day, start_day + n_days).astype("datetime64[D]")
    return d[np.is_busday(d)]


def make_view(n_days=900, start_day=14610, slopes=None):
    # 14610 days after the epoch is a Friday, the 1st of a month
    dates = weekdays(start_day, n_days)
    T = len(dates)
    inst = list(TICKERS) + ["ZZZ"]
    N = len(inst)
    mon = dates.astype("datetime64[M]").astype(np.int64).astype(float)
    sl = SLOPES if slopes is None else slopes
    series = {PREFIX + k: 2.0 + sl[k] * mon for k in KEYS}
    return View(dates, inst, np.ones((T, N), bool), np.ones((T, N), bool), series)


def third_sessions(dates, skip_first_month=False):
    """Hand computation: index of the third row of every calendar month in the data."""
    mon = dates.astype("datetime64[M]").astype(np.int64)
    out = []
    for m in np.unique(mon):
        if skip_first_month and m == mon[0]:
            continue
        rows = np.flatnonzero(mon == m)
        if len(rows) >= 3:
            out.append(rows[2])
    return np.array(out)


def change_rows(w):
    d = np.abs(np.diff(w, axis=0)).sum(1) > 0
    return np.flatnonzero(d) + 1


def test_decisions_on_third_session():
    v = make_view(n_days=500)
    dec = third_sessions(v.dates)
    w = target_weights(v, {"hold_months": 1, "lookback_months": 1})
    # every cohort starts on a third session; first one needs one earlier decision (lookback 1)
    starts = change_rows(w)
    assert starts[0] == dec[1]
    assert set(starts).issubset(set(dec[1:]) | set(dec[1:] + 21))   # changes only at formation or expiry
    a = v.col("FXA")
    for t in dec[1:]:
        assert w[t, a] == pytest.approx(0.25)
    assert np.all(w[: dec[1]] == 0)


def test_first_partial_month_skipped():
    v = make_view(n_days=500, start_day=14610 + 10)      # data start mid-month
    dec = third_sessions(v.dates, skip_first_month=True)
    w = target_weights(v, {"hold_months": 1, "lookback_months": 1})
    # first month's session count is unknown: its 'third row' is not a decision, so the first cohort
    # is at the third session of the third month (needs the second month's decision as its lag)
    assert np.all(w[: dec[1]] == 0)
    assert w[dec[1], v.col("FXA")] == pytest.approx(0.25)


def test_legs_weights_and_overlap():
    v = make_view()
    w = target_weights(v, PARAMS)
    h = PARAMS["hold_months"]
    L = PARAMS["lookback_months"]
    dec = third_sessions(v.dates)
    first = dec[L]
    assert np.all(w[:first] == 0)
    c = {tk: v.col(tk) for tk in TICKERS}
    row = w[first]
    assert row[c["FXA"]] == pytest.approx(0.25 / h)
    assert row[c["FXB"]] == pytest.approx(0.25 / h)
    assert row[c["FXF"]] == pytest.approx(-0.25 / h)
    assert row[c["FXY"]] == pytest.approx(-0.25 / h)
    assert row[c["FXC"]] == 0 and row[c["FXE"]] == 0 and row[v.col("ZZZ")] == 0
    later = dec[L + h + 1]                     # steady state: h cohorts alive
    assert w[later, c["FXA"]] == pytest.approx(0.25)
    assert w[later, c["FXY"]] == pytest.approx(-0.25)
    assert np.abs(w).sum(1).max() <= 1.0 + 1e-12
    assert np.allclose(w.sum(1), 0.0)
    assert np.all(w[:, v.col("ZZZ")] == 0)


@pytest.mark.parametrize("h", [1, 2, 3])
def test_cohort_expires_after_hold_sessions(h):
    v = make_view(n_days=400)
    dec = third_sessions(v.dates)
    L = 1
    t0 = dec[L]
    for k in KEYS:                              # only one cohort can form: later yields missing
        s = v.series[PREFIX + k].copy()
        s[t0 + 1:] = np.nan
        v.series[PREFIX + k] = s
    w = target_weights(v, {"hold_months": h, "lookback_months": L})
    a = v.col("FXA")
    assert np.all(w[:t0] == 0)
    assert np.all(w[t0: t0 + 21 * h, a] == pytest.approx(0.25 / h))
    assert np.all(w[t0 + 21 * h:] == 0)


def test_at_most_hold_months_cohorts():
    v = make_view()
    dec = third_sessions(v.dates)
    a = v.col("FXA")
    for h in (1, 2, 3):
        w = target_weights(v, {"hold_months": h, "lookback_months": 1})
        assert np.abs(w[:, a]).max() == pytest.approx(0.25)
        assert np.abs(w).sum(1).max() <= 1.0 + 1e-12
        # a cohort closes no later than the formation of the h-th later cohort
        gap = dec[1 + h] - dec[1]
        if gap < 21 * h:
            assert w[dec[1 + h] - 1, a] == pytest.approx(0.25)


def test_fewer_than_five_eligible_no_cohort():
    v = make_view(n_days=400)
    v.series[PREFIX + "AU"] = np.full(len(v.dates), np.nan)
    dec = third_sessions(v.dates)
    w5 = target_weights(v, PARAMS)
    assert w5[dec[3], v.col("FXB")] == pytest.approx(0.125)   # FXB now top, FXA excluded
    assert w5[dec[3], v.col("FXC")] == pytest.approx(0.125)
    assert w5[dec[3], v.col("FXA")] == 0
    v.tradable[:, v.col("FXC")] = False
    w4 = target_weights(v, PARAMS)
    assert np.all(w4 == 0)


def test_lagged_yield_missing_excludes():
    v = make_view(n_days=400)
    dec = third_sessions(v.dates)
    s = v.series[PREFIX + "AU"].copy()
    s[: dec[0] + 1] = np.nan                     # lagged value of the first cohort missing
    v.series[PREFIX + "AU"] = s
    w = target_weights(v, PARAMS)
    assert w[dec[3], v.col("FXA")] == 0
    assert w[dec[4], v.col("FXA")] == pytest.approx(0.125)


def test_untradable_at_decision_excluded():
    v = make_view(n_days=400)
    dec = third_sessions(v.dates)
    t = dec[3]
    v.tradable[t, v.col("FXA")] = False
    w = target_weights(v, PARAMS)
    assert w[t, v.col("FXA")] == 0
    assert w[t, v.col("FXC")] == pytest.approx(0.125)


def test_ties_alphabetical_and_disjoint_legs():
    v = make_view(n_days=400, slopes={k: 0.0 for k in KEYS})
    w = target_weights(v, PARAMS)
    t = third_sessions(v.dates)[3]
    assert w[t, v.col("FXA")] == pytest.approx(0.125)
    assert w[t, v.col("FXB")] == pytest.approx(0.125)
    assert w[t, v.col("FXC")] == pytest.approx(-0.125)
    assert w[t, v.col("FXE")] == pytest.approx(-0.125)
    assert w[t, v.col("FXF")] == 0 and w[t, v.col("FXY")] == 0


def test_rank_follows_yield_change_not_level():
    # FXY has the highest level but falls; FXC has the lowest level but rises most
    v = make_view(n_days=400)
    mon = v.dates.astype("datetime64[M]").astype(np.int64).astype(float)
    v.series[PREFIX + "JP"] = 50.0 - 0.05 * mon
    v.series[PREFIX + "CA"] = -50.0 + 0.05 * mon
    w = target_weights(v, PARAMS)
    t = third_sessions(v.dates)[3]
    assert w[t, v.col("FXC")] > 0 and w[t, v.col("FXY")] < 0


def test_params_change_output():
    v = make_view()
    base = target_weights(v, PARAMS)
    assert not np.allclose(base, target_weights(v, {**PARAMS, "hold_months": 3}))
    assert not np.allclose(base, target_weights(v, {**PARAMS, "lookback_months": 6}))


def test_point_in_time_truncation():
    v = make_view()
    full = target_weights(v, PARAMS)
    for cut in (100, 237, 400, 555):
        tv = View(v.dates[:cut], v.instruments, v.tradable[:cut], v.listed[:cut],
                  {k: s[:cut] for k, s in v.series.items()})
        assert np.array_equal(target_weights(tv, PARAMS), full[:cut])


def test_missing_series_raises():
    v = make_view(n_days=200)
    del v.series[PREFIX + "JP"]
    with pytest.raises(KeyError):
        target_weights(v, PARAMS)
