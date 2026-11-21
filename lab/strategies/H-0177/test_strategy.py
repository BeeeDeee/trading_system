import numpy as np
import pytest

from strategy import PARAMS, target_weights, decision_rows, trailing_cum_return, events

NAMES = ("SPY", "IEF", "SHY", "SHV", "BIL", "VGSH", "SCHO")
DAY_2004_01_01 = 12418          # days since 1970-01-01 (a Thursday)


def weekdays(n, start=DAY_2004_01_01):
    d0 = np.array([start], dtype=np.int64).astype("datetime64[D]")[0]
    return np.busday_offset(d0, np.arange(n), roll="forward")


class View:
    def __init__(self, dates, ret_oc, listed=None, ret_co=None):
        T, N = ret_oc.shape
        self.dates = dates
        self.instruments = NAMES
        self.asset_class = tuple("us_etf" for _ in NAMES)
        self.ret_oc = ret_oc.astype(np.float64)
        self.ret_co = np.zeros((T, N)) if ret_co is None else ret_co
        self.listed = np.ones((T, N), dtype=bool) if listed is None else listed
        self.tradable = self.listed.copy()
        self.delisting = np.zeros((T, N), dtype=bool)
        self.close = np.where(self.listed, 100.0, np.nan)
        self.dollar_volume = np.where(self.listed, 1e8, np.nan)
        self.universe = None
        self.extras = {}
        self.series = {}
        self.cash_ret = np.zeros(T)

    def col(self, name):
        return self.instruments.index(name)


def const_view(T, spy, ief, shy, listed=None):
    r = np.zeros((T, len(NAMES)))
    r[:, 0], r[:, 1], r[:, 2] = spy, ief, shy
    r[:, 3:] = 0.0001
    return View(weekdays(T), r, listed=listed)


# ---------------------------------------------------------------- calendar
def test_decision_rows_last_weekday_of_month():
    d = weekdays(260)
    dec = decision_rows(d)
    m = d.astype("datetime64[M]")
    expected = np.zeros(len(d), dtype=bool)
    expected[:-1] = m[1:] != m[:-1]
    # last row: flagged only if its next weekday is in a new month
    expected[-1] = np.busday_offset(d[-1], 1).astype("datetime64[M]") != m[-1]
    assert np.array_equal(dec, expected)
    assert dec.sum() >= 11


def test_decision_rows_holiday_on_last_weekday_falls_to_next_session():
    d = weekdays(200)
    may31 = np.array([DAY_2004_01_01 + 151], dtype=np.int64).astype("datetime64[D]")[0]   # Monday 2004-05-31
    keep = d != may31
    assert (~keep).sum() == 1
    d = d[keep]
    dec = decision_rows(d)
    i_may31_prev = int(np.flatnonzero(d < may31)[-1])     # Friday 2004-05-28
    i_jun1 = i_may31_prev + 1                              # Tuesday 2004-06-01
    assert not dec[i_may31_prev]
    assert dec[i_jun1]
    # May has no decision row, June has two (June 1 for May, June 30 for June), others one
    m = d.astype("datetime64[M]")
    counts = [int(dec[m == mm].sum()) for mm in np.unique(m)[:-1]]
    assert counts[4] == 0 and counts[5] == 2
    assert all(c == 1 for i, c in enumerate(counts) if i not in (4, 5))


# ---------------------------------------------------------------- signal
def test_trailing_cum_return_and_missing_as_zero():
    T = 10
    r = np.zeros((T, len(NAMES)))
    r[:, 1] = 0.01
    r[4, 1] = np.nan
    v = View(weekdays(T), r)
    out = trailing_cum_return(v, 1, 3)
    assert np.all(np.isnan(out[:3]))
    assert out[3] == pytest.approx(1.01 ** 3 - 1)
    assert out[5] == pytest.approx(1.01 ** 2 - 1)   # rows 3..5, row 4 missing -> 0
    assert out[9] == pytest.approx(1.01 ** 3 - 1)


@pytest.mark.parametrize("spy,ief,shy,exp", [
    (0.002, 0.001, 0.0, (0.80, 0.20, 0.00)),          # sB +, sE +
    (-0.002, 0.001, 0.0, (0.80 / 1.4, 0.60 / 1.4, 0.0)),  # sB +, sE -
    (0.002, -0.001, 0.0, (0.40, 0.20, 0.40)),         # sB -, sE +
    (-0.002, -0.001, 0.0, (0.40, 0.60, 0.00)),        # sB -, sE -
])
def test_four_states(spy, ief, shy, exp):
    T = 120
    v = const_view(T, spy, ief, shy)
    w = target_weights(v, {"lookback_days": 20})
    dec = decision_rows(v.dates)
    rows = np.flatnonzero(dec)
    late = rows[rows >= 20]
    assert len(late) >= 3
    for t in late:
        assert w[t, 0] == pytest.approx(exp[0])
        assert w[t, 1] == pytest.approx(exp[1])
        assert w[t, 2] == pytest.approx(exp[2])
        assert np.all(w[t, 3:] == 0.0)
        assert np.abs(w[t]).sum() == pytest.approx(1.0)
        assert np.all(w[t] >= 0)
    assert np.all(np.isnan(w[~dec]))


def test_ties_count_as_negative():
    v = const_view(120, 0.0, 0.0, 0.0)
    w = target_weights(v, {"lookback_days": 20})
    t = np.flatnonzero(decision_rows(v.dates))[-1]
    assert tuple(np.round(w[t, :3], 6)) == (0.40, 0.60, 0.0)   # sB = -1, sE = -1


def test_benchmark_before_full_lookback():
    v = const_view(120, 0.002, 0.001, 0.0)
    w = target_weights(v, {"lookback_days": 60})
    rows = np.flatnonzero(decision_rows(v.dates))
    for t in rows:
        if t < 60:
            assert tuple(np.round(w[t, :3], 6)) == (0.60, 0.40, 0.0)
        else:
            assert tuple(np.round(w[t, :3], 6)) == (0.80, 0.20, 0.0)


def test_late_listing_of_shy_delays_signal():
    T = 200
    listed = np.ones((T, len(NAMES)), dtype=bool)
    listed[:50, 2] = False
    v = const_view(T, 0.002, 0.001, 0.0, listed=listed)
    w = target_weights(v, {"lookback_days": 20})
    rows = np.flatnonzero(decision_rows(v.dates))
    for t in rows:
        if t - 20 < 50:
            assert tuple(np.round(w[t, :3], 6)) == (0.60, 0.40, 0.0)
        else:
            assert tuple(np.round(w[t, :3], 6)) == (0.80, 0.20, 0.0)


def test_lookback_parameter_changes_signal():
    T = 300
    r = np.zeros((T, len(NAMES)))
    r[:, 0] = 0.001
    r[:200, 1] = 0.003       # IEF strong long ago
    r[200:, 1] = -0.002      # weak recently
    v = View(weekdays(T), r)
    t = int(np.flatnonzero(decision_rows(v.dates))[-1])
    w_short = target_weights(v, {"lookback_days": 40})
    w_long = target_weights(v, {"lookback_days": 250})
    assert w_short[t, 0] == pytest.approx(0.40)    # sB = -1
    assert w_long[t, 0] == pytest.approx(0.80)     # sB = +1
    assert PARAMS == {"lookback_days": 252}


def test_point_in_time_truncation():
    rng = np.random.default_rng(0)
    T = 400
    r = rng.normal(0, 0.01, (T, len(NAMES)))
    v = View(weekdays(T), r)
    full = target_weights(v, {"lookback_days": 60})
    for cut in (70, 150, 233, 399):
        vt = View(v.dates[:cut], r[:cut])
        wt = target_weights(vt, {"lookback_days": 60})
        assert np.array_equal(np.isnan(wt), np.isnan(full[:cut]))
        assert np.allclose(np.nan_to_num(wt), np.nan_to_num(full[:cut]))


def test_fails_loudly_on_missing_instrument():
    v = const_view(50, 0.0, 0.0, 0.0)
    v.instruments = ("SPY", "XXX", "SHY", "SHV", "BIL", "VGSH", "SCHO")
    with pytest.raises(ValueError):
        target_weights(v, {"lookback_days": 20})


# ---------------------------------------------------------------- diagnostic
def test_events_sides_and_peer_rule():
    T = 200
    listed = np.ones((T, len(NAMES)), dtype=bool)
    listed[:120, 4] = False      # BIL late
    listed[:120, 5] = False      # VGSH late
    listed[:, 6] = False         # SCHO never
    v = const_view(T, -0.002, 0.001, 0.0, listed=listed)   # sB = +1, sE = -1
    mask, side = events(v, {"lookback_days": 20})
    dec = decision_rows(v.dates)
    rows = np.flatnonzero(mask.any(axis=1))
    # SHY, SHV priced always (2 peers) -> need BIL / VGSH from row 120
    assert np.all(rows >= 120)
    assert np.all(dec[rows])
    expected = np.flatnonzero(dec & (np.arange(T) >= 120))
    assert np.array_equal(rows, expected)
    assert np.all(mask[rows, 0]) and np.all(mask[rows, 1])
    assert not mask[:, 2:].any()
    assert np.all(side[rows, 0] == 1.0)     # SPY side sB
    assert np.all(side[rows, 1] == 1.0)     # IEF side -sE = +1


def test_events_need_full_lookback():
    v = const_view(200, 0.002, -0.001, 0.0)    # sB = -1, sE = +1
    mask, side = events(v, {"lookback_days": 100})
    rows = np.flatnonzero(mask.any(axis=1))
    assert len(rows) > 0 and np.all(rows >= 100)
    assert np.all(side[rows, 0] == -1.0)
    assert np.all(side[rows, 1] == -1.0)
