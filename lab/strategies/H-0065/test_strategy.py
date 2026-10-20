import numpy as np
import pytest

from strategy import target_weights, PARAMS
from strategy import nyse_holidays, quarter_calendar, _date


class FakeData:
    def __init__(self, dates, ret, close, universe, tradable):
        T, N = ret.shape
        self.dates = dates
        self.instruments = tuple("E%d" % (100 + j) for j in range(N))
        self.asset_class = tuple("us_equity" for _ in range(N))
        self.ret_co = np.zeros((T, N))
        self.ret_oc = ret
        self.close = close
        self.dollar_volume = np.where(np.isfinite(close), 1e8, np.nan)
        self.tradable = tradable
        self.listed = np.ones((T, N), dtype=bool)
        self.delisting = np.zeros((T, N), dtype=bool)
        self.universe = universe
        self.extras = {}
        self.series = {}
        self.cash_ret = np.zeros(T)

    def col(self, name):
        return self.instruments.index(name)


def D(y, m, d):
    return _date(y, m, d)


N = 10


def make_data(seed=0, drift_q2=True):
    hol = nyse_holidays(2014, 2017)
    start = np.busday_offset(D(2015, 1, 2), 0, roll="forward", holidays=hol)
    end = D(2016, 8, 31)
    n = int(np.busday_count(start, end, holidays=hol))
    dates = np.busday_offset(start, np.arange(n), holidays=hol)
    rng = np.random.default_rng(seed)
    f = rng.normal(0, 0.01, n)
    ret = f[:, None] + rng.normal(0, 0.002, (n, N))
    if drift_q2:
        inq2 = (dates >= D(2016, 4, 1)) & (dates < D(2016, 7, 1))
        drift = (np.arange(N) - 4.5) * 0.002
        ret[inq2] += drift[None, :]
    close = np.full((n, N), 10.0)
    uni = np.ones((n, N), dtype=bool)
    trad = np.ones((n, N), dtype=bool)
    return FakeData(dates, ret, close, uni, trad)


def row(data, y, m, d):
    i = int(np.searchsorted(data.dates, D(y, m, d)))
    assert data.dates[i] == D(y, m, d)
    return i


# ---------------------------------------------------------------- calendar
def test_holidays_rules():
    h = set(nyse_holidays(2000, 2022).tolist())
    assert D(2000, 4, 21).item() in h          # Good Friday 2000
    assert D(2016, 3, 25).item() in h          # Good Friday 2016
    assert D(2010, 12, 24).item() in h         # Christmas on Saturday -> Friday
    assert D(2010, 12, 31).item() not in h     # New Year on Saturday is not observed
    assert D(2017, 1, 2).item() in h           # New Year on Sunday -> Monday
    assert D(2015, 7, 3).item() in h           # July 4 on Saturday -> Friday
    assert D(2016, 11, 24).item() in h         # Thanksgiving
    assert D(2016, 5, 30).item() in h          # Memorial Day
    assert D(2016, 1, 18).item() in h          # MLK
    assert D(2016, 2, 15).item() in h          # Presidents' Day
    assert D(2016, 9, 5).item() in h           # Labor Day
    assert D(2022, 6, 20).item() in h          # Juneteenth (Sunday -> Monday), from 2022
    assert D(2021, 6, 18).item() not in h


def test_quarter_calendar_E_L():
    dates = np.array([D(2016, 3, 1), D(2016, 6, 1), D(2016, 7, 7)])
    qid, E, L, Lp, k = quarter_calendar(dates, 10)
    assert L[0] == D(2016, 3, 31) and E[0] == D(2016, 3, 16)   # Good Friday skipped
    assert L[1] == D(2016, 6, 30) and E[1] == D(2016, 6, 16)
    assert Lp[2] == D(2016, 6, 30)
    assert k[2] == 4                                          # Jul 1, (Jul 4 holiday), 5, 6, 7
    _, E5, _, _, _ = quarter_calendar(dates, 5)
    assert E5[1] == D(2016, 6, 23)


# ---------------------------------------------------------------- weights
def test_cycle_weights_primary():
    data = make_data()
    W = target_weights(data, PARAMS)
    e = row(data, 2016, 6, 16)
    l_ = row(data, 2016, 6, 30)
    assert np.all(W[e - 1] == 0)                             # flat before E
    exp_p = np.zeros(N)
    exp_p[[8, 9]] = 0.25
    exp_p[[0, 1]] = -0.25
    for t in range(e, l_):
        np.testing.assert_allclose(W[t], exp_p)              # pressure held through L
    np.testing.assert_allclose(W[l_], -exp_p)                # release mirror at L
    for t in range(l_, l_ + 5):                              # L, Jul 1, 5, 6, 7
        np.testing.assert_allclose(W[t], -exp_p)
    assert data.dates[l_ + 5] == D(2016, 7, 8)
    assert np.all(W[l_ + 5:] == 0)                           # exit after L + post_days


def test_gross_and_neutral():
    data = make_data()
    W = target_weights(data, PARAMS)
    assert np.all(np.isfinite(W))
    g = np.abs(W).sum(axis=1)
    assert g.max() <= 1.0 + 1e-9
    on = g > 0
    assert on.any()
    np.testing.assert_allclose(W[on].sum(axis=1), 0.0, atol=1e-12)


def test_universe_exit_not_rescaled():
    data = make_data()
    i = row(data, 2016, 6, 22)
    data.universe[i, 9] = False                              # leaves for one day only
    W = target_weights(data, PARAMS)
    e = row(data, 2016, 6, 16)
    l_ = row(data, 2016, 6, 30)
    assert W[i - 1, 9] == pytest.approx(0.25)
    assert np.all(W[i:l_, 9] == 0)                           # closed for the rest of the phase
    assert np.all(W[i:l_, 8] == pytest.approx(0.25))         # leg not re-scaled
    assert W[e, 0] == pytest.approx(-0.25)
    assert W[l_, 9] == pytest.approx(-0.25)                  # back in at the release re-rank


def test_not_tradable_closes():
    data = make_data()
    i = row(data, 2016, 7, 5)
    data.tradable[i, 0] = False
    W = target_weights(data, PARAMS)
    l_ = row(data, 2016, 6, 30)
    assert W[l_, 0] == pytest.approx(0.25)
    assert np.all(W[i:, 0] == 0)


def test_coverage_rule():
    data = make_data()
    q2 = np.flatnonzero((data.dates >= D(2016, 4, 1)) & (data.dates < D(2016, 7, 1)))
    gone = q2[: int(0.25 * len(q2)) + 1]                    # stock 9 misses > 20 % of the quarter
    data.close[gone, 9] = np.nan
    W = target_weights(data, PARAMS)
    e = row(data, 2016, 6, 16)
    # 9 ranked stocks -> floor(0.2 * 9) = 1 per leg
    assert W[e, 9] == 0
    assert W[e, 8] == pytest.approx(0.5)
    assert W[e, 0] == pytest.approx(-0.5)


def test_beta_warmup_needs_200_pairs():
    data = make_data()
    T = 190
    small = FakeData(data.dates[:T], data.ret_oc[:T], data.close[:T], data.universe[:T], data.tradable[:T])
    W = target_weights(small, PARAMS)
    assert np.all(W == 0)


def test_params_change_output():
    data = make_data()
    base = target_weights(data, PARAMS)
    l_ = row(data, 2016, 6, 30)
    p = dict(PARAMS, pre_days=5)
    W = target_weights(data, p)
    assert np.all(W[row(data, 2016, 6, 22)] == 0)
    assert np.abs(W[row(data, 2016, 6, 23)]).sum() == pytest.approx(1.0)
    assert np.abs(base[row(data, 2016, 6, 22)]).sum() == pytest.approx(1.0)
    p = dict(PARAMS, post_days=3)
    W = target_weights(data, p)
    assert np.abs(W[l_ + 2]).sum() == pytest.approx(1.0)
    assert np.all(W[l_ + 3] == 0)
    p = dict(PARAMS, leg_quantile=0.1)
    W = target_weights(data, p)
    e = row(data, 2016, 6, 16)
    assert W[e, 9] == pytest.approx(0.5) and W[e, 0] == pytest.approx(-0.5)
    assert W[e, 8] == 0


def test_point_in_time_truncation():
    data = make_data()
    full = target_weights(data, PARAMS)
    for cut in (260, row(data, 2016, 6, 16), row(data, 2016, 6, 30), row(data, 2016, 7, 6)):
        T = cut + 1
        small = FakeData(data.dates[:T], data.ret_oc[:T], data.close[:T], data.universe[:T], data.tradable[:T])
        np.testing.assert_array_equal(target_weights(small, PARAMS), full[:T])


def test_missing_universe_raises():
    data = make_data()
    data.universe = None
    with pytest.raises(ValueError):
        target_weights(data, PARAMS)
