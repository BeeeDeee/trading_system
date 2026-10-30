import numpy as np
import pytest

from strategy import target_weights, PARAMS

T = 420
T0 = 360          # shock row (decision after its close)


class FakeView:
    """Minimal stand-in for DataView with the fields the strategy reads."""

    def __init__(self, dates, names, ret_oc, tradable=None, listed=None, universe=None):
        n = len(names)
        t = dates.shape[0]
        self.dates = dates
        self.instruments = tuple(names)
        self.asset_class = tuple("us_equity" for _ in names)
        self.ret_co = np.zeros((t, n), dtype=np.float32)
        self.ret_oc = ret_oc.astype(np.float64)
        self.tradable = np.ones((t, n), bool) if tradable is None else tradable
        self.listed = np.ones((t, n), bool) if listed is None else listed
        self.delisting = np.zeros((t, n), bool)
        self.universe = np.ones((t, n), bool) if universe is None else universe
        self.close = 50.0 * np.cumprod(1.0 + self.ret_oc, axis=0)
        self.dollar_volume = np.full((t, n), 1e8)
        self.extras = {}
        self.series = {}
        self.cash_ret = np.zeros(t)

    def col(self, name):
        return self.instruments.index(name)

    def head(self, k):
        v = FakeView(self.dates[:k], self.instruments, self.ret_oc[:k], self.tradable[:k].copy(),
                     self.listed[:k].copy(), self.universe[:k].copy())
        return v


def make_dates(n):
    days = np.arange(10957, 10957 + 2 * n)          # day numbers since 1970-01-01
    days = days[(days + 3) % 7 < 5][:n]              # weekdays only
    return days.astype("datetime64[D]")


def month_start_before(dates, t):
    months = dates.astype("datetime64[M]")
    starts = [k for k in range(1, len(dates)) if months[k] != months[k - 1]]
    return max(k for k in starts if k <= t)


def twin_returns(rng, n_pairs, shocks, n_single=2):
    """Pairs (2p, 2p+1): leg b = leg a * (1 + eps), eps alternating +-0.001; leg b gets shocks[p] at T0."""
    cols = []
    alt = 0.001 * np.where(np.arange(T) % 2 == 0, 1.0, -1.0)
    for p in range(n_pairs):
        ra = rng.normal(0.0, 0.015, T)
        rb = (1.0 + ra) * (1.0 + alt) - 1.0
        rb[T0] = (1.0 + rb[T0]) * (1.0 + shocks[p]) - 1.0
        cols += [ra, rb]
    for _ in range(n_single):
        cols.append(rng.normal(0.0, 0.015, T))
    return np.column_stack(cols)


def base_view(shock=0.02, **kw):
    rng = np.random.default_rng(7)
    r = twin_returns(rng, 1, [shock])
    return FakeView(make_dates(T), ["A", "B", "C", "D"], r, **kw)


def z_at(view, t, i, j):
    lr = np.log1p(view.ret_oc)
    d = lr[:, i] - lr[:, j]
    w = d[t - 125:t + 1]
    return w[-5:].sum() / (w.std(ddof=1) * np.sqrt(5))


def test_entry_hold_and_sides():
    v = base_view()
    w = target_weights(v, PARAMS)
    a, b = v.col("A"), v.col("B")
    assert np.all(w[:T0] == 0.0)
    for t in range(T0, T0 + 5):           # entered after the close of T0, held 5 decision rows
        assert w[t, a] == 0.25 and w[t, b] == -0.25   # B jumped: B leader (short), A laggard (long)
    assert np.all(w[T0 + 5:] == 0.0)       # exit decided after the close of T0+5
    assert np.all(w[:, [v.col("C"), v.col("D")]] == 0.0)


def test_hold_days_param_and_refractory():
    v = base_view()
    w = target_weights(v, dict(PARAMS, hold_days=3))
    a = v.col("A")
    # T0..T0+2 held; T0+3 exit (pair event 3 sessions ago blocks); T0+4 gap still holds the shock -> new event
    assert list(w[T0:T0 + 8, a]) == [0.25, 0.25, 0.25, 0.0, 0.25, 0.25, 0.25, 0.0]
    w5 = target_weights(v, PARAMS)
    assert not np.array_equal(w, w5)


def test_z_entry_threshold():
    shock = None
    for s in np.arange(0.001, 0.03, 0.0002):
        v = base_view(shock=float(s))
        z = abs(z_at(v, T0, 0, 1))
        if 2.1 < z < 2.4 and abs(z_at(v, T0 - 1, 0, 1)) < 1.5:
            shock = float(s)
            break
    assert shock is not None
    v = base_view(shock=shock)
    assert np.all(target_weights(v, dict(PARAMS, z_entry=2.5)) == 0.0)
    w2 = target_weights(v, dict(PARAMS, z_entry=2.0))
    assert w2[T0, 0] == 0.25 and w2[T0, 1] == -0.25
    assert np.all(w2[:T0] == 0.0)


def test_missing_return_in_window_blocks_event():
    v = base_view()
    v.tradable[T0 - 2, v.col("A")] = False
    w = target_weights(v, PARAMS)
    assert np.all(w == 0.0)


def test_correlation_needs_240_valid_returns():
    dates = make_dates(T)
    m = month_start_before(dates, T0)
    assert m - 251 > 0 and m - 238 < T0 - 125
    v = base_view()
    v.tradable[m - 250:m - 238, v.col("B")] = False      # 12 missing -> 240 valid -> pair
    assert target_weights(v, PARAMS)[T0, 0] == 0.25
    v = base_view()
    v.tradable[m - 251:m - 238, v.col("B")] = False      # 13 missing -> 239 valid -> no pair
    assert np.all(target_weights(v, PARAMS)[m:] == 0.0)


def test_no_pairs_before_lookback():
    rng = np.random.default_rng(3)
    alt = 0.001 * np.where(np.arange(T) % 2 == 0, 1.0, -1.0)
    ra = rng.normal(0.0, 0.015, T)
    rb = (1.0 + ra) * (1.0 + alt) - 1.0
    rb[200] += 0.03                                       # before any month start >= row 251
    v = FakeView(make_dates(T), ["A", "B"], np.column_stack([ra, rb]))
    w = target_weights(v, PARAMS)
    assert np.all(w[:251] == 0.0)


def test_universe_exit_closes_both_legs():
    v = base_view()
    v.universe[T0 + 2, v.col("B")] = False
    w = target_weights(v, PARAMS)
    assert w[T0 + 1, 0] == 0.25 and w[T0 + 1, 1] == -0.25
    assert np.all(w[T0 + 2:] == 0.0)
    assert np.all(w[~v.universe] == 0.0)


def test_two_slots_largest_z_first_and_gross():
    rng = np.random.default_rng(11)
    r = twin_returns(rng, 3, [0.01, 0.02, 0.03])
    v = FakeView(make_dates(T), ["A", "B", "C", "D", "E", "F", "G", "H"], r)
    w = target_weights(v, PARAMS)
    assert np.all(w[T0, 0:2] == 0.0)                      # smallest gap dropped (no queue)
    assert list(w[T0, 2:6]) == [0.25, -0.25, 0.25, -0.25]
    assert np.all(w[T0 + 1:T0 + 5, 0:2] == 0.0)
    assert np.abs(w).sum(axis=1).max() <= 1.0 + 1e-12


def test_stock_in_at_most_one_position():
    rng = np.random.default_rng(5)
    alt = 0.001 * np.where(np.arange(T) % 2 == 0, 1.0, -1.0)
    ra = rng.normal(0.0, 0.015, T)
    rb = (1.0 + ra) * (1.0 + alt) - 1.0
    rc = (1.0 + ra) * (1.0 - 0.7 * alt) - 1.0
    ra[T0] = (1.0 + ra[T0]) * 1.02 - 1.0                  # A jumps: leader in A-B and A-C
    v = FakeView(make_dates(T), ["A", "B", "C"], np.column_stack([ra, rb, rc]))
    w = target_weights(v, PARAMS)
    assert w[T0, 0] == -0.25
    assert sorted(w[T0, 1:]) == [0.0, 0.25]
    assert np.abs(w).sum(axis=1).max() <= 0.5 + 1e-12


def test_point_in_time_truncation():
    v = base_view()
    full = target_weights(v, PARAMS)
    for k in [T0 - 1, T0, T0 + 1, T0 + 3, T0 + 6, 380]:
        part = target_weights(v.head(k), PARAMS)
        assert np.array_equal(part, full[:k])


def test_missing_universe_raises():
    v = base_view()
    v.universe = None
    with pytest.raises(ValueError):
        target_weights(v, PARAMS)
