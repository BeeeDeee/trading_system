import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS

T = 260
N = 12
A = list(range(0, 5))      # group A: ex-days 10, 73, 136 -> regular at 136, E = 199
B = list(range(5, 10))     # group B: ex-days 68, 131, 194 -> regular at 194 (short 194..198 for W=5)
OTHER = [10, 11]           # never pay


def make_view(ex_days, close=None, universe=None, tradable=None, T_=T, amounts=None):
    names = tuple("S%02d" % j for j in range(N))
    dates = np.arange(10000, 10000 + T_).astype("datetime64[D]")
    if close is None:
        close = np.full((T_, N), 100.0)
    close = close[:T_]
    ret_co = np.zeros((T_, N))
    ret_oc = np.zeros((T_, N))
    for j, days in ex_days.items():
        for t in days:
            if t < T_:
                ret_co[t, j] = 0.01            # total return 1 %, unadjusted close flat -> d = 1 %
    for (t, j), r in (amounts or {}).items():
        ret_co[t, j] = r
    if universe is not None:
        universe = universe[:T_]
    if tradable is not None:
        tradable = tradable[:T_]
    listed = np.isfinite(close)
    if universe is None:
        universe = np.ones((T_, N), dtype=bool)
    if tradable is None:
        tradable = listed.copy()
    return DataView(
        dates=dates,
        instruments=names,
        asset_class=tuple("us_equity" for _ in names),
        ret_co=ret_co,
        ret_oc=ret_oc,
        tradable=tradable,
        listed=listed,
        delisting=np.zeros((T_, N), dtype=bool),
        close=close,
        dollar_volume=np.full((T_, N), 1e8),
        universe=universe,
        extras={},
        series={},
        cash_ret=np.zeros(T_),
    )


def base_ex():
    ex = {}
    for j in A:
        ex[j] = [10, 73, 136]
    for j in B:
        ex[j] = [68, 131, 194]
    return ex


def test_params_default():
    assert PARAMS == {"window_days": 5}


def test_known_rows_w5():
    w = target_weights(make_view(base_ex()), {"window_days": 5})
    assert w.shape == (T, N)
    # quiet day: flat
    assert np.all(w[100] == 0)
    # t=136: A just went ex (short 136..140), no long leg -> long leg = basket of B + OTHER (7 names)
    for t in range(136, 141):
        assert np.allclose(w[t, A], -0.1)
        assert np.allclose(w[t, B + OTHER], 0.5 / 7)
    assert np.all(w[141] == 0)
    # t=193: A long (t+1 = 194 = E-5), B not yet ex -> short leg = basket B + OTHER
    assert np.allclose(w[192], 0.0)
    assert np.allclose(w[193, A], 0.1)
    assert np.allclose(w[193, B + OTHER], -0.5 / 7)
    # t=194..197: both legs full
    for t in range(194, 198):
        assert np.allclose(w[t, A], 0.1)
        assert np.allclose(w[t, B], -0.1)
        assert np.allclose(w[t, OTHER], 0.0)
    # t=198: A out (t+1 = E), B still short -> long leg = basket A + OTHER
    assert np.allclose(w[198, B], -0.1)
    assert np.allclose(w[198, A + OTHER], 0.5 / 7)
    # t=199: both legs empty -> flat
    assert np.all(w[199] == 0)


def test_window_param_changes_output():
    v = make_view(base_ex())
    w5 = target_weights(v, {"window_days": 5})
    w3 = target_weights(v, {"window_days": 3})
    assert not np.allclose(w5, w3)
    # W=3: A long for t+1 in [196, 198] -> t in 195..197; B short t in 194..196
    assert np.allclose(w3[193], 0.0)
    assert np.allclose(w3[194, B], -0.1) and np.allclose(w3[194, A + OTHER], 0.5 / 7)
    assert np.allclose(w3[195, A], 0.1) and np.allclose(w3[195, B], -0.1)
    assert np.allclose(w3[197, A], 0.1) and np.allclose(w3[197, B + OTHER], -0.5 / 7)
    assert np.all(w3[198] == 0)


def test_gross_and_neutral():
    for W in (3, 5, 8):
        w = target_weights(make_view(base_ex()), {"window_days": W})
        g = np.abs(w).sum(axis=1)
        assert np.all(g <= 1.0 + 1e-12)
        assert np.allclose(w.sum(axis=1), 0.0)
        assert set(np.round(g, 12)) <= {0.0, 1.0}


def test_first_rows_and_two_detections_not_regular():
    w = target_weights(make_view(base_ex()), PARAMS)
    # before any stock has 3 detections nothing is held (A after 73 has only 2)
    assert np.all(w[:136] == 0)


def test_irregular_gap_not_regular():
    ex = base_ex()
    for j in A:
        ex[j] = [10, 90, 136]          # last gap 46 < 55
    w = target_weights(make_view(ex), PARAMS)
    assert np.all(w[136:141] == 0)
    assert np.all(w[193] == 0)          # A has no predicted window
    # B alone (short, 194..198) -> long leg is the basket A + OTHER
    assert np.allclose(w[195, B], -0.1)
    assert np.allclose(w[195, A + OTHER], 0.5 / 7)


def test_too_many_detections_not_regular():
    ex = base_ex()
    for j in A:
        ex[j] = [5, 20, 40, 60, 73, 136]   # 6 in the last 252 -> not regular
    w = target_weights(make_view(ex), PARAMS)
    assert np.all(w[136:141, A] <= 0.5 / 7 + 1e-12)
    assert np.all(w[136:141] == 0)


def test_split_day_and_bounds_not_detected():
    ex = base_ex()
    close = np.full((T, N), 100.0)
    # A's third ex-day at 136 happens together with a 2:1 split (unadjusted close halves): not detected
    close[136:, A] = 50.0
    w = target_weights(make_view(ex, close=close), PARAMS)
    assert np.all(w[136:141] == 0)
    # distribution above 10 % is not detected
    v = make_view(base_ex(), amounts={(136, j): 0.2 for j in A})
    assert np.all(target_weights(v, PARAMS)[136:141] == 0)
    # distribution below 0.1 % is not detected
    v3 = make_view(base_ex(), amounts={(136, j): 0.0005 for j in A})
    assert np.all(target_weights(v3, PARAMS)[136:141] == 0)
    # just under 10 % with an unadjusted drop is detected: close falls 100 -> 95, total return +4.4 %
    close4 = np.full((T, N), 100.0)
    close4[136:, A] = 95.0
    v4 = make_view(base_ex(), close=close4, amounts={(136, j): 0.044 for j in A})
    # d = 1.044 * 100 / 95 - 1 = 0.0989
    w4 = target_weights(v4, PARAMS)
    assert np.allclose(w4[136, A], -0.1)


def test_missing_close_means_no_detection():
    close = np.full((T, N), 100.0)
    close[135, A] = np.nan              # no bar on t-1 of the third ex-day
    w = target_weights(make_view(base_ex(), close=close), PARAMS)
    assert np.all(w[136:141] == 0)


def test_early_ex_day_leaves_long_set():
    ex = base_ex()
    ex[0] = [10, 73, 136, 195]          # stock 0 goes ex at 195, before predicted E = 199
    w = target_weights(make_view(ex), PARAMS)
    assert np.allclose(w[194, A], 0.1)
    # at 195: stock 0 is a regular payer at x=195 (gap 59) -> moves to the short side (6 names);
    # the long set has 4 names -> long leg = basket of the eligible members not in either set (OTHER, 2);
    # the thin long set itself is not held
    assert w[195, 0] < 0
    assert np.allclose(w[195, 0], -0.5 / 6)
    assert np.allclose(w[195, B], -0.5 / 6)
    assert np.allclose(w[195, A[1:]], 0.0)
    assert np.allclose(w[195, OTHER], 0.25)


def test_universe_and_tradable_exclusion():
    uni = np.ones((T, N), dtype=bool)
    uni[195:, 0] = False                # stock 0 leaves the universe at 195
    w = target_weights(make_view(base_ex(), universe=uni), PARAMS)
    assert np.allclose(w[194, A], 0.1)
    assert w[195, 0] == 0
    # long set has 4 -> replaced by the basket of eligible members in neither set: OTHER (2)
    assert np.allclose(w[195, A[1:]], 0.0)
    assert np.allclose(w[195, OTHER], 0.25)
    assert np.allclose(w[195, B], -0.1)
    trd = np.ones((T, N), dtype=bool)
    trd[193, 11] = False                # not tradable on 193 -> not in the basket that day
    w2 = target_weights(make_view(base_ex(), tradable=trd), PARAMS)
    assert w2[193, 11] == 0
    assert np.allclose(w2[193, B + [10]], -0.5 / 6)
    assert np.allclose(w2[192 + 5, 11], 0.0) and np.allclose(w2[198, 11], 0.5 / 7)


def test_point_in_time_truncation():
    full = make_view(base_ex())
    wf = target_weights(full, PARAMS)
    for cut in (137, 194, 197, 199):
        part = make_view(base_ex(), T_=cut)
        wp = target_weights(part, PARAMS)
        assert np.allclose(wp, wf[:cut])


def test_point_in_time_perturbation():
    rng = np.random.default_rng(0)
    base = make_view(base_ex())
    wf = target_weights(base, PARAMS)
    for cut in (136, 193, 196):
        close = np.full((T, N), 100.0)
        close[cut + 1:] = 100.0 * np.exp(rng.normal(0, 0.05, size=(T - cut - 1, N)))
        amounts = {(t, j): float(rng.uniform(0.0, 0.05)) for t in range(cut + 1, T) for j in range(N)}
        v = make_view(base_ex(), close=close, amounts=amounts)
        wp = target_weights(v, PARAMS)
        assert np.allclose(wp[:cut + 1], wf[:cut + 1])


def test_diagnostic_events():
    from strategy import events
    mask, side = events(make_view(base_ex()), {"window_days": 5})
    # post-ex events: A at 136, B at 194 (regular payers on their detected ex-day), side -1
    assert mask[136, A].all() and np.all(side[136, A] == -1)
    assert mask[194, B].all() and np.all(side[194, B] == -1)
    # pre-ex event: A at t = E - W - 1 = 193 (t+1 = 194 = E - 5), side +1
    assert mask[193, A].all() and np.all(side[193, A] == 1)
    # B pre-ex: E = 257 -> t = 251
    assert mask[251, B].all() and np.all(side[251, B] == 1)
    assert mask.sum() == 20
    m3, _ = events(make_view(base_ex()), {"window_days": 3})
    assert m3[195, A].all() and not m3[193, A].any()


def test_missing_universe_raises():
    v = make_view(base_ex())
    v2 = DataView(dates=v.dates, instruments=v.instruments, asset_class=v.asset_class, ret_co=v.ret_co,
                  ret_oc=v.ret_oc, tradable=v.tradable, listed=v.listed, delisting=v.delisting, close=v.close,
                  dollar_volume=v.dollar_volume, universe=None, extras={}, series={}, cash_ret=v.cash_ret)
    with pytest.raises(ValueError):
        target_weights(v2, PARAMS)
