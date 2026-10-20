import numpy as np
import pytest
from lab.framework.data import DataView
from strategy import target_weights, PARAMS, events

N = 10
NAMES = tuple("E%d" % (i + 1) for i in range(N))


def weekday_dates(n_rows, drop=()):
    # day numbers since the epoch, weekdays only, no date literals
    days = np.arange(12000, 12000 + 2 * n_rows).astype("datetime64[D]")
    days = days[np.is_busday(days)]
    keep = np.ones(days.size, dtype=bool)
    for d in drop:
        keep[d] = False
    return days[keep][:n_rows]


def make_view(dates, ea, ret_oc, universe=None, tradable=None):
    T = dates.size
    N = ea.shape[1]
    NAMES = tuple("E%d" % (i + 1) for i in range(N))
    ret_co = np.zeros((T, N))
    if universe is None:
        universe = np.ones((T, N), dtype=bool)
    if tradable is None:
        tradable = np.ones((T, N), dtype=bool)
    return DataView(
        dates=dates,
        instruments=NAMES,
        asset_class=tuple("us_equity" for _ in range(N)),
        ret_co=ret_co,
        ret_oc=ret_oc,
        tradable=tradable,
        listed=np.ones((T, N), dtype=bool),
        delisting=np.zeros((T, N), dtype=bool),
        close=np.full((T, N), 100.0),
        dollar_volume=np.full((T, N), 1e8),
        universe=universe,
        extras={"ea_reaction_day": ea.astype(np.float32),
                "liq_rank": np.tile(np.arange(1, N + 1, dtype=np.float32), (T, 1))},
        series={},
        cash_ret=np.zeros(T),
    )


def plus_days(d, k):
    return (d.astype(np.int64) + k).astype("datetime64[D]")


def row_of(dates, d):
    i = np.nonzero(dates == d)[0]
    return int(i[0]) if i.size else None


def base_case(drop=(), shock_offset=0, N=N):
    """Stocks 0,1,2,4 have a reaction day at row 5. Their predicted session P = date(5) + 364 days.
    At the candidate row t the formation return is s0 -5 %, s1 +5 %, others 0.001*j.
    s2 is a loser too but had a reaction day 10 sessions before t (already released)."""
    full = weekday_dates(330)
    L = 5
    D = plus_days(full[L], 364)
    dates = weekday_dates(330, drop=drop)
    p = row_of(dates, D)
    if p is None:  # D is a "holiday": P is the next row
        p = int(np.searchsorted(dates, D))
    t = p - 3 + shock_offset
    T = dates.size
    ea = np.zeros((T, N))
    ea[L, [0, 1, 2, 4]] = 1.0
    ea[t - 10, 2] = 1.0
    ret = np.zeros((T, N))
    ret[t] = 0.001 * np.arange(N)
    ret[t, 0] = -0.05
    ret[t, 1] = 0.05
    return dates, ea, ret, p, t


def test_basic_events_and_hold():
    dates, ea, ret, p, t = base_case()
    w = target_weights(make_view(dates, ea, ret), PARAMS)
    h = PARAMS["hold_days"]
    assert np.all(w[:t] == 0)
    assert np.allclose(w[t:t + h, 0], 0.05)       # loser long, rows t..t+4
    assert np.allclose(w[t:t + h, 1], -0.05)      # winner short
    assert np.all(w[t + h:, :] == 0)              # closed at the open of t+1+hold
    assert np.all(w[:, 2] == 0)                   # recent reaction: not a candidate
    assert np.all(w[:, 4] == 0)                   # middle return: no event
    assert np.all(w[:, 3] == 0) and np.all(w[:, 9] == 0)   # no 8-K history
    assert np.all(np.abs(w).sum(axis=1) <= 1.0 + 1e-12)


def test_holiday_on_predicted_day_keeps_lead():
    full = weekday_dates(330)
    D = plus_days(full[5], 364)
    hol = row_of(full, D)
    dates, ea, ret, p, t = base_case(drop=(hol,))
    assert dates[p] > D
    w = target_weights(make_view(dates, ea, ret), PARAMS)
    assert np.allclose(w[t, 0], 0.05)
    assert np.all(w[:t, 0] == 0)


def test_holiday_between_t_and_p_fires_one_session_late():
    full = weekday_dates(330)
    D = plus_days(full[5], 364)
    hol = row_of(full, D) - 2
    # with the holiday unknown in advance, the event fires at P-2 instead of P-3
    dates, ea, ret, p, t = base_case(drop=(hol,), shock_offset=1)
    assert t == p - 2
    w = target_weights(make_view(dates, ea, ret), PARAMS)
    assert np.allclose(w[t, 0], 0.05)
    assert np.all(w[:t, 0] == 0)


def test_hold_days_param():
    dates, ea, ret, p, t = base_case()
    for h in (4, 6):
        w = target_weights(make_view(dates, ea, ret), dict(PARAMS, hold_days=h))
        assert np.allclose(w[t:t + h, 0], 0.05)
        assert np.all(w[t + h:, 0] == 0)


def test_formation_days_param():
    dates, ea, ret, p, t = base_case()
    ret = ret.copy()
    # an offsetting move 4 sessions back only matters for formation_days=5
    ret[t - 4, 0] = 0.20
    w3 = target_weights(make_view(dates, ea, ret), PARAMS)
    w5 = target_weights(make_view(dates, ea, ret), dict(PARAMS, formation_days=5))
    assert w3[t, 0] > 0
    assert w5[t, 0] < 0   # +14 % over 5 sessions: now the top winner


def test_leg_quantile_param():
    dates, ea, ret, p, t = base_case()
    w = target_weights(make_view(dates, ea, ret), dict(PARAMS, leg_quantile=0.5))
    # with the median as both cut-offs s4 (0.004 < median 0.0055) becomes a loser
    assert w[t, 4] > 0
    w1 = target_weights(make_view(dates, ea, ret), PARAMS)
    assert w1[t, 4] == 0


def test_universe_exit_closes_slot_and_entry_needs_member():
    dates, ea, ret, p, t = base_case()
    uni = np.ones((dates.size, N), dtype=bool)
    uni[t + 2:, 0] = False
    w = target_weights(make_view(dates, ea, ret, universe=uni), PARAMS)
    assert np.allclose(w[t:t + 2, 0], 0.05)
    assert np.all(w[t + 2:, 0] == 0)
    uni2 = np.ones((dates.size, N), dtype=bool)
    uni2[t, 1] = False
    w2 = target_weights(make_view(dates, ea, ret, universe=uni2), PARAMS)
    assert np.all(w2[:, 1] == 0)


def test_missing_return_no_event():
    dates, ea, ret, p, t = base_case()
    v = make_view(dates, ea, ret)
    v.close[t - 1, 0] = np.nan
    w = target_weights(v, PARAMS)
    assert np.all(w[:, 0] == 0)


def test_gross_not_scaled_at_10_slots():
    dates, ea, ret, p, t = base_case()
    ea2 = ea.copy()
    ea2[5, :] = 1.0
    ea2[t - 10, 2] = 0.0
    w = target_weights(make_view(dates, ea2, ret), dict(PARAMS, leg_quantile=0.5))
    assert np.isclose(np.abs(w[t]).sum(), 0.5)
    assert np.allclose(np.abs(w[t]), 0.05)


def test_gross_scaled_when_more_than_20_slots():
    dates, ea, ret, p, t = base_case(N=30)
    ea2 = ea.copy()
    ea2[5, :] = 1.0
    ea2[t - 10, 2] = 0.0
    w = target_weights(make_view(dates, ea2, ret), dict(PARAMS, leg_quantile=0.5))
    g = np.abs(w).sum(axis=1)
    assert g.max() <= 1.0
    assert np.isclose(g[t], 1.0)
    assert np.allclose(np.abs(w[t]), 1.0 / 30)
    assert np.allclose(w[t, 0], 1.0 / 30) and np.allclose(w[t, 1], -1.0 / 30)


def test_point_in_time_truncation():
    dates, ea, ret, p, t = base_case()
    full = target_weights(make_view(dates, ea, ret), PARAMS)
    for cut in (t - 1, t, t + 2, dates.size - 5):
        v = make_view(dates[:cut + 1], ea[:cut + 1], ret[:cut + 1])
        part = target_weights(v, PARAMS)
        assert np.array_equal(part, full[:cut + 1])


def test_point_in_time_perturbation():
    dates, ea, ret, p, t = base_case()
    full = target_weights(make_view(dates, ea, ret), PARAMS)
    rng = np.random.default_rng(7)
    for cut in (t - 1, t, t + 3):
        ea2 = ea.copy()
        ret2 = ret.copy()
        ret2[cut + 1:] = rng.normal(0, 0.05, ret2[cut + 1:].shape)
        ea2[cut + 1:] = (rng.random(ea2[cut + 1:].shape) < 0.2).astype(float)
        w = target_weights(make_view(dates, ea2, ret2), PARAMS)
        assert np.array_equal(w[:cut + 1], full[:cut + 1])


def test_event_mask_and_side():
    dates, ea, ret, p, t = base_case()
    m1, s1 = events(make_view(dates, ea, ret), PARAMS)
    assert m1.sum() == 2 and m1[t, 0] and m1[t, 1]
    assert s1[t, 0] == 1.0 and s1[t, 1] == -1.0


def test_missing_input_raises():
    dates, ea, ret, p, t = base_case()
    v = make_view(dates, ea, ret)
    v.extras.pop("ea_reaction_day")
    with pytest.raises(KeyError):
        target_weights(v, PARAMS)
