import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS, leg_state, switch_rows, min_obs, _holidays


def ymd(y_since_1970, month, day):
    m = (np.array([y_since_1970]).astype("datetime64[Y]").astype("datetime64[M]")
         + np.array([month - 1]).astype("timedelta64[M]"))
    return (m.astype("datetime64[D]") + np.array([day - 1]).astype("timedelta64[D]"))[0]


def year_start(y_since_1970):
    return np.array([y_since_1970]).astype("datetime64[Y]").astype("datetime64[D]")[0]


def sessions(y_since_1970, n, holidays=None):
    start = year_start(y_since_1970)
    days = start + np.arange(int(n * 1.6) + 30)
    hol = holidays if holidays is not None else np.array([], dtype="datetime64[D]")
    keep = np.is_busday(days, holidays=hol)
    return days[keep][:n]


def make_data(dates, R, universe=None, listed=None, tradable=None):
    T, N = R.shape
    if listed is None:
        listed = np.ones((T, N), dtype=bool)
    if tradable is None:
        tradable = listed.copy()
    if universe is None:
        universe = listed.copy()
    elif universe is False:
        universe = None
    R = np.where(listed, R, 0.0)
    close = np.where(listed, 10.0 * np.cumprod(1.0 + R, axis=0), np.nan)
    insts = tuple("S%02d" % j if j < 100 else "S%03d" % j for j in range(N))
    return DataView(
        dates=dates,
        instruments=insts,
        asset_class=tuple("us_equity" for _ in range(N)),
        ret_co=np.zeros((T, N), dtype=np.float32),
        ret_oc=R.astype(np.float32),
        tradable=tradable,
        listed=listed,
        delisting=np.zeros((T, N), dtype=bool),
        close=close.astype(np.float32),
        dollar_volume=np.where(listed, 1e8, np.nan).astype(np.float32),
        universe=universe,
        extras={},
        series={},
        cash_ret=np.zeros(T),
    )


def factor_market(T, N, seed=0):
    """r_i = b_i * f: with no noise beta_i = b_i / mean(b) exactly, so the ranking is by b."""
    rng = np.random.default_rng(seed)
    f = rng.normal(0, 0.01, T)
    b = np.linspace(0.2, 2.0, N)
    perm = rng.permutation(N)
    b = b[perm]
    return b, f[:, None] * b[None, :]


# ------------------------------------------------------------------------------ calendar
def test_holidays_known():
    hol = _holidays(sessions(54, 5))                          # year 1970 + 54
    assert ymd(54, 3, 29) in hol          # Good Friday
    assert ymd(54, 5, 27) in hol          # Memorial Day
    assert ymd(54, 11, 28) in hol         # Thanksgiving
    hol51 = _holidays(sessions(51, 5))
    assert ymd(51, 4, 2) in hol51 and ymd(51, 5, 31) in hol51 and ymd(51, 11, 25) in hol51


def test_switch_dates_known_year():
    Y = 54
    hol = _holidays(sessions(Y, 5))
    d = sessions(Y, 250, holidays=hol)                        # Good Friday etc. removed
    sw = switch_rows(d, 3)
    days = [(d[t], a) for t, a in sw]
    # Jan 31 (Wed) is day -1 -> switch A after Jan 30; Feb 5 is the 3rd session of Feb -> switch B
    assert (ymd(Y, 1, 30), True) in days
    assert (ymd(Y, 2, 5), False) in days
    # Feb 29 (Thu) is day -1 -> A after Feb 28; March sessions 1, 4, 5 -> B after Mar 5
    assert (ymd(Y, 2, 28), True) in days
    assert (ymd(Y, 3, 5), False) in days
    # Good Friday Mar 29: day -1 is Thu Mar 28 -> A after Wed Mar 27; B after Apr 3
    assert (ymd(Y, 3, 27), True) in days
    assert (ymd(Y, 4, 3), False) in days
    # Thanksgiving Nov 28 (Thu): day -1 is Fri Nov 29 -> A after Wed Nov 27
    assert (ymd(Y, 11, 27), True) in days
    # exactly two switches per month
    assert all(sum(1 for t, a in sw if d[t].astype("datetime64[M]") == mo) == 2
               for mo in np.unique(d[30:-30].astype("datetime64[M]")))
    # post_days = 2 moves switch B one session earlier, switch A unchanged
    days2 = [(d[t], a) for t, a in switch_rows(d, 2)]
    assert (ymd(Y, 2, 2), False) in days2 and (ymd(Y, 1, 30), True) in days2
    days4 = [(d[t], a) for t, a in switch_rows(d, 4)]
    assert (ymd(Y, 2, 6), False) in days4


def test_memorial_day_month_end():
    Y = 51                                                    # May 31 is Memorial Day (Monday)
    hol = _holidays(sessions(Y, 5))
    d = sessions(Y, 200, holidays=hol)
    days = [(d[t], a) for t, a in switch_rows(d, 3)]
    assert (ymd(Y, 5, 27), True) in days                      # day -1 = Fri May 28


def test_leg_state_point_in_time():
    hol = _holidays(sessions(54, 5))
    d = sessions(54, 120, holidays=hol)
    full = leg_state(d, 3)
    for cut in (20, 41, 62, 63, 64, 90):
        assert np.array_equal(leg_state(d[: cut + 1], 3), full[: cut + 1])


# ------------------------------------------------------------------------------ weights
def test_legs_and_weights():
    T, N = 330, 120
    d = sessions(53, T)
    b, R = factor_market(T, N)
    data = make_data(d, R)
    W = target_weights(data, PARAMS)
    sw = switch_rows(d, PARAMS["post_days"])
    first = [(t, a) for t, a in sw if t >= 189][:4]
    nleg = 24                                                 # round(0.2 * 120)
    top = set(np.argsort(-b)[:nleg])
    bot = set(np.argsort(b)[:nleg])
    for t, a in first:
        row = W[t]
        assert np.all(np.isfinite(row))
        held = set(np.nonzero(row > 0)[0])
        assert held == (top if a else bot)
        assert np.allclose(row[list(held)], 1.0 / nleg)
        assert row.sum() <= 1.0 + 1e-9 and row.min() >= 0.0
    # rows between switches are "hold" (NaN)
    t0, t1 = first[0][0], first[1][0]
    assert np.all(np.isnan(W[t0 + 1:t1]))
    # before enough history: no decision at all
    assert np.all(np.isnan(W[:188]))


def test_min_leg_size_and_quantile_param():
    T, N = 330, 60
    d = sessions(53, T)
    b, R = factor_market(T, N, seed=3)
    data = make_data(d, R)
    W = target_weights(data, PARAMS)                          # round(0.2*60)=12 -> at least 20
    t = [t for t, a in switch_rows(d, 3) if t >= 189][0]
    assert (W[t] > 0).sum() == 20
    W3 = target_weights(data, dict(PARAMS, leg_quantile=0.4))  # 24 names
    assert (W3[t] > 0).sum() == 24


def test_eligibility_late_listing_and_lookback():
    T, N = 400, 100
    d = sessions(53, T)
    b, R = factor_market(T, N, seed=1)
    listed = np.ones((T, N), dtype=bool)
    hi = int(np.argmax(b))
    listed[:150, hi] = False                                  # highest beta lists late
    data = make_data(d, R, listed=listed)
    sw = [(t, a) for t, a in switch_rows(d, 3) if a]
    W = target_weights(data, PARAMS)
    # valid returns of hi start at row 151; needs 189 -> eligible from row 339 on
    for t, a in sw:
        if np.isfinite(W[t]).all() and t < 339:
            assert W[t, hi] == 0.0
        if t >= 339:
            assert W[t, hi] > 0.0
    # beta_lookback 126 needs ceil(0.75*126)=95 returns in a 126-row window
    assert min_obs(126) == 95 and min_obs(252) == 189 and min_obs(504) == 189
    W126 = target_weights(data, dict(PARAMS, beta_lookback=126))
    early = [t for t, a in sw if 151 + 95 - 1 <= t < 339]
    assert early and all(W126[t, hi] > 0 for t in early)


def test_exit_when_leaving_universe():
    T, N = 330, 120
    d = sessions(53, T)
    b, R = factor_market(T, N, seed=2)
    uni = np.ones((T, N), dtype=bool)
    data0 = make_data(d, R, universe=uni.copy())
    W0 = target_weights(data0, PARAMS)
    sw = [(t, a) for t, a in switch_rows(d, 3) if t >= 189]
    t0, t1 = sw[0][0], sw[1][0]
    held = np.nonzero(W0[t0] > 0)[0]
    j = held[0]
    uni[t0 + 2, j] = False                                    # leaves for one day, then returns
    data = make_data(d, R, universe=uni)
    W = target_weights(data, PARAMS)
    assert np.all(np.isnan(W[t0 + 1, :]))
    assert np.all(W[t0 + 2:t1, j] == 0.0)                     # sold, stays in cash until next switch
    others = np.delete(np.arange(N), j)
    assert np.all(np.isnan(W[t0 + 2:t1][:, others]))
    assert np.array_equal(np.isfinite(W[t1]), np.ones(N, dtype=bool))


def test_untradable_holding_sold():
    T, N = 330, 120
    d = sessions(53, T)
    b, R = factor_market(T, N, seed=2)
    data0 = make_data(d, R)
    W0 = target_weights(data0, PARAMS)
    sw = [(t, a) for t, a in switch_rows(d, 3) if t >= 189]
    t0, t1 = sw[0][0], sw[1][0]
    j = np.nonzero(W0[t0] > 0)[0][1]
    tr = np.ones((T, N), dtype=bool)
    tr[t0 + 3, j] = False
    W = target_weights(make_data(d, R, tradable=tr), PARAMS)
    assert np.all(W[t0 + 3:t1, j] == 0.0)


def test_truncation_invariance_and_determinism():
    T, N = 360, 80
    d = sessions(53, T)
    rng = np.random.default_rng(7)
    R = rng.normal(0, 0.02, (T, N)) + rng.normal(0, 0.01, (T, 1)) * rng.uniform(0.3, 2, N)
    data = make_data(d, R)
    full = target_weights(data, PARAMS)
    assert np.array_equal(full, target_weights(data, PARAMS), equal_nan=True)
    for cut in (200, 230, 260, 300):
        part = make_data(d[: cut + 1], R[: cut + 1])
        w = target_weights(part, PARAMS)
        assert np.array_equal(w, full[: cut + 1], equal_nan=True)


def test_missing_universe_raises():
    T, N = 50, 5
    d = sessions(53, T)
    data = make_data(d, np.zeros((T, N)), universe=False)
    with pytest.raises(Exception):
        target_weights(data, PARAMS)


def test_rotation_alternates_top_and_bottom():
    T, N = 420, 120
    d = sessions(53, T)
    b, R = factor_market(T, N, seed=5)
    W = target_weights(make_data(d, R), PARAMS)
    top = set(np.argsort(-b)[:24])
    bot = set(np.argsort(b)[:24])
    for t, a in switch_rows(d, 3):
        if t < 189:
            continue
        assert set(np.nonzero(W[t] > 0)[0]) == (top if a else bot)
        assert abs(W[t].sum() - 1.0) < 1e-9
