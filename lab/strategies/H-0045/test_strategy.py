import numpy as np
import pytest

from strategy import target_weights, PARAMS, cycle_week_ahead, sessions_since_fomc, SERIES_KEY, events
from lab.framework.data import DataView

# Hand-made schedule: statement days at rows 5, 40 and 80, T = 85.
T = 85
FOMC = [5, 40, 80, 120]   # 120 lies after the last row: the schedule is pre-announced


def schedule():
    n = np.full(T, np.nan)
    for d in range(T):
        nxt = [f for f in FOMC if f > d]
        n[d] = (nxt[0] - d) if nxt else np.nan
    return n


def expected_week():
    """week(d) by hand (NaN = none)."""
    w = np.full(T, np.nan)
    w[5:9] = 0          # k 0..3
    w[9:14] = 1
    w[14:19] = 2
    w[19:24] = 3
    w[24:29] = 4
    w[29:34] = 5
    w[34:39] = 6        # k 29..33
    w[39] = 0           # day before statement 40
    w[40:44] = 0
    w[44:49] = 1
    w[49:54] = 2
    w[54:59] = 3
    w[59:64] = 4
    w[64:69] = 5
    w[69:74] = 6
    # 74..78: k 34..38 -> none
    w[79] = 0           # day before statement 80
    w[80:84] = 0
    w[84] = 1           # k = 4
    return w


def make_view(n, close_nan_rows=()):
    names = ("SPY", "IEF", "TLT", "SHY", "LQD")
    N = len(names)
    dates = np.arange(T).astype("datetime64[D]") + np.timedelta64(12000, "D")
    close = np.full((T, N), 100.0)
    for r in close_nan_rows:
        close[r, 0] = np.nan
    z = np.zeros((T, N))
    return DataView(
        dates=dates,
        instruments=names,
        asset_class=tuple("us_etf" for _ in names),
        ret_co=z.copy(),
        ret_oc=z.copy(),
        tradable=np.ones((T, N), dtype=bool),
        listed=np.ones((T, N), dtype=bool),
        delisting=np.zeros((T, N), dtype=bool),
        close=close,
        dollar_volume=np.full((T, N), 1e9),
        universe=None,
        extras={},
        series={SERIES_KEY: n},
        cash_ret=np.zeros(T),
    )


def test_k_counts_from_previous_day_signal():
    k = sessions_since_fomc(schedule())
    assert np.all(np.isnan(k[:5]))
    assert k[5] == 0 and k[39] == 34 and k[40] == 0 and k[84] == 4


def test_week_today_matches_hand():
    w = cycle_week_ahead(schedule(), 0)
    np.testing.assert_array_equal(w, expected_week())


@pytest.mark.parametrize("h", [1, 2])
def test_week_ahead_equals_future_week(h):
    n = schedule()
    w0 = expected_week()
    wh = cycle_week_ahead(n, h)
    # week(d+h) computed at d from past values only equals the true week h rows later
    # (rows where n is NaN are excluded: unknown schedule)
    for d in range(T - h):
        if np.isfinite(n[d]):
            assert (np.isnan(wh[d]) and np.isnan(w0[d + h])) or wh[d] == w0[d + h], d


def test_weights_primary():
    n = schedule()
    w = target_weights(make_view(n), PARAMS)
    assert w.shape == (T, 5)
    np.testing.assert_allclose(w[0], [0.6, 0.4, 0, 0, 0])      # week none
    np.testing.assert_allclose(w[4], [0.9, 0.1, 0, 0, 0])      # target for row 5 (week 0)
    np.testing.assert_allclose(w[8], [0.3, 0.7, 0, 0, 0])      # target for row 9 (week 1)
    np.testing.assert_allclose(w[13], [0.9, 0.1, 0, 0, 0])     # target for row 14 (week 2)
    np.testing.assert_allclose(w[73], [0.6, 0.4, 0, 0, 0])     # target for row 74 (week none)
    np.testing.assert_allclose(w[78], [0.9, 0.1, 0, 0, 0])     # target for row 79 (day before)
    np.testing.assert_allclose(w[83], [0.3, 0.7, 0, 0, 0])     # target for row 84 (week 1)
    fin = np.isfinite(w).all(axis=1)
    assert np.all(np.abs(w[fin]).sum(axis=1) <= 1 + 1e-12)
    assert np.all(w[fin] >= 0)


def test_params_change_output():
    n = schedule()
    base = target_weights(make_view(n), PARAMS)
    t4 = target_weights(make_view(n), dict(PARAMS, tilt=0.4))
    np.testing.assert_allclose(t4[4], [1.0, 0.0, 0, 0, 0])
    np.testing.assert_allclose(t4[8], [0.2, 0.8, 0, 0, 0])
    om = target_weights(make_view(n), dict(PARAMS, offset_sessions=-1))
    op = target_weights(make_view(n), dict(PARAMS, offset_sessions=1))
    # offset -1: row 8 uses week(8) = 0; offset +1: row 7 uses week(9) = 1
    np.testing.assert_allclose(om[8], [0.9, 0.1, 0, 0, 0])
    np.testing.assert_allclose(op[7], [0.3, 0.7, 0, 0, 0])
    np.testing.assert_allclose(base[7], [0.9, 0.1, 0, 0, 0])


def test_missing_bar_keeps_positions():
    w = target_weights(make_view(schedule(), close_nan_rows=(20,)), PARAMS)
    assert np.all(np.isnan(w[20]))
    assert np.isfinite(w[21]).all()


def test_trailing_missing_schedule_keeps_positions():
    n = schedule()
    n[82:] = np.nan
    w = target_weights(make_view(n), PARAMS)
    assert np.all(np.isnan(w[82:]))
    assert np.isfinite(w[81]).all()


def test_leading_missing_schedule_is_sixty_forty():
    n = schedule()
    n[:3] = np.nan
    w = target_weights(make_view(n), PARAMS)
    np.testing.assert_allclose(w[0], [0.6, 0.4, 0, 0, 0])


def test_internal_gap_raises():
    n = schedule()
    n[30] = np.nan
    with pytest.raises(ValueError):
        target_weights(make_view(n), PARAMS)


def test_missing_series_raises():
    v = make_view(schedule())
    v.series.pop(SERIES_KEY)
    with pytest.raises(KeyError):
        target_weights(v, PARAMS)


def test_point_in_time_truncation():
    n = schedule()
    full = target_weights(make_view(n), PARAMS)
    for cut in (10, 39, 41, 60):
        n2 = n.copy()
        n2[cut + 1:] = np.random.default_rng(cut).integers(1, 40, T - cut - 1)
        part = target_weights(make_view(n2), PARAMS)
        np.testing.assert_array_equal(part[:cut + 1], full[:cut + 1])


def test_events():
    mask, side = events(make_view(schedule()), PARAMS)
    rows = np.flatnonzero(mask[:, 0])
    # week starts at rows 5 (w0), 9, 14, 19, 24, 29, 34, 39 (w0 after w6), 44, ..., 69, 79 (after none)
    exp = [4, 8, 13, 18, 23, 28, 33, 38, 43, 48, 53, 58, 63, 68, 78, 83]
    assert list(rows) == exp
    assert not mask[:, 1:].any()
    assert side[4, 0] == 1 and side[8, 0] == -1 and side[13, 0] == 1 and side[28, 0] == -1
