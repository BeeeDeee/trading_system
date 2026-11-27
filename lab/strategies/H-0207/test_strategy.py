import numpy as np
import pytest

from strategy import target_weights, PARAMS, yield_changes, rolling_scale, release_session, event_side, book_side
from lab.framework.data import DataView

T = 320
NAMES = ("FXE", "FXY", "FXB", "FXA", "FXC", "FXF", "SHY", "SHV", "BIL")


def base_yield(T=T):
    # alternating +/- 1 bp changes: trailing std ~ 0.01
    steps = np.where(np.arange(T) % 2 == 0, 0.01, -0.01)
    steps[0] = 0.0
    return 2.0 + np.cumsum(steps)


def no_cal(T=T):
    return [np.full(T, 5.0), np.full(T, 7.0), np.full(T, 9.0)]


def with_jump(v, row, size):
    v = v.copy()
    v[row:] += size
    return v


def test_yield_changes_skip_missing():
    v = np.array([np.nan, 1.0, 1.1, np.nan, 1.3])
    dy = yield_changes(v)
    assert np.isnan(dy[0]) and np.isnan(dy[1]) and np.isnan(dy[3])
    assert dy[2] == pytest.approx(0.1)
    assert dy[4] == pytest.approx(0.2)  # vs last valid observation


def test_rolling_scale_min_obs_and_window():
    dy = np.arange(400, dtype=float)
    s = rolling_scale(dy, window=252, min_obs=126)
    assert np.all(np.isnan(s[:125])) and np.isfinite(s[125])
    assert s[399] == pytest.approx(np.std(dy[148:400], ddof=1))
    dy2 = dy.copy()
    dy2[::2] = np.nan
    s2 = rolling_scale(dy2, window=252, min_obs=126)
    assert np.isnan(s2[250])  # only 125 valid changes
    assert np.isfinite(s2[251])


def test_release_session_uses_previous_row():
    cal = no_cal(10)
    cal[1][3] = 1.0
    rel = release_session(cal)
    assert rel.tolist() == [i == 4 for i in range(10)]


def test_event_timing_and_side():
    v = with_jump(base_yield(), 200, 0.05)   # row 200 holds DGS2(d=199), a +5 bp move
    cal = no_cal()
    cal[2][198] = 1.0                         # FOMC is the next session (199) as of 198
    side = event_side(v, cal, 1.0)
    assert side[200] == -1.0
    assert np.count_nonzero(side) == 1
    # without the release flag there is no event
    assert np.count_nonzero(event_side(v, no_cal(), 1.0)) == 0
    # calendar one row off: no event
    cal2 = no_cal()
    cal2[2][199] = 1.0
    assert np.count_nonzero(event_side(v, cal2, 1.0)) == 0
    # yield fall -> long
    vd = with_jump(base_yield(), 200, -0.05)
    assert event_side(vd, cal, 1.0)[200] == 1.0


def test_warmup_no_signal():
    v = with_jump(base_yield(), 100, 0.05)
    cal = no_cal()
    cal[0][98] = 1.0
    assert np.count_nonzero(event_side(v, cal, 1.0)) == 0


def test_missing_yield_no_news():
    v = with_jump(base_yield(), 200, 0.05)
    v[200] = np.nan
    cal = no_cal()
    cal[0][198] = 1.0
    assert np.count_nonzero(event_side(v, cal, 1.0)) == 0


def test_threshold_matters():
    v = with_jump(base_yield(), 200, 0.002)  # +1 bp base step + 0.2 bp -> dy = 1.2 bp, z ~ 1.2
    cal = no_cal()
    cal[1][198] = 1.0
    dy = yield_changes(v)
    z = dy[200] / rolling_scale(dy)[200]
    assert 1.0 < abs(z) < 1.5
    assert np.count_nonzero(event_side(v, cal, 1.0)) == 1
    assert np.count_nonzero(event_side(v, cal, 1.5)) == 0
    assert np.count_nonzero(event_side(v, cal, 0.5)) >= 1


def test_book_hold_replace_extend():
    s = np.zeros(30)
    s[5] = -1
    b = book_side(s, 5)
    assert b[5:10].tolist() == [-1] * 5 and b[10] == 0 and b[4] == 0
    s[7] = 1
    b = book_side(s, 5)
    assert b[5:7].tolist() == [-1, -1] and b[7:12].tolist() == [1] * 5 and b[12] == 0
    s2 = np.zeros(30)
    s2[5] = -1
    s2[8] = -1
    b2 = book_side(s2, 3)
    assert b2[5:11].tolist() == [-1] * 6 and b2[11] == 0
    assert np.count_nonzero(book_side(s2, 7)) != np.count_nonzero(b2)


def make_view(v, cal, close=None):
    N = len(NAMES)
    dates = np.arange(T).astype("datetime64[D]") + np.timedelta64(14000, "D")
    z = np.zeros((T, N))
    ones = np.ones((T, N), dtype=bool)
    if close is None:
        close = np.full((T, N), 10.0)
    series = {"fred_macro.DGS2": v}
    for k, c in zip(("next_cpi_sessions", "next_empsit_sessions", "next_fomc_sessions"), cal):
        series["macro_release_calendar_1d." + k] = c
    series["macro_release_calendar_1d.next_release_sessions"] = np.minimum(np.minimum(cal[0], cal[1]), cal[2])
    return DataView(
        dates=dates, instruments=NAMES, asset_class=("us_etf",) * N,
        ret_co=z, ret_oc=z.copy(), tradable=ones, listed=ones.copy(),
        delisting=np.zeros((T, N), dtype=bool), close=close, dollar_volume=np.full((T, N), 1e7),
        universe=None, extras={}, series=series, cash_ret=np.zeros(T),
    )


def test_target_weights_book():
    v = with_jump(base_yield(), 200, 0.05)
    cal = no_cal()
    cal[0][198] = 1.0
    close = np.full((T, len(NAMES)), 10.0)
    close[202, 1] = np.nan                     # FXY without a price on row 202
    w = target_weights(make_view(v, cal, close), PARAMS)
    assert w.shape == (T, len(NAMES))
    H = PARAMS["hold_days"]
    assert np.all(w[:200] == 0)
    for t in range(200, 200 + H):
        for j in range(6):
            exp = 0.0 if (t == 202 and j == 1) else -1.0 / 6
            assert w[t, j] == pytest.approx(exp)
    assert np.all(w[200 + H:] == 0)
    assert np.all(w[:, 6:] == 0)               # SHY, SHV, BIL never traded
    assert np.all(np.abs(w).sum(axis=1) <= 1 + 1e-12)
    w3 = target_weights(make_view(v, cal), {**PARAMS, "hold_days": 3})
    assert np.count_nonzero(w3[:, 0]) == 3


def test_missing_series_raises():
    view = make_view(base_yield(), no_cal())
    del view.series["fred_macro.DGS2"]
    with pytest.raises(KeyError):
        target_weights(view, PARAMS)
