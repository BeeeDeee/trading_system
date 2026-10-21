import numpy as np
import pytest

from strategy import target_weights, PARAMS

INSTR = ("IEF", "SHY", "SHV", "BIL")
T0 = 400


class FakeView:
    def __init__(self, ret_oc_ief, dgs10, close_ief=None, series_key="fred_macro.DGS10"):
        T = len(ret_oc_ief)
        N = len(INSTR)
        self.dates = np.arange(T).astype("datetime64[D]")
        self.instruments = INSTR
        self.asset_class = ("us_etf",) * N
        self.ret_co = np.zeros((T, N))
        self.ret_oc = np.zeros((T, N))
        self.ret_oc[:, 0] = ret_oc_ief
        self.close = np.full((T, N), 100.0)
        if close_ief is not None:
            self.close[:, 0] = close_ief
        self.tradable = np.isfinite(self.close)
        self.listed = np.ones((T, N), dtype=bool)
        self.universe = None
        self.series = {series_key: np.asarray(dgs10, dtype=float)}
        self.extras = {}

    def col(self, name):
        return self.instruments.index(name)


def base_returns(T=T0):
    return 0.001 * np.where(np.arange(T) % 2 == 0, 1.0, -1.0)


def flat_yield(T=T0):
    return np.full(T, 2.0)


def ief(w):
    return w[:, 0]


def test_rally_event_holds_hold_days_rows():
    r = base_returns()
    r[300] = 0.01
    w = target_weights(FakeView(r, flat_yield()), PARAMS)
    assert w.shape == (T0, 4)
    assert np.all(ief(w)[:300] == 0)
    assert np.all(ief(w)[300:305] == 1.0)
    assert np.all(ief(w)[305:] == 0)
    assert np.all(w[:, 1:] == 0)
    assert np.all(np.abs(w).sum(axis=1) <= 1.0 + 1e-12)


def test_selloff_event_is_short():
    r = base_returns()
    r[300] = -0.01
    w = target_weights(FakeView(r, flat_yield()), PARAMS)
    assert np.all(ief(w)[300:305] == -1.0)
    assert ief(w)[305] == 0


def test_new_event_resets_side_and_exit():
    r = base_returns()
    r[300] = 0.01
    r[302] = -0.02
    w = target_weights(FakeView(r, flat_yield()), PARAMS)
    assert np.all(ief(w)[300:302] == 1.0)
    assert np.all(ief(w)[302:307] == -1.0)
    assert ief(w)[307] == 0


def test_hold_days_param():
    r = base_returns()
    r[300] = 0.01
    p = dict(PARAMS, hold_days=3)
    w = target_weights(FakeView(r, flat_yield()), p)
    assert np.all(ief(w)[300:303] == 1.0)
    assert ief(w)[303] == 0


def test_k_param():
    r = base_returns()
    r[300] = 0.0014  # sd ~ 0.001: above 1.25 sd, below 1.5 sd
    v = FakeView(r, flat_yield())
    assert ief(target_weights(v, PARAMS))[300] == 1.0
    assert ief(target_weights(v, dict(PARAMS, k=1.5)))[300] == 0


def test_band_param_and_state_off():
    r = base_returns()
    r[300] = 0.01
    y = 1.0 + 0.004 * np.arange(T0)  # y - ymin over 252 rows = 0.004 * 251 = 1.004
    v = FakeView(r, y)
    assert ief(target_weights(v, PARAMS))[300] == 0
    assert ief(target_weights(v, dict(PARAMS, band_pp=1.0)))[300] == 0
    assert ief(target_weights(v, dict(PARAMS, band_pp=1.01)))[300] == 1.0
    y2 = 1.0 + 0.0025 * np.arange(T0)  # 0.6275 <= 0.75 but > 0.5
    v2 = FakeView(r, y2)
    assert ief(target_weights(v2, PARAMS))[300] == 1.0
    assert ief(target_weights(v2, dict(PARAMS, band_pp=0.5)))[300] == 0


def test_ymin_includes_current_value():
    r = base_returns()
    r[300] = 0.01
    y = flat_yield()
    y[300] = 1.0  # new low on t: y - ymin = 0 -> ON
    assert ief(target_weights(FakeView(r, y), PARAMS))[300] == 1.0
    y[300] = 3.0  # 1.0 above the 252-row minimum of 2.0 -> OFF
    assert ief(target_weights(FakeView(r, y), PARAMS))[300] == 0


def test_sd_warmup_needs_50_returns():
    r = base_returns()
    r[240] = 0.01  # 40 valid returns before it (rows 200..239)
    r[260] = 0.01  # 60 valid returns before it
    close = np.full(T0, 100.0)
    close[:200] = np.nan  # IEF has no bars before row 200 (late listing)
    w = target_weights(FakeView(r, flat_yield(), close_ief=close), PARAMS)
    assert np.all(ief(w)[:260] == 0)
    assert np.all(ief(w)[260:265] == 1.0)


def test_yield_warmup_needs_200_observations():
    r = base_returns()
    r[150] = 0.01
    r[260] = 0.01
    y = flat_yield()
    y[:100] = np.nan  # at row 150 only 51 valid values, at 260 there are 160 (window 9..260)
    w = target_weights(FakeView(r, y), PARAMS)
    assert ief(w)[150] == 0
    assert ief(w)[260] == 0
    y2 = flat_yield()
    y2[:50] = np.nan  # at 260: window 9..260 has 211 valid
    w2 = target_weights(FakeView(r, y2), PARAMS)
    assert ief(w2)[260] == 1.0


def test_forward_fill_limit_five():
    r = base_returns()
    r[300] = 0.01
    y = flat_yield()
    y[295:301] = np.nan  # 6 missing incl. t -> stale > 5 -> OFF
    assert ief(target_weights(FakeView(r, y), PARAMS))[300] == 0
    y = flat_yield()
    y[296:301] = np.nan  # 5 missing -> filled -> ON
    assert ief(target_weights(FakeView(r, y), PARAMS))[300] == 1.0


def test_missing_bar_no_event():
    r = base_returns()
    r[300] = 0.01
    close = np.full(T0, 100.0)
    close[300] = np.nan
    w = target_weights(FakeView(r, flat_yield(), close_ief=close), PARAMS)
    assert np.all(ief(w) == 0)


def test_missing_series_raises():
    with pytest.raises(KeyError):
        target_weights(FakeView(base_returns(), flat_yield(), series_key="other.DGS10"), PARAMS)


def test_point_in_time_truncation():
    r = base_returns()
    r[300] = 0.01
    r[320] = -0.02
    full = target_weights(FakeView(r, flat_yield()), PARAMS)
    for cut in (250, 301, 303, 321, 330):
        part = target_weights(FakeView(r[:cut], flat_yield()[:cut]), PARAMS)
        assert np.array_equal(part, full[:cut])

