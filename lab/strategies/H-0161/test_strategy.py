import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS, signal_parts, HOME, TILT

NAMES = ("SPY", "IEF", "EWJ", "EWH", "FXI", "EWY", "EWT", "EWS", "EWM", "EWA",
         "THD", "EIDO", "EPHE", "ENZL", "INDA")
KEYS = sorted(set(HOME.values()))
T = 120


def make(T=T, seed=1):
    rng = np.random.default_rng(seed)
    N = len(NAMES)
    dates = (np.arange(T) + 11000).astype("datetime64[D]")
    ret_oc = rng.normal(0.0, 0.01, (T, N))
    ret_co = np.zeros((T, N))
    tradable = np.ones((T, N), dtype=bool)
    listed = np.ones((T, N), dtype=bool)
    close = np.full((T, N), 50.0)
    series = {k: np.zeros(T) for k in KEYS}
    return dict(dates=dates, instruments=NAMES, asset_class=tuple(["us_etf"] * N),
                ret_co=ret_co, ret_oc=ret_oc, tradable=tradable, listed=listed,
                delisting=np.zeros((T, N), dtype=bool), close=close,
                dollar_volume=np.full((T, N), 1e8), universe=None, extras={},
                series=series, cash_ret=np.zeros(T))


def view(d):
    return DataView(**d)


def c(name):
    return NAMES.index(name)


def test_no_event_is_exact_60_40():
    W = target_weights(view(make()), PARAMS)
    assert W.shape == (T, len(NAMES))
    assert np.allclose(W[:, c("SPY")], 0.6)
    assert np.allclose(W[:, c("IEF")], 0.4)
    assert np.allclose(np.abs(W).sum(1), 1.0)


def test_one_day_holiday_entry_and_exit():
    d = make()
    t = 90
    d["series"]["exchange_holidays_1d.XTKS.closed_next"][t] = 1.0
    d["ret_oc"][t, c("EWJ")] = -0.08
    W = target_weights(view(d), PARAMS)
    j, s = c("EWJ"), c("SPY")
    assert W[t, j] == pytest.approx(0.3)
    assert W[t, s] == pytest.approx(0.3)
    assert W[t, c("IEF")] == pytest.approx(0.4)
    # flag 0 at t+1 -> closed at open of t+2
    assert W[t + 1, j] == 0.0
    assert W[t + 1, s] == pytest.approx(0.6)
    assert W[t - 1, j] == 0.0


def test_slot_weight_param():
    d = make()
    t = 90
    d["series"]["exchange_holidays_1d.XTKS.closed_next"][t] = 1.0
    d["ret_oc"][t, c("EWJ")] = -0.08
    W = target_weights(view(d), {"slot_weight": 0.2, "z_threshold": 1.0})
    assert W[t, c("EWJ")] == pytest.approx(0.2)
    assert W[t, c("SPY")] == pytest.approx(0.4)


def test_z_value_and_threshold():
    d = make()
    t = 90
    d["series"]["exchange_holidays_1d.XTKS.closed_next"][t] = 1.0
    d["ret_oc"][t, c("EWJ")] = d["ret_oc"][t, c("SPY")] - 0.0
    flags, z, n = signal_parts(view(d), PARAMS)
    jj = TILT.index("EWJ")
    dd = np.log1p(d["ret_oc"][:, c("EWJ")]) - np.log1p(d["ret_oc"][:, c("SPY")])
    sig = np.std(dd[t - 63:t], ddof=1)
    expect = dd[t] / sig            # n = 1 (home traded before session t)
    assert n[t, jj] == 1
    assert z[t, jj] == pytest.approx(expect, rel=1e-9)
    # choose a move that gives z = -1.2: enters at threshold 1.0, not at 1.5
    d["ret_oc"][t, c("EWJ")] = np.expm1(np.log1p(d["ret_oc"][t, c("SPY")]) - 1.2 * sig)
    W1 = target_weights(view(d), {"slot_weight": 0.3, "z_threshold": 1.0})
    W2 = target_weights(view(d), {"slot_weight": 0.3, "z_threshold": 1.5})
    assert W1[t, c("EWJ")] == pytest.approx(0.3)
    assert W2[t, c("EWJ")] == 0.0


def test_unanchored_window_length_and_cap():
    d = make()
    f = d["series"]["exchange_holidays_1d.XHKG.closed_next"]
    f[80] = 1.0
    f[81] = 1.0
    f[100:115] = 1.0
    flags, z, n = signal_parts(view(d), PARAMS)
    jj = TILT.index("EWH")
    assert n[80, jj] == 1
    assert n[81, jj] == 2
    assert n[82, jj] == 3       # anchor = most recent s with closed_next(s-1) = 0 -> s = 80
    assert n[83, jj] == 1
    assert n[110, jj] == 10     # capped at 10 sessions
    assert n[114, jj] == 10
    # U over window: ln(1+oc(a)) + sum (co + oc) for s=a+1..t
    t, a = 82, 80
    lj = np.log1p(d["ret_oc"][:, c("EWH")]) + np.log1p(d["ret_co"][:, c("EWH")])
    ls = np.log1p(d["ret_oc"][:, c("SPY")]) + np.log1p(d["ret_co"][:, c("SPY")])
    u = (np.log1p(d["ret_oc"][a, c("EWH")]) + lj[a + 1:t + 1].sum()) - \
        (np.log1p(d["ret_oc"][a, c("SPY")]) + ls[a + 1:t + 1].sum())
    dd = np.log1p(d["ret_oc"][:, c("EWH")]) - np.log1p(d["ret_oc"][:, c("SPY")])
    sig = np.std(dd[t - 63:t], ddof=1)
    assert z[t, jj] == pytest.approx(u / (sig * np.sqrt(3)), rel=1e-9)


def test_max_hold_five_sessions():
    d = make()
    t = 90
    d["series"]["exchange_holidays_1d.XKRX.closed_next"][t:t + 9] = 1.0
    d["ret_oc"][t, c("EWY")] = -0.10
    d["ret_oc"][t + 1:t + 9, c("EWY")] = 0.0
    d["ret_oc"][t + 1:t + 9, c("SPY")] = 0.0
    W = target_weights(view(d), PARAMS)
    j = c("EWY")
    # entry fill at t+1; held sessions t+1..t+5; exit decided at close of t+5
    assert np.allclose(W[t:t + 5, j], 0.3)
    assert W[t + 5, j] == 0.0


def test_nan_flag_no_event_and_history_required():
    d = make()
    d["series"]["exchange_holidays_1d.XTKS.closed_next"][90] = np.nan
    d["ret_oc"][90, c("EWJ")] = -0.10
    d["series"]["exchange_holidays_1d.XTAI.closed_next"][50] = 1.0   # < 63 rows of history
    d["ret_oc"][50, c("EWT")] = -0.10
    W = target_weights(view(d), PARAMS)
    assert W[90, c("EWJ")] == 0.0
    assert W[50, c("EWT")] == 0.0


def test_late_listing_cannot_enter():
    d = make()
    j = c("INDA")
    d["close"][:70, j] = np.nan
    d["ret_oc"][:70, j] = 0.0
    d["tradable"][:70, j] = False
    d["listed"][:70, j] = False
    d["series"]["exchange_holidays_1d.XBOM.closed_next"][100] = 1.0
    d["ret_oc"][100, j] = -0.10
    W = target_weights(view(d), PARAMS)
    assert W[100, j] == 0.0


def test_untradable_next_open_cancels_entry():
    d = make()
    t = 90
    d["series"]["exchange_holidays_1d.XTKS.closed_next"][t:t + 3] = 1.0
    d["ret_oc"][t, c("EWJ")] = -0.08
    d["tradable"][t + 1, c("EWJ")] = False
    W = target_weights(view(d), PARAMS)
    assert W[t, c("EWJ")] == pytest.approx(0.3)
    assert W[t + 1, c("EWJ")] == 0.0 or W[t + 1, c("EWJ")] == pytest.approx(0.3)
    # the t entry never filled, so row t+1 is not a continuation of a held slot
    assert np.abs(W[t + 1]).sum() == pytest.approx(1.0)


def test_cap_scales_slots():
    d = make()
    t = 90
    for key, name in (("XTKS", "EWJ"), ("XKRX", "EWY"), ("XTAI", "EWT")):
        d["series"]["exchange_holidays_1d." + key + ".closed_next"][t] = 1.0
        d["ret_oc"][t, c(name)] = -0.10
    W = target_weights(view(d), PARAMS)
    for name in ("EWJ", "EWY", "EWT"):
        assert W[t, c(name)] == pytest.approx(0.2)
    assert W[t, c("SPY")] == pytest.approx(0.0)
    assert W[t].min() >= 0.0
    assert np.abs(W[t]).sum() <= 1.0 + 1e-9


def test_shared_exchange_both_enter():
    d = make()
    t = 90
    d["series"]["exchange_holidays_1d.XHKG.closed_next"][t] = 1.0
    d["ret_oc"][t, c("EWH")] = -0.10
    d["ret_oc"][t, c("FXI")] = 0.05
    W = target_weights(view(d), PARAMS)
    assert W[t, c("EWH")] == pytest.approx(0.3)
    assert W[t, c("FXI")] == 0.0


def test_point_in_time_truncation():
    d = make()
    d["series"]["exchange_holidays_1d.XTKS.closed_next"][[85, 86, 95]] = 1.0
    d["ret_oc"][[85, 95], c("EWJ")] = -0.08
    W = target_weights(view(d), PARAMS)
    for cut in (70, 86, 90, 100):
        e = make(T=cut + 1)
        for k in e["series"]:
            e["series"][k] = d["series"][k][:cut + 1].copy()
        for k in ("ret_oc", "ret_co", "tradable", "listed", "close", "dollar_volume", "delisting"):
            e[k] = d[k][:cut + 1].copy()
        Wc = target_weights(view(e), PARAMS)
        assert np.array_equal(Wc, W[:cut + 1])
