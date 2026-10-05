import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS, regime_series, card_universe, triggers, _weekday


def _dates(T, start_offset=0):
    # day numbers counted from the epoch; offset 4 makes row 0 a Monday (epoch day 0 is a Thursday)
    return (np.arange(T) + 4 + start_offset).astype("datetime64[D]")


def _view(close, volume=None, names=None, universe=None, dates=None):
    T, N = close.shape
    names = names or tuple(["BTCUSDT"] + ["P%02dUSDT" % i for i in range(1, N)])
    if volume is None:
        volume = np.ones((T, N))
    if dates is None:
        dates = _dates(T)
    prev = np.vstack([close[:1], close[:-1]])
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.nan_to_num(close / prev - 1.0)
    listed = ~np.isnan(close)
    return DataView(
        dates=dates,
        instruments=names,
        asset_class=tuple(["crypto_spot"] * N),
        ret_co=np.zeros((T, N)),
        ret_oc=r,
        tradable=listed.copy(),
        listed=listed,
        delisting=np.zeros((T, N), dtype=bool),
        close=close,
        dollar_volume=volume,
        universe=universe if universe is not None else np.ones((T, N), dtype=bool),
        extras={},
        series={},
        cash_ret=np.zeros(T),
    )


def test_weekday():
    d = _dates(14)
    assert list(_weekday(d)) == [0, 1, 2, 3, 4, 5, 6] * 2


def test_regime_sunday_close_vs_sma():
    T = 7 * 6
    dates = _dates(T)                       # row 6, 13, 20, ... are Sundays
    c = np.full(T, 100.0)
    sundays = np.arange(6, T, 7)            # 6 Sundays
    c[sundays] = [100, 100, 100, 90, 120, 110]
    reg = regime_series(c, dates, 3)
    # weekly closes: 100,100,100,90,120,110; SMA3 from 3rd: 100, 96.67, 103.3, 106.67
    on_at = {6: False, 13: False, 20: False, 27: False, 34: True, 41: True}
    for s, v in on_at.items():
        assert reg[s] == v
    # regime set at Sunday 34 applies from Sunday 34 to Saturday 40
    assert reg[34:41].all() and not reg[27:34].any()
    # Monday after a Sunday uses that Sunday's regime; before the first Sunday it is OFF
    assert reg[35] and not reg[:6].any()
    # regime_weeks changes the result: with 2 weeks, Sunday 27 (90 vs 95) OFF, Sunday 41 (110 vs 115) OFF
    reg2 = regime_series(c, dates, 2)
    assert reg2[34] and not reg2[41]
    assert not np.array_equal(reg2, reg)


def test_regime_five_weeks():
    T = 7 * 6
    dates = _dates(T)
    c = np.full(T, 100.0)
    c[np.arange(6, T, 7)] = [100, 100, 100, 90, 120, 110]
    reg5 = regime_series(c, dates, 5)
    assert not reg5[:34].any()              # fewer than 5 weekly closes -> OFF
    assert reg5[34]                         # 120 > 102
    assert reg5[41]                         # 110 > mean(100,100,90,120,110)=104


def test_universe_history_and_rank():
    T, N = 70, 3
    close = np.ones((T, N))
    close[:20, 2] = np.nan                  # pair 2 listed late
    vol = np.ones((T, N)) * np.array([1.0, 2.0, 3.0])
    u = card_universe(close, vol)
    assert not u[58].any()                  # 59 closes: not enough history
    assert u[59, 0] and u[59, 1] and not u[59, 2]
    assert not u[69, 2]                     # pair 2 has only 50 closes at row 69
    # rank cut: with 60 pairs qualifying, only the 50 highest-volume ones are members
    T2, N2 = 61, 60
    vol2 = np.tile(np.arange(N2, dtype=float), (T2, 1))
    u2 = card_universe(np.ones((T2, N2)), vol2)
    assert u2[60].sum() == 50 and u2[60, 10:].all() and not u2[60, :10].any()
    # volume is the mean over the last 30 days only: a volume spike 40 days ago does not count
    vol3 = np.tile(np.arange(N2, dtype=float), (T2, 1))
    vol3[20, 0] = 1e9
    assert not card_universe(np.ones((T2, N2)), vol3)[60, 0]
    vol3[31, 0] = 1e9
    assert card_universe(np.ones((T2, N2)), vol3)[60, 0]


def test_triggers_hand_example():
    T = 40
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, T)
    r[35] = -0.10
    close = 100 * np.exp(np.cumsum(r))[:, None]
    trig = triggers(close, 2.0)
    assert trig[35, 0]
    assert not trig[:31].any()              # fewer than 30 return observations before row 31
    # hand check of the threshold: return of row t vs std (ddof 1) of the returns of rows t-30..t-1
    lr = np.log(close[1:, 0] / close[:-1, 0])   # lr[j] is the return of row j+1
    for t in range(31, T):
        sig = np.std(lr[t - 31:t - 1], ddof=1)
        assert trig[t, 0] == (lr[t - 1] <= -2.0 * sig)
    # today's return is excluded from sigma: the drop itself would inflate it
    # a missing close on t-1 suppresses the trigger
    c2 = close.copy()
    c2[34, 0] = np.nan
    assert not triggers(c2, 2.0)[35, 0]


def _scenario(T=7 * 30, drop_row=None):
    """BTC rising steadily (regime ON after 20 weeks), 5 other pairs with small noise."""
    rng = np.random.default_rng(1)
    N = 6
    r = rng.normal(0, 0.01, (T, N))
    r[:, 0] = 0.003 + rng.normal(0, 0.001, T)
    close = 100 * np.exp(np.cumsum(r, axis=0))
    return close


def test_positions_hold_and_weights():
    T = 7 * 30
    close = _scenario(T)
    t0 = 7 * 25 + 2
    close[t0:, 1] *= 0.7                   # extreme drop of pair 1 on row t0
    v = _view(close)
    w = target_weights(v, PARAMS)
    assert w.shape == (T, 6)
    assert w[t0, 1] == pytest.approx(0.25)
    assert w[t0 + 1, 1] == pytest.approx(0.25) and w[t0 + 2, 1] == pytest.approx(0.25)
    assert w[t0 + 3, 1] == 0.0             # closed at the open of t0+1+3
    assert w[t0 - 1, 1] == 0.0
    # hold_days changes the holding
    p = dict(PARAMS, hold_days=5)
    w5 = target_weights(v, p)
    assert w5[t0 + 4, 1] == pytest.approx(0.25) and w5[t0 + 5, 1] == 0.0
    # retrigger resets exit
    close2 = close.copy()
    close2[t0 + 2:, 1] *= 0.7
    w2 = target_weights(_view(close2), PARAMS)
    assert w2[t0 + 4, 1] == pytest.approx(0.25) and w2[t0 + 5, 1] == 0.0
    assert (np.abs(w).sum(axis=1) <= 1 + 1e-12).all() and (w >= 0).all()


def test_regime_off_blocks_trigger():
    T = 7 * 30
    close = _scenario(T)
    close[:, 0] = 100 * np.exp(-0.003 * np.arange(T))    # BTC falling: regime OFF
    t0 = 7 * 25 + 2
    close[t0:, 1] *= 0.7
    w = target_weights(_view(close), PARAMS)
    assert not w.any()


def test_k_sigma_matters_and_weight_split():
    T = 7 * 30
    close = _scenario(T)
    t0 = 7 * 25 + 2
    for j in range(1, 6):
        close[t0:, j] *= 0.7                # 5 pairs trigger together
    w = target_weights(_view(close), PARAMS)
    assert np.allclose(w[t0, 1:], 0.2) and np.isclose(w[t0].sum(), 1.0)
    # small drop: triggers at k=1.5 but not at very large k
    # pair 1: returns alternate +-0.01 (sigma ~ 0.0102), then -0.025 on row t0 (-2.5 sigma)
    close3 = _scenario(T)
    r1 = np.where(np.arange(T) % 2 == 0, 0.01, -0.01)
    r1[0] = 0.0
    r1[t0] = -0.025
    r1[t0 + 1:] = 0.0
    close3[:, 1] = 100 * np.exp(np.cumsum(r1))
    wa = target_weights(_view(close3), dict(PARAMS, k_sigma=1.5))
    wb = target_weights(_view(close3), dict(PARAMS, k_sigma=3.0))
    assert wa[t0, 1] > 0 and wb[t0, 1] == 0


def test_universe_membership_respected():
    T = 7 * 30
    close = _scenario(T)
    t0 = 7 * 25 + 2
    close[t0:, 1] *= 0.7
    uni = np.ones((T, 6), dtype=bool)
    uni[:, 1] = False
    w = target_weights(_view(close, universe=uni), PARAMS)
    assert not w[:, 1].any()


def test_regime_uses_btcusdt_column_by_name():
    # same data, BTCUSDT moved to another column: identical weights per instrument
    T = 7 * 30
    close = _scenario(T)
    t0 = 7 * 25 + 2
    close[t0:, 1] *= 0.7
    names = ("BTCUSDT", "P01USDT", "P02USDT", "P03USDT", "P04USDT", "P05USDT")
    w = target_weights(_view(close, names=names), PARAMS)
    perm = [3, 1, 2, 0, 4, 5]
    w2 = target_weights(_view(close[:, perm], names=tuple(names[i] for i in perm)), PARAMS)
    assert np.array_equal(w[:, perm], w2)
    assert np.array_equal(w, target_weights(_view(close, names=names), PARAMS))


def test_late_listing_and_short_history():
    T = 7 * 30
    close = _scenario(T)
    t0 = 7 * 25 + 2
    close[: t0 - 40, 2] = np.nan            # pair 2 listed 40 days before its drop: < 60 days history
    close[t0:, 2] *= 0.7
    w = target_weights(_view(close), PARAMS)
    assert not w[:, 2].any()
    assert np.isfinite(w).all()
