import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import PARAMS, target_weights, compute_events


def events(data, params):
    # same event code as diagnostic.py (a copy); the tests may only import strategy
    T, N = len(data.dates), len(data.instruments)
    ev, K, cols, uni, jb, je = compute_events(data, params)
    mask = np.zeros((T, N), dtype=bool)
    if len(cols):
        mask[:, cols] = ev
    return mask, np.where(mask, -1.0, 0.0)

COINS = ("ADA", "XRP", "DOT", "LINK")       # perps with an Upbit series and a spot pair
OTHERS = ("BNB", "LTC", "EOS")              # perps without an Upbit series
T0 = 40                                     # default event day


def make_view(T=80, upbit=None, spot=None, ret=None, uni=None, listed=None, spot_coins=COINS,
              upbit_coins=COINS, tradable=None):
    perps = ("BTCUSDT.P", "ETHUSDT.P") + tuple(c + "USDT.P" for c in COINS + OTHERS)
    spots = tuple(c + "USDT" for c in spot_coins)
    instruments = perps + spots
    N = len(instruments)
    asset_class = tuple(["crypto_perp"] * len(perps) + ["crypto_spot"] * len(spots))
    dates = np.arange(18262, 18262 + T).astype("datetime64[D]")
    ret_oc = np.zeros((T, N))
    if ret is not None:
        for name, (t, v) in ret.items():
            ret_oc[t, instruments.index(name)] = v
    lst = np.ones((T, N), dtype=bool) if listed is None else listed
    trd = lst.copy() if tradable is None else tradable
    close = np.ones((T, N))
    dv = np.full((T, N), 1.0e8)
    for c in spot_coins:
        j = instruments.index(c + "USDT")
        dv[:, j] = 1.0e7 if spot is None or c not in spot else spot[c]
    series = {}
    for c in upbit_coins:
        v = np.full(T, 2.0e9)
        if upbit is not None and c in upbit:
            v = upbit[c]
        series["upbit_krw_1d." + c + ".value_krw"] = v
        series["upbit_krw_1d." + c + ".close"] = np.ones(T)
    if uni is None:
        uni = np.zeros((T, N), dtype=bool)
        uni[:, :len(perps)] = True
    return DataView(
        dates=dates, instruments=instruments, asset_class=asset_class,
        ret_co=np.zeros((T, N)), ret_oc=ret_oc, tradable=trd, listed=lst,
        delisting=np.zeros((T, N), dtype=bool), close=close, dollar_volume=dv,
        universe=uni, extras={}, series=series, cash_ret=np.zeros(T),
    )


def surge(T=80, t=T0, mult=2.5, base=2.0e9):
    v = np.full(T, base)
    v[t] = base * mult
    return v


def col(v, name):
    return v.instruments.index(name)


def test_single_event_slot():
    v = make_view(upbit={"ADA": surge()}, ret={"ADAUSDT.P": (T0, 0.05)})
    w = target_weights(v, PARAMS)
    a, b, e = col(v, "ADAUSDT.P"), col(v, "BTCUSDT.P"), col(v, "ETHUSDT.P")
    h = PARAMS["hold_days"]
    assert np.all(w[:T0] == 0)
    assert np.allclose(w[T0:T0 + h, a], -0.15)
    assert np.allclose(w[T0:T0 + h, b], 0.075)
    assert np.allclose(w[T0:T0 + h, e], 0.075)
    assert np.all(w[T0 + h:] == 0)
    assert np.all(w[:, col(v, "ADAUSDT")] == 0)
    mask, side = events(v, PARAMS)
    assert mask.sum() == 1 and mask[T0, a] and side[T0, a] == -1


def test_global_surge_not_marked():
    sp = np.full(80, 1.0e7)
    sp[T0] = 2.5e7                               # spot surges as much as Upbit: K = 1
    v = make_view(upbit={"ADA": surge()}, spot={"ADA": sp}, ret={"ADAUSDT.P": (T0, 0.05)})
    assert np.all(target_weights(v, PARAMS) == 0)


def test_down_move_not_marked():
    v = make_view(upbit={"ADA": surge()}, ret={"ADAUSDT.P": (T0, -0.05)})
    assert np.all(target_weights(v, PARAMS) == 0)


def test_liquidity_floor():
    v = make_view(upbit={"ADA": surge(base=5.0e8)}, ret={"ADAUSDT.P": (T0, 0.05)})
    assert np.all(target_weights(v, PARAMS) == 0)


def test_min_valid_days():
    u = surge(t=25)
    u[:6] = np.nan                               # window t-30..t-1 at t=25 has 19 valid days
    v = make_view(upbit={"ADA": u}, ret={"ADAUSDT.P": (25, 0.05)})
    assert np.all(target_weights(v, PARAMS) == 0)
    u = surge(t=25)
    u[:5] = np.nan                               # 20 valid days -> eligible
    v = make_view(upbit={"ADA": u}, ret={"ADAUSDT.P": (25, 0.05)})
    assert target_weights(v, PARAMS)[25, col(v, "ADAUSDT.P")] == pytest.approx(-0.15)


def test_missing_spot_today_no_event():
    sp = np.full(80, 1.0e7)
    sp[T0] = np.nan
    v = make_view(upbit={"ADA": surge()}, spot={"ADA": sp}, ret={"ADAUSDT.P": (T0, 0.05)})
    assert np.all(target_weights(v, PARAMS) == 0)


def test_no_spot_pair_not_eligible():
    v = make_view(upbit={"ADA": surge()}, ret={"ADAUSDT.P": (T0, 0.05)}, spot_coins=("XRP",))
    assert np.all(target_weights(v, PARAMS) == 0)


def test_k_param():
    v = make_view(upbit={"ADA": surge(mult=2.5)}, ret={"ADAUSDT.P": (T0, 0.05)})
    p = dict(PARAMS, k=3)
    assert np.all(target_weights(v, p) == 0)
    p = dict(PARAMS, k=1.5)
    assert target_weights(v, p)[T0, col(v, "ADAUSDT.P")] == pytest.approx(-0.15)


def test_hold_param():
    v = make_view(upbit={"ADA": surge()}, ret={"ADAUSDT.P": (T0, 0.05)})
    a = col(v, "ADAUSDT.P")
    for h in (3, 7):
        w = target_weights(v, dict(PARAMS, hold_days=h))
        assert np.allclose(w[T0:T0 + h, a], -0.15)
        assert np.all(w[T0 + h:, a] == 0)


def test_slot_cap_largest_k_first():
    ups = {"ADA": surge(mult=2.2), "XRP": surge(mult=3.0), "DOT": surge(mult=2.6), "LINK": surge(mult=4.0)}
    ret = {c + "USDT.P": (T0, 0.05) for c in COINS}
    v = make_view(upbit=ups, ret=ret)
    w = target_weights(v, PARAMS)
    assert w[T0, col(v, "ADAUSDT.P")] == 0      # smallest K dropped
    for c in ("XRP", "DOT", "LINK"):
        assert w[T0, col(v, c + "USDT.P")] == pytest.approx(-0.15)
    assert w[T0, col(v, "BTCUSDT.P")] == pytest.approx(0.225)
    assert np.abs(w).sum(axis=1).max() <= 0.9 + 1e-12
    mask, _ = events(v, PARAMS)
    assert mask[T0].sum() == 4                  # the diagnostic marks all events


def test_no_retrigger_and_later_reentry():
    h = PARAMS["hold_days"]
    u = np.full(80, 2.0e9)
    u[T0] = u[T0 + 2] = u[T0 + h] = u[T0 + h + 1] = 6.0e9
    v = make_view(upbit={"ADA": u})
    a = col(v, "ADAUSDT.P")
    for t in (T0, T0 + 2, T0 + h, T0 + h + 1):
        v.ret_oc[t, a] = 0.05
    w = target_weights(v, PARAMS)
    assert np.allclose(w[T0:T0 + h, a], -0.15)
    assert w[T0 + h, a] == 0                     # day t+h is blocked (event at t within the previous h days)
    assert np.allclose(w[T0 + h + 1:T0 + 2 * h + 1, a], -0.15)
    mask, _ = events(v, PARAMS)
    assert list(np.flatnonzero(mask[:, a])) == [T0, T0 + h + 1]


def test_leaves_universe_closes_slot():
    v = make_view(upbit={"ADA": surge()}, ret={"ADAUSDT.P": (T0, 0.05)})
    a = col(v, "ADAUSDT.P")
    v.universe[T0 + 2:, a] = False
    w = target_weights(v, PARAMS)
    assert np.allclose(w[T0:T0 + 2, a], -0.15)
    assert np.all(w[T0 + 2:] == 0)               # both legs closed


def test_not_member_no_event():
    v = make_view(upbit={"ADA": surge()}, ret={"ADAUSDT.P": (T0, 0.05)})
    v.universe[T0, col(v, "ADAUSDT.P")] = False
    assert np.all(target_weights(v, PARAMS) == 0)


def test_point_in_time_truncation():
    ups = {"ADA": surge(), "XRP": surge(t=T0 + 3, mult=3.0)}
    ret = {"ADAUSDT.P": (T0, 0.05), "XRPUSDT.P": (T0 + 3, 0.04)}
    v = make_view(upbit=ups, ret=ret)
    full = target_weights(v, PARAMS)
    for cut in (10, 25, T0, T0 + 1, T0 + 3, T0 + 6):
        vt = make_view(T=cut + 1, upbit={c: x[:cut + 1] for c, x in ups.items()},
                       ret={k: x for k, x in ret.items() if x[0] <= cut})
        assert np.array_equal(target_weights(vt, PARAMS), full[:cut + 1])


def test_short_view():
    v = make_view(T=12)
    w = target_weights(v, PARAMS)
    assert w.shape == (12, len(v.instruments)) and np.all(w == 0)


def test_missing_upbit_raises():
    v = make_view(upbit_coins=())
    with pytest.raises(KeyError):
        target_weights(v, PARAMS)


def test_missing_spot_dataset_raises():
    v = make_view(spot_coins=())
    with pytest.raises(KeyError):
        target_weights(v, PARAMS)


def test_deterministic():
    v = make_view(upbit={"ADA": surge()}, ret={"ADAUSDT.P": (T0, 0.05)})
    assert np.array_equal(target_weights(v, PARAMS), target_weights(v, PARAMS))
