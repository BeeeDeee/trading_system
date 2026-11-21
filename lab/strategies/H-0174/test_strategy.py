import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS, _events


def events(v, params):
    """Same event code as diagnostic.py (copied there): mechanism events on the perp columns, side -1."""
    ev, _, cols, _ = _events(v, params)
    mask = np.zeros((len(v.dates), len(v.instruments)), dtype=bool)
    for m, j in enumerate(cols):
        mask[:, j] |= ev[:, m]
    return mask, np.where(mask, -1.0, 0.0)

BASES = ["ADA", "XRP", "DOT", "LINK"]
T0 = 60


def make_view(T=T0, upbit=None, spot=None, rets=None, universe=True, extra_perp=True, with_spot=True,
              hedges=True, upbit_level=2.0e9, n_fill=1):
    names, ac = [], []
    if hedges:
        names += ["BTCUSDT.P", "ETHUSDT.P"]
        ac += ["crypto_perp", "crypto_perp"]
    for b in BASES:
        names.append(b + "USDT.P")
        ac.append("crypto_perp")
    if extra_perp:
        for f in range(n_fill):
            names.append("FILL%dUSDT.P" % f)
            ac.append("crypto_perp")
    if with_spot:
        for b in BASES:
            names.append(b + "USDT")
            ac.append("crypto_spot")
    N = len(names)
    t = np.arange(T, dtype=float)
    ret_co = np.zeros((T, N))
    ret_oc = np.zeros((T, N))
    if rets is not None:
        for nm, (row, val) in rets.items():
            ret_oc[row, names.index(nm)] = val
    listed = np.ones((T, N), dtype=bool)
    close = np.ones((T, N))
    dv = np.zeros((T, N))
    for i, nm in enumerate(names):
        dv[:, i] = 1.0e7 * (1.0 + 0.001 * t) * (1.0 + 0.01 * i)
    series = {}
    for bi, b in enumerate(BASES + ["BTC", "ETH"]):
        series["upbit_krw_1d." + b + ".value_krw"] = upbit_level * (1.0 + 0.001 * t) * (1.0 + 0.01 * bi)
        series["upbit_krw_1d." + b + ".close"] = np.ones(T)
    if upbit is not None:
        for b, (rows, mult) in upbit.items():
            v = series["upbit_krw_1d." + b + ".value_krw"].copy()
            v[rows] = v[rows] * mult
            series["upbit_krw_1d." + b + ".value_krw"] = v
    if spot is not None and with_spot:
        for b, (rows, mult) in spot.items():
            dv[rows, names.index(b + "USDT")] *= mult
    perp = np.array([a == "crypto_perp" for a in ac])
    if universe is True:
        uni = np.broadcast_to(perp, (T, N)).copy()
    elif universe is None:
        uni = None
    else:
        uni = universe(names, np.broadcast_to(perp, (T, N)).copy())
    return DataView(
        dates=np.arange(18000, 18000 + T).astype("datetime64[D]"),
        instruments=tuple(names), asset_class=tuple(ac),
        ret_co=ret_co, ret_oc=ret_oc, tradable=listed.copy(), listed=listed,
        delisting=np.zeros((T, N), dtype=bool), close=close, dollar_volume=dv,
        universe=uni, extras={}, series=series, cash_ret=np.zeros(T),
    )


def col(v, nm):
    return v.instruments.index(nm)


def pump(base, row, mult=3.0, ret=0.05):
    return {base: ([row], mult)}, {base + "USDT.P": (row, ret)}


def test_basic_event_and_hold_timing():
    up, rt = pump("ADA", 40)
    v = make_view(upbit=up, rets=rt)
    w = target_weights(v, PARAMS)
    a, b, e = col(v, "ADAUSDT.P"), col(v, "BTCUSDT.P"), col(v, "ETHUSDT.P")
    assert np.all(w[:40] == 0)
    for t in range(40, 45):                      # decision rows t..t+hold-1
        assert w[t, a] == pytest.approx(-0.15)
        assert w[t, b] == pytest.approx(0.075)
        assert w[t, e] == pytest.approx(0.075)
    assert np.all(w[45:] == 0)                    # closed at the open of t+1+hold
    w3 = target_weights(v, dict(PARAMS, hold_days=3))
    assert w3[42, a] == pytest.approx(-0.15) and w3[43, a] == 0


def test_global_surge_is_not_an_event():
    up, rt = pump("ADA", 40)
    v = make_view(upbit=up, rets=rt, spot={"ADA": ([40], 3.0)})
    assert np.all(target_weights(v, PARAMS) == 0)


def test_return_must_beat_universe_median():
    up, _ = pump("ADA", 40)
    v = make_view(upbit=up, rets={"ADAUSDT.P": (40, -0.01)})
    assert np.all(target_weights(v, PARAMS) == 0)
    v0 = make_view(upbit=up)                       # return equal to the median (0): not above
    assert np.all(target_weights(v0, PARAMS) == 0)


def test_k_threshold_changes_output():
    up, rt = pump("ADA", 40, mult=2.5)
    v = make_view(upbit=up, rets=rt)
    a = col(v, "ADAUSDT.P")
    assert target_weights(v, dict(PARAMS, k=2.0))[40, a] == pytest.approx(-0.15)
    assert target_weights(v, dict(PARAMS, k=1.5))[40, a] == pytest.approx(-0.15)
    assert np.all(target_weights(v, dict(PARAMS, k=3.0)) == 0)


def test_liquidity_floor():
    up, rt = pump("ADA", 40)
    v = make_view(upbit=up, rets=rt, upbit_level=5.0e8)
    assert np.all(target_weights(v, PARAMS) == 0)


def test_needs_20_valid_days():
    up, rt = pump("ADA", 15)
    v = make_view(upbit=up, rets=rt)
    assert np.all(target_weights(v, PARAMS) == 0)
    up, rt = pump("ADA", 20)                       # window 0..19 = 20 valid days
    v = make_view(upbit=up, rets=rt)
    assert target_weights(v, PARAMS)[20, col(v, "ADAUSDT.P")] == pytest.approx(-0.15)


def test_short_view_does_not_crash():
    v = make_view(T=12)
    w = target_weights(v, PARAMS)
    assert w.shape == (12, len(v.instruments)) and np.all(w == 0)


def test_no_retrigger_within_hold():
    rows = list(range(40, 52))
    v = make_view(upbit={"ADA": (rows, 3.0)})
    for t in rows:
        v.ret_oc[t, col(v, "ADAUSDT.P")] = 0.05
    mask, side = events(v, PARAMS)
    a = col(v, "ADAUSDT.P")
    assert list(np.flatnonzero(mask[:, a])) == [40, 46]
    assert np.all(side[mask] == -1)
    w = target_weights(v, PARAMS)
    assert np.all(w[40:45, a] == pytest.approx(-0.15))
    assert w[45, a] == 0
    assert np.all(w[46:51, a] == pytest.approx(-0.15))
    assert w[51, a] == 0


def test_slot_cap_takes_largest_K():
    up = {"ADA": ([40], 3.0), "XRP": ([40], 4.0), "DOT": ([40], 5.0), "LINK": ([40], 6.0)}
    rt = {b + "USDT.P": (40, 0.05) for b in BASES}
    v = make_view(upbit=up, n_fill=5)              # 11 perps: 4 up-movers stay above the median (0)
    for nm, (row, val) in rt.items():
        v.ret_oc[row, col(v, nm)] = val
    w = target_weights(v, PARAMS)
    assert w[40, col(v, "ADAUSDT.P")] == 0          # smallest K dropped
    for b in ["XRP", "DOT", "LINK"]:
        assert w[40, col(v, b + "USDT.P")] == pytest.approx(-0.15)
    assert w[40, col(v, "BTCUSDT.P")] == pytest.approx(0.225)
    assert np.sum(np.abs(w[40])) == pytest.approx(0.9)
    assert np.all(np.sum(np.abs(w), axis=1) <= 1.0 + 1e-12)
    mask, _ = events(v, PARAMS)
    assert mask[40, col(v, "ADAUSDT.P")]           # still a mechanism event


def test_slot_closes_when_coin_leaves_universe():
    up, rt = pump("ADA", 40)

    def uni(names, u):
        u[42:, names.index("ADAUSDT.P")] = False
        return u
    v = make_view(upbit=up, rets=rt, universe=uni)
    w = target_weights(v, PARAMS)
    assert w[41, col(v, "ADAUSDT.P")] == pytest.approx(-0.15)
    assert np.all(w[42:] == 0)


def test_non_member_not_eligible():
    up, rt = pump("ADA", 40)

    def uni(names, u):
        u[:, names.index("ADAUSDT.P")] = False
        return u
    v = make_view(upbit=up, rets=rt, universe=uni)
    assert np.all(target_weights(v, PARAMS) == 0)


def test_missing_or_repeated_upbit_value_means_no_event():
    up, rt = pump("ADA", 40)
    v = make_view(upbit=up, rets=rt)
    v.series["upbit_krw_1d.ADA.value_krw"][40] = np.nan
    assert np.all(target_weights(v, PARAMS) == 0)
    v = make_view(upbit=up, rets=rt)
    s = v.series["upbit_krw_1d.ADA.value_krw"]
    s[40] = s[39]                                   # carried-forward value = gap
    assert np.all(target_weights(v, PARAMS) == 0)
    v = make_view(upbit=up, rets=rt)
    v.dollar_volume[40, col(v, "ADAUSDT")] = np.nan  # missing spot volume
    assert np.all(target_weights(v, PARAMS) == 0)


def test_hedge_coins_never_events():
    v = make_view(upbit={"BTC": ([40], 5.0), "ETH": ([40], 5.0)},
                  rets={"BTCUSDT.P": (40, 0.05)})
    v.ret_oc[40, col(v, "ETHUSDT.P")] = 0.05
    assert np.all(target_weights(v, PARAMS) == 0)


def test_missing_spot_pairs_raise():
    v = make_view(with_spot=False)
    with pytest.raises(ValueError):
        target_weights(v, PARAMS)


def test_spot_volume_as_series():
    up, rt = pump("ADA", 40)
    v = make_view(upbit=up, rets=rt)
    ref = target_weights(v, PARAMS)
    v2 = make_view(upbit=up, rets=rt, with_spot=False)
    for b in BASES:
        v2.series["binance_spot_1d." + b + "USDT.quote_volume"] = v.dollar_volume[:, col(v, b + "USDT")].copy()
    w2 = target_weights(v2, PARAMS)
    np.testing.assert_array_equal(w2, ref[:, :len(v2.instruments)])
    assert w2[40, col(v2, "ADAUSDT.P")] == pytest.approx(-0.15)


def test_missing_hedge_raises():
    v = make_view(hedges=False)
    with pytest.raises(Exception):
        target_weights(v, PARAMS)


def test_fallback_universe_when_none():
    up, rt = pump("ADA", 40)
    v = make_view(upbit=up, rets=rt, universe=None)
    w = target_weights(v, PARAMS)
    assert w[40, col(v, "ADAUSDT.P")] == pytest.approx(-0.15)
    assert np.all(w[:, col(v, "ADAUSDT")] == 0)      # spot never held


def test_point_in_time_truncation():
    up = {"ADA": ([40, 50], 3.0), "XRP": ([45], 4.0)}
    v = make_view(upbit=up, rets={"ADAUSDT.P": (40, 0.05)})
    v.ret_oc[50, col(v, "ADAUSDT.P")] = 0.05
    v.ret_oc[45, col(v, "XRPUSDT.P")] = 0.05
    full = target_weights(v, PARAMS)
    for cut in [30, 41, 46, 52]:
        tv = DataView(
            dates=v.dates[:cut], instruments=v.instruments, asset_class=v.asset_class,
            ret_co=v.ret_co[:cut], ret_oc=v.ret_oc[:cut], tradable=v.tradable[:cut], listed=v.listed[:cut],
            delisting=v.delisting[:cut], close=v.close[:cut], dollar_volume=v.dollar_volume[:cut],
            universe=v.universe[:cut], extras={}, series={kk: s[:cut] for kk, s in v.series.items()},
            cash_ret=v.cash_ret[:cut])
        np.testing.assert_array_equal(target_weights(tv, PARAMS), full[:cut])
