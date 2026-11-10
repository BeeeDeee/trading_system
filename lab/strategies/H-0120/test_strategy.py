from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from strategy import (PARAMS, book, build_features, close_universe_exits, decision_rows,
                      effective_universe, target_weights, window_mean, window_sum)


@dataclass
class View:
    """Minimal stand-in with the DataView attributes the helpers read."""
    universe: Any = None
    dollar_volume: Any = None
    listed: Any = None
    dates: Any = None
    instruments: Any = None
    series: Any = None
    extras: Any = None


def _dates(T):
    return np.arange(T).astype("datetime64[D]")


def test_params_match_card():
    assert PARAMS == {"n_estimators": 100, "top": 0.2}
    assert callable(target_weights)


def test_window_sum_by_hand():
    x = np.arange(6, dtype=float)[:, None]           # 0..5
    s = window_sum(x, 0, 2)
    assert np.all(np.isnan(s[:2, 0]))
    assert s[2, 0] == 0 + 1 + 2 and s[5, 0] == 3 + 4 + 5
    m = window_mean(x, 3, 4)                           # rows t-4..t-3
    assert np.all(np.isnan(m[:4, 0]))
    assert m[4, 0] == 0.5 and m[5, 0] == 1.5
    y = x.copy()
    y[3, 0] = np.nan
    s = window_sum(y, 0, 2)
    assert np.all(np.isnan(s[3:6, 0])) and s[2, 0] == 3


def test_window_shorter_than_view():
    x = np.ones((10, 3))
    assert np.all(np.isnan(window_mean(x, 7, 59)))
    assert window_sum(np.ones((3, 2)), 0, 2)[2, 0] == 3


def test_window_point_in_time():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(80, 4))
    full = window_mean(x, 7, 59)
    y = x.copy()
    y[61:] = rng.normal(size=(19, 4)) * 100
    assert np.array_equal(window_mean(y, 7, 59)[:61], full[:61], equal_nan=True)
    assert np.array_equal(window_mean(x[:61], 7, 59), full[:61], equal_nan=True)
    assert np.isfinite(full[59:]).all() and np.isnan(full[:59]).all()


def test_decision_rows_every_fifth_day():
    d = decision_rows(_dates(12))
    assert list(np.flatnonzero(d)) == [0, 5, 10]


def test_book_legs_and_gross():
    T, N = 3, 20
    score = np.tile(np.arange(N, dtype=float), (T, 1))
    uni = np.ones((T, N), dtype=bool)
    w = book(score, uni, 0.2)                          # 4 per leg
    assert np.allclose(w[0, 16:], 0.125) and np.allclose(w[0, :4], -0.125)
    assert np.allclose(w[0, 4:16], 0.0)
    assert np.allclose(np.abs(w).sum(axis=1), 1.0) and np.allclose(w.sum(axis=1), 0.0)
    w15 = book(score, uni, 0.15)                       # floor(3.0) = 3 per leg
    assert (w15[0] > 0).sum() == 3 and (w15[0] < 0).sum() == 3
    w25 = book(score, uni, 0.25)                       # 5 per leg
    assert (w25[0] > 0).sum() == 5


def test_book_min_per_leg_and_min_names():
    N = 12
    score = np.arange(N, dtype=float)[None, :].repeat(2, axis=0)
    uni = np.ones((2, N), dtype=bool)
    uni[1, :3] = False                                 # 9 members on row 1
    w = book(score, uni, 0.15)                         # row 0: floor(1.8)=1 -> 2 per leg
    assert (w[0] > 0).sum() == 2 and (w[0] < 0).sum() == 2
    assert np.allclose(w[0][w[0] > 0], 0.25)
    assert np.all(w[1] == 0)                           # fewer than 10 scored -> flat
    score2 = score.copy()
    score2[0, 5] = np.nan                              # NaN score: not scored
    w2 = book(score2, uni, 0.2)
    assert w2[0, 5] == 0 and (w2[0] > 0).sum() == 2    # 11 scored -> floor(2.2)=2


def test_book_only_members():
    N = 20
    score = np.arange(N, dtype=float)[None, :]
    uni = np.ones((1, N), dtype=bool)
    uni[0, 19] = False
    w = book(score, uni, 0.2)
    assert w[0, 19] == 0 and w[0, 18] > 0


def test_universe_exit_closes_position():
    T, N = 10, 3
    dec = decision_rows(_dates(T))
    w = np.full((T, N), np.nan)
    w[0] = [0.5, -0.5, 0.0]
    w[5] = [0.5, -0.5, 0.0]
    uni = np.ones((T, N), dtype=bool)
    uni[2:4, 0] = False                                # leaves at row 2, back at 4
    uni[7, 2] = False                                  # not held: nothing happens
    out = close_universe_exits(w, dec, uni)
    assert out[2, 0] == 0.0 and np.isnan(out[3, 0]) and np.isnan(out[4, 0])
    assert np.all(np.isnan(out[1])) and np.isnan(out[2, 1])
    assert np.all(np.isnan(out[6:, :]))
    assert out[5, 0] == 0.5


def test_fallback_universe_top_n_by_30d_volume():
    T, N = 40, 25
    dv = np.tile(np.arange(1, N + 1, dtype=float), (T, 1))
    listed = np.ones((T, N), dtype=bool)
    listed[:, 24] = False                              # most liquid but not listed
    dv[:, 24] = np.nan
    v = View(universe=None, dollar_volume=dv, listed=listed)
    u = effective_universe(v)
    assert u.sum(axis=1).max() == 20
    assert not u[:, 24].any() and u[:, 4:24].all() and not u[:, :4].any()
    # short views (fewer rows than the window) work and agree with the long view
    v2 = View(universe=None, dollar_volume=dv[:5], listed=listed[:5])
    assert np.array_equal(effective_universe(v2), u[:5])
    given = np.zeros((T, N), dtype=bool)
    assert effective_universe(View(universe=given, dollar_volume=dv, listed=listed)) is not None


def _feature_view(T=70):
    rng = np.random.default_rng(3)
    inst = ("BTCUSDT.P", "AAAUSDT.P", "BBBUSDT.P", "CCCUSDT.P", "DDDUSDT.P")
    series = {}
    for name in ("BTCUSDT", "AAAUSDT", "BBBUSDT", "DDDUSDT"):  # CCC has no spot key
        for f in ("ret_asia", "ret_europe", "ret_us"):
            series["binance_1h_features." + name + "." + f] = rng.normal(0, 0.01, T).astype(np.float32)
        series["binance_1h_features." + name + ".vshare_us"] = rng.uniform(0.2, 0.5, T).astype(np.float32)
        series["binance_1h_features." + name + ".rv"] = rng.uniform(0.01, 0.05, T).astype(np.float32)
    listed = np.ones((T, 5), dtype=bool)
    listed[:20, 2] = False                                     # BBB perp listed late
    funding = rng.normal(0, 1e-4, (T, 5)).astype(np.float32)
    return View(dates=_dates(T), instruments=inst, series=series, extras={"funding": funding}, listed=listed)


def test_features_nan_handling_and_history():
    v = _feature_view()
    uni = np.ones((70, 5), dtype=bool)
    X = np.asarray(build_features(v, uni))
    assert X.shape == (70, 5, 6)
    assert np.all(np.isnan(X[:59]))                           # rv needs 60 rows; history < 60
    assert np.all(np.isnan(X[:, 3, :5]))                       # no spot key: incomplete vector, not scored
    assert np.isfinite(X[59:, 3, 5]).all()                     # its funding feature alone exists
    assert np.all(np.isnan(X[:, 2]))                           # BBB has < 60 listed days through row 69
    assert np.all(np.isfinite(X[59:, [0, 1, 4]]))


def test_features_point_in_time():
    v = _feature_view()
    uni = np.ones((70, 5), dtype=bool)
    full = np.asarray(build_features(v, uni))
    cut = 63
    v2 = View(dates=v.dates[:cut + 1], instruments=v.instruments,
              series={k: a[:cut + 1] for k, a in v.series.items()},
              extras={"funding": v.extras["funding"][:cut + 1]}, listed=v.listed[:cut + 1])
    part = np.asarray(build_features(v2, uni[:cut + 1]))
    assert np.allclose(part, full[:cut + 1], equal_nan=True)


def test_btc_residual_by_hand():
    v = _feature_view()
    uni = np.ones((70, 5), dtype=bool)
    # BTC's own session residuals are exactly zero; with z-scoring over 2 finite names the AAA value
    # has the sign of its residual sum
    t = 65
    s = v.series
    res = sum(float(s["binance_1h_features.AAAUSDT.ret_us"][k]) - float(s["binance_1h_features.BTCUSDT.ret_us"][k])
              for k in (t - 2, t - 1, t))
    X = np.asarray(build_features(v, uni))
    if abs(res) > 1e-9:
        assert np.sign(X[t, 1, 2] - X[t, 0, 2]) == np.sign(res)


def test_missing_btc_reference_raises():
    v = _feature_view()
    del v.series["binance_1h_features.BTCUSDT.rv"]
    del v.series["binance_1h_features.BTCUSDT.ret_asia"]
    with pytest.raises(KeyError):
        build_features(v, np.ones((70, 5), dtype=bool))
