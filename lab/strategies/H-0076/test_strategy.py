import dataclasses
from typing import Any

import numpy as np
import pytest

from strategy import PARAMS, target_weights, legs, perp_base, nanmedian_last, events, top_n_universe

SUF = "USDT" + ".P"
PFX = "upbit_krw_1d."


@dataclasses.dataclass
class View:
    dates: Any
    instruments: Any
    universe: Any
    close: Any
    extras: Any
    series: Any
    tradable: Any
    dollar_volume: Any = None
    listed: Any = None


T = 60
T0 = 45
N_COIN = 12
# groups: A = 0,1,2 always valid; B = 3,4,5 invalid on t%3==0; C = 6,7,8 on t%3==1; D = 9,10,11 on t%3==2;
# on T0 all valid. So every day except T0 has 9 valid coins (< 10, no cohort) and on T0 each coin has exactly
# 20 valid days in T0-30 .. T0-1 (= 2/3 of baseline_days 30).


def make_view(shocks=None, basis_t0=None, extra_cols=True):
    names = ["K" + str(i).zfill(2) + SUF for i in range(N_COIN)]
    bases = ["K" + str(i).zfill(2) for i in range(N_COIN)]
    if extra_cols:
        names += ["1000" + "ZZ" + SUF, "QQ" + SUF]      # multiplier-prefixed (has an Upbit key) and unmatched
        bases += ["1000" + "ZZ", None]
    N = len(names)
    t = np.arange(T)
    close = np.ones((T, N))
    basis = np.zeros((T, N))
    series = {}
    d = np.zeros((T, N))
    if shocks:
        for j, v in shocks.items():
            d[T0, j] = v
    for j, b in enumerate(bases):
        if b is None:
            continue
        valid = np.ones(T, dtype=bool)
        if j < N_COIN and j >= 3:
            g = (j - 3) // 3
            valid = (t % 3) != g
        valid[T0] = True
        series[PFX + b + ".close"] = np.exp(7.0 + d[:, j])
        series[PFX + b + ".value_krw"] = np.where(valid, 2e9, 5e8)
    if basis_t0:
        for j, v in basis_t0.items():
            basis[T0, j] = v
    return View(
        dates=np.arange(T).astype("datetime64[D]"),
        instruments=tuple(names),
        universe=np.ones((T, N), dtype=bool),
        close=close,
        extras={"basis": basis, "funding": np.zeros((T, N))},
        series=series,
        tradable=np.ones((T, N), dtype=bool),
    )


BASE_SHOCKS = {0: 0.10, 3: 0.08, 6: 0.06, 11: -0.10, 8: -0.08, 5: -0.06, 12: 0.15}


def test_perp_base():
    assert perp_base("1" + "INCH" + SUF) == "1" + "INCH"
    assert perp_base("1000" + "SHIB" + SUF) is None
    assert perp_base("ETH" + "USDT") is None
    assert perp_base(SUF) is None


def test_nanmedian():
    x = np.array([[1.0, np.nan, 3.0, 2.0], [np.nan, np.nan, np.nan, np.nan], [4.0, 1.0, np.nan, np.nan]])
    med, cnt = nanmedian_last(x)
    assert med[0] == 2.0 and np.isnan(med[1]) and med[2] == 2.5
    assert list(cnt) == [3, 0, 2]


def test_legs_on_t0_only():
    v = make_view(BASE_SHOCKS)
    lm, sm, ok = legs(v, PARAMS)
    assert set(np.nonzero(lm[T0])[0]) == {0, 3, 6}
    assert set(np.nonzero(sm[T0])[0]) == {5, 8, 11}
    other = np.ones(T, dtype=bool)
    other[T0] = False
    assert not lm[other].any() and not sm[other].any()
    assert not lm[:, 12:].any() and not sm[:, 12:].any()     # multiplier / unmatched never eligible


def test_book_weights():
    v = make_view(BASE_SHOCKS)
    w = target_weights(v, PARAMS)
    q = 0.5 / (3 * 3)
    for t in range(T0, T0 + 3):
        assert np.allclose(w[t, [0, 3, 6]], q) and np.allclose(w[t, [5, 8, 11]], -q)
        assert np.isclose(np.abs(w[t]).sum(), 1.0 / 3) and np.isclose(w[t].sum(), 0.0)
    assert np.allclose(w[:T0], 0) and np.allclose(w[T0 + 3:], 0)


def test_hold_and_nleg_params():
    v = make_view(BASE_SHOCKS)
    w5 = target_weights(v, dict(PARAMS, hold_days=5))
    assert np.allclose(w5[T0:T0 + 5, 0], 0.5 / 15) and np.allclose(w5[T0 + 5:], 0)
    w2 = target_weights(v, dict(PARAMS, n_leg=2))
    assert np.allclose(w2[T0, [0, 3]], 0.5 / 6) and np.allclose(w2[T0, [8, 11]], -0.5 / 6)
    assert w2[T0, 6] == 0 and w2[T0, 5] == 0


def test_baseline_param():
    v = make_view(BASE_SHOCKS)
    assert np.allclose(target_weights(v, dict(PARAMS, baseline_days=60)), 0)   # not 40 valid days before T0
    # baseline 14 needs 10 valid days in T0-14 .. T0-1 (days 31..44): groups C and D have only 9 -> 6 eligible
    w14 = target_weights(v, dict(PARAMS, baseline_days=14))
    assert np.allclose(w14, 0)
    lm, _, _ = legs(v, dict(PARAMS, baseline_days=14))
    assert not lm.any()


def test_baseline_subtracted():
    # coin 1 ran rich by 0.09 on every earlier day: its abnormal premium at T0 is r - 0.09, so it is not long
    v = make_view(BASE_SHOCKS)
    u = v.series[PFX + "K01.close"].copy()
    u[:T0] = np.exp(7.09)
    v.series[PFX + "K01.close"] = u
    lm, sm, _ = legs(v, PARAMS)
    assert not lm[T0, 1] and sm[T0, 1]         # a = 0 - 0.09 is the lowest


def test_large_relative_premium_dropped():
    shocks = dict(BASE_SHOCKS)
    shocks[1] = 0.5
    v = make_view(shocks)
    lm, sm, _ = legs(v, PARAMS)
    assert not lm[T0, 1]
    assert set(np.nonzero(lm[T0])[0]) == {0, 3, 6}


def test_basis_spot_equivalent():
    v = make_view(BASE_SHOCKS, basis_t0={2: 0.10})   # S = P / 1.1: k up by ln 1.1 = 0.0953
    lm, _, _ = legs(v, PARAMS)
    assert set(np.nonzero(lm[T0])[0]) == {0, 2, 3}
    v2 = make_view(BASE_SHOCKS)
    v2.extras["basis"][:] = np.nan                   # missing basis: S = P
    lm2, _, _ = legs(v2, PARAMS)
    assert set(np.nonzero(lm2[T0])[0]) == {0, 3, 6}


def test_volume_floor_and_min_eligible():
    v = make_view(BASE_SHOCKS)
    for j in (1, 2, 4):
        vk = v.series[PFX + "K" + str(j).zfill(2) + ".value_krw"].copy()
        vk[T0] = 9.9e8
        v.series[PFX + "K" + str(j).zfill(2) + ".value_krw"] = vk
    lm, sm, _ = legs(v, PARAMS)                      # 9 eligible: no cohort
    assert not lm.any() and not sm.any()


def test_leaving_universe_closes_and_not_replaced():
    v = make_view(BASE_SHOCKS)
    v.universe[T0 + 1, 0] = False
    w = target_weights(v, PARAMS)
    assert w[T0, 0] > 0 and w[T0 + 1, 0] == 0 and w[T0 + 2, 0] == 0
    assert w[T0 + 1, 3] > 0
    v2 = make_view(BASE_SHOCKS)
    v2.close[T0 + 1, 11] = np.nan                    # perp data missing
    w2 = target_weights(v2, PARAMS)
    assert w2[T0 + 1, 11] == 0 and w2[T0 + 2, 11] == 0


def test_missing_inputs_raise():
    v = make_view(BASE_SHOCKS)
    v.series = {}
    with pytest.raises(KeyError):
        target_weights(v, PARAMS)
    v = make_view(BASE_SHOCKS)
    v.extras = {}
    with pytest.raises(KeyError):
        target_weights(v, PARAMS)


def random_view(T=260, N=20, seed=1):
    rng = np.random.default_rng(seed)
    names = ["R" + str(i).zfill(2) + SUF for i in range(N)]
    close = np.exp(np.cumsum(rng.normal(0, 0.03, (T, N)), axis=0))
    close[:40, 3] = np.nan                           # late listing
    basis = rng.normal(0, 0.002, (T, N))
    basis[rng.random((T, N)) < 0.05] = np.nan
    uni = rng.random((T, N)) < 0.9
    series = {}
    for j, nm in enumerate(names):
        if j == N - 1:
            continue                                 # one perp without Upbit data
        b = nm[: len(nm) - len(SUF)]
        prem = np.cumsum(rng.normal(0, 0.01, T)) * 0.3
        u = close[:, j] * 1300.0 * np.exp(prem)
        u[rng.random(T) < 0.03] = np.nan
        series[PFX + b + ".close"] = u
        series[PFX + b + ".value_krw"] = np.exp(rng.normal(21.5, 0.7, T))
    return View(np.arange(T).astype("datetime64[D]"), tuple(names), uni, close,
                {"basis": basis}, series, np.isfinite(close))


def cut_view(v, t, noise_seed=None):
    if noise_seed is None:
        sl = slice(0, t + 1)
        return View(v.dates[sl], v.instruments, v.universe[sl], v.close[sl], {"basis": v.extras["basis"][sl]},
                    {k: x[sl] for k, x in v.series.items()}, v.tradable[sl])
    rng = np.random.default_rng(noise_seed)
    c = v.close.copy()
    c[t + 1:] *= np.exp(rng.normal(0, 0.3, c[t + 1:].shape))
    s = {}
    for k, x in v.series.items():
        y = x.copy()
        y[t + 1:] *= np.exp(rng.normal(0, 0.3, y[t + 1:].shape))
        s[k] = y
    return View(v.dates, v.instruments, v.universe, c, {"basis": v.extras["basis"]}, s, v.tradable)


def test_point_in_time_and_gross():
    v = random_view()
    for params in (PARAMS, dict(PARAMS, baseline_days=14, hold_days=5, n_leg=4)):
        w = target_weights(v, params)
        assert np.all(np.isfinite(w))
        assert np.all(np.abs(w).sum(axis=1) <= 1.0 + 1e-12)
        assert np.abs(w).sum() > 0
        assert np.all(w[~v.universe] == 0)
        for t in (50, 100, 177, 230):
            wt = target_weights(cut_view(v, t), params)
            assert np.array_equal(wt, w[: t + 1])
            wp = target_weights(cut_view(v, t, noise_seed=t), params)
            assert np.array_equal(wp[: t + 1], w[: t + 1])


def test_dollar_neutral_without_exits():
    v = random_view()
    v.universe[:] = True                             # nobody leaves: every live cohort is complete
    for params in (PARAMS, dict(PARAMS, hold_days=5, n_leg=4)):
        w = target_weights(v, params)
        assert np.allclose(w.sum(axis=1), 0.0)
        assert np.all(np.abs(w).sum(axis=1) <= 1.0 + 1e-12)
        assert np.abs(w).sum() > 0


def test_fallback_universe_ranking():
    # 35 perps, dollar volume = column index + 1 except: col 0 huge but only on 5 days long ago (out of window),
    # col 1 not listed on the last row, col 2 NaN volume on all days (no volume)
    T_, N_ = 40, 35
    dv = np.tile(np.arange(1, N_ + 1, dtype=float), (T_, 1))
    dv[:, 0] = 0.0
    dv[:5, 0] = 1e9
    dv[:, 1] = 1e6
    dv[:, 2] = np.nan
    listed = np.ones((T_, N_), dtype=bool)
    listed[-1, 1] = False
    v = View(np.arange(T_).astype("datetime64[D]"), tuple("U" + str(i).zfill(2) + SUF for i in range(N_)), None,
             np.ones((T_, N_)), {"basis": np.zeros((T_, N_))}, {}, np.ones((T_, N_), dtype=bool), dv, listed)
    u = top_n_universe(v)
    assert u[3, 0]                                   # big recent volume early on
    last = u[-1]
    assert not last[0] and not last[1] and not last[2]
    assert last.sum() == 30
    assert set(np.nonzero(last)[0]) == set(range(5, 35))
    assert top_n_universe(v, n=5)[-1].sum() == 5
    v.universe = np.zeros((T_, N_), dtype=bool)
    assert not top_n_universe(v).any()               # the gate's universe wins when passed


def test_fallback_universe_used_by_strategy_and_point_in_time():
    v = random_view(N=36, seed=3)
    rng = np.random.default_rng(7)
    dv = np.exp(rng.normal(18, 1.0, v.close.shape))
    dv[~np.isfinite(v.close)] = np.nan
    listed = np.isfinite(v.close)
    vn = View(v.dates, v.instruments, None, v.close, v.extras, v.series, v.tradable, dv, listed)
    uni = top_n_universe(vn)
    assert np.all(uni.sum(axis=1) <= 30) and uni.sum(axis=1).max() == 30
    w = target_weights(vn, PARAMS)
    assert np.abs(w).sum() > 0 and np.all(w[~uni] == 0)
    assert np.all(np.abs(w).sum(axis=1) <= 1.0 + 1e-12)
    vu = View(v.dates, v.instruments, uni, v.close, v.extras, v.series, v.tradable, dv, listed)
    assert np.array_equal(w, target_weights(vu, PARAMS))
    m, s = events(vn, PARAMS)
    lm, sm, _ = legs(vn, PARAMS)
    assert np.array_equal(m, (lm | sm) & uni & vn.tradable)
    for t in (60, 150, 222):
        sl = slice(0, t + 1)
        vc = View(v.dates[sl], v.instruments, None, v.close[sl], {"basis": v.extras["basis"][sl]},
                  {k: x[sl] for k, x in v.series.items()}, v.tradable[sl], dv[sl], listed[sl])
        assert np.array_equal(target_weights(vc, PARAMS), w[sl])
        dv2 = dv.copy()
        dv2[t + 1:] *= np.exp(rng.normal(0, 2.0, dv2[t + 1:].shape))
        vp = View(v.dates, v.instruments, None, v.close, v.extras, v.series, v.tradable, dv2, listed)
        assert np.array_equal(target_weights(vp, PARAMS)[sl], w[sl])


def test_events_match_legs():
    v = make_view(BASE_SHOCKS)
    mask, side = events(v, PARAMS)
    assert set(np.nonzero(mask[T0])[0]) == {0, 3, 6, 5, 8, 11}
    assert np.all(side[T0, [0, 3, 6]] == 1) and np.all(side[T0, [5, 8, 11]] == -1)
    assert mask.sum() == 6
    rv = random_view()
    m2, s2 = events(rv, PARAMS)
    w = target_weights(rv, PARAMS)
    lm, sm, _ = legs(rv, PARAMS)
    assert np.array_equal(m2, (lm | sm) & rv.universe & rv.tradable)
