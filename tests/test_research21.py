"""Research 21: features, proxy-filter flags, criteria and controls (prereg v1.1)."""

from types import SimpleNamespace

import numpy as np
import polars as pl

from qlab.data.panel import Panel
from qlab.research5.costs import DateCost, period_cost
from qlab.research5.sim import Candidates, SimSpec, simulate
from qlab.research21 import evaluate as EV
from qlab.research21 import features as F
from qlab.research21 import setup as S
from qlab.research21 import strategy as ST


def rand_inputs(T=400, N=5, seed=1):
    rng = np.random.default_rng(seed)
    co, oc = rng.normal(0, 0.01, (T, N)), rng.normal(0, 0.015, (T, N))
    ok = np.ones((T, N), bool)
    ok[:30, 0] = False                                   # security 0 listed from day 30
    dv = rng.uniform(1e7, 2e8, (T, N))
    earn = rng.random((T, N)) < 0.02
    etf = rng.normal(0, 0.01, (T, N))
    return co, oc, ok, ok.copy(), dv, earn, etf


def test_prior_stats_window_ends_at_t_minus_1_and_ignores_nan():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(120, 3))
    x[10:15, 1] = np.nan
    mean, sd = F.prior_stats(x, 20, 15)
    for t in (0, 14, 25, 80, 119):
        w = x[max(t - 20, 0):t]                         # rows t-20 .. t-1, NOT t
        for j in range(3):
            v = w[np.isfinite(w[:, j]), j]
            if len(v) >= 15:
                assert np.isclose(mean[t, j], v.mean()) and np.isclose(sd[t, j], v.std(ddof=1))
            else:
                assert np.isnan(mean[t, j]) and np.isnan(sd[t, j])


def test_window_any_is_trading_day_window_with_clipping():
    m = np.zeros((10, 1), bool)
    m[5, 0] = True
    got = F.window_any(m, 2, 2)[:, 0]
    assert got.nonzero()[0].tolist() == [3, 4, 5, 6, 7]           # event at 5 is in the window of t = 3..7
    got2 = F.window_any(m, 2, 0)[:, 0]
    assert got2.nonzero()[0].tolist() == [5, 6, 7]                # r2: only t-2 .. t, i.e. t = 5..7
    m[0, 0] = True
    assert F.window_any(m, 2, 2)[0, 0] and F.window_any(m, 2, 2)[2, 0]


def test_features_are_point_in_time_except_e1():
    args = rand_inputs()
    full = F.compute_chunk(*args)
    for k in (150, 301):                                 # truncate and recompute: rows < k must not change
        cut = F.compute_chunk(*(a[:k] for a in args))
        for name in ("rsi", "valid", "regime", "exit", "O", "volratio", "gapz", "idioz", "e2"):
            a, b = np.asarray(full[name])[:k], np.asarray(cut[name])
            assert np.allclose(a, b, equal_nan=True), name
        # E1 looks 2 days ahead: identical up to k - 3, and it does differ somewhere near the cut
        assert np.array_equal(full["e1"][:k - 2], cut["e1"][:k - 2])


def test_future_perturbation_does_not_change_past_features():
    args = list(rand_inputs())
    base = F.compute_chunk(*args)
    k = 200
    rng = np.random.default_rng(5)
    pert = [a.copy() for a in args]
    for i in (0, 1, 4, 6):                               # returns, dollar volume, etf returns
        pert[i][k:] = rng.normal(0, 0.05, pert[i][k:].shape) if i != 4 else rng.uniform(1e6, 1e9, pert[i][k:].shape)
    new = F.compute_chunk(*pert)
    for name in ("rsi", "regime", "exit", "O", "volratio", "gapz", "idioz", "e2"):
        assert np.allclose(np.asarray(base[name])[:k], np.asarray(new[name])[:k], equal_nan=True), name


def test_volratio_gap_and_idio_hand_example():
    T = 80
    co = np.zeros((T, 1)); oc = np.zeros((T, 1))
    rng = np.random.default_rng(2)
    co[:, 0] = rng.normal(0, 0.01, T)
    dv = np.full((T, 1), 1e8)
    dv[70, 0] = 4e8                                       # 4x the prior 20-day average
    co[70, 0] = -0.20                                     # huge gap down on the same day
    ok = np.ones((T, 1), bool)
    etf = np.zeros((T, 1))
    f = F.compute_chunk(co, oc, ok, ok.copy(), dv, np.zeros((T, 1), bool), etf)
    assert np.isclose(f["volratio"][70, 0], 4.0, atol=1e-5)
    assert np.isnan(f["volratio"][5, 0])                  # fewer than 15 prior observations
    assert f["gapz"][70, 0] < -5 and f["idioz"][70, 0] < -5   # the event does not dilute its own sigma
    assert abs(f["gapz"][60, 0]) < 4


def test_unavailable_filter_blocks_and_e_without_8k_allows():
    nan = np.float32(np.nan)
    m = {"volratio": np.array([[1.0, nan, 5.0]], np.float32), "gapz": np.array([[0.0, nan, -3.0]], np.float32),
         "idioz": np.array([[0.0, nan, -3.0]], np.float32), "e1": np.zeros((1, 3), bool), "e2": np.zeros((1, 3), bool)}
    m["e1"][0, 2] = True
    f = SimpleNamespace(m=m, pos=lambda a: np.asarray(a))
    t = np.zeros(3, int)
    a = np.array([0, 1, 2])
    fl = ST.blocked_flags(f, t, a, ST.Config())
    assert fl["V"].tolist() == [False, True, True]        # 5x > 3x blocks, NaN blocks
    assert fl["G"].tolist() == [False, True, True]
    assert fl["I"].tolist() == [False, True, True]
    assert fl["E"].tolist() == [False, False, True]       # no 8-K means allowed
    assert fl["C"].tolist() == [False, True, True]


def test_net_at_matches_the_simulator_and_scales_costs():
    T, N = 40, 2
    dates = np.arange(np.datetime64("2010-01-04"), np.datetime64("2010-01-04") + T)
    rng = np.random.default_rng(3)
    co, oc = rng.normal(0, 0.01, (T, N)), rng.normal(0, 0.01, (T, N))
    ok = np.ones((T, N), bool)
    p = Panel(dates, np.arange(N), co, oc, ok, ok.copy(), np.zeros((T, N), bool), np.full((T, N), 50.0), np.full((T, N), 1e8))
    ex = np.zeros((T, N), bool)
    ex[12:, 0] = True
    open_c = DateCost(period_cost(dates, True, 1.0))
    out = simulate(p, Candidates(np.array([5]), np.array([0]), np.array([1.0])), SimSpec(2, 10, 1, adv_cap=np.inf),
                   np.broadcast_to(np.inf, (T, N)), 0, T, open_c.at, open_c.at, ex, None, None)
    tr = out.trades.row(0, named=True)
    c = period_cost(dates, True, 1.0)
    assert np.isclose(EV.net_at(np.array([tr["gross_ret"]]), c[tr["entry_day"]], c[tr["exit_day"]])[0], tr["net_ret"])
    g, ce, cx = np.array([0.01]), np.array([0.001]), np.array([0.001])
    assert EV.net_at(g, ce, cx, 2.0)[0] < EV.net_at(g, ce, cx, 1.0)[0] < g[0]


def _trades(n=4000, effect=0.0, seed=0, blocked_share=0.4):
    rng = np.random.default_rng(seed)
    month = rng.integers(0, 120, n)
    blk = rng.random(n) < blocked_share
    net = rng.normal(0.001, 0.03, n) + np.where(blk, 0.0, effect)
    year = 1999 + month // 12 * 27 // 10
    return pl.DataFrame({"net_ret": net, "gross_ret": net + 0.002, "b_V": blk, "month": month,
                         "year": np.clip(year, 1999, 2026)})


def test_spread_ci_detects_a_planted_effect_and_not_noise():
    t = _trades(effect=0.01)
    p, lo, hi = EV.month_boot_spread(t["net_ret"].to_numpy(), t["b_V"].to_numpy(), t["month"].to_numpy(), reps=300)
    assert lo > 0 and p > 0.005
    t0 = _trades(effect=0.0, seed=4)
    p0, lo0, hi0 = EV.month_boot_spread(t0["net_ret"].to_numpy(), t0["b_V"].to_numpy(), t0["month"].to_numpy(), reps=300)
    assert lo0 < 0 < hi0


def test_filter_criteria_pass_fail_and_share_rule():
    n = 4000
    ce = cx = np.full(n, 0.0005)
    good = EV.filter_criteria(_trades(effect=0.01), "V", ce, cx, c2_p95=0.0005, thirds=S.THIRDS, reps=200)
    assert good["verdict"] == "nazek" and good["criteria_passed"] == 5
    bad = EV.filter_criteria(_trades(effect=0.0, seed=7), "V", ce, cx, c2_p95=0.0005, thirds=S.THIRDS, reps=200)
    assert bad["verdict"] == "ne"
    thin = EV.filter_criteria(_trades(effect=0.01, blocked_share=0.05), "V", ce, cx, 0.0005, S.THIRDS, 200)
    assert thin["verdict"] == "nelze posoudit"
    weak_c2 = EV.filter_criteria(_trades(effect=0.01), "V", ce, cx, c2_p95=0.5, thirds=S.THIRDS, reps=200)
    assert not weak_c2["criteria"]["c2_allowed_gt0_and_gt_c2_p95"] and weak_c2["verdict"] == "ne"


def test_c2_control_on_a_trending_universe():
    T, N = 60, 6
    dates = np.arange(np.datetime64("2010-01-04"), np.datetime64("2010-01-04") + T)
    O = np.cumprod(np.full((T, N), 1.001), axis=0)       # every security rises 0.1 % a day, open to open
    ones = np.ones((T, N), bool)
    feats = SimpleNamespace(cols=np.arange(N), m={"valid": ones, "regime": ones, "O": O})
    extra = {"base_ok": ones, "adv20": np.full((T, N), 1e9)}
    panel = SimpleNamespace(shape=(T, N), tradable=ones)
    sig = np.array([10, 10, 20, 30])
    hold = np.array([3, 3, 3, 3])
    m = EV.c2_means(feats, extra, panel, dates, sig, hold, 50e6, 1.0, sims=20, seed=1)
    c = period_cost(dates, True, 1.0)[11]
    expect = 1.001 ** 3 * (1 - c) / (1 + c) - 1
    assert len(m) == 20 and np.allclose(m, expect, atol=1e-9)
