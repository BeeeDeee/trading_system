"""Research 10: signed simulator, sleeve mixing, TSMOM and crypto long/short decisions, point in time."""

import numpy as np
import pytest

from qlab.data.panel import Panel
from qlab.engine.vector import Decisions, simulate
from qlab.research10 import crypto as C, tsmom as B
from qlab.research10.sim import monthly_mix, simulate_signed


def make_panel(T=300, N=4, seed=0, start="2010-01-01"):
    rng = np.random.default_rng(seed)
    dates = np.arange(np.datetime64(start), np.datetime64(start) + np.timedelta64(T, "D"))
    co, oc = rng.normal(0, 0.005, (T, N)), rng.normal(0, 0.01, (T, N))
    true = np.ones((T, N), dtype=bool)
    close = 100 * np.cumprod((1 + co) * (1 + oc), axis=0)
    return Panel(dates, np.arange(N, dtype=np.int64), co, oc, true, true, ~true, close, np.full((T, N), 1e9))


def test_long_only_parity_with_vector_engine():
    p = make_panel()
    rng = np.random.default_rng(1)
    days = np.arange(5, 290, 21)
    w = rng.dirichlet(np.ones(4), len(days)) * 0.9
    cash = np.full(len(p.dates), 0.0001)
    a = simulate(p, Decisions(days, w), 0.001, cash)
    b = simulate_signed(p.ret_co, p.ret_oc, days, w, 0.001, cash)
    assert np.allclose(a.returns, b.returns, atol=1e-14)
    assert np.allclose(a.costs, b.costs, atol=1e-15)


def test_short_pnl_carry_and_no_interest_on_proceeds():
    T = 4
    co, oc = np.zeros((T, 1)), np.zeros((T, 1))
    oc[2, 0] = 0.10                                     # price +10 % on day 2
    r = simulate_signed(co, oc, np.array([0]), np.array([[-0.5]]), 0.0, cash_ret=np.full(T, 0.01),
                        short_carry=np.full((T, 1), 0.001))
    nav0 = 1.01                                         # day 0: interest on 1.0, no position
    open1 = nav0 * 1.01                                 # day 1: interest before the fill
    short = -0.5 * open1
    nav1 = open1 - 0.001 * short                        # carry on the short held over day 1
    assert r.nav[1] == pytest.approx(nav1)
    cash = nav1 - short
    cash += (cash + short) * 0.01                       # day 2: proceeds earn nothing
    cash -= 0.001 * short                               # carry valued at the open (no overnight move)
    assert r.nav[2] == pytest.approx(cash + short * 1.10)
    assert r.returns[2] < 0                             # short loses when the price rises


def test_nan_target_keeps_position_and_gross_limit():
    p = make_panel(N=2)
    r = simulate_signed(p.ret_co, p.ret_oc, np.array([0, 50]), np.array([[0.5, -0.5], [np.nan, 0.0]]), 0.0)
    assert r.gross[60] == pytest.approx(r.net[60]) and r.net[60] > 0   # only the long leg left
    assert r.turnover[51] == pytest.approx((r.gross[50] - r.net[50]) / 2, rel=0.05)  # closed the short only
    with pytest.raises(ValueError):
        simulate_signed(p.ret_co, p.ret_oc, np.array([0]), np.array([[0.7, -0.7]]), 0.0)


def test_monthly_mix_matches_explicit_loop():
    rng = np.random.default_rng(2)
    dates = np.arange(np.datetime64("2010-01-01"), np.datetime64("2010-06-01"))
    r = rng.normal(0, 0.01, (len(dates), 2))
    out = monthly_mix(r, np.array([0.3, 0.7]), dates)
    month = dates.astype("datetime64[M]")
    v, ref = np.array([0.3, 0.7]), []
    for t in range(len(dates)):
        if t > 0 and month[t - 1] != (month[t - 2] if t > 1 else None):   # day after a month's first day
            v = v.sum() * np.array([0.3, 0.7])
        before = v.sum()
        v = v * (1 + r[t])
        ref.append(v.sum() / before - 1)
    assert np.allclose(out, ref)


def test_tsmom_point_in_time_and_weights():
    p = make_panel(T=800, N=4, seed=3, start="2005-01-03")
    cash = np.full(800, 0.0001)
    for cfg in B.grid():
        d, w = B.decisions(p, cash, [0, 1, 2, 3], cfg)
        assert (np.abs(w).sum(axis=1) <= 1 + 1e-12).all()
        if cfg.kind == "LO":
            assert (w >= 0).all()
        cut = 500
        p2 = make_panel(T=800, N=4, seed=3, start="2005-01-03")
        p2.ret_oc[cut + 1:] *= -3
        d2, w2 = B.decisions(p2, cash, [0, 1, 2, 3], cfg)
        k = d <= cut
        assert np.array_equal(d[k], d2[d2 <= cut]) and np.allclose(w[k], w2[d2 <= cut]), cfg.id
    d, w = B.decisions(p, cash, [0, 1, 2, 3], B.Config("LS", 12, "eq"))
    assert np.allclose(np.abs(w[-1]).sum(), 1.0)


def test_crypto_signal_point_in_time_and_decisions():
    rng = np.random.default_rng(4)
    c = 100 * np.cumprod(1 + rng.normal(0, 0.03, 400))
    for n in (20, 50, 100, 200):
        c2 = c.copy()
        c2[301:] *= 0.2
        assert np.array_equal(C.signal(c, n)[:301], C.signal(c2, n)[:301])
    s = {0: C.signal(c, 20), 1: C.signal(c[::-1].copy(), 20)}
    d, w = C.decisions(s, 3, first=30)
    assert d[0] == 30 and np.allclose(w[0, :2], 0.5 * np.array([s[0][30], s[1][30]])) and np.isnan(w[0, 2])
    for k in range(1, len(d)):
        t = d[k]
        for j in (0, 1):
            assert np.isnan(w[k, j]) == (s[j][t] == s[j][t - 1])
    _, ws = C.decisions(s, 3, first=30, short_only=True)
    assert np.nanmax(ws) <= 0
