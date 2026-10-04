"""The lab engine against two independent references: qlab's vector engine (long-only, delistings) and
research 10's signed simulator (shorts)."""

import numpy as np
import pytest

from qlab.data.panel import Panel
from qlab.engine.vector import simulate as vector_simulate
from qlab.research10.sim import simulate_signed

from lab.framework.engine import simulate


def random_market(T=400, N=8, seed=0, delist=True):
    rng = np.random.default_rng(seed)
    ret_co, ret_oc = rng.normal(0, 0.005, (T, N)), rng.normal(0.0003, 0.01, (T, N))
    tradable = rng.random((T, N)) > 0.03
    listed = np.ones((T, N), dtype=bool)
    delisting = np.zeros((T, N), dtype=bool)
    if delist:
        for j, last in ((2, 150), (5, 300)):
            delisting[last, j], ret_co[last, j], tradable[last, j] = True, -0.3, False
            ret_co[last + 1:, j] = ret_oc[last:, j] = 0.0
            tradable[last + 1:, j] = listed[last + 1:, j] = False
    return ret_co, ret_oc, tradable, listed, delisting


def decisions(T, N, seed, long_only=True, gross=0.95):
    rng = np.random.default_rng(seed)
    w = np.full((T, N), np.nan)
    for t in range(0, T, 7):
        x = rng.random(N) if long_only else rng.normal(0, 1, N)
        w[t] = x / np.abs(x).sum() * gross
    return w


@pytest.mark.parametrize("seed", range(4))
def test_long_only_matches_qlab_vector_engine(seed):
    ret_co, ret_oc, tradable, listed, delisting = random_market(seed=seed)
    T, N = ret_co.shape
    w = decisions(T, N, seed)                         # also targets delisted assets: no fill in either engine
    cost = np.random.default_rng(seed).uniform(0.0005, 0.002, (T, N))
    cash = np.full(T, 0.0001)
    panel = Panel(np.arange(T).astype("datetime64[D]"), np.arange(N), ret_co, ret_oc, tradable, listed, delisting,
                  np.ones((T, N)), np.ones((T, N)))
    ref = vector_simulate(panel, w, cost, cash)
    out = simulate(ret_co, ret_oc, w, cost, tradable, delisting, cash)
    np.testing.assert_allclose(out.returns, ref.returns, rtol=0, atol=1e-13)
    np.testing.assert_allclose(out.turnover, ref.turnover, rtol=0, atol=1e-13)


def test_full_investment_with_costs_never_borrows():
    ret_co, ret_oc, tradable, listed, delisting = random_market(delist=False)
    T, N = ret_co.shape
    w = decisions(T, N, 1, gross=1.0)
    out = simulate(ret_co, ret_oc, w, 0.002, np.ones_like(tradable), delisting)
    assert (out.net <= 1 + 1e-12).all()


@pytest.mark.parametrize("seed", range(3))
def test_long_short_matches_signed_simulator(seed):
    ret_co, ret_oc, tradable, _, delisting = random_market(seed=seed, delist=False)
    T, N = ret_co.shape
    w = decisions(T, N, seed, long_only=False)
    days = np.flatnonzero(~np.isnan(w).all(axis=1))
    carry = np.random.default_rng(seed).normal(0.0001, 0.0001, (T, N))
    cash = np.full(T, 0.0001)
    ref = simulate_signed(ret_co, ret_oc, days, w[days], 0.001, cash, carry, tradable)
    out = simulate(ret_co, ret_oc, w, 0.001, tradable, delisting, cash, carry)
    np.testing.assert_allclose(out.returns, ref.returns, rtol=0, atol=1e-13)
    np.testing.assert_allclose(out.gross, ref.gross, rtol=0, atol=1e-13)


def test_delisted_position_is_paid_to_cash():
    ret_co, ret_oc, tradable, _, delisting = random_market(seed=3)
    T, N = ret_co.shape
    w = np.full((T, N), np.nan)
    w[10] = 0.0
    w[10, 2] = 1.0                                   # all in asset 2, which delists at row 150 at -30 %
    out = simulate(ret_co, ret_oc, w, 0.0, np.ones_like(tradable), delisting)
    assert out.held[150, 2] == 0.0 and out.net[151] == 0.0
    assert out.nav[150] == pytest.approx(out.nav[149] * 0.7)


def test_leverage_is_refused():
    ret_co, ret_oc, tradable, _, delisting = random_market()
    w = decisions(*ret_co.shape, 0, gross=1.2)
    with pytest.raises(ValueError, match="gross weight above 1"):
        simulate(ret_co, ret_oc, w, 0.0, tradable, delisting)
