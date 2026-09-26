import numpy as np
import polars as pl
import pytest

from qlab.data.normalize import normalize_prices
from qlab.data.panel import Panel, panel_from_bars
from qlab.data.synthetic import generate_market
from qlab.engine.costs import CostModel, cash_returns
from qlab.engine.vector import simulate


def make_panel(ret_co, ret_oc, tradable=None, delisting=None):
    ret_co, ret_oc = np.asarray(ret_co, float), np.asarray(ret_oc, float)
    shape = ret_co.shape
    return Panel(
        dates=np.arange(np.datetime64("2020-01-01"), np.datetime64("2020-01-01") + np.timedelta64(shape[0], "D")),
        assets=np.arange(shape[1]),
        ret_co=ret_co, ret_oc=ret_oc,
        tradable=np.ones(shape, bool) if tradable is None else np.asarray(tradable),
        listed=np.ones(shape, bool),
        delisting=np.zeros(shape, bool) if delisting is None else np.asarray(delisting),
        close_u=np.full(shape, 10.0), dollar_volume=np.full(shape, 1e6))


def hold(n_days, n_assets):
    return np.full((n_days, n_assets), np.nan)


def test_no_targets_is_cash():
    p = make_panel(np.full((5, 2), 0.01), np.full((5, 2), 0.02))
    rf = np.array([0.0, 0.001, 0.001, 0.002, 0.0])
    r = simulate(p, hold(5, 2), 0.0, rf)
    np.testing.assert_allclose(r.nav, np.cumprod(1 + rf))
    assert (r.exposure == 0).all()


def test_timing_overnight_old_weights_intraday_new():
    # Decide on day 0 to buy asset 0; filled at open of day 1 -> earns only intraday of day 1.
    ret_co = [[0.10], [0.05], [0.00]]
    ret_oc = [[0.10], [0.02], [0.03]]
    targets = hold(3, 1)
    targets[0] = [1.0]
    r = simulate(make_panel(ret_co, ret_oc), targets, 0.0)
    np.testing.assert_allclose(r.nav, [1.0, 1.02, 1.02 * 1.03])


def test_costs_exact():
    targets = hold(3, 2)
    targets[0] = [1.0, 0.0]
    targets[1] = [0.0, 1.0]
    c = 0.001
    r = simulate(make_panel(np.zeros((3, 2)), np.zeros((3, 2))), targets, c)
    after_buy = 1 / (1 + c)                        # all cash spent incl. cost
    after_switch = after_buy * (1 - c) / (1 + c)   # sell A (cost), buy B with proceeds
    np.testing.assert_allclose(r.nav, [1.0, after_buy, after_switch])
    np.testing.assert_allclose(r.costs[1], c * after_buy)
    np.testing.assert_allclose(r.turnover[2], 1 + (1 - c) / (1 + c), rtol=1e-12)


def test_untradable_at_execution_keeps_position():
    targets = hold(3, 1)
    targets[0] = [1.0]
    tradable = np.array([[True], [False], [True]])
    r = simulate(make_panel(np.zeros((3, 1)), np.full((3, 1), 0.01), tradable), targets, 0.0)
    assert r.exposure[1] == 0 and r.exposure[2] == 0  # order lapsed, not retried without decision


def test_delisting_pays_terminal_return_into_cash():
    ret_co = np.array([[0.0], [0.0], [0.25], [0.0]])  # day 2 is the delisting row: +25 % payout
    delisting = np.array([[False], [False], [True], [False]])
    targets = hold(4, 1)
    targets[0] = [1.0]
    r = simulate(make_panel(ret_co, np.zeros((4, 1)), delisting=delisting), targets, 0.0,
                 np.array([0, 0, 0, 0.01]))
    np.testing.assert_allclose(r.nav, [1.0, 1.0, 1.25, 1.25 * 1.01])
    assert r.exposure[2] == 0


def test_rejects_invalid_targets():
    p = make_panel(np.zeros((2, 2)), np.zeros((2, 2)))
    for bad in ([[-0.1, 0.5]], [[0.7, 0.7]], [[np.nan, 0.5]]):
        t = hold(2, 2)
        t[0] = bad[0]
        with pytest.raises(ValueError):
            simulate(p, t, 0.0)


def test_buy_and_hold_matches_total_return_on_synthetic_market():
    m = generate_market(n_assets=30, seed=11)
    bars = normalize_prices(m.prices, m.delistings, m.calendar)
    p = panel_from_bars(bars, m.calendar)
    full = p.listed.all(axis=0) & ~p.delisting.any(axis=0) & p.tradable[1]
    a = int(np.flatnonzero(full)[0])
    targets = hold(*p.shape)
    targets[0] = 0.0
    targets[0, a] = 1.0
    r = simulate(p, targets, 0.0)
    tr = m.truth.filter(pl.col("permaticker") == int(p.assets[a])).sort("date")["tr_index"]
    # Bought at the open of day 1: total return from that open to the last close.
    expected = tr[-1] / tr[0] / (1 + p.ret_co[1, a])
    np.testing.assert_allclose(r.nav[-1], expected, rtol=1e-9)
    assert r.exposure[1:].min() > 0.999


def test_equal_weight_portfolio_survives_delistings():
    m = generate_market(n_assets=40, seed=5)
    p = panel_from_bars(normalize_prices(m.prices, m.delistings, m.calendar), m.calendar)
    targets = hold(*p.shape)
    for t in range(0, p.shape[0], 21):
        eligible = p.tradable[t] & ~p.delisting[t]
        targets[t] = eligible / eligible.sum()
    r = simulate(p, targets, CostModel().rate(np.full(p.shape, 300.0)))
    assert np.isfinite(r.nav).all() and (r.nav > 0).all()
    assert (r.exposure <= 1 + 1e-12).all() and (r.exposure >= 0).all()
    assert r.costs.sum() > 0


def test_cost_model():
    cm = CostModel(commission_bps=1.0)
    ranks = np.array([1.0, 350.0, 900.0, 5000.0, np.nan])
    np.testing.assert_allclose(cm.rate(ranks), np.array([6.0, 7.0, 13.0, 26.0, 26.0]) / 1e4)
    dates = np.array(["2000-06-01", "2010-06-01"], dtype="datetime64[D]")
    both = cm.rate(np.array([[1.0, 900.0], [1.0, 900.0]]), dates)
    np.testing.assert_allclose(both, np.array([[8.5, 31.0], [6.0, 13.0]]) / 1e4)
    np.testing.assert_allclose(CostModel(multiplier=3).rate(np.array([1.0])), [15e-4])


def test_cash_returns_use_previous_rate_act_360():
    dates = np.array(["2024-01-05", "2024-01-08", "2024-01-09"], dtype="datetime64[D]")
    rate_dates = np.array(["2024-01-04", "2024-01-08"], dtype="datetime64[D]")
    r = cash_returns(dates, rate_dates, np.array([3.6, 7.2]))
    np.testing.assert_allclose(r, [0.0, 0.036 * 3 / 360, 0.072 / 360])


def test_sparse_decisions_equal_dense():
    from qlab.engine.vector import Decisions
    rng = np.random.default_rng(0)
    p = make_panel(rng.normal(0, 0.01, (50, 4)), rng.normal(0, 0.01, (50, 4)))
    dense = hold(50, 4)
    days = np.array([0, 7, 20, 33])
    w = rng.dirichlet(np.ones(4), size=4) * 0.9
    dense[days] = w
    a = simulate(p, dense, 0.001)
    b = simulate(p, Decisions(days, w), 0.001)
    np.testing.assert_array_equal(a.nav, b.nav)
    with pytest.raises(ValueError):
        simulate(p, Decisions(np.array([5, 5]), w[:2]), 0.0)


def test_repeated_full_investment_stays_finite():
    # Regression: fully invested, then the same target again -> no buys needed while rounding may
    # leave cash at -1e-17; must not produce 0/0.
    rng = np.random.default_rng(3)
    p = make_panel(rng.normal(0, 0.01, (400, 1)), rng.normal(0, 0.01, (400, 1)))
    targets = hold(400, 1)
    targets[::5] = 1.0
    r = simulate(p, targets, 0.0025, np.full(400, 1e-4))
    assert np.isfinite(r.nav).all()
