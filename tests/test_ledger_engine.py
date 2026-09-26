import numpy as np
import pytest

from qlab.data.normalize import normalize_prices
from qlab.data.panel import Panel, panel_from_bars
from qlab.data.synthetic import generate_market
from qlab.engine.costs import CostModel
from qlab.engine.ledger import LedgerConfig, simulate_ledger
from qlab.engine.vector import simulate


def random_case(seed, n_days=400, n_assets=25):
    rng = np.random.default_rng(seed)
    shape = (n_days, n_assets)
    listed = np.ones(shape, bool)
    delisting = np.zeros(shape, bool)
    ret_co = rng.normal(0, 0.01, shape)
    ret_oc = rng.normal(0, 0.02, shape)
    tradable = rng.random(shape) > 0.05
    for a in rng.choice(n_assets, 5, replace=False):  # some assets get delisted
        d = int(rng.integers(50, n_days - 1))
        delisting[d, a] = True
        ret_co[d, a] = rng.choice([-1.0, -0.3, 0.0, 0.25])
        ret_oc[d, a] = 0.0
        tradable[d:, a] = False
        listed[d + 1:, a] = False
        ret_co[d + 1:, a] = ret_oc[d + 1:, a] = 0.0
    panel = Panel(np.arange(n_days).astype("datetime64[D]"), np.arange(n_assets), ret_co, ret_oc,
                  tradable, listed, delisting, np.full(shape, 20.0), np.full(shape, 1e6))
    targets = np.full(shape, np.nan)
    step = int(rng.integers(1, 15))
    for t in range(0, n_days, step):
        w = rng.random(n_assets) * (rng.random(n_assets) < 0.4) * listed[t]
        if w.sum() > 0:
            w = w / w.sum() * rng.uniform(0.5, 1.0)
        targets[t] = w
    cost = CostModel().rate(rng.uniform(1, 2000, shape))
    cash_ret = rng.uniform(0, 2e-4, n_days)
    return panel, targets, cost, cash_ret


@pytest.mark.parametrize("seed", range(8))
def test_parity_with_vector_engine(seed):
    panel, targets, cost, cash_ret = random_case(seed)
    v = simulate(panel, targets, cost, cash_ret)
    ledger = simulate_ledger(panel, targets, cost, cash_ret)
    np.testing.assert_allclose(ledger.sim.returns, v.returns, rtol=0, atol=1e-10)
    np.testing.assert_allclose(ledger.sim.nav / 10_000, v.nav, rtol=1e-9)
    np.testing.assert_allclose(ledger.sim.turnover, v.turnover, atol=1e-10)
    np.testing.assert_allclose(ledger.sim.costs, v.costs, atol=1e-12)
    np.testing.assert_allclose(ledger.sim.exposure, v.exposure, atol=1e-10)


def test_parity_on_synthetic_market():
    m = generate_market(n_assets=40, seed=21)
    p = panel_from_bars(normalize_prices(m.prices, m.delistings, m.calendar), m.calendar)
    targets = np.full(p.shape, np.nan)
    for t in range(0, p.shape[0], 5):
        ok = p.tradable[t] & ~p.delisting[t]
        targets[t] = ok / max(ok.sum(), 1)
    cost = CostModel().rate(np.full(p.shape, 700.0))
    v = simulate(p, targets, cost)
    ledger = simulate_ledger(p, targets, cost)
    np.testing.assert_allclose(ledger.sim.returns, v.returns, atol=1e-10)


def test_fixed_fee_and_order_log():
    panel, targets, _, _ = random_case(1)
    base = simulate_ledger(panel, targets, 0.0)
    fee = simulate_ledger(panel, targets, 0.0, config=LedgerConfig(fee_per_order=1.0))
    n_orders = fee.orders.filter(fee.orders["side"] != "delisting").height
    assert fee.orders["cost"].sum() == pytest.approx(n_orders * 1.0)
    assert fee.sim.nav[-1] < base.sim.nav[-1]
    assert set(base.orders["side"]) <= {"buy", "sell", "delisting"}


def test_no_trade_band_reduces_orders():
    panel, targets, cost, _ = random_case(2)
    targets[~np.isnan(targets).all(axis=1)] = 1.0 / panel.shape[1]  # rebalance to equal weight
    base = simulate_ledger(panel, targets, cost)
    banded = simulate_ledger(panel, targets, cost, config=LedgerConfig(no_trade_band=0.01))
    assert banded.orders.height < base.orders.height / 2
    assert banded.sim.costs.sum() < base.sim.costs.sum()


def test_never_negative_cash_with_fees():
    panel, targets, cost, _ = random_case(3)
    targets[~np.isnan(targets).all(axis=1)] = 1.0 / panel.shape[1]
    r = simulate_ledger(panel, targets, cost, config=LedgerConfig(fee_per_order=5.0))
    assert (r.sim.exposure <= 1 + 1e-12).all()
