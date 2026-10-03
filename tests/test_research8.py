"""Research 8 (funding carry): simulator arithmetic, liquidation, point in time, parsing."""

from datetime import date

import numpy as np
import pytest

from qlab.research8 import data as D, strategy as S
from qlab.research8.panel import FIELDS, Panel, mondays
from qlab.research8.sim import Costs, simulate

ZERO = Costs(spot_fee=0.0, perp_fee=0.0, tiers=((0.0, 0.0),))


def flat_panel(T=21, N=2, price=100.0, fund=0.0, start="2024-01-01", symbols=("BTCUSDT", "ETHUSDT")):
    dates = np.arange(np.datetime64(start), np.datetime64(start) + np.timedelta64(T, "D"))
    m = {k: np.full((T, N), price) for k in ("po", "ph", "pl", "pc", "so", "sc")}
    m["qv"] = np.full((T, N), 2e9)
    m["fund_hold"] = np.full((T, N), fund)
    m["fund_sig"] = np.full((T, N), fund)
    m["fund_n"] = np.full((T, N), 3.0)
    return Panel(dates, list(symbols), m, np.zeros(T))


def test_constant_prices_no_funding_costs_only_at_entry():
    p = flat_panel()
    costs = Costs()
    r = simulate(p, {3: {0: 1.0}}, costs=costs)
    # entry: spot notional 2/3, both legs: (10 + 2) + (5 + 2) bps of 2/3 NAV
    assert r.nav[3] == pytest.approx(1 - 2 / 3 * (0.0012 + 0.0007))
    assert np.allclose(r.nav[3:], r.nav[3])
    # no volume history yet -> worst slippage tier (25 bps per leg)
    r0 = simulate(p, {0: {0: 1.0}}, costs=costs)
    assert r0.nav[0] == pytest.approx(1 - 2 / 3 * (0.0035 + 0.0030))


def test_funding_income_by_hand():
    f = 0.0003                                     # per day (3 x 0.0001)
    p = flat_panel(fund=f)
    r = simulate(p, {0: {0: 0.5}}, costs=ZERO)
    # short notional = s * w * NAV = 2/3 * 0.5 -> income per day 1/3 * f, credited to the perp account
    assert r.nav[6] == pytest.approx(1 + 7 * f / 3)
    assert r.funding[0] == pytest.approx(f / 3)


def test_delta_neutral_without_costs():
    p = flat_panel(T=10)
    path = np.array([100, 110, 95, 120, 80, 100, 130, 125, 90, 100.0])
    for k in ("po", "so"):
        p.m[k][:, 0] = np.r_[100, path[:-1]]
    for k in ("pc", "sc"):
        p.m[k][:, 0] = path
    p.m["ph"][:, 0] = np.maximum(p.m["po"][:, 0], path)
    r = simulate(p, {0: {0: 1.0}}, costs=ZERO)
    assert np.allclose(r.nav, 1.0)
    assert not r.liquidations


def test_liquidation_loses_perp_account_and_sells_spot():
    p = flat_panel(T=5)
    p.m["ph"][2, 0] = 160.0                          # +60 % intraday, perp 2x on its collateral
    r = simulate(p, {0: {0: 1.0}}, costs=ZERO)
    assert r.liquidations == [(2, 0)]
    # perp account (1/3) lost, spot (2/3) sold at the unchanged close
    assert r.nav[2] == pytest.approx(2 / 3)
    assert r.nav[4] == pytest.approx(2 / 3)


def test_maintenance_rebalances_after_big_move():
    p = flat_panel(T=6)
    p.m["pc"][1, 0] = p.m["sc"][1, 0] = 126.0       # perp account drops below 50 % of its 1/3 target
    p.m["ph"][1, 0] = 126.0
    p.m["pc"][2:, 0] = p.m["sc"][2:, 0] = p.m["po"][2:, 0] = p.m["so"][2:, 0] = p.m["ph"][2:, 0] = 126.0
    r = simulate(p, {0: {0: 1.0}}, costs=Costs())
    assert r.turnover[2] > 0                         # legs re-balanced at the next open
    assert r.costs[2] > 0


def test_delisting_closes_at_last_close_with_slippage():
    p = flat_panel(T=6)
    for k in ("po", "ph", "pl", "pc"):
        p.m[k][3:, 0] = np.nan
    r = simulate(p, {0: {0: 1.0}}, costs=ZERO)
    assert r.delistings == [(3, 0)]
    assert r.nav[3] == pytest.approx(1 - 2 / 3 * 2 * 0.02)


def test_targets_are_point_in_time():
    rng = np.random.default_rng(0)
    T, N = 120, 6
    syms = ["BTCUSDT", "ETHUSDT", "AUSDT", "BUSDT", "CUSDT", "DUSDT"]
    p = flat_panel(T=T, N=N, symbols=syms)
    p.m["fund_sig"] = rng.normal(0, 3e-4, (T, N))
    p.m["qv"] = rng.lognormal(20, 1, (T, N))
    for cfg in S.grid():
        base = S.targets(p, cfg)
        for cut in (60, 90):
            q = Panel(p.dates, p.symbols, {k: v.copy() for k, v in p.m.items()}, p.tbill)
            q.m["fund_sig"][cut:] = rng.normal(0, 1e-2, (T - cut, N))
            q.m["qv"][cut:] = rng.lognormal(25, 3, (T - cut, N))
            pert = S.targets(q, cfg)
            assert {t: w for t, w in base.items() if t <= cut} == {t: w for t, w in pert.items() if t <= cut}, cfg.id


def test_grid_has_17_candidates_and_unique_ids():
    g = S.grid()
    assert len(g) == 17 and len({c.id for c in g}) == 17


def test_mondays():
    d = np.arange(np.datetime64("2024-01-01"), np.datetime64("2024-01-15"))   # 2024-01-01 was a Monday
    assert list(mondays(d)) == [0, 7]


def test_spot_pair_mapping():
    spot = {"BTCUSDT", "PEPEUSDT", "1000SATSUSDT"}
    assert D.spot_pair("BTCUSDT", spot) == ("BTCUSDT", 1.0)
    assert D.spot_pair("1000PEPEUSDT", spot) == ("PEPEUSDT", 1000.0)
    assert D.spot_pair("1000SATSUSDT", spot) == ("1000SATSUSDT", 1.0)
    assert D.spot_pair("USDCUSDT", spot) is None
    assert D.spot_pair("BTCUSDT_230331", spot) is None


def test_invalid_pair_forces_exit_at_actual_prices():
    p = flat_panel(T=6)
    p.m["sc"][2, 0] = 70.0                           # spot close 30 % below perp: pair invalid on day 2
    p.m["so"][3:, 0] = p.m["sc"][3:, 0] = 70.0       # and the spot stays there
    r = simulate(p, {0: {0: 1.0}}, costs=ZERO)
    assert r.mismatches == [(3, 0)]
    # spot leg lost 30 % on 2/3 of NAV, perp unchanged, then 2 % on both legs at the next open
    q = 2 / 3 / 100
    assert r.nav[3] == pytest.approx(1 - 2 / 3 * 0.3 - q * 0.02 * (70 + 100))


def test_universe_needs_valid_pair_for_30_days():
    syms = ["BTCUSDT", "ETHUSDT", "AUSDT"]
    p = flat_panel(T=70, N=3, symbols=syms)
    p.m["sc"][50, 2] = 50.0
    assert 2 in S.universe(p, 45, 3)
    assert 2 not in S.universe(p, 60, 3)
