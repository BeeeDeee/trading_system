"""Research 9 (crypto trend / momentum): symbol filter, panel, trend sleeves, point in time."""

import numpy as np
import pytest

from qlab.data.panel import Panel
from qlab.research9 import strategy as S
from qlab.research9.data import tradable_spot
from qlab.research9.panel import cost_rate


def make_panel(close: np.ndarray, start="2024-01-01") -> tuple[Panel, np.ndarray]:
    T, N = close.shape
    dates = np.arange(np.datetime64(start), np.datetime64(start) + np.timedelta64(T, "D"))
    opn = np.vstack([close[:1], close[:-1]])                  # open = previous close (continuous market)
    ret_oc = close / opn - 1
    ret_co = np.zeros((T, N))
    true = np.ones((T, N), dtype=bool)
    p = Panel(dates, np.arange(N, dtype=np.int64), ret_co, ret_oc, true, true, ~true, close, np.full((T, N), 2e9))
    return p, np.full((T, N), 2e9)


def test_symbol_filter():
    s = {"BTCUSDT", "BTCUPUSDT", "ETHBEARUSDT", "ETHUSDT", "JUPUSDT", "USDCUSDT", "WBTCUSDT", "BTCBUSD"}
    assert tradable_spot(s) == ["BTCUSDT", "ETHUSDT", "JUPUSDT"]


def test_grid_26_unique():
    g = S.grid()
    assert len(g) == 26 and len({c.id for c in g}) == 26


def test_sma_and_signal():
    c = np.arange(1.0, 11.0)
    assert S.sma(c[:, None], 3)[2, 0] == pytest.approx(2.0)
    assert np.isnan(S.sma(c[:, None], 3)[1, 0])
    sig = S.trend_signal(c, S.Config("T", "sma", 3))
    assert list(sig[:2]) == [0.0, 0.0] and sig[5] == 1.0


def test_trend_sleeve_trades_only_on_signal_change():
    T = 12
    close = np.ones((T, 2)) * 100.0
    close[6:, 0] = 110.0                                       # coin 0 jumps on day 6 (held), coin 1 flat
    p, qv = make_panel(close)
    sig = {0: np.ones(T), 1: np.r_[np.ones(8), np.zeros(4)]}   # signal 0 from the close of day 8 -> sold at the open of day 9
    cost = np.full((T, 2), 0.001)
    r = S.simulate_trend(p, sig, cost, first=0)
    # day 1: coin 0 buys 50 % (+10 bps); coin 1 buys what the remaining cash allows
    b1 = 0.4995 / 1.001
    assert r.nav[1] == pytest.approx(1 - 0.0005 - b1 * 0.001)
    # day 9: only coin 1 sold (coin 0 keeps its drifted value, no trade)
    assert r.turnover[9] == pytest.approx(b1 / r.nav[8], rel=1e-9) and r.turnover[8] == 0
    assert r.n_positions[-1] == 1


def test_x_decisions_point_in_time():
    rng = np.random.default_rng(0)
    T, N = 200, 30
    close = 100 * np.cumprod(1 + rng.normal(0, 0.03, (T, N)), axis=0)
    p, qv = make_panel(close)
    qv = rng.lognormal(18, 1, (T, N))
    for cfg in [c for c in S.grid() if c.family == "X"]:
        base = S.x_decisions(p, qv, cfg, btc=0)
        cut = 150
        c2 = close.copy()
        c2[cut + 1:] *= rng.uniform(0.2, 5.0, (T - cut - 1, N))
        q2 = qv.copy()
        q2[cut + 1:] = rng.lognormal(25, 3, (T - cut - 1, N))
        p2, _ = make_panel(c2)
        pert = S.x_decisions(p2, q2, cfg, btc=0)
        k = base.days <= cut
        assert np.array_equal(base.days[k], pert.days[pert.days <= cut]), cfg.id
        assert np.allclose(base.weights[k], pert.weights[pert.days <= cut]), cfg.id


def test_trend_signal_point_in_time():
    rng = np.random.default_rng(1)
    c = 100 * np.cumprod(1 + rng.normal(0, 0.03, 300))
    for cfg in [g for g in S.grid() if g.family == "T"]:
        c2 = c.copy()
        c2[201:] *= 3
        assert np.array_equal(S.trend_signal(c, cfg)[:201], S.trend_signal(c2, cfg)[:201]), cfg.id


def test_cost_rate_tiers_use_past_volume():
    qv = np.full((40, 1), 3e8)
    qv[35:] = 1e12                                            # future volume must not lower today's cost
    c = cost_rate(qv)
    assert c[35, 0] == pytest.approx(0.0010 + 0.0005)
    assert c[0, 0] == pytest.approx(0.0010 + 0.0025)         # no history yet: worst tier
