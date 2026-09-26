from pathlib import Path

import numpy as np
import pytest
import yaml

from qlab.data.normalize import normalize_prices
from qlab.data.panel import panel_from_bars
from qlab.data.synthetic import generate_market
from qlab.strategies.grid import StrategyConfig, build_grid, neighbours
from qlab.schedule import period_starts
from qlab.strategies.portfolio import PortfolioSpec, build_decisions, cap_weights
from qlab.strategies.run import Context, decisions_for, dense_targets, run_candidate
from qlab.strategies.signals import Signal
from qlab.validation.leakage import assert_point_in_time

GRID = yaml.safe_load((Path(__file__).parents[1] / "configs" / "frozen_defaults.yaml")
                      .read_text())["grid"]

SMALL = {  # short lookbacks so a 2-year synthetic market exercises every family
    "xs_momentum": {"lookback": 20, "skip": 5},
    "ts_trend": {"sma": 30},
    "breakout": {"lookback": 30, "recent_days": 5},
    "st_reversal": {"lookback": 5},
    "low_vol": {"lookback": 20},
    "regime_market": {"sma": 30},
}


@pytest.fixture(scope="module")
def panel():
    m = generate_market(n_assets=30, seed=23, start="2010-01-01", end="2011-12-31")
    return panel_from_bars(normalize_prices(m.prices, m.delistings, m.calendar), m.calendar)


def context(panel):
    universe = panel.listed & ~panel.delisting
    universe[:, 0] = False  # column 0 plays SPY
    return Context(panel, universe, spy=0)


def config(family, **kw):
    base = dict(top_n=5, weighting="inverse_vol", rebalance="weekly", hysteresis=1.5,
                overlay="trend_filter")
    base.update(kw)
    if family == "regime_market":
        base = dict(rebalance="weekly")
    return StrategyConfig(family, tuple(sorted(SMALL[family].items())), **base)


@pytest.mark.parametrize("family", list(SMALL))
def test_every_family_is_point_in_time(panel, family):
    cfg = config(family)
    fn = lambda p: dense_targets(decisions_for(cfg, context(p)), p.shape)  # noqa: E731
    assert_point_in_time(fn, panel, family, n_cuts=6, min_day=60)


@pytest.mark.parametrize("family", list(SMALL))
def test_every_family_runs_and_respects_limits(panel, family):
    cfg = config(family, overlay="none")
    ctx = context(panel)
    d = decisions_for(cfg, ctx)
    assert (d.weights >= 0).all() and (d.weights.sum(axis=1) <= 1 + 1e-9).all()
    if family != "regime_market":
        assert (d.weights <= 0.10 + 1e-12).all() or cfg.top_n < 10
        assert not d.weights[:, 0].any()  # SPY is not in the stock universe
        assert (np.count_nonzero(d.weights, axis=1) <= cfg.top_n).all()
    r = run_candidate(cfg, ctx)
    assert np.isfinite(r.nav).all()
    assert r.exposure.max() > 0


def test_hysteresis_reduces_turnover(panel):
    ctx = context(panel)
    tight = run_candidate(config("xs_momentum", hysteresis=1.0, overlay="none"), ctx)
    loose = run_candidate(config("xs_momentum", hysteresis=2.0, overlay="none"), ctx)
    assert loose.turnover.sum() < tight.turnover.sum()


def test_selection_rules_by_hand():
    score = np.array([[3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]])
    entry = np.ones_like(score, bool)
    exits = np.zeros_like(score, bool)
    exits[2, 3] = True
    sig = Signal(score, entry, exits)
    universe = np.ones_like(score, bool)
    d = build_decisions(sig, universe, np.array([True, True, True]),
                        PortfolioSpec(top_n=2, hysteresis=1.5, max_weight=1.0))
    assert d.weights[0].tolist() == [0.5, 0.5, 0, 0]
    # day 1: previous picks ranked 4 and 3; rank 3 <= 2*1.5 stays, rank 4 is replaced by the best
    assert d.weights[1].tolist() == [0, 0.5, 0, 0.5]
    # day 2: asset 3 must exit although it ranks first
    assert d.weights[2].tolist() == [0, 0.5, 0.5, 0]


def test_cap_weights():
    w = cap_weights(np.array([0.5, 0.3, 0.1, 0.1]), 0.3)
    np.testing.assert_allclose(w, [0.3, 0.3, 0.2, 0.2])
    np.testing.assert_allclose(cap_weights(np.array([0.6, 0.4]), 0.3), [0.3, 0.3])  # rest is cash


def test_period_starts():
    d = np.array(["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-12", "2024-01-16",
                  "2024-02-01"], dtype="datetime64[D]")
    assert period_starts(d, "W").tolist() == [True, False, True, False, True, True]
    assert period_starts(d, "M").tolist() == [True, False, False, False, False, True]


def test_grid():
    grid = build_grid(GRID)
    assert len(grid) == 21 * 144 + 3
    assert len({c.candidate_id for c in grid}) == len(grid)
    a = next(c for c in grid if c.family == "xs_momentum")
    near = [c for c in grid if neighbours(a, c, GRID)]
    assert a in near and 1 < len(near) < 30
