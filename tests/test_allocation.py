import numpy as np
import pytest

from qlab.data.normalize import normalize_prices
from qlab.data.panel import panel_from_bars
from qlab.data.synthetic import generate_market
from qlab.schedule import period_starts
from qlab.strategies.allocation import (H1, AllocationConfig, allocation_decisions,
                                        allocation_grid, allocation_neighbours)
from qlab.strategies.run import dense_targets
from qlab.validation.leakage import assert_point_in_time


@pytest.fixture(scope="module")
def panel():
    m = generate_market(n_assets=9, seed=31, start="2010-01-01", end="2013-12-31", p_delist=0.0)
    return panel_from_bars(normalize_prices(m.prices, m.delistings, m.calendar), m.calendar)


def decisions(panel, cfg, min_history=60):
    cash = np.full(panel.shape[0], 1e-4)
    return allocation_decisions(panel, list(range(panel.shape[1])), cfg, cash,
                                period_starts(panel.dates, "M"), min_history)


@pytest.mark.parametrize("cfg", [H1, AllocationConfig("equal", "tsmom126", "top3_126"),
                                 AllocationConfig("iv63", "sma100", "top5_252")])
def test_point_in_time(panel, cfg):
    fn = lambda p: dense_targets(decisions(p, cfg), p.shape)  # noqa: E731
    assert_point_in_time(fn, panel, cfg.key, n_cuts=6, min_day=300)


def test_rules(panel):
    d = decisions(panel, AllocationConfig("equal", "none", "top3_126"))
    late = d.weights[d.days > 300]
    assert (np.count_nonzero(late, axis=1) == 3).all()
    np.testing.assert_allclose(late.sum(axis=1), 1.0)
    d = decisions(panel, H1)
    assert (d.weights <= 0.40 + 1e-12).all() and (d.weights.sum(axis=1) <= 1 + 1e-9).all()
    none = decisions(panel, AllocationConfig("iv126", "none", "none"))
    # the trend filter only ever removes weight (to cash), never adds
    assert (d.weights <= none.weights + 1e-12).all()


def test_grid_and_neighbours():
    grid = allocation_grid()
    assert len(grid) == 100 and len({c.key for c in grid}) == 100
    neigh = allocation_neighbours(grid)
    i = grid.index(H1)
    names = {grid[j].key for j in neigh[i]}
    assert "iv63|sma200|none" in names and "iv126|sma100|none" in names
    assert "iv63|sma100|none" not in names  # two dimensions differ
