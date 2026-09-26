import numpy as np

from qlab.engine.vector import Decisions
from qlab.selection.select import select
from qlab.strategies.grid import StrategyConfig
from qlab.validation.registry import TrialRegistry
from qlab.validation.wfo import SparseDecisions, make_folds, stitch_decisions

CFG = {"hard_filters": {"max_drawdown": 0.45, "min_sharpe": 0.0, "min_share_positive_years": 0.0,
                        "max_turnover_annual": 20.0, "min_avg_positions": 1},
       "ranking": {"dedup_correlation": 0.85}}


def test_make_folds_with_embargo():
    dates = np.arange(np.datetime64("2004-12-01"), np.datetime64("2006-02-01"))
    folds = make_folds(dates, 2005, 2006, train_start=0, embargo=5)
    assert [f.year for f in folds] == [2005, 2006]
    f = folds[0]
    assert str(dates[f.test_start]) == "2005-01-01" and str(dates[f.test_end - 1]) == "2005-12-31"
    assert f.train_end == f.test_start - 5


def test_select_picks_best_of_distinct_clusters():
    rng = np.random.default_rng(0)
    n = 1500
    a, b = rng.normal(0.001, 0.01, n), rng.normal(0.0008, 0.01, n)
    R = np.column_stack([a, a + rng.normal(0, 0.001, n), b, rng.normal(-0.001, 0.01, n)])
    grid = [StrategyConfig("low_vol", (("lookback", 63),), 10)] * 4
    neigh = [np.array([i]) for i in range(4)]
    dates = np.arange(np.datetime64("2010-01-01"), np.datetime64("2010-01-01") + np.timedelta64(n, "D"))
    pol = select(R, np.zeros(n), dates, np.zeros((n, 4)), np.full((n, 4), 10), grid, neigh, CFG, k=2)
    assert set(pol.members.tolist()) == {int(np.argmax([R[:, 0].mean(), R[:, 1].mean()])), 2}
    assert pol.n_passed == 3 and pol.weight_each == 0.5


def test_stitch_switches_members_between_folds():
    d0 = SparseDecisions(Decisions(np.array([0, 10, 20]), np.array([[1.0, 0, 0], [0.5, 0.5, 0], [1, 0, 0]])))
    d1 = SparseDecisions(Decisions(np.array([5, 15]), np.array([[0, 0, 1.0], [0, 1.0, 0]])))
    from qlab.validation.wfo import Fold
    folds = [Fold(1, 0, 3, 8, 14), Fold(2, 0, 9, 14, 30)]
    d = stitch_decisions(folds, [[0], [0, 1]], 0.5, {0: d0, 1: d1}, 3)
    assert d.days.tolist() == [7, 10, 13, 15, 20]
    np.testing.assert_allclose(d.weights[0], [0.5, 0, 0])          # member 0 alone, half capital
    np.testing.assert_allclose(d.weights[2], [0.25, 0.25, 0.5])    # fold 2 start: both members
    np.testing.assert_allclose(d.weights[4], [0.5, 0.5, 0])


def test_registry_has(tmp_path):
    reg = TrialRegistry(tmp_path / "t.sqlite")
    assert not reg.has({"k": 2})
    reg.record("methodology_eval", {"k": 2})
    assert reg.has({"k": 2}) and not reg.has({"k": 3})
