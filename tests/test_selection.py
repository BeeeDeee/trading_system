import numpy as np
import pytest

from qlab.selection.metrics import cagr, max_drawdown, sharpe, share_positive_years
from qlab.selection.ranking import (cluster_representatives, correlation_clusters, hard_filter,
                                    robust_score)


def test_column_metrics_match_scalar_versions():
    from qlab.validation.metrics import summary
    rng = np.random.default_rng(0)
    R = rng.normal(0.0004, 0.01, (504, 3))
    rf = np.full(504, 0.0001)
    for k in range(3):
        s = summary(R[:, k], rf)
        assert sharpe(R, rf)[k] == pytest.approx(s["sharpe"])
        assert cagr(R)[k] == pytest.approx(s["cagr"])
        assert max_drawdown(R)[k] == pytest.approx(s["max_drawdown"])
    dates = np.arange(np.datetime64("2019-01-01"), np.datetime64("2019-01-01") + np.timedelta64(504, "D"))
    assert share_positive_years(np.full((504, 1), 0.001), rf, dates)[0] == 1.0


def test_hard_filter():
    m = {"max_drawdown": np.array([0.3, 0.5, 0.3]), "sharpe": np.array([0.5, 0.5, -0.1]),
         "share_positive_years": np.array([0.7, 0.7, 0.7]),
         "turnover_annual": np.array([5.0, 5.0, 5.0]), "avg_positions": np.array([20, 20, 20.0])}
    cfg = {"max_drawdown": 0.45, "min_sharpe": 0.0, "min_share_positive_years": 0.6,
           "max_turnover_annual": 20.0, "min_avg_positions": 8}
    assert hard_filter(m, cfg).tolist() == [True, False, False]


def test_robust_score_penalizes_isolated_peak():
    sr = np.array([0.1, 1.0, 0.1])
    neigh = [np.array([0, 1]), np.array([0, 1, 2]), np.array([1, 2])]
    score, med = robust_score(sr, neigh)
    assert score[1] == pytest.approx(0.1) and med[1] == pytest.approx(0.1)


def test_clusters_and_representatives():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(2000, 2))
    R = np.column_stack([base[:, 0], base[:, 0] + 0.1 * rng.normal(size=2000),
                         base[:, 1], base[:, 1] + 0.1 * rng.normal(size=2000), rng.normal(size=2000)])
    labels = correlation_clusters(R, 0.85)
    assert labels[0] == labels[1] and labels[2] == labels[3] and len(set(labels)) == 3
    score = np.array([0.1, 0.5, 0.9, 0.2, 0.3])
    reps = cluster_representatives(labels, score)
    assert reps.tolist() == [2, 1, 4]
