"""Book tests: score_weights legs + daily cohorts, leg_fraction and the 50-name floor, on hand-made scores."""
import numpy as np
import pytest

from strategy import MIN_NAMES, cohort_book, target_weights
from lab.framework.ml import score_weights


def _scores(T, N, n_scored):
    s = np.full((T, N), np.nan)
    s[:, :n_scored] = np.arange(n_scored, dtype=float)[None, :]      # stock j has score j
    return s


@pytest.mark.parametrize("lf", [0.05, 0.1, 0.2])
def test_legs_and_cohort_weights(lf):
    T, N = 8, 120
    uni = np.ones((T, N), dtype=bool)
    s = _scores(T, N, 100)
    c = np.nan_to_num(np.asarray(score_weights(s, uni, top=lf, bottom=lf, min_names=MIN_NAMES), dtype=float))
    k = int(round(lf * 100))
    assert (c[0] > 0).sum() == k and (c[0] < 0).sum() == k
    assert (c[0, 100 - k:100] > 0).all() and (c[0, :k] < 0).all()            # top scores long, bottom short
    b = cohort_book(c, uni)
    assert np.allclose(b[4][b[4] > 0].sum(), 0.5) and np.allclose(b[4][b[4] < 0].sum(), -0.5)
    assert np.allclose(b[0][b[0] > 0], 0.1 / k)                                # one cohort: 0.1 per leg


def test_fewer_than_50_scored_no_cohort():
    T, N = 3, 80
    uni = np.ones((T, N), dtype=bool)
    s = _scores(T, N, 49)
    c = np.nan_to_num(np.asarray(score_weights(s, uni, top=0.1, bottom=0.1, min_names=MIN_NAMES), dtype=float))
    assert np.abs(c).sum() == 0
    assert MIN_NAMES == 50


def test_missing_universe_fails_loudly():
    from lab.framework.data import synthetic
    data = synthetic()
    if data.universe is not None:
        pytest.skip("synthetic market has a universe")
    with pytest.raises(Exception):
        target_weights(data, {"leg_fraction": 0.1, "n_estimators": 100})
