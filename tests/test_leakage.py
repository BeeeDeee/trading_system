import numpy as np
import pytest

from qlab.data.normalize import normalize_prices
from qlab.data.panel import panel_from_bars
from qlab.data.synthetic import generate_market
from qlab.features.basic import tr_index, trailing_return, trailing_volatility
from qlab.validation.leakage import assert_point_in_time, check_point_in_time


@pytest.fixture(scope="module")
def panel():
    m = generate_market(n_assets=25, seed=13, start="2010-01-01", end="2011-12-31")
    return panel_from_bars(normalize_prices(m.prices, m.delistings, m.calendar), m.calendar)


def momentum_top5(panel):
    """A toy strategy: monthly, equal weight in the 5 best 63-day performers."""
    score = trailing_return(panel, 63)
    targets = np.full(panel.shape, np.nan)
    for t in range(63, panel.shape[0], 21):
        s = np.where(panel.tradable[t] & np.isfinite(score[t]), score[t], -np.inf)
        best = np.argsort(-s)[:5]
        w = np.zeros(panel.shape[1])
        w[best[np.isfinite(s[best])]] = 0.2
        targets[t] = w
    return targets


HONEST = {
    "tr_index": tr_index,
    "momentum_126_21": lambda p: trailing_return(p, 126, 21),
    "volatility_63": lambda p: trailing_volatility(p, 63),
    "strategy_momentum_top5": momentum_top5,
}

LEAKY = {
    "next_day_return": lambda p: np.vstack([p.ret_oc[1:], np.zeros((1, p.shape[1]))]),
    "centered_average": lambda p: np.apply_along_axis(
        lambda x: np.convolve(x, np.ones(11) / 11, mode="same"), 0, p.ret_oc),
    "full_sample_zscore": lambda p: (p.ret_oc - p.ret_oc.mean(axis=0)) / p.ret_oc.std(axis=0),
    "normalized_by_final_value": lambda p: tr_index(p) / tr_index(p)[-1],
}


@pytest.mark.parametrize("name", HONEST)
def test_honest_functions_pass(panel, name):
    assert_point_in_time(HONEST[name], panel, name)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("name", LEAKY)
def test_leaky_functions_are_caught(panel, name):
    report = check_point_in_time(LEAKY[name], panel, name)
    assert not report.ok


def test_features_are_correct(panel):
    idx = tr_index(panel)
    mom = trailing_return(panel, 20, 5)
    t, a = 300, int(np.flatnonzero(panel.listed[:301].all(axis=0))[0])
    assert mom[t, a] == pytest.approx(idx[t - 5, a] / idx[t - 20, a] - 1)
    r = (1 + panel.ret_co[:, a]) * (1 + panel.ret_oc[:, a]) - 1
    vol = trailing_volatility(panel, 30)
    assert vol[t, a] == pytest.approx(np.std(r[t - 29: t + 1], ddof=1))
    assert np.isnan(mom[5]).all()


def test_rolling_helpers_match_naive():
    from qlab.features.basic import rolling_max, rolling_mean, rolling_min
    x = np.random.default_rng(0).normal(size=(300, 3))
    for w in (1, 2, 5, 55, 126):
        naive_max = np.array([x[max(0, t - w + 1): t + 1].max(axis=0) for t in range(300)])
        naive_min = np.array([x[max(0, t - w + 1): t + 1].min(axis=0) for t in range(300)])
        naive_mean = np.array([x[t - w + 1: t + 1].mean(axis=0) if t >= w - 1 else [np.nan] * 3
                               for t in range(300)])
        np.testing.assert_allclose(rolling_max(x, w)[w - 1:], naive_max[w - 1:])
        np.testing.assert_allclose(rolling_min(x, w)[w - 1:], naive_min[w - 1:])
        np.testing.assert_allclose(rolling_mean(x, w), naive_mean)
        assert np.isnan(rolling_max(x, w)[: w - 1]).all()
