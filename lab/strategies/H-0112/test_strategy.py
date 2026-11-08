import dataclasses

import numpy as np
import pytest

from strategy import PARAMS, target_weights
import strategy as S


@dataclasses.dataclass
class FakeData:
    ret_co: np.ndarray
    ret_oc: np.ndarray
    listed: np.ndarray
    dollar_volume: np.ndarray
    extras: dict


def test_params_match_card():
    assert PARAMS == {"flow_sigma": 1.5, "n_estimators": 100, "top": 0.2}


def test_window_sum_and_prior():
    a = np.arange(1.0, 7.0)
    np.testing.assert_allclose(S._window_sum(a, 3), [1, 3, 6, 9, 12, 15])
    np.testing.assert_allclose(S._window_sum(a, 10), np.cumsum(a))   # view shorter than window
    np.testing.assert_allclose(S._prior(a, 0.0), [0, 1, 2, 3, 4, 5])


def test_daily_returns_compound_and_unlisted():
    data = FakeData(ret_co=np.array([[0.1, 0.0]]), ret_oc=np.array([[0.1, 0.05]]),
                    listed=np.array([[True, False]]), dollar_volume=np.ones((1, 2)), extras={})
    r = S.daily_returns(data)
    assert r[0, 0] == pytest.approx(0.21)
    assert np.isnan(r[0, 1])


def _market_data(T=80, seed=0):
    rng = np.random.default_rng(seed)
    N = 3
    ret_oc = rng.normal(0, 0.01, (T, N))
    mc = np.tile(np.array([1.0, 3.0, np.nan]), (T, 1))
    dv = np.full((T, N), 10.0)
    return FakeData(ret_co=np.zeros((T, N)), ret_oc=ret_oc, listed=np.ones((T, N), bool),
                    dollar_volume=dv, extras={"sf1_arq_marketcap": mc})


def test_market_series_by_hand():
    data = _market_data()
    T = data.ret_oc.shape[0]
    uni = np.ones((T, 3), bool)
    uni[:, 1] = True
    r = S.daily_returns(data)
    m, z, vshock = S.market_series(data, r, uni)
    # cap-weighted over members with a marketcap (column 2 has none)
    np.testing.assert_allclose(m, (1.0 * r[:, 0] + 3.0 * r[:, 1]) / 4.0)
    # z needs 63 prior sessions
    assert np.all(np.isnan(z[:63]))
    t = 70
    sd = np.std(m[t - 63:t], ddof=1)
    assert z[t] == pytest.approx(m[t] / sd)
    # constant member volume: shock 0 once the window is full
    assert np.all(np.isnan(vshock[:63]))
    np.testing.assert_allclose(vshock[63:], 0.0, atol=1e-12)
    # a non-member is excluded from m and from the volume sum
    uni2 = uni.copy()
    uni2[t, 1] = False
    m2, _, v2 = S.market_series(data, r, uni2)
    assert m2[t] == pytest.approx(r[t, 0])
    assert v2[t] == pytest.approx(np.log(20.0 / 30.0))


def test_market_series_point_in_time():
    data = _market_data(T=90)
    uni = np.ones((90, 3), bool)
    r = S.daily_returns(data)
    _, z, v = S.market_series(data, r, uni)
    cut = 75
    short = FakeData(data.ret_co[:cut], data.ret_oc[:cut], data.listed[:cut], data.dollar_volume[:cut],
                     {"sf1_arq_marketcap": data.extras["sf1_arq_marketcap"][:cut]})
    _, zs, vs = S.market_series(short, S.daily_returns(short), uni[:cut])
    np.testing.assert_array_equal(z[:cut], zs)
    np.testing.assert_array_equal(v[:cut], vs)


def test_inverse_turnover():
    T = 110
    dv = np.full((T, 2), 4.0)
    dv[:40, 1] = np.nan                      # late start: fewer than 32 finite values early
    dv[100:, 0] = 1.0
    mc = np.full((T, 2), 400.0)
    mc[65:70, 0] = np.nan                    # unknown marketcap stays NaN
    data = FakeData(np.zeros((T, 2)), np.zeros((T, 2)), np.ones((T, 2), bool), dv, {"sf1_arq_marketcap": mc})
    it = S.inverse_turnover(data)
    assert np.all(np.isnan(it[:31, 0]))
    assert it[31, 0] == pytest.approx(np.log(100.0))
    assert np.isnan(it[66, 0])
    # mean over t-62..t: at row 100 one day of 1.0 among 63
    assert it[100, 0] == pytest.approx(np.log(400.0 / ((62 * 4.0 + 1.0) / 63)))
    first = 40 + S.DV_MIN_OBS - 1
    assert np.isnan(it[first - 1, 1]) and it[first, 1] == pytest.approx(np.log(100.0))


def test_rolling_beta_exact_and_min_pairs():
    rng = np.random.default_rng(1)
    T = 300
    m = rng.normal(0, 0.01, T)
    r = np.column_stack([2.0 * m + 0.001, -0.5 * m, m])
    r[:200, 2] = np.nan                      # only 100 pairs by row 299: below 126
    b = S.rolling_beta(r, m)
    assert np.all(np.isnan(b[:125, 0]))
    assert b[125, 0] == pytest.approx(2.0)
    assert b[299, 1] == pytest.approx(-0.5)
    assert np.all(np.isnan(b[:, 2]))
    np.testing.assert_array_equal(S.rolling_beta(r, m, chunk=1), b)   # column chunks change nothing


def test_prior_return_excludes_today():
    r = np.array([[0.1], [0.1], [0.1], [0.1], [0.1], [0.5], [0.0]])
    p = S.prior_return(r)
    assert np.all(np.isnan(p[:5, 0]))
    assert p[5, 0] == pytest.approx(1.1 ** 5 - 1)          # t-5..t-1, not row 5 itself
    assert p[6, 0] == pytest.approx(1.1 ** 4 * 1.5 - 1)


def test_train_rows_and_flow_days():
    z = np.array([np.nan, 0.5, -1.0, 1.2, -1.6, 2.0])
    np.testing.assert_array_equal(S.train_rows(z), [False, False, True, True, True, True])
    np.testing.assert_array_equal(S.flow_days(z, 1.5), [False, False, False, False, True, True])
    np.testing.assert_array_equal(S.flow_days(z, 2.0), [False] * 5 + [True])


def test_gbm_params_reads_n_estimators():
    p = S.gbm_params({"n_estimators": 200})
    assert p == {"n_estimators": 200, "num_leaves": 15, "learning_rate": 0.05, "min_child_samples": 200}


def _book_inputs(T=14, N=120):
    score = np.tile(np.arange(N, dtype=np.float64), (T, 1))
    uni = np.ones((T, N), bool)
    z = np.zeros(T)
    return score, uni, z


def test_single_cohort_held_five_sessions():
    score, uni, z = _book_inputs()
    z[2] = -1.7
    w = S.build_book(score, uni, z, 1.5, 0.2)
    assert np.all(w[:2] == 0) and np.all(w[7:] == 0)
    for t in range(2, 7):
        assert w[t][w[t] > 0].sum() == pytest.approx(0.25)
        assert w[t][w[t] < 0].sum() == pytest.approx(-0.25)
        assert np.all(w[t, :60] <= 0) and np.all(w[t, 60:] >= 0)   # long the high scores
        np.testing.assert_allclose(w[t], w[2])


def test_top_changes_leg_size():
    score, uni, z = _book_inputs()
    z[2] = 2.0
    n1 = (S.build_book(score, uni, z, 1.5, 0.1)[2] > 0).sum()
    n3 = (S.build_book(score, uni, z, 1.5, 0.3)[2] > 0).sum()
    assert n1 < n3


def test_flow_sigma_changes_decision_days():
    score, uni, z = _book_inputs()
    z[2] = 1.2
    assert np.all(S.build_book(score, uni, z, 1.5, 0.2) == 0)
    assert np.abs(S.build_book(score, uni, z, 1.0, 0.2)[2]).sum() == pytest.approx(0.5)


def test_overlap_scaled_to_gross_one():
    score, uni, z = _book_inputs()
    z[2:7] = 3.0                              # five consecutive flow days
    w = S.build_book(score, uni, z, 1.5, 0.2)
    g = np.abs(w).sum(axis=1)
    np.testing.assert_allclose(g[2:7], [0.5, 1.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(g[7:11], [1.0, 1.0, 1.0, 0.5])
    assert g[11] == 0
    assert np.all(np.abs(w.sum(axis=1)) < 1e-12)                 # dollar-neutral


def test_member_exit_closes_position_and_few_names_no_cohort():
    score, uni, z = _book_inputs()
    z[2] = 2.0
    uni[4:, 119] = False
    w = S.build_book(score, uni, z, 1.5, 0.2)
    assert w[3, 119] > 0 and w[4, 119] == 0
    score2, uni2, z2 = _book_inputs(N=99)
    z2[2] = 2.0
    assert np.all(S.build_book(score2, uni2, z2, 1.5, 0.2) == 0)


def test_unscored_members_not_held():
    score, uni, z = _book_inputs(N=130)
    score[:, 125:] = np.nan
    z[2] = 2.0
    w = S.build_book(score, uni, z, 1.5, 0.2)
    assert np.all(w[:, 125:] == 0)
