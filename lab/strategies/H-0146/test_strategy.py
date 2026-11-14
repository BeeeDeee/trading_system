import numpy as np
import pytest

from strategy import target_weights, PARAMS
from strategy import lag_rows, raw_features, cohort_book, decision_mask, book_from_score
from strategy import LAG, SKIP, HOLD_ROWS, N_COHORTS


def test_params_match_card():
    assert PARAMS == {"num_leaves": 8, "top": 0.2}


def test_lag_rows_shifts_down_and_handles_short_views():
    x = np.arange(10, dtype=np.float32).reshape(5, 2)
    out = lag_rows(x, 2)
    assert np.all(np.isnan(out[:2]))
    assert np.array_equal(out[2:], x[:3])
    short = lag_rows(x, 252)          # view shorter than the lag
    assert short.shape == x.shape and np.all(np.isnan(short))


def _inputs(T=300, N=3):
    assets = np.full((T, N), 100.0, dtype=np.float32)
    capex = np.full((T, N), -10.0, dtype=np.float32)
    ncfo = np.full((T, N), 20.0, dtype=np.float32)
    debt = np.full((T, N), 30.0, dtype=np.float32)
    roa = np.full((T, N), 0.05, dtype=np.float32)
    tri = np.ones((T, N))
    listed = np.ones((T, N), dtype=bool)
    return assets, capex, ncfo, debt, roa, tri, listed


def test_raw_features_by_hand():
    T = 300
    assets, capex, ncfo, debt, roa, tri, listed = _inputs(T)
    t = 280
    assets[t, 0] = 150.0          # grew from 100 (value known 252 rows earlier)
    capex[t, 0] = -30.0           # |capex| / assets: 0.2 now vs 0.1 a year ago
    roa[t, 0] = 0.02              # roa fell by 0.03
    tri[t - SKIP, 0] = 1.3
    tri[t - LAG, 0] = 1.0
    feats, ok = raw_features(assets, capex, ncfo, debt, roa, tri, listed)
    ag, cx, cf, lv, rc, r121 = feats
    assert ag[t, 0] == pytest.approx(np.log(1.5), rel=1e-5)
    assert cx[t, 0] == pytest.approx(0.2 - 0.1, rel=1e-5)
    assert cf[t, 0] == pytest.approx(20.0 / 150.0, rel=1e-5)
    assert lv[t, 0] == pytest.approx(30.0 / 150.0, rel=1e-5)
    assert rc[t, 0] == pytest.approx(-0.03, abs=1e-6)
    assert r121[t, 0] == pytest.approx(0.3, rel=1e-6)
    # the other stocks: no change
    assert ag[t, 1] == pytest.approx(0.0, abs=1e-7) and r121[t, 1] == pytest.approx(0.0, abs=1e-7)
    # first 252 rows: no value 252 rows earlier -> NaN, not scored
    assert np.all(np.isnan(ag[:LAG])) and np.all(np.isnan(cx[:LAG])) and np.all(np.isnan(rc[:LAG]))
    assert np.all(np.isnan(r121[:LAG])) and not ok[:LAG].any()
    assert ok[LAG:].all()


def test_raw_features_nonpositive_or_missing_assets_and_late_listing():
    assets, capex, ncfo, debt, roa, tri, listed = _inputs(T=500)
    assets[290, 0] = 0.0                   # assets_t not > 0
    assets[290 - LAG, 1] = -5.0             # assets_{t-252} not > 0
    assets[290, 2] = np.nan                 # unknown -> stays NaN, never 0
    listed[:100, 2] = False                 # listed late
    feats, ok = raw_features(assets, capex, ncfo, debt, roa, tri, listed)
    assert not ok[290, 0] and not ok[290, 1] and not ok[290, 2]
    assert np.isnan(feats[0][290, 0]) and np.isnan(feats[0][290, 1]) and np.isnan(feats[1][290, 1])
    assert np.isnan(feats[2][290, 0]) and np.isnan(feats[3][290, 0])
    assert np.isnan(feats[2][290, 2])
    # ret_12_1 needs a listing at t-252
    assert np.isnan(feats[5][LAG + 50, 2]) and np.isfinite(feats[5][LAG + 150, 2])
    for f in feats:
        assert not np.isinf(f).any()


def test_raw_features_point_in_time():
    assets, capex, ncfo, debt, roa, tri, listed = _inputs(T=400)
    rng = np.random.default_rng(1)
    assets *= rng.uniform(0.5, 2.0, assets.shape).astype(np.float32)
    tri = np.cumprod(1 + rng.normal(0, 0.01, tri.shape), axis=0)
    full, _ = raw_features(assets, capex, ncfo, debt, roa, tri, listed)
    cut = 330
    part, _ = raw_features(assets[:cut], capex[:cut], ncfo[:cut], debt[:cut], roa[:cut], tri[:cut],
                           listed[:cut])
    for a, b in zip(full, part):
        assert np.array_equal(a[:cut], b, equal_nan=True)


def test_cohort_book_hold_share_and_gross():
    T, N = 400, 4
    decision = np.zeros(T, dtype=bool)
    decision[[10, 30]] = True
    w = np.zeros((T, N))
    w[10] = [0.5, 0.5, 0.0, 0.0]
    w[30] = [0.0, 0.5, 0.5, 0.0]
    alive = np.ones((T, N), dtype=bool)
    W = cohort_book(decision, w / N_COHORTS, alive)
    assert np.all(W[:10] == 0)                              # flat before the first cohort
    assert W[10, 0] == pytest.approx(0.5 / 6)               # warm-up: rest is cash
    assert W[30, 1] == pytest.approx(1.0 / 6)               # stock in two cohorts holds the sum
    assert W[10 + HOLD_ROWS - 1, 0] == pytest.approx(0.5 / 6)
    assert W[10 + HOLD_ROWS, 0] == 0                        # cohort expires after 126 rows
    assert W[10 + HOLD_ROWS, 1] == pytest.approx(0.5 / 6)
    assert W[30 + HOLD_ROWS, 2] == 0
    assert np.all(W.sum(axis=1) <= 1 + 1e-12) and np.all(W >= 0)


def test_cohort_book_member_exit_is_dropped_for_good():
    T, N = 300, 3
    decision = np.zeros(T, dtype=bool)
    decision[5] = True
    w = np.zeros((T, N))
    w[5] = [1 / 3, 1 / 3, 1 / 3]
    alive = np.ones((T, N), dtype=bool)
    alive[50, 0] = False          # leaves LIQ-500 for one row, then re-enters
    alive[5, 2] = False           # not a member at formation -> never held
    W = cohort_book(decision, w / N_COHORTS, alive)
    assert W[49, 0] == pytest.approx(1 / 18)
    assert np.all(W[50:, 0] == 0)                           # sold, not re-bought in this cohort
    assert W[60, 1] == pytest.approx(1 / 18)                # not redistributed
    assert np.all(W[:, 2] == 0)


def test_cohort_book_at_most_six_live_and_unformed_cohort_is_cash():
    T, N = 400, 2
    decision = np.zeros(T, dtype=bool)
    rows = [0, 20, 40, 60, 80, 100, 120, 140]  # seventh cohort arrives before the first's 126 rows end
    decision[rows] = True
    w = np.zeros((T, N))
    for r in rows:
        w[r] = [1.0, 0.0]
    w[60] = 0.0                                  # cohort not formed (fewer than 20 names): cash
    W = cohort_book(decision, w / N_COHORTS, np.ones((T, N), dtype=bool))
    assert W[119, 0] == pytest.approx(5 / 6)     # 6 decisions, one unformed
    assert W[120, 0] == pytest.approx(5 / 6)     # cohort 0 replaced by cohort 6 at row 120
    assert W[140, 0] == pytest.approx(5 / 6)
    assert np.all(W.sum(axis=1) <= 1 + 1e-12)


def _score_case(T=60, N=50):
    dates = np.arange(0, T, dtype="int64").astype("datetime64[D]") + np.timedelta64(10957, "D")
    uni = np.ones((T, N), dtype=bool)
    uni[:, 45:] = False                               # non-members are never held
    score = np.tile(np.arange(N, dtype=np.float64), (T, 1))
    score[:, 40:45] = np.nan                          # members without a score
    listed = np.ones((T, N), dtype=bool)
    return score, uni, listed, dates


def test_book_from_score_top_parameter_and_min_names():
    score, uni, listed, dates = _score_case()
    W2 = book_from_score(score, uni, listed, dates, 0.2)
    W1 = book_from_score(score, uni, listed, dates, 0.1)
    d = 0                                             # 2000-01-01 is the first row: a decision row
    held2 = np.flatnonzero(W2[d] > 0)
    held1 = np.flatnonzero(W1[d] > 0)
    assert len(held2) == 8 and len(held1) == 4        # 40 scored members
    assert set(held2) == set(range(32, 40)) and set(held1) == set(range(36, 40))
    assert W2[d].sum() == pytest.approx(1 / 6) and W1[d].sum() == pytest.approx(1 / 6)
    assert np.all(W2[:, 40:] == 0)
    few = score.copy()
    few[:, 19:40] = np.nan                            # 19 scored members: no cohort, cash
    assert np.all(book_from_score(few, uni, listed, dates, 0.2) == 0)


def test_decision_mask_first_trading_day_of_month():
    dates = np.arange(0, 70, dtype="int64").astype("datetime64[D]") + np.timedelta64(10957, "D")
    m = decision_mask(dates)                     # 10957 days after the epoch = 2000-01-01
    months = dates.astype("datetime64[M]")
    expected = np.r_[True, months[1:] != months[:-1]]
    assert np.array_equal(m, expected)
