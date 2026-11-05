import numpy as np
import pytest

from strategy import target_weights, PARAMS
from strategy import (weekly_decision_days, filing_events, sessions_since_filing, trailing_std_full,
                      winsorize_rows, compute_features, book_from_scores, last_true_index)
from lab.framework.data import DataView


def _dates(start_int, n_business):
    # build business dates from an integer day count (no date literals)
    base = np.zeros(1, dtype="datetime64[D]")[0] + np.timedelta64(start_int, "D")
    return np.busday_offset(base, np.arange(n_business), roll="forward")


def test_weekly_decisions_fridays_and_holiday_fallback():
    d = _dates(19000, 15)                      # three full weeks of business days
    wd = (d.astype(np.int64) + 3) % 7
    keep = np.ones(d.size, dtype=bool)
    second_friday = np.flatnonzero(wd == 4)[1]
    keep[second_friday] = False                # a holiday Friday
    d = d[keep]
    dec = weekly_decision_days(d)
    wd = (d.astype(np.int64) + 3) % 7
    fridays = np.flatnonzero(wd == 4)
    assert dec[fridays].all()
    # the week without Friday decides on the first session of the next week (instead of a Thursday)
    after = np.flatnonzero(wd[:-1] == 3)       # Thursdays
    hol_thu = [i for i in after if wd[i + 1] == 0]
    assert len(hol_thu) == 1
    assert dec[hol_thu[0] + 1]
    assert dec.sum() == len(fridays) + 1


def test_filing_events_and_count():
    T = 10
    a = np.full((T, 2), np.nan)
    a[3:, 0] = 1.0
    a[7:, 0] = 2.0
    a[:, 1] = 5.0                               # present from row 0, never changes
    same = np.where(np.isfinite(a), 1.0, np.nan)
    ev = filing_events(same, same, a)
    assert ev[:, 0].tolist() == [False, False, False, True, False, False, False, True, False, False]
    assert not ev[:, 1].any()
    cnt = sessions_since_filing(same, same, a)
    assert np.isnan(cnt[:3, 0]).all()
    assert cnt[3:, 0].tolist() == [0, 1, 2, 3, 0, 1, 2]
    assert np.isnan(cnt[:, 1]).all()            # unknown age before EVENT_CAP rows


def test_count_capped_and_known_after_cap():
    T = 300
    a = np.full((T, 1), 1.0)
    a[10:] = 2.0
    cnt = sessions_since_filing(a, a, a)
    assert cnt[10, 0] == 0 and cnt[50, 0] == 40
    assert cnt[200, 0] == 126
    b = np.full((T, 1), 1.0)
    cb = sessions_since_filing(b, b, b)
    assert np.isnan(cb[125, 0]) and cb[126, 0] == 126


def test_trailing_std():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, (100, 2))
    valid = np.ones((100, 2), dtype=bool)
    valid[50, 1] = False
    sd = trailing_std_full(np.where(valid, r, 0.0), valid, 63)
    assert np.isnan(sd[61]).all()
    assert sd[62, 0] == pytest.approx(np.std(r[:63, 0], ddof=1))
    assert sd[99, 0] == pytest.approx(np.std(r[37:100, 0], ddof=1))
    assert np.isnan(sd[99, 1])                 # a missing return inside the window
    short = trailing_std_full(r[:10], valid[:10], 63)
    assert np.isnan(short).all()


def test_winsorize_rows():
    x = np.arange(1.0, 102.0)[None, :]
    scored = np.ones_like(x, dtype=bool)
    scored[0, -1] = False                       # 101 not scored
    w = winsorize_rows(x, scored)
    assert np.isnan(w[0, -1])
    assert w[0, 0] == pytest.approx(np.percentile(np.arange(1.0, 101.0), 1))
    assert w[0, 99] == pytest.approx(np.percentile(np.arange(1.0, 101.0), 99))
    assert w[0, 50] == 51.0


def test_last_true_index():
    m = np.array([[False], [True], [False], [True], [False]])
    assert last_true_index(m)[:, 0].tolist() == [-1, 1, 1, 3, 3]


def _view(T, N, rng, flag_rows=None, assets=1.0):
    dates = _dates(16000, T)
    ret_co = rng.normal(0, 0.005, (T, N))
    ret_oc = rng.normal(0, 0.01, (T, N))
    close = 100 * np.cumprod((1 + ret_co) * (1 + ret_oc), axis=0)
    flag = np.zeros((T, N))
    if flag_rows is not None:
        for t, j in flag_rows:
            flag[t, j] = 1.0
    netinc = np.full((T, N), 2.0)
    ncfo = np.full((T, N), 1.0)
    ast = np.full((T, N), assets)
    netinc[100:] = 3.0                          # a filing on row 100 for everyone
    names = tuple(f"E{i}" for i in range(N))
    ones = np.ones((T, N), dtype=bool)
    return DataView(dates=dates, instruments=names, asset_class=tuple("us_equity" for _ in names),
                    ret_co=ret_co, ret_oc=ret_oc, tradable=ones, listed=ones,
                    delisting=np.zeros((T, N), dtype=bool), close=close,
                    dollar_volume=np.full((T, N), 1e8), universe=ones.copy(),
                    extras={"sf1_arq_netinc": netinc.astype(np.float32), "sf1_arq_ncfo": ncfo.astype(np.float32),
                            "sf1_arq_assets": ast.astype(np.float32), "ea_reaction_day": flag.astype(np.float32)},
                    series={}, cash_ret=np.zeros(T))


def test_features_by_hand():
    rng = np.random.default_rng(1)
    T, N = 260, 5
    data = _view(T, N, rng, flag_rows=[(80, 0), (120, 0)])
    feats, scored, count = compute_features(data)
    f1, f2, f3, f4, f5, f6 = feats
    ret = (1 + data.ret_co) * (1 + data.ret_oc) - 1
    ret[0] = 0.0
    # F1, F2 (accruals and cash profitability) after the filing on row 100
    assert f1[150, 0] == pytest.approx((3.0 - 1.0) / 1.0)
    assert f2[150, 0] == pytest.approx(1.0)
    # F3 at t = 130: r = 120, reaction vs member median over the 63-session std ending at 119
    sd119 = np.std(ret[57:120, 0], ddof=1)
    expect3 = (ret[120, 0] - np.median(ret[120])) / sd119
    assert f3[130, 0] == pytest.approx(expect3, rel=1e-6)
    # F4 at t = 130: stock compounded from close of 120 vs equal-weight member index, scaled by sqrt(10)
    stock = np.prod(1 + ret[121:131, 0]) - 1
    mkt = np.prod(1 + ret[121:131].mean(axis=1)) - 1
    assert f4[130, 0] == pytest.approx((stock - mkt) / (sd119 * np.sqrt(10)), rel=1e-5)
    assert f4[120, 0] == 0.0
    # F3 before r = 80 is NaN; stocks without any release are not scored
    assert np.isnan(f3[79, 0]) and not scored[130, 1]
    # 126-session window: r = 120 still counts at 245, not at 246
    assert np.isfinite(f3[245, 0]) and np.isnan(f3[246, 0])
    # F5 sessions since filing
    assert count[100, 0] == 0 and count[130, 0] == 30 and f5[130, 0] == pytest.approx(30 / 126)
    # F6: 63-session std ending at t
    assert f6[130, 0] == pytest.approx(np.std(ret[68:131, 0], ddof=1))
    assert scored[130, 0]


def test_assets_nonpositive_not_scored():
    rng = np.random.default_rng(2)
    data = _view(200, 3, rng, flag_rows=[(150, 0)], assets=-1.0)
    _, scored, _ = compute_features(data)
    assert not scored.any()


def test_not_member_not_scored():
    rng = np.random.default_rng(3)
    data = _view(200, 3, rng, flag_rows=[(150, 0)])
    data.universe[160, 0] = False
    _, scored, _ = compute_features(data)
    assert scored[159, 0] and not scored[160, 0] and scored[161, 0]


def _scores(T, N, rng, n_scored):
    s = rng.normal(size=(T, N))
    s[:, n_scored:] = np.nan
    return s


def test_book_top_fraction_keep_and_exit():
    rng = np.random.default_rng(4)
    T, N = 30, 120
    dates = _dates(19000, T)
    dec = weekly_decision_days(dates)
    uni = np.ones((T, N), dtype=bool)
    score = _scores(T, N, rng, 100)
    d_rows = np.flatnonzero(dec)
    score[d_rows[1]] = np.nan
    score[d_rows[1], :40] = 1.0                 # only 40 scored: keep the previous book
    out_row = d_rows[0] + 2
    w0 = book_from_scores(score, uni, dates, 0.2)
    first_book = w0[d_rows[0]]
    held = np.flatnonzero(first_book > 0)
    uni[out_row, held[0]] = False               # leaves the universe for one day
    w = book_from_scores(score, uni, dates, 0.2)
    # decision rows: 20 names equal weight, gross 1
    assert np.isclose(np.nansum(w[d_rows[0]]), 1.0)
    assert (w[d_rows[0]] > 0).sum() == 20
    top20 = np.argsort(-np.nan_to_num(score[d_rows[0]], nan=-np.inf))[:20]
    assert set(np.flatnonzero(w[d_rows[0]] > 0)) == set(top20)
    # rows before the first decision: no weights
    assert np.isnan(w[: d_rows[0]]).all()
    # exit from the universe: sold on that row and stays sold until the next decision
    assert w[out_row, held[0]] == 0.0
    assert np.isclose(np.nansum(w[out_row]), 1.0 - 1.0 / 20)
    nxt = d_rows[1]
    assert np.isnan(w[out_row + 1: nxt]).all()
    # fewer than 50 scored on the second decision: previous book (minus the exit) kept
    assert np.array_equal(w[nxt], w[out_row])   # the kept book
    # third decision: new book
    assert (w[d_rows[2]] > 0).sum() == 20
    gross = np.nansum(np.abs(w), axis=1)
    assert (gross <= 1.0 + 1e-9).all()
    assert (np.nan_to_num(w) >= 0).all()


def test_top_param_changes_book():
    rng = np.random.default_rng(5)
    T, N = 15, 100
    dates = _dates(19000, T)
    uni = np.ones((T, N), dtype=bool)
    score = _scores(T, N, rng, 100)
    d = np.flatnonzero(weekly_decision_days(dates))[0]
    a = book_from_scores(score, uni, dates, 0.1)
    b = book_from_scores(score, uni, dates, 0.3)
    assert (a[d] > 0).sum() == 10 and (b[d] > 0).sum() == 30


def test_flat_before_any_score():
    T, N = 15, 60
    dates = _dates(19000, T)
    uni = np.ones((T, N), dtype=bool)
    w = book_from_scores(np.full((T, N), np.nan), uni, dates, 0.2)
    assert np.all(np.isnan(w) | (w == 0))


def test_params_defaults():
    assert PARAMS == {"num_leaves": 15, "top": 0.2}
