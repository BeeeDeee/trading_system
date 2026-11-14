import numpy as np
import pytest

from strategy import target_weights, PARAMS
from strategy import (decision_rows, book_to_market, realized_vol, winsorize_rows, buffered_book,
                      rank_sets, _ratio, _lag_rows, fresh_entries)

DAY_2020_01_01 = 18262   # days since 1970-01-01


def weekdays(first_day, n_days):
    d = (np.arange(n_days) + first_day).astype("datetime64[D]")
    return d[np.is_busday(d)]


def test_params():
    assert PARAMS == {"n_estimators": 100, "top": 0.2}
    assert callable(target_weights)


def test_decision_rows_last_weekday_of_month():
    d = weekdays(DAY_2020_01_01, 63)          # 2020-01-01 .. 2020-03-03
    flags = decision_rows(d)
    days = (d[flags].astype(np.int64) - DAY_2020_01_01).tolist()
    assert days == [30, 58]                    # 2020-01-31 (Fri), 2020-02-28 (Fri; 29th is Saturday)
    # point-in-time: flags on a prefix are the same
    for k in (5, 21, 22, 40):
        assert np.array_equal(decision_rows(d[:k]), flags[:k])


def test_decision_rows_month_end_holiday_falls_to_next_row():
    d = weekdays(DAY_2020_01_01, 63)
    d = d[d.astype(np.int64) != DAY_2020_01_01 + 30]   # drop 2020-01-31 as if a holiday
    flags = decision_rows(d)
    days = (d[flags].astype(np.int64) - DAY_2020_01_01).tolist()
    assert days == [33, 58]                    # 2020-02-03 (first row of Feb), 2020-02-28


def test_book_to_market_moves_with_price_since_usable():
    pb = np.array([[2.0], [2.0], [2.0], [4.0], [4.0], [np.nan], [-1.0], [5.0]])
    idx = np.array([[1.0], [1.1], [1.21], [1.21], [2.42], [2.42], [2.42], [4.84]])
    bm = book_to_market(pb, idx)[:, 0]
    assert bm[0] == pytest.approx(0.5)
    assert bm[1] == pytest.approx(0.5 / 1.1)
    assert bm[2] == pytest.approx(0.5 / 1.21)
    assert bm[3] == pytest.approx(0.25)        # new pb usable at row 3: anchored at the row-2 close
    assert bm[4] == pytest.approx(0.125)
    assert np.isnan(bm[5]) and np.isnan(bm[6])  # missing, pb <= 0
    assert bm[7] == pytest.approx(0.2 / 2.0)    # re-anchored at row 6 close


def test_ratio_and_lag():
    r = _ratio(np.array([[1.0, 1.0, 1.0, np.nan]]), np.array([[2.0, 0.0, -1.0, 2.0]]))
    assert r[0, 0] == 0.5 and np.isnan(r[0, 1:]).all()
    x = np.arange(5.0)[:, None]
    lag = _lag_rows(x, 2)
    assert np.isnan(lag[:2]).all() and np.array_equal(lag[2:, 0], [0.0, 1.0, 2.0])


def test_realized_vol_window_and_min_valid():
    r = np.array([[0.01], [0.03], [-0.02], [0.05], [0.0]])
    valid = np.array([[True], [True], [True], [True], [False]])
    v = realized_vol(r, valid, window=3, min_valid=3)[:, 0]
    assert np.isnan(v[:2]).all()
    assert v[2] == pytest.approx(np.std([0.01, 0.03, -0.02], ddof=1))
    assert v[3] == pytest.approx(np.std([0.03, -0.02, 0.05], ddof=1))
    assert np.isnan(v[4])                      # only 2 valid rows in the window
    v2 = realized_vol(r, valid, window=3, min_valid=2)[:, 0]
    assert v2[4] == pytest.approx(np.std([-0.02, 0.05], ddof=1))


def test_winsorize_rows():
    x = np.full((2, 103), np.nan)
    x[0, :101] = np.arange(101.0)
    x[0, 0] = -1000.0
    x[0, 100] = 1000.0
    w = winsorize_rows(x)
    lo = np.percentile(x[0, :101], 1)
    hi = np.percentile(x[0, :101], 99)
    assert w[0, 0] == pytest.approx(lo) and w[0, 100] == pytest.approx(hi)
    assert np.array_equal(w[0, 2:99], x[0, 2:99])
    assert np.isnan(w[0, 101:]).all() and np.isnan(w[1]).all()


def _book_case():
    T, N = 6, 25
    score = np.full((T, N), np.nan)
    member = np.ones((T, N), dtype=bool)
    dec = np.array([True, False, True, False, True, False])
    score[0] = np.arange(N)                    # top 5 = cols 20..24
    member[1:, 24] = False                     # col 24 leaves the universe at row 1
    s2 = np.arange(N, dtype=float)
    s2[[15, 16, 17, 18, 19]] = 100 + np.arange(5)   # new top 5; cols 20..23 rank 6..9 -> kept (buffer)
    score[2] = s2
    s4 = s2.copy()
    s4[20] = -5.0                              # col 20 falls out of the top 10 -> sold
    s4[21] = np.nan                            # col 21 unscored -> sold
    score[4] = s4
    return score, member, dec


def test_buffered_book():
    score, member, dec = _book_case()
    w = buffered_book(score, member, dec, 0.2)
    assert np.allclose(w[0, 20:], 0.2) and np.all(w[0, :20] == 0)
    # row 1: only the universe exit is a decision
    assert w[1, 24] == 0 and np.isnan(np.delete(w[1], 24)).all()
    # row 2: 24 scored members (col 24 left) -> n_top floor(4.8) = 4, n_keep floor(9.6) = 9
    # top 4 = 19, 18, 17, 16 (bought); 15 is rank 5 -> not bought; held 20..23 rank 6..9 -> kept
    held2 = np.nonzero(w[2] > 0)[0].tolist()
    assert held2 == [16, 17, 18, 19, 20, 21, 22, 23]
    assert np.allclose(w[2, held2], 1 / 8)
    assert np.isnan(w[3]).all()
    # row 4: 23 scored -> n_top 4, n_keep 9 = 19..15, 23, 22, 14, 13; 20 (low) and 21 (unscored) sold
    held4 = np.nonzero(w[4] > 0)[0].tolist()
    assert held4 == [16, 17, 18, 19, 22, 23]
    for t in (0, 2, 4):
        assert w[t].sum() == pytest.approx(1.0) and (w[t] >= 0).all()


def test_buffer_vs_no_hold_and_top_param():
    score, member, dec = _book_case()
    w15 = buffered_book(score, member, dec, 0.15)
    w20 = buffered_book(score, member, dec, 0.2)
    assert not np.array_equal(np.nan_to_num(w15, nan=-1), np.nan_to_num(w20, nan=-1))
    # with 24 scored at row 2 and top 0.15: n_top 3, n_keep 7
    assert np.count_nonzero(w15[0] > 0) == 3


def test_flat_with_fewer_than_20_scored():
    T, N = 2, 25
    score = np.full((T, N), np.nan)
    score[0, :19] = np.arange(19)
    member = np.ones((T, N), dtype=bool)
    w = buffered_book(score, member, np.array([True, True]), 0.2)
    assert np.all(w == 0)
    it, ik, n = rank_sets(score[0], np.isfinite(score[0]), 0.2)
    assert n == 19 and not it.any()


def test_non_members_never_held():
    score, member, dec = _book_case()
    member[:, 3] = False
    score[:, 3] = 1e9
    w = buffered_book(score, member, dec, 0.2)
    assert np.all(np.nan_to_num(w[:, 3]) == 0)


def test_fresh_entries():
    score, member, dec = _book_case()
    tradable = np.ones_like(member)
    m = fresh_entries(score, member, tradable, dec, 0.2)
    assert m[0, 20:].all() and m[0].sum() == 5          # first scored decision: nothing was in the top 10
    assert m[2].sum() == 0                              # 16..19 were in the top 10 at row 0: not fresh
    assert not m[1].any() and not m[3].any()            # only decision rows
    score[4, 0] = 1000.0                                # col 0 jumps from the bottom into the top
    m = fresh_entries(score, member, tradable, dec, 0.2)
    assert m[4].sum() == 1 and m[4, 0]
    tradable[4, 0] = False
    m = fresh_entries(score, member, tradable, dec, 0.2)
    assert not m[4].any()
