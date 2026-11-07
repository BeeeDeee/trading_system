import dataclasses

import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS
from strategy import (
    _announcement_risk, _book, _reaction, _recent_reaction, _roll_median_members, _roll_std_full,
    _roll_sum_full, _sets, _window_candidates, _entrants_and_scores,
)


def _bdays(start, n):
    """n weekdays from integer day `start` (integer day 0 is a Thursday)."""
    d = np.arange(start, start + 2 * n)
    d = d[((d + 3) % 7) < 5][:n]
    return d


def _view(T=1300, N=200, seed=0, period=65, late_col=3, late_row=400):
    rng = np.random.default_rng(seed)
    days = _bdays(12800, T)
    dates = days.astype("datetime64[D]")
    ret_co = rng.normal(0, 0.005, (T, N))
    ret_oc = rng.normal(0, 0.012, (T, N))
    listed = np.ones((T, N), dtype=bool)
    listed[:late_row, late_col] = False
    ret_co[~listed] = 0.0
    ret_oc[~listed] = 0.0
    close = 50 * np.cumprod((1 + ret_co) * (1 + ret_oc), axis=0)
    close[~listed] = np.nan
    dv = np.exp(rng.normal(18, 0.5, (T, N))) * np.exp(np.linspace(0, 1, N))[None, :]
    dv[~listed] = np.nan
    ea = np.zeros((T, N))
    off = rng.integers(0, period, N)
    for j in range(N):
        ea[off[j]::period, j] = 1.0
    ea[:100, 5] = np.nan
    ea[~listed] = np.nan
    uni = listed.copy()
    uni[700:760, 7] = False
    return DataView(
        dates=dates, instruments=tuple("E%d" % (j + 1) for j in range(N)), asset_class=("us_equity",) * N,
        ret_co=ret_co, ret_oc=ret_oc, tradable=listed.copy(), listed=listed, delisting=np.zeros((T, N), dtype=bool),
        close=close, dollar_volume=dv, universe=uni, extras={"ea_reaction_day": ea}, series={}, cash_ret=None,
    )


def _cut(v, n):
    return dataclasses.replace(
        v, dates=v.dates[:n], ret_co=v.ret_co[:n], ret_oc=v.ret_oc[:n], tradable=v.tradable[:n],
        listed=v.listed[:n], delisting=v.delisting[:n], close=v.close[:n], dollar_volume=v.dollar_volume[:n],
        universe=v.universe[:n], extras={k: x[:n] for k, x in v.extras.items()},
    )


# ------------------------------------------------------------------ rolling helpers
def test_roll_sum_full_needs_full_window():
    x = np.array([[1.0], [2.0], [np.nan], [4.0], [5.0], [6.0]])
    s = _roll_sum_full(x, 2)[:, 0]
    assert np.isnan(s[0]) and s[1] == 3.0 and np.isnan(s[2]) and np.isnan(s[3]) and s[4] == 9.0 and s[5] == 11.0
    assert np.all(np.isnan(_roll_sum_full(x[:1], 2)))       # view shorter than the window


def test_roll_std_and_median():
    x = np.array([[1.0], [2.0], [3.0], [10.0]])
    sd = _roll_std_full(x, 3)[:, 0]
    assert np.isnan(sd[1]) and sd[2] == pytest.approx(1.0) and sd[3] == pytest.approx(np.std([2, 3, 10], ddof=1))
    m = _roll_median_members(x, 3, np.array([[True], [True], [True], [False]]))[:, 0]
    assert np.isnan(m[1]) and m[2] == 2.0 and np.isnan(m[3])   # non-member row not computed


# ------------------------------------------------------------------ entrants
def test_reaction_nan_is_no_release():
    assert _reaction(np.array([[1.0, np.nan, 0.0, 2.0]])).tolist() == [[True, False, False, False]]


def test_window_candidates_first_session_before_anniversary():
    days = _bdays(12800, 400)
    T = len(days)
    react = np.zeros((T, 3), dtype=bool)
    # col 0: a Wednesday reaction; A - 5 is the Friday before A
    s0 = int(np.nonzero(((days + 3) % 7) == 2)[0][0])
    react[s0, 0] = True
    # col 1: a Thursday reaction; A - 5 is a Saturday -> entrant on the Monday (A - 3)
    s1 = int(np.nonzero(((days + 3) % 7) == 3)[0][0])
    react[s1, 1] = True
    cand = _window_candidates(days, react)
    r0 = np.nonzero(cand[:, 0])[0]
    assert len(r0) == 1 and days[r0[0]] == days[s0] + 359
    r1 = np.nonzero(cand[:, 1])[0]
    assert len(r1) == 1 and days[r1[0]] == days[s1] + 361
    assert not cand[:, 2].any()
    # a gap in the calendar from A-5 to A: first session at/after A-5 is not before A -> no entry
    a = days[s0] + 364
    keep = (days < a - 5) | (days >= a)
    d2 = days[keep]
    r2 = react[keep]
    assert not _window_candidates(d2, r2)[:, 0].any()


def test_recent_reaction_40_sessions():
    react = np.zeros((100, 1), dtype=bool)
    react[10, 0] = True
    rr = _recent_reaction(react, 40)[:, 0]
    assert not rr[9] and rr[10] and rr[49] and not rr[50]


def test_announcement_risk_last_four_min_two():
    T = 20
    react = np.zeros((T, 2), dtype=bool)
    ex = np.zeros((T, 2))
    rows = [2, 5, 9, 12, 15]
    vals = [0.01, 0.03, 0.05, 0.07, 0.09]
    for r, v in zip(rows, vals):
        react[r, 0] = True
        ex[r, 0] = v
    react[3, 1] = True
    ex[3, 1] = 0.5
    ar = _announcement_risk(react, ex)
    assert np.all(np.isnan(ar[:5, 0]))                       # one reaction only: NaN
    assert ar[5, 0] == pytest.approx(0.02) and ar[8, 0] == pytest.approx(0.02)
    assert ar[9, 0] == pytest.approx(0.03)
    assert ar[12, 0] == pytest.approx(0.04)
    assert ar[15, 0] == pytest.approx(0.06) and ar[19, 0] == pytest.approx(0.06)   # last 4 only
    assert np.all(np.isnan(ar[:, 1]))


# ------------------------------------------------------------------ sets
def test_sets_trailing_quantiles_and_min_pool():
    T, N = 70, 60
    score = np.full((T, N), np.nan)
    score[0, :30] = np.arange(30)                          # pool 30 < 50: no sets on row 0
    score[2, 30:] = np.arange(30, 60)                      # pool 60 on row 2 (rows 0..2)
    side = _sets(score, 0.2)
    assert not side[0].any()
    lo, hi = np.quantile(np.arange(60), 0.2), np.quantile(np.arange(60), 0.8)
    assert lo < 30                                         # no LOW on row 2 (all its scores >= 30)
    assert (side[2] == -1).sum() == 0
    assert (side[2] == 1).sum() == int((np.arange(30, 60) >= hi).sum())
    # 63 sessions later the row-0 scores leave the pool: row 65 (pool rows 3..65) has 30+? only its own
    score[65, :20] = np.arange(20)
    side = _sets(score, 0.2)
    assert not side[65].any()                              # pool = rows 3..65 -> 20 < 50
    side3 = _sets(score, 0.3)
    assert (side3[2] == 1).sum() > (side[2] == 1).sum()    # tail changes the sets


def test_sets_low_side():
    score = np.full((1, 100), np.nan)
    score[0] = np.arange(100.0)
    side = _sets(score, 0.1)
    assert (side[0] == -1).sum() == int((np.arange(100) <= np.quantile(np.arange(100), 0.1)).sum())
    assert (side[0] == 1).sum() == int((np.arange(100) >= np.quantile(np.arange(100), 0.9)).sum())
    assert side[0, 0] == -1 and side[0, 99] == 1 and side[0, 50] == 0


# ------------------------------------------------------------------ book
def test_book_hold_five_and_single_slot():
    T, N = 12, 3
    side = np.zeros((T, N))
    uni = np.ones((T, N), dtype=bool)
    side[1, 0] = 1
    side[3, 0] = -1                                        # already in a slot: ignored
    side[2, 1] = -1
    uni[4, 1] = False                                      # leaves the universe: closed at row 4
    w = _book(side, uni)
    assert w[0, 0] == 0 and np.all(w[1:6, 0] == 0.04) and w[6, 0] == 0
    assert np.all(w[2:4, 1] == -0.04) and np.all(w[4:, 1] == 0)
    side[6, 0] = 1                                         # new slot allowed after the old one closed
    w = _book(side, uni)
    assert np.all(w[6:11, 0] == 0.04) and w[11, 0] == 0


def test_book_gross_cap_scales_new_slots_only():
    T, N = 8, 40
    side = np.zeros((T, N))
    uni = np.ones((T, N), dtype=bool)
    side[0, :20] = 1                                       # 0.8 gross
    side[1, 20:30] = -1                                    # 10 new slots, only 0.2 room
    w = _book(side, uni)
    assert np.all(w[1, :20] == 0.04)
    assert np.allclose(w[1, 20:30], -0.02, rtol=1e-6)
    assert np.abs(w).sum(axis=1).max() <= 1.0
    assert np.allclose(w[5, 20:30], -0.02, rtol=1e-6) and np.all(w[5, :20] == 0)


# ------------------------------------------------------------------ pipeline
@pytest.fixture(scope="module")
def full():
    v = _view()
    return v, target_weights(v, PARAMS)


def test_pipeline_shape_gross_and_first_scores(full):
    v, w = full
    assert w.shape == (len(v.dates), len(v.instruments))
    assert np.all(np.isfinite(w))
    assert np.abs(w).sum(axis=1).max() <= 1.0 + 1e-12
    assert np.all(w[:504] == 0)                            # no model before the minimum training sample
    assert np.all(w[~v.universe] == 0)                     # only members held
    nz = np.abs(w[w != 0])
    assert np.all(nz <= 0.04 + 1e-12)


def test_pipeline_point_in_time(full):
    v, w = full
    n = 1000
    wc = target_weights(_cut(v, n), PARAMS)
    assert np.array_equal(wc, w[:n])


def test_num_leaves_reaches_the_model(full):
    v, _ = full
    _, ent4, s4 = _entrants_and_scores(v, {"num_leaves": 4, "tail": 0.2})
    _, ent16, s16 = _entrants_and_scores(v, {"num_leaves": 16, "tail": 0.2})
    assert np.array_equal(ent4, ent16)                     # entrants do not depend on the model
    f = np.isfinite(s4)
    assert np.array_equal(f, np.isfinite(s16)) and f.sum() > 0
    assert not np.allclose(s4[f], s16[f])


def test_sets_match_book_entries(full):
    # diagnostic.py marks exactly these sets (same code); every new book slot is a set member with its sign
    v, w = full
    uni, ent, score = _entrants_and_scores(v, PARAMS)
    assert np.all(np.isnan(score[~ent]))                   # scores only on window entrants
    assert ent.sum() > 500 and np.isfinite(score).sum() > 100
    side = _sets(score, PARAMS["tail"])
    assert np.all(ent[side != 0])
    assert (side == 1).sum() > 0 and (side == -1).sum() > 0
    opened = (w != 0) & np.vstack([np.ones((1, w.shape[1]), dtype=bool), w[:-1] == 0])
    assert opened.sum() > 0
    assert np.all(np.sign(w[opened]) == side[opened])
