import numpy as np
import pytest

from strategy import PARAMS, target_weights
from strategy import (
    build_book, detect_events, hedge_basket, row_median, select_entries, slot_beta,
    trailing_compound_return, trailing_vol,
)


def test_params_match_card():
    assert PARAMS == {"n_estimators": 100, "top": 0.5}


# ------------------------------------------------------------------ events
def test_event_rise_and_dedup():
    T, N = 20, 3
    b = np.zeros((T, N))
    b[3:, 0] = 1          # rise at row 3
    b[5:, 0] = 2          # rise at row 5 -> within 5 sessions of row 3: not an event
    b[9:, 0] = 3          # rise at row 9 -> candidate at 5 is in 4..8: not an event
    b[15:, 0] = 4         # rise at row 15 -> no candidate in 10..14: event
    b[:, 1] = np.nan
    b[8:, 1] = 1.0        # NaN -> value is not a rise
    b[10:, 1] = 2.0       # rise at row 10: event
    b[12:, 2] = 1.0       # rise at row 12, but not a member
    uni = np.ones((T, N), dtype=bool)
    uni[12, 2] = False
    trad = np.ones((T, N), dtype=bool)
    ev = detect_events(b, uni, trad)
    assert set(zip(*np.nonzero(ev))) == {(3, 0), (15, 0), (10, 1)}


def test_event_dedup_boundary_and_tradable():
    T = 15
    b = np.zeros((T, 2))
    b[2:, 0] = 1
    b[8:, 0] = 2          # 6 rows later: prior window 3..7 empty -> event
    b[4:, 1] = 1          # rise at row 4 but not tradable
    trad = np.ones((T, 2), dtype=bool)
    trad[4, 1] = False
    b[6:, 1] = 2          # rise at row 6: the non-tradable rise is no candidate -> event
    ev = detect_events(b, np.ones((T, 2), dtype=bool), trad)
    assert set(zip(*np.nonzero(ev))) == {(2, 0), (8, 0), (6, 1)}


def test_event_point_in_time():
    rng = np.random.default_rng(1)
    b = np.cumsum(rng.random((60, 4)) < 0.2, axis=0).astype(float)
    uni = rng.random((60, 4)) < 0.9
    trad = np.ones((60, 4), dtype=bool)
    full = detect_events(b, uni, trad)
    for cut in (10, 33, 59):
        assert np.array_equal(detect_events(b[:cut], uni[:cut], trad[:cut]), full[:cut])


# ------------------------------------------------------------------ rolling inputs
def test_trailing_return_and_vol():
    r = np.full((30, 2), 0.01)
    r[:5, 1] = np.nan                       # late listing
    r21 = trailing_compound_return(r, 21)
    assert np.isnan(r21[19, 0]) and np.isclose(r21[20, 0], 1.01 ** 21 - 1)
    assert np.isnan(r21[24, 1]) and np.isclose(r21[25, 1], 1.01 ** 21 - 1)
    x = np.zeros((70, 1))
    x[::2, 0] = 0.02
    v = trailing_vol(x, 63)
    assert np.isnan(v[61, 0])
    seg = x[1:64, 0]
    assert np.isclose(v[63, 0], seg.std())


def test_row_median_members_only():
    x = np.array([[1.0, 2.0, 100.0, np.nan], [np.nan, np.nan, np.nan, np.nan]])
    m = np.array([[True, True, False, True], [True, True, True, True]])
    med = row_median(x, m)
    assert med[0] == 1.5 and np.isnan(med[1])


# ------------------------------------------------------------------ entry rule
def test_select_entries_quantile_and_minimum():
    T, N = 400, 5
    s = np.full((T, N), np.nan)
    # rows 0..49: one scored event each, scores 0..49
    for t in range(50):
        s[t, t % N] = float(t)
    # row 50: two events: threshold over 50 prior = median of 0..49 = 24.5
    s[50, 0] = 24.4
    s[50, 1] = 24.5
    e = select_entries(s, 0.5)
    assert not e[:50].any()                  # fewer than 50 prior scored events
    assert not e[50, 0] and e[50, 1]
    # top = 0.33: threshold = 0.67 quantile of 0..49
    thr = np.quantile(np.arange(50.0), 0.67)
    s2 = s.copy()
    s2[50, 0], s2[50, 1] = thr - 1e-9, thr
    e2 = select_entries(s2, 0.33)
    assert not e2[50, 0] and e2[50, 1]


def test_select_entries_window_strictly_before():
    T, N = 600, 2
    s = np.full((T, N), np.nan)
    for t in range(60):
        s[t, 0] = 100.0                      # old high scores, leave the window later
    for t in range(300, 360):
        s[t, 0] = 0.0
    s[400, 1] = 1.0
    e = select_entries(s, 0.5)
    # at row 400 the window 148..399 holds the 60 zeros only -> 1.0 enters
    assert e[400, 1]
    s[400, 1] = -1.0
    assert not select_entries(s, 0.5)[400, 1]
    # a same-row score never counts in its own threshold
    s3 = np.full((T, N), np.nan)
    s3[10, :] = 1.0
    assert not select_entries(s3, 0.5, min_prior=1)[10].any()


def test_top_changes_entries():
    rng = np.random.default_rng(3)
    s = np.where(rng.random((500, 20)) < 0.1, rng.normal(size=(500, 20)), np.nan)
    n = [select_entries(s, top).sum() for top in (0.33, 0.5, 0.67)]
    assert n[0] < n[1] < n[2]


# ------------------------------------------------------------------ book
def _market(T=300, N=60, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.01, (T, N))
    dvol = np.tile(np.arange(N, 0, -1, dtype=float), (T, 1))   # column 0 most liquid
    uni = np.ones((T, N), dtype=bool)
    trad = np.ones((T, N), dtype=bool)
    return r, dvol, uni, trad


def test_hedge_basket_excludes_held_and_nonmembers():
    r, dvol, uni, trad = _market()
    uni[10, 3] = False
    b = hedge_basket(dvol, uni[10], 10, {0, 5})
    assert b.size == 50
    assert 0 not in b and 5 not in b and 3 not in b
    assert list(b[:3]) == [1, 2, 4]


def test_beta_short_history_and_clip():
    r, dvol, uni, trad = _market()
    basket = np.arange(10, 60)
    assert slot_beta(r, 100, 1, basket) == 1.0          # 101 sessions < 126
    rb = np.nanmean(r[:, basket], axis=1)
    r2 = r.copy()
    r2[:, 1] = 3.0 * rb
    assert slot_beta(r2, 299, 1, basket) == 1.5
    r2[:, 1] = 0.8 * rb
    assert np.isclose(slot_beta(r2, 299, 1, basket), 0.8)


def test_book_slot_hold_and_hedge():
    r, dvol, uni, trad = _market()
    T, N = r.shape
    entry = np.zeros((T, N), dtype=bool)
    entry[50, 0] = True                                  # beta 1.0 (51 sessions)
    w = build_book(entry, uni, trad, r, dvol)
    assert np.allclose(w[:50], 0) and np.allclose(w[55:], 0)
    for t in range(50, 55):
        assert np.isclose(w[t, 0], 0.08)
        assert np.isclose(w[t].sum(), 0.0)               # beta 1 -> dollar neutral
        assert np.isclose(np.abs(w[t]).sum(), 0.16)
        assert np.allclose(w[t, 1:51], -0.08 / 50) and np.allclose(w[t, 51:], 0)


def test_book_early_close_and_no_extension():
    r, dvol, uni, trad = _market()
    T, N = r.shape
    entry = np.zeros((T, N), dtype=bool)
    entry[50, 7] = True
    entry[52, 7] = True                                  # already held: does not extend
    trad[53, 7] = False                                  # not tradable -> slot closes at row 53
    w = build_book(entry, uni, trad, r, dvol)
    assert np.isclose(w[52, 7], 0.08)
    assert np.allclose(w[53:], 0)
    # leaving the universe closes as well
    trad[53, 7] = True
    uni[52, 7] = False
    w = build_book(entry, uni, trad, r, dvol)
    assert np.isclose(w[51, 7], 0.08) and np.allclose(w[52:], 0)


def test_book_gross_cap_pro_rata():
    r, dvol, uni, trad = _market(N=80)
    T, N = r.shape
    entry = np.zeros((T, N), dtype=bool)
    entry[60, 60:75] = True                              # 15 slots x 0.16 = 2.4 gross
    w = build_book(entry, uni, trad, r, dvol)
    g = np.abs(w).sum(axis=1)
    assert g.max() <= 1.0 + 1e-9
    assert np.allclose(w[60, 60:75], 0.08 / 2.4)
    assert (g[61:65] > 0).all() and np.allclose(g[65:], 0)


def test_book_hedge_name_leaving_universe_is_dropped():
    r, dvol, uni, trad = _market()
    T, N = r.shape
    entry = np.zeros((T, N), dtype=bool)
    entry[50, 0] = True
    uni[52:, 1] = False
    w = build_book(entry, uni, trad, r, dvol)
    assert np.isclose(w[51, 1], -0.08 / 50)
    assert np.allclose(w[52:55, 1], 0) and np.isclose(w[53, 2], -0.08 / 50)


# ------------------------------------------------------------------ features
class _FakeView:
    def __init__(self, T=100, N=6, seed=0):
        rng = np.random.default_rng(seed)
        self.ret_co = rng.normal(0, 0.005, (T, N))
        self.ret_oc = rng.normal(0, 0.01, (T, N))
        self.listed = np.ones((T, N), dtype=bool)
        self.listed[:50, 5] = False                      # late listing: < 63 sessions at row 80
        f = lambda v: np.full((T, N), v, dtype=np.float32)
        mcap = f(1e9)
        mcap[:, 4] = np.nan                              # no market cap yet
        self.extras = {"ins_buy_value_91d": f(1e6), "ins_buy_n_91d": f(3.0), "ins_buy_n_365d": f(5.0),
                       "ins_sell_value_182d": f(2e6), "sf1_arq_marketcap": mcap}


def test_build_features_event_rows_and_missing():
    from strategy import build_features
    d = _FakeView()
    T, N = d.ret_co.shape
    uni = np.ones((T, N), dtype=bool)
    event = np.zeros((T, N), dtype=bool)
    event[80, :] = True
    event[40, 0] = True                                   # fewer than 63 sessions of returns
    feats, comp = build_features(d, event, uni)
    assert len(feats) == 7
    assert comp[80, :4].all() and not comp[80, 4] and not comp[80, 5] and not comp[40, 0]
    assert comp.sum() == 4
    for f in feats:
        assert f.dtype == np.float32 and np.isnan(f[~comp]).all()
    assert np.isclose(feats[0][80, 0], np.log1p(1e6))
    assert np.isclose(feats[1][80, 0], 1e-3)
    assert feats[2][80, 0] == 3.0 and feats[3][80, 0] == 2.0
    assert np.isclose(feats[4][80, 0], 2e-3)


# ------------------------------------------------------------------ full pipeline
def test_full_pipeline_needs_universe():
    class V:
        universe = None
    with pytest.raises(Exception):
        target_weights(V(), PARAMS)
