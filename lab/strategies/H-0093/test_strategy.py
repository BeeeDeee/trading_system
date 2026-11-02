import numpy as np
import pytest

from lab.framework.data import DataView, synthetic
from strategy import target_weights, PARAMS

T = 2000


def stock_columns(rel=-3, n_q=32):
    """One stock: quarter p usable on row s_p = 20 + 63 p, its 8-K reaction session on s_p + rel.

    Net income v_p = 100 + p + 50 * (p % 4 == 0): the p % 4 == 0 season is the best of its five-quarter
    block (r = 1), seasons 1 and 2 beat one neighbour (r = 0.25), season 3 beats two (r = 0.5).
    """
    ni = np.full(T, np.nan)
    ea = np.zeros(T)
    starts = [20 + 63 * p for p in range(n_q) if 20 + 63 * p < T]
    for p, s in enumerate(starts):
        ni[s:] = 100.0 + p + (50.0 if p % 4 == 0 else 0.0)
        r = s + rel
        if 0 <= r < T:
            ea[r] = 1.0
    return ni, ea, starts


def make_view(ni_cols, ea_cols, n_hedge=5, uni=None, tradable=None, liq=None):
    n_ev = len(ni_cols)
    N = n_ev + n_hedge
    ni = np.full((T, N), np.nan, dtype=np.float32)
    ea = np.zeros((T, N), dtype=np.float32)
    for k in range(n_ev):
        ni[:, k] = ni_cols[k]
        ea[:, k] = ea_cols[k]
    if liq is None:
        liq = np.tile(np.arange(1, N + 1, dtype=np.float32), (T, 1))
    if uni is None:
        uni = np.ones((T, N), dtype=bool)
    if tradable is None:
        tradable = np.ones((T, N), dtype=bool)
    z = np.zeros((T, N))
    return DataView(
        dates=synthetic().dates[:T], instruments=tuple("E%d" % (k + 1) for k in range(N)),
        asset_class=tuple("us_equity" for _ in range(N)), ret_co=z, ret_oc=z.copy(), tradable=tradable,
        listed=np.ones((T, N), dtype=bool), delisting=np.zeros((T, N), dtype=bool), close=np.ones((T, N)),
        dollar_volume=np.ones((T, N)), universe=uni,
        extras={"sf1_arq_netinc": ni, "ea_reaction_day": ea, "liq_rank": liq}, series={}, cash_ret=None)


def truncate(v, k):
    return DataView(
        dates=v.dates[:k], instruments=v.instruments, asset_class=v.asset_class, ret_co=v.ret_co[:k],
        ret_oc=v.ret_oc[:k], tradable=v.tradable[:k], listed=v.listed[:k], delisting=v.delisting[:k],
        close=v.close[:k], dollar_volume=v.dollar_volume[:k], universe=v.universe[:k],
        extras={key: a[:k] for key, a in v.extras.items()}, series={}, cash_ret=None)


def expected(qs_sides):
    # event decided on t = r_q + 248 = 17 + 63 q + 248
    return [(265 + 63 * q, s) for q, s in qs_sides]


BASE = [(18, -1), (20, 1), (21, -1), (22, -1), (24, 1), (25, -1), (26, -1)]


def event_starts(W, j=0):
    w = W[:, j]
    prev = np.concatenate([[0.0], w[:-1]])
    idx = np.nonzero((w != 0) & (prev == 0))[0]
    return [(int(t), float(np.sign(w[t]))) for t in idx]


def test_events_sides_and_hold():
    ni, ea, _ = stock_columns()
    W = target_weights(make_view([ni], [ea]), PARAMS)
    assert event_starts(W) == expected(BASE)
    for t, s in expected(BASE):
        assert np.allclose(W[t:t + 5, 0], 0.05 * s)       # held decision rows t .. t+4
        assert W[t + 5, 0] == 0.0                         # closed at the open of t+6
        assert np.allclose(W[t:t + 5, 1:], -0.05 * s / 5)  # hedge: 5 non-event members
    assert np.all(W[:1399] == 0.0)                       # before 22 quarters of history: cash
    assert np.allclose(W.sum(axis=1), 0.0)               # dollar-neutral every row
    assert np.all(np.abs(W).sum(axis=1) <= 1.0 + 1e-12)


def test_threshold_and_years_back():
    ni, ea, _ = stock_columns()
    v = make_view([ni], [ea])
    W8 = target_weights(v, {"threshold": 0.8, "years_back": 5})
    assert event_starts(W8) == expected([(20, 1), (24, 1)])      # S = 0.25 no longer <= 0.2
    W7 = target_weights(v, {"threshold": 0.7, "years_back": 5})
    assert event_starts(W7) == expected(BASE)                     # S = 0.5 still neutral
    W4 = target_weights(v, {"threshold": 0.75, "years_back": 4})
    q4 = [(q, {0: 1, 1: -1, 2: -1}[q % 4]) for q in range(14, 28) if q % 4 != 3]
    assert event_starts(W4) == expected(q4)
    W6 = target_weights(v, {"threshold": 0.75, "years_back": 6})
    assert event_starts(W6) == expected([(22, -1), (24, 1), (25, -1), (26, -1)])


def test_amendment_vs_new_quarter():
    ni, ea, starts = stock_columns()
    base = target_weights(make_view([ni], [ea]), PARAMS)
    s10, s11 = starts[10], starts[11]
    a = ni.copy()
    a[s10 + 10:s11] = 1000.0          # within 40 sessions: overwrites quarter 10 -> its season scores change
    Wa = target_weights(make_view([a], [ea]), PARAMS)
    assert event_starts(Wa) != event_starts(base)
    b = ni.copy()
    b[s10 + 45:s11] = 1000.0          # 45 sessions: new quarter, then quarter 11's value overwrites it
    Wb = target_weights(make_view([b], [ea]), PARAMS)
    assert np.array_equal(Wb, base)
    c = ni.copy()
    c[s10 + 10:s11] = ni[s10] + 0.1   # small amendment that keeps every comparison: no change
    assert np.array_equal(target_weights(make_view([c], [ea]), PARAMS), base)


def test_missing_quarter_gap_voids_score():
    ni, ea, starts = stock_columns()
    g = ni.copy()
    g[starts[10]:starts[11]] = ni[starts[9]]     # quarter 10 never shows: gap of 126 sessions
    W = target_weights(make_view([g], [ea]), PARAMS)
    assert np.all(W == 0.0)


def test_alignment_guard():
    # event q = 20 on t: R1 = r_23 = t-59, latest quarter (23) usable on t-56
    ni, ea, _ = stock_columns()
    t = 265 + 63 * 20
    assert ea[t - 59] == 1.0
    late = ea.copy()
    late[t - 59] = 0.0
    late[t - 45] = 1.0                           # R1 = t-45: quarter 23 usable 11 sessions before R1 -> no score
    W = target_weights(make_view([ni], [late]), PARAMS)
    assert event_starts(W) == [x for x in expected(BASE) if x[0] != t]
    edge = ea.copy()
    edge[t - 59] = 0.0
    edge[t - 51] = 1.0                           # R1 = t-51: usable exactly 5 sessions before R1 -> scored
    W2 = target_weights(make_view([ni], [edge]), PARAMS)
    assert event_starts(W2) == expected(BASE)


def test_cluster_and_recent_release():
    ni, ea, _ = stock_columns()
    e2 = ea.copy()
    t = 265 + 63 * 20
    e2[t - 248 - 10] = 1.0       # earlier filing in the cluster: it anchors (on t-10), t-248 no longer does
    W = target_weights(make_view([ni], [e2]), PARAMS)
    want = [(tt - 10, s) if tt == t else (tt, s) for tt, s in expected(BASE)]
    assert event_starts(W) == want
    e3 = ea.copy()
    e3[t - 5] = 1.0                              # release already came early
    W3 = target_weights(make_view([ni], [e3]), PARAMS)
    assert t not in [x[0] for x in event_starts(W3)]


def test_universe_exit_and_tradable():
    ni, ea, _ = stock_columns()
    t = 265 + 63 * 20
    uni = np.ones((T, 6), dtype=bool)
    uni[t + 2, 0] = False
    W = target_weights(make_view([ni], [ea], uni=uni), PARAMS)
    assert np.allclose(W[t:t + 2, 0], 0.05)
    assert np.all(W[t + 2:t + 5, 0] == 0.0)      # closed at the first non-member row, not reopened
    assert np.all(W[t + 2:t + 5] == 0.0)          # no open event, no hedge
    uni2 = np.ones((T, 6), dtype=bool)
    uni2[t, 0] = False
    W2 = target_weights(make_view([ni], [ea], uni=uni2), PARAMS)
    assert np.all(W2[t:t + 5, 0] == 0.0)
    tr = np.ones((T, 6), dtype=bool)
    tr[t, 0] = False
    W3 = target_weights(make_view([ni], [ea], tradable=tr), PARAMS)
    assert np.all(W3[t:t + 5, 0] == 0.0)


def test_hedge_uses_100_most_liquid_non_event_members():
    ni, ea, _ = stock_columns()
    N = 105
    liq = np.tile(np.arange(1, N + 1, dtype=np.float32), (T, 1))
    liq[:, 0] = 50.0
    W = target_weights(make_view([ni], [ea], n_hedge=104, liq=liq), PARAMS)
    t = 265 + 63 * 20
    assert np.allclose(W[t, 1:101], -0.05 / 100)
    assert np.all(W[t, 101:] == 0.0)


def test_gross_cap_scales_everything():
    ni, ea, _ = stock_columns()
    W = target_weights(make_view([ni] * 25, [ea] * 25, n_hedge=5), PARAMS)
    t = 265 + 63 * 20
    assert np.allclose(W[t, :25], 0.05 / 2.5)
    assert np.allclose(W[t, 25:], -1.25 / 5 / 2.5)
    assert np.isclose(np.abs(W[t]).sum(), 1.0)
    assert np.isclose(W[t].sum(), 0.0)


def test_no_coverage_no_events_and_missing_inputs_raise():
    ni, ea, _ = stock_columns()
    W = target_weights(make_view([ni], [np.full(T, np.nan)]), PARAMS)
    assert np.all(W == 0.0)
    v = make_view([ni], [ea])
    del v.extras["ea_reaction_day"]
    with pytest.raises(KeyError):
        target_weights(v, PARAMS)


def test_point_in_time_truncation():
    ni, ea, starts = stock_columns()
    a = ni.copy()
    a[starts[25] + 10:starts[26]] = 1000.0       # a later amendment must not change earlier rows
    v = make_view([ni, a], [ea, ea])
    W = target_weights(v, PARAMS)
    for k in (1300, 1400, 1527, 1530, 1600, 1700, 1905):
        Wk = target_weights(truncate(v, k), PARAMS)
        assert np.array_equal(Wk, W[:k])
