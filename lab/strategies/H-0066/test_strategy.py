import numpy as np
import pytest

from strategy import target_weights, PARAMS
from strategy import peer_table, announcer_z, month_starts, build_book, market_adjusted, valid_returns


# ---------------------------------------------------------------- peer map

def test_peer_table_picks_most_correlated():
    rng = np.random.default_rng(1)
    base = rng.normal(size=300)
    c0 = base
    c1 = base + 0.1 * rng.normal(size=300)     # very close to 0
    c2 = base + 1.0 * rng.normal(size=300)     # medium
    c3 = rng.normal(size=300)                  # unrelated
    W = np.column_stack([c0, c1, c2, c3])
    tab = peer_table(W, 2)
    assert list(tab[0]) == [1, 2]
    assert list(tab[1]) == [0, 2]
    assert tab.shape == (4, 2)
    assert 0 not in tab[0] and 1 not in tab[1]    # never itself


def test_peer_table_min_obs_and_fewer_than_k():
    rng = np.random.default_rng(2)
    base = rng.normal(size=252)
    W = np.column_stack([base, base + 0.1 * rng.normal(size=252), base + 0.1 * rng.normal(size=252)])
    W[:100, 2] = np.nan                        # only 152 common observations -> does not qualify
    tab = peer_table(W, 5)
    assert list(tab[0]) == [1, -1, -1, -1, -1]
    assert list(tab[2]) == [-1, -1, -1, -1, -1]


def test_peer_table_matches_numpy_corr_pairwise():
    rng = np.random.default_rng(3)
    W = rng.normal(size=(252, 6))
    W[:, 1] += W[:, 0]
    W[:30, 3] = np.nan
    tab = peer_table(W, 5)
    # reference: pairwise complete corr
    ref = np.full((6, 6), -np.inf)
    for i in range(6):
        for j in range(6):
            if i != j:
                ok = np.isfinite(W[:, i]) & np.isfinite(W[:, j])
                ref[i, j] = np.corrcoef(W[ok, i], W[ok, j])[0, 1]
    for i in range(6):
        assert list(tab[i]) == list(np.argsort(-ref[i], kind="stable")[:5])


# ---------------------------------------------------------------- announcer z

def test_announcer_z_uses_rows_before_t_and_min_obs():
    a = np.zeros((100, 1), dtype=np.float32)
    a[:, 0] = np.where(np.arange(100) % 2 == 0, 0.01, -0.01)
    a[80, 0] = 0.1
    z = announcer_z(a, [80], [0])
    sd = np.std(a[17:80, 0].astype(np.float64), ddof=1)
    assert z[0] == pytest.approx(0.1 / sd, rel=1e-5)
    # the big value on t itself does not enter the sd: changing a[t] scales z linearly
    a2 = a.copy()
    a2[80, 0] = 0.2
    assert announcer_z(a2, [80], [0])[0] == pytest.approx(2 * z[0], rel=1e-5)
    # fewer than 40 valid observations -> NaN
    a3 = a.copy()
    a3[17:60, 0] = np.nan                      # 20 valid left
    assert np.isnan(announcer_z(a3, [80], [0])[0])
    # early row: only 30 rows of history -> NaN
    assert np.isnan(announcer_z(a, [30], [0])[0])
    assert np.isfinite(announcer_z(a, [45], [0])[0])


# ---------------------------------------------------------------- returns

def test_market_adjustment_and_validity():
    T, N = 3, 3
    co = np.zeros((T, N), dtype=np.float32)
    oc = np.array([[0, 0, 0], [0.02, 0.04, 0.5], [0.01, 0.0, 0.0]], dtype=np.float32)
    close = np.ones((T, N))
    listed = np.ones((T, N), bool)
    tradable = np.ones((T, N), bool)
    tradable[2, 1] = False
    r, valid = valid_returns(co, oc, close, listed, tradable)
    assert not valid[0].any()                  # no previous close on row 0
    assert valid[1].all() and not valid[2, 1]
    uni = np.ones((T, N), bool)
    uni[1, 2] = False                          # non-member: excluded from the mean, still gets a
    a = market_adjusted(r, valid, uni)
    assert a[1, 0] == pytest.approx(-0.01, abs=1e-6)
    assert a[1, 1] == pytest.approx(0.01, abs=1e-6)
    assert a[1, 2] == pytest.approx(0.47, abs=1e-6)
    assert np.isnan(a[2, 1])


def test_month_starts():
    # day numbers since the epoch: Jan 30, Jan 31, Feb 1, Feb 2, Mar 1 (1970)
    d = np.array([29, 30, 31, 32, 59], dtype="datetime64[D]")
    assert list(month_starts(d)) == [False, False, True, False, True]


# ---------------------------------------------------------------- book

def _flat(T, N):
    ea = np.zeros((T, N), dtype=np.float32)
    ones = np.ones((T, N), bool)
    return ea, ones.copy(), ones.copy(), ones.copy()


def test_book_hold_and_replace():
    T, N = 15, 2
    ea, uni, trad, lst = _flat(T, N)
    ev = np.zeros((T, N))
    ev[2, 0] = 1
    ev[3, 1] = -1
    ev[5, 1] = 1                               # replaces: new side, new clock
    W = build_book(ev, ea, uni, trad, lst, 5)
    assert list(W[:, 0] != 0) == [False, False] + [True] * 5 + [False] * 8
    assert np.allclose(W[2:7, 0], 0.01)
    assert np.allclose(W[3:5, 1], -0.01)
    assert np.allclose(W[5:10, 1], 0.01)
    assert W[10, 1] == 0
    W3 = build_book(ev, ea, uni, trad, lst, 3)
    assert list(W3[:, 0] != 0) == [False, False] + [True] * 3 + [False] * 10


def test_book_early_exits():
    T, N = 12, 3
    ea, uni, trad, lst = _flat(T, N)
    ev = np.zeros((T, N))
    ev[1, :] = [1, -1, 1]
    ea[3, 0] = 1                               # own release on row 3 -> 0 from row 3 on
    uni[4, 1] = False                          # leaves universe on row 4
    trad[5, 2] = False                         # stops trading on row 5
    W = build_book(ev, ea, uni, trad, lst, 7)
    assert np.allclose(W[1:3, 0], 0.01) and np.all(W[3:, 0] == 0)
    assert np.allclose(W[1:4, 1], -0.01) and np.all(W[4:, 1] == 0)
    assert np.allclose(W[1:5, 2], 0.01) and np.all(W[5:, 2] == 0)


def test_book_scaling_above_100():
    T, N = 3, 150
    ea, uni, trad, lst = _flat(T, N)
    ev = np.zeros((T, N))
    ev[0, :100] = 1
    ev[1, 100:] = -1
    W = build_book(ev, ea, uni, trad, lst, 5)
    assert np.allclose(np.abs(W[0, :100]), 0.01)
    assert np.allclose(np.abs(W[1]), 1.0 / 150)
    assert np.abs(W).sum(axis=1).max() <= 1.0 + 1e-12


# ---------------------------------------------------------------- end to end on a hand-made market
#
# Two blocks of six stocks: 0..5 share factor A, 6..11 share factor B. On row T0 stock 0 releases with
# a +15 % move and stock 6 with a -15 % move. Stock 1 reported 20 rows earlier, stock 2 is out of the
# universe on T0 (but was a member at the month end, so it is in the peer map), stock 7 has no 8-K
# coverage (NaN). Expected events on T0 with peers_k = 5: +1 on 3, 4, 5 and -1 on 8, 9, 10, 11.

T, N, T0 = 400, 12, 350


def _market(seed=0):
    from lab.framework.data import DataView
    rng = np.random.default_rng(seed)
    fa = rng.normal(0, 0.02, T)
    fb = rng.normal(0, 0.02, T)
    r = np.empty((T, N))
    r[:, :6] = fa[:, None] + rng.normal(0, 0.005, (T, 6))
    r[:, 6:] = fb[:, None] + rng.normal(0, 0.005, (T, 6))
    r[T0 - 20] = 0.0                          # stock 1's own release row: no move -> z = 0
    r[T0, 0] = 0.15
    r[T0, 6] = -0.15
    ret_co = np.zeros((T, N), dtype=np.float32)
    ret_oc = r.astype(np.float32)
    ea = np.zeros((T, N), dtype=np.float32)
    ea[T0, 0] = 1
    ea[T0, 6] = 1
    ea[T0 - 20, 1] = 1
    ea[:, 7] = np.nan
    uni = np.ones((T, N), bool)
    uni[T0, 2] = False
    ones = np.ones((T, N), bool)
    return DataView(
        dates=(np.arange(T) + 12000).astype("datetime64[D]"),
        instruments=tuple("E%d" % (100 + i) for i in range(N)),
        asset_class=tuple("us_equity" for _ in range(N)),
        ret_co=ret_co, ret_oc=ret_oc, tradable=ones.copy(), listed=ones.copy(),
        delisting=np.zeros((T, N), bool), close=np.ones((T, N), dtype=np.float32),
        dollar_volume=np.ones((T, N), dtype=np.float32), universe=uni,
        extras={"ea_reaction_day": ea}, series={}, cash_ret=np.zeros(T),
    )


def _cut(data, t, noise_seed=None):
    """Rows 0..t of `data`, or all rows with rows > t replaced by noise."""
    from lab.framework.data import DataView
    sl = slice(0, t + 1)
    if noise_seed is None:
        return DataView(
            dates=data.dates[sl], instruments=data.instruments, asset_class=data.asset_class,
            ret_co=data.ret_co[sl], ret_oc=data.ret_oc[sl], tradable=data.tradable[sl], listed=data.listed[sl],
            delisting=data.delisting[sl], close=data.close[sl], dollar_volume=data.dollar_volume[sl],
            universe=data.universe[sl], extras={"ea_reaction_day": data.extras["ea_reaction_day"][sl]},
            series={}, cash_ret=data.cash_ret[sl])
    rng = np.random.default_rng(noise_seed)
    oc = data.ret_oc.copy()
    oc[t + 1:] = rng.normal(0, 0.1, oc[t + 1:].shape)
    ea = data.extras["ea_reaction_day"].copy()
    ea[t + 1:] = (rng.random(ea[t + 1:].shape) < 0.2).astype(np.float32)
    return DataView(
        dates=data.dates, instruments=data.instruments, asset_class=data.asset_class,
        ret_co=data.ret_co, ret_oc=oc, tradable=data.tradable, listed=data.listed,
        delisting=data.delisting, close=data.close, dollar_volume=data.dollar_volume,
        universe=data.universe, extras={"ea_reaction_day": ea}, series={}, cash_ret=data.cash_ret)


def test_end_to_end_events_and_book():
    data = _market()
    p = dict(PARAMS, peers_k=5)
    W = target_weights(data, p)
    assert W.shape == (T, N)
    nz_rows = np.nonzero(np.abs(W).sum(axis=1))[0]
    assert list(nz_rows) == list(range(T0, T0 + 5))
    expected = np.zeros(N)
    expected[[3, 4, 5]] = 0.01
    expected[[8, 9, 10, 11]] = -0.01
    for t in range(T0, T0 + 5):
        assert np.allclose(W[t], expected)
    assert np.abs(W).sum(axis=1).max() <= 1.0


def test_end_to_end_params_change_output():
    data = _market()
    base = dict(PARAMS, peers_k=5)
    W = target_weights(data, base)
    W3 = target_weights(data, dict(base, hold_days=3))
    assert list(np.nonzero(np.abs(W3).sum(axis=1))[0]) == list(range(T0, T0 + 3))
    Wk = target_weights(data, dict(base, peers_k=2))      # at most 2 peers per announcer
    assert 0 < np.count_nonzero(Wk[T0]) <= 4
    assert not np.array_equal(Wk, W)
    Wz = target_weights(data, dict(base, z_threshold=1000.0))
    assert not np.any(Wz)


def test_end_to_end_point_in_time():
    data = _market()
    p = dict(PARAMS, peers_k=5)
    W = target_weights(data, p)
    for t in [T0 - 1, T0, T0 + 2, T0 + 30]:
        Wc = target_weights(_cut(data, t), p)
        assert np.array_equal(Wc, W[:t + 1])
        Wn = target_weights(_cut(data, t, noise_seed=t), p)
        assert np.array_equal(Wn[:t + 1], W[:t + 1])


def test_peer_events_on_hand_made_market():
    from strategy import peer_events
    data = _market()
    ev = peer_events(data, dict(PARAMS, peers_k=5))
    mask = ev != 0
    assert mask.sum() == 7 and set(np.nonzero(mask)[0]) == {T0}
    assert list(ev[T0]) == [0, 0, 0, 1, 1, 1, 0, 0, -1, -1, -1, -1]
