import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS
from strategy import _shock_rows, _trailing_mean_all_valid, _rolling_beta, _select, _universe, _trailing_nansum
from strategy import books

T = 120
N = 16          # BTCUSDT.P + 15 coins
NAMES = tuple(["BTCUSDT.P"] + ["C%02dUSDT.P" % i for i in range(1, N)])


def make_view(vix=None, funding=None, universe=None, seed=1, betas=None, dv=None):
    rng = np.random.default_rng(seed)
    btc = rng.normal(0.0, 0.03, T)
    if betas is None:
        betas = np.linspace(0.2, 2.0, N)
        betas[0] = 1.0
    ret = btc[:, None] * betas[None, :] + rng.normal(0.0, 0.002, (T, N))
    ret[:, 0] = btc
    ret_co = np.zeros((T, N))
    ret_oc = ret
    close = 100.0 * np.cumprod(1.0 + ret, axis=0)
    if funding is None:
        funding = np.tile(np.arange(N, dtype=float)[None, :] * 1e-4, (T, 1))
    if vix is None:
        vix = np.full(T, 20.0)
    if dv is None:
        dv = np.tile(np.linspace(2e9, 1e9, N)[None, :], (T, 1))
    uni = np.ones((T, N), dtype=bool) if universe is None else universe
    dates = np.arange(18262, 18262 + T).astype("datetime64[D]")
    return DataView(
        dates=dates,
        instruments=NAMES,
        asset_class=tuple(["crypto_perp"] * N),
        ret_co=ret_co,
        ret_oc=ret_oc,
        tradable=np.ones((T, N), dtype=bool),
        listed=np.ones((T, N), dtype=bool),
        delisting=np.zeros((T, N), dtype=bool),
        close=close,
        dollar_volume=dv,
        universe=uni,
        extras={"funding": funding},
        series={"fred_macro.VIXCLS": vix},
        cash_ret=np.zeros(T),
    )


def vix_with_shock_at(rows, jump=0.2):
    v = np.full(T, 20.0)
    level = 20.0
    for t in range(T):
        if t in rows:
            level = level * np.exp(jump)
        v[t] = level
    return v


# ---------- (1) shock detection ----------

def test_shock_needs_new_observation_and_threshold():
    v = np.array([np.nan, 20.0, 20.0, 22.2, 22.2, np.nan, 24.0, 20.0, 22.0])
    s = _shock_rows(v, 0.1)
    # row 3: log(22.2/20)=0.104 fires; row 4 carries no new obs; row 6: log(24/22.2)=0.078 no;
    # row 8: log(22/20)=0.0953 no
    assert s.tolist() == [False, False, False, True, False, False, False, False, False]
    s2 = _shock_rows(v, 0.075)
    assert s2.tolist() == [False, False, False, True, False, False, True, False, True]


def test_shock_weekend_rows_never_fire():
    v = np.array([20.0, 23.0, 23.0, 23.0])        # forward-filled weekend after a jump
    assert _shock_rows(v, 0.1).tolist() == [False, True, False, False]


# ---------- (2) crowding ----------

def test_crowd_needs_all_valid():
    f = np.array([[1.0], [2.0], [np.nan], [4.0], [5.0], [6.0]])
    m = _trailing_mean_all_valid(f, 2)
    assert np.isnan(m[0, 0]) and m[1, 0] == pytest.approx(1.5)
    assert np.isnan(m[2, 0]) and np.isnan(m[3, 0])
    assert m[4, 0] == pytest.approx(4.5) and m[5, 0] == pytest.approx(5.5)


def test_short_views_do_not_crash():
    f = np.ones((3, 2))
    assert np.isnan(_trailing_mean_all_valid(f, 7)).all()
    assert _trailing_nansum(f, 30)[2, 0] == 3.0


# ---------- (3) beta ----------

def test_beta_recovers_slope_and_min_pairs():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.02, 100)
    r = np.column_stack([x, 2.0 * x, 0.5 * x])
    b = _rolling_beta(r, 0)
    assert np.isnan(b[43]).all()                  # 44 pairs: not enough
    assert b[44, 0] == 1.0
    assert b[99, 1] == pytest.approx(2.0) and b[99, 2] == pytest.approx(0.5)
    r2 = r.copy()
    r2[80:99, 1] = np.nan                         # only 41 valid pairs in rows 40..99 -> ineligible
    assert np.isnan(_rolling_beta(r2, 0)[99, 1])


# ---------- (4) selection ----------

def test_select_halves_and_legs():
    n = 14
    beta = np.arange(n, dtype=float)              # low half cols 0..6, high half 7..13
    crowd = np.array([5, 1, 2, 9, 8, 3, 7, 0, 1, 9, 8, 7, 2, 3], dtype=float)
    qv = np.ones(n)
    s, l = _select(crowd, beta, qv, np.ones(n, bool))
    assert sorted(s[:3]) == [3, 4, 6] and sorted(l[:3]) == [1, 2, 5]
    assert sorted(s[3:]) == [9, 10, 11] and sorted(l[3:]) == [7, 8, 12]


def test_select_ties_by_higher_volume_and_min_half():
    n = 12
    beta = np.arange(n, dtype=float)
    crowd = np.zeros(n)
    qv = np.arange(n, dtype=float)                # higher column = higher volume
    s, l = _select(crowd, beta, qv, np.ones(n, bool))
    assert sorted(s[:3]) == [3, 4, 5] and sorted(l[:3]) == [0, 1, 2]
    assert _select(crowd, beta, qv, np.arange(n) < 11) is None    # 11 eligible -> halves of 5


# ---------- (5) book and holding ----------

def test_book_weights_and_hold_window():
    d = make_view(vix=vix_with_shock_at({70}))
    w = target_weights(d, PARAMS)
    assert w.shape == (T, N)
    row = w[70]
    assert np.sum(row > 0) == 6 and np.sum(row < 0) == 6
    assert np.nansum(np.abs(row)) == pytest.approx(1.0) and np.nansum(row) == pytest.approx(0.0)
    assert np.isnan(w[71:75]).all()               # held, no rebalancing
    assert (w[75] == 0).all() and (w[:70] == 0).all() and (w[76:] == 0).all()
    # betas: cols 1..6 < 1 = BTC (col 0) < col 7 < ... -> low half {0..7}, high half {8..15};
    # funding rises with the column: shorts are the top 3 of each half, longs the bottom 3
    assert sorted(np.flatnonzero(row < 0)) == [5, 6, 7, 13, 14, 15]
    assert sorted(np.flatnonzero(row > 0)) == [0, 1, 2, 8, 9, 10]
    assert np.allclose(row[row != 0], np.sign(row[row != 0]) / 12.0)


def test_no_book_before_beta_warmup():
    d = make_view(vix=vix_with_shock_at({30}))
    assert (target_weights(d, PARAMS) == 0).all()


def test_new_shock_restarts_clock():
    d = make_view(vix=vix_with_shock_at({70, 73}))
    w = target_weights(d, PARAMS)
    assert np.isfinite(w[73]).all() and np.nansum(np.abs(w[73])) == pytest.approx(1.0)
    assert np.isnan(w[74:78]).all() and (w[78] == 0).all()


def test_universe_exit_closes_without_replacement():
    uni = np.ones((T, N), dtype=bool)
    d0 = make_view(vix=vix_with_shock_at({70}))
    held = np.flatnonzero(target_weights(d0, PARAMS)[70] != 0)
    j = held[0]
    uni[72:, j] = False
    w = target_weights(make_view(vix=vix_with_shock_at({70}), universe=uni), PARAMS)
    assert w[72, j] == 0.0
    assert np.isnan(np.delete(w[72], j)).all()
    assert np.isnan(w[73, j])                     # already closed, nothing new


def test_ineligible_half_means_no_book():
    uni = np.zeros((T, N), dtype=bool)
    uni[:, :11] = True                            # 11 members -> halves of 5 < 6
    w = target_weights(make_view(vix=vix_with_shock_at({70}), universe=uni), PARAMS)
    assert (w == 0).all()


def test_params_change_output():
    v = np.full(T, 20.0)
    v[70:] = 20.0 * np.exp(0.11)                  # log change 0.11
    d = make_view(vix=v)
    assert np.any(target_weights(d, {"funding_days": 7, "vix_shock": 0.1}) != 0)
    assert (target_weights(d, {"funding_days": 7, "vix_shock": 0.125}) == 0).all()
    # funding_days: recent funding reverses the ranking only for a short window
    f = np.tile(np.arange(N, dtype=float)[None, :] * 1e-4, (T, 1))
    f[68:71] = f[68:71, ::-1] * 2                 # fd=3: reversed ranking; fd=14: original ranking
    d2 = make_view(vix=vix_with_shock_at({70}), funding=f)
    w3 = target_weights(d2, {"funding_days": 3, "vix_shock": 0.1})[70]
    w14 = target_weights(d2, {"funding_days": 14, "vix_shock": 0.1})[70]
    assert not np.array_equal(w3, w14)


def test_point_in_time_truncation():
    d = make_view(vix=vix_with_shock_at({70, 90}))
    full = target_weights(d, PARAMS)
    for cut in (65, 71, 80, 92):
        dc = make_view(vix=vix_with_shock_at({70, 90}))
        dc = DataView(
            dates=dc.dates[:cut], instruments=dc.instruments, asset_class=dc.asset_class,
            ret_co=dc.ret_co[:cut], ret_oc=dc.ret_oc[:cut], tradable=dc.tradable[:cut],
            listed=dc.listed[:cut], delisting=dc.delisting[:cut], close=dc.close[:cut],
            dollar_volume=dc.dollar_volume[:cut], universe=dc.universe[:cut],
            extras={"funding": dc.extras["funding"][:cut]},
            series={"fred_macro.VIXCLS": dc.series["fred_macro.VIXCLS"][:cut]}, cash_ret=dc.cash_ret[:cut])
        np.testing.assert_array_equal(target_weights(dc, PARAMS), full[:cut])


def test_fallback_universe_top_n():
    dv = np.tile(np.arange(40, dtype=float)[None, :], (5, 1))
    dv[:, 0] = np.nan

    class V:
        universe = None
        close = np.ones((5, 40))
        listed = np.ones((5, 40), dtype=bool)
    m = _universe(V, _trailing_nansum(dv, 30))
    assert m.sum(axis=1).tolist() == [30] * 5
    assert m[0, 39] and m[0, 10] and not m[0, 9] and not m[0, 0]


def test_missing_vix_raises():
    d = make_view()
    d.series.pop("fred_macro.VIXCLS")
    with pytest.raises(KeyError):
        target_weights(d, PARAMS)


def test_books_match_weights():
    d = make_view(vix=vix_with_shock_at({70, 90}))
    w = target_weights(d, PARAMS)
    bl, _ = books(d, PARAMS)
    assert [b[0] for b in bl] == [70, 90]
    for t, s, l in bl:
        assert (w[t, s] < 0).all() and (w[t, l] > 0).all() and len(set(s) | set(l)) == 12
