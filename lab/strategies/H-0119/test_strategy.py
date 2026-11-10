import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import (
    PARAMS,
    SIGNAL_ETFS,
    target_weights,
    build_features,
    commodity_moves,
    exposure_regression,
    last_session_of_week,
    model_book,
)


def weekdays(n_rows):
    """n_rows weekdays starting on epoch day 4, a Monday."""
    d = np.arange(4, 4 + 2 * n_rows)
    d = d[(d + 3) % 7 < 5][:n_rows]
    return d.astype("datetime64[D]")


def make_view(T=700, n_stocks=130, seed=0, etfs=SIGNAL_ETFS, late=None, leave=None):
    rng = np.random.default_rng(seed)
    names = tuple(etfs) + tuple(f"E{100 + i}" for i in range(n_stocks))
    N = len(names)
    spy = rng.normal(0, 0.01, T)
    uso = rng.normal(0, 0.02, T) + 0.3 * spy
    com = rng.normal(0, 0.012, (T, 4)) + 0.2 * spy[:, None]
    r = np.zeros((T, N))
    etf_r = {"SPY": spy, "USO": uso, "GLD": com[:, 0], "UNG": com[:, 1], "DBA": com[:, 2], "DBB": com[:, 3]}
    for j, e in enumerate(etfs):
        r[:, j] = etf_r[e]
    ne = len(etfs)
    bm = rng.uniform(0.5, 1.5, n_stocks)
    bg = np.zeros(n_stocks)
    bg[:10] = 2.0                       # strong gold producers
    bg[10:20] = -2.0                    # gold users
    r[:, ne:] = spy[:, None] * bm + com[:, [0]] * bg + rng.normal(0, 0.015, (T, n_stocks))
    ret_co = r * 0.3
    ret_oc = (1 + r) / (1 + ret_co) - 1
    listed = np.ones((T, N), dtype=bool)
    close = 50 + np.cumsum(rng.normal(0, 0.5, (T, N)), axis=0)
    close = np.abs(close) + 5
    dv = rng.uniform(1e7, 1e9, (T, N))
    uni = np.zeros((T, N), dtype=bool)
    uni[:, ne:] = True
    if late is not None:                # stock column ne + 0 lists at row `late`
        j = ne
        listed[:late, j] = False
        close[:late, j] = np.nan
        dv[:late, j] = np.nan
        ret_co[:late, j] = 0.0
        ret_oc[:late, j] = 0.0
        uni[:late, j] = False
    if leave is not None:               # stocks leave the universe on given rows
        for row, j in leave:
            uni[row:, ne + j] = False
    return DataView(
        dates=weekdays(T),
        instruments=names,
        asset_class=tuple("us_etf" if n in etfs else "us_equity" for n in names),
        ret_co=ret_co,
        ret_oc=ret_oc,
        tradable=listed.copy(),
        listed=listed,
        delisting=np.zeros((T, N), dtype=bool),
        close=close,
        dollar_volume=dv,
        universe=uni,
    )


def test_last_session_of_week_is_friday():
    dates = weekdays(20)            # Monday .. Friday, 4 full weeks
    f = last_session_of_week(dates)
    assert f.tolist() == [False, False, False, False, True] * 4


def test_commodity_moves_known_shock():
    rng = np.random.default_rng(1)
    L = 252
    spy = rng.normal(0, 0.01, L)
    uso = rng.normal(0, 0.02, L)
    eps = rng.normal(0, 0.01, (L, 4))
    eps[-10:, 1] += 0.02            # UNG gets a +20 % residual move over the last 10 sessions
    com = 0.001 + 0.5 * spy[:, None] - 0.2 * uso[:, None] + eps
    e, R, z = commodity_moves(spy, uso, com)
    X = np.column_stack([np.ones(L), spy, uso])
    assert np.allclose(X.T @ e, 0.0, atol=1e-10)          # residual to [1, SPY, USO]
    assert np.allclose(R, e[-10:].sum(axis=0))
    assert np.allclose(z, R / (e.std(axis=0, ddof=1) * np.sqrt(10)))
    assert z[1] > 3.0
    assert np.all(np.abs(z[[0, 2, 3]]) < 3.0)


def test_exposure_regression_recovers_betas_and_masks():
    rng = np.random.default_rng(2)
    L, n = 252, 4
    F = rng.normal(0, 0.01, (L, 6))
    B = np.array([[0.0, 1.0, 0.2, 2.0, 0.0, 0.0, 0.0],
                  [0.001, 0.8, 0.0, 0.0, -1.5, 0.0, 0.0],
                  [0.0, 1.2, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    X = np.column_stack([np.ones(L), F])
    noise = rng.normal(0, 0.001, (L, n))
    Y = X @ B.T + noise
    M = np.ones((L, n), dtype=bool)
    M[:30, 1] = False               # 222 valid rows: still estimated
    Y[:30, 1] = 1e3                 # garbage on masked rows must be ignored
    M[:60, 3] = False               # 192 valid rows: below 200, not estimated
    b, t, ivol, ok = exposure_regression(Y, M, F)
    assert ok.tolist() == [True, True, True, False]
    assert np.allclose(b[:3], B[:3], atol=0.05)
    assert np.all(np.isnan(b[3])) and np.isnan(ivol[3]) and np.all(np.isnan(t[3]))
    assert abs(t[0, 3]) > 50 and abs(t[2, 3]) < 5
    resid = Y[-63:, 0] - X[-63:] @ b[0]
    assert np.isclose(ivol[0], resid.std(ddof=1))
    assert np.isclose(ivol[0], 0.001, rtol=0.3)


def test_features_point_in_time_shapes_and_late_listing():
    data = make_view(T=700, n_stocks=40, late=200)
    feats, scored, decision, shock, im = build_features(data, 1.5)
    T = 700
    assert feats.shape == (5, T, data.ret_co.shape[1])
    assert not shock[:251].any() and not scored[:251].any()
    assert np.all(decision[shock])                 # shocks only on decision dates
    assert not scored[~decision].any()
    assert not scored[:, :6].any()                 # signal ETFs never scored
    assert np.all(data.universe[scored])
    assert shock.sum() > 5
    rows = np.flatnonzero(shock)
    # the late stock (listed from row 200) needs 200 valid returns: first scored row >= 399
    late_rows = np.flatnonzero(scored[:, 6])
    assert late_rows.size > 0 and late_rows.min() >= 399
    assert not scored[rows[rows < 399], 6].any()
    # F5 is date-level: identical for all scored members of a date
    for t in rows:
        s = feats[4, t, scored[t]]
        assert np.allclose(s, s[0]) and s[0] >= 1.5
    # unscored entries are NaN
    assert np.all(np.isnan(feats[:, ~scored]))
    assert np.all(np.isnan(im[~scored]))


def test_implied_move_sign_follows_exposure():
    data = make_view(T=700, n_stocks=40)
    feats, scored, decision, shock, im = build_features(data, 1.0)
    # gold producers (stocks 0-9, b = +2) and users (10-19, b = -2) have opposite implied moves on
    # dates where only GLD is shocked; check via correlation over shock dates
    ne = len(SIGNAL_ETFS)
    prod = im[:, ne:ne + 10]
    user = im[:, ne + 10:ne + 20]
    rows = np.flatnonzero(shock)
    prod_m = np.nanmean(prod[rows], axis=1)
    user_m = np.nanmean(user[rows], axis=1)
    big = np.abs(prod_m) > np.nanmedian(np.abs(prod_m))
    assert np.mean(np.sign(prod_m[big]) == -np.sign(user_m[big])) > 0.8


def test_shock_z_changes_shock_dates():
    data = make_view(T=700, n_stocks=20)
    s10 = build_features(data, 1.0)[3]
    s15 = build_features(data, 1.5)[3]
    s20 = build_features(data, 2.0)[3]
    assert s10.sum() > s15.sum() > s20.sum()
    assert np.all(s10[s15]) and np.all(s15[s20])


def test_missing_signal_etf_raises():
    data = make_view(T=300, n_stocks=10, etfs=("SPY", "USO", "GLD", "DBA", "DBB"))
    with pytest.raises(Exception):
        target_weights(data, PARAMS)


@pytest.fixture(scope="module")
def full_run():
    data = make_view(T=900, n_stocks=130, seed=3, leave=[(700, j) for j in range(0, 130, 3)])
    w = target_weights(data, PARAMS)
    book = model_book(data, PARAMS)
    w_small = target_weights(data, {"shock_z": 1.5, "top": 0.05})
    return data, w, book, w_small


def test_book_structure(full_run):
    data, w, (book, decision, shock, im, uni), w_small = full_run
    T, N = w.shape
    assert w.shape == data.ret_co.shape
    # decision rows are fully specified; non-shock decision rows are flat
    assert np.all(np.isfinite(w[decision]))
    assert np.all(w[decision & ~shock] == 0.0)
    # non-decision rows: NaN except zeros that close a non-member
    nd = ~decision[:, None] & np.isfinite(w)
    assert np.all(w[nd] == 0.0) and not np.any(uni[nd])
    # gross <= 1, dollar neutral, members only, ETFs never held
    gross = np.nansum(np.abs(w), axis=1)
    assert np.all(gross <= 1.0 + 1e-9)
    active = np.flatnonzero(gross > 0)
    assert active.size > 0, "no book after the first fit"
    for t in active:
        assert np.isclose(w[t][w[t] > 0].sum(), 0.5) and np.isclose(w[t][w[t] < 0].sum(), -0.5)
    assert np.all(uni[np.nan_to_num(w) != 0])
    assert np.all(np.nan_to_num(w[:, :6]) == 0)
    # first book only after warm-up (252 rows) plus a fit
    assert active.min() > 251


def test_universe_exit_closed_between_decisions(full_run):
    data, w, (book, decision, shock, im, uni), w_small = full_run
    held_after = book[np.flatnonzero(decision & (np.arange(len(decision)) < 700)).max()]
    leavers = [6 + j for j in range(0, 130, 3) if held_after[6 + j] != 0]
    if not decision[700]:
        for j in leavers:
            assert w[700, j] == 0.0


def test_top_changes_leg_size(full_run):
    data, w, book, w_small = full_run
    n_big = (np.nan_to_num(w) != 0).sum(axis=1)
    n_small = (np.nan_to_num(w_small) != 0).sum(axis=1)
    act = n_big > 0
    assert act.any()
    assert np.all(n_small[act] < n_big[act])


def test_truncation_point_in_time(full_run):
    data, w, book, w_small = full_run
    cut = 650
    sub = DataView(
        dates=data.dates[:cut], instruments=data.instruments, asset_class=data.asset_class,
        ret_co=data.ret_co[:cut], ret_oc=data.ret_oc[:cut], tradable=data.tradable[:cut],
        listed=data.listed[:cut], delisting=data.delisting[:cut], close=data.close[:cut],
        dollar_volume=data.dollar_volume[:cut], universe=data.universe[:cut],
    )
    ws = target_weights(sub, PARAMS)
    assert np.array_equal(np.isnan(ws), np.isnan(w[:cut]))
    assert np.allclose(np.nan_to_num(ws), np.nan_to_num(w[:cut]))
