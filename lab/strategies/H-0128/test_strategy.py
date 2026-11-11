import numpy as np
import pytest

from strategy import target_weights, PARAMS
from strategy import (decision_mask, weekday, ffill, lagged_sum, diff_lag, shock_z, betas, legs,
                      book_row, leverage, filing_age_ok, market_return)
from lab.framework.data import DataView

MONDAY0 = 4 + 7 * 1500   # epoch day number of a Monday


def bdays(n_cal, drop=()):
    d = (MONDAY0 + np.arange(n_cal)).astype("datetime64[D]")
    keep = np.is_busday(d)
    for k in drop:
        keep[k] = False
    return d[keep]


# ----------------------------------------------------------------------------------------------------- calendar
def test_weekday_and_decision_days():
    d = (MONDAY0 + np.arange(14)).astype("datetime64[D]")
    assert list(weekday(d)) == [0, 1, 2, 3, 4, 5, 6] * 2
    b = bdays(21)
    m = decision_mask(b)
    assert m.sum() == 3
    assert np.all(weekday(b[m]) == 4)


def test_holiday_friday_week_has_no_decision():
    b = bdays(21, drop=(11,))          # the second Friday is a holiday
    m = decision_mask(b)
    assert m.sum() == 2                # Thursday of that week is not knowable as the last session
    assert np.all(weekday(b[m]) == 4)


# ----------------------------------------------------------------------------------------------------- helpers
def test_ffill_lagged_sum_diff():
    x = np.array([np.nan, 1.0, np.nan, 3.0, np.nan])
    assert np.allclose(ffill(x)[1:], [1, 1, 3, 3]) and np.isnan(ffill(x)[0])
    y = np.arange(8, dtype=float)
    ls = lagged_sum(y, 3)
    assert np.all(np.isnan(ls[:3]))
    assert np.allclose(ls[3:], [0 + 1 + 2, 1 + 2 + 3, 2 + 3 + 4, 3 + 4 + 5, 4 + 5 + 6])
    y2 = y.copy()
    y2[4] = np.nan
    ls2 = lagged_sum(y2, 3)
    assert np.isnan(ls2[5]) and np.isnan(ls2[7]) and np.isclose(ls2[4], 6.0)
    dl = diff_lag(y ** 2, 2)
    assert np.all(np.isnan(dl[:2])) and np.isclose(dl[5], 25 - 9)


def test_market_return_members_only():
    r = np.array([[0.01, 0.03, 0.5], [np.nan, 0.02, 0.0], [0.0, 0.0, 0.0]])
    mem = np.array([[True, True, False], [True, True, False], [False, False, False]])
    M = market_return(r, mem)
    assert np.isclose(M[0], 0.02) and np.isclose(M[1], 0.02) and np.isnan(M[2])


def _ref_z(S, Y, M, t):
    """Independent reference: explicit window loops."""
    rows = []
    for k in range(50):
        e = t - 5 * k                  # window of row e covers sessions e-5 .. e-1
        if e - 10 < 0:
            rows.append(None)
            continue
        dS = S[e] - S[e - 5]
        dY = Y[e] - Y[e - 5]
        rM = sum(M[e - 5:e])
        rML = sum(M[e - 10:e - 5])
        rows.append((dS, rM, rML, dY))
    good = [x for x in rows if x is not None]
    A = np.array([[1.0, g[1], g[2], g[3]] for g in good])
    y = np.array([g[0] for g in good])
    b = np.linalg.lstsq(A, y, rcond=None)[0]
    e = y - A @ b
    return e[0] / np.sqrt(np.sum(e ** 2) / (len(y) - 4)), len(y)


def test_shock_z_matches_reference_and_is_point_in_time():
    rng = np.random.default_rng(1)
    T = 400
    S = 2.0 + np.cumsum(rng.normal(0, 0.02, T))
    Y = 4.0 + np.cumsum(rng.normal(0, 0.03, T))
    M = rng.normal(0, 0.01, T)
    S = S + 0.5 * np.cumsum(np.concatenate([[0.0], M[:-1]]))   # spread reacts to the market
    t = 380
    z = shock_z(S, Y, M, np.array([t]))
    ref, n = _ref_z(S, Y, M, t)
    assert n == 50
    assert np.isclose(z[t], ref)
    # row t's own market return and everything after row t are not used; S/Y rows after t not used
    M2, S2, Y2 = M.copy(), S.copy(), Y.copy()
    M2[t:] = 0.3
    S2[t + 1:] = 99.0
    Y2[t + 1:] = -5.0
    assert np.isclose(shock_z(S2, Y2, M2, np.array([t]))[t], z[t])


def test_shock_z_sign_and_min_windows():
    rng = np.random.default_rng(2)
    T = 400
    S = 2.0 + np.cumsum(rng.normal(0, 0.02, T))
    Y = 4.0 + np.cumsum(rng.normal(0, 0.03, T))
    M = rng.normal(0, 0.01, T)
    t = 380
    up = S.copy()
    up[t:] += 1.0
    dn = S.copy()
    dn[t:] -= 1.0
    assert shock_z(up, Y, M, np.array([t]))[t] > 3
    assert shock_z(dn, Y, M, np.array([t]))[t] < -3
    # fewer than 40 valid windows -> NaN: row 199 has only 38 windows with a complete rM_lag
    assert np.isnan(shock_z(S, Y, M, np.array([199]))[199])
    assert np.isfinite(shock_z(S, Y, M, np.array([205]))[205])
    short = np.array([t])
    M3 = M.copy()
    M3[: t - 5 * 39] = np.nan          # leaves 38 windows with complete rM and rM_lag
    assert np.isnan(shock_z(S, Y, M3, short)[t])
    S4 = S.copy()
    S4[t - 3:] = np.nan                # current window: carried forward value is used, not NaN
    assert np.isfinite(shock_z(S4, Y, M, short)[t])


def test_betas():
    rng = np.random.default_rng(3)
    T, N = 300, 3
    M = rng.normal(0, 0.01, T)
    r = np.column_stack([1.5 * M, -0.5 * M + 1e-6 * rng.normal(size=T), M])
    valid = np.ones((T, N), dtype=bool)
    valid[:120, 2] = False             # only 180 valid days in the last 252 rows for column 2
    b = betas(r, M, valid, np.array([T - 1, 150]))
    assert np.isclose(b[0, 0], 1.5) and np.isclose(b[0, 1], -0.5, atol=1e-3)
    assert np.isnan(b[0, 2])
    assert np.all(np.isnan(b[1]))      # row 150: only 151 days of history
    b2 = betas(r, M, valid, np.array([251]))
    assert np.isclose(b2[0, 0], 1.5)


def test_leverage_and_eligibility():
    L = leverage(np.array([50.0, 0.0, 10.0, 10.0, np.nan, 10.0]),
                 np.array([10.0, 20.0, 0.0, np.nan, 0.0, 0.0]),
                 np.array([150.0, 80.0, 0.0, 10.0, 10.0, -5.0]))
    assert np.isclose(L[0], 40 / 200) and np.isclose(L[1], -20 / 80)
    assert np.all(np.isnan(L[2:]))


def test_filing_age():
    dates = bdays(1000)
    T = dates.size
    F = np.full((T, 3), np.nan)
    F[:, 0] = 1.0                      # filed on row 0, never again -> stale after 400 calendar days
    F[:, 1] = 1.0 + (np.arange(T) // 63)
    F[:, 2] = np.nan
    rows = np.array([10, T - 1])
    ok = filing_age_ok([F, F], dates, rows)
    assert ok[0, 0] and not ok[1, 0]
    assert ok[0, 1] and ok[1, 1]
    assert not ok[0, 2] and not ok[1, 2]


def test_legs_terciles():
    beta = np.arange(9, dtype=float)               # terciles: cols 0-2, 3-5, 6-8
    L = np.array([0.3, 0.1, 0.2, 0.9, 0.5, 0.7, 0.0, 0.4, -0.2])
    elig = np.ones(9, dtype=bool)
    lg = legs(beta, L, elig, 0.34)                 # k = 1 per tercile
    assert list(np.flatnonzero(lg == 1)) == [0, 3, 7]
    assert list(np.flatnonzero(lg == -1)) == [1, 4, 8]
    elig2 = elig.copy()
    elig2[0] = False                               # 8 eligible -> terciles of 3,3,2: cols 1-3, 4-6, 7-8
    lg2 = legs(beta, L, elig2, 0.34)
    assert list(np.flatnonzero(lg2 == 1)) == [3, 5]   # tercile 7-8: k = floor(0.68) = 0
    assert list(np.flatnonzero(lg2 == -1)) == [1, 6]
    assert np.all(legs(beta, L, np.zeros(9, dtype=bool), 0.2) == 0)


def test_book_row_sides():
    leg = np.array([1, 1, -1, -1, 0, 0], dtype=np.int8)
    w = book_row(leg, 1.2, 1.0)                    # widening: long LOW, short HIGH
    assert np.allclose(w, [-0.25, -0.25, 0.25, 0.25, 0, 0])
    w = book_row(leg, -1.0, 1.0)                   # tightening (|z| == z_news counts)
    assert np.allclose(w, [0.25, 0.25, -0.25, -0.25, 0, 0])
    assert np.all(book_row(leg, 0.9, 1.0) == 0)
    assert np.all(book_row(leg, np.nan, 1.0) == 0)


# ----------------------------------------------------------------------------------------------------- full view
N = 30
SHOCK_WEEK = 110       # index of the Friday whose window carries the spread jump


def make_view(jump=1.0):
    rng = np.random.default_rng(7)
    dates = bdays(1100)
    T = dates.size
    fri = np.flatnonzero(weekday(dates) == 4)
    t_shock = fri[SHOCK_WEEK]
    base = rng.normal(0.0003, 0.01, T)
    bet = 0.5 + 0.05 * np.arange(N)
    r = base[:, None] * bet[None, :] + 1e-5 * rng.normal(size=(T, N))
    ret_co = r * 0.3
    ret_oc = (1 + r) / (1 + ret_co) - 1
    listed = np.ones((T, N), dtype=bool)
    listed[:t_shock - 150, 28] = False             # late listing: 150 days of history at the shock
    ret_co[~listed] = 0.0
    ret_oc[~listed] = 0.0
    tradable = listed.copy()
    close = np.where(listed, 50.0, np.nan)
    uni = listed.copy()
    uni[:, 29] = False                             # never a member
    rows = np.arange(T)[:, None]
    mcap = np.tile(100.0 * (1 + 0.001 * (rows // 63)), (1, N))
    mcap[:, 27] = 100.0                            # frozen filing: stale
    debt = np.tile(((np.arange(N) * 7) % 26 + 1.0)[None, :], (T, 1))
    debt[:, 27] = 3.0
    cash = np.zeros((T, N))
    cash[:, 26] = np.nan                           # missing cashneq
    baa = 7.0 + np.cumsum(rng.normal(0, 0.02, T))
    dgs = 4.5 + np.cumsum(rng.normal(0, 0.03, T))
    baa[t_shock:] += jump                          # row t holds the value stamped t-1
    view = DataView(
        dates=dates, instruments=tuple(f"E{j}" for j in range(N)), asset_class=tuple("us_equity" for _ in range(N)),
        ret_co=ret_co, ret_oc=ret_oc, tradable=tradable, listed=listed, delisting=np.zeros((T, N), dtype=bool),
        close=close, dollar_volume=np.where(listed, 1e8, np.nan), universe=uni,
        extras={"sf1_arq_debt": debt.astype(np.float32), "sf1_arq_cashneq": cash.astype(np.float32),
                "sf1_arq_marketcap": mcap.astype(np.float32)},
        series={"fred_moody_yields_1d.DBAA": baa, "fred_macro.DGS10": dgs}, cash_ret=np.zeros(T))
    return view, t_shock, debt[0]


def truncate(v, n):
    return DataView(
        dates=v.dates[:n], instruments=v.instruments, asset_class=v.asset_class, ret_co=v.ret_co[:n],
        ret_oc=v.ret_oc[:n], tradable=v.tradable[:n], listed=v.listed[:n], delisting=v.delisting[:n],
        close=v.close[:n], dollar_volume=v.dollar_volume[:n], universe=v.universe[:n],
        extras={k: x[:n] for k, x in v.extras.items()}, series={k: x[:n] for k, x in v.series.items()},
        cash_ret=v.cash_ret[:n])


def expected_legs(debt, k):
    terc = [list(range(0, 9)), list(range(9, 18)), list(range(18, 26))]   # 26 eligible, beta rises with column
    hi, lo = [], []
    for g in terc:
        s = sorted(g, key=lambda j: debt[j])
        lo += s[:k]
        hi += s[-k:]
    return sorted(hi), sorted(lo)


def test_full_widening_book():
    v, t, debt = make_view(1.0)
    W = target_weights(v, PARAMS)
    assert W.shape == (v.dates.size, N)
    dec = decision_mask(v.dates)
    assert np.all(np.isnan(W[~dec]))
    assert np.all(np.isfinite(W[dec]))
    assert np.all(np.abs(W[dec]).sum(axis=1) <= 1 + 1e-9)
    assert np.all(W[dec][:, 29] == 0)
    hi, lo = expected_legs(debt, 1)                # q = 0.2: floor(1.8), floor(1.8), floor(1.6) = 1 each
    assert sorted(np.flatnonzero(W[t] > 0)) == lo
    assert sorted(np.flatnonzero(W[t] < 0)) == hi
    assert np.allclose(W[t][W[t] > 0], 0.5 / 3) and np.allclose(W[t][W[t] < 0], -0.5 / 3)
    assert np.all(W[t, 26:] == 0)                  # missing cash, stale filing, short history, non-member
    # warm-up: 40 windows with rM_lag need row >= 5 * 39 + 10 = 205; the 39th Friday is at row <= 199
    assert np.all(W[dec][:39] == 0)


def test_full_tightening_and_params():
    v, t, debt = make_view(-1.0)
    W = target_weights(v, PARAMS)
    hi, lo = expected_legs(debt, 1)
    assert sorted(np.flatnonzero(W[t] > 0)) == hi
    assert sorted(np.flatnonzero(W[t] < 0)) == lo
    W25 = target_weights(v, dict(PARAMS, leg_quantile=0.25))   # floor(2.25), floor(2.25), floor(2.0) = 2 each
    hi2, lo2 = expected_legs(debt, 2)
    assert sorted(np.flatnonzero(W25[t] > 0)) == hi2
    assert np.allclose(W25[t][W25[t] > 0], 0.5 / 6)
    Wz = target_weights(v, dict(PARAMS, z_news=1000.0))
    dec = decision_mask(v.dates)
    assert np.all(Wz[dec] == 0)
    Wl = target_weights(v, dict(PARAMS, z_news=0.75))
    assert (np.abs(Wl[dec]).sum(axis=1) > 0).sum() >= (np.abs(W[dec]).sum(axis=1) > 0).sum()


def test_truncation_point_in_time():
    v, t, _ = make_view(1.0)
    W = target_weights(v, PARAMS)
    for n in (t + 1, t + 3, t - 40):
        Wt = target_weights(truncate(v, n), PARAMS)
        assert np.array_equal(np.isnan(Wt), np.isnan(W[:n]))
        assert np.allclose(np.nan_to_num(Wt), np.nan_to_num(W[:n]))


def test_missing_series_raises():
    v, _, _ = make_view(1.0)
    v.series.pop("fred_macro.DGS10")
    with pytest.raises(KeyError):
        target_weights(v, PARAMS)
