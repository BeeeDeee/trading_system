import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS
from strategy import _week_end_rows, _legs

SMALL = {"corr_window": 60, "formation_days": 5, "peers_k": 5}


def weekday_dates(T, start_monday=4 + 7 * 1500):
    d = np.arange(start_monday, start_monday + 2 * T)
    d = d[(d + 3) % 7 < 5][:T]
    return d.astype("datetime64[D]")


def make_view(T=300, N=150, seed=0, groups=15, missing=0.02, universe=None, dates=None):
    rng = np.random.default_rng(seed)
    g = np.arange(N) % groups
    fac = rng.normal(0, 0.01, (T, groups))
    mkt = rng.normal(0, 0.01, (T, 1))
    r = mkt + fac[:, g] + rng.normal(0, 0.01, (T, N))
    ret_co = r * 0.3
    ret_oc = (1 + r) / (1 + ret_co) - 1
    listed = np.ones((T, N), dtype=bool)
    close = 50 * np.cumprod(1 + r, axis=0)
    miss = rng.random((T, N)) < missing
    close[miss] = np.nan
    ret_co[miss] = 0.0
    ret_oc[miss] = 0.0
    tradable = ~miss
    if universe is None:
        universe = np.ones((T, N), dtype=bool)
    return DataView(
        dates=weekday_dates(T) if dates is None else dates,
        instruments=tuple(f"E{100 + j}" for j in range(N)),
        asset_class=tuple("us_equity" for _ in range(N)),
        ret_co=ret_co, ret_oc=ret_oc, tradable=tradable, listed=listed,
        delisting=np.zeros((T, N), dtype=bool), close=close,
        dollar_volume=np.full((T, N), 1e8), universe=universe,
        extras={}, series={}, cash_ret=np.zeros(T),
    )


def sub(view, rows):
    def cut(x):
        return x[rows] if isinstance(x, np.ndarray) else x
    return DataView(
        dates=view.dates[rows], instruments=view.instruments, asset_class=view.asset_class,
        ret_co=cut(view.ret_co), ret_oc=cut(view.ret_oc), tradable=cut(view.tradable),
        listed=cut(view.listed), delisting=cut(view.delisting), close=cut(view.close),
        dollar_volume=cut(view.dollar_volume), universe=cut(view.universe),
        extras={}, series={}, cash_ret=view.cash_ret[rows],
    )


def naive_legs(view, params, t):
    """Independent loop implementation of the card at decision row t."""
    W, F, K = params["corr_window"], params["formation_days"], params["peers_k"]
    N = len(view.instruments)
    valid = view.listed & np.isfinite(view.close)
    r = (1 + view.ret_co) * (1 + view.ret_oc) - 1
    lo = max(0, t + 1 - W)
    E = [j for j in range(N)
         if view.universe[t, j] and view.tradable[t, j]
         and t + 1 >= F and valid[t + 1 - F:t + 1, j].all()
         and valid[lo:t + 1, j].sum() >= 0.75 * W]
    if len(E) < 50:
        return set(), set()
    win = r[lo:t + 1][:, E].copy()
    vm = valid[lo:t + 1][:, E]
    win[~vm] = np.nan
    x = win - np.nanmean(win, axis=1, keepdims=True)
    n = len(E)
    C = np.full((n, n), -np.inf)
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            both = vm[:, a] & vm[:, b]
            if both.sum() < 0.6 * W:
                continue
            C[a, b] = np.corrcoef(x[both, a], x[both, b])[0, 1]
    own = np.array([np.prod(1 + r[t + 1 - F:t + 1, j]) - 1 for j in E])
    peer = np.array([own[np.argsort(-C[a])[:K]].mean() for a in range(n)])
    pct = np.argsort(np.argsort(own)) / (n - 1)
    M = [a for a in range(n) if 0.2 <= pct[a] <= 0.8]
    if len(M) < 50:
        return set(), set()
    Ms = sorted(M, key=lambda a: peer[a])
    q = len(M) // 5
    return {E[a] for a in Ms[-q:]}, {E[a] for a in Ms[:q]}


def test_week_end_rows_are_fridays_and_holiday_week_is_skipped():
    d = weekday_dates(20)
    dec = _week_end_rows(d)
    wd = (d.astype(np.int64) + 3) % 7
    assert np.array_equal(dec, wd == 4)
    # drop the second Friday (holiday): that week has no decision row
    keep = np.ones(20, dtype=bool)
    keep[9] = False
    d2 = d[keep]
    dec2 = _week_end_rows(d2)
    assert dec2.sum() == dec.sum() - 1
    assert not dec2[5:9].any()      # Mon-Thu of the holiday week


def test_matches_naive_reference():
    v = make_view(T=200, N=150, seed=1)
    dec, lg, sh = _legs(v, SMALL)
    rows = np.flatnonzero(dec)[-3:]
    for t in rows:
        el, es = naive_legs(v, SMALL, t)
        assert len(el) > 0
        assert set(np.flatnonzero(lg[t])) == el
        assert set(np.flatnonzero(sh[t])) == es


def test_weights_shape_gross_net_and_nan_rows():
    v = make_view(T=300, seed=2)
    w = target_weights(v, SMALL)
    dec = _week_end_rows(v.dates)
    assert w.shape == (300, 150)
    assert np.isnan(w[~dec]).all()
    assert np.isfinite(w[dec]).all()
    traded = np.abs(w[dec]).sum(axis=1) > 0
    assert traded.sum() > 10
    wt = w[dec][traded]
    assert np.allclose(np.abs(wt).sum(axis=1), 1.0)
    assert np.allclose(wt.sum(axis=1), 0.0)
    assert np.allclose(wt[wt > 0].sum() / traded.sum(), 0.5)
    # quintiles of M(t): equal leg sizes, leg = floor(|M| / 5) >= 10
    n_l = (wt > 0).sum(axis=1)
    assert np.array_equal(n_l, (wt < 0).sum(axis=1))
    assert (n_l >= 10).all()
    # first decision rows before the corr window is 75 % full are flat
    first = np.flatnonzero(dec)
    early = first[first + 1 < 0.75 * SMALL["corr_window"]]
    assert len(early) > 0 and (w[early] == 0).all()


def test_peer_catch_up_logic():
    # stock 0's group mates (same group) rise strongly in the formation week while it stays flat
    v = make_view(T=200, seed=3, missing=0.0)
    dec = np.flatnonzero(_week_end_rows(v.dates))
    t = dec[-1]
    g = np.arange(150) % 15
    rc, ro = v.ret_co.copy(), v.ret_oc.copy()
    for j in range(150):
        rc[t - 4:t + 1, j] = 0.0
        ro[t - 4:t + 1, j] = 0.0005 * ((j * 37) % 150) / 150   # small spread of own returns
    mates = np.flatnonzero((g == 0) & (np.arange(150) != 0))
    losers = np.flatnonzero((g == 1) & (np.arange(150) != 1))
    ro[t - 4:t + 1, mates] = 0.02
    ro[t - 4:t + 1, losers] = -0.02
    ro[t - 4:t + 1, 0] = 0.0002
    ro[t - 4:t + 1, 1] = 0.0002
    v2 = DataView(dates=v.dates, instruments=v.instruments, asset_class=v.asset_class, ret_co=rc,
                  ret_oc=ro, tradable=v.tradable, listed=v.listed, delisting=v.delisting, close=v.close,
                  dollar_volume=v.dollar_volume, universe=v.universe, extras={}, series={},
                  cash_ret=v.cash_ret)
    w = target_weights(v2, SMALL)
    assert w[t, 0] > 0          # flat stock whose peers rose: long
    assert w[t, 1] < 0          # flat stock whose peers fell: short
    assert (w[t, mates] == 0).all() and (w[t, losers] == 0).all()   # big own movers excluded


def test_universe_and_tradable_and_late_listing():
    T, N = 300, 150
    uni = np.ones((T, N), dtype=bool)
    uni[:, :5] = False
    v = make_view(T=T, N=N, seed=4, universe=uni)
    v.listed[:250, 140] = False
    v.close[:250, 140] = np.nan
    v.tradable[:, 141] = False
    w = target_weights(v, SMALL)
    dec = _week_end_rows(v.dates)
    assert (w[dec][:, :5] == 0).all()
    assert (w[dec][:, 141] == 0).all()
    # listed at row 250: eligible only once 75 % of the 60-day window has returns (row >= 294)
    rows = np.flatnonzero(dec)
    assert (w[rows[rows < 294], 140] == 0).all()
    with pytest.raises(ValueError):
        target_weights(DataView(
            dates=v.dates, instruments=v.instruments, asset_class=v.asset_class, ret_co=v.ret_co,
            ret_oc=v.ret_oc, tradable=v.tradable, listed=v.listed, delisting=v.delisting,
            close=v.close, dollar_volume=v.dollar_volume, universe=None, extras={}, series={},
            cash_ret=v.cash_ret), SMALL)


def test_too_few_names_is_flat():
    v = make_view(T=200, N=80, seed=5)      # |M(t)| <= 49 < 50
    w = target_weights(v, SMALL)
    dec = _week_end_rows(v.dates)
    assert (w[dec] == 0).all()


def test_point_in_time_truncation_and_perturbation():
    v = make_view(T=260, seed=6)
    w = target_weights(v, SMALL)
    rng = np.random.default_rng(9)
    dec = np.flatnonzero(_week_end_rows(v.dates))
    for t in list(dec[-6:]) + [200, 233]:
        wt = target_weights(sub(v, slice(0, t + 1)), SMALL)
        assert np.array_equal(np.nan_to_num(wt, nan=9.0), np.nan_to_num(w[:t + 1], nan=9.0))
        rc = v.ret_co.copy()
        ro = v.ret_oc.copy()
        rc[t + 1:] = rng.normal(0, 0.05, rc[t + 1:].shape)
        ro[t + 1:] = rng.normal(0, 0.05, ro[t + 1:].shape)
        vp = DataView(dates=v.dates, instruments=v.instruments, asset_class=v.asset_class, ret_co=rc,
                      ret_oc=ro, tradable=v.tradable, listed=v.listed, delisting=v.delisting,
                      close=v.close, dollar_volume=v.dollar_volume, universe=v.universe, extras={},
                      series={}, cash_ret=v.cash_ret)
        wp = target_weights(vp, SMALL)
        assert np.array_equal(np.nan_to_num(wp[:t + 1], nan=9.0), np.nan_to_num(w[:t + 1], nan=9.0))


def test_each_param_changes_output():
    v = make_view(T=300, seed=7)
    base = np.nan_to_num(target_weights(v, SMALL), nan=9.0)
    for name, val in [("corr_window", 40), ("formation_days", 3), ("peers_k", 10)]:
        p = dict(SMALL)
        p[name] = val
        assert not np.array_equal(base, np.nan_to_num(target_weights(v, p), nan=9.0)), name


def test_deterministic_and_primary_params():
    assert PARAMS == {"corr_window": 252, "formation_days": 5, "peers_k": 10}
    v = make_view(T=320, seed=8)
    a = target_weights(v, PARAMS)
    b = target_weights(v, PARAMS)
    assert np.array_equal(np.nan_to_num(a, nan=9.0), np.nan_to_num(b, nan=9.0))
