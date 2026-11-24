import numpy as np
import pytest

from lab.framework.data import DataView
import strategy as S
from strategy import target_weights, PARAMS

N = 30
T = 130
T0 = 100          # event row
EV = 0            # event stock


def make_view(T=T, N=N, mutate=None, drop_key=None):
    # deterministic +-1 % pattern: half the stocks up, half down each day -> member mean 0, |z| ~ 1
    sign = np.where((np.arange(T)[:, None] + np.arange(N)[None, :]) % 2 == 0, 1.0, -1.0)
    ret_oc = 0.01 * sign
    ret_co = np.zeros((T, N))
    ret_oc[T0, EV] = -0.10                               # big idiosyncratic drop
    dv = np.tile(np.arange(N, dtype=float)[None, :] * 1e6 + 1e6, (T, 1))   # higher column -> more liquid
    ea = np.zeros((T, N))
    ea[T0 - 5, :] = 1.0                                   # release 5 sessions before: window open
    nk = np.zeros((T, N))
    ncf = np.full((T, N), 1e6)                            # issuers by default
    mcap = np.full((T, N), 1e9)
    ncf[:, EV] = -5e7                                     # stock 0 the only repurchaser
    uni = np.ones((T, N), dtype=bool)
    d = dict(ret_oc=ret_oc, ret_co=ret_co, dv=dv, ea=ea, nk=nk, ncf=ncf, mcap=mcap, uni=uni)
    d["close"] = 100.0 * np.cumprod(1.0 + ret_oc, axis=0)
    if mutate is not None:
        mutate(d)
    extras = {
        "ea_reaction_day": d["ea"].astype(np.float32),
        "nonearn_8k_reaction_day": d["nk"].astype(np.float32),
        "sf1_art_ncfcommon": d["ncf"].astype(np.float32),
        "sf1_art_marketcap": d["mcap"].astype(np.float32),
    }
    if drop_key is not None:
        del extras[drop_key]
    names = tuple("S%02d" % i for i in range(N))
    return DataView(
        dates=np.arange(12000, 12000 + T).astype("datetime64[D]"),
        instruments=names,
        asset_class=tuple("us_equity" for _ in names),
        ret_co=d["ret_co"].astype(np.float32),
        ret_oc=d["ret_oc"].astype(np.float32),
        tradable=np.ones((T, N), dtype=bool),
        listed=np.ones((T, N), dtype=bool),
        delisting=np.zeros((T, N), dtype=bool),
        close=d["close"].astype(np.float32),
        dollar_volume=d["dv"].astype(np.float32),
        universe=d["uni"],
        extras=extras,
        series={},
        cash_ret=np.zeros(T),
    )


def truncate(v, cut):
    k = cut + 1
    return DataView(
        dates=v.dates[:k], instruments=v.instruments, asset_class=v.asset_class,
        ret_co=v.ret_co[:k], ret_oc=v.ret_oc[:k], tradable=v.tradable[:k], listed=v.listed[:k],
        delisting=v.delisting[:k], close=v.close[:k], dollar_volume=v.dollar_volume[:k],
        universe=v.universe[:k], extras={n: x[:k] for n, x in v.extras.items()}, series={},
        cash_ret=v.cash_ret[:k],
    )


def events_of(view, params=PARAMS):
    return S.event_table(view, params)[0]


def test_basic_event_and_slot():
    v = make_view()
    ev = events_of(v)
    assert ev[T0, EV]
    assert ev.sum() == 1
    w = target_weights(v, PARAMS)
    assert w.shape == (T, N)
    for t in range(T0, T0 + 5):                         # long 0.06 on decision rows T0..T0+4
        assert w[t, EV] == pytest.approx(0.06)
        # hedge: the 20 most liquid members (columns 10..29), beta 1 (history < 200), 0.06/20 each
        assert np.allclose(w[t, 10:30], -0.06 / 20)
        assert np.all(w[t, 1:10] == 0.0)
    assert np.all(w[:T0] == 0.0)
    assert np.all(w[T0 + 5:] == 0.0)                    # closed at the open of T0+6


def test_news_blocks_event():
    def m(d):
        d["nk"][T0 - 1, EV] = 1.0
    assert not events_of(make_view(mutate=m))[T0, EV]

    def m1(d):
        d["nk"][T0, EV] = 1.0
    assert not events_of(make_view(mutate=m1))[T0, EV]

    def m2(d):
        d["ea"][T0, EV] = np.nan
    assert not events_of(make_view(mutate=m2))[T0, EV]

    def m3(d):
        d["nk"][T0 - 1, EV] = np.nan
    assert not events_of(make_view(mutate=m3))[T0, EV]


def test_window_bounds_and_param():
    def k2(d):
        d["ea"][:, EV] = 0.0
        d["ea"][T0 - 2, EV] = 1.0                       # k = 2: window not yet open
    assert not events_of(make_view(mutate=k2))[T0, EV]

    def k3(d):
        d["ea"][:, EV] = 0.0
        d["ea"][T0 - 3, EV] = 1.0                       # k = 3: open
    assert events_of(make_view(mutate=k3))[T0, EV]

    def k40(d):
        d["ea"][:, EV] = 0.0
        d["ea"][T0 - 40, EV] = 1.0                      # k = 40
    v = make_view(mutate=k40)
    assert not events_of(v, {"window_close": 35, "z_min": 2.0})[T0, EV]
    assert events_of(v, {"window_close": 45, "z_min": 2.0})[T0, EV]
    w35 = target_weights(v, {"window_close": 35, "z_min": 2.0})
    w45 = target_weights(v, {"window_close": 45, "z_min": 2.0})
    assert w35[T0, EV] == 0.0 and w45[T0, EV] == pytest.approx(0.06)

    def none(d):
        d["ea"][:, EV] = 0.0                            # no release at all
    assert not events_of(make_view(mutate=none), {"window_close": 45, "z_min": 2.0})[T0, EV]


def test_window_100_session_limit():
    ea = np.zeros((130, 1))
    ea[10, 0] = 1.0
    ok = S._open_window(ea, 200)[:, 0]
    assert ok[13] and ok[110] and not ok[111] and not ok[12]


def test_z_min_param():
    def m(d):
        d["ret_oc"][T0, EV] = 0.0
    base = make_view(mutate=m)
    r = S._close_to_close(base)
    hist = r[T0 - 63:T0]
    e_hist = hist - np.nanmean(hist, axis=1, keepdims=True)
    s = np.std(e_hist[:, EV], ddof=1)
    target_e = -2.25 * s
    others = (np.nansum(r[T0]) - r[T0, EV]) / (N - 1)
    x = target_e * N / (N - 1) + others                 # e = x - mean(r incl. x)

    def m2(d):
        d["ret_oc"][T0, EV] = x
    v = make_view(mutate=m2)
    _, z, _ = S.event_table(v, PARAMS)
    assert z[T0, EV] == pytest.approx(-2.25, abs=0.01)
    assert events_of(v, {"window_close": 35, "z_min": 2.0})[T0, EV]
    assert not events_of(v, {"window_close": 35, "z_min": 2.5})[T0, EV]
    w20 = target_weights(v, {"window_close": 35, "z_min": 2.0})
    w25 = target_weights(v, {"window_close": 35, "z_min": 2.5})
    assert w20[T0, EV] > 0 and w25[T0, EV] == 0.0


def test_resid_std_needs_50_values():
    def m(d):
        d["close"][: T0 - 49, EV] = np.nan              # only 48 valid returns in t-63..t-1
    v = make_view(mutate=m)
    _, z, _ = S.event_table(v, PARAMS)
    assert np.isnan(z[T0, EV])
    assert not events_of(v)[T0, EV]

    def m50(d):
        d["close"][: T0 - 51, EV] = np.nan              # exactly 50 valid
    _, z, _ = S.event_table(make_view(mutate=m50), PARAMS)
    assert np.isfinite(z[T0, EV])


def test_repurchaser_rules():
    def issuer(d):
        d["ncf"][:, EV] = 1e6
    assert not events_of(make_view(mutate=issuer))[T0, EV]

    def nan_b(d):
        d["ncf"][:, EV] = np.nan
    assert not events_of(make_view(mutate=nan_b))[T0, EV]

    uni = np.ones((1, 30), dtype=bool)
    mcap = np.full((1, 30), 1e9)
    ncf = np.full((1, 30), 1e6)
    ncf[0, :11] = -np.linspace(5e7, 1e7, 11)            # 11 positive yields, top third of 30 = 10 names
    top = S._repurchaser(ncf, mcap, uni)[0]
    assert top[:10].all() and not top[10:].any()
    uni2 = uni.copy()
    uni2[0, 20:] = False                                # 20 valid members -> top 7
    top = S._repurchaser(ncf, mcap, uni2)[0]
    assert top[:7].all() and not top[7:].any()
    ncf2 = np.full((1, 30), 1e6)
    ncf2[0, 0] = -1e6                                    # positive b required even inside the top third
    top = S._repurchaser(ncf2, mcap, uni)[0]
    assert top[0] and not top[1:].any()
    mc_nan = mcap.copy()
    mc_nan[0, 0] = np.nan
    assert not S._repurchaser(ncf2, mc_nan, uni)[0, 0]


def test_non_member_not_entered_and_exit_on_leaving():
    def notmem(d):
        d["uni"][T0, EV] = False
    v = make_view(mutate=notmem)
    assert not events_of(v)[T0, EV]
    assert np.all(target_weights(v, PARAMS) == 0.0)

    def leave(d):
        d["uni"][T0 + 2, EV] = False
    w = target_weights(make_view(mutate=leave), PARAMS)
    assert w[T0 + 1, EV] == pytest.approx(0.06)
    assert np.all(w[T0 + 2] == 0.0)                      # slot and its hedge closed
    assert np.all(w[T0 + 3:] == 0.0)

    def hedge_leaves(d):
        d["uni"][T0 + 1, 29] = False
    w = target_weights(make_view(mutate=hedge_leaves), PARAMS)
    assert w[T0 + 1, 29] == 0.0 and w[T0 + 2, 29] == pytest.approx(-0.003)
    assert w[T0 + 1, EV] == pytest.approx(0.06)


def test_slot_cap_takes_most_negative_z():
    drops = np.linspace(-0.08, -0.17, 10)                # stocks 0..9 all drop; 9 most negative

    def m(d):
        d["ret_oc"][T0, :10] = drops
        d["ncf"][:, :10] = -1e8                          # all ten repurchasers in the top third
    v = make_view(mutate=m)
    ev, z, _ = S.event_table(v, PARAMS)
    assert ev[T0, :10].all() and ev.sum() == 10
    w = target_weights(v, PARAMS)
    order = np.argsort(z[T0, :10])
    chosen = set(order[:8].tolist())
    assert chosen == set(range(2, 10))
    assert set(np.nonzero(w[T0] > 0)[0].tolist()) == chosen
    assert np.all(w[T0 + 1, :2] == 0.0)                  # dropped events are not queued
    assert np.abs(w[T0]).sum() == pytest.approx(0.96)    # 8 x 0.06 long + 8 x 0.06 short
    assert np.all(np.abs(w).sum(axis=1) <= 1.0 + 1e-12)


def test_gross_scaled_to_one(monkeypatch):
    drops = np.linspace(-0.08, -0.17, 10)

    def m(d):
        d["ret_oc"][T0, :10] = drops
        d["ncf"][:, :10] = -1e8
    v = make_view(mutate=m)
    monkeypatch.setattr(S, "SLOT_WEIGHT", 0.1)
    w = target_weights(v, PARAMS)
    g = np.abs(w[T0]).sum()
    assert g == pytest.approx(1.0)
    longs = w[T0][w[T0] > 0]
    assert np.allclose(longs, 0.1 / 1.6)                 # proportional scaling of 1.6 raw gross


def test_slot_freed_after_hold_and_new_entries():
    def m(d):
        d["ret_oc"][T0 + 5, EV] = -0.10                  # new event on the row the first slot closes
    v = make_view(mutate=m)
    ev = events_of(v)
    assert ev[T0, EV] and ev[T0 + 5, EV]
    w = target_weights(v, PARAMS)
    assert w[T0 + 5, EV] == pytest.approx(0.06) and w[T0 + 9, EV] == pytest.approx(0.06)
    assert w[T0 + 10, EV] == 0.0


def test_no_reentry_while_open():
    def m(d):
        d["ret_oc"][T0 + 2, EV] = -0.10                  # second drop while the slot is open
    v = make_view(mutate=m)
    ev = events_of(v)
    assert ev[T0, EV] and ev[T0 + 2, EV]
    w = target_weights(v, PARAMS)
    assert w[T0 + 4, EV] == pytest.approx(0.06)
    assert w[T0 + 5, EV] == 0.0                          # not extended by the second event


def test_hedge_beta_and_clip():
    Tl, Nl = 300, 30
    rng = np.random.default_rng(3)
    r = rng.normal(0, 0.01, size=(Tl, Nl))
    dv = np.tile(np.arange(Nl, dtype=float)[None, :] + 1.0, (Tl, 1))
    uni_t = np.ones(Nl, dtype=bool)
    mkt = r[:, 10:30].mean(axis=1)
    r[:, 0] = 1.2 * mkt + rng.normal(0, 0.001, Tl)
    basket, beta = S._hedge(280, 0, r, dv, uni_t)
    assert sorted(basket.tolist()) == list(range(10, 30))
    assert beta == pytest.approx(1.2, abs=0.05)
    r[:, 0] = 3.0 * mkt
    assert S._hedge(280, 0, r, dv, uni_t)[1] == pytest.approx(1.5)
    r[:, 0] = -1.0 * mkt
    assert S._hedge(280, 0, r, dv, uni_t)[1] == pytest.approx(0.5)
    r2 = r.copy()
    r2[:100, 0] = np.nan                                 # 252 window t-252..t-1 = rows 28..279 -> 180 valid
    assert S._hedge(280, 0, r2, dv, uni_t)[1] == pytest.approx(1.0)
    basket, _ = S._hedge(280, 29, r, dv, uni_t)          # event stock excluded from its own basket
    assert 29 not in basket.tolist() and 9 in basket.tolist()
    uni_t2 = uni_t.copy()
    uni_t2[25] = False                                   # non-members never in the basket
    basket, _ = S._hedge(280, 0, r, dv, uni_t2)
    assert 25 not in basket.tolist() and len(basket) == 20


def test_point_in_time_truncation():
    def m(d):
        d["ret_oc"][T0 + 3, 3] = -0.10
        d["ncf"][:, 3] = -6e7
    v = make_view(mutate=m)
    w_full = target_weights(v, PARAMS)
    for cut in (T0 - 1, T0, T0 + 2, T0 + 3, T0 + 7):
        w_cut = target_weights(truncate(v, cut), PARAMS)
        assert np.array_equal(w_cut, w_full[:cut + 1])


def test_short_view_runs():
    v = truncate(make_view(), 5)
    w = target_weights(v, PARAMS)
    assert w.shape == (6, N) and np.all(w == 0.0)


def test_missing_input_raises():
    with pytest.raises(KeyError):
        target_weights(make_view(drop_key="ea_reaction_day"), PARAMS)
    with pytest.raises(KeyError):
        target_weights(make_view(drop_key="sf1_art_marketcap"), PARAMS)
