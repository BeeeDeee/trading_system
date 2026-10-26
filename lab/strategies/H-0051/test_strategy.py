import math

import numpy as np
import pytest

from lab.framework.data import DataView, synthetic
from strategy import target_weights, PARAMS

MARKET = {
    "GLD": "088691", "SLV": "084691", "USO": "067651", "UNG": "023651", "UGA": "111659",
    "CORN": "002602", "WEAT": "001602", "SOYB": "005602", "CPER": "085692", "PPLT": "076651",
    "PALL": "075651", "CANE": "080732", "FXE": "099741", "FXY": "097741", "FXB": "096742",
    "FXA": "232741", "FXC": "090741", "FXF": "092741",
}
NAMES = tuple(MARKET.keys())
DS = "cftc_cot_legacy_1d."


def _ffill(x):
    out = x.copy()
    for t in range(1, out.shape[0]):
        if not np.isfinite(out[t]):
            out[t] = out[t - 1]
    return out


def ref_z(asof, nl, ns, oi, W):
    """Slow reference of card rules (1)-(2): z of the latest report at each row (loop, by the card's text)."""
    T = len(asof)
    reps = []          # (row, asof, net, oi)
    z_row = np.full(T, np.nan)
    rep_row = np.full(T, -1)
    zs = []
    flows = []
    last = np.nan
    for t in range(T):
        if np.isfinite(asof[t]) and asof[t] != last:
            last = asof[t]
            reps.append((t, asof[t], nl[t] - ns[t], oi[t]))
            if len(reps) > 1 and reps[-1][1] - reps[-2][1] == 7 and oi[t] > 0:
                f = (reps[-1][2] - reps[-2][2]) / oi[t]
            else:
                f = np.nan
            prev = [x for x in flows[-W:] if np.isfinite(x)]
            if len(prev) >= W / 2 and len(prev) >= 2 and np.isfinite(f):
                zs.append(f / np.std(prev, ddof=1))
            else:
                zs.append(np.nan)
            flows.append(f)
        if reps:
            z_row[t] = zs[-1]
            rep_row[t] = reps[-1][0]
    return z_row, rep_row


def make_view(T=600, seed=0, stop_reports_after=None, low_dv=(), late=None, ffill=True, shift_asof=None):
    rng = np.random.default_rng(seed)
    base = synthetic()
    dates = base.dates[:T]
    N = len(NAMES)
    day = dates.astype(np.int64)
    wd = (day + 3) % 7                     # 0 = Monday (1970-01-01 was a Thursday)
    vols = np.linspace(0.005, 0.03, N)
    ret_co = rng.normal(0, 1, (T, N)) * vols * 0.3
    ret_oc = rng.normal(0, 1, (T, N)) * vols
    listed = np.ones((T, N), dtype=bool)
    if late is not None:
        name, row = late
        listed[:row, NAMES.index(name)] = False
    ret_co[~listed] = 0.0
    ret_oc[~listed] = 0.0
    close = np.where(listed, 50.0, np.nan)
    dv = np.where(listed, 2e7, np.nan)
    for name in low_dv:
        dv[:, NAMES.index(name)] = 1e6
    series = {}
    rep = wd == 4                          # Friday release
    if stop_reports_after is not None:
        rep &= np.arange(T) <= stop_reports_after
    for name, code in MARKET.items():
        a = np.where(rep, day - 3.0, np.nan)   # Tuesday as-of
        if shift_asof is not None and shift_asof[0] == name:
            a[shift_asof[1]] -= 1.0              # a Monday as-of: gaps of 6 and 8 days
        nl = np.where(rep, 1e5 + np.cumsum(rng.normal(0, 3e3, T)), np.nan)
        ns = np.where(rep, 8e4 + np.cumsum(rng.normal(0, 3e3, T)), np.nan)
        oi = np.where(rep, 5e5, np.nan)
        if ffill:
            a, nl, ns, oi = _ffill(a), _ffill(nl), _ffill(ns), _ffill(oi)
        series[DS + code + ".asof_date"] = a
        series[DS + code + ".nc_long"] = nl
        series[DS + code + ".nc_short"] = ns
        series[DS + code + ".open_interest"] = oi
    return DataView(
        dates=dates,
        instruments=NAMES,
        asset_class=tuple("us_etf" for _ in NAMES),
        ret_co=ret_co,
        ret_oc=ret_oc,
        tradable=listed.copy(),
        listed=listed,
        delisting=np.zeros((T, N), dtype=bool),
        close=close,
        dollar_volume=dv,
        universe=None,
        extras={},
        series=series,
        cash_ret=np.zeros(T),
    ), rep


def zmatrix(dv, W):
    T, N = len(dv.dates), len(dv.instruments)
    Z = np.full((T, N), np.nan)
    for name, code in MARKET.items():
        s = [dv.series[DS + code + k] for k in (".asof_date", ".nc_long", ".nc_short", ".open_interest")]
        Z[:, dv.col(name)] = ref_z(*s, W)[0]
    return Z


def decision_rows(w):
    return np.flatnonzero(np.isfinite(w).all(axis=1))


def test_reference_z_by_hand():
    # net = 0,1,3,6,10,15,21 with OI 10 -> flows .1 .. .6 ; W = 4 needs >= 2 earlier flows
    a = np.array([0, 7, 14, 21, 28, 35, 42], float)
    nl = np.array([0, 1, 3, 6, 10, 15, 21], float) + 100
    z, _ = ref_z(a, nl, np.full(7, 100.0), np.full(7, 10.0), 4)
    assert np.isnan(z[:3]).all()
    assert z[3] == pytest.approx(0.3 / np.std([.1, .2], ddof=1))
    assert z[6] == pytest.approx(0.6 / np.std([.2, .3, .4, .5], ddof=1))


def test_weights_legs_and_gross():
    dv, rep = make_view()
    w = target_weights(dv, PARAMS)
    dec = decision_rows(w)
    assert set(dec) <= set(np.flatnonzero(rep))
    nonflat = [t for t in dec if np.abs(w[t]).sum() > 0]
    assert len(nonflat) > 10
    Z = zmatrix(dv, PARAMS["z_window_weeks"])
    r = (1 + dv.ret_co) * (1 + dv.ret_oc) - 1
    k = int(math.floor(0.2 * 18 + 0.5))   # 4
    for t in nonflat:
        assert t >= 251
        row = w[t]
        assert np.abs(row).sum() == pytest.approx(1.0)
        assert row[row > 0].sum() == pytest.approx(0.5)
        assert row[row < 0].sum() == pytest.approx(-0.5)
        assert (row > 0).sum() == k and (row < 0).sum() == k
        order = np.argsort(Z[t])
        assert set(np.flatnonzero(row > 0)) == set(order[:k])
        assert set(np.flatnonzero(row < 0)) == set(order[-k:])
        for leg in (np.flatnonzero(row > 0), np.flatnonzero(row < 0)):
            vol = np.std(r[t - 62:t + 1][:, leg], axis=0, ddof=1)
            prod = row[leg] * vol
            assert prod == pytest.approx(np.full(k, prod[0]))
    other = np.setdiff1d(np.arange(len(dv.dates)), dec)
    assert np.isnan(w[other]).all()        # positions kept between rebalances


def test_unfilled_series_same_result():
    a, _ = make_view(seed=3, ffill=True)
    b, _ = make_view(seed=3, ffill=False)
    np.testing.assert_array_equal(target_weights(a, PARAMS), target_weights(b, PARAMS))


def test_non_7_day_gap_makes_flow_missing():
    dv0, rep = make_view(T=600, seed=7)
    reprows = np.flatnonzero(rep)
    r = int(reprows[reprows > 400][0])
    dv, _ = make_view(T=600, seed=7, shift_asof=("GLD", r))
    Z = zmatrix(dv, PARAMS["z_window_weeks"])
    j = dv.col("GLD")
    nxt = int(reprows[reprows > r][0])
    assert np.isnan(Z[r, j]) and np.isnan(Z[nxt, j])
    w = target_weights(dv, PARAMS)
    assert w[r, j] == 0 and w[nxt, j] == 0
    assert np.abs(w[r]).sum() == pytest.approx(1.0)          # 17 eligible still trade


def test_fewer_than_six_eligible_is_flat():
    dv, rep = make_view(low_dv=NAMES[:13])
    w = target_weights(dv, PARAMS)
    dec = decision_rows(w)
    assert len(dec) > 0
    assert (w[dec] == 0).all()
    dv2, _ = make_view(low_dv=NAMES[:12])           # 6 eligible -> k = round(1.2) = 1
    w2 = target_weights(dv2, PARAMS)
    t = [t for t in decision_rows(w2) if np.abs(w2[t]).sum() > 0][0]
    assert (w2[t] > 0).sum() == 1 and (w2[t] < 0).sum() == 1
    assert (w2[t, [dv2.col(n) for n in NAMES[:12]]] == 0).all()


def test_stale_reports_close_book():
    dv, rep = make_view(T=600, stop_reports_after=450)
    w = target_weights(dv, PARAMS)
    last = np.flatnonzero(rep)[-1]
    assert np.abs(w[last]).sum() == pytest.approx(1.0)
    day = dv.dates.astype(np.int64)
    after = np.arange(last + 1, 600)
    stale = after[day[after] - day[last] > 10]
    keep = after[day[after] - day[last] <= 10]
    assert len(stale) > 0 and len(keep) > 0
    assert np.isnan(w[keep]).all()
    assert (w[stale] == 0).all()


def test_late_listing_needs_252_sessions():
    dv, rep = make_view(T=600, late=("FXF", 200))
    w = target_weights(dv, PARAMS)
    j = dv.col("FXF")
    dec = decision_rows(w)
    assert all(w[t, j] == 0 for t in dec if t < 200 + 251)
    assert any(w[t, j] != 0 for t in dec if t >= 200 + 251)


def test_params_change_output():
    dv, _ = make_view(T=900, seed=5)
    base = target_weights(dv, PARAMS)
    w_lf = target_weights(dv, dict(PARAMS, leg_fraction=0.3))
    w_zw = target_weights(dv, dict(PARAMS, z_window_weeks=26))
    assert not np.array_equal(np.nan_to_num(base, nan=9), np.nan_to_num(w_lf, nan=9))
    assert not np.array_equal(np.nan_to_num(base, nan=9), np.nan_to_num(w_zw, nan=9))
    t = [t for t in decision_rows(w_lf) if np.abs(w_lf[t]).sum() > 0][0]
    assert (w_lf[t] > 0).sum() == 5          # round(0.3 * 18) = 5
    # shorter z window -> earliest z (13 earlier flows) arrives before the 252-session history; same first trade row
    Z26 = zmatrix(dv, 26)
    t26 = [t for t in decision_rows(w_zw) if np.abs(w_zw[t]).sum() > 0][0]
    order = np.argsort(Z26[t26])
    assert set(np.flatnonzero(w_zw[t26] > 0)) == set(order[:4])


def test_truncation_point_in_time():
    dv, _ = make_view(T=700, seed=4)
    full = target_weights(dv, PARAMS)
    cut = 560
    sub = DataView(
        dates=dv.dates[:cut], instruments=dv.instruments, asset_class=dv.asset_class,
        ret_co=dv.ret_co[:cut], ret_oc=dv.ret_oc[:cut], tradable=dv.tradable[:cut], listed=dv.listed[:cut],
        delisting=dv.delisting[:cut], close=dv.close[:cut], dollar_volume=dv.dollar_volume[:cut],
        universe=None, extras={}, series={k: v[:cut] for k, v in dv.series.items()}, cash_ret=dv.cash_ret[:cut],
    )
    np.testing.assert_array_equal(np.nan_to_num(full[:cut], nan=9),
                                  np.nan_to_num(target_weights(sub, PARAMS), nan=9))
