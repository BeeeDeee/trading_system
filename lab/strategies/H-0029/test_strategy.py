"""Mechanics tests for H-0029 on hand-made markets.

Base market: T=30 business days starting on a Monday (Fridays = rows 4, 9, 14, 19, 24, 29),
N stocks E000..E(N-1), constant price, market cap 1e6 filed on row 0, dollar volume 1000*(j+1)
(so turnover rises with j), and on row 29 an open->close return of j*1e-5 (so R rises with j).
With N=130, F=3, q=0.2: eligible 130 -> high-TO group = top 26 = E104..E129 -> legs of 5:
long E125..E129 (+0.1 each), short E104..E108 (-0.1 each).
"""
import numpy as np
import pytest
from lab.framework.data import DataView
from strategy import target_weights, PARAMS, select_legs, decision_rows, _marketcap_start_rows

P3 = {"formation_days": 3, "quantile": 0.2}
LONG = [125, 126, 127, 128, 129]
SHORT = [104, 105, 106, 107, 108]


def biz_dates(T):
    monday = np.array([4]).astype("datetime64[D]")  # day 4 after the epoch is a Monday
    return np.busday_offset(monday, np.arange(T))


def make_arrays(T=30, N=130):
    a = {}
    a["dates"] = biz_dates(T)
    a["ret_co"] = np.zeros((T, N))
    a["ret_oc"] = np.zeros((T, N))
    a["ret_oc"][29] = np.arange(N) * 1e-5
    a["close"] = np.ones((T, N))
    a["dollar_volume"] = np.tile(1000.0 * (np.arange(N) + 1), (T, 1))
    a["mc"] = np.full((T, N), 1e6)
    a["universe"] = np.ones((T, N), dtype=bool)
    a["tradable"] = np.ones((T, N), dtype=bool)
    return a


def view(a):
    T, N = a["close"].shape
    listed = np.isfinite(a["close"])
    return DataView(dates=a["dates"], instruments=tuple("E%03d" % j for j in range(N)),
                    asset_class=("us_equity",) * N, ret_co=a["ret_co"], ret_oc=a["ret_oc"],
                    tradable=a["tradable"], listed=listed, delisting=np.zeros((T, N), dtype=bool),
                    close=a["close"], dollar_volume=a["dollar_volume"], universe=a["universe"],
                    extras={"sf1_arq_marketcap": a["mc"]}, series={}, cash_ret=np.zeros(T))


def expected_row(N, longs, shorts, wl=0.1, ws=0.1):
    e = np.zeros(N)
    e[longs] = wl
    e[shorts] = -ws
    return e


def test_params_match_card():
    assert PARAMS == {"formation_days": 21, "quantile": 0.2}


def test_decision_rows_are_fridays():
    d = biz_dates(30)
    assert list(decision_rows(d)) == [4, 9, 14, 19, 24, 29]


def test_marketcap_start_rows():
    mc = np.array([np.nan, np.nan, 5, 5, 5, 7, 7, np.nan, 7])[:, None]
    assert list(_marketcap_start_rows(mc)[:, 0]) == [-1, -1, 2, 2, 2, 5, 5, -1, 8]


def test_base_selection_and_carry():
    a = make_arrays()
    w = target_weights(view(a), P3)
    N = 130
    np.testing.assert_allclose(w[29], expected_row(N, LONG, SHORT))
    # rows 4..24: all R tie at 0 -> order by id: long = first 5 ids of the group, short = last 5
    np.testing.assert_allclose(w[24], expected_row(N, SHORT, LONG))
    nondec = [t for t in range(30) if t not in (4, 9, 14, 19, 24, 29)]
    assert np.isnan(w[nondec]).all()
    dec = w[[4, 9, 14, 19, 24, 29]]
    assert np.all(np.abs(dec).sum(axis=1) <= 1.0 + 1e-12)
    np.testing.assert_allclose(np.where(dec > 0, dec, 0).sum(axis=1), 0.5)
    np.testing.assert_allclose(np.where(dec < 0, dec, 0).sum(axis=1), -0.5)


def test_lookback_not_full_is_flat():
    a = make_arrays()
    w = target_weights(view(a), {"formation_days": 21, "quantile": 0.2})
    for t in (4, 9, 14, 19):        # t - 21 < 0
        assert np.all(w[t] == 0.0)
    assert np.abs(w[24]).sum() == pytest.approx(1.0)


@pytest.mark.parametrize("kind", ["close_in_window", "mc_nan", "universe", "tradable", "late_listing"])
def test_ineligible_stock_is_skipped(kind):
    a = make_arrays()
    j = 125
    if kind == "close_in_window":
        a["close"][26, j] = np.nan          # row t-F for t=29, F=3
        a["dollar_volume"][26, j] = np.nan
    elif kind == "mc_nan":
        a["mc"][29, j] = np.nan
    elif kind == "universe":
        a["universe"][29, j] = False
    elif kind == "tradable":
        a["tradable"][29, j] = False
    elif kind == "late_listing":
        a["close"][:27, j] = np.nan
        a["dollar_volume"][:27, j] = np.nan
    w = target_weights(view(a), P3)
    # 129 eligible -> group = top 25 by TO = E104..E129 without E125; legs of 5
    np.testing.assert_allclose(w[29], expected_row(130, [124, 126, 127, 128, 129], SHORT))


def test_missing_price_outside_window_does_not_exclude():
    a = make_arrays()
    a["close"][25, 125] = np.nan
    a["dollar_volume"][25, 125] = np.nan
    w = target_weights(view(a), P3)
    np.testing.assert_allclose(w[29], expected_row(130, LONG, SHORT))


def test_leg_below_five_is_flat():
    a = make_arrays(N=100)              # group of 20 -> legs of 4
    w = target_weights(view(a), P3)
    assert np.all(w[29] == 0.0)
    assert np.isnan(w[28]).all()


def test_marketcap_rolled_forward_with_price():
    # E104 doubles on row 2 (after its filing on row 0, before the window of t=29): M doubles, TO halves,
    # it leaves the high-TO group and E103 enters (as the lowest-R member: short leg).
    a = make_arrays()
    a["ret_oc"][2, 104] = 1.0
    w = target_weights(view(a), P3)
    np.testing.assert_allclose(w[29], expected_row(130, LONG, [103, 105, 106, 107, 108]))
    # a refiling on row 3 at the new value (2e6) resets s to row 3: same M_t, same answer
    a["mc"][3:, 104] = 2e6
    w2 = target_weights(view(a), P3)
    np.testing.assert_allclose(w2[29], w[29])
    # a frozen value is not rolled before its first usable row: refiling on row 3 at 1e6 gives M=1e6
    a["mc"][3:, 104] = 1e6 + 1.0
    w3 = target_weights(view(a), P3)
    np.testing.assert_allclose(w3[29], expected_row(130, LONG, SHORT))


def test_formation_days_changes_output():
    a = make_arrays()
    a["dollar_volume"][:, 110] = 1e8     # always the highest turnover
    a["ret_oc"][25, 110] = 0.5           # inside a 5-day window ending row 29, outside a 3-day one
    w3 = target_weights(view(a), P3)
    w5 = target_weights(view(a), {"formation_days": 5, "quantile": 0.2})
    assert w3[29, 110] == 0.0
    assert w5[29, 110] == pytest.approx(0.1)
    np.testing.assert_allclose(w5[29], expected_row(130, [110, 126, 127, 128, 129], SHORT))


def test_quantile_changes_output():
    a = make_arrays()
    w = target_weights(view(a), {"formation_days": 3, "quantile": 0.25})
    # group = top floor(32.5)=32 = E098..E129, legs of 8
    np.testing.assert_allclose(w[29], expected_row(130, list(range(122, 130)), list(range(98, 106)),
                                                   0.0625, 0.0625))


def test_truncation_is_point_in_time():
    a = make_arrays()
    full = target_weights(view(a), P3)
    b = {k: v[:25] for k, v in a.items()}
    part = target_weights(view(b), P3)
    np.testing.assert_array_equal(np.isnan(part), np.isnan(full[:25]))
    np.testing.assert_allclose(np.nan_to_num(part), np.nan_to_num(full[:25]))


def test_select_legs_disjoint():
    a = make_arrays()
    rows, legs = select_legs(view(a), P3)
    for sel in legs:
        if sel is not None:
            assert not set(sel[0]) & set(sel[1])
