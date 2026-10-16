"""Mechanics tests for H-0042 on hand-made inputs.

Calendar used: weekdays from the numpy epoch (Thursday, January 1 of 1970), built with np.busday_offset.
Row  0 = Jan 1, 1 = Jan 2, 2 = Jan 5, ... 21 = Jan 30 (last weekday of January),
row 22 = Feb 2, 23 = Feb 3, 24 = Feb 4 (first three of February), 41 = Feb 27 (last of February),
row 42 = Mar 2, 43 = Mar 3, 44 = Mar 4, ...
Month-turn window rows: 0-2, 21-24, 41-44, ...
"""
import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS
import strategy as S

UNI = ("SPY", "QQQ", "IWM", "EFA", "EEM", "SHY", "IEF", "TLT", "LQD", "GLD")
# betas to SPY by construction (y = b * x exactly)
BETAS = {"SPY": 1.0, "QQQ": 1.5, "IWM": 1.3, "EFA": 0.9, "EEM": 1.2, "SHY": -0.1,
         "IEF": -0.3, "TLT": -0.6, "LQD": 0.1, "GLD": 0.05}
T = 80


def sessions_to_next(n, rows):
    out = np.full(n, np.nan)
    for d in range(n):
        later = [r - d for r in rows if r > d]
        if later:
            out[d] = min(later)
    return out


def make_view(cpi=(), emp=(), fomc=(), betas=None, late=None, nobar=None, n=T):
    betas = BETAS if betas is None else betas
    epoch = np.zeros(1, dtype="datetime64[D]")
    dates = np.busday_offset(epoch[0], np.arange(n), roll="forward").astype("datetime64[D]")
    x = 0.01 * np.sin(np.arange(n) * 1.7 + 0.3) + 0.002
    N = len(UNI)
    r = np.stack([betas[u] * x for u in UNI], axis=1)
    listed = np.ones((n, N), dtype=bool)
    for name, first in (late or {}).items():
        listed[:first, UNI.index(name)] = False
    r = np.where(listed, r, 0.0)
    close = np.where(listed, 100.0, np.nan)
    for name, row in (nobar or []):
        close[row, UNI.index(name)] = np.nan
    series = {
        "macro_release_calendar_1d.next_cpi_sessions": sessions_to_next(n, cpi),
        "macro_release_calendar_1d.next_empsit_sessions": sessions_to_next(n, emp),
        "macro_release_calendar_1d.next_fomc_sessions": sessions_to_next(n, fomc),
    }
    allr = sorted(set(cpi) | set(emp) | set(fomc))
    series["macro_release_calendar_1d.next_release_sessions"] = sessions_to_next(n, allr)
    return DataView(
        dates=dates, instruments=UNI, asset_class=tuple("us_etf" for _ in UNI),
        ret_co=np.zeros((n, N)), ret_oc=r, tradable=listed.copy(), listed=listed,
        delisting=np.zeros((n, N), dtype=bool), close=close, dollar_volume=np.where(listed, 1e8, np.nan),
        universe=None, extras={}, series=series, cash_ret=np.zeros(n),
    )


P = {"beta_lookback": 5, "tilt": 0.4}
C = {u: UNI.index(u) for u in UNI}


def core(tilt):
    w = np.zeros(len(UNI))
    w[C["SPY"]] = 0.60 - tilt / 5
    w[C["IEF"]] = 0.40 + tilt / 5
    return w


def tilted(tilt, top):
    w = core(tilt)
    w[C["IEF"]] -= tilt
    for u in top:
        w[C[u]] += tilt / 3
    return w


def test_params_match_card():
    assert PARAMS == {"beta_lookback": 252, "tilt": 0.4}


def test_calendar_positions_by_hand():
    v = make_view()
    mt = S.in_month_turn(v.dates)
    expected = {0, 1, 2, 21, 22, 23, 24, 41, 42, 43, 44}
    assert set(np.nonzero(mt[:45])[0]) == expected


def test_single_window_weights_and_drift():
    # CPI on row 12 (Jan 19): tilt sessions rows 11, 12 -> tilted decision at row 10, back to core at row 12
    w = target_weights(make_view(cpi=[12]), P)
    np.testing.assert_allclose(w[10], tilted(0.4, ["QQQ", "IWM", "EEM"]), atol=1e-12)
    assert np.all(np.isnan(w[11]))
    np.testing.assert_allclose(w[12], core(0.4), atol=1e-12)
    # first decision is core; rows without state change or month start are NaN (drift)
    np.testing.assert_allclose(w[0], core(0.4), atol=1e-12)
    for r in list(range(1, 10)) + list(range(13, 22)) + list(range(23, 42)):
        assert np.all(np.isnan(w[r])), r
    # first decision of each month re-applies the core
    np.testing.assert_allclose(w[22], core(0.4), atol=1e-12)
    np.testing.assert_allclose(w[42], core(0.4), atol=1e-12)


def test_spy_in_top_three_adds_to_core():
    b = dict(BETAS, EEM=0.8)   # top three other than IEF: QQQ 1.5, IWM 1.3, SPY 1.0
    w = target_weights(make_view(fomc=[12], betas=b), P)
    exp = tilted(0.4, ["QQQ", "IWM", "SPY"])
    np.testing.assert_allclose(w[10], exp, atol=1e-12)
    assert abs(w[10, C["SPY"]] - (0.52 + 0.4 / 3)) < 1e-12


def test_ief_never_in_top_three():
    b = dict(BETAS, IEF=5.0)
    w = target_weights(make_view(cpi=[12], betas=b), P)
    np.testing.assert_allclose(w[10], tilted(0.4, ["QQQ", "IWM", "EEM"]), atol=1e-12)


@pytest.mark.parametrize("a", [21, 22, 23, 24, 25, 41, 42, 43, 44, 45])
def test_month_turn_exclusion(a):
    # A or A-1 in the month-turn window -> no tilt anywhere
    w = target_weights(make_view(emp=[a]), P)
    rows = ~np.all(np.isnan(w), axis=1)
    for r in np.nonzero(rows)[0]:
        np.testing.assert_allclose(w[r], core(0.4), atol=1e-12)


@pytest.mark.parametrize("a", [20, 26, 40, 46])
def test_month_turn_edges_kept(a):
    # A = Jan 29 (A-1 Jan 28), Feb 6 (A-1 Feb 5), Feb 26, Mar 6: windows do not touch the month turn
    w = target_weights(make_view(emp=[a]), P)
    np.testing.assert_allclose(w[a - 2], tilted(0.4, ["QQQ", "IWM", "EEM"]), atol=1e-12)
    np.testing.assert_allclose(w[a], core(0.4), atol=1e-12)


def test_excluded_cpi_followed_by_valid_fomc():
    # CPI row 25 (Feb 5) is excluded (A-1 = Feb 4 is the third session); FOMC row 26 (Feb 6) is valid:
    # tilt sessions 25, 26 -> tilted decision at row 24, core again at row 26
    w = target_weights(make_view(cpi=[25], fomc=[26]), P)
    np.testing.assert_allclose(w[24], tilted(0.4, ["QQQ", "IWM", "EEM"]), atol=1e-12)
    assert np.all(np.isnan(w[25]))
    np.testing.assert_allclose(w[26], core(0.4), atol=1e-12)


def test_joined_windows_single_entry():
    # CPI row 12 and FOMC row 13: tilt sessions 11, 12, 13 -> one entry (row 10) and one exit (row 13)
    w = target_weights(make_view(cpi=[12], fomc=[13]), P)
    np.testing.assert_allclose(w[10], tilted(0.4, ["QQQ", "IWM", "EEM"]), atol=1e-12)
    assert np.all(np.isnan(w[11])) and np.all(np.isnan(w[12]))
    np.testing.assert_allclose(w[13], core(0.4), atol=1e-12)
    # back-to-back windows (A at 12 and 14: sessions 11..14) also join
    w = target_weights(make_view(cpi=[12], fomc=[14]), P)
    np.testing.assert_allclose(w[10], tilted(0.4, ["QQQ", "IWM", "EEM"]), atol=1e-12)
    assert all(np.all(np.isnan(w[r])) for r in (11, 12, 13))
    np.testing.assert_allclose(w[14], core(0.4), atol=1e-12)


def test_needs_nine_eligible_and_late_listing():
    # EEM and GLD listed from row 30: with lookback 5 they are eligible from row 35.
    # A at row 12: only 8 eligible -> core. A at row 50: 10 eligible -> tilt.
    v = make_view(cpi=[12, 50], late={"EEM": 30, "GLD": 30})
    w = target_weights(v, P)
    assert np.all(np.isnan(w[10]))
    np.testing.assert_allclose(w[48], tilted(0.4, ["QQQ", "IWM", "EEM"]), atol=1e-12)
    # not-yet-listed ETFs (no bar) keep their weight on decision rows: NaN element
    assert np.isnan(w[0, C["EEM"]]) and np.isnan(w[0, C["GLD"]])
    np.testing.assert_allclose(w[0, [C["SPY"], C["IEF"]]], [0.52, 0.48], atol=1e-12)
    # only GLD late: 9 eligible is enough
    w = target_weights(make_view(cpi=[12], late={"GLD": 30}), P)
    exp = tilted(0.4, ["QQQ", "IWM", "EEM"])
    exp[C["GLD"]] = np.nan
    np.testing.assert_allclose(w[10], exp, atol=1e-12)


def test_no_bar_on_decision_day_keeps_weight():
    w = target_weights(make_view(cpi=[12], nobar=[("IWM", 10)]), P)
    assert np.isnan(w[10, C["IWM"]])
    assert abs(w[10, C["QQQ"]] - 0.4 / 3) < 1e-12


def test_missing_calendar_means_core():
    v = make_view(cpi=[12])
    for k in list(v.series):
        v.series[k] = np.full(T, np.nan)
    w = target_weights(v, P)
    for r in np.nonzero(~np.all(np.isnan(w), axis=1))[0]:
        np.testing.assert_allclose(w[r], core(0.4), atol=1e-12)


def test_missing_series_raises():
    v = make_view(cpi=[12])
    del v.series["macro_release_calendar_1d.next_fomc_sessions"]
    with pytest.raises(KeyError):
        target_weights(v, P)


def test_params_change_output():
    v = make_view(cpi=[12, 50])
    base = target_weights(v, P)
    w_t = target_weights(v, {"beta_lookback": 5, "tilt": 0.5})
    np.testing.assert_allclose(w_t[10], tilted(0.5, ["QQQ", "IWM", "EEM"]), atol=1e-12)
    assert abs(w_t[10, C["IEF"]]) < 1e-12
    assert not np.allclose(np.nan_to_num(base[10]), np.nan_to_num(w_t[10]))
    # longer lookback: nobody eligible at row 10 (needs 20 returns) -> core only; eligible by row 48
    w_l = target_weights(v, {"beta_lookback": 20, "tilt": 0.4})
    assert np.all(np.isnan(w_l[10]))
    np.testing.assert_allclose(w_l[48], tilted(0.4, ["QQQ", "IWM", "EEM"]), atol=1e-12)


def test_gross_and_long_only():
    v = make_view(cpi=[12, 30, 50, 70], fomc=[13, 33, 52], late={"GLD": 30})
    for p in ({"beta_lookback": 5, "tilt": t} for t in (0.3, 0.4, 0.5)):
        w = target_weights(v, p)
        x = np.nan_to_num(w)
        assert np.all(x >= 0)
        assert np.all(x.sum(axis=1) <= 1 + 1e-9)


def test_point_in_time_truncation():
    v = make_view(cpi=[12, 30, 50, 70], fomc=[13, 33, 52])
    full = target_weights(v, P)
    for cut in (11, 12, 31, 49, 60):
        tv = DataView(
            dates=v.dates[:cut], instruments=v.instruments, asset_class=v.asset_class,
            ret_co=v.ret_co[:cut], ret_oc=v.ret_oc[:cut], tradable=v.tradable[:cut], listed=v.listed[:cut],
            delisting=v.delisting[:cut], close=v.close[:cut], dollar_volume=v.dollar_volume[:cut],
            universe=None, extras={}, series={k: s[:cut] for k, s in v.series.items()}, cash_ret=v.cash_ret[:cut],
        )
        np.testing.assert_array_equal(target_weights(tv, P), full[:cut])
