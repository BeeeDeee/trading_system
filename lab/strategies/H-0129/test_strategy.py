import numpy as np
import pytest
from lab.framework.data import DataView
from strategy import target_weights, PARAMS, decision_mask, legs

N_ELIG = 30
N = 32          # 30 eligible stocks + one bank (deposits > 0) + one without assets
T = 320
D_LAST = 314    # last Friday row with a full hold inside the data (rows are Mon..Fri, Friday = 4 mod 5)

# Hand-derived legs for leg_quantile 0.2: leverage j/100 -> terciles {0..9}, {10..19}, {20..29};
# MA = ((3j) mod 10) * 5 / 100 -> inside each tercile the lowest two are j%10 in {0, 7}, the highest {3, 6}.
REAL = [0, 7, 10, 17, 20, 27]
MON = [3, 6, 13, 16, 23, 26]


def make_dates(n, skip=()):
    base = np.array([4 + 7 * 1800], dtype=np.int64).astype("datetime64[D]")   # a Monday
    days = np.busday_offset(base[0], np.arange(n + len(skip)), roll="forward")
    keep = np.ones(days.size, dtype=bool)
    keep[list(skip)] = False
    return days[keep]


def fundamentals(n_rows):
    j = np.arange(N, dtype=np.float64)
    assets = np.full(N, 100.0)
    liab = j.copy()
    cash = ((np.arange(N) * 3) % 10) * 5.0
    rec = np.zeros(N)
    inv = np.zeros(N)
    dep = np.full(N, np.nan)
    dep[5] = 0.0                       # zero deposits: still eligible
    rec[0] = np.nan                    # missing component counts as 0 -> stock 0 stays real-asset-heavy
    inv[13] = np.nan                   # missing component, top-MA stock stays top
    # bank: lowest MA of all, but deposits > 0 -> excluded
    cash[30] = 0.0
    liab[30] = 15.0
    dep[30] = 50.0
    # no assets -> excluded
    assets[31] = np.nan
    liab[31] = 1.0
    cash[31] = 0.0
    rows = lambda v: np.tile(v.astype(np.float32), (n_rows, 1))
    return {
        "sf1_arq_cashneq": rows(cash),
        "sf1_arq_receivables": rows(rec),
        "sf1_arq_investmentsc": rows(inv),
        "sf1_arq_liabilities": rows(liab),
        "sf1_arq_assets": rows(assets),
        "sf1_arq_deposits": rows(dep),
    }


def make_data(n_rows=T, jump=0.0, jump_row=D_LAST, seed=7, uni=None, trad=None):
    rng = np.random.default_rng(seed)
    dates = make_dates(n_rows)
    ret_co = rng.normal(0, 0.004, (n_rows, N))
    ret_oc = rng.normal(0, 0.008, (n_rows, N)) + rng.normal(0, 0.006, (n_rows, 1))
    t5 = 2.0 + np.cumsum(rng.normal(0, 0.02, n_rows))
    r10 = 1.0 + np.cumsum(rng.normal(0, 0.03, n_rows))
    oil = 50.0 * np.exp(np.cumsum(rng.normal(0, 0.02, n_rows)))
    aaa = 5.0 + np.cumsum(rng.normal(0, 0.02, n_rows))
    baa = aaa + 1.0 + np.cumsum(rng.normal(0, 0.01, n_rows))
    if jump != 0.0:
        t5[jump_row:] += jump
    series = {
        "fred_macro.T5YIE": t5,
        "fred_macro.DFII10": r10,
        "fred_macro.DCOILWTICO": oil,
        "fred_moody_yields_1d.DBAA": baa,
        "fred_moody_yields_1d.DAAA": aaa,
    }
    if uni is None:
        uni = np.ones((n_rows, N), dtype=bool)
    if trad is None:
        trad = np.ones((n_rows, N), dtype=bool)
    names = tuple("S%02d" % k for k in range(N))
    return DataView(
        dates=dates,
        instruments=names,
        asset_class=tuple("us_equity" for _ in names),
        ret_co=ret_co,
        ret_oc=ret_oc,
        tradable=trad,
        listed=np.ones((n_rows, N), dtype=bool),
        delisting=np.zeros((n_rows, N), dtype=bool),
        close=np.full((n_rows, N), 20.0),
        dollar_volume=np.full((n_rows, N), 1e8),
        universe=uni,
        extras=fundamentals(n_rows),
        series=series,
        cash_ret=np.zeros(n_rows),
    )


def nz(row):
    return sorted(np.flatnonzero(np.nan_to_num(row) != 0).tolist())


def test_decision_days_are_fridays_and_holiday_friday_skipped():
    d = make_dates(10)                       # Mon..Fri, Mon..Fri
    assert np.flatnonzero(decision_mask(d)).tolist() == [4, 9]
    d2 = make_dates(9, skip=(4,))            # first Friday is a holiday
    m = decision_mask(d2)
    assert np.flatnonzero(m).tolist() == [8]  # Thursday (row 3) is not known to be the last session


def test_legs_hand_derived():
    ex = {k: v[0].astype(np.float64) for k, v in fundamentals(1).items()}
    real, mon = legs(ex, np.ones(N, bool), np.ones(N, bool), 0.2)
    assert real.tolist() == REAL
    assert mon.tolist() == MON
    assert 30 not in real and 31 not in real and 31 not in mon
    real15, mon15 = legs(ex, np.ones(N, bool), np.ones(N, bool), 0.15)
    assert real15.tolist() == [0, 10, 20] and mon15.tolist() == [3, 13, 23]
    # a non-member / non-tradable stock is not eligible and changes the tercile split
    uni = np.ones(N, bool)
    uni[0] = False
    r2, _ = legs(ex, uni, np.ones(N, bool), 0.2)
    assert 0 not in r2


def test_up_shock_long_real_short_monetary_hold_and_exit():
    W = target_weights(make_data(jump=1.0), PARAMS)
    d = D_LAST
    assert nz(W[d]) == sorted(REAL + MON)
    assert np.allclose(W[d, REAL], 0.5 / 6) and np.allclose(W[d, MON], -0.5 / 6)
    assert abs(np.nansum(W[d])) < 1e-12 and abs(np.nansum(np.abs(W[d])) - 1.0) < 1e-12
    assert np.all(np.isnan(W[d + 1:d + 5]))          # held 5 sessions, no trades
    assert np.all(W[d + 5] == 0.0)                    # exit at the open of t+6


def test_down_shock_mirror():
    W = target_weights(make_data(jump=-1.0), PARAMS)
    d = D_LAST
    assert np.allclose(W[d, MON], 0.5 / 6) and np.allclose(W[d, REAL], -0.5 / 6)


def test_shock_visible_only_the_day_after():
    # the value of session d is in row d+1 (next_morning): it must not move the decision at d
    W = target_weights(make_data(jump=1.0, jump_row=D_LAST + 1), {"leg_quantile": 0.2, "z_shock": 3.0})
    assert np.all(W[D_LAST] == 0.0)
    W2 = target_weights(make_data(jump=1.0), {"leg_quantile": 0.2, "z_shock": 3.0})
    assert len(nz(W2[D_LAST])) == 12


def test_params_change_output():
    data = make_data(jump=1.0)
    W_hi = target_weights(data, {"leg_quantile": 0.2, "z_shock": 1e6})
    assert np.nansum(np.abs(W_hi)) == 0.0
    W15 = target_weights(data, {"leg_quantile": 0.15, "z_shock": 1.0})
    assert nz(W15[D_LAST]) == sorted([0, 10, 20, 3, 13, 23])
    assert np.allclose(W15[D_LAST, [0, 10, 20]], 0.5 / 3)


def test_leaving_universe_or_untradable_closed_next_open_not_replaced():
    uni = np.ones((T, N), dtype=bool)
    uni[D_LAST + 2, 7] = False
    trad = np.ones((T, N), dtype=bool)
    trad[D_LAST + 3, 13] = False
    W = target_weights(make_data(jump=1.0, uni=uni, trad=trad), PARAMS)
    d = D_LAST
    assert W[d + 2, 7] == 0.0 and np.isnan(np.delete(W[d + 2], 7)).all()
    assert W[d + 3, 13] == 0.0 and np.isnan(np.delete(W[d + 3], 13)).all()
    assert np.all(np.isnan(W[d + 1])) and np.all(np.isnan(W[d + 4]))


def test_too_short_history_no_signal():
    data = make_data(n_rows=150, jump=1.0, jump_row=144)
    W = target_weights(data, PARAMS)
    assert np.nansum(np.abs(W)) == 0.0


def test_missing_fred_value_blocks_change():
    data = make_data(jump=1.0)
    data.series["fred_macro.T5YIE"][D_LAST - 5] = np.nan     # the five-session change at s needs both ends
    W = target_weights(data, PARAMS)
    assert np.all(W[D_LAST] == 0.0)


def test_gross_and_dollar_neutral_everywhere():
    W = target_weights(make_data(jump=1.0), {"leg_quantile": 0.25, "z_shock": 0.75})
    g = np.nansum(np.abs(W), axis=1)
    assert np.all(g <= 1.0 + 1e-12)
    full = np.all(np.isfinite(W), axis=1) & (g > 0)
    assert np.allclose(np.nansum(W[full], axis=1), 0.0)


def test_point_in_time_truncation():
    data = make_data(jump=1.0)
    W = target_weights(data, PARAMS)
    cut = 250
    short = make_data(jump=1.0)
    short.ret_co[cut + 1:] = 0.3
    short.ret_oc[cut + 1:] = -0.2
    short.series["fred_macro.T5YIE"][cut + 1:] = 9.0
    W2 = target_weights(short, PARAMS)
    assert np.array_equal(np.isnan(W[:cut + 1]), np.isnan(W2[:cut + 1]))
    assert np.allclose(np.nan_to_num(W[:cut + 1]), np.nan_to_num(W2[:cut + 1]))
