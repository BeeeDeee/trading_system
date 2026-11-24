import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS
from strategy import decision_rows, day_numbers, core

N = 10
BETAS = np.linspace(0.5, 1.4, N)
START = 12786          # day number of a Monday (weekday = (d + 3) % 7 == 0)
WEEKS = 156


def make_arrays(seed=0):
    days = np.arange(START, START + 7 * WEEKS)
    days = days[(days + 3) % 7 < 5]
    T = days.size
    rng = np.random.default_rng(seed)
    f = rng.normal(0.0, 0.01, T)
    r = f[:, None] * BETAS[None, :]
    flag = np.zeros((T, N), dtype=np.float32)
    mondays = np.flatnonzero((days + 3) % 7 == 0)
    for s in mondays:
        c = ((days[s] - START) // 7) % 5
        flag[s, :c] = 1.0
    return {
        "days": days,
        "ret_co": np.zeros((T, N)),
        "ret_oc": r,
        "close": np.full((T, N), 50.0),
        "listed": np.ones((T, N), dtype=bool),
        "tradable": np.ones((T, N), dtype=bool),
        "universe": np.ones((T, N), dtype=bool),
        "flag": flag,
    }


def view(a, upto=None):
    sl = slice(None) if upto is None else slice(0, upto + 1)
    T = a["days"][sl].size
    return DataView(
        dates=a["days"][sl].astype("datetime64[D]"),
        instruments=tuple("E%d" % (100 + j) for j in range(N)),
        asset_class=tuple("us_equity" for _ in range(N)),
        ret_co=a["ret_co"][sl],
        ret_oc=a["ret_oc"][sl],
        tradable=a["tradable"][sl],
        listed=a["listed"][sl],
        delisting=np.zeros((T, N), dtype=bool),
        close=a["close"][sl],
        dollar_volume=np.full((T, N), 1e8),
        universe=a["universe"][sl],
        extras={"ea_reaction_day": a["flag"][sl]},
        series={},
        cash_ret=np.zeros(T),
    )


def window_monday_row(a, t):
    """Row of the Monday inside [date(t) + 1 - 364, date(t) + 7 - 364] for a Friday t."""
    return int(np.flatnonzero(a["days"] == a["days"][t] + 3 - 364)[0])


def fridays(a):
    return np.flatnonzero((a["days"] + 3) % 7 == 4)


def pick_rows(a):
    fr = fridays(a)
    late = fr[fr > 120 * 5]
    # F: a late Friday whose window gets every stock flagged (S = 1, the maximum) -> peak
    F = int(late[0])
    # G: a later Friday whose window Monday has c = 0 (S = 0, the minimum) -> off-peak
    G = None
    for t in late[3:]:
        s = window_monday_row(a, t)
        if ((a["days"][s] - START) // 7) % 5 == 0:
            G = int(t)
            break
    assert G is not None
    return F, G


def peak_data():
    a = make_arrays()
    F, G = pick_rows(a)
    a["flag"][window_monday_row(a, F), :] = 1.0
    return a, F, G


def test_decisions_only_on_fridays():
    a = make_arrays()
    W = target_weights(view(a), PARAMS)
    dec = np.flatnonzero(~np.isnan(W).all(axis=1))
    assert np.all((a["days"][dec] + 3) % 7 == 4)
    assert np.array_equal(decision_rows(a["days"].astype("datetime64[D]")), fridays(a))


def test_peak_week_holds_top_quintile_and_offpeak_bottom():
    a, F, G = peak_data()
    W = target_weights(view(a), PARAMS)
    expect_top = np.zeros(N)
    expect_top[[8, 9]] = 0.5
    expect_bot = np.zeros(N)
    expect_bot[[0, 1]] = 0.5
    assert np.allclose(W[F], expect_top)
    assert np.allclose(W[G], expect_bot)
    out, _, _ = core(view(a), PARAMS)
    peaks = {t: p for t, p, _, _ in out}
    assert peaks[F] and not peaks[G]


def test_before_52_weekly_values_book_is_bottom_quintile():
    a = make_arrays()
    a["flag"][:, :] = 1.0          # every week maximal, but no S history yet in year 1-2
    fr = fridays(a)
    t = int(fr[fr > 230][0])       # beta valid (>= 0.8 * 252 rows), S history < 52 values
    W = target_weights(view(a), PARAMS)
    expect = np.zeros(N)
    expect[[0, 1]] = 0.5
    assert np.allclose(W[t], expect)


def test_no_beta_before_enough_history_is_flat():
    a = make_arrays()
    W = target_weights(view(a), PARAMS)
    t = int(fridays(a)[2])
    assert np.allclose(W[t], 0.0)


def test_stock_without_beta_is_skipped():
    a, F, G = peak_data()
    lo = F - 252 + 1
    a["close"][lo:lo + 60, 9] = np.nan      # 60 of 252 missing -> under 80 % -> no beta
    W = target_weights(view(a), PARAMS)
    assert W[F, 9] == 0.0
    # 9 stocks with beta -> quintile of 1: stock 8 alone
    assert np.isclose(W[F, 8], 1.0)
    assert np.isclose(np.nansum(W[F]), 1.0)


def test_late_listing_has_no_beta():
    a, F, G = peak_data()
    a["listed"][:F - 150, 9] = False
    a["close"][:F - 150, 9] = np.nan
    a["universe"][:F - 150, 9] = False
    W = target_weights(view(a), PARAMS)
    assert W[F, 9] == 0.0
    W2 = target_weights(view(a), dict(PARAMS, beta_lookback=126))
    assert np.isclose(W2[F, 9], 0.5)


def test_non_member_never_held_and_midweek_exit():
    a, F, G = peak_data()
    a["universe"][F + 2, 9] = False
    W = target_weights(view(a), PARAMS)
    assert W[F + 2, 9] == 0.0
    assert np.isnan(W[F + 2, 8]) and np.isnan(W[F + 1]).all()
    a2, F2, _ = peak_data()
    a2["universe"][F2, 9] = False
    W2 = target_weights(view(a2), PARAMS)
    assert W2[F2, 9] == 0.0


def test_gross_and_long_only():
    a, F, G = peak_data()
    for p in (PARAMS, dict(PARAMS, beta_lookback=126), dict(PARAMS, peak_fraction=0.5)):
        W = target_weights(view(a), p)
        g = np.nansum(np.abs(W), axis=1)
        assert np.all(g <= 1.0 + 1e-9)
        assert np.all(np.nan_to_num(W) >= 0.0)


def test_params_change_output():
    a, F, G = peak_data()
    base = target_weights(view(a), PARAMS)
    pf = target_weights(view(a), dict(PARAMS, peak_fraction=0.5))
    lb = target_weights(view(a), dict(PARAMS, beta_lookback=126))
    assert not np.array_equal(np.nan_to_num(base, nan=-1), np.nan_to_num(pf, nan=-1))
    assert not np.array_equal(np.nan_to_num(base, nan=-1), np.nan_to_num(lb, nan=-1))


def test_point_in_time_truncation():
    a, F, G = peak_data()
    full = target_weights(view(a), PARAMS)
    for cut in (F - 3, F, G - 1, G):
        part = target_weights(view(a, cut), PARAMS)
        assert np.array_equal(np.nan_to_num(part, nan=-1), np.nan_to_num(full[:cut + 1], nan=-1))


def test_intensity_uses_year_old_flags_only():
    a, F, G = peak_data()
    full = target_weights(view(a), PARAMS)
    b = make_arrays()
    b["flag"][window_monday_row(b, F), :] = 1.0
    b["flag"][F - 250:, :] = 0.0              # flags newer than 357 days do not matter at F
    W = target_weights(view(b), PARAMS)
    assert np.array_equal(np.nan_to_num(W[:F + 1], nan=-1), np.nan_to_num(full[:F + 1], nan=-1))


def test_missing_flag_input_raises():
    a = make_arrays()
    v = view(a)
    v.extras.pop("ea_reaction_day")
    with pytest.raises(KeyError):
        target_weights(v, PARAMS)
