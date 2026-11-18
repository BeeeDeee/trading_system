import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS

FUNDS = ("ADX", "GAB", "USA", "TY", "GAM", "RVT", "RMT", "NAD", "NEA", "NUV", "NZF", "NVG", "NMZ", "MQY",
         "MYI", "BFK", "VKQ", "PML", "PMF", "PHK", "DHY", "HIX", "EHI", "DSU", "HYT", "PCN", "PTY", "RQI",
         "RNP", "IGR", "JRS", "UTF", "UTG", "DNP", "GUT", "MXF", "TWN", "KF", "GF", "SWZ", "JOF", "TDF",
         "EMF", "IFN")
ANCHORS = ("SPY", "IWM", "MUB", "HYG", "IYR", "XLU", "EWW", "EWT", "EWY", "EWG", "EWL", "EWJ", "EEM")
ANCHOR_OF = dict(zip(FUNDS, ["SPY"] * 5 + ["IWM"] * 2 + ["MUB"] * 12 + ["HYG"] * 8 + ["IYR"] * 4
                     + ["XLU"] * 4 + ["EWW", "EWT", "EWY", "EWG", "EWL", "EWJ"] + ["EEM"] * 3))
NAMES = FUNDS + ANCHORS
T = 420


def make_returns(seed=0, beta=1.0, gauss=False):
    """Anchors N(0, 1%); each fund = beta * anchor + noise. Default noise alternates +-0.5 % so that a
    5-session residual sum is +-0.5 % and |z| stays near 0.45: no event unless one is planted."""
    rng = np.random.default_rng(seed)
    N = len(NAMES)
    r = np.zeros((T, N))
    alt = 0.005 * np.where(np.arange(T) % 2 == 0, 1.0, -1.0)
    for a in ANCHORS:
        r[:, NAMES.index(a)] = rng.normal(0.0, 0.01, T)
    for f in FUNDS:
        noise = rng.normal(0.0, 0.005, T) if gauss else alt
        r[:, NAMES.index(f)] = beta * r[:, NAMES.index(ANCHOR_OF[f])] + noise
    return r


def make_view(r, listed=None, tradable=None, dollar_volume=None):
    N = len(NAMES)
    if listed is None:
        listed = np.ones((T, N), dtype=bool)
    if tradable is None:
        tradable = listed.copy()
    if dollar_volume is None:
        dollar_volume = np.full((T, N), 2e6)
    close = np.where(listed, 10.0 * np.cumprod(1.0 + r, axis=0), np.nan)
    dates = np.arange(T).astype("datetime64[D]")
    return DataView(
        dates=dates,
        instruments=NAMES,
        asset_class=tuple("us_etf" for _ in NAMES),
        ret_co=np.where(listed, r / 2.0, 0.0),
        ret_oc=np.where(listed, (1.0 + r) / (1.0 + r / 2.0) - 1.0, 0.0),
        tradable=tradable,
        listed=listed,
        delisting=np.zeros((T, N), dtype=bool),
        close=close,
        dollar_volume=np.where(listed, dollar_volume, np.nan),
        universe=None,
        extras={},
        series={},
        cash_ret=np.zeros(T),
    )


def col(name):
    return NAMES.index(name)


def entries(w, name):
    """Rows where the weight of `name` goes from 0 to positive."""
    x = w[:, col(name)]
    prev = np.concatenate([[0.0], x[:-1]])
    return np.flatnonzero((x > 0) & (prev <= 0))


def dimson_beta(r, f, a, e):
    """Reference Dimson beta over rows e-251..e by least squares (all rows valid)."""
    rows = np.arange(max(2, e - 251), e + 1)
    y = r[rows, col(f)]
    X = np.column_stack([np.ones(len(rows)), r[rows, col(a)], r[rows - 1, col(a)]])
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(np.clip(coef[1] + coef[2], 0.3, 2.5))


def test_params_match_card():
    assert PARAMS == {"hold_sessions": 21, "z_entry": 2.0}


def test_single_event_weights_and_hold():
    r = make_returns(1, beta=1.4)
    r[350, col("ADX")] -= 0.20          # one-day 20 % shortfall vs SPY
    w = target_weights(make_view(r), PARAMS)
    assert 350 in entries(w, "ADX")
    b = dimson_beta(r, "ADX", "SPY", 345)   # window ends at t-5
    wf = 0.20 / (1.0 + b)
    # held on rows 350..370 (entry at open 351, exit at open 372 = 351 + 21)
    assert np.allclose(w[350:371, col("ADX")], wf, atol=1e-9)
    assert w[371, col("ADX")] == 0.0
    assert w[349, col("ADX")] == 0.0


def test_beta_sized_hedge_and_slot_gross():
    r = make_returns(2, beta=1.4)
    r[350, col("TWN")] -= 0.20          # TWN is the only fund on EWT
    w = target_weights(make_view(r), PARAMS)
    b = dimson_beta(r, "TWN", "EWT", 345)
    assert w[350, col("TWN")] == pytest.approx(0.20 / (1 + b), abs=1e-9)
    assert w[350, col("EWT")] == pytest.approx(-0.20 * b / (1 + b), abs=1e-9)
    assert w[350, col("TWN")] - w[350, col("EWT")] == pytest.approx(0.20, abs=1e-12)


def test_no_event_before_beta_window_is_valid():
    r = make_returns(3)
    r[150, col("ADX")] -= 0.20          # fewer than 189 sessions in the beta window
    w = target_weights(make_view(r), PARAMS)
    assert np.all(w[:200, col("ADX")] == 0.0)


def test_dedup_within_hold_and_hold_param():
    r = make_returns(4)
    r[300, col("ADX")] -= 0.20
    r[340, col("ADX")] -= 0.20          # 40 sessions later: new event for hold 21, blocked for hold 42
    w21 = target_weights(make_view(r), PARAMS)
    w42 = target_weights(make_view(r), {**PARAMS, "hold_sessions": 42})
    assert list(entries(w21, "ADX")) == [300, 340]
    assert list(entries(w42, "ADX")) == [300]
    assert np.all(w42[300:342, col("ADX")] > 0) and w42[342, col("ADX")] == 0.0
    r[310, col("ADX")] -= 0.20          # 10 sessions after the first event: no second event at hold 21
    w = target_weights(make_view(r), PARAMS)
    assert 310 not in entries(w, "ADX")


def test_slot_cap_most_negative_first_and_netting():
    r = make_returns(5)
    shocks = {"ADX": 0.30, "GAB": 0.25, "USA": 0.22, "NAD": 0.20, "PHK": 0.18, "RQI": 0.10}
    for f, s in shocks.items():
        r[300, col(f)] -= s
    w = target_weights(make_view(r), PARAMS)
    opened = [f for f in shocks if w[300, col(f)] > 0]
    # same residual sigma for all funds, so z orders as the shocks: the smallest (RQI) is dropped
    assert opened == ["ADX", "GAB", "USA", "NAD", "PHK"]
    assert np.sum(np.abs(w[300])) <= 1.0 + 1e-12
    # the dropped event is not queued
    assert np.all(w[:, col("RQI")] == 0.0)
    # three SPY funds net in one SPY short
    spy_funds = [f for f in ("ADX", "GAB", "USA") if f in opened]
    expected = -sum(0.20 - w[300, col(f)] for f in spy_funds)
    assert w[300, col("SPY")] == pytest.approx(expected, abs=1e-12)


def test_late_listing_is_ineligible():
    r = make_returns(6)
    listed = np.ones((T, len(NAMES)), dtype=bool)
    listed[:200, col("KF")] = False
    r[300, col("KF")] -= 0.20           # beta window has < 189 joint sessions
    w = target_weights(make_view(r, listed=listed), PARAMS)
    assert np.all(w[:, col("KF")] == 0.0)


def test_liquidity_floor():
    r = make_returns(7)
    r[350, col("ADX")] -= 0.20
    dv = np.full((T, len(NAMES)), 2e6)
    dv[:, col("ADX")] = 4e5
    w = target_weights(make_view(r, dollar_volume=dv), PARAMS)
    assert np.all(w[:, col("ADX")] == 0.0)
    dv[:, col("ADX")] = 4e5
    dv[350 - 31:351, col("ADX")] = 6e5  # 32 of the last 63 sessions above the floor: median above
    w = target_weights(make_view(r, dollar_volume=dv), PARAMS)
    assert 350 in entries(w, "ADX")
    dv[319, col("ADX")] = 4e5          # 31 of 63: median below the floor
    w = target_weights(make_view(r, dollar_volume=dv), PARAMS)
    assert np.all(w[:, col("ADX")] == 0.0)


def test_slot_closes_when_a_leg_stops_trading():
    r = make_returns(8)
    r[350, col("ADX")] -= 0.20
    listed = np.ones((T, len(NAMES)), dtype=bool)
    listed[360:, col("ADX")] = False
    w = target_weights(make_view(r, listed=listed), PARAMS)
    assert np.all(w[350:360, col("ADX")] > 0)
    assert np.all(w[360:, col("ADX")] == 0.0)
    assert np.all(w[360:, col("SPY")] == 0.0)


def test_z_entry_changes_output_and_gross():
    r = make_returns(9, gauss=True)
    v = make_view(r)
    w15 = target_weights(v, {**PARAMS, "z_entry": 1.5})
    w25 = target_weights(v, {**PARAMS, "z_entry": 2.5})
    assert not np.array_equal(w15, w25)
    # planted shock: residual sigma 0.5 %, alternating noise sums to +0.5 % on even rows, -0.5 % on odd rows
    # z(349) = -0.45, z(350) = (-0.0296 + 0.005) / (0.005 sqrt 5) = -2.20, z(351) = (-0.0296 - 0.005) / ... = -3.09
    r = make_returns(9)
    r[350, col("ADX")] -= 0.0296
    v1 = make_view(r)
    assert list(entries(target_weights(v1, PARAMS), "ADX")) == [350]
    assert list(entries(target_weights(v1, {**PARAMS, "z_entry": 2.5}), "ADX")) == [351]
    for w in (w15, w25, target_weights(v, PARAMS)):
        assert np.all(np.abs(w).sum(axis=1) <= 1.0 + 1e-12)
        assert np.all(w[:, [col(f) for f in FUNDS]] >= 0)
        assert np.all(w[:, [col(a) for a in ANCHORS]] <= 0)


def test_deterministic_and_prefix_stable():
    r = make_returns(10)
    w1 = target_weights(make_view(r), PARAMS)
    w2 = target_weights(make_view(r), PARAMS)
    assert np.array_equal(w1, w2)
    r2 = r.copy()
    r2[330:] = np.random.default_rng(99).normal(0, 0.03, r2[330:].shape)
    w3 = target_weights(make_view(r2), PARAMS)
    assert np.array_equal(w1[:330], w3[:330])
