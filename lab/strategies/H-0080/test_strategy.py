import numpy as np
import pytest
from lab.framework.data import DataView
from strategy import target_weights, PARAMS, residual_z, events_from_z, CANDIDATES, diag_events

NAMES = ("SPY", "IEF", "SHY", "AGG", "LQD", "HYG", "JNK", "EMB", "MUB", "BKLN")


def make_view(T=420, seed=0, listed_from=None, shocks=None):
    """Synthetic panel: candidates = 0.4*SPY + 0.5*IEF + noise. shocks: {name: (row, size)} adds
    a residual of `size` spread over the 5 sessions ending at `row` (so the 5-session sum is `size`)."""
    rng = np.random.default_rng(seed)
    N = len(NAMES)
    dates = np.busday_offset(np.zeros(T, dtype="datetime64[D]"), np.arange(T), roll="forward")
    spy = rng.normal(0.0003, 0.01, T)
    ief = rng.normal(0.0001, 0.003, T)
    r = np.zeros((T, N))
    r[:, 0] = spy
    r[:, 1] = ief
    for i in range(2, N):
        r[:, i] = 0.4 * spy + 0.5 * ief + rng.normal(0, 0.002, T)
    if shocks:
        for name, (row, size) in shocks.items():
            r[row, NAMES.index(name)] += size
    listed = np.ones((T, N), dtype=bool)
    if listed_from:
        for name, row in listed_from.items():
            listed[:row, NAMES.index(name)] = False
    r = np.where(listed, r, 0.0)
    ret_co = np.zeros((T, N))
    ret_oc = r.copy()
    close = np.where(listed, 100.0 * np.cumprod(1 + r, axis=0), np.nan)
    tradable = listed.copy()
    return DataView(
        dates=dates,
        instruments=NAMES,
        asset_class=tuple("us_etf" for _ in NAMES),
        ret_co=ret_co,
        ret_oc=ret_oc,
        tradable=tradable,
        listed=listed,
        delisting=np.zeros((T, N), dtype=bool),
        close=close,
        dollar_volume=np.where(listed, 1e9, np.nan),
        universe=None,
        extras={},
        series={},
        cash_ret=np.zeros(T),
    )


def ref_z(data, name, t):
    """Direct OLS by lstsq on the card's windows."""
    r = (1 + data.ret_co) * (1 + data.ret_oc) - 1
    j, s, i = data.col(name), data.col("SPY"), data.col("IEF")
    fit = slice(t - 256, t - 4)
    X = np.column_stack([np.ones(252), r[fit, s], r[fit, i]])
    beta, *_ = np.linalg.lstsq(X, r[fit, j], rcond=None)
    res = r[fit, j] - X @ beta
    sig = np.std(res, ddof=1)
    win = slice(t - 4, t + 1)
    Xw = np.column_stack([np.ones(5), r[win, s], r[win, i]])
    S = np.sum(r[win, j] - Xw @ beta)
    return S / (sig * np.sqrt(5))


def test_z_matches_direct_ols():
    d = make_view()
    z = residual_z(d)
    for t in (300, 350, 419):
        for name in ("LQD", "BKLN"):
            assert z[t, CANDIDATES.index(name)] == pytest.approx(ref_z(d, name, t), rel=1e-6, abs=1e-9)


def test_no_signal_before_history_and_late_listing():
    d = make_view(listed_from={"EMB": 100})
    z = residual_z(d)
    k = CANDIDATES.index("LQD")
    assert np.all(np.isnan(z[:256, k])) and np.isfinite(z[256, k])
    e = CANDIDATES.index("EMB")
    # needs 257 sessions of history and 200 valid fit sessions: first signal at row 100+256
    assert np.all(np.isnan(z[:356, e])) and np.isfinite(z[356, e])


def test_event_spacing():
    z = np.full((20, 1), 0.0)
    z[[3, 4, 8, 9, 15], 0] = -3.0
    ev = events_from_z(z, 2.0, 5)
    assert list(np.flatnonzero(ev[:, 0])) == [3, 9, 15]
    ev3 = events_from_z(z, 2.0, 3)
    assert list(np.flatnonzero(ev3[:, 0])) == [3, 8, 15]
    assert not events_from_z(z, 3.5, 5).any()


def test_slot_weights_and_hold():
    p = dict(PARAMS, k_sigma=5.0)
    d = make_view(shocks={"HYG": (330, -0.05)})
    z = residual_z(d)
    k = CANDIDATES.index("HYG")
    assert z[330, k] < -5.0
    w = target_weights(d, p)
    hy, ief, spy = d.col("HYG"), d.col("IEF"), d.col("SPY")
    t0 = int(np.flatnonzero(events_from_z(z, 5.0, 5)[:, k])[0])
    assert t0 == 330
    assert w[t0, hy] == pytest.approx(0.2) and w[t0, ief] == pytest.approx(0.2)
    assert w[t0, spy] == pytest.approx(0.6)
    h = PARAMS["hold_days"]
    # targets between are NaN (keep) or the same slot targets
    for t in range(t0 + 1, t0 + h):
        assert np.isnan(w[t, hy]) or w[t, hy] == pytest.approx(0.2)
    assert w[t0 + h, hy] == 0.0 and w[t0 + h, ief] == pytest.approx(0.4)


def test_two_slot_cap_most_negative_first():
    # k_sigma = 5 so that only the injected shocks fire (noise events would occupy slots)
    p = dict(PARAMS, k_sigma=5.0)
    d = make_view(shocks={"LQD": (330, -0.03), "JNK": (330, -0.06), "MUB": (330, -0.045)})
    z = residual_z(d)
    ev = events_from_z(z, 5.0, 5)
    assert not ev[:330].any()
    assert sorted(CANDIDATES[k] for k in np.flatnonzero(ev[330])) == ["JNK", "LQD", "MUB"]
    assert z[330, CANDIDATES.index("JNK")] < z[330, CANDIDATES.index("MUB")] < z[330, CANDIDATES.index("LQD")]
    w = target_weights(d, p)
    row = w[330]
    assert np.isfinite(row).all()
    held = sorted(n for n in CANDIDATES if row[d.col(n)] > 0)
    assert held == ["JNK", "MUB"]          # LQD dropped: no free slot
    assert row[d.col("IEF")] == pytest.approx(0.0)
    assert row[d.col("SPY")] == pytest.approx(0.6)
    # LQD does not enter later when a slot frees (its event was dropped)
    assert np.all(np.nan_to_num(w[331:, d.col("LQD")]) == 0)
    assert w[335, d.col("IEF")] == pytest.approx(0.4)


def test_book_invariants_and_month_rebalance():
    d = make_view(seed=3, shocks={"EMB": (300, -0.05), "LQD": (380, -0.05)})
    w = target_weights(d, PARAMS)
    dec = np.isfinite(w).all(axis=1)
    assert dec[0]
    assert np.all(np.isnan(w[~dec]))
    assert np.all(w[dec] >= 0)
    assert np.allclose(w[dec].sum(axis=1), 1.0)
    assert np.allclose(w[dec, d.col("SPY")], 0.6)
    assert np.all(w[dec][:, [d.col("SHY"), d.col("AGG")]] == 0)
    months = d.dates.astype("datetime64[M]")
    first = np.r_[True, months[1:] != months[:-1]]
    assert np.all(dec[first])


def test_params_change_output():
    d = make_view(seed=5, shocks={"HYG": (330, -0.035), "LQD": (400, -0.06)})
    base = target_weights(d, PARAMS)
    w_h = target_weights(d, dict(PARAMS, hold_days=3))
    w_k = target_weights(d, dict(PARAMS, k_sigma=2.5))
    w_k2 = target_weights(d, dict(PARAMS, k_sigma=1.5))
    eq = lambda a, b: np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(np.nan_to_num(a), np.nan_to_num(b))
    assert not eq(base, w_h)
    assert not (eq(base, w_k) and eq(base, w_k2))


def truncate(d, n):
    return DataView(
        dates=d.dates[:n], instruments=d.instruments, asset_class=d.asset_class,
        ret_co=d.ret_co[:n], ret_oc=d.ret_oc[:n], tradable=d.tradable[:n], listed=d.listed[:n],
        delisting=d.delisting[:n], close=d.close[:n], dollar_volume=d.dollar_volume[:n],
        universe=None, extras={}, series={}, cash_ret=d.cash_ret[:n],
    )


def test_truncation_invariance():
    d = make_view(seed=7, shocks={"BKLN": (330, -0.05)})
    w = target_weights(d, PARAMS)
    for cut in (300, 331, 333, 380):
        wt = target_weights(truncate(d, cut + 1), PARAMS)
        a = w[:cut + 1]
        assert np.array_equal(np.isnan(a), np.isnan(wt))
        assert np.allclose(np.nan_to_num(a), np.nan_to_num(wt))


def test_slot_released_when_price_stops():
    p = dict(PARAMS, k_sigma=5.0)
    d = make_view(shocks={"HYG": (330, -0.05)})
    t0 = 330
    hy = d.col("HYG")
    d.listed[t0 + 2, hy] = False
    d.close[t0 + 2, hy] = np.nan
    w = target_weights(d, p)
    assert w[t0, hy] == pytest.approx(0.2)
    assert w[t0 + 2, hy] == 0.0 and w[t0 + 2, d.col("IEF")] == pytest.approx(0.4)


def test_diagnostic_events_match_strategy():
    d = make_view(shocks={"HYG": (330, -0.05), "MUB": (360, -0.05)})
    mask, side = diag_events(d, PARAMS)
    ev = events_from_z(residual_z(d), PARAMS["k_sigma"], PARAMS["hold_days"])
    for k, n in enumerate(CANDIDATES):
        assert np.array_equal(mask[:, d.col(n)], ev[:, k])
    for n in ("SPY", "IEF", "SHY", "AGG"):
        assert not mask[:, d.col(n)].any()
    assert np.all(side == 1.0)
