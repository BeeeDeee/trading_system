import numpy as np
import pytest

from strategy import target_weights, PARAMS
from strategy import oil_returns, rolling_slope, rolling_sd, signals, book


# ---------------------------------------------------------------- building blocks

def test_params_match_card():
    assert PARAMS == {"n_leg": 3, "shock_z": 1.5}


def test_oil_returns_missing_stale_negative_winsor():
    p = np.array([np.nan, 50.0, 50.0, 55.0, -10.0, 20.0, 30.0, 100.0, np.nan, 110.0])
    p[6] = 22.0
    o = oil_returns(p)
    assert np.isnan(o[0])                      # no price
    assert np.isnan(o[1])                      # no previous available value
    assert np.isnan(o[2])                      # repeated (forward-filled) value: not a new print
    assert o[3] == pytest.approx(0.10)
    assert np.isnan(o[4])                      # price <= 0
    assert np.isnan(o[5])                      # previous price <= 0
    assert o[6] == pytest.approx(0.10)
    assert o[7] == pytest.approx(0.20)         # +355 % winsorized at +20 %
    assert np.isnan(o[8])
    assert o[9] == pytest.approx(0.10)         # measured against the last available value (100)


def test_rolling_slope_matches_polyfit_and_min_obs():
    rng = np.random.default_rng(1)
    x = rng.normal(size=80)
    y = 0.7 * x + rng.normal(scale=0.3, size=80)
    b = rolling_slope(y, x, 30, 20)
    ref = np.polyfit(x[21:51], y[21:51], 1)[0]
    assert b[50] == pytest.approx(ref, rel=1e-9)
    assert np.all(np.isnan(b[:19]))            # fewer than 20 observations
    assert np.isfinite(b[19])
    y2 = y.copy()
    y2[40:52] = np.nan                         # 30-row window ending 55 has 18 joint obs
    b2 = rolling_slope(y2, x, 30, 20)
    assert np.isnan(b2[55])


def test_rolling_sd_matches_numpy():
    rng = np.random.default_rng(2)
    x = rng.normal(size=60)
    s = rolling_sd(x, 25, 20)
    assert s[40] == pytest.approx(np.std(x[16:41], ddof=1), rel=1e-9)
    assert np.isnan(s[18])


# ---------------------------------------------------------------- book mechanics (hand-made z / beta)

def _flat_inputs(T=30, N=12):
    beta = np.tile(np.arange(N, dtype=float), (T, 1))
    elig = np.ones((T, N), bool)
    alive = np.ones((T, N), bool)
    z = np.zeros(T)
    return z, beta, elig, alive


def test_book_event_hold_exit_and_flat_rule():
    z, beta, elig, alive = _flat_inputs()
    z[3] = 2.0      # event, oil up
    z[5] = 3.0      # inside the slot: ignored
    z[8] = 2.0      # exit decision row t0+5: position still open, ignored
    z[9] = -2.0     # first flat row: event, oil down
    W, ev = book(z, beta, elig, alive, 3, 1.5)
    assert ev == [(3, 1.0), (9, -1.0)]
    exp = np.zeros(12)
    exp[[9, 10, 11]] = 0.5 / 3
    exp[[0, 1, 2]] = -0.5 / 3
    np.testing.assert_allclose(W[3], exp)
    assert np.all(np.isnan(W[4:8]))            # held, no rebalancing
    np.testing.assert_allclose(W[8], 0.0)      # exit at the open of t0+6
    np.testing.assert_allclose(W[9], -exp)     # mirror: long lowest, short highest
    assert np.all(np.isnan(W[10:14]))
    np.testing.assert_allclose(W[14], 0.0)
    assert np.all(np.isnan(W[15:]))
    assert np.all(np.isnan(W[:3]))
    for row in W:
        if np.all(np.isfinite(row)):
            assert np.abs(row).sum() <= 1.0 + 1e-12
            assert abs(row.sum()) < 1e-12      # dollar-neutral


def test_book_threshold_is_inclusive_and_param_sensitive():
    z, beta, elig, alive = _flat_inputs()
    z[3] = 1.5
    W, ev = book(z, beta, elig, alive, 3, 1.5)
    assert ev == [(3, 1.0)]
    W, ev = book(z, beta, elig, alive, 3, 1.75)
    assert ev == []
    z[3] = 2.0
    W2, _ = book(z, beta, elig, alive, 2, 1.5)
    assert np.count_nonzero(W2[3]) == 4
    assert W2[3, 11] == pytest.approx(0.25) and W2[3, 0] == pytest.approx(-0.25)
    W4, _ = book(z, beta, elig, alive, 4, 1.5)
    assert np.count_nonzero(W4[3]) == 8
    assert W4[3, 8] == pytest.approx(0.125) and W4[3, 3] == pytest.approx(-0.125)


def test_book_needs_ten_eligible_and_ranks_eligible_only():
    z, beta, elig, alive = _flat_inputs()
    z[3] = 2.0
    elig[3, :] = False
    elig[3, :9] = True                         # 9 eligible: not enough
    W, ev = book(z, beta, elig, alive, 3, 1.5)
    assert ev == []
    elig[3, 9] = True                          # 10 eligible: cols 0..9
    W, ev = book(z, beta, elig, alive, 3, 1.5)
    assert ev == [(3, 1.0)]
    assert W[3, 9] > 0 and W[3, 7] > 0 and W[3, 11] == 0.0 and W[3, 10] == 0.0


def test_book_name_losing_price_is_closed_to_cash():
    z, beta, elig, alive = _flat_inputs()
    z[3] = 2.0
    alive[5:, 10] = False
    W, _ = book(z, beta, elig, alive, 3, 1.5)
    assert W[5, 10] == 0.0
    others = np.delete(np.arange(12), 10)
    assert np.all(np.isnan(W[5, others]))      # the rest is held, weight not redistributed
    assert np.all(np.isnan(W[6:8]))
    np.testing.assert_allclose(W[8], 0.0)


# ---------------------------------------------------------------- full signal pipeline on generated data

def _market(T=700, N=12, seed=7, shock_at=600):
    rng = np.random.default_rng(seed)
    m = rng.normal(0, 0.01, T)
    e_true = rng.normal(0, 0.02, T)
    e_true[shock_at:shock_at + 5] = 0.06
    beta_true = np.linspace(-0.3, 0.3, N)
    r = m[:, None] + e_true[:, None] * beta_true[None, :] + rng.normal(0, 0.005, (T, N))
    o = 0.5 * m + e_true
    p_sess = 50.0 * np.cumprod(1.0 + o)
    oil = np.full(T, np.nan)
    oil[1:] = p_sess[:-1]                      # next_morning: row t holds session t-1
    close = np.ones((T, N))
    listed = np.ones((T, N), bool)
    ret_co = np.zeros((T, N))
    ret_oc = r
    return close, listed, ret_co, ret_oc, oil


def test_pipeline_detects_shock_and_ranks_true_exposure():
    close, listed, ret_co, ret_oc, oil = _market()
    z, beta, elig, alive = signals(close, listed, ret_co, ret_oc, oil)
    assert np.all(np.isnan(z[:200]))           # needs 200 observations first
    assert z[605] > 3.0                        # five +6 % oil-specific sessions 600..604
    order = np.argsort(beta[605])
    assert set(order[-3:]) == {9, 10, 11} and set(order[:3]) == {0, 1, 2}
    W, ev = book(z, beta, elig, alive, 3, 1.5)
    hits = [(t, s) for t, s in ev if 601 <= t <= 607]
    assert hits and hits[0][1] == 1.0
    t = hits[0][0]
    assert W[t, 11] > 0 and W[t, 0] < 0


def test_pipeline_late_listing_becomes_eligible_after_history():
    close, listed, ret_co, ret_oc, oil = _market()
    close[:300, 11] = np.nan
    listed[:300, 11] = False
    ret_oc[:300, 11] = 0.0
    z, beta, elig, alive = signals(close, listed, ret_co, ret_oc, oil)
    assert not elig[:500, 11].any()
    assert elig[520, 11]
    # e needs 200 joint obs for b (from about row 201), exposure 200 more obs of e: about row 401
    assert not elig[:400, 0].any()
    assert elig[450, 0]


def test_pipeline_point_in_time_truncation_and_perturbation():
    close, listed, ret_co, ret_oc, oil = _market()
    full = signals(close, listed, ret_co, ret_oc, oil)
    Wf, _ = book(*full, 3, 1.5)
    rng = np.random.default_rng(3)
    for cut in (450, 603, 650):
        tr = signals(close[:cut], listed[:cut], ret_co[:cut], ret_oc[:cut], oil[:cut])
        Wt, _ = book(*tr, 3, 1.5)
        np.testing.assert_array_equal(Wt, Wf[:cut])
        oc2 = ret_oc.copy()
        oc2[cut:] = rng.normal(0, 0.05, oc2[cut:].shape)
        oil2 = oil.copy()
        oil2[cut:] = rng.uniform(1, 200, oil2[cut:].shape)
        pt = signals(close, listed, ret_co, oc2, oil2)
        Wp, _ = book(*pt, 3, 1.5)
        np.testing.assert_array_equal(Wp[:cut], Wf[:cut])


def test_target_weights_signature_on_dataview():
    from lab.framework.data import DataView
    close, listed, ret_co, ret_oc, oil = _market(N=33)
    T, N = close.shape
    import strategy as S
    dates = np.arange(11000, 11000 + T).astype("datetime64[D]")
    dv = DataView(
        dates=dates, instruments=tuple(S.UNIVERSE), asset_class=tuple(["us_etf"] * N),
        ret_co=ret_co, ret_oc=ret_oc, tradable=listed.copy(), listed=listed, delisting=np.zeros((T, N), bool),
        close=close, dollar_volume=np.full((T, N), 1e7), universe=None, extras={},
        series={"fred_macro.DCOILWTICO": oil}, cash_ret=np.zeros(T),
    )
    W = target_weights(dv, PARAMS)
    assert W.shape == (T, N)
    fin = np.all(np.isfinite(W), axis=1)
    assert fin.any()
    assert np.all(np.abs(W[fin]).sum(axis=1) <= 1.0 + 1e-12)
    with pytest.raises(Exception):
        dv2 = DataView(
            dates=dates, instruments=tuple(S.UNIVERSE), asset_class=tuple(["us_etf"] * N),
            ret_co=ret_co, ret_oc=ret_oc, tradable=listed.copy(), listed=listed,
            delisting=np.zeros((T, N), bool), close=close, dollar_volume=np.full((T, N), 1e7), universe=None,
            extras={}, series={}, cash_ret=np.zeros(T),
        )
        target_weights(dv2, PARAMS)            # missing oil series must raise, not go flat
