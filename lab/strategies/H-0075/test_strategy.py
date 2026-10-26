import numpy as np
import pytest

from strategy import target_weights, PARAMS
from strategy import _event_weights, _hold, _yield_changes, _prev_window_std
from strategy import _marks

K = 10
T = 320
T0 = 250          # shock session t; action row d = T0 + 1
B = 126


def market(seed=0, shock=0.6, shock_row=T0, stress=0.0):
    rng = np.random.default_rng(seed)
    dy_obs = rng.normal(0.0, 0.05, T)
    dy_obs[shock_row] = shock
    yld_obs = 4.0 + np.cumsum(dy_obs)
    yld = np.full(T, np.nan)
    yld[1:] = yld_obs[:-1]                       # next_morning: row u holds the observation of session u-1
    b = np.linspace(-0.1, 0.1, K)                # true rate betas, mean zero -> EW barely moves on the shock
    r = 0.0003 + dy_obs[:, None] * b[None, :] + rng.normal(0.0, 0.004, (T, K))
    r[shock_row] = dy_obs[shock_row] * b + rng.normal(0.0, 0.0005, K)
    r[shock_row] += stress
    ret_co = np.zeros((T, K))
    ret_oc = r
    close = 100.0 * np.cumprod(1.0 + r, axis=0)
    listed = np.ones((T, K), dtype=bool)
    tradable = np.ones((T, K), dtype=bool)
    member = np.ones((T, K), dtype=bool)
    return dict(ret_co=ret_co, ret_oc=ret_oc, close=close, listed=listed, tradable=tradable,
                member=member, yld=yld), r, dy_obs


def run(mk, beta_days=B, k_sigma=5.0):
    return _event_weights(mk["ret_co"], mk["ret_oc"], mk["close"], mk["listed"], mk["tradable"],
                          mk["member"], mk["yld"], beta_days, k_sigma)


def test_params_match_card():
    assert PARAMS == {"beta_days": 252, "k_sigma": 1.5}


def test_prev_window_std_matches_numpy():
    x = np.random.default_rng(1).normal(size=200)
    out = _prev_window_std(x, 63)
    assert np.all(np.isnan(out[:63]))
    for j in (63, 100, 199):
        assert out[j] == pytest.approx(np.std(x[j - 63:j], ddof=1), rel=1e-9)
    x[150] = np.nan
    out = _prev_window_std(x, 63)
    assert np.isnan(out[151]) and np.isnan(out[199])
    assert np.isfinite(out[150]) and np.isfinite(out[149])


def test_yield_changes_skip_missing_days():
    yld = np.full(80, np.nan)
    yld[1:] = 4.0 + np.cumsum(np.random.default_rng(2).normal(0, 0.05, 79))
    yld[40] = np.nan                               # FRED had no value
    dyk, sig_y = _yield_changes(yld)
    assert np.isnan(dyk[0]) and np.isnan(dyk[1]) and np.isnan(dyk[40])
    assert dyk[41] == pytest.approx(yld[41] - yld[39])
    valid = dyk[np.isfinite(dyk)]
    # first sigma: the 63 valid changes before the 64th valid change
    j = np.flatnonzero(np.isfinite(dyk))[63]
    assert sig_y[j] == pytest.approx(np.std(valid[:63], ddof=1))
    assert np.all(np.isnan(sig_y[:j]))


def test_event_legs_and_betas_by_hand():
    mk, r, dy_obs = market()
    event, ev_w = run(mk)
    d = T0 + 1
    assert event[d]
    assert np.flatnonzero(event).tolist() == [d]
    # independent beta: OLS of r_i - EW on dy over sessions [t-B, t-1]
    ew = r.mean(axis=1)
    s = np.arange(T0 - B, T0)
    beta = np.array([np.polyfit(dy_obs[s], r[s, i] - ew[s], 1)[0] for i in range(K)])
    p = beta * dy_obs[T0]
    order = np.argsort(p)
    expect = np.zeros(K)
    expect[order[:2]] = 0.25
    expect[order[2:][-2:]] = -0.25
    assert np.allclose(ev_w[d], expect)
    assert ev_w[d].tolist() == [0.25, 0.25] + [0.0] * 6 + [-0.25, -0.25]
    assert np.all(np.isnan(ev_w[:d])) and np.all(np.isnan(ev_w[d + 1:]))


def test_negative_shock_flips_legs():
    mk, _, _ = market(shock=-0.6)
    event, ev_w = run(mk)
    assert ev_w[T0 + 1].tolist() == [-0.25, -0.25] + [0.0] * 6 + [0.25, 0.25]


def test_hold_five_sessions_then_flat():
    mk, _, _ = market()
    event, ev_w = run(mk)
    w = _hold(event, ev_w)
    d = T0 + 1
    assert np.all(w[:d] == 0)
    for row in range(d, d + 5):
        assert np.allclose(w[row], ev_w[d])
    assert np.all(w[d + 5:] == 0)
    assert np.all(np.abs(w).sum(axis=1) <= 1 + 1e-12)
    assert np.allclose(w.sum(axis=1), 0)


def test_new_event_replaces_and_restarts_clock():
    ev = np.zeros(30, dtype=bool)
    ev_w = np.full((30, 4), np.nan)
    ev[5] = True
    ev_w[5] = [0.25, 0.25, -0.25, -0.25]
    ev[8] = True
    ev_w[8] = [-0.25, -0.25, 0.25, 0.25]
    w = _hold(ev, ev_w)
    assert np.all(w[5:8] == ev_w[5])
    assert np.all(w[8:13] == ev_w[8])
    assert np.all(w[13:] == 0) and np.all(w[:5] == 0)


def test_equity_stress_day_excluded():
    mk, _, _ = market(stress=0.1)
    event, _ = run(mk)
    assert not event[T0 + 1]


def test_fewer_than_six_rankable_no_event():
    mk, _, _ = market()
    mk["tradable"][T0 + 1, 0:5] = False
    event, _ = run(mk)
    assert not event[T0 + 1]
    mk, _, _ = market()
    mk["tradable"][T0 + 1, 0:4] = False             # 6 rankable left
    event, ev_w = run(mk)
    assert event[T0 + 1]
    assert ev_w[T0 + 1].tolist() == [0.0] * 4 + [0.25, 0.25, 0.0, 0.0, -0.25, -0.25]


def test_no_return_on_shock_day_not_rankable():
    mk, _, _ = market()
    mk["tradable"][T0, [0, 9]] = False              # no return of ETFs 0 and 9 on t (EW stays calm)
    event, ev_w = run(mk)
    assert event[T0 + 1]
    assert ev_w[T0 + 1].tolist() == [0.0, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0, -0.25, -0.25, 0.0]


def test_late_listing_needs_beta_days_of_history():
    # first return at the row after listing; rankable iff first return <= t - B = 124
    for listed_from, rankable in ((123, True), (124, False)):
        mk, _, _ = market()
        mk["listed"][:listed_from, 9] = False
        mk["close"][:listed_from, 9] = np.nan
        event, ev_w = run(mk)
        assert event[T0 + 1]
        if rankable:
            assert ev_w[T0 + 1, 9] == -0.25
        else:
            assert ev_w[T0 + 1, 9] == 0 and ev_w[T0 + 1, 7] == -0.25


def test_no_events_before_warmup():
    mk, _, _ = market(shock_row=100)
    event, _ = run(mk, k_sigma=1.0)
    assert not event[: B + 2].any()


def test_parameters_change_output():
    mk, _, _ = market()
    e10, _ = run(mk, k_sigma=1.0)
    e15, _ = run(mk, k_sigma=1.5)
    e20, _ = run(mk, k_sigma=2.0)
    assert e10.sum() > e15.sum() > e20.sum() >= 1
    assert not np.any(e20 & ~e15) and not np.any(e15 & ~e10)
    mk, _, _ = market()
    a, _ = run(mk, beta_days=126, k_sigma=1.0)
    b, _ = run(mk, beta_days=252, k_sigma=1.0)
    assert np.flatnonzero(a)[0] < np.flatnonzero(b)[0]


def test_point_in_time_truncation_and_noise():
    mk, _, _ = market()
    ev_full, w_full = run(mk, k_sigma=1.0)
    full = _hold(ev_full, w_full)
    rng = np.random.default_rng(7)
    for cut in (130, 200, 251, 252, 290):
        sub = {k: v[: cut + 1].copy() for k, v in mk.items()}
        ev, w = run(sub, k_sigma=1.0)
        assert np.array_equal(_hold(ev, w), full[: cut + 1])
        noisy = {k: v.copy() for k, v in mk.items()}
        noisy["ret_oc"][cut + 1:] = rng.normal(0, 0.05, noisy["ret_oc"][cut + 1:].shape)
        noisy["yld"][cut + 1:] = rng.normal(4, 1, T - cut - 1)
        ev, w = run(noisy, k_sigma=1.0)
        assert np.array_equal(_hold(ev, w)[: cut + 1], full[: cut + 1])


def test_diagnostic_matches_strategy_legs():
    mk, _, _ = market()
    event, ev_w = run(mk, k_sigma=1.0)
    mask, side = _marks(mk["ret_co"], mk["ret_oc"], mk["close"], mk["listed"], mk["tradable"],
                        mk["member"], mk["yld"], B, 1.0)
    assert np.array_equal(mask.any(axis=1), event)
    rows = np.flatnonzero(event)
    assert np.array_equal(mask[rows], ev_w[rows] != 0)
    assert np.array_equal(side[rows][mask[rows]], np.sign(ev_w[rows][mask[rows]]))
    assert np.all(mask.sum(axis=1)[rows] == 4)
